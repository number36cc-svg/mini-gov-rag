"""检索 + 生成：问题 → 向量检索 → 置信度判定 → MiniCPM 生成带引用的回答。

三个差异化设计都在这个文件里：
1. 版本时效管理 —— 问题问"修订改了什么"时，自动把同一法规的多个版本都捞出来对比
2. 可追责拒答  —— 检索相似度不够就直接拒答，不把没有依据的问题丢给大模型去编
3. 引用可溯源  —— 每条回答都能点回文件名 + 版本 + 页码
"""
from __future__ import annotations

import json
import re

import numpy as np
import requests

from .config import (
    CONF_GRAY,
    CONF_HIGH,
    CONF_LOW,
    INDEX_DIR,
    LLM_KEEP_ALIVE,
    LLM_MODEL,
    MAX_TOKENS,
    OLLAMA_URL,
    REQUEST_TIMEOUT,
    TEMPERATURE,
    TOP_K,
)
from .ingest import get_model

# bge 系列做检索时，给 query 加指令前缀能明显提升召回（passage 侧不加）
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

SYSTEM_PROMPT = """你是政企制度问答助手，服务对象是财政/政务单位的业务人员。

【工作方式】下面会给你若干条【参考资料】，都是从知识库检索出来的原文。你的任务是从资料里找到答案并转述。

【必须遵守】
1. 资料里只要有答案，就必须回答。资料中出现「｜」（竖线）是表格的分隔符，
   一行表示一条记录，按位置对应：报错信息｜原因｜处理步骤。
2. 每个结论后用「（依据：《文件名》版本 第X页）」标注出处。
3. 涉及数字、比例、时限、条数上限，必须照抄原文的具体数值，不要概括成"一定数量"。
4. 如果资料同时包含同一文件的不同版本，直说两者是否一致；不一致就分别说明。
5. 不要寒暄、不要重复问题、不要加"希望对您有帮助"。
6. 如果用户消息里给出了【程序核对结果】，必须直接采用该结论（一致性/差异以它为准），
   你只负责把它讲清楚，不得自行判断两版是否相同。
7. 回答必须带事实：涉及日期、比例、条号、条数上限时，必须把具体值写出来。
   禁止只给"没有变化""有区别""可以办理"这类结论而不写具体内容。

【示例】
资料：序号｜报错信息｜原因｜处理步骤：2｜「导出失败：超出导出条数上限」｜A学院导出上限配置为 5000 条｜缩小日期区间分批导出
问题：导出超上限怎么处理？
回答：原因是 A学院导出条数上限为 5000 条。处理方式：缩小日期区间分批导出；确需超量请联系单位管理员临时调整上限。（依据：《常见报错对照_A学院》现行 第1页）

【唯一允许拒答的情形】资料与问题完全无关（例如问的是法规条文，资料却是系统操作手册），
此时只输出这一句，不输出任何其他文字：依据现有制度文件无法回答该问题。
只要资料里出现了与问题相关的信息，就按上面规则回答，不得拒答。"""

# 闸三触发后的二次确认提示词：不给任何"可以不答"的选项，只做抽取。
# 为什么需要它：8B 级小模型在"资料里有答案但自己没把握"时，会抓住
# system prompt 里的拒答条款当默认出口（实测同一问题 3 次里 2 次误拒）。
# 与其把提示词写得更严（更严只会让它更爱拒），不如在拒答前换一个没有出口的提示词重试一次：
# 能抽出来就说明资料里确实有答案，是第一遍的误判；抽不出来才真正按拒答处理。
EXTRACT_PROMPT = """从下面【资料】中找出能回答【问题】的句子，直接给出答案并注明出处。
如果【资料】里确实没有相关内容，只输出两个字：无。"""

# 模型自述"答不了"的判定词。命中且回答很短 → 后处理改判为拒答。
# 这是第二道闸：向量相似度会被"字面像制度"的问题骗过，模型的可答性判断更可靠。
REFUSAL_MARKERS = ("无法回答", "无法确定", "没有相关", "未检索到", "不足以回答", "无法依据")
REFUSAL_MAX_LEN = 80

# 判定"这个问题是在问版本差异"
COMPARE_RE = re.compile(r"修订|修改|变化|变动|区别|差异|对比|新旧|新增|删去|调整|相比|一样|一致|相同|不同|两个版本|两版")
# 拒答话术：结果里带上人工渠道，这是政企客户验收时最看重的"兜底"
REFUSAL = (
    "依据现有制度文件无法回答该问题（知识库中未检索到足够相关的条款）。\n\n"
    "建议：① 换用制度中的原文表述再问一次；② 咨询本单位归口管理部门；"
    "③ 如需纳入知识库，请管理员补充上传相关文件。"
)

_index_cache = {"chunks": None, "vectors": None}

# 混合检索权重：语义相似度 + 字面重合度。
# 纯向量检索在"账号已锁定"这类原词直问的问题上反而不如字面匹配稳，
# 因为 bge-small 只有 512 维、对短查询的语义压缩有损。两者加权是最省事的补救。
LEX_WEIGHT = 0.4


def _bigrams(text: str) -> set:
    """中文按二元字组切，去掉标点与空白。"""
    compact = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", text)
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def _lexical_coverage(query: str, text: str) -> float:
    """query 的二元字组有多大比例出现在 text 里（0~1）。"""
    q = _bigrams(query)
    if not q:
        return 0.0
    return len(q & _bigrams(text)) / len(q)


# ---------------------------------------------------------------- 索引加载
def load_index() -> tuple[list[dict], np.ndarray]:
    """加载入库阶段落地的切片与向量（进程内只读一次）。"""
    if _index_cache["chunks"] is None:
        chunks_path = INDEX_DIR / "chunks.json"
        vectors_path = INDEX_DIR / "vectors.npy"
        if not chunks_path.exists():
            raise RuntimeError("索引不存在，请先运行：python -m app.ingest")
        _index_cache["chunks"] = json.loads(chunks_path.read_text(encoding="utf-8"))
        _index_cache["vectors"] = np.load(vectors_path)
    return _index_cache["chunks"], _index_cache["vectors"]


def knowledge_stats() -> dict:
    """给界面侧边栏用：知识库规模。"""
    meta_path = INDEX_DIR / "meta.json"
    if not meta_path.exists():
        return {"chunks": 0, "dim": 0, "sources": [], "scenarios": []}
    return json.loads(meta_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 检索
def _encode_query(query: str) -> np.ndarray:
    vec = get_model().encode(
        [QUERY_INSTRUCTION + query], normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vec, dtype=np.float32)[0]


def search(
    query: str, top_k: int = TOP_K, scenario: str | None = None, source: str | None = None
) -> list[dict]:
    """混合检索（语义 + 字面）。scenario 限定场景，source 限定法规名（版本对比时用）。"""
    chunks, vectors = load_index()
    qvec = _encode_query(query)
    cos = vectors @ qvec  # 向量已归一化，点积 = 余弦相似度
    lex = np.array([_lexical_coverage(query, c["text"]) for c in chunks], dtype=np.float32)
    scores = (1.0 - LEX_WEIGHT) * cos + LEX_WEIGHT * lex

    order = np.argsort(-scores)
    hits: list[dict] = []
    for i in order:
        item = chunks[int(i)]
        if scenario and item["scenario"] != scenario:
            continue
        if source and item["source"] != source:
            continue
        hits.append(dict(item, score=float(scores[int(i)]), cos=float(cos[int(i)])))
        if len(hits) >= top_k:
            break
    return hits


def is_version_question(query: str) -> bool:
    """问题里同时出现法规名与"修订/变化/区别"等词，判定为版本对比问题。"""
    if not COMPARE_RE.search(query):
        return False
    chunks, _ = load_index()
    return any(c["source"] and c["source"] in query for c in chunks)


def _version_key(version: str) -> int:
    """版本排序键：'2018修正' → 2018，'现行' → 9999（视为最新）。"""
    if not version or version == "现行":
        return 9999
    m = re.match(r"(\d{4})", version)
    return int(m.group(1)) if m else 0


# 切片开头的条款号，用于跨版本对齐同一条文
_ART_HEAD_RE = re.compile(r"^\s*(第[一二三四五六七八九十百零〇\d]+条)")


def _align_key(hit: dict) -> tuple:
    """跨版本对齐键：同一份文件的「同一条文」在各版本里必须得到同一个 key。

    不能用切片序号（chunk_id 末尾）——两版条款编号存在偏移时序号对不上，
    去重会失效（实测：问预备费比例，引用里同时出现 2014 与 2018 两版）。
    条款号才是跨版本稳定的锚点；非条款类文档退回用正文前缀做键。
    """
    m = _ART_HEAD_RE.match(hit["text"])
    if m:
        return (hit["source"], m.group(1))
    return (hit["source"], hit["text"][:40])


def keep_latest_versions(hits: list[dict]) -> list[dict]:
    """同一份文件的多版本，只保留最新生效的那一版。

    这是"版本时效管理"的另一半：客户问"预备费比例是多少"，答案应该是现行有效的那一版，
    而不是把 2014 和 2018 两版并排贴出来让用户自己判断——那是把判断责任推回给客户，
    政企场景里等于没做版本管理。只有明确问"改了什么"时才走多版本对比。
    """
    best: dict = {}
    order: list = []
    for h in hits:
        key = _align_key(h)
        if key not in best:
            best[key] = h
            order.append(key)
        elif _version_key(h["version"]) > _version_key(best[key]["version"]):
            best[key] = h
    return [best[k] for k in order]


def retrieve(query: str, scenario: str | None = None, top_k: int = TOP_K) -> list[dict]:
    """检索入口：普通问题走"多取候选 → 版本去重 → 取 top_k"；版本对比问题走多版本。"""
    if not is_version_question(query):
        raw = search(query, top_k=top_k * 3, scenario=scenario)
        return keep_latest_versions(raw)[:top_k]

    chunks, _ = load_index()
    all_sources = {c["source"] for c in chunks if not scenario or c["scenario"] == scenario}
    mentioned = [s for s in all_sources if s and s in query]
    if not mentioned:
        return search(query, top_k=top_k, scenario=scenario)

    target = mentioned[0]
    versions = sorted({c["version"] for c in chunks if c["source"] == target})
    if len(versions) < 2:  # 只有一个版本，没有可比的东西
        return search(query, top_k=top_k, scenario=scenario)

    # 结构化优先：问题点名了「第X条」，就直接按条款号把每个版本的这一条捞出来。
    # 这比向量检索准得多——「两版第五十一条引用的条款号一样吗」这类元问题，
    # 语义向量根本表达不出检索意图，但条款号是精确线索。
    refs = re.findall(r"第[一二三四五六七八九十百零〇\d]+条", re.sub(r"\s+", "", query))
    if refs:
        structured: list[dict] = []
        for ver in versions:
            for c in chunks:
                if c["source"] == target and c["version"] == ver and any(
                    r in re.sub(r"\s+", "", c["text"]) for r in refs
                ):
                    structured.append(dict(c, score=1.0, cos=1.0))
                    break
        if structured:
            return structured

    per_version = max(2, top_k // len(versions))
    hits: list[dict] = []
    for ver in versions:
        part = [
            h
            for h in search(query, top_k=per_version * 3, scenario=scenario, source=target)
            if h["version"] == ver
        ]
        hits.extend(part[:per_version])
    hits.sort(key=lambda h: -h["score"])
    return hits


# ---------------------------------------------------------------- 置信度
def grade_confidence(hits: list[dict]) -> str:
    """第一道闸：按纯语义余弦分三档。没有依据时宁可拒答，这是政企场景的底线。

    注意用的是 cos 而不是混合排序分——字面重合度会让「我们单位去年非税收入多少」
    这种问题的排序分虚高，用它判档必然误判。
    """
    if not hits:
        return "无法回答"
    top = hits[0].get("cos", hits[0]["score"])
    if top >= CONF_HIGH:
        return "依据充分"
    if top >= CONF_LOW:
        return "部分依据"
    return "无法回答"


def article_ref_missing(query: str, hits: list[dict]) -> str:
    """第二道闸之一：问题点名了某个条款，但检索结果里根本没有这条。

    典型场景：「预算法第200条规定了什么」——向量相似度很高（全是制度词），
    但法规只有 101 条，属于凭空捏造的条款，必须拒答而不是硬凑一条相近的。
    返回缺失的条款号，没有缺失则返回空串。
    """
    compact_q = re.sub(r"\s+", "", query)  # 用户常写成「第 200 条」
    for ref in re.findall(r"第[一二三四五六七八九十百零〇\d]+条", compact_q):
        if not any(ref in re.sub(r"\s+", "", h["text"]) for h in hits):
            return ref
    return ""


def llm_says_cannot_answer(text: str) -> bool:
    """第二道闸之二：模型自己说答不了。

    要求模型"答不了就只输出一句话"，所以命中拒答词且回答很短时，
    可以放心地把整条结果改判为拒答并清空引用——这是代码判定，不靠模型自觉。
    """
    body = text.strip()
    return len(body) <= REFUSAL_MAX_LEN and any(m in body for m in REFUSAL_MARKERS)


# ---------------------------------------------------------------- 生成
def _build_context(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        lines.append(
            "【参考资料 %d】《%s》%s 第%s页\n%s"
            % (i, h["source"], h["version"], h["page"], h["text"])
        )
    return "\n\n".join(lines)


def _extract_citations(hits: list[dict]) -> list[dict]:
    """按 文件名+版本+页码 去重，供界面展开"依据条款"用。"""
    seen, cites = set(), []
    for h in hits:
        key = (h["source"], h["version"], h["page"])
        if key in seen:
            continue
        seen.add(key)
        cites.append(
            {
                "source": h["source"],
                "version": h["version"],
                "page": h["page"],
                "score": round(h.get("cos", h["score"]), 3),
                "excerpt": h["text"][:160].replace("\n", " "),
            }
        )
    return cites


# 模型不可用时的统一标记。回答里带这个前缀，说明是环境没准备好，不是模型答的，
# 不能按"依据充分"返回给客户演示——那等于把环境故障伪装成系统能力。
LLM_ERROR_MARK = "【模型不可用】"


def version_article_diff(question: str, hits: list[dict] | None = None) -> str:
    """版本对比题的确定性比对：程序负责判定，模型只负责表述。

    为什么不让模型自己比对：两版第五十一条只差一个条号引用（2014 引"第四十条"、
    2018 引"第五十条"），8B 模型会不看原文直接下结论，实测给出了自相矛盾的答案
    （"不一样，两版引用的都是第五十一条"）。这类字符级差异必须用代码判。

    比对对象：问题里点名的条款；没点名就比对检索命中的条款——因为"有什么变化"这类
    开放提问，模型同样会直接下"没有变化"的结论（实测 V01 就答错了）。
    """
    chunks, _ = load_index()
    compact = re.sub(r"\s+", "", question)
    refs = re.findall(r"第[一二三四五六七八九十百零〇\d]+条", compact)
    sources = {c["source"] for c in chunks if c["source"] and c["source"] in question}
    if not sources:
        return ""
    src = sorted(sources)[0]

    if not refs and hits:
        for h in hits:
            if h["source"] != src:
                continue
            m = _ART_HEAD_RE.match(h["text"])
            if m and m.group(1) not in refs:
                refs.append(m.group(1))
    if not refs:
        return ""

    lines: list[str] = []
    for ref in refs[:3]:
        texts: dict = {}
        for c in chunks:
            if c["source"] != src:
                continue
            if re.sub(r"\s+", "", c["text"]).startswith(ref):  # 必须是该条本身
                texts.setdefault(c["version"], c["text"].strip())
        if len(texts) < 2:
            continue
        norm = {v: re.sub(r"\s+", "", t) for v, t in texts.items()}
        if len(set(norm.values())) == 1:
            lines.append("· %s：各版本条文文本完全一致（仅空格/标点排版差异）。" % ref)
        else:
            lines.append("· %s：存在实质差异，原文如下：" % ref)
            for v in sorted(texts, key=_version_key):
                lines.append("  - %s：%s" % (v, texts[v][:200]))
    if not lines:
        return ""
    return "【程序核对结果】以下为逐字比对结论，必须直接采用，不得自行判断：\n" + "\n".join(lines)


def warmup_llm() -> None:
    """预热：把模型装进内存并常驻（keep_alive）。

    Ollama 默认闲置 5 分钟卸载模型，重新加载 5.5GB 要 30-60 秒——
    演示第一问"卡很久"的元凶。界面启动时在后台线程调一次本函数，
    用户点第一个问题时模型已经是热的。
    """
    try:
        requests.post(
            OLLAMA_URL + "/api/generate",
            json={"model": LLM_MODEL, "prompt": "好", "keep_alive": LLM_KEEP_ALIVE, "stream": False},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException:
        pass  # 预热失败不打扰用户，ask_llm 里会给出可操作的报错


def ask_llm(
    question: str, hits: list[dict], system_prompt: str = SYSTEM_PROMPT, extra: str = ""
) -> str:
    """调用本地 Ollama（/api/chat）。模型没拉、服务没起时给出可操作的报错。"""
    user_content = "【参考资料】\n%s\n\n【问题】%s" % (_build_context(hits), question)
    if extra:
        user_content += "\n\n" + extra
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "stream": False,
        "keep_alive": LLM_KEEP_ALIVE,
        "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
    }
    try:
        resp = requests.post(OLLAMA_URL + "/api/chat", json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        return LLM_ERROR_MARK + " 本地模型服务未启动。请执行：brew services start ollama"
    except requests.exceptions.HTTPError:
        if "not found" in resp.text.lower():
            return LLM_ERROR_MARK + " 模型 %s 未下载。请执行：ollama pull %s" % (LLM_MODEL, LLM_MODEL)
        return LLM_ERROR_MARK + " 模型调用失败（HTTP %s）：%s" % (resp.status_code, resp.text[:200])
    except requests.exceptions.Timeout:
        return LLM_ERROR_MARK + " 模型响应超时（%d 秒）。首次加载模型较慢，可稍后重试。" % REQUEST_TIMEOUT
    data = resp.json()
    return (data.get("message") or {}).get("content", "").strip() or "（模型返回为空）"


def answer(question: str, scenario: str | None = None, on_stage=None) -> dict:
    """完整问答流程：检索 → 三道闸 → 生成 → 返回结构化结果。

    on_stage: 可选回调，函数签名 (stage_text: str)，界面用它展示真实进度
    （检索毫秒级、生成 10 秒级，首次或闲置后模型加载另需 30-60 秒——
    把等待时间花在哪说清楚，比一个笼统的转圈体验好得多）。

    为什么要三道闸，而不是"相似度够高就答"：
    课程 demo 里一个阈值就够用，但政企客户验收时会拿"预算法第 200 条"这种
    不存在的问题来考你——它的向量相似度比很多真问题都高。答错了要担责，
    所以这里的原则是：宁可少答，不可错答。
    """
    def _stage(text: str) -> None:
        if on_stage:
            on_stage(text)

    _stage("检索制度条款…")
    hits = retrieve(question, scenario=scenario)
    # 界面上显示的"最高相似度"必须与判档用同一个数（cos）。
    # 早期用融合分（score），数值和阈值对不上，演示时看着自相矛盾。
    top_score = round(hits[0].get("cos", hits[0]["score"] if hits else 0.0), 3) if hits else 0.0

    def _refuse(reason: str) -> dict:
        return {
            "answer": reason + "\n\n" + REFUSAL,
            "confidence": "无法回答",
            "citations": [],
            "top_score": top_score,
        }

    # 闸一：相似度不够
    confidence = grade_confidence(hits)
    if confidence == "无法回答":
        # 灰档：cos 在 [CONF_GRAY, CONF_LOW) 的问题不算"明显无关"——
        # 实测"目的类"真问题（"预算法为什么编制"→第一条"为了…制定本法"）cos 只有
        # 0.46~0.48，硬拒会让用户觉得"库里有却答不了"。灰档不生成答案、只展示
        # 最相关条款原文，即使放进来的是超纲问题也没有编造风险。
        top_cos = hits[0].get("cos", 0.0) if hits else 0.0
        if top_cos >= CONF_GRAY:
            ref = hits[0]
            return {
                "answer": (
                    "知识库中未找到能直接回答该问题的条款。相关度最高的条款原文如下，供参考：\n\n"
                    "> 《%s》%s 第%s页\n> %s"
                    % (ref["source"], ref["version"], ref["page"], ref["text"][:220])
                ) + "\n\n" + REFUSAL,
                "confidence": "无法回答",
                "citations": _extract_citations(hits[:1]),
                "top_score": round(top_cos, 3),
            }
        return _refuse("检索到的条款与问题相关度不足。")

    # 闸二：问题点名了不存在的条款
    missing = article_ref_missing(question, hits)
    if missing:
        return _refuse("知识库中没有「%s」这一条款，可能条款号有误或该条款不在已收录文件中。" % missing)

    # 闸三：模型读完后自己判断答不了
    diff_note = version_article_diff(question, hits)  # 版本对比题：先给程序核对结论
    _stage("本地模型生成回答中…（首次或闲置后需加载模型，约 30-60 秒；常驻后 10 秒内）")
    text = ask_llm(question, hits, extra=diff_note)
    if text.startswith(LLM_ERROR_MARK):
        # 环境问题要如实暴露，不能伪装成"依据充分"的答案
        return {
            "answer": text,
            "confidence": "服务不可用",
            "citations": [],
            "top_score": top_score,
        }
    if llm_says_cannot_answer(text):
        # 二次确认：换一个"没有拒答出口"的提示词重试。
        # 能抽出答案 → 第一遍是误拒（置信度降一档，诚实反映这道题模型没把握）；
        # 抽不出来或仍拒答 → 真拒答。
        retry = ask_llm(question, hits, system_prompt=EXTRACT_PROMPT, extra=diff_note)
        hollow = retry.strip().strip("。.") in ("", "无", "没有", "无相关内容")
        if retry.startswith(LLM_ERROR_MARK) or hollow or llm_says_cannot_answer(retry):
            return _refuse("已检索到相关文件，但其中没有能直接回答该问题的条款。")
        text = retry
        if confidence == "依据充分":
            confidence = "部分依据"

    return {
        "answer": text,
        "confidence": confidence,
        "citations": _extract_citations(hits),
        "top_score": top_score,
    }
