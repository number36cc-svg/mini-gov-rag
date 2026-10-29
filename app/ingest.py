"""入库：读文档 → 按条款切分 → 向量化 → 落地索引。

设计要点（面试会被问到的三件事）：
1. 为什么按「第X条」切而不是固定字数切？
   制度条文是一个完整规范单元，拦腰截断会产生"半条"，检索到了也答不全。
2. 为什么自己算余弦相似度而不用向量数据库？
   本场景切片量在千级，numpy 一次矩阵乘就出结果（毫秒级），
   引入 Milvus/Chroma 只是多一层运维负担。上量到十万级再换，这个判断本身就是工程取舍。
3. 为什么 embedding 和 LLM 分开？
   embedding 只管"找得到"，LLM 只管"说得清"。两者解耦后换模型互不影响。

命令行用法：
    python -m app.ingest          # 重建索引
"""
from __future__ import annotations

import json
import re
import sys
import threading
from pathlib import Path

import numpy as np

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_DIR,
    EMBED_MODEL,
    INDEX_DIR,
)

# 条款标题，如「第三十二条」；兼容阿拉伯数字写法
ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百零〇\d]+条")
# 只认「行首」的条款标题。法规正文里会出现「依照本法第四十条规定报送备案的预算」
# 这类引述——不做行首约束就会把它误判成新条款，导致后面所有条款编号整体错位。
ARTICLE_BOL_RE = re.compile(r"(?m)^[ \t\u3000]*(第[一二三四五六七八九十百零〇\d]+条)")
# Markdown 表格行与表头分隔行
TABLE_ROW_RE = re.compile(r"(?m)^\s*\|.*\|\s*$")
TABLE_SEP_RE = re.compile(r"(?m)^\s*\|[\s:\-|]+\|\s*$")
# 文件名里的版本标识，如「预算法_2014修正.md」→ (预算法, 2014修正)
VERSION_RE = re.compile(r"^(?P<source>.+?)_(?P<version>\d{4}[^_]*?)$")

_model = None


# ---------------------------------------------------------------- 读文档
def _read_txt(path: Path) -> list[str]:
    return [path.read_text(encoding="utf-8", errors="ignore")]


def _read_pdf(path: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


def _read_docx(path: Path) -> list[str]:
    from docx import Document

    doc = Document(str(path))
    return ["\n".join(p.text for p in doc.paragraphs)]


READERS = {".md": _read_txt, ".txt": _read_txt, ".pdf": _read_pdf, ".docx": _read_docx}


def parse_filename(stem: str) -> tuple[str, str]:
    """从文件名解析 (source, version)。无版本标识时 version 记为「现行」。"""
    m = VERSION_RE.match(stem)
    if m:
        return m.group("source"), m.group("version")
    return stem, "现行"


def load_documents(docs_dir: Path = DOCS_DIR) -> list[dict]:
    """读取 docs_dir 下所有支持的文档，返回 [{text, source, version, page, scenario}]。

    按页/段返回，page 为该页（PDF）或 1（文本类文档）。
    单个文件解析失败只打印警告，不中断整体入库。
    """
    docs: list[dict] = []
    used_files = 0
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in READERS:
            continue
        # 场景 = 相对 docs_dir 的第一层目录名；直接放在根目录下的归为「其他」
        rel = path.relative_to(docs_dir)
        scenario = rel.parts[0] if len(rel.parts) > 1 else "其他"
        source, version = parse_filename(path.stem)

        try:
            pages = READERS[path.suffix.lower()](path)
        except Exception as exc:  # noqa: BLE001 - 一个坏文件不该拖垮整批
            print("[警告] 跳过无法解析的文件 %s：%s" % (path.name, exc))
            continue

        used_files += 1
        for i, text in enumerate(pages, start=1):
            text = text.strip()
            if len(text) < 20:  # 空页、封面页直接丢
                continue
            docs.append(
                {
                    "text": text,
                    "source": source,
                    "version": version,
                    "page": i,
                    "scenario": scenario,
                }
            )
    print("[入库] 读取 %d 个文件，共 %d 个文本片段" % (used_files, len(docs)))
    return docs


# ---------------------------------------------------------------- 切分
def _split_articles(text: str) -> list[str]:
    """按「第X条」边界切分，返回条款块列表（含条款编号本身）。

    优先只认行首的条款标题，避免把"依照本法第四十条规定"这种正文引述当成新条款；
    整篇没有换行的极端情况（如 PDF 抽取成一大段）再退化为全文匹配。
    """
    marks = [m.start(1) for m in ARTICLE_BOL_RE.finditer(text)]
    if not marks:
        marks = [m.start() for m in ARTICLE_RE.finditer(text)]
    if not marks:
        return [text]
    blocks = []
    # 第一条之前的内容（标题、修正日期、发布机关）一律丢弃。
    # 它不含任何规范内容，却塞满了"预算法""修正""全国人大常委会"这类制度词，
    # 检索时余弦分虚高，会把真正的条款挤出候选——实测「预算法自什么时间起施行」
    # 的 top1 长期是这段文件头而不是第一百零一条，导致误拒答。
    for i, start in enumerate(marks):
        end = marks[i + 1] if i + 1 < len(marks) else len(text)
        block = text[start:end].strip()
        if block:
            blocks.append(block)
    return blocks


def _split_long(block: str) -> list[str]:
    """单条款超长时按句号/分号再切。"""
    if len(block) <= CHUNK_SIZE:
        return [block]
    pieces, buf = [], ""
    for sent in re.split(r"(?<=[。；！？])", block):
        if len(buf) + len(sent) > CHUNK_SIZE and buf:
            pieces.append(buf)
            buf = sent
        else:
            buf += sent
    if buf:
        pieces.append(buf)
    return pieces


def _is_table_doc(text: str) -> bool:
    """判断文档是否以 Markdown 表格为主体（如 A/B 学院报错对照表）。"""
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    table_lines = sum(1 for l in lines if l.strip().startswith("|"))
    return table_lines / len(lines) > 0.25


def _table_chunks(text: str) -> list[str]:
    """表格型文档按行切：一条报错/一条规则就是一个完整检索单元。

    如果按 500 字合并，一张 8 行的报错表会被压成两三个切片，
    每张切片里混着七八条不相干的报错，语义被稀释，检索必然失准。
    """
    header = ""
    for line in text.split("\n"):
        if line.strip().startswith("|") and not TABLE_SEP_RE.match(line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not header:
                header = "｜".join(cells)  # 表头，给每行补上字段语义
                continue
            body = "｜".join(c for c in cells if c)
            if len(body) >= 10:
                yield "%s：%s" % (header, body)


def chunk_documents(docs: list[dict]) -> list[dict]:
    """把文档切成检索单元，尽量让一个切片 = 一个完整语义单元。

    三套策略按文档形态分流：
    - 表格型（报错对照表）→ 每行一个切片，补表头当字段说明
    - 条款型（法规）→ 按「第X条」切，相邻小块合并到 CHUNK_SIZE
    - 其他（操作手册）→ 按段落合并

    返回 [{text, source, version, page, scenario, chunk_id}]
    """
    chunks: list[dict] = []
    for doc in docs:
        is_table = _is_table_doc(doc["text"])
        if is_table:
            merged = list(_table_chunks(doc["text"]))  # 表格行不合并，一行一条最好检索
        else:
            blocks: list[str] = []
            for block in _split_articles(doc["text"]):
                blocks.extend(_split_long(block))

            if len(blocks) >= 5:
                # 条款型文档（法规）：「第X条」是权威的规范边界，一条一片，不做合并。
                # 合并会让"第三十九条+第四十条"挤进同一片，检索到"预备费"也说不清
                # 是第几条，引用溯源就失去了意义。
                merged = blocks
            else:
                # 说明型文档（操作手册）：按段落合并成较大切片，保住操作步骤的连贯性
                merged, buf = [], ""
                for block in blocks:
                    if len(buf) + len(block) <= CHUNK_SIZE:
                        buf += block
                    else:
                        if buf:
                            merged.append(buf)
                        # 注意 CHUNK_OVERLAP=0 时必须显式跳过——Python 里 buf[-0:] 等于
                        # 整个 buf，不判 0 会让每个切片把上一块全文吞进来，切片越滚越大。
                        tail = ""
                        if CHUNK_OVERLAP > 0 and buf and len(block) < CHUNK_SIZE:
                            tail = buf[-CHUNK_OVERLAP:]
                        buf = tail + block
                if buf:
                    merged.append(buf)

        for idx, text in enumerate(merged):
            text = text.strip()
            # 短条款（如"本法自公布之日起施行"）也是完整规范，不能因为短就丢
            if len(text) < 12:
                continue
            chunks.append(
                {
                    "text": text,
                    "source": doc["source"],
                    "version": doc["version"],
                    "page": doc["page"],
                    "scenario": doc["scenario"],
                    "chunk_id": "%s|%s|p%d|c%d" % (doc["source"], doc["version"], doc["page"], idx),
                }
            )

    avg = sum(len(c["text"]) for c in chunks) / len(chunks) if chunks else 0
    print("[入库] 切分为 %d 个检索切片，平均 %.0f 字" % (len(chunks), avg))
    return chunks


# ---------------------------------------------------------------- 向量化
_model_lock = threading.Lock()


def get_model():
    """懒加载 embedding 模型（首次会下载约 95MB）。

    加锁是为了界面预热线程和用户提问并发触发加载时不会重复装载——
    双重检查锁：锁外先查一次（热路径无锁开销），锁内再查一次（防竞态）。
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                print("[入库] 加载 embedding 模型 %s ..." % EMBED_MODEL)
                _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """把文本编码成 L2 归一化向量——归一化之后，点积就等于余弦相似度。"""
    vecs = get_model().encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vecs, dtype=np.float32)


# ---------------------------------------------------------------- 建索引
def build_index() -> dict:
    """完整入库流程，落地到 data/index/（vectors.npy + chunks.json + meta.json）。"""
    docs = load_documents()
    if not docs:
        raise SystemExit("[入库] data/docs 下没有可读文档，请先放入语料")

    chunks = chunk_documents(docs)
    vectors = embed_texts([c["text"] for c in chunks])

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vectors)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    stats = {
        "chunks": len(chunks),
        "dim": int(vectors.shape[1]),
        "sources": sorted({"%s（%s）" % (c["source"], c["version"]) for c in chunks}),
        "scenarios": sorted({c["scenario"] for c in chunks}),
    }
    (INDEX_DIR / "meta.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("[入库] 完成：%d 切片 × %d 维 → %s" % (stats["chunks"], stats["dim"], INDEX_DIR))
    print("[入库] 语料清单：" + "、".join(stats["sources"]))
    return stats


if __name__ == "__main__":
    build_index()
