# PROMPTS.md · 给 Cursor 的分步施工清单（v2）

> 用法：装好 Cursor、打开本项目文件夹后，`Cmd+I` 进入 Agent 模式，
> **按顺序**把每一步的 prompt 整段粘贴进去。每步完成后先按"验收"自测，
> 通过再 `git commit`，然后进行下一步。
>
> v2 新增：版本时效管理（第 4 步）、POC 报告生成器（第 7 步）——
> 这两步是本项目的差异化所在，别跳过。

---

## 第 0 步：环境准备（自己做，不用 Cursor）

```bash
# 在项目目录下
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 安装 Ollama: https://ollama.com/download/mac
ollama pull minicpm-v
```

**测试语料（重要，为版本管理和多场景准备）**，按子目录组织放入 `data/docs/`：

```
data/docs/
├── 制度文件/     # 公开法规，至少 1 份含两个版本（文件名带版本标识）
│   ├── 预算法_2014修正.pdf      # 全国人大网可查
│   ├── 预算法_2018修正.pdf
│   └── 政府非税收入管理办法.pdf
└── 运维手册/     # 自编演示语料（后面 Cursor 会帮你生成）
    ├── 演示系统操作手册.md
    ├── 常见报错对照_A学院.md
    └── 常见报错对照_B学院.md
```

红线：只能用公开文件或自编演示数据，绝不放真实客户资料、内部系统名、内部厂商名。

验收：`ollama run minicpm-v "你好"` 能正常回答；`data/docs/` 里至少 3 份 PDF，且含同名不同版本文件；`data/docs/运维手册/` 子目录已存在。

**生成运维演示语料（0.5 步，✅ 已由 WorkBuddy 免费完成，跳过）**

> 三个文件已在 data/docs/运维手册/ 中：演示系统操作手册.md、常见报错对照_A学院.md、常见报错对照_B学院.md。
> 抽查 1 分钟确认内容无真实内部信息即可。

验收：3 个文件内容完整，肉眼检查无真实内部信息。

---

## 第 1 步：文档解析与加载（含版本元数据）

**粘贴给 Agent：**

> 阅读 app/ingest.py 的 load_documents 函数和文件顶部 TODO，实现它：
> 用 pypdf 解析 data/docs/ 下所有 .pdf（按页提取，记录页码），
> 用 python-docx 解析 .docx，直接读取 .txt/.md；
> 单个文件解析失败时打印警告并跳过，不要中断；
> 返回 [{text, source, version, page}] 列表。
> **额外要求 1：从文件名中解析版本信息**——形如"预算法_2014修正.pdf"的文件，
> source 记为"预算法"，并在元数据中额外记录 version="2014修正"；
> 无版本标识的文件 version 记为"现行"。
> **额外要求 2：按子目录划分场景**——data/docs/制度文件/ 下记 scenario="制度文件"，
> data/docs/运维手册/ 下记 scenario="运维手册"（对应 config.SCENARIOS）；
> 文件名形如"常见报错对照_A学院.md"的，额外记录 org="A学院"。

**验收**：终端执行 `python -c "from app.ingest import load_documents; from pathlib import Path; d=load_documents(Path('data/docs')); print(len(d)); print({x['source']: x.get('version') for x in d[:5]})"`，能看到版本信息。

---

## 第 2 步：切分与向量化入库

**粘贴给 Agent：**

> 实现 app/ingest.py 的 chunk_documents 和 build_vector_store 两个函数（按其中 TODO）：
> 切片按 config.py 的 CHUNK_SIZE/CHUNK_OVERLAP，优先在"第X条"开头处断开；
> 用 sentence-transformers 加载 config.EMBED_MODEL 向量化，
> 存入 Chroma 持久化库（目录 config.CHROMA_DIR），
> metadata 包含 source、page、version、scenario、org（若有）。
> 重复运行 ingest 不应产生重复数据
> （以 source+page+chunk 序号做 id 去重，或先清空重建）。

**验收**：`python app/ingest.py` 跑完，`data/chroma/` 目录有内容，重复跑一次不报错不翻倍。

---

## 第 3 步：检索、生成与可追责拒答

**粘贴给 Agent：**

> 实现 app/rag.py 的 retrieve、ask_llm、answer 三个函数（按其中 TODO）：
> 1. retrieve 加载同一 embedding 模型查询 Chroma，返回 top_k 条 {text, source, version, distance}；
> 2. ask_llm 调用 Ollama（config.OLLAMA_URL + /api/generate，stream=False，model=config.LLM_MODEL）。
>    prompt 结构 = SYSTEM_PROMPT + 编号后的检索条款（每条标注文件名和版本）+ 用户问题。
>    **拒答机制（本项目核心设计）**：prompt 中要求模型先判断所给条款是否足以回答，
>    分三级输出——"依据充分"/"部分依据（需人工核实）"/"依据不足（拒答，建议查阅原文或咨询业务部门）"，
>    绝不允许编造条款内容；
> 3. answer 返回 {answer, confidence, citations}，confidence 取自模型输出的级别判断，
>    citations 为引用条款的 {source, version, snippet} 去重列表。

**验收**：跑 `python -c "from app.rag import answer; r=answer('预算包括哪些'); print(r['confidence'], r['answer'][:200])"`；
再问一个明显超纲的问题（如"差旅费报销标准是多少"），确认它**拒答而不是编造**。

---

## 第 4 步：版本时效管理（差异化特性一）

**粘贴给 Agent：**

> 在 app/rag.py 中新增版本时效逻辑：
> 1. retrieve 结果中同一 source 存在多个 version 时，默认只保留"最新生效版本"的条款
>    （文件名中年份大的视为新版本）；
> 2. 新增函数 compare_versions(source, question)：当用户问题是新旧对比类
>    （包含"修订""变化""改了什么""区别"等关键词）时走此函数，
>    检索该 source 的所有版本条款，prompt 中要求模型分版本对照回答；
> 3. answer() 里加自动路由：检测到对比类问题就调用 compare_versions。
> 在 app/config.py 中如需新增常量（如对比类关键词表）就加在那里。

**验收**：问"预算法对预算公开是怎么规定的"→ 回答只引用 2018 修正版；
问"预算法这次修订主要改了什么"→ 回答分版本对照。

---

## 第 5 步：API 服务

**粘贴给 Agent：**

> 实现 app/main.py 的 ask 和 health 接口（按其中 TODO）。
> /ask 调用 rag.answer 并透传 confidence；向量库为空或 Ollama 未启动时返回
> 带明确提示的错误信息（HTTP 503），不要抛裸异常。
> /health 返回向量库切片数、文档数、模型名。

**验收**：启动 `python app/main.py`，另开终端 `curl "localhost:8000/ask" -X POST -H "Content-Type: application/json" -d '{"question":"预算包括哪些"}'` 返回含 confidence 的 JSON。

---

## 第 6 步：Web 界面

**粘贴给 Agent：**

> 实现 app/ui.py 的 Streamlit 界面（按其中 TODO）：多轮对话、
> 通过 http://localhost:8000/ask 调用后端。要求：
> 1. 每条回答上方用彩色标签显示置信度（绿=依据充分 / 黄=部分依据 / 红=无法回答）；
> 2. 回答下方展开"依据条款"，列出每条引用的文件名、版本、页码、原文片段；
> 3. 侧边栏调 /health 显示服务状态和知识库规模，
>    并提供场景切换（制度问答 / 操作辅助），切换后提问时在请求中带上 scenario 过滤；
> 界面风格简洁正式，适合给政企客户演示。

**验收**：`streamlit run app/ui.py` 浏览器打开，正常问题、超纲问题、版本对比问题各问一遍，标签/引用/版本展示全部正确。

---

## 第 7 步：一键 POC 报告生成器（差异化特性二，售前杀手锏）

**粘贴给 Agent：**

> 新建 app/benchmark.py 和 tests/qa_set.json：
> 1. qa_set.json 预设 24 个测试问答对，分四类：
>    制度常规问答 10 条（附标准答案要点，用于人工核对）、
>    制度版本对比 4 条、
>    应拒答的超纲问题 6 条（标准答案为"应拒答"）、
>    操作辅助 4 条（运维手册场景：报错解读、按学院查配置规则）；
> 2. benchmark.py 逐条跑完整问答，统计：
>    平均端到端延迟（秒）、命中率（人工核对后填）、拒答正确率（超纲题正确拒答的比例）、
>    各置信度分布；用 psutil 采集峰值内存；
> 3. 生成 docs/poc_report.md——**按给客户看的 POC 报告格式写**：
>    测试概述、测试环境（含模型名与硬件）、分类测试结果表、拒答机制验证、
>    数据不出域声明、端侧 vs 云端成本对比结论、待改进项。
>    语气正式，像一份真实交付给客户的测试报告。

**验收**：`python app/benchmark.py` 生成 docs/poc_report.md，打开看内容完整、数字真实、格式像交付物。

---

## 第 8 步：实测数据 + 收尾

**粘贴给 Agent：**

> 检查全部代码：删除多余 TODO 和死代码，确认 README 快速开始命令与实际一致；
> 把 benchmark 输出的关键数字填进 README 第 5 节的表格。

**验收**：README 实测数据表填上了真实数字；按快速开始从零走一遍，命令全部可跑。

---

## 第 9 步：发布

- 录屏：Mac 上 `Cmd+Shift+5` 录 60-90 秒演示——依次演示：
  制度问答（绿标签+引用）、超纲拒答（红标签）、版本对比问答、
  切换到操作辅助场景问一条报错解读、最后展示生成的 POC 报告。放进 README
- `git push` 到 GitHub（命令见 CURSOR_GUIDE.md 第 6 节）
- 简历/打招呼话术加一句：GitHub 开源了 MiniCPM 端侧政企 RAG 问答 demo，含版本时效管理与 POC 报告自动生成（附链接）

---

## 时间预算（业余时间）

| 阶段 | 步骤 | 预估 |
|------|------|------|
| 第一周 | 0-3（基础盘跑通） | 3-4 个晚上 |
| 第二周 | 4-6（差异化特性+界面） | 3-4 个晚上 |
| 第三周 | 7-9（POC 报告+数据+发布） | 2-3 个晚上 |
