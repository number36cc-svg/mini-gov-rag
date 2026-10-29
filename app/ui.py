"""Streamlit 对话界面：提问 → 展示回答 + 置信度标签 + 引用出处。

启动（在项目根目录）：
    streamlit run app/ui.py

界面分三块：
- 侧边栏：知识库规模、当前模型、场景筛选——客户演示时一眼看到"库里有几份文件几条款"
- 主区：对话流，每条回答上方带置信度标签（绿/黄/红）
- 回答下方：可展开的"依据条款"，逐条显示 文件名 · 版本 · 页码 · 命中度 · 原文片段
"""
from __future__ import annotations

import os
import sys

import streamlit as st

# 让 streamlit 直接跑脚本时也能 import 到 app 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import rag  # noqa: E402
from app.config import LLM_MODEL, SCENARIOS  # noqa: E402

st.set_page_config(page_title="政企制度问答（端侧 RAG）", page_icon="📄", layout="wide")

CONF_STYLE = {
    "依据充分": ("#EAF3DE", "#3B6D11", "依据充分"),
    "部分依据": ("#FAEEDA", "#854F0B", "部分依据"),
    "无法回答": ("#FCEBEB", "#A32D2D", "无法回答 · 已拒答"),
}


def conf_badge(confidence: str) -> str:
    bg, fg, text = CONF_STYLE.get(confidence, CONF_STYLE["部分依据"])
    return (
        '<span style="background:%s;color:%s;padding:2px 10px;border-radius:10px;'
        'font-size:12px;font-weight:500">%s</span>' % (bg, fg, text)
    )


def render_answer(result: dict) -> None:
    st.markdown(
        conf_badge(result["confidence"])
        + ' <span style="color:#888780;font-size:12px">最高相似度 %.3f</span>'
        % result.get("top_score", 0),
        unsafe_allow_html=True,
    )
    st.markdown(result["answer"])

    cites = result.get("citations") or []
    if cites:
        with st.expander("依据条款（%d 处）" % len(cites)):
            for i, c in enumerate(cites, start=1):
                st.markdown(
                    "**%d. 《%s》%s · 第%s页**　命中度 `%.3f`" % (i, c["source"], c["version"], c["page"], c["score"])
                )
                st.caption(c["excerpt"] + "…")


# ---------------------------------------------------------------- 侧边栏
with st.sidebar:
    st.subheader("知识库")
    stats = rag.knowledge_stats()
    if stats.get("chunks"):
        st.metric("检索切片", stats["chunks"])
        st.metric("向量维度", stats["dim"])
        st.caption("语料：" + "、".join(stats.get("sources", [])))
    else:
        st.warning("索引未构建，请先运行：\n\n`python -m app.ingest`")

    st.divider()
    scenario = st.selectbox("检索范围", ["全部"] + SCENARIOS)
    st.caption("本地模型：`%s`（Ollama）" % LLM_MODEL)
    st.caption("数据全程本地，不出内网。")

    st.divider()
    st.caption("示例问题（点击直接提问，演示用）")
    EXAMPLES = [
        "各级一般公共预算应当按照什么比例设置预备费？",
        "预算法第五十一条在两个版本中引用的条款号一样吗？",
        "提示「导出失败：超出导出条数上限」应该怎么处理？",
        "预算法第 200 条规定了什么内容？",
    ]
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex, key="ex%d" % i, use_container_width=True):
            st.session_state.pending_question = ex

# ---------------------------------------------------------------- 主区
st.title("政企制度智能问答")
st.caption("本地端侧模型 + 本地向量检索 · 每条回答可溯源到条款页码 · 无依据时主动拒答")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("result"):
            render_answer(msg["result"])
        else:
            st.markdown(msg["content"])

question = st.session_state.pop("pending_question", None) or st.chat_input(
    "例如：预备费按本级一般公共预算支出额的百分之几设置？"
)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not stats.get("chunks"):
            result = {"answer": "索引未构建，请先运行 `python -m app.ingest`", "confidence": "无法回答",
                      "citations": [], "top_score": 0}
        else:
            with st.spinner("检索制度条款中…"):
                result = rag.answer(question, scenario=None if scenario == "全部" else scenario)
        render_answer(result)

    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "result": result})
