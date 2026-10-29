"""FastAPI 服务：/ask 问答、/health 健康检查、/stats 知识库规模、/ingest 重建索引。

启动（在项目根目录）：
    uvicorn app.main:app --port 8000
文档：
    http://localhost:8000/docs
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from . import rag


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """预热：把索引读进内存、把 embedding 模型加载好，首次提问才不会卡十几秒。"""
    try:
        rag.load_index()
        print("[启动] 索引已加载：%s" % rag.knowledge_stats())
    except Exception as exc:  # noqa: BLE001
        print("[启动] 索引未就绪（%s），请先运行 python -m app.ingest" % exc)
    yield


app = FastAPI(
    title="mini-gov-rag",
    description="端侧政企制度智能问答（本地 MiniCPM + 本地向量检索，数据不出域）",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str
    scenario: Optional[str] = None  # 制度文件 / 运维手册；None 表示全库检索


class Citation(BaseModel):
    source: str
    version: str
    page: int
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    confidence: str
    citations: List[Citation]
    top_score: float
    latency_ms: int


@app.get("/health")
def health():
    stats = rag.knowledge_stats()
    return {
        "status": "ok" if stats.get("chunks") else "索引未构建",
        "chunks": stats.get("chunks", 0),
        "llm": rag.LLM_MODEL,
        "ollama": rag.OLLAMA_URL,
    }


@app.get("/stats")
def stats():
    return rag.knowledge_stats()


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    started = time.time()
    result = rag.answer(req.question, scenario=req.scenario)
    result["latency_ms"] = int((time.time() - started) * 1000)
    return result


@app.post("/ingest")
def ingest():
    """重建索引（会重新读 data/docs 下全部文档，耗时几十秒）。"""
    from .ingest import build_index

    stats_out = build_index()
    rag._index_cache["chunks"] = None  # 清缓存，下次请求重新加载
    return {"ok": True, **stats_out}
