import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import llm
from .retrieval import MODES, Retriever

app = FastAPI(title="Production RAG Assistant", version="1.0")


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    mode: str = "hybrid_rerank"
    k: int = Field(5, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    if req.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    t0 = time.perf_counter()
    ctx = get_retriever().search(req.question, req.mode, req.k)
    t1 = time.perf_counter()
    try:
        text = llm.answer(req.question, ctx)
    except Exception as e:  # Ollama not running etc.
        raise HTTPException(503, f"LLM backend unavailable: {e}")
    t2 = time.perf_counter()
    return {
        "answer": text,
        "sources": [{"source": c["source"], "chunk_id": c["id"], "text": c["text"][:300]} for c in ctx],
        "latency_ms": {"retrieval": round((t1 - t0) * 1000), "generation": round((t2 - t1) * 1000)},
    }
