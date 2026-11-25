import json
import re

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import CHUNKS_FILE, EMBED_MODEL, INDEX_DIR, QUERY_PREFIX, RERANK_MODEL
from .fusion import rrf

MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self) -> None:
        self.chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        self.by_id = {c["id"]: c for c in self.chunks}
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.col = chromadb.PersistentClient(path=str(INDEX_DIR)).get_collection("docs")
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self._reranker = None

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANK_MODEL)
        return self._reranker

    def _vector(self, query: str, n: int) -> list[str]:
        emb = self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True).tolist()
        return self.col.query(query_embeddings=emb, n_results=n)["ids"][0]

    def _bm25(self, query: str, n: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        return [self.chunks[i]["id"] for i in np.argsort(scores)[::-1][:n]]

    def search(self, query: str, mode: str = "hybrid_rerank", k: int = 5, candidates: int = 20) -> list[dict]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "vector":
            ids = self._vector(query, k)
        elif mode == "bm25":
            ids = self._bm25(query, k)
        else:
            fused = rrf([self._vector(query, candidates), self._bm25(query, candidates)])[:candidates]
            if mode == "hybrid":
                ids = fused[:k]
            else:
                scores = self.reranker.predict([(query, self.by_id[i]["text"]) for i in fused])
                ids = [fused[i] for i in np.argsort(scores)[::-1][:k]]
        return [self.by_id[i] for i in ids]
