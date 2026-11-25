"""Build the vector index + BM25 corpus from data/raw/*.txt.
First line of each file = document title, rest = body."""
import json

import chromadb
from sentence_transformers import SentenceTransformer

from .chunking import chunk_text
from .config import CHUNK_OVERLAP, CHUNK_WORDS, CHUNKS_FILE, EMBED_MODEL, INDEX_DIR, RAW_DIR


def main() -> None:
    files = sorted(RAW_DIR.glob("*.txt"))
    if not files:
        raise SystemExit("No files in data/raw. Run: python scripts/download_data.py")
    chunks = []
    for f in files:
        title, _, body = f.read_text(encoding="utf-8").partition("\n")
        for i, c in enumerate(chunk_text(body, CHUNK_WORDS, CHUNK_OVERLAP)):
            chunks.append({"id": f"{f.stem}-{i}", "source": title.strip(), "text": c})
    print(f"{len(files)} documents -> {len(chunks)} chunks")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    embedder = SentenceTransformer(EMBED_MODEL)
    vectors = embedder.encode(
        [c["text"] for c in chunks], batch_size=32, normalize_embeddings=True, show_progress_bar=True
    )
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    try:
        client.delete_collection("docs")
    except Exception:
        pass
    col = client.create_collection("docs", metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), 500):
        batch = chunks[i : i + 500]
        col.add(
            ids=[c["id"] for c in batch],
            embeddings=vectors[i : i + 500].tolist(),
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"]} for c in batch],
        )
    CHUNKS_FILE.write_text(json.dumps(chunks), encoding="utf-8")
    print("Index built.")


if __name__ == "__main__":
    main()
