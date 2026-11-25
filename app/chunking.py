"""Word-window chunking with overlap (no heavy dependencies, easy to unit test)."""


def chunk_text(text: str, size: int = 180, overlap: int = 40, min_words: int = 20) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        # drop tiny trailing fragments (already covered by the overlap)
        if len(window) < min_words and chunks:
            break
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks
