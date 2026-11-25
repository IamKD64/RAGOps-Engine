import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from app.chunking import chunk_text
from app.fusion import rrf


def test_chunk_overlap_and_size():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, size=100, overlap=20)
    assert all(len(c.split()) <= 100 for c in chunks)
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]


def test_chunk_empty_and_bad_overlap():
    assert chunk_text("") == []
    with pytest.raises(ValueError):
        chunk_text("a b c", size=10, overlap=10)


def test_rrf_prefers_docs_ranked_high_in_both():
    assert rrf([["a", "b", "c"], ["b", "a", "d"]])[:2] in (["a", "b"], ["b", "a"])
    assert rrf([["x", "y"], ["z", "x"]])[0] == "x"
