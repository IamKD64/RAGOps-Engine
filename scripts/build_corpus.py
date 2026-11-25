"""Rebuild data/raw from open-source AI documentation on GitHub (Apache-2.0 / CC-BY licensed docs).
The cleaned corpus is already committed in data/raw, so you only need this to refresh or extend it.
Add your own sources to SOURCES (any raw GitHub markdown URL)."""
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import RAW_DIR  # noqa: E402

R = "https://raw.githubusercontent.com"
HF = f"{R}/huggingface/transformers/main/docs/source/en"
PEFT = f"{R}/huggingface/peft/main/docs/source"
A2A = f"{R}/a2aproject/A2A/main/docs/topics"
SOURCES = {  # name: (title prefix, url)
    **{f"hf_transformers_{n}": ("Transformers", f"{HF}/{p}.md") for n, p in {
        "quicktour": "quicktour", "pipeline_tutorial": "pipeline_tutorial", "tokenizer_summary": "tokenizer_summary",
        "llm_tutorial": "llm_tutorial", "chat_templating": "chat_templating",
        "generation_strategies": "generation_strategies", "kv_cache": "kv_cache", "perplexity": "perplexity",
        "training": "training", "peft": "peft", "glossary": "glossary",
        "quantization_overview": "quantization/overview", "bitsandbytes": "quantization/bitsandbytes",
        "model_memory": "model_memory_anatomy", "text_classification": "tasks/sequence_classification",
        "question_answering": "tasks/question_answering", "summarization": "tasks/summarization",
        "tasks_language_modeling": "tasks/language_modeling"}.items()},
    **{f"hf_peft_{n}": ("PEFT", f"{PEFT}/{p}.md") for n, p in {
        "quicktour": "quicktour", "quantization": "developer_guides/quantization",
        "troubleshooting": "developer_guides/troubleshooting", "model_merging": "developer_guides/model_merging"}.items()},
    **{f"a2a_{n}": ("A2A Protocol", f"{A2A}/{p}.md") for n, p in {
        "what_is": "what-is-a2a", "key_concepts": "key-concepts", "and_mcp": "a2a-and-mcp",
        "agent_discovery": "agent-discovery", "life_of_a_task": "life-of-a-task",
        "streaming": "streaming-and-async", "enterprise": "enterprise-ready"}.items()},
}


def clean(t: str) -> str:
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"^---.*?---\s*", "", t, flags=re.S)
    t = re.sub(r"```.*?```", "", t, flags=re.S)  # drop code blocks, keep prose
    t = re.sub(r"</?(Tip|Youtube|Note|Warning|hfoptions|hfoption|frameworkcontent|pt|tf|jax|div|iframe|img|details|summary)[^>]*>", "", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"\[\[autodoc\]\].*", "", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, (prefix, url) in SOURCES.items():
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print("skip", name, r.status_code)
            continue
        text = clean(r.text)
        m = re.search(r"^#\s+(.+)$", text, re.M)
        title = f"{prefix}: {(m.group(1) if m else name).replace('`', '').strip()}"
        (RAW_DIR / f"{name}.txt").write_text(f"{title}\n{text}", encoding="utf-8")
        print("saved", title)


if __name__ == "__main__":
    main()
