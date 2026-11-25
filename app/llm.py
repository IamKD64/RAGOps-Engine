import requests

from .config import LLM_MODEL, OLLAMA_URL

SYSTEM = (
    "You answer questions using ONLY the numbered context passages. "
    "Cite passages like [1], [2]. If the answer is not in the context, say \"I don't know based on the provided documents.\""
)


def build_prompt(question: str, contexts: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] ({c['source']}) {c['text']}" for i, c in enumerate(contexts))
    return f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"


def chat(messages: list[dict], temperature: float = 0.1) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": LLM_MODEL, "messages": messages, "stream": False, "options": {"temperature": temperature}},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def answer(question: str, contexts: list[dict]) -> str:
    return chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(question, contexts)}])
