"""Retrieval evaluation (+ optional LLM-judge faithfulness).
Usage (from project1-rag/):
  python -m eval.run_eval                       # compare all retrieval modes
  python -m eval.run_eval --min-recall 0.7      # CI gate on hybrid_rerank recall@k
  python -m eval.run_eval --judge 10            # also judge faithfulness on 10 answers (needs Ollama)
"""
import argparse
import json
import sys
from pathlib import Path

from app.retrieval import MODES, Retriever

GOLD = Path(__file__).parent / "golden.jsonl"


def metrics(retriever: Retriever, gold: list[dict], mode: str, k: int, tier: str = "all") -> dict:
    gold = [g for g in gold if tier == "all" or g.get("tier") == tier]
    hits, rr = 0, 0.0
    for g in gold:
        sources = [c["source"] for c in retriever.search(g["q"], mode, k)]
        if g["source"] in sources:
            hits += 1
            rr += 1.0 / (sources.index(g["source"]) + 1)
    n = len(gold)
    return {"mode": mode, "tier": tier, f"recall@{k}": round(hits / n, 3), "mrr": round(rr / n, 3)}


def judge_faithfulness(retriever: Retriever, gold: list[dict], n: int, k: int) -> float:
    from app import llm

    ok = 0
    for g in gold[:n]:
        ctx = retriever.search(g["q"], "hybrid_rerank", k)
        ans = llm.answer(g["q"], ctx)
        verdict = llm.chat([{"role": "user", "content":
            "Context:\n" + "\n".join(c["text"] for c in ctx) + f"\n\nAnswer:\n{ans}\n\n"
            "Is every claim in the answer supported by the context? Reply with only YES or NO."}], temperature=0)
        ok += verdict.upper().startswith("YES")
    return round(ok / n, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-recall", type=float, default=0.0)
    ap.add_argument("--judge", type=int, default=0)
    args = ap.parse_args()

    gold = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    r = Retriever()
    results = [metrics(r, gold, m, args.k, t) for t in ("all", "easy", "hard") for m in MODES]
    print(f"{'tier':<7}{'mode':<15}{'recall@' + str(args.k):<12}{'mrr':<8}")
    for row in results:
        print(f"{row['tier']:<7}{row['mode']:<15}{row[f'recall@{args.k}']:<12}{row['mrr']:<8}")
    out = {"k": args.k, "n_questions": len(gold), "retrieval": results}
    if args.judge:
        out["faithfulness"] = judge_faithfulness(r, gold, args.judge, args.k)
        print("faithfulness:", out["faithfulness"])
    Path(__file__).with_name("results.json").write_text(json.dumps(out, indent=2))

    best = next(x for x in results if x["mode"] == "hybrid_rerank" and x["tier"] == "all")[f"recall@{args.k}"]
    if best < args.min_recall:
        print(f"FAIL: hybrid_rerank recall {best} < {args.min_recall}")
        sys.exit(1)


if __name__ == "__main__":
    main()
