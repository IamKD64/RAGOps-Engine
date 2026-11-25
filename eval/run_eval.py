
"""RAG retrieval evaluation with document-level and chunk-level metrics.

Usage:
  python -m eval.run_eval
  python -m eval.run_eval --k 5 --bootstrap-samples 2000
  python -m eval.run_eval --min-recall 0.95
  python -m eval.run_eval --judge 10

Chunk-level evaluation requires 'relevant_chunk_ids' in golden.jsonl.
Faithfulness judging currently uses the configured Ollama model and is
therefore a preliminary self-judge, not an independent evaluation.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from app.retrieval import MODES, Retriever

GOLD = Path(__file__).parent / "golden.jsonl"
RESULTS = Path(__file__).with_name("results.json")


def load_gold() -> list[dict]:
    """Load and validate benchmark questions."""
    if not GOLD.exists():
        raise FileNotFoundError(f"Benchmark not found: {GOLD}")

    questions = [
        json.loads(line)
        for line in GOLD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not questions:
        raise ValueError("Benchmark contains no questions.")

    for index, question in enumerate(questions, start=1):
        for field in ("q", "source", "tier"):
            if not question.get(field):
                raise ValueError(
                    f"Question {index} is missing required field: {field}"
                )

        if "relevant_chunk_ids" in question:
            ids = question["relevant_chunk_ids"]
            if not isinstance(ids, list) or not ids:
                raise ValueError(
                    f"Question {index}: relevant_chunk_ids must be "
                    "a non-empty list when provided."
                )

    return questions


def bootstrap_ci(
    values: list[float],
    samples: int = 2000,
    seed: int = 42,
) -> dict:
    """Calculate percentile-based 95% bootstrap confidence intervals."""
    if not values:
        return {"lower": None, "upper": None}

    rng = random.Random(seed)
    n = len(values)
    means = []

    for _ in range(samples):
        sampled = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sampled) / n)

    means.sort()

    return {
        "lower": round(means[int(0.025 * samples)], 4),
        "upper": round(means[min(int(0.975 * samples), samples - 1)], 4),
    }


def summarize(values: list[float], samples: int) -> dict:
    """Summarize a metric and its confidence interval."""
    if not values:
        return {
            "n": 0,
            "mean": None,
            "ci_95": {"lower": None, "upper": None},
        }

    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 4),
        "ci_95": bootstrap_ci(values, samples),
    }


def document_metrics(
    retriever: Retriever,
    questions: list[dict],
    mode: str,
    k: int,
    samples: int,
) -> dict:
    """Measure whether the expected source document is retrieved."""
    hits = []
    reciprocal_ranks = []

    for question in questions:
        results = retriever.search(question["q"], mode, k)
        sources = [chunk["source"] for chunk in results]

        if question["source"] in sources:
            rank = sources.index(question["source"]) + 1
            hits.append(1.0)
            reciprocal_ranks.append(1.0 / rank)
        else:
            hits.append(0.0)
            reciprocal_ranks.append(0.0)

    return {
        "mode": mode,
        "n_questions": len(questions),
        f"recall@{k}": summarize(hits, samples),
        "mrr": summarize(reciprocal_ranks, samples),
    }


def chunk_metrics(
    retriever: Retriever,
    questions: list[dict],
    mode: str,
    k: int,
    samples: int,
) -> dict:
    """Measure retrieval of manually labeled answer-bearing chunks."""
    hits = []
    coverage = []
    reciprocal_ranks = []

    for question in questions:
        relevant = set(question["relevant_chunk_ids"])
        results = retriever.search(question["q"], mode, k)
        retrieved = [chunk["id"] for chunk in results]

        relevant_ranks = [
            rank
            for rank, chunk_id in enumerate(retrieved, start=1)
            if chunk_id in relevant
        ]

        hits.append(1.0 if relevant_ranks else 0.0)

        retrieved_relevant = len(set(retrieved) & relevant)
        coverage.append(retrieved_relevant / len(relevant))

        reciprocal_ranks.append(
            1.0 / relevant_ranks[0] if relevant_ranks else 0.0
        )

    return {
        "mode": mode,
        "n_questions": len(questions),
        f"hit_recall@{k}": summarize(hits, samples),
        f"chunk_coverage_recall@{k}": summarize(coverage, samples),
        "mrr": summarize(reciprocal_ranks, samples),
    }


def judge_faithfulness(
    retriever: Retriever,
    questions: list[dict],
    n: int,
    k: int,
) -> dict:
    """Preliminary faithfulness check using the configured Ollama model."""
    from app import llm

    selected = questions[:n]
    passed = 0
    evaluated = 0

    for question in selected:
        context = retriever.search(
            question["q"],
            mode="hybrid_rerank",
            k=k,
        )

        answer = llm.answer(question["q"], context)

        prompt = (
            "Evaluate whether every factual claim in the answer is "
            "supported by the supplied context.\n\n"
            "Context:\n"
            + "\n\n".join(chunk["text"] for chunk in context)
            + f"\n\nAnswer:\n{answer}\n\n"
            "Reply with only YES or NO."
        )

        verdict = llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0,
        ).strip().upper()

        evaluated += 1

        if verdict.startswith("YES"):
            passed += 1

    return {
        "judge": "configured_ollama_model",
        "independent_judge": False,
        "evaluated": evaluated,
        "passed": passed,
        "score": round(passed / evaluated, 4) if evaluated else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval quality."
    )

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--judge", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)

    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be at least 1")

    if args.judge < 0:
        parser.error("--judge cannot be negative")

    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be at least 1")

    if not 0.0 <= args.min_recall <= 1.0:
        parser.error("--min-recall must be between 0 and 1")

    gold = load_gold()

    annotated = [
        question
        for question in gold
        if question.get("relevant_chunk_ids")
    ]

    print(f"Loaded {len(gold)} benchmark questions.")
    print(f"Questions with chunk annotations: {len(annotated)}")

    retriever = Retriever()

    document_results = []
    chunk_results = []

    for tier in ("all", "easy", "hard"):
        questions = [
            question
            for question in gold
            if tier == "all" or question.get("tier") == tier
        ]

        chunk_questions = [
            question
            for question in questions
            if question.get("relevant_chunk_ids")
        ]

        for mode in MODES:
            document_results.append({
                "tier": tier,
                **document_metrics(
                    retriever,
                    questions,
                    mode,
                    args.k,
                    args.bootstrap_samples,
                ),
            })

            if chunk_questions:
                chunk_results.append({
                    "tier": tier,
                    **chunk_metrics(
                        retriever,
                        chunk_questions,
                        mode,
                        args.k,
                        args.bootstrap_samples,
                    ),
                })

    print("\nDOCUMENT-LEVEL RESULTS")
    print(f"{'Tier':<8}{'Mode':<17}{'Recall':<12}{'MRR':<12}")

    for result in document_results:
        recall = result[f"recall@{args.k}"]["mean"]
        mrr = result["mrr"]["mean"]

        print(
            f"{result['tier']:<8}"
            f"{result['mode']:<17}"
            f"{recall:.3f}       "
            f"{mrr:.3f}"
        )

    print(
        f"\nCHUNK-LEVEL RESULTS "
        f"({len(annotated)}/{len(gold)} annotated questions)"
    )

    for result in chunk_results:
        hit = result[f"hit_recall@{args.k}"]["mean"]
        coverage = result[f"chunk_coverage_recall@{args.k}"]["mean"]
        mrr = result["mrr"]["mean"]

        print(
            f"{result['tier']:<8}"
            f"{result['mode']:<17}"
            f"Hit Recall: {hit:.3f}  "
            f"Coverage: {coverage:.3f}  "
            f"MRR: {mrr:.3f}"
        )

    output = {
        "k": args.k,
        "bootstrap_samples": args.bootstrap_samples,
        "n_questions": len(gold),
        "n_chunk_annotated": len(annotated),
        "retrieval": document_results,
        "chunk_retrieval": chunk_results,
    }

    if args.judge:
        print("\nRunning preliminary faithfulness evaluation...")
        output["faithfulness"] = judge_faithfulness(
            retriever,
            gold,
            args.judge,
            args.k,
        )
        print("Faithfulness:", output["faithfulness"])

    RESULTS.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {RESULTS}")

    baseline = next(
        result
        for result in document_results
        if result["mode"] == "hybrid_rerank"
        and result["tier"] == "all"
    )

    score = baseline[f"recall@{args.k}"]["mean"]

    if score < args.min_recall:
        print(
            f"FAIL: hybrid_rerank Recall@{args.k} "
            f"{score:.3f} < {args.min_recall}"
        )
        sys.exit(1)

    print("Evaluation completed.")


if __name__ == "__main__":
    main()
