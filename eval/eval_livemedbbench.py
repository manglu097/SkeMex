"""
LiveMedBench evaluator.

Scoring formula (per sample)
-----------------------------
For each rubric item the LLM judge returns met (bool).

    score_i          = 1 if met_i else 0
    weighted_score_i = points_i × score_i
    numerator        = Σ weighted_score_i
    denominator      = Σ max(points_i, 0)   [positive rubrics only]
    case_total       = clamp(numerator / denominator, 0.0, 1.0)

The benchmark score is the mean of all case_total values.

Ground-truth field
------------------
``record["ground_truth"]`` is a list of rubric dicts, each with:
    - "criterion" : str
    - "points"    : int  (can be negative for undesirable behaviours)

Usage
-----
python eval/eval_livemedbbench.py \\
    --input  output/batch_eval/LiveMedBench_20260329.jsonl \\
    [--model gpt-4.1-mini] \\
    [--output eval/results/livemedbbench_eval.jsonl]
"""
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from eval_utils import call_llm, extract_json, load_jsonl, save_jsonl, print_summary, EvalConfig, init_client
from eval_prompts import LMB_GRADER_SYSTEM, LMB_GRADER_USER


# ---------------------------------------------------------------------------
# Batch rubric grading (one LLM call for all rubrics)
# ---------------------------------------------------------------------------

def grade_all_rubrics(
    query: str,
    prediction: str,
    rubrics: List[Dict[str, Any]],
    model: str,
) -> List[bool]:
    """One LLM call to grade all rubric items. Returns a bool list in rubric order."""
    criteria_list = "\n".join(
        f"{i+1}. {r.get('criterion', '')}" for i, r in enumerate(rubrics)
    )
    raw = call_llm(
        messages=[
            {"role": "system", "content": LMB_GRADER_SYSTEM},
            {"role": "user", "content": LMB_GRADER_USER.format(
                query=query,
                prediction=prediction,
                criteria_list=criteria_list,
            )},
        ],
        model=model,
    )
    parsed = extract_json(raw)
    if isinstance(parsed, list) and len(parsed) == len(rubrics):
        return [bool(v) for v in parsed]
    return [False] * len(rubrics)


# ---------------------------------------------------------------------------
# Per-sample scoring
# ---------------------------------------------------------------------------

def score_sample(
    record: Dict[str, Any],
    model: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Grade all rubric items in one LLM call and compute case_total."""
    query = record.get("query", "")
    prediction = record.get("prediction", "") or ""
    rubrics: List[Dict[str, Any]] = record.get("ground_truth", [])

    if not rubrics:
        return {**record, "rubric_results": [], "case_total": 0.0}

    met_list = grade_all_rubrics(query, prediction, rubrics, model) if prediction else [False] * len(rubrics)

    rubric_results = []
    numerator = 0.0
    denominator = 0.0

    for item, met in zip(rubrics, met_list):
        points = float(item.get("points", 0))
        if points > 0:
            denominator += points
        weighted = points * (1.0 if met else 0.0)
        numerator += weighted
        rubric_results.append({
            "criterion": item.get("criterion", ""),
            "points": points,
            "met": met,
            "weighted_score": weighted,
        })
        if verbose:
            status = "✓" if met else "✗"
            print(f"      {status} [{points:+.0f} → {weighted:+.0f}] {item.get('criterion', '')[:80]}")

    case_total = min(max(numerator / denominator, 0.0), 1.0) if denominator > 0 else 0.0
    return {**record, "rubric_results": rubric_results, "case_total": case_total}


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    records: List[Dict[str, Any]],
    model: str,
    verbose: bool = False,
    desc: str = "LiveMedBench",
) -> List[Dict[str, Any]]:
    results = []
    for i, rec in tqdm(enumerate(records), total=len(records), desc=desc, unit="sample"):
        if verbose:
            print(f"\n[{i+1}/{len(records)}] id={rec.get('id','?')}")
        if not rec.get("success", True):
            results.append({**rec, "rubric_results": [], "case_total": 0.0})
            continue
        scored = score_sample(rec, model=model, verbose=verbose)
        results.append(scored)
    return results


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    success = [r for r in results if r.get("success", True)]
    failed = total - len(success)
    totals = [r["case_total"] for r in success if "case_total" in r]
    mean_case_total = sum(totals) / len(totals) if totals else 0.0
    return {
        "total_samples": total,
        "evaluated": len(success),
        "skipped_failed": failed,
        "mean_case_total": mean_case_total,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LiveMedBench evaluator")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default=None, help="Path to eval config JSON")
    parser.add_argument("--model", default=None, help="Override judge model")
    parser.add_argument("--output", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config_path = args.config or str(Path(__file__).parent.parent / "configs/eval/default.json")
    cfg = EvalConfig(config_path)
    init_client(cfg)
    model = args.model or cfg.model

    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    results = evaluate(records, model=model, verbose=args.verbose)
    metrics = compute_metrics(results)
    print_summary("LiveMedBench", metrics)

    if args.output:
        save_jsonl(results, args.output)
        print(f"Per-sample results saved to: {args.output}")


if __name__ == "__main__":
    main()
