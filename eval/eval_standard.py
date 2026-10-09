"""
Standard evaluator for datasets that use a single ground-truth answer.

Supported datasets
------------------
MMMU, MMMU_Pro, MediQ, AgentClinic_Text, AgentClinic_MM,
LiveClin_Text, LiveClin_MM, MedXpertQA_Text, MedXpertQA_MM, MedJourney

Scoring
-------
MC datasets (default): exact string match on the option letter.
MC datasets (llm mode): LLM judge decides whether prediction matches ground truth.
Open-ended datasets: LLM judge, semantic equivalence check.
Final metric is accuracy = correct_count / total.

Usage
-----
python eval/eval_standard.py \
    --input  output/predictions.jsonl \
    --dataset MMMU 
    [--config configs/eval/default.json] \\
    [--model gpt-4.1-mini] \\
    [--output eval/results/MMMU_eval.jsonl]
"""
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from eval_utils import call_llm, extract_json, load_jsonl, save_jsonl, print_summary, mc_exact_match, EvalConfig, init_client
from eval_prompts import (
    MC_JUDGE_SYSTEM, MC_JUDGE_USER,
    OPEN_JUDGE_SYSTEM, OPEN_JUDGE_USER,
)

# Datasets that use open-ended matching (not strict letter comparison)
OPEN_ENDED_DATASETS = {"AgentClinic_Text", "MedJourney"}


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def judge_sample(
    prediction: str,
    ground_truth: Any,
    dataset: str,
    model: str,
    mc_mode: str = "exact",
) -> Dict[str, Any]:
    """Evaluate one sample. Returns a dict with 'correct'."""
    if not prediction:
        return {"correct": False, "raw": ""}

    gt_str = str(ground_truth)
    pred_str = str(prediction)

    # Open-ended datasets always use LLM judge
    if dataset in OPEN_ENDED_DATASETS:
        raw = call_llm(
            messages=[
                {"role": "system", "content": OPEN_JUDGE_SYSTEM},
                {"role": "user", "content": OPEN_JUDGE_USER.format(
                    ground_truth=gt_str, prediction=pred_str)},
            ],
            model=model,
        )
        parsed = extract_json(raw)
        if isinstance(parsed, dict) and "correct" in parsed:
            return {"correct": parsed["correct"] is True, "raw": raw}
        return {"correct": False, "raw": raw, "judge_parse_error": True}

    # MC datasets: exact match (default) or LLM judge
    if mc_mode == "exact":
        correct = mc_exact_match(pred_str, gt_str)
        return {"correct": correct, "raw": "exact_match"}

    # mc_mode == "llm"
    raw = call_llm(
        messages=[
            {"role": "system", "content": MC_JUDGE_SYSTEM},
            {"role": "user", "content": MC_JUDGE_USER.format(
                ground_truth=gt_str, prediction=pred_str)},
        ],
        model=model,
    )
    parsed = extract_json(raw)
    if isinstance(parsed, dict) and "correct" in parsed:
        return {"correct": parsed["correct"] is True, "raw": raw}
    return {"correct": False, "raw": raw, "judge_parse_error": True}


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    records: List[Dict[str, Any]],
    dataset: str,
    model: str,
    mc_mode: str = "exact",
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    results = []
    for i, rec in tqdm(enumerate(records), total=len(records), desc=dataset, unit="sample"):
        if not rec.get("success", True):
            result = {**rec, "eval_correct": False, "eval_raw": "skipped (failed sample)"}
            results.append(result)
            continue

        judgment = judge_sample(
            prediction=rec.get("prediction", ""),
            ground_truth=rec.get("ground_truth", ""),
            dataset=dataset,
            model=model,
            mc_mode=mc_mode,
        )
        result = {**rec, "eval_correct": judgment["correct"], "eval_raw": judgment["raw"]}
        results.append(result)

        if verbose:
            status = "✓" if judgment["correct"] else "✗"
            print(f"  [{i+1:>4}/{len(records)}] {status}  id={rec.get('id','?')}")

    return results


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    success = [r for r in results if r.get("success", True)]
    correct = [r for r in success if r.get("eval_correct")]
    failed = total - len(success)
    accuracy = len(correct) / len(success) if success else 0.0
    return {
        "total_samples": total,
        "evaluated": len(success),
        "skipped_failed": failed,
        "correct": len(correct),
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Standard dataset evaluator")
    parser.add_argument("--input", required=True, help="Path to batch-eval JSONL output")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. MMMU)")
    parser.add_argument("--config", default=None, help="Path to eval config JSON")
    parser.add_argument("--model", default=None, help="Override judge model")
    parser.add_argument("--mc_mode", default=None, choices=["exact", "llm"],
                        help="Override MC evaluation mode")
    parser.add_argument("--output", default=None, help="Path to save per-sample eval JSONL")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config_path = args.config or str(Path(__file__).parent.parent / "configs/eval/default.json")
    cfg = EvalConfig(config_path)
    init_client(cfg)

    model = args.model or cfg.model
    mc_mode = args.mc_mode or cfg.mc_mode

    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} records from {args.input}")
    print(f"MC mode: {mc_mode}  |  Judge model: {model}")

    results = evaluate(records, dataset=args.dataset, model=model,
                       mc_mode=mc_mode, verbose=args.verbose)
    metrics = compute_metrics(results)
    print_summary(args.dataset, metrics)

    if args.output:
        save_jsonl(results, args.output)
        print(f"Per-sample results saved to: {args.output}")


if __name__ == "__main__":
    main()
