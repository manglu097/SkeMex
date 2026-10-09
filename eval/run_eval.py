"""
Unified evaluation entry point.

Routes each dataset to the appropriate evaluator:
  - healthbench   → eval_healthbench.py  (rubric-based, normalised score)
  - LiveMedBench  → eval_livemedbbench.py (rubric-based, normalised score [0,1])
  - all others    → eval_standard.py      (accuracy, MC exact/llm mode)

Usage
-----
# Single dataset
python eval/run_eval.py \
    --dataset healthbench \
    --input   output/predictions.jsonl
    [--config  configs/eval/default.json] \\
    [--model   gpt-4.1-mini] \\
    [--mc_mode exact|llm] \\
    [--output  eval/results/MMMU_eval.jsonl] \\
    [--verbose]

# Batch: evaluate all JSONL files in a directory
python eval/run_eval.py \
    --input_dir ./result \
    --output_dir eval/results/ \\
    [--config configs/eval/default.json]
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import eval_healthbench
import eval_livemedbbench
import eval_standard
from eval_utils import load_jsonl, save_jsonl, print_summary, EvalConfig, init_client

# ---------------------------------------------------------------------------
# Dataset → evaluator routing
# ---------------------------------------------------------------------------
RUBRIC_DATASETS = {"healthbench", "LiveMedBench"}

STANDARD_DATASETS = {
    "MMMU", "MMMU_Pro", "MediQ",
    "AgentClinic_Text", "AgentClinic_MM",
    "LiveClin_Text", "LiveClin_MM",
    "MedXpertQA_Text", "MedXpertQA_MM",
    "MedJourney",
}

ALL_DATASETS = RUBRIC_DATASETS | STANDARD_DATASETS


def infer_dataset_from_filename(path: Path) -> Optional[str]:
    stem = path.stem  # e.g. "MMMU_20260329_205915"
    for ds in ALL_DATASETS:
        if stem.startswith(ds):
            return ds
    return None


# ---------------------------------------------------------------------------
# Single-file evaluation
# ---------------------------------------------------------------------------

def run_one(
    dataset: str,
    input_path: str,
    model: str,
    mc_mode: str,
    output_path: Optional[str],
    verbose: bool,
) -> None:
    records = load_jsonl(input_path)
    print(f"\nEvaluating [{dataset}]  ({len(records)} records)  model={model}")

    if dataset == "healthbench":
        results = eval_healthbench.evaluate(records, model=model, verbose=verbose)
        metrics = eval_healthbench.compute_metrics(results)
    elif dataset == "LiveMedBench":
        results = eval_livemedbbench.evaluate(records, model=model, verbose=verbose)
        metrics = eval_livemedbbench.compute_metrics(results)
    else:
        print(f"MC mode: {mc_mode}")
        results = eval_standard.evaluate(records, dataset=dataset, model=model,
                                         mc_mode=mc_mode, verbose=verbose)
        metrics = eval_standard.compute_metrics(results)

    print_summary(dataset, metrics)

    if output_path:
        save_jsonl(results, output_path)
        print(f"Per-sample results saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified evaluation runner")

    # Single-file mode
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)

    # Batch-directory mode
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)

    # Config and overrides
    parser.add_argument("--config", default=None,
                        help="Path to eval config JSON (default: configs/eval/default.json)")
    parser.add_argument("--model", default=None,
                        help="Override judge model from config")
    parser.add_argument("--mc_mode", default=None, choices=["exact", "llm"],
                        help="Override MC evaluation mode from config")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load config
    config_path = args.config or str(Path(__file__).parent.parent / "configs/eval/default.json")
    cfg = EvalConfig(config_path)
    init_client(cfg)

    model = args.model or cfg.model
    mc_mode = args.mc_mode or cfg.mc_mode

    # ── Batch-directory mode ──────────────────────────────────────────────
    if args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) if args.output_dir else input_dir / "eval_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        jsonl_files = sorted(input_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"No JSONL files found in {input_dir}")
            return

        for jf in jsonl_files:
            dataset = infer_dataset_from_filename(jf)
            if dataset is None:
                print(f"  [SKIP] Cannot infer dataset from filename: {jf.name}")
                continue
            out_path = str(output_dir / f"{jf.stem}_eval.jsonl")
            run_one(dataset, str(jf), model=model, mc_mode=mc_mode,
                    output_path=out_path, verbose=args.verbose)
        return

    # ── Single-file mode ──────────────────────────────────────────────────
    if not args.input:
        parser.error("--input is required when not using --input_dir")
    if not args.dataset:
        dataset = infer_dataset_from_filename(Path(args.input))
        if dataset is None:
            parser.error("--dataset is required (could not infer from filename)")
    else:
        dataset = args.dataset

    run_one(dataset, args.input, model=model, mc_mode=mc_mode,
            output_path=args.output, verbose=args.verbose)


if __name__ == "__main__":
    main()
