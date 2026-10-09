"""Batch evaluation script for all datasets.

New features vs. original:
  --dataset           now OPTIONAL; omit to run ALL datasets under --data_root
  --data_root         now reads evo_config.offline.test_root as default (CLI overrides)
  --mode react|cot    react = full tool-using agent (default); cot = pure chain-of-thought,
                      no tools, single-turn LLM call
  --llm_config        now reads configs/llms/demo.json by default (was hardcoded)
  --global_config     now reads configs/global_config.json by default (was hardcoded)
  --evo_config        reads configs/evolution/offline.json for test_root default
  --resume            skip already-completed samples and append to the existing output file
  --retry_failed      re-run samples whose "success" is false in existing output files and
                      replace the failed records in-place; the output file is rewritten
                      atomically so no data is lost on interruption
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from medmem_agent.agent import AgentLoop
from medmem_agent.dataset_loader import DATASET_LOADERS
from medmem_agent.evolution.config import load_evo_config
from medmem_agent.evolution.dataset_mix import (
    DATASET_FILES,
    discover_dataset_paths,
    resolve_selected_benches,
)
from medmem_agent.llm import build_llm
from medmem_agent.memory import ConversationMemory
from medmem_agent.tools.registry import ToolRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure CoT system prompt (no tools)
# ---------------------------------------------------------------------------

COT_SYSTEM_PROMPT = """You are a knowledgeable medical assistant.
Answer the question directly using your internal knowledge and reasoning.
Do NOT call any external tools.

## Output Format — STRICT RULES
Your response MUST follow this structure:

  <reasoning> … </reasoning>   ← REQUIRED. Think step by step concisely.
  <response> … </response>     ← REQUIRED. Your final answer.

Rules:
1. <reasoning> must appear before <response>.
2. No text outside the tags.
3. Do NOT output <tool> blocks — no tools are available.
4. Provide a complete, evidence-based answer inside <response>.
"""


# ---------------------------------------------------------------------------
# Single-sample runners
# ---------------------------------------------------------------------------

def run_single_sample(
    agent: AgentLoop,
    sample: Dict[str, Any],
    dataset_name: str,
    supports_vision: bool,
) -> Dict[str, Any]:
    """Run the full ReAct agent on a single sample."""
    try:
        agent.memory.clear()
        agent._steps_history = []
        runtime_ctx = sample.get("runtime_context") or None
        sample_start = time.monotonic()
        result = agent.run(
            sample["query"],
            images=sample["images"],
            tool_runtime_context=runtime_ctx,
        )
        total_duration_s = round(time.monotonic() - sample_start, 3)
        return {
            "id": sample["id"],
            "dataset": dataset_name,
            "query": sample["query"],
            "images": sample["images"],
            "prediction": result,
            "ground_truth": sample["ground_truth"],
            "steps": len(agent._steps_history),
            "total_duration_s": total_duration_s,
            "success": True,
            "metadata": sample["metadata"],
        }
    except Exception as exc:
        logger.error(f"Error processing sample {sample['id']}: {exc}")
        return {
            "id": sample["id"],
            "dataset": dataset_name,
            "query": sample["query"],
            "prediction": None,
            "ground_truth": sample["ground_truth"],
            "steps": 0,
            "total_duration_s": 0.0,
            "success": False,
            "error": str(exc),
            "metadata": sample["metadata"],
        }


def run_single_sample_cot(
    llm,
    sample: Dict[str, Any],
    dataset_name: str,
    supports_vision: bool,
) -> Dict[str, Any]:
    """Run a pure CoT (no-tool) single-turn LLM call on a single sample.

    The LLM receives only the system prompt + user query.  No tool registry,
    no memory, no multi-turn loop.
    """
    try:
        sample_start = time.monotonic()
        images = sample.get("images") or []
        if not supports_vision:
            images = []
        response = llm.generate(
            system_prompt=COT_SYSTEM_PROMPT,
            user_prompt=sample["query"],
            images=images,
        )
        total_duration_s = round(time.monotonic() - sample_start, 3)

        # Extract <response>…</response> if present, otherwise use raw content
        content = response.content.strip()
        import re
        m = re.search(r"<response>(.*?)</response>", content, re.DOTALL)
        prediction = m.group(1).strip() if m else content

        return {
            "id": sample["id"],
            "dataset": dataset_name,
            "query": sample["query"],
            "images": sample.get("images", []),
            "prediction": prediction,
            "ground_truth": sample["ground_truth"],
            "steps": 1,
            "total_duration_s": total_duration_s,
            "success": True,
            "metadata": sample["metadata"],
            "mode": "cot",
        }
    except Exception as exc:
        logger.error(f"Error processing sample {sample['id']}: {exc}")
        return {
            "id": sample["id"],
            "dataset": dataset_name,
            "query": sample["query"],
            "prediction": None,
            "ground_truth": sample["ground_truth"],
            "steps": 0,
            "total_duration_s": 0.0,
            "success": False,
            "error": str(exc),
            "metadata": sample["metadata"],
            "mode": "cot",
        }


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _load_completed_queries_from_file(path: Path) -> Set[str]:
    """Parse a single JSONL file and return the set of completed sample queries.

    ``query`` is used instead of ``id`` because some dataset loaders
    (AgentClinic_Text, MedJourney) generate ids via Python's built-in
    ``hash()``, which is re-seeded randomly on every process start
    (PYTHONHASHSEED).  This makes hash-based ids unreliable across runs and
    causes --resume to re-run already-completed samples.  ``query`` is a
    stable, verbatim string copied from the dataset file and is therefore a
    safe deduplication key.

    Truncated or malformed lines are silently skipped so that a partial last
    line written during a crash does not block the resume.
    """
    completed: Set[str] = set()
    if not path.exists():
        return completed
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                query = record.get("query")
                if query is not None:
                    completed.add(str(query))
            except json.JSONDecodeError:
                pass
    return completed


def _collect_completed_queries(output_dir: Path, dataset_name: str, mode: str) -> Set[str]:
    """Return the union of completed sample queries found across **all** output
    files for the given dataset and mode in *output_dir*.

    Two file-naming conventions are recognised so that both old runs
    (timestamped names like ``{dataset}_{mode}_20260414_170054.jsonl``) and
    new resume runs (stable name ``{dataset}_{mode}.jsonl``) are picked up
    automatically:

    * ``{dataset}_{mode}.jsonl``          — stable resume file
    * ``{dataset}_{mode}_*.jsonl``        — any timestamped file from a
                                            previous run
    """
    completed: Set[str] = set()
    prefix = f"{dataset_name}_{mode}"
    for path in output_dir.glob(f"{prefix}*.jsonl"):
        queries = _load_completed_queries_from_file(path)
        if queries:
            logger.info(
                f"[resume] Reading {len(queries)} completed query/queries from {path.name}"
            )
        completed |= queries
    return completed


def _resume_output_file(output_dir: Path, dataset_name: str, mode: str) -> Path:
    """Return the stable (timestamp-free) output file path used for resume runs.

    New results are always appended to this fixed-name file regardless of
    whether prior results came from timestamped files.
    """
    return output_dir / f"{dataset_name}_{mode}.jsonl"


# ---------------------------------------------------------------------------
# Retry-failed helpers
# ---------------------------------------------------------------------------

def _find_output_file_for_dataset(output_dir: Path, dataset_name: str, mode: str) -> Optional[Path]:
    """Locate the most recent output JSONL file for a dataset.

    Search order:
    1. Stable name ``{dataset}_{mode}.jsonl`` (created by --resume runs).
    2. Most recently modified timestamped file ``{dataset}_{mode}_*.jsonl``.

    Returns ``None`` if no matching file exists.
    """
    stable = output_dir / f"{dataset_name}_{mode}.jsonl"
    if stable.exists():
        return stable
    candidates = sorted(
        output_dir.glob(f"{dataset_name}_{mode}_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_all_records(path: Path) -> List[Dict[str, Any]]:
    """Read every valid JSON line from *path* into a list.

    Malformed / truncated trailing lines are silently dropped.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _rewrite_file_atomically(path: Path, records: List[Dict[str, Any]]) -> None:
    """Overwrite *path* with *records* using a write-then-rename strategy.

    A temporary sibling file is written first so that a crash mid-write
    never leaves *path* in a partially written state.
    """
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _rebuild_sample_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct a runnable sample dict from a previously saved output record.

    The output record already contains every field needed to re-run the sample
    (``id``, ``query``, ``images``, ``ground_truth``, ``metadata``) so we can
    bypass the dataset loader entirely.  This avoids the id-mismatch problem
    that arises when the loader generates ids via ``hash()`` (which is
    randomised per process in Python 3.3+).

    ``runtime_context`` is not stored in the output record; it is set to
    ``None`` here.  For datasets that require a runtime context (e.g.
    AgentClinic) the agent will simply run without it — acceptable for a
    retry because the original run also only had the context available through
    the tool registry, not through the record itself.
    """
    return {
        "id": record["id"],
        "query": record.get("query", ""),
        "images": record.get("images") or [],
        "ground_truth": record.get("ground_truth"),
        "metadata": record.get("metadata") or {},
        "runtime_context": None,
    }


def _retry_failed_dataset(
    dataset_name: str,
    data_path: str,
    mode: str,
    llm,
    llm_config,
    global_config,
    output_dir: Path,
    max_samples: Optional[int],
) -> None:
    """Re-run failed samples for one dataset and replace their records in-place.

    Steps:
    1. Locate the existing output file for this dataset.
    2. Load all records; identify those with ``success == False``.
    3. Reconstruct runnable sample dicts directly from the failed records
       (using ``_rebuild_sample_from_record``) — this avoids id-mismatch
       issues caused by hash-based ids that differ across process runs.
    4. Re-run each failed sample; replace the old record with the new result.
    5. Atomically rewrite the output file with the updated records.
    """
    output_file = _find_output_file_for_dataset(output_dir, dataset_name, mode)
    if output_file is None:
        logger.warning(
            f"[{dataset_name}] [retry_failed] No existing output file found; skipping."
        )
        return

    existing_records = _load_all_records(output_file)
    failed_records = [r for r in existing_records if not r.get("success", True)]

    if not failed_records:
        logger.info(f"[{dataset_name}] [retry_failed] No failed samples found; nothing to do.")
        return

    failed_ids = [str(r["id"]) for r in failed_records]
    logger.info(
        f"[{dataset_name}] [retry_failed] Found {len(failed_records)} failed sample(s): "
        f"{failed_ids}"
    )

    supports_vision = bool(getattr(llm_config, "supports_vision", False))

    # Build agent only for react mode.
    agent = None
    if mode == "react":
        registry = ToolRegistry.from_config(
            f"configs/datasets/{dataset_name}/tools.json",
            global_tools_config=global_config.tools,
        )
        memory = ConversationMemory()
        agent = AgentLoop(
            llm=llm,
            tool_registry=registry,
            memory=memory,
            max_steps=global_config.max_steps,
            output_dir=global_config.output_dir,
            history_filename=global_config.history_filename,
            history_save_mode=global_config.history_save_mode,
        )

    # Re-run each failed sample directly from the saved record fields.
    # Keyed by id for fast lookup during the replacement step.
    new_results: Dict[str, Dict[str, Any]] = {}
    for record in failed_records:
        sid = str(record["id"])
        sample = _rebuild_sample_from_record(record)
        logger.info(f"[{dataset_name}] [retry_failed] Re-running sample {sid} ...")
        if mode == "react":
            result = run_single_sample(agent, sample, dataset_name, supports_vision)
        else:
            result = run_single_sample_cot(llm, sample, dataset_name, supports_vision)
        new_results[sid] = result
        logger.info(
            f"[{dataset_name}] [retry_failed] Sample {sid}: "
            f"success={result['success']}, duration={result['total_duration_s']}s"
        )

    # Replace failed records with new results; preserve original order.
    updated_records = []
    for record in existing_records:
        sid = str(record.get("id", ""))
        if sid in new_results:
            updated_records.append(new_results[sid])
        else:
            updated_records.append(record)

    _rewrite_file_atomically(output_file, updated_records)

    success_count = sum(1 for r in new_results.values() if r["success"])
    logger.info(
        f"[{dataset_name}] [retry_failed] Done. "
        f"Re-ran {len(new_results)} sample(s): "
        f"{success_count} now succeeded, {len(new_results) - success_count} still failed. "
        f"Output file updated: {output_file}"
    )


# ---------------------------------------------------------------------------
# Per-dataset runner (shared by both modes)
# ---------------------------------------------------------------------------

def _run_dataset(
    dataset_name: str,
    data_path: str,
    mode: str,
    llm,
    llm_config,
    global_config,
    max_samples: Optional[int],
    output_dir: Path,
    resume: bool = False,
) -> None:
    """Load, run, and save results for one dataset."""
    loader_class = DATASET_LOADERS[dataset_name]
    loader = loader_class(str(data_path))
    samples = loader.load_samples()
    if max_samples:
        samples = samples[:max_samples]
    logger.info(f"[{dataset_name}] Loaded {len(samples)} samples (mode={mode})")

    supports_vision = bool(getattr(llm_config, "supports_vision", False))

    # Build ReAct agent only when needed
    agent = None
    if mode == "react":
        registry = ToolRegistry.from_config(
            f"configs/datasets/{dataset_name}/tools.json",
            global_tools_config=global_config.tools,
        )
        memory = ConversationMemory()
        agent = AgentLoop(
            llm=llm,
            tool_registry=registry,
            memory=memory,
            max_steps=global_config.max_steps,
            output_dir=global_config.output_dir,
            history_filename=global_config.history_filename,
            history_save_mode=global_config.history_save_mode,
        )

    # ------------------------------------------------------------------
    # Determine output file path.
    #
    # * resume=True  → stable name (no timestamp) so we can find and
    #                  append to the same file across runs.
    # * resume=False → timestamped name (original behaviour).
    # ------------------------------------------------------------------
    if resume:
        output_file = _resume_output_file(output_dir, dataset_name, mode)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{dataset_name}_{mode}_{timestamp}.jsonl"
    logger.info(f"[{dataset_name}] Results → {output_file}")

    # ------------------------------------------------------------------
    # Resume: collect already-completed sample queries from ALL existing
    # output files for this dataset (both timestamped and stable names).
    # query is used instead of id because some loaders (AgentClinic_Text,
    # MedJourney) generate ids via Python's hash(), which is re-seeded
    # randomly per process and therefore unreliable across runs.
    # ------------------------------------------------------------------
    completed_queries: Set[str] = set()
    if resume:
        completed_queries = _collect_completed_queries(output_dir, dataset_name, mode)
        if completed_queries:
            logger.info(
                f"[{dataset_name}] [resume] Found {len(completed_queries)} completed "
                f"sample(s) in total; skipping them."
            )
        else:
            logger.info(
                f"[{dataset_name}] [resume] No prior results found; starting fresh."
            )

    results: List[Dict[str, Any]] = []
    batch_start = time.monotonic()
    skipped = 0

    for i, sample in enumerate(samples):
        sample_query = str(sample.get("query", ""))

        # Skip already-completed samples when resuming (matched by query).
        if resume and sample_query in completed_queries:
            skipped += 1
            continue

        logger.info(f"[{dataset_name}] Sample {i + 1}/{len(samples)}: {sample['id']}")
        if mode == "react":
            result = run_single_sample(agent, sample, dataset_name, supports_vision)
        else:
            result = run_single_sample_cot(llm, sample, dataset_name, supports_vision)
        results.append(result)
        with open(output_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
        logger.info(
            f"[{dataset_name}] Sample {i + 1} saved "
            f"(success={result['success']}, duration={result['total_duration_s']}s)"
        )

    if resume:
        logger.info(
            f"[{dataset_name}] [resume] skipped={skipped}, "
            f"newly processed={len(results)}"
        )

    batch_duration = round(time.monotonic() - batch_start, 3)
    success_count = sum(1 for r in results if r["success"])
    successful_durations = [r["total_duration_s"] for r in results if r["success"]]
    avg_duration = (
        round(sum(successful_durations) / len(successful_durations), 3)
        if successful_durations
        else 0.0
    )
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Dataset:          {dataset_name}")
    logger.info(f"Mode:             {mode}")
    logger.info(f"Total samples:    {len(results)}")
    logger.info(f"Successful:       {success_count}")
    logger.info(f"Failed:           {len(results) - success_count}")
    logger.info(f"Avg duration:     {avg_duration}s / sample")
    logger.info(f"Total wall time:  {batch_duration}s")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch evaluation on datasets. "
        "All parameters default to values in configs/; CLI flags only override."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=list(DATASET_FILES.keys()),
        help="Dataset name. Omit to run ALL datasets found under --data_root.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root directory of test datasets. "
        "Defaults to offline.test_root in --evo_config.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="react",
        choices=["react", "cot"],
        help="react = full ReAct agent with tools (default); "
        "cot = pure chain-of-thought, no tools, single-turn.",
    )
    parser.add_argument(
        "--llm_config",
        type=str,
        default="configs/llms/demo.json",
        help="LLM config file (default: configs/llms/demo.json).",
    )
    parser.add_argument(
        "--global_config",
        type=str,
        default="configs/global_config.json",
        help="Global agent config file.",
    )
    parser.add_argument(
        "--evo_default_config",
        type=str,
        default="configs/evolution/default.json",
    )
    parser.add_argument(
        "--evo_override_config",
        type=str,
        default="configs/evolution/offline.json",
        help="Evolution config that provides offline.test_root.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap the number of samples per dataset (default: unlimited).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: output/batch_eval).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Resume an interrupted run.  Already-completed samples (identified by "
            "their 'query' field) are skipped and new results are appended to the "
            "existing output file.  When --resume is active the output file uses a "
            "stable name '{dataset}_{mode}.jsonl' (no timestamp) so the same file "
            "is reused across runs."
        ),
    )
    parser.add_argument(
        "--retry_failed",
        action="store_true",
        default=False,
        help=(
            "Re-run samples that previously failed (\"success\": false) in the "
            "existing output files under --output_dir.  New results replace the "
            "failed records in-place; the file is rewritten atomically so no data "
            "is lost on interruption.  The output file is located automatically: "
            "a stable '{dataset}_{mode}.jsonl' is preferred; otherwise the most "
            "recently modified timestamped file is used.  Cannot be combined with "
            "--resume."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Validate mutually exclusive flags                                    #
    # ------------------------------------------------------------------ #
    if args.resume and args.retry_failed:
        parser.error("--resume and --retry_failed are mutually exclusive.")

    # ------------------------------------------------------------------ #
    # 1. Load configs                                                       #
    # ------------------------------------------------------------------ #
    from medmem_agent.config import GlobalConfig, LLMConfig

    evo_config = load_evo_config(args.evo_default_config, args.evo_override_config)
    llm_config = LLMConfig.from_file(args.llm_config)
    global_config = GlobalConfig.from_file(args.global_config)
    llm = build_llm(llm_config)

    # ------------------------------------------------------------------ #
    # 2. Resolve data_root: CLI > evo_config.offline.test_root             #
    # ------------------------------------------------------------------ #
    data_root = (
        args.data_root
        or evo_config.offline.test_root
        or getattr(evo_config.offline, "test_split_path", None)
    )
    if not data_root:
        parser.error(
            "--data_root is required (or set offline.test_root in "
            f"{args.evo_override_config})"
        )

    # ------------------------------------------------------------------ #
    # 3. Resolve which datasets to run                                      #
    # ------------------------------------------------------------------ #
    if args.dataset:
        # Single dataset specified on CLI
        rel_path = DATASET_FILES.get(args.dataset)
        if not rel_path:
            parser.error(f"Unknown dataset: {args.dataset}")
        full_path = Path(data_root) / rel_path
        if not full_path.exists():
            logger.error(f"Dataset file not found: {full_path}")
            return
        datasets_to_run = {args.dataset: str(full_path)}
    else:
        # No dataset specified → discover all available under data_root
        try:
            datasets_to_run = discover_dataset_paths(data_root, selected_benches=None)
        except FileNotFoundError as exc:
            logger.error(str(exc))
            return
        logger.info(
            f"No --dataset specified; running all {len(datasets_to_run)} "
            f"available datasets: {sorted(datasets_to_run)}"
        )

    # ------------------------------------------------------------------ #
    # 4. Output directory                                                   #
    # ------------------------------------------------------------------ #
    output_dir = Path(args.output_dir) if args.output_dir else Path("output/batch_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 5. Run each dataset                                                   #
    # ------------------------------------------------------------------ #
    for dataset_name, data_path in sorted(datasets_to_run.items()):
        if args.retry_failed:
            _retry_failed_dataset(
                dataset_name=dataset_name,
                data_path=data_path,
                mode=args.mode,
                llm=llm,
                llm_config=llm_config,
                global_config=global_config,
                output_dir=output_dir,
                max_samples=args.max_samples,
            )
        else:
            _run_dataset(
                dataset_name=dataset_name,
                data_path=data_path,
                mode=args.mode,
                llm=llm,
                llm_config=llm_config,
                global_config=global_config,
                max_samples=args.max_samples,
                output_dir=output_dir,
                resume=args.resume,
            )


if __name__ == "__main__":
    main()
