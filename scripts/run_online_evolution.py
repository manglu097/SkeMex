import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import math
import shutil
from datetime import datetime
from typing import Dict, List, Optional

from medmem_agent.agent import AgentLoop
from medmem_agent.config import GlobalConfig, LLMConfig
from medmem_agent.llm import build_llm
from medmem_agent.memory import ConversationMemory
from medmem_agent.tools.registry import ToolRegistry
from medmem_agent.evolution.benchmark import build_benchmark_scorers
from medmem_agent.evolution.buffer import GatedTrajectoryBuffer
from medmem_agent.evolution.categories import DynamicTaskCategoryRegistry
from medmem_agent.evolution.config import load_evo_config
from medmem_agent.evolution.context_guard import ContextGuard
from medmem_agent.evolution.dataset_mix import build_mixed_dataset_bundle, reshuffle_mixed_samples, should_run_evaluation
from medmem_agent.evolution.embeddings import EmbeddingManager
from medmem_agent.evolution.encode import TrajectoryEncoder
from medmem_agent.evolution.evaluation import RewardEvaluator
from medmem_agent.evolution.governance import SkillGovernance
from medmem_agent.evolution.online import OnlineEvolutionRunner
from medmem_agent.evolution.retrieve import SkillRetriever
from medmem_agent.evolution.storage import SkillStore
from medmem_agent.evolution.utility import UtilityUpdater
from run_batch import run_single_sample


def _parse_benches(raw: Optional[List[str]], fallback_single: Optional[str] = None) -> List[str]:
    if raw:
        values: List[str] = []
        for item in raw:
            values.extend([part.strip() for part in item.split(",") if part.strip()])
        if values:
            return values
    return [fallback_single] if fallback_single else []


def _build_agent(dataset_name: str, llm_config: LLMConfig, global_config: GlobalConfig, evo_config) -> AgentLoop:
    llm = build_llm(llm_config)
    registry = ToolRegistry.from_config(
        f"configs/datasets/{dataset_name}/tools.json",
        global_tools_config=global_config.tools,
    )
    agent = AgentLoop(
        llm=llm,
        tool_registry=registry,
        memory=ConversationMemory(),
        max_steps=global_config.max_steps,
        output_dir=global_config.output_dir,
        history_filename=global_config.history_filename,
        history_save_mode=global_config.history_save_mode,
    )
    evo_provider = evo_config.api.provider or llm_config.provider
    evo_api_base = evo_config.api.api_base or llm_config.api_base
    evo_api_key = evo_config.api.api_key or llm_config.api_key
    agent.set_context_guard(
        ContextGuard(
            evo_config=evo_config,
            provider=evo_provider,
            api_base=evo_api_base,
            api_key=evo_api_key,
        )
    )
    return agent


def _snapshot_skills(store: "SkillStore", output_dir: Path, epoch_index: int) -> Path:
    """Copy the current skill library into a per-epoch checkpoint directory.

    The checkpoint captures everything needed to inspect or replay the skill
    library as it stood at the end of ``epoch_index``:

    * ``skills/``        — all skill Markdown files (all branches)
    * ``index/``         — ``utility_index.json``, ``categories.json``,
                           ``id_counters.json``
    * ``embeddings/``    — ``embeddings.json``

    The ``usage_traces/`` directory is intentionally excluded because it can
    be very large and is not needed for post-hoc skill inspection.

    Returns the path of the created checkpoint directory.
    """
    ckpt_dir = output_dir / "checkpoints" / f"epoch_{epoch_index}"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    src_root = store.experiment_root
    for sub in (store.evo_config.storage.skills_dirname,
                store.evo_config.storage.index_dirname,
                store.evo_config.storage.embeddings_dirname):
        src = src_root / sub
        if src.exists():
            shutil.copytree(src, ckpt_dir / sub)

    # Write a small manifest so the checkpoint is self-describing.
    manifest = {
        "epoch": epoch_index,
        "source": str(src_root),
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (ckpt_dir / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[checkpoint] Epoch {epoch_index} skills saved → {ckpt_dir}")
    return ckpt_dir


# ---------------------------------------------------------------------------
# Evaluation progress state helpers
# ---------------------------------------------------------------------------
# When --resume is active, we need to distinguish two situations after a crash:
#
#   (A) Crashed during evolution  → utility_index.json reflects a partially
#       completed epoch; resume_window_in_epoch > 0; evaluation was never
#       reached for this epoch.
#
#   (B) Crashed during evaluation → the epoch's evolution is 100 % done
#       (utility_index.json is fully written); but the evaluation snapshot
#       is incomplete.  infer_current_window() cannot tell us this because
#       it only looks at the utility index.
#
# We solve (B) by writing a small JSON state file
# ``<output_dir>/eval_progress/<tag>.json`` at the start of every evaluation
# pass (status="running") and updating it to status="done" when the pass
# completes.  On resume we scan for any "running" state files and re-enter
# the corresponding evaluation pass before continuing with evolution.
# ---------------------------------------------------------------------------

_EVAL_PROGRESS_DIR = "eval_progress"


def _eval_state_path(output_dir: Path, tag: str) -> Path:
    return output_dir / _EVAL_PROGRESS_DIR / f"{tag}.json"


def _mark_eval_running(output_dir: Path, tag: str, epoch_index: int) -> None:
    """Write (or overwrite) the state file for *tag* as 'running'."""
    path = _eval_state_path(output_dir, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"tag": tag, "epoch_index": epoch_index, "status": "running"}, indent=2) + "\n",
        encoding="utf-8",
    )


def _mark_eval_done(output_dir: Path, tag: str, epoch_index: int) -> None:
    """Overwrite the state file for *tag* as 'done'."""
    path = _eval_state_path(output_dir, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"tag": tag, "epoch_index": epoch_index, "status": "done"}, indent=2) + "\n",
        encoding="utf-8",
    )


def _find_incomplete_eval(output_dir: Path) -> Optional[dict]:
    """Return the state dict of the first incomplete evaluation pass, or None.

    Two detection strategies are used in order:

    1. **State-file strategy** (new runs): scan ``eval_progress/`` for any
       ``status='running'`` file written by ``_mark_eval_running()``.  This
       works for crashes that happened after the new code was deployed.

    2. **Directory-scan strategy** (legacy / cold-start): scan
       ``evaluations/`` for any snapshot directory that is missing
       ``summary.json``.  A missing summary means the evaluation pass never
       completed.  The epoch index is parsed from the directory name
       (``epoch_<N>_<timestamp>``).

    Returns a dict with keys ``tag``, ``epoch_index``, ``status`` (always
    ``'running'`` for the caller's convenience), or ``None`` if every
    evaluation pass appears complete.
    """
    # Strategy 1: state-file (written by _mark_eval_running)
    progress_dir = output_dir / _EVAL_PROGRESS_DIR
    if progress_dir.exists():
        for state_file in sorted(progress_dir.glob("*.json")):
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
                if state.get("status") == "running":
                    return state
            except Exception:
                continue

    # Strategy 2: directory scan — look for evaluation snapshot dirs that
    # lack a summary.json (i.e. the pass was never fully completed).
    eval_root = output_dir / "evaluations"
    if not eval_root.exists():
        return None
    for snap_dir in sorted(eval_root.iterdir()):
        if not snap_dir.is_dir():
            continue
        if (snap_dir / "summary.json").exists():
            continue  # fully completed
        # Parse epoch index from directory name, e.g. "epoch_1_20260419_225231"
        parts = snap_dir.name.split("_")  # ["epoch", "1", "20260419", "225231"]
        epoch_index = 1
        if len(parts) >= 2 and parts[0] == "epoch":
            try:
                epoch_index = int(parts[1])
            except ValueError:
                pass
        print(
            f"[eval-resume] Found incomplete evaluation snapshot '{snap_dir.name}' "
            f"(no summary.json). Will resume it."
        )
        return {"tag": snap_dir.name, "epoch_index": epoch_index, "status": "running"}
    return None


def _load_completed_eval_results(dataset_file: Path) -> List[dict]:
    """Read already-written results from a partial evaluation JSONL file.

    Returns a list of result dicts.  On any read/parse error the file is
    treated as empty so that the evaluation restarts from scratch for that
    dataset rather than crashing the whole run.
    """
    if not dataset_file.exists():
        return []
    completed: List[dict] = []
    try:
        with open(dataset_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    completed.append(json.loads(line))
    except Exception as exc:
        print(f"[eval-resume] Warning: could not read {dataset_file}: {exc}. Restarting dataset from scratch.")
        return []
    return completed


def _completed_queries(dataset_file: Path) -> set:
    """Return the set of *query* strings already written to *dataset_file*.

    We match on ``query`` rather than ``id`` because several dataset loaders
    generate ``id`` dynamically at runtime (e.g. ``agentclinic_text_<hash>``,
    ``medjourney_<hash>``).  These hashes can differ between process restarts
    even for the same sample, making ID-based matching unreliable.

    ``query`` is built deterministically from the raw data fields so it is
    stable across restarts and can be used as a reliable deduplication key.
    """
    queries: set = set()
    for record in _load_completed_eval_results(dataset_file):
        q = record.get("query")
        if q is not None:
            queries.add(str(q))
    return queries


def _evaluate_snapshot(*, output_dir: Path, tag: str, epoch_index: int, samples_by_dataset: Dict[str, List[dict]], agent_pool: Dict[str, AgentLoop], retriever: SkillRetriever, category_registry: DynamicTaskCategoryRegistry, benchmark_scorers, supports_vision: bool, resume: bool = False) -> Dict[str, dict]:
    # Mark this evaluation pass as "running" before doing any work so that a
    # crash mid-evaluation is detectable on the next --resume invocation.
    _mark_eval_running(output_dir, tag, epoch_index)

    summary: Dict[str, dict] = {}
    eval_dir = output_dir / "evaluations" / tag
    eval_dir.mkdir(parents=True, exist_ok=True)
    overall_rewards: List[float] = []

    for dataset_name, samples in samples_by_dataset.items():
        dataset_file = eval_dir / f"{dataset_name}.jsonl"

        # ------------------------------------------------------------------
        # Resume: load already-written results and build a set of completed
        # sample IDs.  We use pure ID matching (not positional counting) so
        # that the skip logic is correct even when the prior run wrote results
        # out-of-order or when the dataset loader changes between runs.
        # ------------------------------------------------------------------
        completed_results: List[dict] = []
        done_queries: set = set()
        if resume and dataset_file.exists():
            completed_results = _load_completed_eval_results(dataset_file)
            done_queries = _completed_queries(dataset_file)
            if done_queries:
                print(
                    f"[eval-resume] {dataset_name}: found {len(done_queries)} already-evaluated "
                    f"samples in {dataset_file.name} (matched by query), skipping them."
                )

        results: List[dict] = list(completed_results)
        total_reward = sum(float(r.get("eval_result", 0.0)) for r in completed_results)
        for r in completed_results:
            overall_rewards.append(float(r.get("eval_result", 0.0)))

        for sample in samples:
            if resume and str(sample.get("query", "")) in done_queries:
                # Query-based skip: this sample was already evaluated.
                # We match on query rather than id because some loaders
                # generate id dynamically (e.g. hash-based), making id
                # unstable across process restarts.
                continue
            agent = agent_pool[dataset_name]
            task_category = category_registry.classify(sample.get("query", ""))
            retrieval = retriever.retrieve(sample.get("query", ""), task_category)
            if hasattr(agent, "set_evolution_runtime_context"):
                agent.set_evolution_runtime_context({
                    "task_category": task_category,
                    "retrieved_skills_block": retrieval.retrieved_skills_block,
                    "injected_skill_ids": retrieval.injected_skill_ids,
                })
            result = run_single_sample(agent, sample, dataset_name, supports_vision)
            reward = float(benchmark_scorers[dataset_name](sample, result))
            result["eval_result"] = reward
            result["task_category"] = task_category
            result["injected_skill_ids"] = retrieval.injected_skill_ids
            # Persist query so that resume can match by query string.
            result.setdefault("query", sample.get("query", ""))
            results.append(result)
            total_reward += reward
            overall_rewards.append(reward)
            with open(dataset_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        summary[dataset_name] = {
            "count": len(results),
            "mean_reward": total_reward / max(len(results), 1),
            "output_file": str(dataset_file),
        }
    summary["__overall__"] = {
        "count": len(overall_rewards),
        "mean_reward": sum(overall_rewards) / max(len(overall_rewards), 1),
    }
    (eval_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Mark this evaluation pass as fully completed.
    _mark_eval_done(output_dir, tag, epoch_index)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MedAgent-Evo in online mode with mixed-dataset evolution")
    parser.add_argument("--dataset", default=None, help="Backward-compatible single dataset selector")
    parser.add_argument("--benches", nargs="*", default=None, help="Evolution benches")
    parser.add_argument("--eval_benches", nargs="*", default=None, help="Optional evaluation benches; defaults to evolution benches")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--experiment_root", default=None, help="Experiment-local skills root, e.g. skills/main_exp_seed1")
    parser.add_argument("--llm_config", default="configs/llms/demo.json")
    parser.add_argument("--global_config", default="configs/global_config.json")
    parser.add_argument("--evo_default_config", default="configs/evolution/default.json")
    parser.add_argument("--evo_override_config", default="configs/evolution/online.json")
    parser.add_argument("--output_dir", default="output/evolution")
    _eval_cfg_path = Path(__file__).resolve().parent.parent / "configs" / "eval" / "default.json"
    _default_judge_model = "gpt-4.1-mini"
    if _eval_cfg_path.exists():
        try:
            _default_judge_model = json.loads(_eval_cfg_path.read_text(encoding="utf-8")).get("model", _default_judge_model)
        except Exception:
            pass
    parser.add_argument("--judge_model", default=_default_judge_model)
    parser.add_argument("--disable_benchmark_eval", action="store_true")
    parser.add_argument("--max_samples_per_dataset", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Resume training from the last completed window. "
            "The completed window index is inferred from the utility index on disk "
            "(max last_adopted_window across all skills). "
            "Supports cross-epoch resume: if the run crashed in epoch 2 or 3, "
            "earlier epochs are skipped entirely and training continues from the "
            "correct window within the crashed epoch."
        ),
    )
    args = parser.parse_args()

    evo_config = load_evo_config(args.evo_default_config, args.evo_override_config)
    if args.experiment_root:
        evo_config.storage.experiment_root = args.experiment_root
    llm_config = LLMConfig.from_file(args.llm_config)
    global_config = GlobalConfig.from_file(args.global_config)

    selected_benches = _parse_benches(args.benches, args.dataset) or evo_config.online.selected_benches
    eval_benches = _parse_benches(args.eval_benches) or evo_config.online.evaluation.selected_benches or selected_benches
    benchmark_root = args.data_root or evo_config.online.benchmark_root
    if not benchmark_root:
        raise ValueError("Online mode requires --data_root or online.benchmark_root in evolution config.")

    max_samples_per_dataset = args.max_samples_per_dataset or evo_config.online.max_samples_per_dataset or None
    train_bundle = build_mixed_dataset_bundle(
        benchmark_root,
        selected_benches=selected_benches,
        max_samples_per_dataset=max_samples_per_dataset,
        shuffle_across_datasets=True,
        shuffle_seed=evo_config.online.shuffle_seed,
    )
    eval_bundle = build_mixed_dataset_bundle(
        benchmark_root,
        selected_benches=eval_benches,
        max_samples_per_dataset=max_samples_per_dataset,
        shuffle_across_datasets=False,
        shuffle_seed=evo_config.online.shuffle_seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Timestamp: when resuming, reuse the timestamp from the most recent
    # prior run so that evaluation snapshot directories (which embed the
    # timestamp in their name) are found and resumed rather than created
    # fresh.  We infer the prior timestamp from the newest
    # online_mixed_epoch*.jsonl file in output_dir, falling back to a new
    # timestamp when none exists (i.e. first run or non-resume mode).
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume:
        existing_epoch_files = sorted(output_dir.glob("online_mixed_epoch*.jsonl"))
        if existing_epoch_files:
            # Extract the timestamp suffix from the filename, e.g.
            # "online_mixed_epoch1_20260422_120000.jsonl" → "20260422_120000"
            stem = existing_epoch_files[-1].stem  # e.g. "online_mixed_epoch1_20260422_120000"
            parts = stem.split("_", 3)  # ["online", "mixed", "epoch1", "20260422_120000"]
            if len(parts) == 4:
                timestamp = parts[3]
                print(f"[resume] Reusing prior run timestamp: {timestamp}")

    all_agent_datasets = sorted(set(train_bundle.samples_by_dataset.keys()) | set(eval_bundle.samples_by_dataset.keys()))
    agent_pool = {dataset_name: _build_agent(dataset_name, llm_config, global_config, evo_config) for dataset_name in all_agent_datasets}
    first_agent = next(iter(agent_pool.values()))

    store = SkillStore(repo_root=".", evo_config=evo_config)
    evo_provider = evo_config.api.provider or llm_config.provider
    evo_api_base = evo_config.api.api_base or llm_config.api_base
    evo_api_key = evo_config.api.api_key or llm_config.api_key
    embeddings = EmbeddingManager(
        cache_path=str(store.embeddings_path),
        model=evo_config.retrieval.embedding_model,
        api_base=evo_api_base,
        api_key=evo_api_key,
        global_query_cache_path=str(store.global_query_cache_path),
    )
    benchmark_scorers = {} if args.disable_benchmark_eval else build_benchmark_scorers(eval_bundle.samples_by_dataset.keys(), judge_model=args.judge_model)

    runner = OnlineEvolutionRunner(
        agent=first_agent,
        evo_config=evo_config,
        store=store,
        category_registry=DynamicTaskCategoryRegistry(store=store, evo_config=evo_config, provider=evo_provider, api_base=evo_api_base, api_key=evo_api_key),
        retriever=SkillRetriever(store=store, embeddings=embeddings, evo_config=evo_config, provider=evo_provider, api_base=evo_api_base, api_key=evo_api_key),
        evaluator=RewardEvaluator(),
        encoder=TrajectoryEncoder(store=store, evo_config=evo_config, provider=evo_provider, api_base=evo_api_base, api_key=evo_api_key),
        utility_updater=UtilityUpdater(store=store, evo_config=evo_config),
        governance=SkillGovernance(store=store, embeddings=embeddings, evo_config=evo_config, provider=evo_provider, api_base=evo_api_base, api_key=evo_api_key),
        buffer=GatedTrajectoryBuffer(config=evo_config.buffer),
        sample_runner=run_single_sample,
        supports_vision=llm_config.supports_vision,
        agent_selector=lambda dataset_name: agent_pool[dataset_name],
    )

    if not args.disable_benchmark_eval:
        def _on_window_end(event: dict) -> None:
            schedule = evo_config.online.evaluation
            decision = should_run_evaluation(
                granularity=schedule.granularity,
                epoch_index=event["epoch_index"],
                window_index=event["window_index"],
                evaluate_every_n_epochs=schedule.evaluate_every_n_epochs,
                evaluate_every_n_windows=schedule.evaluate_every_n_windows,
                enabled=schedule.enabled,
            )
            if decision.should_evaluate and schedule.granularity.lower() == "window":
                _evaluate_snapshot(
                    output_dir=output_dir,
                    tag=f"window_{event['window_index']}_{timestamp}",
                    epoch_index=event["epoch_index"],
                    samples_by_dataset=eval_bundle.samples_by_dataset,
                    agent_pool=agent_pool,
                    retriever=runner.retriever,
                    category_registry=runner.category_registry,
                    benchmark_scorers=benchmark_scorers,
                    supports_vision=llm_config.supports_vision,
                    resume=args.resume,
                )
        runner.window_end_callback = _on_window_end

    # ------------------------------------------------------------------
    # Resolve resume_window: infer from utility index on disk when
    # --resume is active, then decompose into (resume_epoch,
    # resume_window_in_epoch) so that cross-epoch resume works correctly.
    #
    # Two sources are consulted and the larger value wins:
    #
    # 1. Utility index  — updated at the end of every window that produces
    #    at least one skill adoption.  May lag behind if a window produced
    #    no adoptions (e.g. the very first window of a new epoch).
    #
    # 2. Performance log (online_mixed_epoch*.performance.json) — written
    #    atomically after *every* window regardless of skill adoption.  The
    #    max ``window_index`` across all performance logs is therefore the
    #    most up-to-date record of how far training has progressed.  This
    #    source catches the case where epoch N has already started (its
    #    .jsonl file exists) but the utility index still shows epoch N-1
    #    because the first window of epoch N crashed before any skill was
    #    adopted.
    # ------------------------------------------------------------------
    from medmem_agent.evolution.offline import OfflineEvolutionRunner
    global_resume_window = 0
    resume_epoch = 1
    resume_window_in_epoch = 0
    if args.resume:
        windows_per_epoch = math.ceil(
            len(train_bundle.mixed_samples) / evo_config.buffer.window_size
        )

        # Source 1 (highest priority): eval_progress 'done' state files.
        # A 'done' file means both training and evaluation for that epoch
        # fully completed.  The highest such epoch number tells us exactly
        # which epoch to resume from (next epoch, window 0).
        max_done_epoch = 0
        progress_dir = output_dir / _EVAL_PROGRESS_DIR
        if progress_dir.exists():
            for state_file in progress_dir.glob("*.json"):
                try:
                    state = json.loads(state_file.read_text(encoding="utf-8"))
                    if state.get("status") == "done":
                        ep = int(state.get("epoch_index", 0))
                        if ep > max_done_epoch:
                            max_done_epoch = ep
                except Exception:
                    pass

        if max_done_epoch > 0:
            resume_epoch = max_done_epoch + 1
            resume_window_in_epoch = 0
            global_resume_window = max_done_epoch * windows_per_epoch
            print(
                f"[resume] eval_progress shows epoch {max_done_epoch} fully done. "
                f"Resuming training from epoch {resume_epoch} "
                f"(global window {global_resume_window})."
            )
        else:
            # Source 2 (fallback): utility index — used when no eval_progress
            # 'done' file exists yet (e.g. training crashed before the first
            # evaluation pass completed).
            utility_window = OfflineEvolutionRunner.infer_current_window(store)
            global_resume_window = utility_window
            if global_resume_window > 0:
                resume_epoch = (global_resume_window // windows_per_epoch) + 1
                resume_window_in_epoch = global_resume_window % windows_per_epoch
                print(
                    f"[resume] Found existing utility index. "
                    f"Global window {global_resume_window} → "
                    f"epoch {resume_epoch}, window-in-epoch {resume_window_in_epoch} "
                    f"(skipping first {resume_window_in_epoch * evo_config.buffer.window_size} "
                    f"samples in epoch {resume_epoch})."
                )
            else:
                print("[resume] No prior progress found; starting from the beginning.")

    # ------------------------------------------------------------------
    # Resume: check for an incomplete evaluation pass that was interrupted
    # before the evolution index was advanced.  If found, re-enter that
    # evaluation pass first (the per-dataset resume logic inside
    # _evaluate_snapshot will skip already-completed samples).
    # ------------------------------------------------------------------
    if args.resume and not args.disable_benchmark_eval:
        incomplete = _find_incomplete_eval(output_dir)
        if incomplete:
            inc_tag = incomplete["tag"]
            inc_epoch = incomplete["epoch_index"]
            print(
                f"[eval-resume] Detected incomplete evaluation pass '{inc_tag}' "
                f"(epoch {inc_epoch}). Resuming it before continuing evolution."
            )
            _evaluate_snapshot(
                output_dir=output_dir,
                tag=inc_tag,
                epoch_index=inc_epoch,
                samples_by_dataset=eval_bundle.samples_by_dataset,
                agent_pool=agent_pool,
                retriever=runner.retriever,
                category_registry=runner.category_registry,
                benchmark_scorers=benchmark_scorers,
                supports_vision=llm_config.supports_vision,
                resume=True,
            )

    eval_index: List[dict] = []
    for epoch_index in range(1, max(int(evo_config.online.epochs), 1) + 1):
        if args.resume and global_resume_window > 0 and epoch_index < resume_epoch:
            # This entire epoch was already completed; skip it but keep
            # total_windows_seen consistent so memory-strength scoring is correct.
            windows_per_epoch = math.ceil(
                len(train_bundle.mixed_samples) / evo_config.buffer.window_size
            )
            runner.total_windows_seen = epoch_index * windows_per_epoch
            print(f"[resume] Skipping epoch {epoch_index} (already completed).")
            continue
        rw = resume_window_in_epoch if (args.resume and epoch_index == resume_epoch) else 0
        epoch_samples = reshuffle_mixed_samples(train_bundle.mixed_samples, shuffle_seed=evo_config.online.shuffle_seed + epoch_index - 1)
        epoch_output_path = str(output_dir / f"online_mixed_epoch{epoch_index}_{timestamp}.jsonl")
        runner.run_epoch(
            samples=epoch_samples,
            dataset_name="mixed_online",
            output_path=epoch_output_path,
            benchmark_evaluator=benchmark_scorers if benchmark_scorers else None,
            epoch_index=epoch_index,
            resume_window=rw,
        )
        # ------------------------------------------------------------------
        # Epoch checkpoint: snapshot the skill library after every epoch so
        # the state at each training stage can be inspected independently.
        # ------------------------------------------------------------------
        _snapshot_skills(store, output_dir, epoch_index)

        if not args.disable_benchmark_eval:
            schedule = evo_config.online.evaluation
            decision = should_run_evaluation(
                granularity=schedule.granularity,
                epoch_index=epoch_index,
                window_index=runner.total_windows_seen,
                evaluate_every_n_epochs=schedule.evaluate_every_n_epochs,
                evaluate_every_n_windows=schedule.evaluate_every_n_windows,
                enabled=schedule.enabled,
            )
            if decision.should_evaluate and schedule.granularity.lower() != "window":
                summary = _evaluate_snapshot(
                    output_dir=output_dir,
                    tag=f"epoch_{epoch_index}_{timestamp}",
                    epoch_index=epoch_index,
                    samples_by_dataset=eval_bundle.samples_by_dataset,
                    agent_pool=agent_pool,
                    retriever=runner.retriever,
                    category_registry=runner.category_registry,
                    benchmark_scorers=benchmark_scorers,
                    supports_vision=llm_config.supports_vision,
                    resume=args.resume,
                )
                eval_index.append({"tag": f"epoch_{epoch_index}", "summary": summary})

    if eval_index:
        (output_dir / f"online_eval_index_{timestamp}.json").write_text(json.dumps(eval_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
