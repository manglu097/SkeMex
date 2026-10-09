import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional, Set

from medmem_agent.agent import AgentLoop
from medmem_agent.config import GlobalConfig, LLMConfig
from medmem_agent.llm import build_llm
from medmem_agent.memory import ConversationMemory
from medmem_agent.tools.registry import ToolRegistry
from medmem_agent.evolution.benchmark import build_benchmark_scorers
from medmem_agent.evolution.categories import DynamicTaskCategoryRegistry
from medmem_agent.evolution.config import load_evo_config
from medmem_agent.evolution.context_guard import ContextGuard
from medmem_agent.evolution.dataset_mix import build_mixed_dataset_bundle, reshuffle_mixed_samples, should_run_evaluation
from medmem_agent.evolution.embeddings import EmbeddingManager
from medmem_agent.evolution.evaluation import RewardEvaluator
from medmem_agent.evolution.offline import OfflineEvolutionRunner
from medmem_agent.evolution.online import OnlineEvolutionRunner
from medmem_agent.evolution.retrieve import SkillRetriever
from medmem_agent.evolution.storage import SkillStore

from medmem_agent.evolution.buffer import GatedTrajectoryBuffer
from medmem_agent.evolution.encode import TrajectoryEncoder
from medmem_agent.evolution.utility import UtilityUpdater
from medmem_agent.evolution.governance import SkillGovernance

from run_batch import run_single_sample

_VALID_RUN_MODES = {"train_then_eval", "train_only", "eval_only"}


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
    registry = ToolRegistry.from_config(f"configs/datasets/{dataset_name}/tools.json", global_tools_config=global_config.tools)
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
    agent.set_context_guard(ContextGuard(evo_config=evo_config, provider=evo_provider, api_base=evo_api_base, api_key=evo_api_key))
    return agent


def _summarize_results(results_by_dataset: Dict[str, List[dict]], output_path: Path) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    overall: List[float] = []
    for dataset_name, records in results_by_dataset.items():
        rewards = [float(item.get("eval_result", 0.0)) for item in records]
        overall.extend(rewards)
        summary[dataset_name] = {"count": len(records), "mean_reward": sum(rewards) / max(len(rewards), 1)}
    summary["__overall__"] = {"count": len(overall), "mean_reward": sum(overall) / max(len(overall), 1)}
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _read_completed_queries(dataset_file: Path) -> Set[str]:
    """Return the set of query strings already written to *dataset_file*.

    Matches on ``query`` (not ``id``) because several loaders generate ``id``
    dynamically via hash(), which is randomised per-process in Python 3.3+.
    ``query`` is built deterministically from raw data and is stable across
    process restarts.
    """
    queries: Set[str] = set()
    if not dataset_file.exists():
        return queries
    try:
        with open(dataset_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    q = record.get("query")
                    if q is not None:
                        queries.add(str(q))
    except Exception as exc:
        print(f"[eval-resume] Warning: could not read {dataset_file}: {exc}. Restarting dataset from scratch.")
        return set()
    return queries


def _find_incomplete_eval_only_snapshot(output_dir: Path) -> Optional[str]:
    """Scan output_dir/snapshots/ for an eval_only snapshot that has no summary.json.

    Returns the snapshot tag (e.g. 'eval_only_20260419_225231') if found,
    or None if all snapshots are complete (or none exist).
    """
    snapshots_dir = output_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    # Sort by modification time descending so we pick the most recent incomplete run
    candidates = sorted(
        snapshots_dir.iterdir(),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for snap_dir in candidates:
        if not snap_dir.is_dir():
            continue
        if not snap_dir.name.startswith("eval_only_"):
            continue
        if not (snap_dir / "summary.json").exists():
            return snap_dir.name  # incomplete snapshot found
    return None


def _run_test_snapshot(*, output_dir: Path, tag: str, samples_by_dataset: Dict[str, List[dict]], agent_pool: Dict[str, AgentLoop], retriever: SkillRetriever, category_registry: DynamicTaskCategoryRegistry, benchmark_scorers, supports_vision: bool) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    snapshot_dir = output_dir / "evaluations" / tag
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    result_map: Dict[str, List[dict]] = {}

    for dataset_name, samples in samples_by_dataset.items():
        dataset_file = snapshot_dir / f"{dataset_name}.jsonl"
        dataset_results: List[dict] = []
        for sample in samples:
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
            result["eval_result"] = float(benchmark_scorers[dataset_name](sample, result))
            result["task_category"] = task_category
            result["injected_skill_ids"] = retrieval.injected_skill_ids
            dataset_results.append(result)
            with open(dataset_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        result_map[dataset_name] = dataset_results
    summary = _summarize_results(result_map, snapshot_dir / "summary.json")
    return summary


def require_existing_repository(evo_config) -> None:
    """Do not silently evaluate an empty repository when a checkpoint is missing."""
    root = Path(evo_config.storage.experiment_root)
    index = root / evo_config.storage.index_dirname / "utility_index.json"
    if not index.is_file():
        raise FileNotFoundError(
            "eval_only requires an existing skill repository with a utility index. "
            "Run train_only/train_then_eval first or set --experiment_root to a released checkpoint."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MedAgent-Evo in offline mode with mixed-dataset training and per-dataset testing"
    )
    parser.add_argument("--dataset", default=None, help="Backward-compatible single dataset selector")
    parser.add_argument("--benches", nargs="*", default=None, help="Training benches")
    parser.add_argument("--eval_benches", nargs="*", default=None, help="Optional test benches; defaults to training benches")
    parser.add_argument("--train_root", default=None)
    parser.add_argument("--test_root", default=None)
    parser.add_argument("--experiment_root", default=None, help="Experiment-local skills root, e.g. skills/offline")
    parser.add_argument("--llm_config", default="configs/llms/demo.json")
    parser.add_argument("--global_config", default="configs/global_config.json")
    parser.add_argument("--evo_default_config", default="configs/evolution/default.json")
    parser.add_argument("--evo_override_config", default="configs/evolution/offline.json")
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
    parser.add_argument("--max_train_samples_per_dataset", type=int, default=None)
    parser.add_argument("--max_test_samples_per_dataset", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume training from the last completed window")
    parser.add_argument(
        "--run_mode",
        default=None,
        choices=list(_VALID_RUN_MODES),
        help=(
            "Controls what to run: "
            "'train_then_eval' (default) trains then evaluates in one pass; "
            "'train_only' trains and skips evaluation; "
            "'eval_only' evaluates an existing skill library without training "
            "(launch one process per bench for maximum parallelism)."
        ),
    )
    args = parser.parse_args()

    evo_config = load_evo_config(args.evo_default_config, args.evo_override_config)
    if args.experiment_root:
        evo_config.storage.experiment_root = args.experiment_root

    # CLI --run_mode overrides config value
    run_mode = args.run_mode or evo_config.offline.run_mode or "train_then_eval"
    if run_mode not in _VALID_RUN_MODES:
        raise ValueError(f"Invalid run_mode '{run_mode}'. Must be one of {_VALID_RUN_MODES}.")

    llm_config = LLMConfig.from_file(args.llm_config)
    global_config = GlobalConfig.from_file(args.global_config)

    train_benches = _parse_benches(args.benches, args.dataset) or evo_config.offline.selected_benches
    eval_benches = _parse_benches(args.eval_benches) or evo_config.offline.evaluation.selected_benches or train_benches

    if args.train_root:
        evo_config.offline.train_root = args.train_root
    if args.test_root:
        evo_config.offline.test_root = args.test_root
    train_root = evo_config.offline.train_root or evo_config.offline.train_split_path
    test_root = evo_config.offline.test_root or evo_config.offline.test_split_path

    # Validate roots based on mode
    if run_mode in ("train_then_eval", "train_only") and not train_root:
        raise ValueError(f"run_mode='{run_mode}' requires --train_root or offline.train_root in config.")
    if run_mode in ("train_then_eval", "eval_only") and not test_root:
        raise ValueError(f"run_mode='{run_mode}' requires --test_root or offline.test_root in config.")

    train_max = args.max_train_samples_per_dataset or evo_config.offline.max_train_samples_per_dataset or None
    test_max = args.max_test_samples_per_dataset or evo_config.offline.max_test_samples_per_dataset or None

    # Build dataset bundles only as needed
    train_bundle = (
        build_mixed_dataset_bundle(
            train_root,
            selected_benches=train_benches,
            max_samples_per_dataset=train_max,
            shuffle_across_datasets=True,
            shuffle_seed=evo_config.offline.shuffle_seed,
        )
        if run_mode != "eval_only"
        else None
    )
    test_bundle = (
        build_mixed_dataset_bundle(
            test_root,
            selected_benches=eval_benches,
            max_samples_per_dataset=test_max,
            shuffle_across_datasets=False,
            shuffle_seed=evo_config.offline.shuffle_seed,
        )
        if run_mode != "train_only"
        else None
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build agent pools only as needed
    train_datasets = sorted(train_bundle.samples_by_dataset.keys()) if train_bundle else []
    test_datasets = sorted(test_bundle.samples_by_dataset.keys()) if test_bundle else []
    all_datasets = sorted(set(train_datasets) | set(test_datasets))

    train_agent_pool = (
        {ds: _build_agent(ds, llm_config, global_config, evo_config) for ds in (train_datasets or all_datasets)}
        if run_mode != "eval_only"
        else {}
    )
    test_agent_pool = (
        {ds: _build_agent(ds, llm_config, global_config, evo_config) for ds in test_datasets}
        if run_mode != "train_only"
        else {}
    )

    evo_provider = evo_config.api.provider or llm_config.provider
    evo_api_base = evo_config.api.api_base or llm_config.api_base
    evo_api_key = evo_config.api.api_key or llm_config.api_key

    if run_mode == "eval_only":
        require_existing_repository(evo_config)

    train_store = SkillStore(repo_root=".", evo_config=evo_config, read_only=False)
    frozen_store = SkillStore(
        repo_root=".", evo_config=evo_config,
        read_only=bool(evo_config.offline.freeze_skill_library_during_test),
    )
    embeddings = EmbeddingManager(
        cache_path=str(train_store.embeddings_path),
        model=evo_config.retrieval.embedding_model,
        api_base=evo_api_base,
        api_key=evo_api_key,
        global_query_cache_path=str(train_store.global_query_cache_path),
    )

    benchmark_scorers = (
        {}
        if args.disable_benchmark_eval or run_mode == "train_only"
        else build_benchmark_scorers(test_datasets, judge_model=args.judge_model)
    )

    # Full SkeMex components; test retrieval reads the frozen skill store.
    model_kwargs = dict(provider=evo_provider, api_base=evo_api_base, api_key=evo_api_key)
    buffer = GatedTrajectoryBuffer(config=evo_config.buffer)
    encoder = TrajectoryEncoder(store=train_store, evo_config=evo_config, **model_kwargs)
    train_retriever = SkillRetriever(store=train_store, embeddings=embeddings, evo_config=evo_config, **model_kwargs)
    test_retriever = SkillRetriever(store=frozen_store, embeddings=embeddings, evo_config=evo_config, **model_kwargs)
    utility_updater = UtilityUpdater(store=train_store, evo_config=evo_config)
    governance = SkillGovernance(store=train_store, embeddings=embeddings, evo_config=evo_config, **model_kwargs)

    # Always build train_runner (needed by OfflineEvolutionRunner even in eval_only
    # for category_registry and evo_config access).
    first_train_agent = (
        next(iter(train_agent_pool.values()))
        if train_agent_pool
        else _build_agent(
            (test_datasets or ["healthbench"])[0], llm_config, global_config, evo_config
        )
    )
    train_runner = OnlineEvolutionRunner(
        agent=first_train_agent,
        evo_config=evo_config,
        store=train_store,
        output_dir=str(output_dir),
        category_registry=DynamicTaskCategoryRegistry(
            store=train_store, evo_config=evo_config,
            provider=evo_provider, api_base=evo_api_base, api_key=evo_api_key,
        ),
        retriever=train_retriever,
        evaluator=RewardEvaluator(),
        encoder=encoder,
        utility_updater=utility_updater,
        governance=governance,
        buffer=buffer,
        sample_runner=run_single_sample,
        supports_vision=llm_config.supports_vision,
        agent_selector=(lambda ds: train_agent_pool[ds]) if train_agent_pool else None,
    )

    runner = OfflineEvolutionRunner(
        train_runner=train_runner,
        test_agent=next(iter(test_agent_pool.values())) if test_agent_pool else first_train_agent,
        test_retriever=test_retriever,
        frozen_store=frozen_store,
        sample_runner=run_single_sample,
        evo_config=evo_config,
        supports_vision=llm_config.supports_vision,
        test_agent_selector=(lambda ds: test_agent_pool[ds]) if test_agent_pool else None,
    )

    eval_index: List[dict] = []
    epochs = max(int(evo_config.offline.epochs), 1)

    # ------------------------------------------------------------------ #
    # eval_only: load existing skill library and evaluate without training #
    # ------------------------------------------------------------------ #
    if run_mode == "eval_only":
        current_window = OfflineEvolutionRunner.infer_current_window(frozen_store)

        # ------------------------------------------------------------------
        # Resume: find an incomplete eval_only snapshot (no summary.json)
        # and reuse its tag/timestamp so we write into the same directory.
        # ------------------------------------------------------------------
        eval_only_tag: Optional[str] = None
        if args.resume:
            eval_only_tag = _find_incomplete_eval_only_snapshot(output_dir)
            if eval_only_tag:
                print(f"[eval-resume] Found incomplete eval_only snapshot '{eval_only_tag}'. Resuming it.")
            else:
                print("[eval-resume] No incomplete eval_only snapshot found. Starting fresh.")
        if not eval_only_tag:
            eval_only_tag = f"eval_only_{timestamp}"

        snap_dir = output_dir / "snapshots" / eval_only_tag
        snap_dir.mkdir(parents=True, exist_ok=True)

        # Build per-dataset output paths (reuse existing files if resuming)
        test_output_paths = {
            ds: str(snap_dir / f"{ds}.jsonl")
            for ds in test_bundle.samples_by_dataset.keys()
        }

        # test_only() handles per-sample resume internally (id + query matching)
        test_results = runner.test_only(
            test_samples_by_dataset=test_bundle.samples_by_dataset,
            test_output_paths=test_output_paths,
            benchmark_evaluators=benchmark_scorers if benchmark_scorers else None,
            current_window=current_window,
            resume=args.resume,
        )

        # ------------------------------------------------------------------
        # Rebuild summary from all JSONL files (including pre-existing ones)
        # so that the final summary is always complete even after resume.
        # ------------------------------------------------------------------
        _summarize_results(test_results, snap_dir / "summary.json")
        summary_path = output_dir / f"eval_only_summary_{eval_only_tag.replace('eval_only_', '')}.json"
        _summarize_results(test_results, summary_path)
        return

    # ------------------------------------------------------------------ #
    # train_only / train_then_eval: epoch loop                            #
    # ------------------------------------------------------------------ #
    resume_window = 0
    if args.resume and run_mode != "eval_only":
        resume_window = OfflineEvolutionRunner.infer_current_window(train_store)
        if resume_window > 0:
            print(f"[resume] Found existing utility index. Resuming from window {resume_window}.")

    for epoch_index in range(1, epochs + 1):
        epoch_shuffle_seed = evo_config.offline.shuffle_seed + epoch_index - 1
        epoch_train_samples = reshuffle_mixed_samples(
            train_bundle.mixed_samples,
            shuffle_seed=epoch_shuffle_seed,
        )
        train_output = str(output_dir / f"offline_train_mixed_epoch{epoch_index}_{timestamp}.jsonl")

        if run_mode == "train_only":
            runner.evolve_only(
                train_samples=epoch_train_samples,
                train_output_path=train_output,
                benchmark_evaluator=benchmark_scorers if benchmark_scorers else None,
                epoch_index=epoch_index,
                resume_window=resume_window if epoch_index == 1 else 0,
            )
            continue  # skip evaluation entirely

        # train_then_eval
        test_output_paths = {
            ds: str(output_dir / "snapshots" / f"epoch_{epoch_index}_{timestamp}" / f"{ds}.jsonl")
            for ds in test_bundle.samples_by_dataset.keys()
        }
        runner.evolve_then_test(
            train_samples=epoch_train_samples,
            test_samples_by_dataset={} if args.disable_benchmark_eval else test_bundle.samples_by_dataset,
            train_output_path=train_output,
            test_output_paths=test_output_paths,
            benchmark_evaluators=benchmark_scorers if benchmark_scorers else None,
            epoch_index=epoch_index,
            resume_window=resume_window if epoch_index == 1 else 0,
        )

        if not args.disable_benchmark_eval:
            schedule = evo_config.offline.evaluation
            decision = should_run_evaluation(
                granularity=schedule.granularity,
                epoch_index=epoch_index,
                window_index=train_runner.total_windows_seen,
                evaluate_every_n_epochs=schedule.evaluate_every_n_epochs,
                evaluate_every_n_windows=schedule.evaluate_every_n_windows,
                enabled=schedule.enabled,
            )
            if decision.should_evaluate:
                summary = _run_test_snapshot(
                    output_dir=output_dir,
                    tag=f"offline_epoch_{epoch_index}_{timestamp}",
                    samples_by_dataset=test_bundle.samples_by_dataset,
                    agent_pool=test_agent_pool,
                    retriever=runner.test_retriever,
                    category_registry=train_runner.category_registry,
                    benchmark_scorers=benchmark_scorers,
                    supports_vision=llm_config.supports_vision,
                )
                eval_index.append({"tag": f"epoch_{epoch_index}", "summary": summary})

    if eval_index:
        (output_dir / f"offline_eval_index_{timestamp}.json").write_text(
            json.dumps(eval_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
