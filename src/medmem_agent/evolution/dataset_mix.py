from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from medmem_agent.dataset_loader import DATASET_LOADERS


DATASET_FILES = {
    'healthbench': 'healthbench/hard_sampled_35pct.jsonl',
    'MMMU': 'MMMU/all_test_hard_sampled.jsonl',
    'MMMU_Pro': 'MMMU_Pro/test_medical_hard_sampled.jsonl',
    'MediQ': 'MediQ/medqa_test_convo_sampled.jsonl',
    'AgentClinic_Text': 'AgentClinic_Text/agentclinic_medqa_extended.jsonl',
    'AgentClinic_MM': 'AgentClinic_MM/agentclinic_nejm_extended.jsonl',
    'LiveClin_Text': 'LiveClin_Text/2025_H1_stages_no_image_sampled.jsonl',
    'LiveClin_MM': 'LiveClin_MM/2025_H1_stages_with_image_sampled.jsonl',
    'MedXpertQA_Text': 'MedXpertQA_Text/test_sampled.jsonl',
    'MedXpertQA_MM': 'MedXpertQA_MM/test_sampled.jsonl',
    'MedJourney': 'MedJourney/medjourney_sampled_en.jsonl',
    'LiveMedBench': 'LiveMedBench/LiveMedBench_v202602_sampled_en.jsonl',
}


@dataclass
class MixedDatasetBundle:
    mixed_samples: List[dict]
    samples_by_dataset: Dict[str, List[dict]]
    dataset_paths: Dict[str, str]


@dataclass
class EvaluationScheduleDecision:
    should_evaluate: bool
    reason: str


def resolve_selected_benches(selected_benches: Optional[Iterable[str]] = None) -> List[str]:
    benches = [item for item in (selected_benches or []) if item]
    return benches or list(DATASET_FILES.keys())


def discover_dataset_paths(root_dir: str, selected_benches: Optional[Iterable[str]] = None) -> Dict[str, str]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    resolved: Dict[str, str] = {}
    missing = []
    for dataset_name in resolve_selected_benches(selected_benches):
        relative = DATASET_FILES.get(dataset_name)
        if not relative:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        data_path = root / relative
        if data_path.exists():
            resolved[dataset_name] = str(data_path)
        else:
            missing.append(dataset_name)
    if selected_benches and missing:
        raise FileNotFoundError(f"Missing requested datasets under {root}: {missing}")
    if not resolved:
        raise FileNotFoundError(
            f"No supported datasets were found under root={root}. Selected benches={list(selected_benches or [])}"
        )
    return resolved


def load_processed_samples_from_file(dataset_name: str, dataset_path: str, max_samples: Optional[int] = None) -> List[dict]:
    loader = DATASET_LOADERS[dataset_name](dataset_path)
    # load_samples() already calls process_sample() internally on each raw
    # JSON line and returns standardised dicts.  Calling process_sample()
    # again on the result would cause KeyError because the standardised dict
    # no longer contains raw fields (e.g. 'prompt', 'question', 'context').
    samples = loader.load_samples()
    if max_samples is not None and max_samples > 0:
        samples = samples[:max_samples]
    for item in samples:
        item["dataset"] = dataset_name
        item.setdefault("metadata", {})
        item["metadata"].setdefault("dataset", dataset_name)
    return samples


def build_mixed_dataset_bundle(
    root_dir: str,
    *,
    selected_benches: Optional[Iterable[str]] = None,
    max_samples_per_dataset: Optional[int] = None,
    shuffle_across_datasets: bool = True,
    shuffle_seed: int = 42,
) -> MixedDatasetBundle:
    dataset_paths = discover_dataset_paths(root_dir, selected_benches)
    samples_by_dataset: Dict[str, List[dict]] = {}
    mixed_samples: List[dict] = []

    for dataset_name, dataset_path in dataset_paths.items():
        samples = load_processed_samples_from_file(
            dataset_name,
            dataset_path,
            max_samples=max_samples_per_dataset,
        )
        samples_by_dataset[dataset_name] = samples
        mixed_samples.extend(samples)

    if shuffle_across_datasets:
        rng = random.Random(shuffle_seed)
        rng.shuffle(mixed_samples)

    return MixedDatasetBundle(
        mixed_samples=mixed_samples,
        samples_by_dataset=samples_by_dataset,
        dataset_paths=dataset_paths,
    )


def reshuffle_mixed_samples(samples: List[dict], *, shuffle_seed: int) -> List[dict]:
    cloned = list(samples)
    rng = random.Random(shuffle_seed)
    rng.shuffle(cloned)
    return cloned


def should_run_evaluation(
    *,
    granularity: str,
    epoch_index: int,
    window_index: int,
    evaluate_every_n_epochs: int,
    evaluate_every_n_windows: int,
    enabled: bool = True,
) -> EvaluationScheduleDecision:
    if not enabled:
        return EvaluationScheduleDecision(False, "evaluation_disabled")

    normalized = (granularity or "epoch").strip().lower()
    if normalized == "window":
        freq = max(int(evaluate_every_n_windows or 1), 1)
        if window_index > 0 and window_index % freq == 0:
            return EvaluationScheduleDecision(True, f"window_{window_index}")
        return EvaluationScheduleDecision(False, "window_not_due")

    freq = max(int(evaluate_every_n_epochs or 1), 1)
    if epoch_index > 0 and epoch_index % freq == 0:
        return EvaluationScheduleDecision(True, f"epoch_{epoch_index}")
    return EvaluationScheduleDecision(False, "epoch_not_due")
