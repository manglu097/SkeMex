from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from medmem_agent.dataset_loader import DATASET_LOADERS


DEFAULT_DATASET_FILES = {
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


def resolve_selected_benches(selected_benches: Optional[Iterable[str]] = None) -> List[str]:
    selected = [item for item in (selected_benches or []) if item]
    return selected or list(DEFAULT_DATASET_FILES.keys())


def discover_dataset_files(
    root_dir: str,
    *,
    selected_benches: Optional[Iterable[str]] = None,
    dataset_files: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    file_map = dataset_files or DEFAULT_DATASET_FILES
    root = Path(root_dir)
    discovered: Dict[str, str] = {}
    for dataset_name in resolve_selected_benches(selected_benches):
        rel_path = file_map.get(dataset_name)
        if not rel_path:
            continue
        path = root / rel_path
        if path.exists():
            discovered[dataset_name] = str(path)
    return discovered


def load_processed_dataset(
    dataset_name: str,
    dataset_path: str,
    *,
    max_samples: Optional[int] = None,
) -> List[dict]:
    loader = DATASET_LOADERS[dataset_name](dataset_path)
    # load_samples() already calls process_sample() internally on each raw
    # JSON line and returns standardised dicts.  Calling process_sample()
    # again on the result would cause KeyError because the standardised dict
    # no longer contains raw fields (e.g. 'prompt', 'question', 'context').
    samples = loader.load_samples()
    if max_samples:
        samples = samples[:max_samples]
    for sample in samples:
        sample["dataset"] = dataset_name
    return samples


def load_grouped_samples_from_root(
    root_dir: str,
    *,
    selected_benches: Optional[Iterable[str]] = None,
    dataset_files: Optional[Dict[str, str]] = None,
    max_samples_per_dataset: Optional[int] = None,
) -> Dict[str, List[dict]]:
    discovered = discover_dataset_files(
        root_dir,
        selected_benches=selected_benches,
        dataset_files=dataset_files,
    )
    grouped: Dict[str, List[dict]] = {}
    for dataset_name, dataset_path in discovered.items():
        grouped[dataset_name] = load_processed_dataset(
            dataset_name,
            dataset_path,
            max_samples=max_samples_per_dataset,
        )
    return grouped


def flatten_grouped_samples(
    grouped_samples: Dict[str, List[dict]],
    *,
    shuffle: bool = True,
    shuffle_seed: int = 42,
) -> List[dict]:
    flattened = [sample for dataset_samples in grouped_samples.values() for sample in dataset_samples]
    if shuffle:
        rng = random.Random(shuffle_seed)
        rng.shuffle(flattened)
    return flattened


def load_mixed_samples_from_root(
    root_dir: str,
    *,
    selected_benches: Optional[Iterable[str]] = None,
    dataset_files: Optional[Dict[str, str]] = None,
    max_samples_per_dataset: Optional[int] = None,
    shuffle: bool = True,
    shuffle_seed: int = 42,
) -> Tuple[List[dict], Dict[str, List[dict]]]:
    grouped = load_grouped_samples_from_root(
        root_dir,
        selected_benches=selected_benches,
        dataset_files=dataset_files,
        max_samples_per_dataset=max_samples_per_dataset,
    )
    mixed = flatten_grouped_samples(grouped, shuffle=shuffle, shuffle_seed=shuffle_seed)
    return mixed, grouped
