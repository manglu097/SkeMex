#!/usr/bin/env python3
"""Build the paper splits from locally prepared, categorized JSONL files.

No downloads or model calls. The default replays recorded row indices and
requires the exact processed-input checksum. --mode stratified recomputes the
category split. Both modes verify output counts and checksums before
writing anything; --dry-run performs only those checks.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

MANIFEST = Path(__file__).resolve().parents[1] / 'splits/manifest.json'


def read_records(path):
    records = []
    with Path(path).open(encoding='utf-8') as handle:
        lines = list(handle)
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{path.name}:{n}: malformed JSON') from exc
        if not isinstance(record, dict):
            raise ValueError(f'{path.name}:{n}: expected a JSON object')
        records.append(record)
    return records


def primary_category(record):
    value = record.get('task_category')
    if isinstance(value, list) and value:
        value = value[0]
    if not isinstance(value, str) or not value:
        raise ValueError('Stratified splitting requires a nonempty task_category for every ID record')
    return value


def stratified_indices(records, seed=42, train_ratio=.5):
    """Split sorted categories with a fresh RNG per group; singleton → train."""
    if not 0 < train_ratio < 1:
        raise ValueError('train_ratio must be between zero and one')
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[primary_category(record)].append(i)
    train, test = [], []
    for category in sorted(groups):
        group = list(groups[category])
        random.Random(seed).shuffle(group)
        n = 1 if len(group) == 1 else min(max(1, round(len(group) * train_ratio)), len(group) - 1)
        train.extend(group[:n])
        test.extend(group[n:])
    return {'train': train, 'test': test}


def validate_partition(indices, count):
    if set(indices) != {'train', 'test'}:
        raise ValueError('A partition must contain train and test lists')
    combined = indices['train'] + indices['test']
    if any(type(i) is not int for i in combined) or sorted(combined) != list(range(count)):
        raise ValueError('Split indices must cover each input record exactly once without overlap')


def serialize(records):
    # Keep serialization and record order deterministic.
    return ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records).encode('utf-8')


def prepare_plan(input_root, output_root, manifest, mode='recorded', seed=42):
    plan = []
    for entry in manifest['datasets']:
        relative = Path(entry['relative_file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Manifest paths must be relative and stay within the dataset root')
        source = Path(input_root) / relative
        raw = source.read_bytes()
        if mode == 'recorded' and hashlib.sha256(raw).hexdigest() != entry['processed_input_sha256']:
            raise ValueError(f'{entry["dataset"]}: input checksum differs; use the matching processed snapshot')
        records = read_records(source)
        if len(records) != entry['total']:
            raise ValueError(f'{entry["dataset"]}: expected {entry["total"]} processed records, found {len(records)}')
        if mode == 'recorded':
            indices = entry['indices']
        elif mode == 'stratified':
            indices = {'train': [], 'test': list(range(len(records)))} if entry['type'] == 'OOD' else stratified_indices(records, seed)
        else:
            raise ValueError('mode must be recorded or stratified')
        validate_partition(indices, len(records))
        if entry['type'] == 'OOD' and indices['train']:
            raise ValueError('OOD records must never enter training')
        for split in ('train', 'test'):
            chosen = [records[i] for i in indices[split]]
            if len(chosen) != entry[split]:
                raise ValueError(f'{entry["dataset"]}/{split}: count differs from paper')
            destination = Path(output_root) / split / relative
            if destination.exists():
                raise FileExistsError(f'Refusing to overwrite existing split: {destination}')
            if not chosen:
                continue
            payload = serialize(chosen)
            if hashlib.sha256(payload).hexdigest() != entry['local_snapshot_sha256'][split]:
                raise ValueError(f'{entry["dataset"]}/{split}: output differs from inspected paper split')
            plan.append((destination, payload, len(chosen)))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True, help='Prepared and categorized files, using paths from the manifest')
    parser.add_argument('--output-root', type=Path, required=True, help='Local destination containing train/ and test/')
    parser.add_argument('--mode', choices=['recorded', 'stratified'], default='recorded')
    parser.add_argument('--seed', type=int, default=42, help='Used only by stratified mode')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    plan = prepare_plan(args.input_root, args.output_root, manifest, args.mode, args.seed)
    for path, payload, count in plan:
        if not args.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as handle:
                handle.write(payload)
        print(f'{path.relative_to(args.output_root)}: {count} records; checksum matched')
    print(f'{len(plan)} files verified; {sum(n for _, _, n in plan)} records; '+('no files written' if args.dry_run else 'splits written'))


if __name__ == '__main__':
    main()
