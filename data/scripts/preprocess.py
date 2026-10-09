#!/usr/bin/env python3
"""Run local sampling recipes; no downloading, translation or API calls.

These are individual processing stages, not an end-to-end recreation of all
paper datasets. See data/PREPROCESSING.md for inputs, remaining stages and
verified scope. Malformed records raise an error instead of silently dropping.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from split_dataset import read_records, serialize

SETTINGS = Path(__file__).resolve().parents[1] / 'sampling.json'


def sample_records(records, recipe, settings, seed=42):
    rng = random.Random(seed)
    if recipe == 'mediq':
        return rng.sample(records, max(1, round(len(records) * settings['sample_ratio']))) if records else []
    groups = defaultdict(list)
    for row in records:
        if recipe == 'healthbench':
            tags = [tag for tag in row.get('example_tags', []) if tag.startswith('theme:')]
            category = tags[0] if tags else 'no_theme'
        elif recipe in {'liveclin_text', 'liveclin_mm'}:
            if recipe == 'liveclin_text' and ('Figure' in (row.get('scenario') or '') or 'Figure' in (row.get('question') or '')):
                continue
            category = row.get('Level1', 'N/A')
        else:
            raise ValueError('Unsupported record-sampling recipe')
        groups[category].append(row)
    items = sorted(groups.items()) if recipe == 'healthbench' else groups.items()
    output = []
    for key, group in items:
        ratio = settings['sample_ratio'] if recipe == 'healthbench' else settings['ratios'].get(key, settings['default_ratio'])
        output.extend(rng.sample(group, max(1, round(len(group) * ratio))))
    return output


def sample_medjourney(input_dir, settings, seed=42):
    rng = random.Random(seed)
    output = []
    # Historical source order tp, ep, mp, dp matters for the RNG stream.
    for name, ratio in settings['ratios'].items():
        rows = read_records(Path(input_dir) / f'{name}.jsonl')
        if not rows:
            raise ValueError(f'{name}.jsonl is empty')
        output.extend({**row, 'source': name} for row in rng.sample(rows, max(1, round(len(rows) * ratio))))
    rng.shuffle(output)
    return output


def main():
    settings = json.loads(SETTINGS.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True, choices=list(settings['recipes']))
    parser.add_argument('--input', type=Path, required=True, help='JSONL input; directory with tp/ep/mp/dp.jsonl for medjourney')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=settings['seed'])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite: {args.output}')
    recipe = settings['recipes'][args.recipe]
    rows = sample_medjourney(args.input, recipe, args.seed) if args.recipe == 'medjourney' else sample_records(read_records(args.input), args.recipe, recipe, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        handle.write(serialize(rows))
    print(f'{args.recipe}: wrote {len(rows)} locally sampled records. See the data guide for remaining processing stages.')


if __name__ == '__main__':
    main()
