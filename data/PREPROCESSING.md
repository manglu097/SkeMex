# 🧹 Processing stages and recipes

The paper describes a 3,278-record processed pool, difficulty-oriented selection, modality checks, removal of unevaluable records, format normalization, and near-duplicate removal (Appendix D). This guide documents the dataset-specific sampling and split procedures included in the repository. Some upstream filtering and translation decisions are not distributed with the public datasets.

## Per-dataset stages

| Dataset configuration | Processing before category annotation and split | Available implementation / boundary |
|---|---|---|
| AgentClinic_Text | Extended MedQA cases; use subjective patient context and reference diagnosis with the AgentClinic loader | Preserve the published case subset and its diagnosis fields |
| AgentClinic_MM | Extended NEJM cases and corresponding images | Preserve case/image associations |
| healthbench | HealthBench Hard snapshot; group by first `theme:` tag in `example_tags`; sample 35% per group, seed 42 | Executable `healthbench` recipe; use sorted theme groups and `max(1, round(n * .35))` |
| LiveClin_Text | 2025_H1 stage-level text-only records; remove records containing case-sensitive `Figure` in `scenario` or `question`; sample within `Level1` chapter | Executable `liveclin_text` recipe; per-chapter ratios in `sampling.json`, default 8% |
| LiveClin_MM | 2025_H1 stage-level image-bearing records; sample within `Level1` chapter | Executable `liveclin_mm` recipe: 50% mental/behavioural and ear/mastoid chapters, 7% neoplasms, 10% otherwise |
| LiveMedBench | `v202602` sampled subset; recursively translate Chinese-containing string values to English while retaining non-string fields and rubric structure | Translation service and sampling snapshot must be recorded by the user |
| MediQ | MedQA test conversations; sample 10%, seed 42 | Executable `mediq` recipe; `max(1, round(n * .10))`; retain the conversation/runtime-context fields |
| MedJourney | Sample task files in order `tp`, `ep`, `mp`, `dp` at 50%, 20%, 5%, 3%; attach source label; shuffle combined records; translate `prompt`, `target`, `keyentity` | Executable `medjourney` recipe; record the translation model and version |
| MedXpertQA_Text / MedXpertQA_MM | Keep modalities separate; normalize options, answer labels and image references; use sampled test files | Loader and final membership supplied; upstream filtering/sampling details are not available |
| MMMU | Health & Medicine track; sampled test records with images and answer options | Loader and final membership supplied; exact difficulty sampling/export recipe is not available |
| MMMU_Pro | Health & Medicine portion; sampled records | Loader and final membership supplied; exact upstream selection/export recipe is not available |

The LiveClin Text check is deliberately precise: it is a literal `Figure` filter, not a general image-consistency detector. The paper's full manual/automatic filtering and near-duplicate procedure cannot be inferred from that one filter.

## Run the sampling stages

These commands operate on files you have obtained and exported locally. They do not download data or call an API.

```bash
python data/scripts/preprocess.py --recipe healthbench \
  --input data/raw/healthbench/hard_2025-05-08-21-00-10.jsonl \
  --output data/intermediate/healthbench/hard_sampled_35pct.jsonl

python data/scripts/preprocess.py --recipe liveclin_text \
  --input data/raw/LiveClin_Text/2025_H1_stages_no_image.jsonl \
  --output data/intermediate/LiveClin_Text/2025_H1_stages_no_image_sampled.jsonl

python data/scripts/preprocess.py --recipe liveclin_mm \
  --input data/raw/LiveClin_MM/2025_H1_stages_with_image.jsonl \
  --output data/intermediate/LiveClin_MM/2025_H1_stages_with_image_sampled.jsonl

python data/scripts/preprocess.py --recipe mediq \
  --input data/raw/MediQ/medqa_test_convo.jsonl \
  --output data/intermediate/MediQ/medqa_test_convo_sampled.jsonl

# Input directory contains tp.jsonl, ep.jsonl, mp.jsonl and dp.jsonl.
python data/scripts/preprocess.py --recipe medjourney \
  --input data/raw/MedJourney \
  --output data/intermediate/MedJourney/medjourney_sampled.jsonl
```

The helper raises an error for malformed JSON instead of silently skipping it. It preserves physical JSONL lines, including Unicode line-separator characters inside JSON strings. Outputs refuse overwrite. Group iteration and RNG initialization are deterministic; the sampling RNG is shared across groups, whereas the final split resets the RNG for each category.

## Category labels, translation, and the final split

The final splitter expects `task_category` labels already present and uses the first category for stratification. Changing the classifier, translation, category labels, input order, or upstream revision may change the resulting samples even with seed 42.

The published [split manifest](splits/manifest.json) therefore records exact categorized-input checksums, output checksums and row membership. The indices refer to non-empty JSONL records in that processed input; they are **not upstream dataset IDs**. They contain no patient text or answers. They allow exact replay when the matching processed inputs are available, but they do not recreate unpublished translations or annotations from upstream data alone.

Use `python data/scripts/split_dataset.py --help` and the [data guide](README.md) for final split commands. Dataset loaders then convert each raw benchmark record to the common runtime structure (`id`, `query`, `images`, `runtime_context`, `ground_truth`, `metadata`). Do not feed already-normalized trajectory outputs back into those raw loaders.
