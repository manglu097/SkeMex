# 🗂️ Data sources and preparation

This directory contains **source links, preparation code, and split metadata**. Download benchmark files yourself from their publishers when you are ready to run experiments. No patient cases, benchmark answers, images, or pretrained skill memories are bundled here.

📥 [Sources](#-upstream-sources) · 🧹 [Processing details](PREPROCESSING.md) · ✂️ [Split manifest](splits/manifest.json) · 🧪 [Experiments](../scripts/README.md)

## 📥 Upstream sources

Use the snapshot/subset described below rather than automatically selecting the newest version of a changing benchmark. The machine-readable index is [sources.json](sources.json).

| Benchmark | Official source | SkeMex input snapshot / subset |
|---|---|---|
| AgentClinic | [GitHub](https://github.com/samuelschmidgall/agentclinic) | Extended MedQA (text) and extended NEJM (multimodal) cases |
| HealthBench | [GitHub](https://github.com/openai/simple-evals) · [Hugging Face](https://huggingface.co/datasets/openai/healthbench) | HealthBench Hard; local source filename hard_2025-05-08-21-00-10.jsonl |
| LiveClin | [GitHub](https://github.com/AQ-MedAI/LiveClin) · [Hugging Face](https://huggingface.co/datasets/AQ-MedAI/LiveClin) | 2025_H1; local stage-level text/image separation |
| LiveMedBench | [Hugging Face](https://huggingface.co/datasets/JuelieYann/LiveMedBench) | v202602; sampled and translated English local records |
| MediQ | [GitHub](https://github.com/stellalisy/MediQ) | MedQA test conversations (medqa_test_convo.jsonl) |
| MedJourney | [GitHub](https://github.com/Medical-AI-Learning/MedJourney) | tp / ep / mp / dp tasks; sampled and translated English local records |
| MedXpertQA | [GitHub](https://github.com/TsinghuaC3I/MedXpertQA) · [Hugging Face](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) | Separate Text and MM test subsets |
| MMMU | [GitHub](https://github.com/MMMU-Benchmark/MMMU) · [Hugging Face](https://huggingface.co/datasets/MMMU/MMMU) | Health & Medicine; local all_test_hard_sampled.jsonl |
| MMMU-Pro | [Hugging Face](https://huggingface.co/datasets/MMMU/MMMU_Pro) | Health & Medicine; local test_medical_hard_sampled.jsonl |

Each benchmark retains its own terms. SkeMex's [MIT license](../LICENSE) covers this repository's code, not a new license grant for third-party datasets.

## 🧹 Preparation workflow

1. Obtain the relevant benchmark and images from the links above, recording the source revision.
2. Export to the raw per-benchmark JSONL schema consumed by `src/medmem_agent/dataset_loader.py`. Separate LiveClin text/image stages and MedXpertQA Text/MM. Restrict MMMU/MMMU-Pro to Health & Medicine.
3. Apply the documented filtering, sampling, normalization and, for MedJourney/LiveMedBench, translation stages described in [PREPROCESSING.md](PREPROCESSING.md).
4. Use the original `task_category` annotations for category-stratified splitting. Newly generated annotations can change the split; do not label those outputs as the archived paper split.
5. Build the splits and check their counts and hashes before starting experiments.

## ✂️ Paper split

These counts follow **camera-ready Appendix F, Table 10** (Table 9 in the public arXiv v3). The released manifest contains input/output file checksums and zero-based row positions, **not sample text or answer labels**.

| Dataset key | Total | Train | Test | Setting |
|---|---:|---:|---:|---|
| `AgentClinic_Text` | 214 | 0 | 214 | OOD |
| `AgentClinic_MM` | 120 | 0 | 120 | OOD |
| `MediQ` | 127 | 0 | 127 | OOD |
| `MMMU_Pro` | 56 | 0 | 56 | OOD |
| `MedJourney` | 264 | 0 | 264 | OOD |
| `MMMU` | 277 | 138 | 139 | ID |
| `healthbench` | 350 | 175 | 175 | ID |
| `LiveClin_Text` | 205 | 104 | 101 | ID |
| `LiveClin_MM` | 371 | 186 | 185 | ID |
| `LiveMedBench` | 623 | 313 | 310 | ID |
| `MedXpertQA_Text` | 372 | 187 | 185 | ID |
| `MedXpertQA_MM` | 299 | 151 | 148 | ID |
| **Total** | **3,278** | **1,254** | **2,024** | |

For ID datasets, split on the primary category (`task_category[0]`, or the string value). Iterate categories in sorted order; initialize a fresh `random.Random(42)` in each category; shuffle; take `round(n * 0.5)` records for training, bounded to `[1, n-1]` for `n >= 2`. A singleton belongs to training. OOD datasets are entirely test-only. Python's rounding convention, category annotations, and input record order are part of the recipe.

## ▶️ Reconstruct and verify locally

Place the **prepared and categorized inputs** under `data/processed/` using each `relative_file` in [the manifest](splits/manifest.json). These inputs are the stage immediately before splitting, not arbitrary upstream downloads.

```bash
# Verify the recorded partition before writing anything.
python data/scripts/split_dataset.py \
  --input-root data/processed --output-root data --dry-run

# Write data/train/ and data/test/ from the recorded row positions.
python data/scripts/split_dataset.py \
  --input-root data/processed --output-root data

# Alternatively, recompute the stratified recipe and verify its output.
python data/scripts/split_dataset.py \
  --input-root data/processed --output-root data --mode stratified --dry-run

python scripts/verify_paper.py \
  --train-root data/train --test-root data/test --strict-hash
```

The splitter refuses wrong input checksums in recorded mode, incorrect output counts/checksums in either mode, and existing output files. Run the two reconstruction modes into different fresh destinations when comparing them. The manifest enables deterministic replay when the matching processed inputs are available.

```text
data/
├── README.md                 # Sources and preparation entry point
├── PREPROCESSING.md          # Dataset-specific processing stages
├── sources.json              # Official project/data links
├── sampling.json             # Recovered sampling parameters
├── scripts/
│   ├── preprocess.py         # Five local sampling recipes
│   └── split_dataset.py      # Recorded or category-stratified split
├── splits/manifest.json      # Counts, hashes, and row positions
├── processed/                # Your local prepared inputs; ignored
├── train/                    # Your generated ID training records; ignored
├── test/                     # Your generated ID/OOD test records; ignored
└── ...                       # Your original image folders; ignored
```

Set `SKEMEX_DATA_ROOT` to the root of the original image folders for MMMU/MMMU-Pro and `AGENTCLINIC_IMAGE_ROOT` to AgentClinic's image folder. Other image paths follow their dataset-specific loader rules.

Raw downloads, local images, intermediate files and generated splits remain ignored. For a lightweight illustration of evolved skills, see the [15 skill demos](../examples/skill_demo/README.md).
