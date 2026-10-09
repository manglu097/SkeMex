# 🧪 Running experiments

Install the package from the repository root with `python -m pip install -e .`, configure the services described in [`configs/CONFIGURATION.md`](../configs/CONFIGURATION.md), and prepare the benchmark splits described in [`data/README.md`](../data/README.md).

## Offline ID and OOD evaluation

```bash
python scripts/run_offline_evolution.py \
  --run_mode train_then_eval \
  --train_root data/train --test_root data/test \
  --experiment_root skills/offline --output_dir output/offline

python scripts/run_offline_evolution.py \
  --run_mode eval_only --test_root data/test \
  --experiment_root skills/offline --output_dir output/ood \
  --eval_benches AgentClinic_Text AgentClinic_MM MediQ MMMU_Pro MedJourney
```

The offline configuration selects the seven ID training datasets. `eval_only` requires an existing skill repository produced by training. Evaluation freezes skill and utility updates; embedding and category caches may still be written. Use `train_only` to separate accumulation from later evaluation, and `--resume` to recover an interrupted run.

## Online evaluation

The online configuration runs three rounds, corresponding to epoch@1, epoch@2, and epoch@3 in Table 3 of the paper. Keep a separate experiment directory for each benchmark:

```bash
python scripts/run_online_evolution.py \
  --benches AgentClinic_Text --data_root data/test \
  --experiment_root skills/online_agentclinic_text --output_dir output/online_agentclinic_text
```

Run the command independently for `AgentClinic_Text`, `AgentClinic_MM`, `LiveClin_Text`, `LiveClin_MM`, and `LiveMedBench`. The online configuration disables additional post-epoch evaluation. Stream scores and any later diagnostic evaluation should be recorded separately.

## Metrics

- Multiple-choice evaluation uses the repository's label-occurrence matcher, which normalizes both strings and checks whether the reference label occurs in the prediction.
- AgentClinic Text and MedJourney use semantic equivalence judgments.
- HealthBench and LiveMedBench use rubric-weighted scores normalized by positive rubric weights.
- Evolution summaries include zero rewards for failed records; inspect failure counts before comparing them with standalone evaluator metrics.
- The offline `__overall__` value is sample-weighted. For paper-style benchmark tables, average benchmark metrics macro-wise.

The scripts do not download benchmark data automatically. Keep credentials, API responses, generated skill stores, and run outputs outside version control.
