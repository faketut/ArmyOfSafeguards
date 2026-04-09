# Meta aggregator training pipeline

## Overview

1. **Label data** — Follow [docs/meta_training_labeling.md](../../docs/meta_training_labeling.md).
2. **Seed JSONL** — `seed_meta_train.jsonl`: rows with `text`, `label`, optional `id` / `domain` / `source`.
3. **Expert features** — Run `generate_expert_features.py` to append `individual_results` (calls all experts).
4. **Optional: expert-Q labels** — Use [../teacher_dataset/README.md](../teacher_dataset/README.md) to build CSV / meta JSONL from the four experts plus a **max-Q threshold** label. If you already have **human `label`** in JSONL, use `generate_expert_features.py` only — see `seed_meta_train_with_experts.jsonl` (generated from `seed_meta_train.jsonl`).
5. **Train meta** — `python -m meta_classifier.train_meta --data ...` (see [meta_classifier/train_meta.py](../../meta_classifier/train_meta.py) for OOF and calibration flags), or **tabular** models: [meta_classifier/train_meta_tabular.py](../../meta_classifier/train_meta_tabular.py) (`--algo xgb|mlp`, writes a directory with `manifest.json`).
6. **Point runtime at the artifact** — Set `AOS_META_MODEL_PATH` to a **JSON file** (legacy logistic, e.g. `meta_lr.json`) or a **directory** containing `manifest.json` (XGBoost / MLP from `train_meta_tabular.py`). [meta_classifier/predict.py](../../meta_classifier/predict.py) loads both.

### Native HF labels (no teacher)

Build meta-ready JSONL directly from Hugging Face rows + native label maps (see [build_meta_from_hf_labels.py](build_meta_from_hf_labels.py)):

```bash
# Single dataset
python training/meta/build_meta_from_hf_labels.py --dataset toxigen/toxigen-data --split train --limit 500 --out training/meta/tox_native.jsonl

# Multi-dataset pool (unified unsafe = dataset-native unsafe for each row)
python training/meta/build_meta_from_hf_labels.py --datasets-manifest training/meta/meta_native_pool.example.json --out training/meta/pool_native.jsonl
```

JBB-Behaviors needs split names `harmful` / `benign` in the manifest so labels resolve correctly.

## Quick commands

```bash
# From repo root (with venv + HF models available)
python training/meta/generate_expert_features.py \
  --input training/meta/seed_meta_train.jsonl \
  --output training/meta/seed_meta_train_with_experts.jsonl

python meta_classifier/train_meta.py \
  --data training/meta/seed_meta_train_with_experts.jsonl \
  --out meta_classifier/artifacts/meta_lr.json \
  --n-folds 5 \
  --calibrate temperature
```

## Domain-aware meta (recommended for higher AUC)

When training data mixes very different domains (e.g. `toxicity` + `sexual` + `jailbreak`), a single meta model can underperform under `GroupKFold(source)`.
You can split and train one meta model per `domain`, then route at runtime.

```bash
python training/meta/split_jsonl_by_field.py \
  --in training/meta/teacher_all_for_meta.jsonl \
  --field domain \
  --out-dir training/meta/splits_domain \
  --min-rows 50

# Example: train a toxicity-only meta model (repeat per domain file)
python -m meta_classifier.train_meta \
  --data training/meta/splits_domain/domain__toxicity.jsonl \
  --n-folds 5 \
  --group-field source \
  --class-weight balanced \
  --calibrate temperature \
  --out meta_classifier/artifacts/meta_lr_toxicity.json
```

At runtime, you can route by domain using env vars:

- `AOS_META_MODEL_PATH_TOXICITY`, `AOS_META_MODEL_PATH_SEXUAL`, `AOS_META_MODEL_PATH_JAILBREAK`, `AOS_META_MODEL_PATH_MIXED`
- or `AOS_META_MODEL_MAP_JSON` (a JSON dict string mapping domain->path)

## Offline / CI without loading HF experts

Use `synthetic_meta_train.jsonl` (checked in) to smoke-test `train_meta.py` without downloading models. For production, always generate `individual_results` from real experts.
