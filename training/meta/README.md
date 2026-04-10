# Meta aggregator training pipeline

## Overview

1. **Label data** — Follow [docs/meta_training_labeling.md](../../docs/meta_training_labeling.md).
2. **Seed JSONL** — `seed_meta_train.jsonl`: rows with `text`, `label`, optional `id` / `domain` / `source`.
3. **Expert features** — Run `generate_expert_features.py` to append `individual_results` (calls all experts).
4. **Optional: expert-Q heuristic labels** — Use [`generate_expert_q_meta_jsonl.py`](generate_expert_q_meta_jsonl.py) / [`label_manifest_expert_q.py`](label_manifest_expert_q.py) (same folder) to build meta JSONL with **max expert P(unsafe) ≥ threshold** (not native HF labels). If you already have **human `label`** in JSONL, use `generate_expert_features.py` only — see `seed_meta_train_with_experts.jsonl`.
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

### Expert-Q heuristic meta (optional)

Use this when you want meta supervision aligned with **runtime Q features**, or when native labels are weak. **Do not** use it when you need **dataset-original** safe/unsafe semantics — use `build_meta_from_hf_labels.py` above instead.

| Script | Role |
|--------|------|
| [`expert_q_label.py`](expert_q_label.py) | `label_from_expert_q(...)` helper |
| [`generate_expert_q_meta_jsonl.py`](generate_expert_q_meta_jsonl.py) | Local JSONL or single HF dataset → CSV / full JSONL / `train_meta`-ready JSONL |
| [`label_manifest_expert_q.py`](label_manifest_expert_q.py) | Multi-dataset manifest → one merged meta JSONL |
| [`run_expert_q_meta_pipeline.sh`](run_expert_q_meta_pipeline.sh) | One-shot: seed JSONL → expert-Q meta JSONL → `train_meta` |

From a local JSONL with a `text` field:

```bash
python3 training/meta/generate_expert_q_meta_jsonl.py \
  --input-jsonl training/meta/seed_meta_train.jsonl \
  --text-field text \
  --threshold 0.5 \
  --output-csv training/meta/expert_q_labels.csv \
  --output-jsonl training/meta/expert_q_full.jsonl \
  --output-meta-jsonl training/meta/expert_q_for_meta.jsonl
```

Batch HF datasets ([`manifest.expert_q.example.json`](manifest.expert_q.example.json)):

```bash
python training/meta/label_manifest_expert_q.py \
  --manifest training/meta/manifest.expert_q.example.json \
  --threshold 0.5 \
  --require-expert-outputs \
  --out training/meta/pool_expert_q_for_meta.jsonl
```

One-shot pipeline (env: `INPUT_JSONL`, `META_OUT`, `ARTIFACT_OUT`, `LIMIT`, `THRESHOLD`, `N_FOLDS`, `CALIBRATE`, `GROUP_FIELD`):

```bash
LIMIT=200 ./training/meta/run_expert_q_meta_pipeline.sh
THRESHOLD=0.45 GROUP_FIELD=source LIMIT=500 ./training/meta/run_expert_q_meta_pipeline.sh
```

**Breaking change:** scripts previously under `training/teacher_dataset/` now live in this folder. Map: `generate_teacher_labeled_dataset.py` → `generate_expert_q_meta_jsonl.py`, `label_manifest.py` → `label_manifest_expert_q.py`, `run_teacher_meta_pipeline.sh` → `run_expert_q_meta_pipeline.sh`, `manifest.example.json` → `manifest.expert_q.example.json`. See also [`../teacher_dataset/README.md`](../teacher_dataset/README.md).

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
