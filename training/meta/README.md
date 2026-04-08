# Meta aggregator training pipeline

## Overview

1. **Label data** — Follow [docs/meta_training_labeling.md](../../docs/meta_training_labeling.md).
2. **Seed JSONL** — `seed_meta_train.jsonl`: rows with `text`, `label`, optional `id` / `domain` / `source`.
3. **Expert features** — Run `generate_expert_features.py` to append `individual_results` (calls all experts).
4. **Optional: teacher labels** — Use [../teacher_dataset/README.md](../teacher_dataset/README.md) to build CSV / meta JSONL with **ShieldGemma** or **Granite Guardian** (requires HF access token for gated models). If you already have **human `label`** in JSONL, use `generate_expert_features.py` only — see `seed_meta_train_with_experts.jsonl` (generated from `seed_meta_train.jsonl`).
5. **Train meta** — `python -m meta_classifier.train_meta --data ...` (see [meta_classifier/train_meta.py](../../meta_classifier/train_meta.py) for OOF and calibration flags).
6. **Point runtime at the artifact** — Set `AOS_META_MODEL_PATH` or place `meta_classifier/artifacts/meta_lr.json`.

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

## Offline / CI without loading HF experts

Use `synthetic_meta_train.jsonl` (checked in) to smoke-test `train_meta.py` without downloading models. For production, always generate `individual_results` from real experts.
