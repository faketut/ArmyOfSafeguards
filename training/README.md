# Training

Entry points for fine-tuning experts, building meta-aggregator data, and running end-to-end pipelines.

| Path | Purpose |
|------|---------|
| [meta/README.md](meta/README.md) | Meta JSONL schema helpers, synthetic data, `generate_expert_features.py`, `train_meta` / **native HF** meta build |
| [experts/curriculum_native.yaml](experts/curriculum_native.yaml) | Per-expert **native-label** pools (jailbreak / toxicity / sexual / factuality) — no teacher |
| [experts/build_expert_sft_jsonl.py](experts/build_expert_sft_jsonl.py) | Build SFT JSONL from the curriculum (`--expert`, optional `--target-train-pos-rate`) |
| [common/sequence_classifier_train.py](common/sequence_classifier_train.py) | Shared HF Trainer loop used by `toxicity/train.py` and `jailbreak/train.py` |
| [teacher_dataset/README.md](teacher_dataset/README.md) | **Q-values** (per-expert P(unsafe)) + in-repo binary label (max-Q vs threshold) → CSV + meta-ready JSONL |
| [teacher_dataset/run_teacher_meta_pipeline.sh](teacher_dataset/run_teacher_meta_pipeline.sh) | One-shot: teacher-labeled JSONL → `meta_classifier/artifacts/meta_lr.json` |
| [jailbreak/train.py](jailbreak/train.py) | Fine-tune jailbreak expert from curriculum JSONL (`AOS_JAILBREAK_MODEL` at runtime) |
| [toxicity/train.py](toxicity/train.py) | Fine-tune toxicity (or other domain rows) on JSONL or `--hf-manifest` |
| `sexual/` | Notebooks for sensitive-content expert experiments |

**Docs:** [docs/experts_training_status.md](../docs/experts_training_status.md), [docs/meta_training_labeling.md](../docs/meta_training_labeling.md).

### Quick: expert-Q labels → trained meta

```bash
export LIMIT=500   # optional cap for iteration
./training/teacher_dataset/run_teacher_meta_pipeline.sh
```

### Native experts + unified meta

1. **Per-expert SFT JSONL** — [experts/curriculum_native.yaml](experts/curriculum_native.yaml) gives each expert positives **only** from that head’s README-native unsafe data; rows that are unsafe under *other* heads (jailbreak / toxicity / sexual / factuality) are pooled as **negatives** (`y=0`) for this head. Edit pools if needed, then:

```bash
python training/experts/build_expert_sft_jsonl.py --expert jailbreak --out data/jailbreak_native.jsonl --target-train-pos-rate 0.3
python training/jailbreak/train.py --data data/jailbreak_native.jsonl --domain jailbreak --output-dir experts/artifacts/jailbreak_ft
```

Repeat with `--expert toxicity|sexual|factuality` and `python training/toxicity/train.py --data ... --domain <expert> --output-dir experts/artifacts/<expert>_ft` (set `AOS_TOXICITY_MODEL` etc. for runtime).

2. **Meta JSONL with native labels** — Multi-dataset manifest (example: [meta/meta_native_pool.example.json](meta/meta_native_pool.example.json)):

```bash
python training/meta/build_meta_from_hf_labels.py --datasets-manifest training/meta/meta_native_pool.example.json --out training/meta/meta_native_with_experts.jsonl
```

Point experts at fine-tuned weights via env vars (`AOS_JAILBREAK_MODEL`, `AOS_TOXICITY_MODEL`, …) before running so `individual_results` match production.

3. **Meta classifier** — Logistic: `python meta_classifier/train_meta.py --data ...`. **XGBoost / MLP:**

```bash
python meta_classifier/train_meta_tabular.py --data training/meta/meta_native_with_experts.jsonl --out-dir meta_classifier/artifacts/meta_xgb_native --algo xgb --n-folds 5 --calibrate temperature
```

Set `AOS_META_MODEL_PATH` to that **directory** (it must contain `manifest.json` + `xgb_model.json`).
