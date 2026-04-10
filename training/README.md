# Training

Entry points for fine-tuning experts, building meta-aggregator data, and running end-to-end pipelines.

| Path | Purpose |
|------|---------|
| [meta/README.md](meta/README.md) | Meta JSONL schema helpers, synthetic data, `generate_expert_features.py`, `train_meta` / **native HF** meta build |
| [experts/curriculum_native.yaml](experts/curriculum_native.yaml) | Per-expert **native-label** pools (jailbreak / toxicity / sexual / factuality) — no teacher |
| [experts/build_expert_sft_jsonl.py](experts/build_expert_sft_jsonl.py) | Build SFT JSONL from the curriculum (`--expert`, optional `--target-train-pos-rate`) |
| [common/sequence_classifier_train.py](common/sequence_classifier_train.py) | Shared HF Trainer loop used by expert `train.py` entry points below |
| [common/hf_datasets.py](common/hf_datasets.py) | `load_hf_split` — loads splits without deprecated `trust_remote_code`; **FEVER** uses Hub `refs/convert/parquet` (no script) |
| [teacher_dataset/README.md](teacher_dataset/README.md) | **Q-values** (per-expert P(unsafe)) + in-repo binary label (max-Q vs threshold) → CSV + meta-ready JSONL |
| [teacher_dataset/run_teacher_meta_pipeline.sh](teacher_dataset/run_teacher_meta_pipeline.sh) | One-shot: teacher-labeled JSONL → `meta_classifier/artifacts/meta_lr.json` |
| [jailbreak/train.py](jailbreak/train.py) | Fine-tune jailbreak expert from curriculum JSONL (`AOS_JAILBREAK_MODEL` at runtime) |
| [toxicity/train.py](toxicity/train.py) | Fine-tune toxicity on JSONL or `--hf-manifest` (`AOS_TOXICITY_MODEL` at runtime) |
| [factuality/train.py](factuality/train.py) | Fine-tune factuality from curriculum JSONL (`AOS_FACTUALITY_MODEL` at runtime) |
| [sexual/train.py](sexual/train.py) | Fine-tune sexual/sensitive head from curriculum JSONL (`AOS_SEXUAL_MODEL` at runtime) |
| [sexual/](sexual/) | `train.py` for SFT (curriculum → JSONL via `experts/build_expert_sft_jsonl.py`) |

**Docs:** [docs/experts_training_status.md](../docs/experts_training_status.md), [docs/meta_training_labeling.md](../docs/meta_training_labeling.md).

### Quick: expert-Q labels → trained meta

```bash
export LIMIT=500   # optional cap for iteration
./training/teacher_dataset/run_teacher_meta_pipeline.sh
```

### Native experts + unified meta

1. **Per-expert SFT JSONL** — [experts/curriculum_native.yaml](experts/curriculum_native.yaml) gives each expert positives **only** from that head’s README-native unsafe data; rows that are unsafe under *other* heads (jailbreak / toxicity / sexual / factuality) are pooled as **negatives** (`y=0`) for this head. Edit pools if needed, then:

   **HF access:** Run `huggingface-cli login` or set `HF_TOKEN` for **gated** datasets (e.g. `cardiffnlp/x_sensitive`, JailbreakBench pools, some benchmarks). Build SFT JSONL **before** `train.py`; if the builder fails, you will not have `--data` yet (missing JSONL is expected until the builder succeeds).

```bash
python training/experts/build_expert_sft_jsonl.py --expert jailbreak --out data/jailbreak_native.jsonl --target-train-pos-rate 0.3
python training/jailbreak/train.py --data data/jailbreak_native.jsonl
```

Toxicity, sexual, and factuality (same curriculum builder; use the matching `train.py` defaults):

```bash
python training/experts/build_expert_sft_jsonl.py --expert toxicity --out data/toxicity_native.jsonl
python training/toxicity/train.py --data data/toxicity_native.jsonl --domain toxicity --output-dir experts/artifacts/toxicity_ft

python training/experts/build_expert_sft_jsonl.py --expert sexual --out data/sexual_native.jsonl
python training/sexual/train.py --data data/sexual_native.jsonl

python training/experts/build_expert_sft_jsonl.py --expert factuality --out data/factuality_native.jsonl
python training/factuality/train.py --data data/factuality_native.jsonl
```

You can still use `training/toxicity/train.py --data ... --domain <other>` for ad-hoc domains; the dedicated scripts set the right backbone, `max_length`, and output dir.

2. **Meta JSONL with native labels** — Multi-dataset manifest (example: [meta/meta_native_pool.example.json](meta/meta_native_pool.example.json)):

```bash
python training/meta/build_meta_from_hf_labels.py --datasets-manifest training/meta/meta_native_pool.example.json --out training/meta/meta_native_with_experts.jsonl
```

Point experts at fine-tuned weights via env vars before running so `individual_results` match production: `AOS_JAILBREAK_MODEL`, `AOS_TOXICITY_MODEL`, `AOS_SEXUAL_MODEL`, `AOS_FACTUALITY_MODEL` (each accepts a local path or Hugging Face model id).

3. **Meta classifier** — Logistic: `python meta_classifier/train_meta.py --data ...`. **XGBoost / MLP:**

```bash
python meta_classifier/train_meta_tabular.py --data training/meta/meta_native_with_experts.jsonl --out-dir meta_classifier/artifacts/meta_xgb_native --algo xgb --n-folds 5 --calibrate temperature
```

Set `AOS_META_MODEL_PATH` to that **directory** (it must contain `manifest.json` + `xgb_model.json`).
