# Teacher-labeled dataset (Q-values + label)

Use a **teacher** model (ShieldGemma or Granite Guardian) to assign **binary unsafe/safe labels**, while each row includes the four expert **Q values** \(P(\text{unsafe})\) used by the meta-classifier.

## Format

- **CSV** (default 4 columns + label):

  `q_jailbreak,q_toxicity,q_sexual,q_factuality,label`

  `label` is `1` = unsafe, `0` = safe (from teacher triage: not `is_safe`).

- **JSONL** (full record): includes `q`, `text`, `individual_results`, and teacher metadata.

- **Meta JSONL** (`--output-meta-jsonl`): rows compatible with `meta_classifier/train_meta.py` (`individual_results` + `label` as `safe`/`unsafe`).

## Examples

From a local JSONL with a `text` field (runs all four experts + ShieldGemma):

```bash
cd /path/to/ArmyOfSafeguards

## If you need gated model/dataset access:
## - Copy `.env.example` -> `.env` and set HF_TOKEN, or run `huggingface-cli login`

python3 training/teacher_dataset/generate_teacher_labeled_dataset.py \
  --input-jsonl training/meta/seed_meta_train.jsonl \
  --text-field text \
  --teacher shieldgemma \
  --device cuda \
  --output-csv training/meta/teacher_q_labels.csv \
  --output-jsonl training/meta/teacher_full.jsonl \
  --output-meta-jsonl training/meta/teacher_for_meta.jsonl
```

**Dry-run** (no ShieldGemma/Granite; labels from a heuristic on Q only — for plumbing tests):

```bash
python3 training/teacher_dataset/generate_teacher_labeled_dataset.py \
  --input-jsonl training/meta/seed_meta_train.jsonl \
  --text-field text \
  --dry-run \
  --output-csv training/meta/dryrun_q_labels.csv \
  --output-meta-jsonl training/meta/dryrun_for_meta.jsonl
```

From Hugging Face (example: JailbreakBench harmful behaviors):

```bash
python3 training/teacher_dataset/generate_teacher_labeled_dataset.py \
  --hf-dataset JailbreakBench/JBB-Behaviors \
  --hf-config behaviors \
  --hf-split harmful \
  --hf-text-field Goal \
  --limit 500 \
  --teacher granite \
  --output-meta-jsonl training/meta/jbb_harmful_teacher.jsonl
```

Then train the meta-classifier:

```bash
python3 -m meta_classifier.train_meta \
  --data training/meta/teacher_for_meta.jsonl \
  --n-folds 5 \
  --group-field source \
  --calibrate temperature \
  --out meta_classifier/artifacts/meta_lr.json
```

### One-shot script

`run_teacher_meta_pipeline.sh` runs the generator and `train_meta` in one go. Optional environment variables:

| Variable | Meaning |
|----------|---------|
| `INPUT_JSONL` | Input JSONL (default `training/meta/seed_meta_train.jsonl`) |
| `TEACHER` | `shieldgemma` or `granite` |
| `LIMIT` | Max rows |
| `DEVICE` | `cuda` or `cpu` |
| `DRY_RUN` | Set to `1` for `--dry-run` (no teacher load) |
| `META_OUT` / `ARTIFACT_OUT` | Output paths |

```bash
LIMIT=200 ./training/teacher_dataset/run_teacher_meta_pipeline.sh
DRY_RUN=1 LIMIT=50 ./training/teacher_dataset/run_teacher_meta_pipeline.sh
```

## Teachers

| Flag | Model | Notes |
|------|-------|--------|
| `--teacher shieldgemma` | `google/shieldgemma-2b` | **Gated on Hugging Face** — accept terms on the model page, then `huggingface-cli login` or set `HF_TOKEN` (e.g. via `.env`). |
| `--teacher granite` | `ibm-granite/granite-guardian-3.3-8b` | Large; may need vLLM + VRAM; check HF access terms. |

Use `--require-expert-outputs` to refuse writing rows when any expert fails (avoids all-0.5 Q vectors). Teacher rows are skipped if the wrapper returns an error (e.g. 401 on gated repos without a token).

See also [docs/experts_training_status.md](../../docs/experts_training_status.md).
