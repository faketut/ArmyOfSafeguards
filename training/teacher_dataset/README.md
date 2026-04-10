# Expert-Q dataset (Q-values + binary label)

Each row runs the four in-repo experts and stores their **Q values** \(P(\text{unsafe})\). The **binary label** is derived in-repo: **unsafe** if `max(Q₁…Q₄) >= --threshold`, else **safe** (same spirit as the meta-classifier feature vector; no external teacher models).

## Format

- **CSV** (default 4 columns + label):

  `q_jailbreak,q_toxicity,q_sexual,q_factuality,label`

  `label` is `1` = unsafe, `0` = safe.

- **JSONL** (full record): includes `q`, `text`, `individual_results`, and fields such as `label_source`, `max_expert_q`, `label_threshold`.

- **Meta JSONL** (`--output-meta-jsonl`): rows compatible with `meta_classifier/train_meta.py` (`individual_results` + `label` as `safe`/`unsafe`).

## Examples

From a local JSONL with a `text` field:

```bash
cd /path/to/ArmyOfSafeguards

python3 training/teacher_dataset/generate_teacher_labeled_dataset.py \
  --input-jsonl training/meta/seed_meta_train.jsonl \
  --text-field text \
  --threshold 0.5 \
  --output-csv training/meta/expert_q_labels.csv \
  --output-jsonl training/meta/expert_q_full.jsonl \
  --output-meta-jsonl training/meta/expert_q_for_meta.jsonl
```

`--dry-run` is a deprecated no-op (behavior matches the default above).

From Hugging Face (example: JailbreakBench harmful behaviors):

```bash
python3 training/teacher_dataset/generate_teacher_labeled_dataset.py \
  --hf-dataset JailbreakBench/JBB-Behaviors \
  --hf-config behaviors \
  --hf-split harmful \
  --hf-text-field Goal \
  --limit 500 \
  --threshold 0.5 \
  --output-meta-jsonl training/meta/jbb_harmful_expert_q.jsonl
```

Then train the meta-classifier:

```bash
python3 -m meta_classifier.train_meta \
  --data training/meta/expert_q_for_meta.jsonl \
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
| `LIMIT` | Max rows |
| `META_OUT` / `ARTIFACT_OUT` | Output paths |

```bash
LIMIT=200 ./training/teacher_dataset/run_teacher_meta_pipeline.sh
```

## Batch: multiple HF datasets (manifest)

To process **multiple** Hugging Face datasets from one manifest and merge into one meta-training JSONL:

```bash
python training/teacher_dataset/label_manifest.py \
  --manifest training/teacher_dataset/manifest.example.json \
  --threshold 0.5 \
  --require-expert-outputs \
  --out training/meta/pool_expert_q_for_meta.jsonl
```

`manifest.example.json` is the only checked-in sample manifest here. Larger pools should be your own JSON with the same entry shape; each row needs a **non-empty** `text_field` (`label_manifest.py` does not infer text columns the way `training/meta/build_meta_from_hf_labels.py` does).

### Manifest filters (optional)

To increase the unsafe rate for hard domains (e.g. `toxicity`, `sexual`), you can pre-filter HF rows **before** running experts:

- `filters_all`: all conditions must match
- `filters_any`: at least one condition must match

Each condition is:

`{"field": "...", "op": "eq|neq|gt|ge|lt|le|in|contains", "value": ...}`

Example (keep only the most toxic rows):

```json
{
  "hf_id": "cglez/civil_comments_clean",
  "splits": ["train"],
  "text_field": "text",
  "domain": "toxicity",
  "filters_any": [
    {"field": "toxicity", "op": "eq", "value": 1},
    {"field": "severe_toxicity", "op": "eq", "value": 1},
    {"field": "threat", "op": "eq", "value": 1}
  ]
}
```

Then train a unified meta policy:

```bash
python -m meta_classifier.train_meta \
  --data training/meta/pool_expert_q_for_meta.jsonl \
  --n-folds 5 \
  --group-field source \
  --calibrate temperature \
  --out meta_classifier/artifacts/meta_lr.json
```

## Label rule & quality

- Tune `--threshold` to trade precision vs recall on your pool; it applies to **max** of the four expert unsafe probabilities.
- Use `--require-expert-outputs` to skip rows when any expert fails (avoids degenerate Q vectors).

See also [docs/experts_training_status.md](../../docs/experts_training_status.md).
