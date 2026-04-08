#!/usr/bin/env bash
# Generate teacher-labeled meta JSONL, then train meta_lr artifact.
#
# Environment (optional):
#   INPUT_JSONL   input JSONL with a "text" field (default: training/meta/seed_meta_train.jsonl)
#   TEACHER       shieldgemma | granite (default: shieldgemma)
#   META_OUT      output meta JSONL path
#   ARTIFACT_OUT  output meta_lr.json path
#   LIMIT         max rows (passed to generator)
#   DEVICE        cuda | cpu
#   DRY_RUN       if set to 1, skip teacher models; heuristic labels only
#
# Usage: ./training/teacher_dataset/run_teacher_meta_pipeline.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

INPUT_JSONL="${INPUT_JSONL:-training/meta/seed_meta_train.jsonl}"
TEACHER="${TEACHER:-shieldgemma}"
META_OUT="${META_OUT:-training/meta/teacher_for_meta.jsonl}"
ARTIFACT_OUT="${ARTIFACT_OUT:-meta_classifier/artifacts/meta_lr.json}"
LIMIT="${LIMIT:-}"
DEVICE="${DEVICE:-cuda}"
DRY_RUN="${DRY_RUN:-0}"

CSV_OUT="${META_OUT%.jsonl}.csv"
JSONL_FULL="${META_OUT%.jsonl}_full.jsonl"

if [ -n "$LIMIT" ] && [ "$DRY_RUN" = "1" ]; then
  python3 training/teacher_dataset/generate_teacher_labeled_dataset.py \
    --input-jsonl "$INPUT_JSONL" \
    --text-field text \
    --teacher "$TEACHER" \
    --device "$DEVICE" \
    --require-expert-outputs \
    --output-meta-jsonl "$META_OUT" \
    --output-csv "$CSV_OUT" \
    --output-jsonl "$JSONL_FULL" \
    --limit "$LIMIT" \
    --dry-run
elif [ -n "$LIMIT" ]; then
  python3 training/teacher_dataset/generate_teacher_labeled_dataset.py \
    --input-jsonl "$INPUT_JSONL" \
    --text-field text \
    --teacher "$TEACHER" \
    --device "$DEVICE" \
    --require-expert-outputs \
    --output-meta-jsonl "$META_OUT" \
    --output-csv "$CSV_OUT" \
    --output-jsonl "$JSONL_FULL" \
    --limit "$LIMIT"
elif [ "$DRY_RUN" = "1" ]; then
  python3 training/teacher_dataset/generate_teacher_labeled_dataset.py \
    --input-jsonl "$INPUT_JSONL" \
    --text-field text \
    --teacher "$TEACHER" \
    --device "$DEVICE" \
    --require-expert-outputs \
    --output-meta-jsonl "$META_OUT" \
    --output-csv "$CSV_OUT" \
    --output-jsonl "$JSONL_FULL" \
    --dry-run
else
  python3 training/teacher_dataset/generate_teacher_labeled_dataset.py \
    --input-jsonl "$INPUT_JSONL" \
    --text-field text \
    --teacher "$TEACHER" \
    --device "$DEVICE" \
    --require-expert-outputs \
    --output-meta-jsonl "$META_OUT" \
    --output-csv "$CSV_OUT" \
    --output-jsonl "$JSONL_FULL"
fi

python3 -m meta_classifier.train_meta \
  --data "$META_OUT" \
  --n-folds 5 \
  --calibrate temperature \
  --out "$ARTIFACT_OUT"

echo "Done. Artifact: $ARTIFACT_OUT"
