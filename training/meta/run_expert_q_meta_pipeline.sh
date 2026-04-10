#!/usr/bin/env bash
# Build expert-Q–labeled meta JSONL (max expert P(unsafe) vs threshold), then train meta_lr.
#
# Environment (optional):
#   INPUT_JSONL   input JSONL with a "text" field (default: training/meta/seed_meta_train.jsonl)
#   META_OUT      output meta JSONL path (default: training/meta/expert_q_for_meta.jsonl)
#   ARTIFACT_OUT  output meta_lr.json path (default: meta_classifier/artifacts/meta_lr.json)
#   LIMIT         max rows (passed to generator)
#   THRESHOLD     --threshold for expert-Q label (default: 0.5)
#   N_FOLDS       train_meta --n-folds (default: 5)
#   CALIBRATE     train_meta --calibrate (default: temperature)
#   GROUP_FIELD   if set, passed as train_meta --group-field (e.g. source)
#
# Usage: ./training/meta/run_expert_q_meta_pipeline.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

INPUT_JSONL="${INPUT_JSONL:-training/meta/seed_meta_train.jsonl}"
META_OUT="${META_OUT:-training/meta/expert_q_for_meta.jsonl}"
ARTIFACT_OUT="${ARTIFACT_OUT:-meta_classifier/artifacts/meta_lr.json}"
LIMIT="${LIMIT:-}"
THRESHOLD="${THRESHOLD:-0.5}"
N_FOLDS="${N_FOLDS:-5}"
CALIBRATE="${CALIBRATE:-temperature}"
GROUP_FIELD="${GROUP_FIELD:-}"

CSV_OUT="${META_OUT%.jsonl}.csv"
JSONL_FULL="${META_OUT%.jsonl}_full.jsonl"

GEN=(python3 training/meta/generate_expert_q_meta_jsonl.py
  --input-jsonl "$INPUT_JSONL"
  --text-field text
  --threshold "$THRESHOLD"
  --require-expert-outputs
  --output-meta-jsonl "$META_OUT"
  --output-csv "$CSV_OUT"
  --output-jsonl "$JSONL_FULL")

if [ -n "$LIMIT" ]; then
  GEN+=(--limit "$LIMIT")
fi

"${GEN[@]}"

TRAIN=(python3 -m meta_classifier.train_meta
  --data "$META_OUT"
  --n-folds "$N_FOLDS"
  --calibrate "$CALIBRATE"
  --out "$ARTIFACT_OUT")

if [ -n "$GROUP_FIELD" ]; then
  TRAIN+=(--group-field "$GROUP_FIELD")
fi

"${TRAIN[@]}"

echo "Done. Artifact: $ARTIFACT_OUT"
