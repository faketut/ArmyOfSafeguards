#!/usr/bin/env python3
"""
Phase 2 / Step C: build a multi-label meta training JSONL from existing
per-source HF jsonl rows in `training/meta/`.

Each input row has a single `label` (safe/unsafe) but its `dataset` field
identifies the home axis (toxicity / sexual / jailbreak / factuality).
We emit `labels: {axis: 0/1/null}` so per-axis heads can train only on
rows where their axis is supervised.

Usage:
    python3 -m training.meta.build_multilabel_meta \\
        --input training/meta/hf_meta_all.jsonl \\
        --out training/meta/hf_meta_multilabel.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

# Single source of truth for dataset -> axis routing.
DATASET_TO_AXIS: Dict[str, str] = {
    "toxigen/toxigen-data": "toxicity",
    "pravalika-9/hate_speech_twitter": "toxicity",
    "cglez/civil_comments_clean": "toxicity",
    "cardiffnlp/x_sensitive": "sexual",
    "jackhhao/jailbreak-classification": "jailbreak",
    "jailbreakbench/jbb-behaviors": "jailbreak",
    "allenai/wildguardmix": "jailbreak",
    # Factuality sources (Phase 2 final slice). build_meta_from_hf_labels.py
    # already knows how to map their native labels.
    "truthful_qa": "factuality",
    "domenicrosati/TruthfulQA": "factuality",
    "fever": "factuality",
    "fever/fever": "factuality",
    "tals/vitaminc": "factuality",
    "climate_fever": "factuality",
    "tdiggelm/climate_fever": "factuality",
}

AXES = ("jailbreak", "toxicity", "sexual", "factuality")


def _label_to_int(v) -> Optional[int]:
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return 1 if int(v) == 1 else 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"unsafe", "harmful", "1", "true", "yes"}:
            return 1
        if s in {"safe", "unharmful", "benign", "0", "false", "no"}:
            return 0
    return None


def _axis_for(dataset: str) -> Optional[str]:
    return DATASET_TO_AXIS.get(dataset.strip().lower()) if dataset else None


def transform(row: Dict) -> Optional[Dict]:
    if not isinstance(row.get("individual_results"), dict):
        return None
    axis = _axis_for(str(row.get("dataset", "")))
    if axis is None:
        return None
    y = _label_to_int(row.get("label"))
    if y is None:
        return None
    labels = {a: None for a in AXES}
    labels[axis] = y
    return {
        "text": row.get("text", ""),
        "dataset": row.get("dataset", ""),
        "home_axis": axis,
        "labels": labels,
        "individual_results": row["individual_results"],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    src = Path(args.input)
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = 0
    counts = {a: 0 for a in AXES}
    pos = {a: 0 for a in AXES}
    skipped_no_axis = 0
    with src.open() as f, dst.open("w") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            out = transform(row)
            if out is None:
                if _axis_for(str(row.get("dataset", ""))) is None:
                    skipped_no_axis += 1
                continue
            n_out += 1
            for a, v in out["labels"].items():
                if v is not None:
                    counts[a] += 1
                    pos[a] += int(v)
            g.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"in={n_in} out={n_out} skipped_no_axis={skipped_no_axis}")
    print("per-axis labeled counts (and positives):")
    for a in AXES:
        c = counts[a]
        pr = (pos[a] / c) if c else 0.0
        print(f"  {a:<12} labeled={c:<5} pos={pos[a]:<5} pos_rate={pr:.2%}")
    print(f"wrote: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
