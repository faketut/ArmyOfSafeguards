#!/usr/bin/env python3
"""
Resample a meta-training JSONL to enforce a target unsafe rate per group.

Strategy (per group, e.g. source):
  - Keep ALL unsafe rows.
  - Downsample safe rows so unsafe_rate ~= target.

This is useful when GroupKFold metrics are unstable due to strong label
imbalance and distribution shift across groups.

Example:
  python training/meta/resample_meta_jsonl.py \
    --in training/meta/teacher_all_for_meta.jsonl \
    --out training/meta/teacher_all_for_meta.r20.jsonl \
    --group-field source \
    --target-unsafe-rate 0.2 \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _is_unsafe(row: Dict[str, Any], label_field: str) -> bool:
    v = row.get(label_field)
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, float)):
        return int(v) == 1
    if isinstance(v, str):
        return v.strip().lower() in {"unsafe", "harmful", "1", "true", "yes"}
    return False


def _safe_keep_count(u: int, target: float) -> int:
    # Want u / (u + s) ~= target => s ~= u * (1-target)/target
    if u <= 0:
        return 0
    target = float(target)
    if not (0.0 < target < 1.0):
        raise ValueError("--target-unsafe-rate must be in (0,1)")
    s = u * (1.0 - target) / target
    return int(math.floor(s))


def main() -> int:
    ap = argparse.ArgumentParser(description="Resample meta JSONL to target unsafe rate per group")
    ap.add_argument("--in", dest="inp", type=str, required=True, help="Input JSONL path")
    ap.add_argument("--out", type=str, required=True, help="Output JSONL path")
    ap.add_argument("--label-field", type=str, default="label")
    ap.add_argument("--group-field", type=str, default="source")
    ap.add_argument("--target-unsafe-rate", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--min-unsafe-per-group",
        type=int,
        default=1,
        help="Drop groups with fewer unsafe than this (default: 1).",
    )
    ap.add_argument(
        "--max-safe-per-unsafe",
        type=int,
        default=0,
        help="Optional cap: keep at most K safe rows per unsafe row (0 disables).",
    )
    ap.add_argument(
        "--shuffle-output",
        action="store_true",
        help="Shuffle output rows (recommended).",
    )
    args = ap.parse_args()

    rnd = random.Random(int(args.seed))
    rows = _load_jsonl(Path(args.inp))
    if not rows:
        raise SystemExit("input JSONL is empty")

    gfield = args.group_field.strip()
    if not gfield:
        raise SystemExit("--group-field must be non-empty")

    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g = str(r.get(gfield, "default"))
        by_group[g].append(r)

    kept: List[Dict[str, Any]] = []
    dropped_groups = 0
    for g, rs in by_group.items():
        unsafe = [r for r in rs if _is_unsafe(r, args.label_field)]
        safe = [r for r in rs if not _is_unsafe(r, args.label_field)]

        u = len(unsafe)
        if u < int(args.min_unsafe_per_group):
            dropped_groups += 1
            continue

        safe_keep = min(len(safe), _safe_keep_count(u, args.target_unsafe_rate))
        if args.max_safe_per_unsafe and int(args.max_safe_per_unsafe) > 0:
            safe_keep = min(safe_keep, u * int(args.max_safe_per_unsafe))

        rnd.shuffle(safe)
        kept.extend(unsafe)
        kept.extend(safe[:safe_keep])

    if args.shuffle_output:
        rnd.shuffle(kept)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Report
    n = len(kept)
    u = sum(1 for r in kept if _is_unsafe(r, args.label_field))
    rate = (u / n) if n else 0.0
    print(
        f"Wrote {n} rows to {out_path} (unsafe={u}, unsafe_rate={rate:.3f}). "
        f"Dropped groups: {dropped_groups}/{len(by_group)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

