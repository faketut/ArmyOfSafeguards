#!/usr/bin/env python3
"""
Split a JSONL file into multiple JSONLs by a field (e.g. domain/source).

Example:
  python training/meta/split_jsonl_by_field.py \
    --in training/meta/teacher_all_for_meta.jsonl \
    --field domain \
    --out-dir training/meta/splits_domain
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _slug(s: str) -> str:
    s = s.strip().lower()
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
        else:
            out.append("_")
    return "".join(out).strip("_") or "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="Split JSONL by a field value")
    ap.add_argument("--in", dest="inp", type=str, required=True, help="Input JSONL")
    ap.add_argument("--field", type=str, required=True, help="Field to split by (e.g. domain, source)")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory")
    ap.add_argument("--unknown-value", type=str, default="(missing)", help="Bucket for missing field")
    ap.add_argument("--min-rows", type=int, default=1, help="Only write buckets with >= min-rows")
    args = ap.parse_args()

    inp = Path(args.inp)
    rows = _load_jsonl(inp)
    if not rows:
        raise SystemExit("input JSONL is empty")

    field = args.field.strip()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    counts = Counter()
    for r in rows:
        v = r.get(field, None)
        key = args.unknown_value if v is None else str(v)
        counts[key] += 1
        buckets.setdefault(key, []).append(r)

    written_files = 0
    for key, rs in sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True):
        if len(rs) < int(args.min_rows):
            continue
        name = _slug(f"{field}__{key}")
        out_path = out_dir / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        written_files += 1

    print(f"Read {len(rows)} rows from {inp}")
    print(f"Wrote {written_files} file(s) to {out_dir}")
    print("Top buckets:")
    for k, c in counts.most_common(10):
        print(f"  - {k}: {c}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

