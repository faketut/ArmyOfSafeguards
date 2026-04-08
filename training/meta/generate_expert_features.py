#!/usr/bin/env python3
"""
Append `individual_results` to each JSONL row by running all safeguard experts.

Input rows must include a `text` field. Existing `individual_results` are overwritten
unless --skip-existing is set.

Usage:
  python training/meta/generate_expert_features.py \\
    --input training/meta/seed_meta_train.jsonl \\
    --output training/meta/seed_meta_train_with_experts.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aggregator.expert_runner import run_all_safeguards  # noqa: E402


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run experts and fill individual_results in JSONL")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSONL path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output JSONL path")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not overwrite rows that already have a non-empty individual_results dict",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not load models; write placeholder individual_results (p_unsafe=0.5 via errors)",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    rows = _load_jsonl(inp)
    out.parent.mkdir(parents=True, exist_ok=True)

    for row in rows:
        if args.skip_existing and row.get("individual_results"):
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            row["individual_results"] = {"error": "missing or empty text"}
            continue
        if args.dry_run:
            row["individual_results"] = {
                "jailbreak": {"label": False, "confidence": 0.5},
                "toxicity": {"label": "LABEL_0", "confidence": 0.5},
                "sexual": {"label": "LABEL_0", "confidence": 0.5},
                "factuality": {"label": "LABEL_0", "confidence": 0.5},
            }
            row["_dry_run"] = True
        else:
            row["individual_results"] = run_all_safeguards(text)

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
