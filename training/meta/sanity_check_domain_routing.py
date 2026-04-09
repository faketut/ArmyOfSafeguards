#!/usr/bin/env python3
"""
Quick sanity-check for domain-aware meta routing.

Loads a meta-training JSONL and prints routed P(unsafe) for a few rows.

Example:
  python training/meta/sanity_check_domain_routing.py \
    --data training/meta/teacher_all_for_meta.jsonl \
    --n 5

You can control routing via env:
  - AOS_META_MODEL_PATH_TOXICITY / _SEXUAL / _JAILBREAK / _MIXED
  - or AOS_META_MODEL_MAP_JSON
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from meta_classifier.predict import meta_predict_proba_routed  # noqa: E402


def _load_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanity-check domain-aware meta routing artifacts")
    ap.add_argument("--data", type=str, required=True, help="Meta JSONL (must contain individual_results)")
    ap.add_argument("--n", type=int, default=5, help="Number of rows to print")
    args = ap.parse_args()

    path = Path(args.data)
    rows = _load_jsonl(path, limit=max(1, int(args.n)))
    if not rows:
        raise SystemExit("no rows loaded")

    print("env routing overrides:")
    for k in (
        "AOS_META_MODEL_PATH",
        "AOS_META_MODEL_MAP_JSON",
        "AOS_META_MODEL_PATH_TOXICITY",
        "AOS_META_MODEL_PATH_SEXUAL",
        "AOS_META_MODEL_PATH_JAILBREAK",
        "AOS_META_MODEL_PATH_MIXED",
    ):
        if os.environ.get(k, "").strip():
            print(f"  {k}=<set>")

    print()
    for i, r in enumerate(rows):
        ind = r.get("individual_results") or {}
        domain = str(r.get("domain", "") or "").strip() or "toxicity"
        text = str(r.get("text", "") or "")[:140].replace("\n", " ")
        p = meta_predict_proba_routed(ind, domain=domain)
        print(f"[{i}] domain={domain} p_unsafe={p:.4f} label={r.get('label')!r} source={r.get('source')!r}")
        print(f"     text={text!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

