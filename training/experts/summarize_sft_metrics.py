#!/usr/bin/env python3
"""
Print a compact text table from expert SFT metrics JSONL (see sequence_classifier_train --metrics-registry).

Example:
  python training/experts/summarize_sft_metrics.py
  python training/experts/summarize_sft_metrics.py path/to/sft_metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize expert SFT metrics JSONL as a table.")
    ap.add_argument(
        "path",
        nargs="?",
        default=str(_REPO_ROOT / "training/experts/sft_metrics.jsonl"),
        help="JSONL path (default: training/experts/sft_metrics.jsonl under repo root)",
    )
    args = ap.parse_args()
    path = Path(args.path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    rows = _load_rows(path)
    if not rows:
        print(f"(no rows in {path})", file=sys.stderr)
        return 1

    # Columns: time, domain, roc_auc, ap, f1, train_n, base_model (truncated)
    print(f"{'recorded_at':<28} {'domain':<12} {'roc_auc':>7} {'ap':>7} {'f1':>6} {'tr_n':>6} {'base_model'}")
    for r in rows:
        m = r.get("metrics") or {}
        ts = str(r.get("recorded_at", ""))[:26]
        dom = str(r.get("expert_domain", ""))[:12]
        auc = m.get("roc_auc")
        apv = m.get("ap")
        f1 = m.get("f1")
        split = r.get("split") or {}
        tn = split.get("train_n", "")
        bm = str(r.get("base_model", ""))
        if len(bm) > 42:
            bm = bm[:39] + "..."
        auc_s = f"{auc:.4f}" if isinstance(auc, (int, float)) else "n/a"
        ap_s = f"{apv:.4f}" if isinstance(apv, (int, float)) else "n/a"
        f1_s = f"{f1:.4f}" if isinstance(f1, (int, float)) else "n/a"
        print(f"{ts:<28} {dom:<12} {auc_s:>7} {ap_s:>7} {f1_s:>6} {str(tn):>6} {bm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
