#!/usr/bin/env python3
"""
Evaluate how predictive each *single* expert is of the teacher label.

This helps estimate the achievable ceiling of the meta model:
if the best single expert AUC is ~0.7 on a domain, it's hard for a simple
meta over the same expert outputs to exceed ~0.8 without better features/data.

Uses p_unsafe_* mapping from `meta_classifier.feature_builder`.

Example:
  python training/meta/eval_single_expert_auc.py \
    --data training/meta/teacher_all_for_meta.jsonl \
    --domain toxicity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# Ensure repo root import works when executed as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_label(v: Any) -> int:
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return 1 if int(v) == 1 else 0
    if isinstance(v, str):
        return 1 if v.strip().lower() in {"unsafe", "harmful", "1", "true", "yes"} else 0
    raise ValueError(f"bad label: {v!r}")


def _p_unsafe(individual_results: Dict[str, Any], expert_name: str) -> float:
    from meta_classifier.feature_builder import _unsafe_probability_from_result

    res = individual_results.get(expert_name, {})
    if not isinstance(res, dict) or "error" in res:
        return 0.5
    return float(_unsafe_probability_from_result(expert_name, res))


def _metrics(y: np.ndarray, p: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (roc_auc, ap). If undefined (single class), returns None.
    """
    if len(np.unique(y)) < 2:
        return None, None
    try:
        auc = float(roc_auc_score(y, p))
    except Exception:
        auc = None
    try:
        ap = float(average_precision_score(y, p))
    except Exception:
        ap = None
    return auc, ap


def main() -> int:
    ap = argparse.ArgumentParser(description="Single-expert AUC vs teacher labels")
    ap.add_argument("--data", type=str, required=True, help="Meta JSONL with label + individual_results")
    ap.add_argument("--domain", type=str, default="", help="Optional domain filter (e.g. toxicity)")
    ap.add_argument("--label-field", type=str, default="label")
    args = ap.parse_args()

    rows = _load_jsonl(Path(args.data))
    if not rows:
        raise SystemExit("empty JSONL")

    domain = args.domain.strip().lower()
    if domain:
        rows = [r for r in rows if str(r.get("domain", "")).strip().lower() == domain]
        if not rows:
            raise SystemExit(f"no rows for domain={domain!r}")

    expert_names = ["jailbreak", "toxicity", "sexual", "factuality"]

    y_list: List[int] = []
    p_by_expert: Dict[str, List[float]] = {e: [] for e in expert_names}
    for r in rows:
        try:
            y = _parse_label(r.get(args.label_field))
        except Exception:
            continue
        ind = r.get("individual_results") or {}
        if not isinstance(ind, dict):
            continue
        y_list.append(int(y))
        for e in expert_names:
            p_by_expert[e].append(_p_unsafe(ind, e))

    y_arr = np.asarray(y_list, dtype=np.int64)
    print(f"rows={len(y_arr)} unsafe={int(y_arr.sum())} unsafe_rate={(float(y_arr.mean()) if len(y_arr) else 0.0):.3f}")
    if domain:
        print(f"domain={domain}")

    # Compute metrics
    results: List[Tuple[str, Optional[float], Optional[float]]] = []
    for e in expert_names:
        p = np.asarray(p_by_expert[e], dtype=np.float64)
        auc, ap_s = _metrics(y_arr, p)
        results.append((e, auc, ap_s))

    # Pretty print
    print()
    print("single-expert metrics (higher is better)")
    print("expert     | ROC-AUC | AP (PR-AUC)")
    print("-----------+---------+-----------")
    for e, auc, ap_s in sorted(results, key=lambda t: (t[1] is not None, t[1] or -1.0), reverse=True):
        auc_s = f"{auc:.4f}" if auc is not None else "n/a"
        ap_s_s = f"{ap_s:.4f}" if ap_s is not None else "n/a"
        print(f"{e:<10} | {auc_s:>7} | {ap_s_s:>9}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

