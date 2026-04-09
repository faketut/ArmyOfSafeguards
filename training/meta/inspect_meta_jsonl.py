#!/usr/bin/env python3
"""
Inspect meta-training JSONL distribution and CV splits.

Useful when CV metrics look inconsistent (e.g. mean fold AUC vs micro OOF AUC),
often due to strong source/domain shifts or label imbalance within groups.

Example:
  python training/meta/inspect_meta_jsonl.py \
    --data training/meta/teacher_all_for_meta.jsonl \
    --group-field source \
    --n-folds 5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_label(row: Dict[str, Any], label_field: str) -> int:
    v = row.get(label_field)
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return 1 if int(v) == 1 else 0
    if isinstance(v, str):
        return 1 if v.lower() in {"unsafe", "harmful", "1", "true", "yes"} else 0
    raise ValueError(f"Unsupported label value: {v!r}")


def _rate(pos: int, n: int) -> float:
    return float(pos) / float(n) if n else 0.0


def _summarize_groups(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    top_k: int,
) -> List[Tuple[str, int, int, float]]:
    by_g: Dict[str, List[int]] = defaultdict(list)
    for yi, gi in zip(y.tolist(), groups.tolist()):
        by_g[str(gi)].append(int(yi))
    rows: List[Tuple[str, int, int, float]] = []
    for g, ys in by_g.items():
        n = len(ys)
        pos = int(sum(ys))
        rows.append((g, n, pos, _rate(pos, n)))
    rows.sort(key=lambda t: t[1], reverse=True)
    if top_k > 0:
        return rows[:top_k]
    return rows


def _print_table(title: str, headers: List[str], rows: List[List[Any]]) -> None:
    print()
    print(title)
    print("-" * len(title))
    if not rows:
        print("(empty)")
        return
    # Simple fixed-width formatting
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(x)) for x in col) for col in cols]
    fmt = " | ".join("{:" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*[str(x) for x in r]))


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect meta JSONL distributions and CV split balance")
    ap.add_argument("--data", type=str, required=True, help="Path to JSONL (meta training)")
    ap.add_argument("--label-field", type=str, default="label")
    ap.add_argument("--group-field", type=str, default="source", help="Group field (default: source)")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--top-k-groups", type=int, default=25, help="Show top K groups by size (0=all)")
    args = ap.parse_args()

    path = Path(args.data)
    rows = _load_jsonl(path)
    if not rows:
        print(f"empty: {path}")
        return 2

    # Labels
    y_list: List[int] = []
    bad_labels = 0
    for r in rows:
        try:
            y_list.append(_parse_label(r, args.label_field))
        except Exception:
            bad_labels += 1
    if bad_labels:
        print(f"warning: {bad_labels} row(s) have unsupported labels (ignored for stats)")
    y = np.asarray(y_list, dtype=np.int64)

    n = len(y)
    pos = int(y.sum())
    neg = int(n - pos)
    print(f"rows: {n}  unsafe: {pos} ({_rate(pos, n):.3f})  safe: {neg} ({_rate(neg, n):.3f})")

    # Quick metadata distributions (if present)
    for key in ("source", "dataset", "domain", "split", "config", "language", "teacher"):
        c = Counter(str(r.get(key, "")) for r in rows if r.get(key) is not None)
        if not c:
            continue
        common = c.most_common(10)
        _print_table(
            f"top {key} values",
            [key, "count"],
            [[k if k else "(missing)", v] for k, v in common],
        )

    # Group stats (for GroupKFold)
    g_field = args.group_field.strip()
    groups: Optional[np.ndarray] = None
    if g_field:
        g_list = [str(r.get(g_field, "default")) for r in rows][: len(y)]
        groups = np.asarray(g_list, dtype=object)
        uniq = len(set(g_list))
        print()
        print(f"group_field={g_field!r} unique_groups={uniq}")
        top_groups = _summarize_groups(y, groups, top_k=args.top_k_groups)
        _print_table(
            f"top groups by size ({g_field})",
            [g_field, "n", "unsafe", "unsafe_rate"],
            [[g, n_i, pos_i, f"{rate:.3f}"] for (g, n_i, pos_i, rate) in top_groups],
        )

    # CV split balance
    if args.n_folds and args.n_folds > 1:
        print()
        print(f"cv: n_folds={args.n_folds}")
        if groups is not None and len(set(groups.tolist())) >= args.n_folds:
            splitter = GroupKFold(n_splits=args.n_folds)
            splits = list(splitter.split(np.zeros((n, 1)), y, groups))
            cv_name = "GroupKFold"
        else:
            splitter = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
            splits = list(splitter.split(np.zeros((n, 1)), y))
            cv_name = "StratifiedKFold"
        print(f"splitter: {cv_name}")

        fold_rows: List[List[Any]] = []
        for k, (tr, va) in enumerate(splits):
            y_va = y[va]
            fold_rows.append(
                [
                    k,
                    len(tr),
                    int(y[tr].sum()),
                    f"{_rate(int(y[tr].sum()), len(tr)):.3f}",
                    len(va),
                    int(y_va.sum()),
                    f"{_rate(int(y_va.sum()), len(va)):.3f}",
                ]
            )
        _print_table(
            "fold label balance",
            ["fold", "train_n", "train_unsafe", "train_rate", "val_n", "val_unsafe", "val_rate"],
            fold_rows,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

