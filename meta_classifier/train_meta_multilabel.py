#!/usr/bin/env python3
"""
Phase 2 / Step B: train per-axis logistic meta heads with monotonicity
constraint on the own axis (own-axis coefficient >= 0).

Why: the current global LR in `meta_classifier/artifacts/meta_lr.json` learned
a *negative* coefficient on the toxicity feature, which means the fused
prediction can move opposite to the toxicity expert on its own axis — that
violates the initial intent ("don't regress vs the specialist").

This trainer fits one head per axis with L-BFGS-B and bounds:
    own-axis coef >= 0, all other coefs free, L2 regularization on all.

Outputs `meta_classifier/artifacts/meta_lr_axis_<axis>.json` per axis using
the existing `LogisticArtifact` JSON schema (feature_names, coef, intercept,
temperature). These are drop-in compatible with `LogisticArtifact.load`.

Usage:
    python3 -m meta_classifier.train_meta_multilabel \\
        --data training/meta/hf_meta_multilabel.jsonl \\
        --outdir meta_classifier/artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from meta_classifier.feature_builder import (
    DEFAULT_EXPERTS,
    FeatureSpec,
    build_feature_vector,
)
from meta_classifier.logistic_model import LogisticArtifact


AXES = DEFAULT_EXPERTS  # ("jailbreak", "toxicity", "sexual", "factuality")


def _load_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _featurize(rows: List[Dict], spec: FeatureSpec) -> np.ndarray:
    return np.asarray(
        [build_feature_vector(r.get("individual_results", {}), spec=spec) for r in rows],
        dtype=np.float64,
    )


def _axis_subset(rows: List[Dict], axis: str) -> Tuple[List[int], np.ndarray]:
    idx: List[int] = []
    y: List[int] = []
    for i, r in enumerate(rows):
        lab = (r.get("labels") or {}).get(axis)
        if lab is None:
            continue
        idx.append(i)
        y.append(int(lab))
    return idx, np.asarray(y, dtype=np.int64)


def _neg_log_lik_l2(
    theta: np.ndarray, X: np.ndarray, y: np.ndarray, l2: float
) -> Tuple[float, np.ndarray]:
    """Logistic NLL + L2 (on weights, not bias). Returns (loss, grad)."""
    w, b = theta[:-1], theta[-1]
    z = X @ w + b
    # log(1+exp(z)) numerically stable
    log1pexp = np.where(z >= 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
    nll = float(np.mean(log1pexp - y * z))
    reg = float(0.5 * l2 * np.dot(w, w))
    sigm = 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
    grad_w = X.T @ (sigm - y) / len(y) + l2 * w
    grad_b = float(np.mean(sigm - y))
    return nll + reg, np.concatenate([grad_w, [grad_b]])


def fit_constrained_lr(
    X: np.ndarray,
    y: np.ndarray,
    own_axis_idx: int,
    l2: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """L-BFGS-B fit with own-axis coef >= 0. Returns (coef, intercept)."""
    n_features = X.shape[1]
    theta0 = np.zeros(n_features + 1, dtype=np.float64)
    bounds: List[Tuple[Optional[float], Optional[float]]] = [(None, None)] * n_features
    bounds[own_axis_idx] = (0.0, None)
    bounds.append((None, None))  # intercept

    res = minimize(
        _neg_log_lik_l2,
        theta0,
        args=(X, y, l2),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    theta = res.x
    return theta[:-1], float(theta[-1])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="multi-label JSONL")
    p.add_argument(
        "--outdir",
        default=str(Path(__file__).parent / "artifacts"),
        help="output directory for per-axis artifacts",
    )
    p.add_argument("--l2", type=float, default=0.1)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-rows", type=int, default=20)
    args = p.parse_args()

    spec = FeatureSpec()
    feat_names = spec.feature_names()
    rows = _load_rows(Path(args.data))
    if not rows:
        raise SystemExit(f"no rows in {args.data}")

    X_all = _featurize(rows, spec)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"features: {feat_names}")
    print(f"trained heads (own-axis coef constrained >= 0):\n")
    print(f"  {'axis':<12} {'rows':>6} {'pos%':>6} {'AUC':>6}  coef + intercept")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6}  {'-'*40}")

    summary: Dict[str, Dict] = {}
    for axis in AXES:
        if axis not in feat_names and f"p_unsafe_{axis}" not in feat_names:
            continue
        own_idx = feat_names.index(f"p_unsafe_{axis}")

        idx, y = _axis_subset(rows, axis)
        if len(idx) < args.min_rows:
            print(f"  {axis:<12} {len(idx):>6}  (skipped: too few rows)")
            continue
        X = X_all[idx]

        # Stratified split for an AUC sanity check.
        if y.sum() > 1 and (len(y) - y.sum()) > 1:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=args.test_size, random_state=args.seed, stratify=y
            )
        else:
            X_tr, X_te, y_tr, y_te = X, X, y, y

        coef, intercept = fit_constrained_lr(X_tr, y_tr, own_idx, l2=args.l2)

        # AUC on held-out (or train if too tiny)
        z = X_te @ coef + intercept
        p_hat = 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
        try:
            auc = float(roc_auc_score(y_te, p_hat))
        except Exception:
            auc = float("nan")

        # Final refit on all axis data.
        coef, intercept = fit_constrained_lr(X, y, own_idx, l2=args.l2)

        artifact = LogisticArtifact(
            feature_names=feat_names,
            coef=[float(c) for c in coef],
            intercept=float(intercept),
            temperature=1.0,
        )
        out_path = outdir / f"meta_lr_axis_{axis}.json"
        artifact.dump(out_path)

        pos_pct = float(y.mean())
        coef_str = ", ".join(f"{n.replace('p_unsafe_', '')}={c:+.2f}" for n, c in zip(feat_names, coef))
        print(f"  {axis:<12} {len(idx):>6} {pos_pct:>6.1%} {auc:>6.3f}  [{coef_str}, b={intercept:+.2f}]")
        summary[axis] = {
            "rows": int(len(idx)),
            "pos_rate": pos_pct,
            "auc": auc,
            "own_axis_coef": float(coef[own_idx]),
            "artifact": str(out_path),
        }

    print(f"\nsaved per-axis artifacts to: {outdir}")
    print("\nfidelity check (own-axis coefs, must be >= 0):")
    ok = True
    for axis, s in summary.items():
        flag = "OK" if s["own_axis_coef"] >= -1e-9 else "FAIL"
        if s["own_axis_coef"] < -1e-9:
            ok = False
        print(f"  {axis:<12} own_axis_coef = {s['own_axis_coef']:+.4f}  [{flag}]")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
