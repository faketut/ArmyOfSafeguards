from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

from meta_classifier.feature_builder import FeatureSpec, build_feature_vector
from meta_classifier.logistic_model import LogisticArtifact


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


def build_dataset(
    rows: List[Dict[str, Any]],
    spec: FeatureSpec,
    label_field: str,
    group_field: str = "",
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[np.ndarray]]:
    X_list: List[List[float]] = []
    y_list: List[int] = []
    group_list: List[str] = []
    skipped: List[str] = []
    for i, row in enumerate(rows):
        ind = row.get("individual_results") or row.get("experts") or {}
        if not isinstance(ind, dict):
            skipped.append(f"row {i}: no individual_results")
            continue
        try:
            X_list.append(build_feature_vector(ind, spec=spec))
            y_list.append(_parse_label(row, label_field))
            if group_field:
                group_list.append(str(row.get(group_field, "default")))
        except Exception as e:
            skipped.append(f"row {i}: {e}")
    if skipped and len(skipped) <= 10:
        for s in skipped:
            print("warning:", s)
    elif skipped:
        print(f"warning: skipped {len(skipped)} rows (showing first 5)")
        for s in skipped[:5]:
            print("warning:", s)
    X = np.asarray(X_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.int64)
    groups: Optional[np.ndarray]
    if group_field and group_list:
        groups = np.asarray(group_list, dtype=object)
    else:
        groups = None
    return X, y, skipped, groups


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def _apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    eps = 1e-9
    p = np.clip(p, eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))
    return _sigmoid(logit / float(temperature))


def tune_temperature(y_true: np.ndarray, p_raw: np.ndarray, grid: Sequence[float]) -> float:
    """Pick T minimizing binary cross-entropy on calibration predictions."""
    y_true = np.asarray(y_true, dtype=np.float64)
    p_raw = np.clip(np.asarray(p_raw, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    best_t, best_loss = 1.0, float("inf")
    for t in grid:
        if t <= 0:
            continue
        p_cal = _apply_temperature(p_raw, t)
        loss = log_loss(y_true, p_cal, labels=[0, 1])
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return best_t


def cross_val_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    n_folds: int,
    seed: int,
) -> Tuple[np.ndarray, List[float]]:
    """
    Out-of-fold positive-class probabilities.
    Returns (oof_proba, fold_aucs).
    """
    oof = np.zeros(len(y), dtype=np.float64)
    fold_aucs: List[float] = []

    if groups is not None:
        splitter = GroupKFold(n_splits=n_folds)
        splits = list(splitter.split(X, y, groups))
    else:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y))

    for train_idx, val_idx in splits:
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(X[train_idx], y[train_idx])
        p = clf.predict_proba(X[val_idx])[:, 1]
        oof[val_idx] = p
        try:
            fold_aucs.append(float(roc_auc_score(y[val_idx], p)))
        except Exception:
            fold_aucs.append(float("nan"))

    return oof, fold_aucs


def main() -> int:
    parser = argparse.ArgumentParser(description="Train LR meta-classifier over expert outputs")
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL: individual_results + label")
    parser.add_argument("--label-field", type=str, default="label", help="Label field name (default: label)")
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "artifacts" / "meta_lr.json"), help="Output artifact path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction when --n-folds is 0 (default: 0.2)")
    parser.add_argument(
        "--n-folds",
        type=int,
        default=0,
        help="If >1, run stratified (or group) K-fold OOF metrics; final model still trained on all data unless --no-final-fit-all",
    )
    parser.add_argument(
        "--group-field",
        type=str,
        default="",
        help="Optional JSONL field for GroupKFold (e.g. source) to reduce template leakage",
    )
    parser.add_argument(
        "--calibrate",
        type=str,
        choices=["none", "temperature"],
        default="none",
        help="Post-hoc temperature scaling on meta probability (fit on OOF predictions when n-folds>1, else on holdout)",
    )
    parser.add_argument(
        "--no-final-fit-all",
        action="store_true",
        help="After CV, fit only on training part of a single train_test_split (not recommended for deployment)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    spec = FeatureSpec()
    data_path = Path(args.data)
    rows = _load_jsonl(data_path)
    if not rows:
        raise SystemExit(
            f"No rows loaded from {str(data_path)!r} (empty file or only blank lines). "
            f"Run label_manifest first and ensure it prints a positive 'Wrote N rows' count."
        )
    X, y, _, groups = build_dataset(
        rows, spec=spec, label_field=args.label_field, group_field=args.group_field
    )

    if len(X) < 20:
        raise SystemExit(
            f"Not enough training rows after featurization: {len(X)} (loaded {len(rows)} JSONL lines). "
            f"Often all rows lack usable individual_results — see warnings above."
        )

    if groups is not None and args.n_folds > 1 and len(np.unique(groups)) < args.n_folds:
        print("warning: not enough unique groups for K-fold; falling back to stratified KFold")
        groups = None

    temperature = 1.0
    oof_proba: Optional[np.ndarray] = None

    if args.n_folds and args.n_folds > 1:
        oof_proba, fold_aucs = cross_val_oof(X, y, groups, args.n_folds, args.seed)
        print(f"OOF ROC-AUC (mean of folds): {np.nanmean(fold_aucs):.4f}")
        try:
            print(f"OOF ROC-AUC (micro on all OOF preds): {roc_auc_score(y, oof_proba):.4f}")
        except Exception as e:
            print("OOF ROC-AUC (full):", e)
        if args.calibrate == "temperature" and oof_proba is not None:
            grid = [0.25 + 0.1 * i for i in range(28)]  # 0.25 .. 2.95
            temperature = tune_temperature(y, oof_proba, grid)
            print(f"Calibrated temperature (from OOF): {temperature:.4f}")
            p_cal = _apply_temperature(oof_proba, temperature)
            print("OOF log-loss (raw vs calibrated):", log_loss(y, oof_proba), log_loss(y, p_cal))

    # Final fit
    clf = LogisticRegression(max_iter=2000, random_state=args.seed)

    if args.no_final_fit_all:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        print(classification_report(y_test, clf.predict(X_test), digits=4))
        try:
            print("Holdout ROC-AUC:", roc_auc_score(y_test, y_proba))
        except Exception:
            pass
        if args.calibrate == "temperature" and oof_proba is None:
            grid = [0.25 + 0.1 * i for i in range(28)]
            temperature = tune_temperature(y_test, y_proba, grid)
            print(f"Calibrated temperature (from holdout): {temperature:.4f}")
    elif args.n_folds > 1:
        # OOF already computed temperature when calibrate=temperature
        clf.fit(X, y)
    else:
        # Single train/val split for reporting + optional temperature when no OOF
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        print(classification_report(y_test, clf.predict(X_test), digits=4))
        try:
            print("Holdout ROC-AUC:", roc_auc_score(y_test, y_proba))
        except Exception:
            pass
        if args.calibrate == "temperature":
            grid = [0.25 + 0.1 * i for i in range(28)]
            temperature = tune_temperature(y_test, y_proba, grid)
            print(f"Calibrated temperature (from holdout): {temperature:.4f}")
        clf.fit(X, y)

    coef = clf.coef_[0].tolist()
    intercept = float(clf.intercept_[0])

    artifact = LogisticArtifact(
        feature_names=spec.feature_names(),
        coef=[float(x) for x in coef],
        intercept=intercept,
        temperature=float(temperature),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact.dump(out_path)
    print(f"Saved artifact to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
