from __future__ import annotations

"""
Train meta-classifier on expert feature vectors using XGBoost or a small MLP.

Artifacts:
  --out-dir/
    manifest.json   # model_type, feature_names, temperature, paths / scaler
    xgb_model.json  # if algo=xgb
    mlp.pt          # if algo=mlp

Inference: set AOS_META_MODEL_PATH to the directory (or to manifest.json in that directory).
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import classification_report, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from meta_classifier.feature_builder import FeatureSpec  # noqa: E402
from meta_classifier.train_meta import (  # noqa: E402
    _apply_temperature,
    _load_jsonl,
    build_dataset,
    tune_temperature,
)

_TEMP_CALIBRATION_GRID = [0.25 + 0.1 * i for i in range(28)]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def _cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    n_folds: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if groups is not None:
        return list(GroupKFold(n_splits=n_folds).split(X, y, groups))
    return list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X, y))


def _xgb_classifier_kwargs(*, seed: int, scale_pos_weight: Optional[float]) -> Dict[str, Any]:
    kw: Dict[str, Any] = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=int(seed),
        n_jobs=-1,
        eval_metric="logloss",
    )
    if scale_pos_weight is not None:
        kw["scale_pos_weight"] = float(scale_pos_weight)
    return kw


def _train_xgb(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    scale_pos_weight: Optional[float],
) -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise SystemExit("Install xgboost: pip install xgboost") from e

    clf = XGBClassifier(**_xgb_classifier_kwargs(seed=seed, scale_pos_weight=scale_pos_weight))
    clf.fit(X, y)
    return clf


def _cross_val_oof_xgb(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    n_folds: int,
    seed: int,
    scale_pos_weight: Optional[float],
) -> Tuple[np.ndarray, List[float]]:
    from xgboost import XGBClassifier

    oof = np.zeros(len(y), dtype=np.float64)
    fold_aucs: List[float] = []

    splits = _cv_splits(X, y, groups, n_folds, seed)
    kwargs = _xgb_classifier_kwargs(seed=seed, scale_pos_weight=scale_pos_weight)

    for train_idx, val_idx in splits:
        clf = XGBClassifier(**kwargs)
        clf.fit(X[train_idx], y[train_idx])
        p = clf.predict_proba(X[val_idx])[:, 1]
        oof[val_idx] = p
        try:
            fold_aucs.append(float(roc_auc_score(y[val_idx], p)))
        except Exception:
            fold_aucs.append(float("nan"))
    return oof, fold_aucs


def _train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    hidden: int,
    epochs: int,
    lr: float,
    device: str,
) -> Tuple[Any, StandardScaler]:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    d_in = Xs.shape[1]
    model = nn.Sequential(
        nn.Linear(d_in, int(hidden)),
        nn.ReLU(),
        nn.Linear(int(hidden), 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(Xs, device=device)
    y_t = y_t.to(device)
    model.to(device)
    model.train()
    for _ in range(int(epochs)):
        opt.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        opt.step()
    model.eval()
    return model, scaler


def _mlp_cross_val_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    n_folds: int,
    seed: int,
    hidden: int,
    epochs: int,
    lr: float,
    device: str,
) -> Tuple[np.ndarray, List[float]]:
    import torch

    oof = np.zeros(len(y), dtype=np.float64)
    fold_aucs: List[float] = []

    splits = _cv_splits(X, y, groups, n_folds, seed)
    for train_idx, val_idx in splits:
        model, scaler = _train_mlp(
            X[train_idx],
            y[train_idx],
            seed=seed,
            hidden=hidden,
            epochs=epochs,
            lr=lr,
            device=device,
        )
        X_va = scaler.transform(X[val_idx]).astype(np.float32)
        model.eval()
        with torch.no_grad():
            xt = torch.tensor(X_va, device=device)
            logits = model(xt).cpu().numpy().ravel()
            p = _sigmoid(logits)
        oof[val_idx] = p
        try:
            fold_aucs.append(float(roc_auc_score(y[val_idx], p)))
        except Exception:
            fold_aucs.append(float("nan"))
    return oof, fold_aucs


def main() -> int:
    parser = argparse.ArgumentParser(description="Train XGBoost or MLP meta-classifier on expert features")
    parser.add_argument("--data", type=str, required=True, help="JSONL with individual_results + label")
    parser.add_argument("--label-field", type=str, default="label")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for manifest + model weights")
    parser.add_argument("--algo", type=str, choices=["xgb", "mlp"], default="xgb")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-folds", type=int, default=0, help="If >1, OOF metrics (same as train_meta.py)")
    parser.add_argument("--group-field", type=str, default="", help="GroupKFold field e.g. source")
    parser.add_argument(
        "--calibrate",
        type=str,
        choices=["none", "temperature"],
        default="none",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlp-hidden", type=int, default=32)
    parser.add_argument("--mlp-epochs", type=int, default=400)
    parser.add_argument("--mlp-lr", type=float, default=1e-2)
    parser.add_argument(
        "--mlp-device",
        type=str,
        default="cpu",
        help="cpu or cuda (for algo=mlp)",
    )
    parser.add_argument(
        "--xgb-scale-pos-weight",
        type=float,
        default=0.0,
        help="If >0, passed to XGBClassifier as scale_pos_weight; 0 = leave default",
    )
    args = parser.parse_args()

    spec = FeatureSpec()
    data_path = Path(args.data)
    rows = _load_jsonl(data_path)
    if not rows:
        raise SystemExit("empty --data JSONL")

    X, y, _, groups = build_dataset(
        rows, spec=spec, label_field=args.label_field, group_field=args.group_field
    )
    if len(X) < 20:
        raise SystemExit(f"not enough rows after featurization: {len(X)}")

    if groups is not None and args.n_folds > 1 and len(np.unique(groups)) < args.n_folds:
        print("warning: not enough unique groups for K-fold; falling back to stratified KFold")
        groups = None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "version": 1,
        "model_type": str(args.algo),
        "feature_names": spec.feature_names(),
        "temperature": 1.0,
    }

    spw = float(args.xgb_scale_pos_weight) if args.algo == "xgb" else None
    if spw is not None and spw <= 0:
        spw = None

    temperature = 1.0
    oof_proba: Optional[np.ndarray] = None

    if args.n_folds and args.n_folds > 1:
        if args.algo == "xgb":
            oof_proba, fold_aucs = _cross_val_oof_xgb(X, y, groups, args.n_folds, args.seed, spw)
            print(f"OOF ROC-AUC (mean of folds): {np.nanmean(fold_aucs):.4f}")
            try:
                print(f"OOF ROC-AUC (micro on all OOF preds): {roc_auc_score(y, oof_proba):.4f}")
            except Exception as e:
                print("OOF ROC-AUC (full):", e)
        else:
            # MLP OOF: lighter — reuse stratified splits; train per fold on CPU
            oof_proba, fold_aucs = _mlp_cross_val_oof(
                X, y, groups, args.n_folds, args.seed, int(args.mlp_hidden), int(args.mlp_epochs), float(args.mlp_lr), args.mlp_device
            )
            print(f"OOF ROC-AUC (mean of folds): {np.nanmean(fold_aucs):.4f}")

        if args.calibrate == "temperature" and oof_proba is not None:
            temperature = tune_temperature(y, oof_proba, _TEMP_CALIBRATION_GRID)
            print(f"Calibrated temperature (from OOF): {temperature:.4f}")
            p_cal = _apply_temperature(oof_proba, temperature)
            print("OOF log-loss (raw vs calibrated):", log_loss(y, oof_proba), log_loss(y, p_cal))

    manifest["temperature"] = float(temperature)

    # Holdout diagnostics when not using K-fold OOF (final model is still fit on all data below).
    if not args.n_folds or args.n_folds <= 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )
        if args.algo == "xgb":
            from xgboost import XGBClassifier

            clf_h = XGBClassifier(**_xgb_classifier_kwargs(seed=args.seed, scale_pos_weight=spw))
            clf_h.fit(X_train, y_train)
            y_proba = clf_h.predict_proba(X_test)[:, 1]
        else:
            import torch

            model_h, sc_h = _train_mlp(
                X_train,
                y_train,
                seed=args.seed,
                hidden=args.mlp_hidden,
                epochs=args.mlp_epochs,
                lr=args.mlp_lr,
                device=args.mlp_device,
            )
            X_test_s = sc_h.transform(X_test).astype(np.float32)
            model_h.eval()
            with torch.no_grad():
                xt = torch.tensor(X_test_s, device=args.mlp_device)
                logits = model_h(xt).cpu().numpy().ravel()
                y_proba = _sigmoid(logits)
        print(classification_report(y_test, (y_proba >= 0.5).astype(int), digits=4))
        try:
            print("Holdout ROC-AUC:", roc_auc_score(y_test, y_proba))
        except Exception:
            pass
        if args.calibrate == "temperature" and oof_proba is None:
            temperature = tune_temperature(y_test, y_proba, _TEMP_CALIBRATION_GRID)
            manifest["temperature"] = float(temperature)
            print(f"Calibrated temperature (from holdout): {temperature:.4f}")

    if args.algo == "xgb":
        clf = _train_xgb(X, y, seed=args.seed, scale_pos_weight=spw)
        model_path = out_dir / "xgb_model.json"
        clf.get_booster().save_model(str(model_path))
        manifest["model_file"] = model_path.name
    else:
        import torch

        model, scaler = _train_mlp(
            X,
            y,
            seed=args.seed,
            hidden=args.mlp_hidden,
            epochs=args.mlp_epochs,
            lr=args.mlp_lr,
            device=args.mlp_device,
        )
        mlp_path = out_dir / "mlp.pt"
        torch.save(model.state_dict(), mlp_path)
        manifest["model_file"] = mlp_path.name
        manifest["mlp_hidden"] = int(args.mlp_hidden)
        manifest["mlp_d_in"] = int(X.shape[1])
        manifest["scaler_mean"] = scaler.mean_.tolist()
        manifest["scaler_scale"] = scaler.scale_.tolist()

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote artifact directory: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
