from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from meta_classifier.feature_builder import FeatureSpec, build_feature_vector
from meta_classifier.logistic_model import LogisticArtifact, predict_proba
from meta_classifier.train_meta import _apply_temperature


DEFAULT_ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "meta_lr.json"
DEFAULT_DOMAIN_TO_ARTIFACT = {
    "toxicity": Path(__file__).parent / "artifacts" / "meta_lr_toxicity.json",
    "sexual": Path(__file__).parent / "artifacts" / "meta_lr_sexual.json",
    "jailbreak": Path(__file__).parent / "artifacts" / "meta_lr_jailbreak.json",
    "mixed": Path(__file__).parent / "artifacts" / "meta_lr_mixed.json",
}


def get_artifact_path_from_env() -> Path:
    p = os.environ.get("AOS_META_MODEL_PATH", "").strip()
    if p:
        return Path(p)
    return DEFAULT_ARTIFACT_PATH


def _get_domain_artifact_from_env(domain: str) -> Optional[Path]:
    """
    Domain routing:
    - If AOS_META_MODEL_MAP_JSON is set: JSON dict of {domain: path}
    - Else use AOS_META_MODEL_PATH_<DOMAIN> overrides (e.g. AOS_META_MODEL_PATH_TOXICITY)
    - Else fall back to DEFAULT_DOMAIN_TO_ARTIFACT if the file exists
    - Else None
    """
    d = (domain or "").strip().lower()
    if not d:
        return None

    map_json = os.environ.get("AOS_META_MODEL_MAP_JSON", "").strip()
    if map_json:
        try:
            m = json.loads(map_json)
            if isinstance(m, dict) and d in m and isinstance(m[d], str) and m[d].strip():
                return Path(m[d].strip())
        except Exception:
            pass

    key = f"AOS_META_MODEL_PATH_{d.upper()}"
    p = os.environ.get(key, "").strip()
    if p:
        return Path(p)

    default = DEFAULT_DOMAIN_TO_ARTIFACT.get(d)
    if default and default.exists():
        return default
    return None


def _resolve_manifest_path(path: Path) -> Path:
    path = path.resolve()
    if path.is_dir():
        return path / "manifest.json"
    return path


def _tabular_proba_raw(
    payload: Dict[str, Any],
    base_dir: Path,
    x: np.ndarray,
) -> float:
    """x shape (1, n_features)."""
    mt = str(payload.get("model_type", "")).lower()
    model_file = base_dir / str(payload["model_file"])

    if mt == "xgb":
        try:
            from xgboost import XGBClassifier

            clf = XGBClassifier()
            clf.load_model(str(model_file))
            return float(clf.predict_proba(x)[0, 1])
        except Exception:
            import xgboost as xgb

            booster = xgb.Booster()
            booster.load_model(str(model_file))
            dmat = xgb.DMatrix(x)
            return float(booster.predict(dmat)[0])

    if mt == "mlp":
        import torch
        import torch.nn as nn

        d_in = int(payload["mlp_d_in"])
        h = int(payload["mlp_hidden"])
        model = nn.Sequential(
            nn.Linear(d_in, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )
        try:
            state = torch.load(model_file, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(model_file, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        mean = np.asarray(payload["scaler_mean"], dtype=np.float64)
        scale = np.asarray(payload["scaler_scale"], dtype=np.float64)
        xs = (x - mean) / scale
        with torch.no_grad():
            logit = model(torch.tensor(xs, dtype=torch.float32)).numpy().ravel()[0]
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -50.0, 50.0))))

    raise ValueError(f"Unsupported tabular model_type: {mt!r}")


def meta_predict_proba(
    individual_results: Dict[str, Any],
    *,
    artifact_path: Optional[str | Path] = None,
    spec: Optional[FeatureSpec] = None,
) -> float:
    """
    Predict P(unsafe) from expert outputs using a learned meta model:
    legacy logistic JSON (coef), or a directory / manifest with XGBoost or MLP (see train_meta_tabular.py).
    """
    path = Path(artifact_path) if artifact_path is not None else get_artifact_path_from_env()
    manifest_path = _resolve_manifest_path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Meta artifact not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if spec is None:
        spec = FeatureSpec()
    feats = build_feature_vector(individual_results, spec=spec)
    x = np.asarray([feats], dtype=np.float32)

    mt = str(payload.get("model_type", "")).lower()
    if "coef" in payload and mt not in ("xgb", "mlp"):
        artifact = LogisticArtifact.load(manifest_path)
        p = float(predict_proba(artifact, feats))
        return p

    base_dir = manifest_path.parent
    t = float(payload.get("temperature", 1.0))
    p_raw = _tabular_proba_raw(payload, base_dir, x)
    return float(_apply_temperature(np.array([p_raw]), t)[0])


def meta_predict_proba_routed(
    individual_results: Dict[str, Any],
    *,
    domain: str,
    fallback_artifact_path: Optional[str | Path] = None,
    spec: Optional[FeatureSpec] = None,
) -> float:
    """
    Predict P(unsafe) with domain-aware routing when available.

    If no domain-specific artifact is configured/found, falls back to:
      - fallback_artifact_path if provided
      - else AOS_META_MODEL_PATH / DEFAULT_ARTIFACT_PATH
    """
    dpath = _get_domain_artifact_from_env(domain)
    if dpath is not None:
        return meta_predict_proba(individual_results, artifact_path=dpath, spec=spec)
    if fallback_artifact_path is not None:
        return meta_predict_proba(individual_results, artifact_path=fallback_artifact_path, spec=spec)
    return meta_predict_proba(individual_results, spec=spec)
