from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from meta_classifier.feature_builder import FeatureSpec, build_feature_vector
from meta_classifier.logistic_model import LogisticArtifact, predict_proba


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
        import json

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


def meta_predict_proba(
    individual_results: Dict[str, Any],
    *,
    artifact_path: Optional[str | Path] = None,
    spec: Optional[FeatureSpec] = None,
) -> float:
    """
    Predict P(unsafe) from expert outputs using a learned logistic model.
    """
    path = Path(artifact_path) if artifact_path is not None else get_artifact_path_from_env()
    artifact = LogisticArtifact.load(path)

    if spec is None:
        spec = FeatureSpec()
    x = build_feature_vector(individual_results, spec=spec)
    return float(predict_proba(artifact, x))


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

