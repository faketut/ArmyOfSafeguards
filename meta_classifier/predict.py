from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from meta_classifier.feature_builder import FeatureSpec, build_feature_vector
from meta_classifier.logistic_model import LogisticArtifact, predict_proba


DEFAULT_ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "meta_lr.json"


def get_artifact_path_from_env() -> Path:
    p = os.environ.get("AOS_META_MODEL_PATH", "").strip()
    if p:
        return Path(p)
    return DEFAULT_ARTIFACT_PATH


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

