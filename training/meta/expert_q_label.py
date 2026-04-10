"""Binary safe/unsafe label from the same Q features as meta_classifier (no external teacher)."""
from __future__ import annotations

from typing import Any, Dict, Tuple

from meta_classifier.feature_builder import FeatureSpec, build_feature_vector


def label_from_expert_q(individual_results: Dict[str, Any], *, threshold: float) -> Tuple[str, str, float]:
    """
    Returns (label, verdict, max_q) where label/verdict are 'safe'|'unsafe'.
    Uses max of the four expert P(unsafe) features (no rules tag).
    """
    spec = FeatureSpec(include_bias_features=False)
    x = build_feature_vector(individual_results, spec=spec)
    if len(x) < 4:
        m = max(x) if x else 0.0
    else:
        m = max(float(x[i]) for i in range(4))
    if m >= float(threshold):
        return "unsafe", "unsafe", float(m)
    return "safe", "safe", float(m)
