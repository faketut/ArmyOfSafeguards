from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_EXPERTS: Tuple[str, ...] = ("jailbreak", "toxicity", "sexual", "factuality")


def _unsafe_probability_from_result(safeguard_name: str, result: Dict[str, Any]) -> float:
    """
    Convert legacy safeguard result to P(unsafe) in [0, 1].

    This mirrors logic in `aggregator/weighted_aggregator.py` but is kept
    standalone to avoid cyclic imports.
    """
    confidence = float(result.get("confidence", 0.0) or 0.0)
    label = result.get("label", "")

    confidence = min(max(confidence, 0.0), 1.0)

    if safeguard_name == "factuality":
        if label == "LABEL_1":
            return confidence
        if label == "LABEL_0":
            return 1.0 - confidence
        return 0.5

    if safeguard_name == "sexual":
        if label == "LABEL_1":
            return confidence
        if label == "LABEL_0":
            return 1.0 - confidence
        return 0.5

    if safeguard_name == "jailbreak":
        if label is True:
            return confidence
        if label is False:
            return 1.0 - confidence
        return 0.5

    if safeguard_name == "toxicity":
        if label in ("LABEL_1", "unsafe"):
            return confidence
        if label in ("LABEL_0", "safe"):
            return 1.0 - confidence
        return 0.5

    if label in ("LABEL_0", "safe", False):
        return 1.0 - confidence
    if label in ("LABEL_1", "unsafe", True):
        return confidence
    return confidence


@dataclass(frozen=True)
class FeatureSpec:
    expert_names: Tuple[str, ...] = DEFAULT_EXPERTS
    include_bias_features: bool = True

    def feature_names(self) -> List[str]:
        feats = [f"p_unsafe_{name}" for name in self.expert_names]
        if self.include_bias_features:
            feats.append("has_rules_tag")
        return feats


def build_feature_vector(
    individual_results: Dict[str, Any],
    spec: FeatureSpec | None = None,
) -> List[float]:
    """
    Build a fixed-length feature vector from aggregator-style `individual_results`.
    """
    if spec is None:
        spec = FeatureSpec()

    x: List[float] = []

    for name in spec.expert_names:
        res = individual_results.get(name, {})
        if not isinstance(res, dict) or "error" in res:
            x.append(0.5)
        else:
            x.append(_unsafe_probability_from_result(name, res))

    if spec.include_bias_features:
        rules_res = individual_results.get("rules", {})
        has_rules = bool(isinstance(rules_res, dict) and rules_res.get("matches"))
        x.append(1.0 if has_rules else 0.0)

    return x

