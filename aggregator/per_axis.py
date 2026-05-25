"""
Per-axis verdict helper.

Builds a per-expert `{p_unsafe, verdict}` map from aggregator-style
`individual_results`. This preserves each expert's specialty in the aggregator
output so the fused system can be compared per-axis against each expert
("army of safeguards" initial intent).

Reuses `_unsafe_probability_from_result` and `DEFAULT_EXPERTS` from
`meta_classifier.feature_builder` to avoid duplicating the label→probability
mapping.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from meta_classifier.feature_builder import (
    DEFAULT_EXPERTS,
    _unsafe_probability_from_result,
)


def build_per_axis(
    individual_results: Dict[str, Any],
    threshold: float = 0.5,
    expert_names: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Args:
        individual_results: Mapping `expert_name -> {"label", "confidence"}` or
            `{"error": ...}`, as produced by `run_all_safeguards`.
        threshold: Per-axis decision threshold on P(unsafe). Defaults to 0.5.
        expert_names: Iterable of axis names. Defaults to `DEFAULT_EXPERTS`.

    Returns:
        Dict keyed by expert name with `{"p_unsafe": float, "verdict": str,
        "available": bool}`. Missing or errored experts get
        `p_unsafe=0.5, verdict="safe", available=False`.
    """
    names = tuple(expert_names) if expert_names is not None else DEFAULT_EXPERTS
    out: Dict[str, Dict[str, Any]] = {}
    for name in names:
        res = individual_results.get(name) if isinstance(individual_results, dict) else None
        if not isinstance(res, dict) or "error" in res:
            out[name] = {"p_unsafe": 0.5, "verdict": "safe", "available": False}
            continue
        p = _unsafe_probability_from_result(name, res)
        out[name] = {
            "p_unsafe": float(p),
            "verdict": "unsafe" if p >= threshold else "safe",
            "available": True,
        }
    return out
