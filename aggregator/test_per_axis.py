"""Unit tests for `aggregator.per_axis.build_per_axis`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aggregator.per_axis import build_per_axis
from meta_classifier.feature_builder import DEFAULT_EXPERTS


def _safe_result(name: str) -> dict:
    # P(unsafe) → 0.0 for each expert (well below 0.5 threshold).
    if name == "jailbreak":
        return {"label": False, "confidence": 1.0}
    return {"label": "LABEL_0", "confidence": 1.0}


def _unsafe_result(name: str) -> dict:
    # P(unsafe) → 1.0 for each expert.
    if name == "jailbreak":
        return {"label": True, "confidence": 1.0}
    if name == "toxicity":
        return {"label": "unsafe", "confidence": 1.0}
    return {"label": "LABEL_1", "confidence": 1.0}


def test_each_axis_isolated_unsafe():
    for hot in DEFAULT_EXPERTS:
        ir = {name: (_unsafe_result(name) if name == hot else _safe_result(name))
              for name in DEFAULT_EXPERTS}
        per_axis = build_per_axis(ir, threshold=0.5)

        assert per_axis[hot]["verdict"] == "unsafe", (hot, per_axis)
        assert per_axis[hot]["p_unsafe"] == 1.0
        assert per_axis[hot]["available"] is True

        for other in DEFAULT_EXPERTS:
            if other == hot:
                continue
            assert per_axis[other]["verdict"] == "safe", (hot, other, per_axis)
            assert per_axis[other]["p_unsafe"] == 0.0
            assert per_axis[other]["available"] is True


def test_missing_or_errored_expert_defaults_to_safe():
    ir = {
        "jailbreak": {"error": "boom"},
        # toxicity, sexual, factuality entirely missing
    }
    per_axis = build_per_axis(ir, threshold=0.5)

    for name in DEFAULT_EXPERTS:
        assert per_axis[name]["available"] is False, (name, per_axis)
        assert per_axis[name]["verdict"] == "safe"
        assert per_axis[name]["p_unsafe"] == 0.5


def test_threshold_is_respected():
    # Confidence 0.6 → p_unsafe=0.6.
    ir = {name: {"label": "LABEL_1", "confidence": 0.6} for name in DEFAULT_EXPERTS}
    ir["jailbreak"] = {"label": True, "confidence": 0.6}

    low = build_per_axis(ir, threshold=0.5)
    high = build_per_axis(ir, threshold=0.7)

    for name in DEFAULT_EXPERTS:
        assert low[name]["verdict"] == "unsafe"
        assert high[name]["verdict"] == "safe"


if __name__ == "__main__":
    test_each_axis_isolated_unsafe()
    test_missing_or_errored_expert_defaults_to_safe()
    test_threshold_is_respected()
    print("ok")
