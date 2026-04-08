from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

from policy.dynamic_threshold import Context, dynamic_threshold

Verdict = Literal["safe", "unsafe", "needs_review"]


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


@dataclass(frozen=True)
class TriageConfig:
    base_threshold: float = 0.5
    gray_low: float = 0.6
    gray_high: float = 0.8

    @staticmethod
    def from_env(default_threshold: float) -> "TriageConfig":
        return TriageConfig(
            base_threshold=float(default_threshold),
            gray_low=_env_float("AOS_GRAY_LOW", 0.6),
            gray_high=_env_float("AOS_GRAY_HIGH", 0.8),
        )


def triage_score(
    score_unsafe: float,
    *,
    config: TriageConfig,
    ctx: Optional[Context] = None,
) -> Tuple[Verdict, Dict[str, Any]]:
    """
    Convert a scalar P(unsafe) into a verdict.

    Conservative mapping:
    - score >= threshold -> unsafe
    - gray_low <= score < gray_high -> needs_review
    - else -> safe

    The threshold may be dynamically adjusted if enabled.
    """
    score = float(score_unsafe)
    threshold_used = dynamic_threshold(config.base_threshold, ctx=ctx)

    gray_low = float(config.gray_low)
    gray_high = float(config.gray_high)
    if gray_low > gray_high:
        gray_low, gray_high = gray_high, gray_low

    if score >= threshold_used:
        verdict: Verdict = "unsafe"
    elif gray_low <= score < gray_high:
        verdict = "needs_review"
    else:
        verdict = "safe"

    payload: Dict[str, Any] = {
        "score_unsafe": score,
        "threshold_used": threshold_used,
        "gray_low": gray_low,
        "gray_high": gray_high,
    }
    if verdict == "needs_review":
        payload["review_payload"] = {
            "reason": "gray_zone",
            "score_unsafe": score,
            "suggestion": "route_to_human_or_llm_second_pass",
        }
    return verdict, payload

