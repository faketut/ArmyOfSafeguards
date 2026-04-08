from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Context:
    """
    Optional context used for dynamic thresholding.

    All fields are optional and default to None; callers can gradually adopt them.
    """

    user_tier: Optional[str] = None  # e.g. "anonymous", "new", "trusted"
    send_rate_per_min: Optional[float] = None
    prior_violations: Optional[int] = None


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


def dynamic_threshold(base_threshold: float, ctx: Optional[Context] = None) -> float:
    """
    Compute an adjusted threshold given context.

    Defaults to base_threshold. You can tune via env vars:
    - AOS_DYNAMIC_THRESHOLD_ENABLE: 1/true to enable adjustments
    - AOS_TIER_DELTA_ANON, AOS_TIER_DELTA_NEW, AOS_TIER_DELTA_TRUSTED: deltas
    - AOS_RATE_DELTA: delta added when send_rate_per_min exceeds AOS_RATE_LIMIT
    - AOS_RATE_LIMIT: default 30
    - AOS_VIOLATION_DELTA: delta per prior violation
    """
    enable = os.environ.get("AOS_DYNAMIC_THRESHOLD_ENABLE", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
    if not enable:
        return float(base_threshold)

    ctx = ctx or Context()
    t = float(base_threshold)

    tier = (ctx.user_tier or "").strip().lower()
    if tier in {"anon", "anonymous"}:
        t += _env_float("AOS_TIER_DELTA_ANON", 0.05)
    elif tier in {"new"}:
        t += _env_float("AOS_TIER_DELTA_NEW", 0.03)
    elif tier in {"trusted"}:
        t += _env_float("AOS_TIER_DELTA_TRUSTED", -0.02)

    # rate based
    rate_limit = _env_float("AOS_RATE_LIMIT", 30.0)
    if ctx.send_rate_per_min is not None and ctx.send_rate_per_min > rate_limit:
        t += _env_float("AOS_RATE_DELTA", 0.03)

    # prior violations
    if ctx.prior_violations is not None and ctx.prior_violations > 0:
        t += _env_float("AOS_VIOLATION_DELTA", 0.01) * float(ctx.prior_violations)

    # clamp to sensible range
    if t < 0.01:
        t = 0.01
    if t > 0.99:
        t = 0.99
    return t


def context_from_dict(d: Optional[Dict[str, Any]]) -> Context:
    if not d:
        return Context()
    return Context(
        user_tier=d.get("user_tier"),
        send_rate_per_min=d.get("send_rate_per_min"),
        prior_violations=d.get("prior_violations"),
    )

