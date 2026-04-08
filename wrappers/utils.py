from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from policy.dynamic_threshold import context_from_dict
from policy.triage import TriageConfig, triage_score


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def get_effective_device(device_param: Optional[str]) -> str:
    """
    Wrapper device selection:
    - if device_param provided: use it
    - else: map AOS_DEVICE (auto/cpu/cuda) to wrapper device string
    """
    if device_param:
        return device_param

    aos = os.environ.get("AOS_DEVICE", "auto").strip().lower()
    if aos in {"cpu", "cuda"}:
        return aos
    return "cuda"  # preserve old wrapper default (auto -> prefer cuda)


def error_default_verdict() -> str:
    return os.environ.get("AOS_WRAPPER_ERROR_VERDICT", "needs_review").strip().lower()


def safe_error_result(
    *,
    wrapper_name: str,
    reason: str,
    threshold: float,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Safer default on errors: needs_review (or configured verdict).
    """
    verdict = error_default_verdict()
    if verdict not in {"safe", "unsafe", "needs_review"}:
        verdict = "needs_review"

    # Use a neutral score so triage can still be computed deterministically.
    score_unsafe = 0.5
    triage_cfg = TriageConfig.from_env(default_threshold=threshold)
    v, triage = triage_score(score_unsafe, config=triage_cfg, ctx=context_from_dict(context))

    # If user forced a verdict via env, honor it.
    final_verdict = verdict
    if final_verdict != v:
        triage["forced_verdict"] = final_verdict

    is_safe = final_verdict == "safe"

    return {
        "is_safe": is_safe,
        "average_confidence": float(score_unsafe),
        "flags": [
            {
                "safeguard": wrapper_name,
                "label": "error",
                "confidence": float(score_unsafe),
                "reason": reason,
            }
        ]
        if not is_safe
        else [],
        "individual_results": {
            wrapper_name: {
                "error": reason,
                "score_unsafe": float(score_unsafe),
            }
        },
        "verdict": final_verdict,
        "triage": triage,
    }


def apply_triage(
    *,
    wrapper_name: str,
    score_unsafe: float,
    threshold: float,
    context: Optional[Dict[str, Any]] = None,
    individual_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    triage_cfg = TriageConfig.from_env(default_threshold=threshold)
    verdict, triage = triage_score(score_unsafe, config=triage_cfg, ctx=context_from_dict(context))
    is_safe = verdict == "safe"

    flags = []
    if verdict != "safe":
        flags.append(
            {
                "safeguard": wrapper_name,
                "label": verdict,
                "confidence": float(score_unsafe),
            }
        )

    return {
        "is_safe": is_safe,
        "average_confidence": float(score_unsafe),
        "flags": flags,
        "individual_results": {wrapper_name: individual_payload or {"score_unsafe": float(score_unsafe)}},
        "verdict": verdict,
        "triage": triage,
    }

