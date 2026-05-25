"""
Meta-classifier aggregator (learned logistic regression).

This aggregator uses the expert safeguards' outputs as features and predicts
P(unsafe) using a learned logistic model. It preserves the legacy return keys
used by `evaluation/run_benchmark.py`.

Per-axis routing (Phase 2 D): if per-axis meta heads are present on disk
(meta_classifier/artifacts/meta_lr_axis_<axis>.json), they are used to produce
the `per_axis` field and the fused `meta_probability_unsafe` is the max
across axes (OR semantics). Set `AOS_META_DISABLE_PER_AXIS=1` to fall back
to the legacy single global LR path.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from aggregator.expert_runner import run_all_safeguards
from aggregator.per_axis import build_per_axis
from meta_classifier.feature_builder import DEFAULT_EXPERTS
from meta_classifier.predict import (
    meta_predict_per_axis,
    meta_predict_proba,
    meta_predict_proba_routed,
)
from policy.dynamic_threshold import context_from_dict
from policy.triage import TriageConfig, triage_score
from rules.engine import load_rule_engine_from_env, rules_enabled


def _per_axis_enabled() -> bool:
    return os.environ.get("AOS_META_DISABLE_PER_AXIS", "").strip() not in {"1", "true", "TRUE", "yes", "YES"}


def _per_axis_from_meta(
    experts: Dict[str, Any], threshold: float
) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Build `per_axis` using per-axis meta heads when available.

    For axes with a meta head: p_unsafe comes from the head.
    For axes without a head: fall back to the raw expert mapping (same as
    `build_per_axis`).
    Returns None when no head artifacts exist (caller falls back to legacy).
    """
    head_probs = meta_predict_per_axis(experts)
    if not head_probs:
        return None

    fallback = build_per_axis(experts, threshold=threshold)
    out: Dict[str, Dict[str, Any]] = {}
    for axis in DEFAULT_EXPERTS:
        if axis in head_probs:
            p = float(head_probs[axis])
            out[axis] = {
                "p_unsafe": p,
                "verdict": "unsafe" if p >= threshold else "safe",
                "available": True,
                "source": "meta_head",
            }
        else:
            base = fallback.get(axis, {"p_unsafe": 0.5, "verdict": "safe", "available": False})
            out[axis] = {**base, "source": "expert_raw"}
    return out


def evaluate_text(
    text: str,
    threshold: float = 0.5,
    meta_model_path: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    domain: str = "",
) -> Dict[str, Any]:
    rule_matches = []
    if rules_enabled():
        engine = load_rule_engine_from_env()
        rule_matches = engine.match(text)
        hard_blocks = [m for m in rule_matches if m.action == "block"]
        if hard_blocks:
            ir = {
                "rules": {"label": "block", "confidence": 1.0, "matches": [m.__dict__ for m in rule_matches]}
            }
            return {
                "is_safe": False,
                "flags": [{"safeguard": "rules", "label": m.tag, "confidence": 1.0} for m in hard_blocks],
                "average_confidence": 1.0,
                "individual_results": ir,
                "per_axis": build_per_axis(ir, threshold=threshold),
                "rule_matches": [m.__dict__ for m in rule_matches],
            }

    experts = run_all_safeguards(text)

    per_axis_meta: Optional[Dict[str, Dict[str, Any]]] = None
    meta_source = "global_lr"
    p_unsafe: float
    if _per_axis_enabled():
        per_axis_meta = _per_axis_from_meta(experts, threshold=threshold)

    if per_axis_meta is not None:
        # OR aggregation across axes: fused unsafe iff any axis exceeds threshold.
        p_unsafe = max(info["p_unsafe"] for info in per_axis_meta.values())
        meta_source = "per_axis_heads"
    else:
        try:
            d = domain or (context or {}).get("domain", "") if isinstance(context, dict) else domain
            if isinstance(d, str) and d.strip():
                p_unsafe = meta_predict_proba_routed(
                    experts,
                    domain=d,
                    fallback_artifact_path=meta_model_path,
                )
            else:
                p_unsafe = meta_predict_proba(experts, artifact_path=meta_model_path)
        except Exception as e:
            # Fallback behavior: if meta model isn't available, remain conservative.
            p_unsafe = 0.5
            experts.setdefault("meta", {})["error"] = str(e)

    cfg = TriageConfig.from_env(default_threshold=threshold)
    ctx = context_from_dict(context)
    verdict, triage = triage_score(p_unsafe, config=cfg, ctx=ctx)

    # Conservative: treat needs_review as not safe for binary evaluation.
    is_safe = verdict == "safe"
    flags = []
    if verdict != "safe":
        flags.append({"safeguard": "meta", "label": verdict, "confidence": float(p_unsafe)})

    avg_conf = float(p_unsafe)  # legacy key
    out: Dict[str, Any] = {
        "is_safe": is_safe,
        "flags": flags,
        "average_confidence": avg_conf,
        "individual_results": experts,
        "per_axis": per_axis_meta if per_axis_meta is not None else build_per_axis(experts, threshold=threshold),
        "meta_probability_unsafe": float(p_unsafe),
        "meta_threshold": float(threshold),
        "meta_source": meta_source,
        "verdict": verdict,
        "triage": triage,
    }

    if rule_matches:
        out["rule_matches"] = [m.__dict__ for m in rule_matches]
        out.setdefault("individual_results", {})["rules"] = {
            "label": "tag",
            "confidence": 1.0,
            "matches": [m.__dict__ for m in rule_matches],
        }

    return out

