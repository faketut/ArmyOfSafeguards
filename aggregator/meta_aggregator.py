"""
Meta-classifier aggregator (learned logistic regression).

This aggregator uses the expert safeguards' outputs as features and predicts
P(unsafe) using a learned logistic model. It preserves the legacy return keys
used by `evaluation/run_benchmark.py`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from aggregator.expert_runner import run_all_safeguards
from meta_classifier.predict import meta_predict_proba, meta_predict_proba_routed
from policy.dynamic_threshold import context_from_dict
from policy.triage import TriageConfig, triage_score
from rules.engine import load_rule_engine_from_env, rules_enabled


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
            return {
                "is_safe": False,
                "flags": [{"safeguard": "rules", "label": m.tag, "confidence": 1.0} for m in hard_blocks],
                "average_confidence": 1.0,
                "individual_results": {
                    "rules": {"label": "block", "confidence": 1.0, "matches": [m.__dict__ for m in rule_matches]}
                },
                "rule_matches": [m.__dict__ for m in rule_matches],
            }

    experts = run_all_safeguards(text)

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
        "meta_probability_unsafe": float(p_unsafe),
        "meta_threshold": float(threshold),
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

