"""
Weighted Aggregator for Army of Safeguards.

This module imports all individual safeguard critics and provides
a unified interface for running multiple safeguards on input text.
Uses weighted sum aggregation where each safeguard's probability
of "unsafe" is weighted by its importance and performance.

Weighted aggregation formula:
    score = Σ(wi * probability_i)
    
Where:
    - wi: weight for safeguard i
    - probability_i: probability that safeguard i flags content as unsafe (0.0-1.0)
"""
from typing import Dict, List, Any, Optional

from aggregator.expert_runner import run_all_safeguards
from aggregator.per_axis import build_per_axis
from policy.dynamic_threshold import context_from_dict
from policy.triage import TriageConfig, triage_score
from rules.engine import load_rule_engine_from_env, rules_enabled


# Default weights based on task importance and heuristic priorities
# Higher weight = more important in final decision
# Weights can be:
#   - Learned from dataset performance (accuracy/F1)
#   - Set heuristically by task importance
#   - Adjusted based on confidence calibration
DEFAULT_WEIGHTS = {
    'jailbreak': 0.35,    # Highest priority - jailbreaks are critical security issues
    'toxicity': 0.25,     # High priority - toxic content causes harm
    'sexual': 0.20,       # Medium-high priority - inappropriate content
    'factuality': 0.20,   # Medium priority - misinformation is important but less immediate
}

# Normalize weights to sum to 1.0 (optional, for interpretability)
def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weights so they sum to 1.0."""
    total = sum(weights.values())
    if total > 0:
        return {k: v / total for k, v in weights.items()}
    return weights


def _get_unsafe_probability(safeguard_name: str, result: Dict[str, Any]) -> float:
    """
    Convert safeguard result to probability of content being unsafe (0.0-1.0).
    
    Args:
        safeguard_name: Name of the safeguard
        result: Result dictionary with 'label' and 'confidence'
        
    Returns:
        Probability that content is unsafe (0.0 = definitely safe, 1.0 = definitely unsafe)
    """
    confidence = result.get('confidence', 0.0)
    label = result.get('label', '')
    
    # Convert label + confidence to probability of "unsafe"
    if safeguard_name == 'factuality':
        # LABEL_0 = factual (safe), LABEL_1 = non-factual (unsafe)
        if label == 'LABEL_1':
            return confidence  # Confidence in non-factual = probability of unsafe
        elif label == 'LABEL_0':
            return 1.0 - confidence  # Confidence in factual = 1 - probability of unsafe
        else:
            return 0.5  # Unknown label
    
    elif safeguard_name == 'sexual':
        # LABEL_0 = safe, LABEL_1 = sexual/inappropriate (unsafe)
        if label == 'LABEL_1':
            return confidence
        elif label == 'LABEL_0':
            return 1.0 - confidence
        else:
            return 0.5
    
    elif safeguard_name == 'jailbreak':
        # Boolean: False = safe, True = jailbreak (unsafe)
        if label is True:
            return confidence  # Confidence in jailbreak = probability of unsafe
        elif label is False:
            return 1.0 - confidence  # Confidence in not jailbreak = 1 - probability of unsafe
        else:
            return 0.5
    
    elif safeguard_name == 'toxicity':
        # LABEL_0 = safe, LABEL_1 or "unsafe" = toxic (unsafe)
        if label == 'LABEL_1' or label == 'unsafe':
            return confidence
        elif label == 'LABEL_0' or label == 'safe':
            return 1.0 - confidence
        else:
            return 0.5
    
    else:
        # Default: assume higher label numbers or "unsafe" strings = problematic
        if label in ['LABEL_0', 'safe', False]:
            return 1.0 - confidence
        elif label in ['LABEL_1', 'unsafe', True]:
            return confidence
        else:
            # Unknown format - use confidence directly if it seems like a probability
            return min(max(confidence, 0.0), 1.0)


def aggregate_results(
    results: Dict[str, Dict[str, Any]], 
    weights: Optional[Dict[str, float]] = None,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Aggregate results from multiple safeguards using weighted sum.

    Formula: score = Σ(wi * probability_i)
    where wi is the weight for safeguard i and probability_i is the
    probability that safeguard i flags content as unsafe.

    Args:
        results: Dictionary of results from each safeguard
        weights: Dictionary of weights for each safeguard. If None, uses DEFAULT_WEIGHTS.
                 Weights can be based on:
                 - Task importance (e.g., jailbreak > toxicity)
                 - Dataset performance (accuracy/F1 scores)
                 - Confidence calibration
        threshold: Optional threshold for flagging. If None, uses weighted score > 0.5.
                   With threshold=None, the decision is purely based on weighted majority.

    Returns:
        Dictionary containing:
        - weighted_score: Final weighted aggregation score (0.0-1.0)
        - is_safe: Boolean indicating if content is considered safe
        - flags: List of individual safeguard flags (if any)
        - weighted_contributions: Individual weighted contributions
        - individual_results: Original results from each safeguard
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    
    # Use default threshold of 0.5 if not specified
    if threshold is None:
        threshold = 0.5
    
    flags = []
    weighted_contributions = {}
    probabilities = {}
    
    weighted_sum = 0.0
    total_weight = 0.0
    weighted_confidence_sum = 0.0  # For weighted average confidence
    
    # First pass: collect all safeguards that have results and weights
    active_safeguards = []
    for safeguard_name, result in results.items():
        if 'error' in result:
            continue
        if safeguard_name in weights:
            active_safeguards.append((safeguard_name, result))
    
    # Normalize weights only for active safeguards (those with results)
    if active_safeguards:
        active_weights = {name: weights[name] for name, _ in active_safeguards}
        active_weights = _normalize_weights(active_weights)
    else:
        active_weights = {}
    
    # Second pass: calculate weighted contributions
    for safeguard_name, result in active_safeguards:
        # Get probability of unsafe
        prob_unsafe = _get_unsafe_probability(safeguard_name, result)
        probabilities[safeguard_name] = prob_unsafe
        
        # Get normalized weight for this safeguard
        weight = active_weights.get(safeguard_name, 0.0)
        
        # Calculate weighted contribution for probability
        contribution = weight * prob_unsafe
        weighted_contributions[safeguard_name] = {
            'weight': weight,
            'probability_unsafe': prob_unsafe,
            'weighted_contribution': contribution
        }
        
        weighted_sum += contribution
        total_weight += weight
        
        # Calculate weighted confidence contribution
        confidence = result.get('confidence', 0.0)
        weighted_confidence_sum += weight * confidence
        
        # Track individual flags (safeguards that individually flag as unsafe)
        label = result.get('label', '')
        
        # Individual flag logic (for reporting purposes)
        is_problematic = prob_unsafe >= threshold
        
        if is_problematic:
            flags.append({
                'safeguard': safeguard_name,
                'label': label,
                'confidence': confidence,
                'probability_unsafe': prob_unsafe
            })
    
    # Calculate final weighted score
    # Since weights are normalized to sum to 1.0, total_weight should be ~1.0
    # Dividing by total_weight provides robustness and ensures score stays in [0, 1]
    if total_weight > 0:
        weighted_score = weighted_sum / total_weight
        weighted_average_confidence = weighted_confidence_sum / total_weight
    else:
        weighted_score = 0.0
        weighted_average_confidence = 0.0
    
    # Store probabilities for all safeguards (even those not in weights) for completeness
    for safeguard_name, result in results.items():
        if 'error' not in result and safeguard_name not in probabilities:
            probabilities[safeguard_name] = _get_unsafe_probability(safeguard_name, result)
    
    # Store normalized weights used in the calculation
    weights_used = active_weights
    
    # Final safety decision
    is_safe = weighted_score < threshold
    
    return {
        'weighted_score': weighted_score,
        'is_safe': is_safe,
        'threshold_used': threshold,
        'average_confidence': weighted_average_confidence,
        'flags': flags,
        'weighted_contributions': weighted_contributions,
        'probabilities': probabilities,
        'weights': weights_used,  # Return the normalized weights actually used
        'individual_results': results,
        'per_axis': build_per_axis(results, threshold=threshold),
    }


def evaluate_text(
    text: str, 
    weights: Optional[Dict[str, float]] = None,
    threshold: Optional[float] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run all safeguards and aggregate results using weighted sum.

    Args:
        text: Input text to evaluate
        weights: Dictionary of weights for each safeguard. If None, uses DEFAULT_WEIGHTS.
        threshold: Optional threshold for flagging (default: 0.5). If None, uses weighted score > 0.5.

    Returns:
        Aggregated safety assessment with weighted score
    """
    rule_matches = []
    if rules_enabled():
        engine = load_rule_engine_from_env()
        rule_matches = engine.match(text)
        hard_blocks = [m for m in rule_matches if m.action == "block"]
        if hard_blocks:
            ir = {
                "rules": {
                    "label": "block",
                    "confidence": 1.0,
                    "matches": [m.__dict__ for m in rule_matches],
                }
            }
            return {
                "is_safe": False,
                "flags": [
                    {"safeguard": "rules", "label": m.tag, "confidence": 1.0}
                    for m in hard_blocks
                ],
                "average_confidence": 1.0,
                "individual_results": ir,
                "per_axis": build_per_axis(ir, threshold=threshold if threshold is not None else 0.5),
                "rule_matches": [m.__dict__ for m in rule_matches],
            }

    results = run_all_safeguards(text)
    aggregated = aggregate_results(results, weights=weights, threshold=threshold)
    if rule_matches:
        aggregated["rule_matches"] = [m.__dict__ for m in rule_matches]
        aggregated.setdefault("individual_results", {})["rules"] = {
            "label": "tag",
            "confidence": 1.0,
            "matches": [m.__dict__ for m in rule_matches],
        }

    # Gray-zone triage (conservative: needs_review counts as not safe).
    # Use the *effective* threshold from aggregation for compatibility.
    score = float(aggregated.get("weighted_score", 0.0))
    effective_threshold = float(aggregated.get("threshold_used", 0.5))
    cfg = TriageConfig.from_env(default_threshold=effective_threshold)
    verdict, triage = triage_score(score, config=cfg, ctx=context_from_dict(context))
    aggregated["verdict"] = verdict
    aggregated["triage"] = triage
    aggregated["is_safe"] = verdict == "safe"
    if verdict != "safe":
        aggregated.setdefault("flags", []).append(
            {"safeguard": "triage", "label": verdict, "confidence": score}
        )

    return aggregated


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        test_text = " ".join(sys.argv[1:])
    else:
        test_text = input("Enter text to evaluate: ")

    print("\nRunning all safeguards...")
    print("=" * 60)

    result = evaluate_text(test_text)

    print(
        f"\nOverall Safety: {'✅ SAFE' if result['is_safe'] else '⚠️  FLAGGED'}")
    print(f"Weighted Average Confidence: {result.get('weighted_average_confidence', 0.0):.2%}")
    
    # Show weighted score and threshold if available
    if 'weighted_score' in result:
        threshold = result.get('threshold_used', 0.5)
        print(f"Weighted Score: {result['weighted_score']:.4f} (threshold: {threshold:.2f})")
    
    # Show weighted contributions if available
    if 'weighted_contributions' in result:
        print(f"\nWeighted Contributions:")
        for safeguard, contrib in result['weighted_contributions'].items():
            print(
                f"  {safeguard}: "
                f"P(unsafe)={contrib['probability_unsafe']:.3f}, "
                f"weight={contrib['weight']:.3f}, "
                f"contribution={contrib['weighted_contribution']:.4f}"
            )

    if result['flags']:
        print(f"\nIndividual Flags ({len(result['flags'])}):")
        for flag in result['flags']:
            print(
                f"  - {flag['safeguard']}: {flag['label']} "
                f"(confidence: {flag['confidence']:.2%})")

    print("\nIndividual Results:")
    for safeguard, res in result['individual_results'].items():
        if 'error' not in res:
            prob_info = ""
            if 'probabilities' in result and safeguard in result['probabilities']:
                prob_info = f", P(unsafe): {result['probabilities'][safeguard]:.3f}"
            print(
                f"  {safeguard}: {res.get('label')} "
                f"(confidence: {res.get('confidence', 0):.2%}{prob_info})")
        else:
            print(f"  {safeguard}: {res['error']}")

    print("=" * 60)
