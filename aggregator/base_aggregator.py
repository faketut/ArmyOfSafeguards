"""
Aggregator for Army of Safeguards.

This module imports all individual safeguard critics and provides
a unified interface for running multiple safeguards on input text.
"""
from typing import Dict, List, Any

from aggregator.expert_runner import run_all_safeguards
from aggregator.per_axis import build_per_axis
from rules.engine import load_rule_engine_from_env, rules_enabled


def aggregate_results(results: Dict[str, Dict[str, Any]], threshold: float = 0.7) -> Dict[str, Any]:
    """
    Aggregate results from multiple safeguards.

    Args:
        results: Dictionary of results from each safeguard
        threshold: Confidence threshold for flagging (default: 0.7)

    Returns:
        Aggregated decision with overall safety assessment
    """
    flags = []
    confidences = []

    for safeguard_name, result in results.items():
        if 'error' in result:
            continue

        confidence = result.get('confidence', 0.0)
        label = result.get('label', '')

        # Check if this safeguard flagged the content
        # For factuality: LABEL_1 = non-factual (unsafe), LABEL_0 = factual (safe)
        # For sexual: LABEL_1 = sensitive/sexual content (unsafe), LABEL_0 = safe
        # For jailbreak: label is boolean (True = jailbreak/unsafe, False = safe)
        is_problematic = False

        if safeguard_name == 'factuality':
            # LABEL_1 means non-factual/misinformation
            is_problematic = (label == 'LABEL_1' and confidence >= threshold)
        elif safeguard_name == 'sexual':
            # LABEL_1 means sensitive/sexual content
            is_problematic = (label == 'LABEL_1' and confidence >= threshold)
        elif safeguard_name == 'jailbreak':
            # Boolean label: True = jailbreak (unsafe)
            is_problematic = (label is True and confidence >= threshold)
        elif safeguard_name == 'toxicity':
            # "unsafe" or LABEL_1 means toxic/racist/hateful
            is_problematic = ((label == 'unsafe' or label ==
                              'LABEL_1') and confidence >= threshold)
        else:
            # For other safeguards, assume higher label numbers = problematic
            # Adjust this logic when adding new safeguards
            is_problematic = (confidence >= threshold and label not in [
                              'LABEL_0', 'safe'])

        if is_problematic:
            flags.append({
                'safeguard': safeguard_name,
                'label': label,
                'confidence': confidence
            })

        confidences.append(confidence)

    # Calculate overall safety score
    avg_confidence = sum(confidences) / \
        len(confidences) if confidences else 0.0
    is_safe = len(flags) == 0

    return {
        'is_safe': is_safe,
        'flags': flags,
        'average_confidence': avg_confidence,
        'individual_results': results,
        'per_axis': build_per_axis(results, threshold=threshold),
    }


def evaluate_text(text: str, threshold: float = 0.7) -> Dict[str, Any]:
    """
    Convenience function to run all safeguards and aggregate results.

    Args:
        text: Input text to evaluate
        threshold: Confidence threshold for flagging

    Returns:
        Aggregated safety assessment
    """
    rule_matches = []
    if rules_enabled():
        engine = load_rule_engine_from_env()
        rule_matches = engine.match(text)
        hard_blocks = [m for m in rule_matches if m.action == "block"]
        if hard_blocks:
            # Preserve existing output keys and add rule info.
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
                    {
                        "safeguard": "rules",
                        "label": m.tag,
                        "confidence": 1.0,
                    }
                    for m in hard_blocks
                ],
                "average_confidence": 1.0,
                "individual_results": ir,
                "per_axis": build_per_axis(ir, threshold=threshold),
                "rule_matches": [m.__dict__ for m in rule_matches],
            }

    results = run_all_safeguards(text)
    aggregated = aggregate_results(results, threshold)
    if rule_matches:
        aggregated["rule_matches"] = [m.__dict__ for m in rule_matches]
        aggregated.setdefault("individual_results", {})["rules"] = {
            "label": "tag",
            "confidence": 1.0,
            "matches": [m.__dict__ for m in rule_matches],
        }
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
    print(f"Average Confidence: {result['average_confidence']:.2%}")

    if result['flags']:
        print(f"\nFlags ({len(result['flags'])}):")
        for flag in result['flags']:
            print(
                f"  - {flag['safeguard']}: {flag['label']} (confidence: {flag['confidence']:.2%})")

    print("\nIndividual Results:")
    for safeguard, res in result['individual_results'].items():
        if 'error' not in res:
            print(
                f"  {safeguard}: {res.get('label')} (confidence: {res.get('confidence', 0):.2%})")
        else:
            print(f"  {safeguard}: {res['error']}")

    print("=" * 60)
