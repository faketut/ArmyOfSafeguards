"""
Compatibility aggregator entrypoint.

Repository docs refer to `aggregator/aggregator.py`. This file restores that
entrypoint while delegating implementation to the maintained aggregators.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Literal

# Ensure project root is importable when executed as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

AggregatorType = Literal["base", "weighted", "meta"]


def evaluate_text(
    text: str,
    threshold: float = 0.7,
    aggregator: AggregatorType = "base",
) -> Dict[str, Any]:
    """
    Evaluate input text using all safeguards and an aggregator.

    Args:
        text: Input text to evaluate.
        threshold: Threshold used by the chosen aggregator.
        aggregator: Which aggregator logic to use ("base", "weighted", or "meta").

    Returns:
        Dict with at least:
        - is_safe: bool
        - flags: list
        - average_confidence: float
        - individual_results: dict
    """
    if aggregator == "meta":
        from aggregator.meta_aggregator import evaluate_text as _eval

        return _eval(text, threshold=threshold)

    if aggregator == "weighted":
        from aggregator.weighted_aggregator import evaluate_text as _eval

        return _eval(text, threshold=threshold)

    from aggregator.base_aggregator import evaluate_text as _eval

    return _eval(text, threshold=threshold)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Army of Safeguards aggregator")
    parser.add_argument("text", nargs="?", help="Text snippet to evaluate")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Threshold for flagging (default: 0.7)",
    )
    parser.add_argument(
        "--aggregator",
        type=str,
        choices=["base", "weighted", "meta"],
        default="base",
        help="Aggregator to use (default: base)",
    )
    return parser


if __name__ == "__main__":
    cli = _build_cli()
    args = cli.parse_args()

    sample_text = args.text or input("Enter text to evaluate: ")
    result = evaluate_text(sample_text, threshold=args.threshold, aggregator=args.aggregator)

    print("\nRunning all safeguards...")
    print("=" * 60)
    print(f"\nOverall Safety: {'✅ SAFE' if result['is_safe'] else '⚠️  FLAGGED'}")
    print(f"Average Confidence: {result.get('average_confidence', 0.0):.2%}")

    if result.get("flags"):
        print(f"\nFlags ({len(result['flags'])}):")
        for flag in result["flags"]:
            print(
                f"  - {flag.get('safeguard')}: {flag.get('label')} "
                f"(confidence: {flag.get('confidence', 0.0):.2%})"
            )

    print("\nIndividual Results:")
    for safeguard, res in result.get("individual_results", {}).items():
        if isinstance(res, dict) and "error" not in res:
            print(
                f"  {safeguard}: {res.get('label')} "
                f"(confidence: {res.get('confidence', 0):.2%})"
            )
        else:
            err = res.get("error") if isinstance(res, dict) else "Unknown error"
            print(f"  {safeguard}: {err}")

    print("=" * 60)

