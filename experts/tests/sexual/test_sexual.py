"""Comprehensive unit tests for the sexual content expert."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experts.sexual import aggregate, predict


def test_safe_content():
    print("\n" + "=" * 60)
    print("TEST 1: Safe Content Predictions")
    print("=" * 60)

    safe_examples = [
        "I love learning new things!",
        "What a beautiful day it is today.",
        "Thank you for your help, I really appreciate it.",
        "The movie was entertaining and well-made.",
        "Let's work together to solve this problem.",
    ]

    passed = 0
    for text in safe_examples:
        result = predict(text)
        is_safe = result["label"] == "LABEL_0"
        status = "✓ PASS" if is_safe else "✗ FAIL"
        print(f"{status}: '{text[:50]}...' → {result['label']} ({result['confidence']:.2%})")
        if is_safe:
            passed += 1
    return passed == len(safe_examples)


def test_sensitive_content():
    print("\n" + "=" * 60)
    print("TEST 2: Sensitive Content Predictions")
    print("=" * 60)

    sensitive_examples = [
        "This is so fucking cool!",
        "That's a damn good idea.",
        "I can't believe this shit.",
    ]

    passed = 0
    for text in sensitive_examples:
        result = predict(text)
        is_sensitive = result["label"] == "LABEL_1"
        status = "✓ PASS" if is_sensitive else "✗ FAIL"
        print(f"{status}: '{text[:50]}...' → {result['label']} ({result['confidence']:.2%})")
        if is_sensitive:
            passed += 1
    return passed == len(sensitive_examples)


def test_aggregation():
    predictions = [
        {"label": "LABEL_0", "confidence": 0.85},
        {"label": "LABEL_0", "confidence": 0.90},
        {"label": "LABEL_1", "confidence": 0.75},
    ]
    result = aggregate(predictions)
    return result["label"] == "LABEL_0" and result["votes"] == 2


def main():
    ok = all([test_safe_content(), test_sensitive_content(), test_aggregation()])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

