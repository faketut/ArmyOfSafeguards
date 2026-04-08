"""Comprehensive unit tests for the toxicity expert."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experts.toxicity import aggregate, predict


def test_safe_content():
    """Test predictions on safe, non-toxic content."""
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
        label = result["label"]
        confidence = result["confidence"]

        is_safe = label == "LABEL_0"
        status = "✓ PASS" if is_safe else "✗ FAIL"

        print(f"{status}: '{text[:50]}...' → {label} ({confidence:.2%})")

        if is_safe:
            passed += 1

    print(f"\nPassed: {passed}/{len(safe_examples)}")
    return passed == len(safe_examples)


def test_toxic_content():
    """Test predictions on toxic/racist content."""
    print("\n" + "=" * 60)
    print("TEST 2: Toxic Content Predictions")
    print("=" * 60)

    toxic_examples = [
        "I hate those people, they are all terrible.",
        "You are worthless and stupid.",
        "All [group] are criminals and should be banned.",
    ]

    passed = 0
    for text in toxic_examples:
        result = predict(text)
        label = result["label"]
        confidence = result["confidence"]

        is_toxic = label == "LABEL_1"
        status = "✓ PASS" if is_toxic else "✗ FAIL"

        print(f"{status}: '{text[:50]}...' → {label} ({confidence:.2%})")

        if is_toxic:
            passed += 1

    print(f"\nPassed: {passed}/{len(toxic_examples)}")
    return passed == len(toxic_examples)


def test_aggregation():
    """Test majority-vote aggregation."""
    print("\n" + "=" * 60)
    print("TEST 3: Aggregation Function")
    print("=" * 60)

    predictions = [
        {"label": "LABEL_0", "confidence": 0.85},
        {"label": "LABEL_0", "confidence": 0.90},
        {"label": "LABEL_1", "confidence": 0.75},
    ]

    result = aggregate(predictions)

    passed = result["label"] == "LABEL_0" and result["votes"] == 2
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n{status}: Majority vote correctly selected LABEL_0")

    return passed


def main():
    print("\n" + "=" * 60)
    print("TOXICITY EXPERT - COMPREHENSIVE UNIT TESTS")
    print("=" * 60)

    results = []
    results.append(("Safe Content", test_safe_content()))
    results.append(("Toxic Content", test_toxic_content()))
    results.append(("Aggregation", test_aggregation()))

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print("=" * 60)
    print(f"Overall: {passed}/{total} test suites passed")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

