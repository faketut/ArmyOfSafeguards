"""Quick sanity check for the toxicity expert."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experts.toxicity import predict


def main():
    """Run quick sanity check with simple examples."""

    print("=" * 60)
    print("TOXICITY EXPERT - QUICK TEST")
    print("=" * 60)
    print()

    test_cases = [
        ("Hello, how are you today?", "safe"),
        ("I love spending time with my friends.", "safe"),
        ("This is a hateful and racist statement.", "toxic"),
    ]

    for i, (text, expected_type) in enumerate(test_cases, 1):
        print(f"Test {i}: {expected_type.upper()}")
        print(f"  Text: '{text}'")

        result = predict(text)
        label = result["label"]
        confidence = result["confidence"]

        is_safe = label in ["safe", "LABEL_0"]
        result_type = "safe" if is_safe else "toxic"

        status = "OK" if result_type == expected_type else "FAIL"
        print(f"  {status} Prediction: {label} (confidence: {confidence:.2%})")
        print()

    print("=" * 60)
    print("Quick test complete!")
    print("For comprehensive testing, run: python experts/tests/toxicity/test_toxicity.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

