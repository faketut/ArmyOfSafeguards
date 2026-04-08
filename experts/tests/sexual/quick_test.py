"""Quick sanity check for the sexual content expert."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experts.sexual import predict


def main():
    """Run quick sanity check with simple examples."""

    print("=" * 60)
    print("SEXUAL CONTENT EXPERT - QUICK TEST")
    print("=" * 60)
    print()

    test_cases = [
        ("Hello, how are you today?", "safe"),
        ("I love spending time with my friends.", "safe"),
        ("This is so fucking cool!", "sensitive"),
    ]

    for i, (text, expected_type) in enumerate(test_cases, 1):
        print(f"Test {i}: {expected_type.upper()}")
        print(f"  Text: '{text}'")

        result = predict(text)
        label = result["label"]
        confidence = result["confidence"]

        is_safe = label == "LABEL_0"
        result_type = "safe" if is_safe else "sensitive"

        status = "OK" if result_type == expected_type else "FAIL"
        print(f"  {status} Prediction: {label} (confidence: {confidence:.2%})")
        print()

    print("=" * 60)
    print("Quick test complete!")
    print("For comprehensive testing, run: python experts/tests/sexual/test_sexual.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

