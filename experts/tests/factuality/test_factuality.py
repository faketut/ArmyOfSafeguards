"""Test script for the factuality expert."""

from experts.factuality import aggregate, predict


def test_basic_predictions():
    """Test basic factuality predictions."""
    print("=" * 60)
    print("TEST 1: Basic Factuality Predictions")
    print("=" * 60)

    test_cases = [
        "The sky is blue.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Earth is flat.",
        "Paris is the capital of France.",
        "Humans can breathe underwater without equipment.",
    ]

    results = []
    for text in test_cases:
        result = predict(text)
        results.append(result)
        print(f"\nText: {text}")
        print(f"  Label: {result['label']}")
        print(f"  Confidence: {result['confidence']:.4f}")

    return results


def test_aggregation(predictions):
    """Test the aggregation function."""
    print("\n" + "=" * 60)
    print("TEST 2: Aggregation of Multiple Predictions")
    print("=" * 60)

    aggregated = aggregate(predictions)
    print(f"\nAggregated Result:")
    print(f"  Label: {aggregated['label']}")
    print(f"  Confidence: {aggregated['confidence']:.4f}")
    print(f"  Votes: {aggregated['votes']}/{aggregated['total']}")

    return aggregated


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 60)
    print("TEST 3: Edge Cases")
    print("=" * 60)

    # Test long text (will be truncated)
    long_text = "This is a test sentence. " * 100
    print(f"\n1. Long text (truncated to 512 tokens):")
    result = predict(long_text)
    print(f"   Label: {result['label']}")
    print(f"   Confidence: {result['confidence']:.4f}")

    # Test short text
    short_text = "Yes."
    print(f"\n2. Very short text: '{short_text}'")
    result = predict(short_text)
    print(f"   Label: {result['label']}")
    print(f"   Confidence: {result['confidence']:.4f}")

    # Test empty string (should raise error)
    print(f"\n3. Empty string test:")
    try:
        predict("")
        print("   ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {e}")

    # Test whitespace only (should raise error)
    print(f"\n4. Whitespace-only string test:")
    try:
        predict("   ")
        print("   ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {e}")


def test_custom_aggregation():
    """Test aggregation with custom predictions."""
    print("\n" + "=" * 60)
    print("TEST 4: Custom Aggregation Scenarios")
    print("=" * 60)

    # Scenario 1: Clear majority
    print("\n1. Clear majority vote:")
    preds = [
        {"label": "factual", "confidence": 0.9},
        {"label": "factual", "confidence": 0.85},
        {"label": "non-factual", "confidence": 0.6},
    ]
    result = aggregate(preds)
    print(f"   Label: {result['label']}, Confidence: {result['confidence']:.4f}")
    print(f"   Votes: {result['votes']}/{result['total']}")

    # Scenario 2: Tie-breaking by confidence
    print("\n2. Tie-breaking scenario:")
    preds = [
        {"label": "factual", "confidence": 0.95},
        {"label": "non-factual", "confidence": 0.7},
    ]
    result = aggregate(preds)
    print(f"   Label: {result['label']}, Confidence: {result['confidence']:.4f}")
    print(f"   Votes: {result['votes']}/{result['total']}")

    # Scenario 3: Empty predictions (should raise error)
    print("\n3. Empty predictions test:")
    try:
        aggregate([])
        print("   ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {e}")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "FACTUALITY EXPERT TEST SUITE" + " " * 19 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # Run tests
    predictions = test_basic_predictions()
    test_aggregation(predictions)
    test_edge_cases()
    test_custom_aggregation()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()

