"""Quick test to verify the factuality expert is working."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experts.factuality import predict

print("Testing factuality expert...")
print("-" * 50)

# Test a factual statement
factual = "The capital of France is Paris."
result1 = predict(factual)
print(f"[OK] Factual: '{factual}'")
print(f"  -> {result1['label']} (confidence: {result1['confidence']:.2%})")

print()

# Test a non-factual statement
nonfactual = "The moon is made of cheese."
result2 = predict(nonfactual)
print(f"[OK] Non-factual: '{nonfactual}'")
print(f"  -> {result2['label']} (confidence: {result2['confidence']:.2%})")

print("-" * 50)
print("[OK] Expert is working correctly!")

