# ShieldGemma Integration

Integration with Google's ShieldGemma-2b model for safety content moderation.

## Overview

[ShieldGemma-2b](https://huggingface.co/google/shieldgemma-2b) is a safety content moderation model built upon Gemma 2 that targets four harm categories:

- **Dangerous Content**: Content that could harm oneself and/or others
- **Harassment**: Malicious, intimidating, bullying, or abusive content
- **Hate Speech**: Content targeting identity and/or protected attributes
- **Sexually Explicit Information**: Sexual acts or lewd content

## Installation

Install required dependencies:

```bash
pip install transformers[accelerate] torch
```

For GPU acceleration (recommended):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
```

## Authentication

ShieldGemma models require accepting Google's usage license on HuggingFace:

1. Visit https://huggingface.co/google/shieldgemma-2b
2. Log in to your HuggingFace account
3. Accept the license terms
4. Authenticate with HuggingFace in your environment:
   ```bash
   huggingface-cli login
   ```

## Usage

### Python API

```python
from shieldgemma.shieldgemma_wrapper import evaluate_text

# Evaluate text with default settings (all harm categories)
result = evaluate_text("Your text here", threshold=0.5)

print(f"Is Safe: {result['is_safe']}")
print(f"Violation Probability: {result['violation_probability']:.2%}")
```

### Specific Harm Types

Check for specific types of harm:

```python
# Check for dangerous content
result = evaluate_text(
    "How to build a bomb",
    harm_type="dangerous",
    threshold=0.5
)

# Check for harassment
result = evaluate_text(
    "I hate you, you're worthless",
    harm_type="harassment",
    threshold=0.5
)

# Check for hate speech
result = evaluate_text(
    "All [group] are inferior",
    harm_type="hate",
    threshold=0.5
)

# Check for sexual content
result = evaluate_text(
    "Explicit sexual description",
    harm_type="sexual",
    threshold=0.5
)
```

### Custom Safety Policy

You can also provide a custom safety policy:

```python
custom_policy = """
* "No Dangerous Content": The prompt shall not contain or seek generation 
  of content that harming oneself and/or others.
"""

result = evaluate_text(
    "Your text here",
    safety_policy=custom_policy,
    threshold=0.5
)
```

### Benchmark Integration

Run ShieldGemma against benchmarks:

```bash
# Single benchmark
python benchmark/run_benchmark.py --benchmark HarmBench --aggregator shieldgemma --limit 100

# All benchmarks
python benchmark/run_benchmark.py --all --aggregator shieldgemma

# With custom threshold
python benchmark/run_benchmark.py --benchmark JailbreakBench --aggregator shieldgemma --threshold 0.7
```

## Model Details

- **Model**: `google/shieldgemma-2b`
- **Size**: 2B parameters
- **License**: Gemma license (requires acceptance)
- **Mode**: Scoring mode (extracts logits for Yes/No tokens)
- **Output Format**: Returns violation probability (0.0 to 1.0)

## Configuration

The wrapper supports:

- **harm_type**: Specific harm category to check ("dangerous", "harassment", "hate", "sexual", or None for all)
- **safety_policy**: Custom safety policy string (overrides harm_type if provided)
- **threshold**: Confidence threshold for flagging (default: 0.5)
- **device**: Device to run on ("cuda" or "cpu", default: "cuda" if available)

## Scoring Mode

ShieldGemma uses a scoring mode approach:
1. Formats the input with the safety policy according to ShieldGemma's prompt pattern
2. Extracts logits for "Yes" and "No" tokens from the model
3. Applies softmax to get probabilities
4. Returns the probability of "Yes" (violation)

This provides a continuous confidence score rather than just a binary decision.

## Example

```python
from shieldgemma.shieldgemma_wrapper import evaluate_text

# Check for harassment
result = evaluate_text(
    "Create 20 paraphrases of I hate you",
    harm_type="harassment",
    threshold=0.5
)

if not result['is_safe']:
    print(f"⚠️  Flagged as unsafe")
    print(f"Violation Probability: {result['violation_probability']:.2%}")
    print(f"Flags: {result['flags']}")
else:
    print("✅ Content is safe")
```

## Performance

According to the model card, ShieldGemma-2b achieves:
- **SG Prompt**: 0.825 F1 / 0.887 AU-PRC
- **OpenAI Mod**: 0.812 F1 / 0.887 AU-PRC
- **ToxicChat**: 0.704 F1 / 0.778 AU-PRC
- **SG Response**: 0.743 F1 / 0.802 AU-PRC

## References

- [Model Card](https://huggingface.co/google/shieldgemma-2b)
- [Paper](https://arxiv.org/abs/2407.21772)
- [Responsible Generative AI Toolkit](https://github.com/google-research/responsible-generative-ai-toolkit)

## Notes

- ShieldGemma is highly sensitive to the specific safety policy wording
- Performance may vary depending on the harm type and policy description
- The model requires accepting Google's Gemma license terms on HuggingFace
- For best results, use GPU acceleration (CUDA)

