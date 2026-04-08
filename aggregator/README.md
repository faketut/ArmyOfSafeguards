# Aggregator

The aggregator module provides a unified interface for running multiple safeguards on input text.

## Usage

### Command Line

```bash
python aggregator/aggregator.py "Your text to evaluate here"
```

### Python API

```python
from aggregator.aggregator import evaluate_text
from aggregator.expert_runner import run_all_safeguards

# Run all safeguards and get aggregated result
result = evaluate_text("Your text here", threshold=0.7)

print(f"Is Safe: {result['is_safe']}")
print(f"Flags: {result['flags']}")

# Or run safeguards individually
individual_results = run_all_safeguards("Your text here")
print(individual_results)
```

## How It Works

1. **run_all_safeguards()** - Imports and runs each available safeguard critic
2. **aggregate_results()** - Combines results using configurable threshold logic
3. **evaluate_text()** - Convenience function that does both steps

## Adding New Safeguards

When a new safeguard is added to the repo (e.g., `experts/toxicity.py`):

1. Add it under `experts/`
2. Import it in `aggregator/expert_runner.py`
3. The aggregator will automatically include it in evaluations

## Configuration

- **threshold**: Confidence threshold for flagging content (default: 0.7)
- Adjust aggregation logic in `aggregate_results()` as needed

