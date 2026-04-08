# Toxicity Expert

This expert detects racist, hateful, and toxic content in text. It uses the Hugging Face model `SohamNagi/tiny-toxicity-classifier`.

## Model
- **Model**: `SohamNagi/tiny-toxicity-classifier`
- **Labels**: typically `LABEL_0` (safe) and `LABEL_1` (unsafe) depending on model config.

## Usage

### Command Line
```bash
python experts/toxicity.py "Your text to evaluate"
```

### Python API
```python
from experts.toxicity import predict, aggregate
```

## Tests
```bash
python experts/tests/toxicity/quick_test.py
python experts/tests/toxicity/test_toxicity.py
python experts/tests/toxicity/evaluate_toxicity.py --limit 100
```

