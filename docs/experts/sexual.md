# Sexual Content Expert

This expert detects sexual and sensitive content in text. It uses the Hugging Face model `faketut/x-sensitive-deberta-binary`.

## Model
- **Model**: `faketut/x-sensitive-deberta-binary`
- **Labels**:
  - `LABEL_0`: Safe
  - `LABEL_1`: Sensitive/sexual content

## Usage

### Command Line
```bash
python experts/sexual.py "Your text to evaluate"
```

### Python API
```python
from experts.sexual import predict, aggregate
```

## Tests
```bash
python experts/tests/sexual/quick_test.py
python experts/tests/sexual/test_sexual.py
python experts/tests/sexual/evaluate_sexual.py --limit 100
```

## Training
- Notebooks live in `training/sexual/`.

