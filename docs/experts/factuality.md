# Factuality Expert

This expert flags model outputs that contradict verified facts or propagate misinformation. It is powered by the fine-tuned `ajith-bondili/deberta-v3-factuality-small` DeBERTa-v3 sequence classifier.

## Model
- **Model**: `ajith-bondili/deberta-v3-factuality-small`
- **Labels**:
  - `LABEL_0`: Factual
  - `LABEL_1`: Non-factual/Uncertain

## Usage

### Command Line
```bash
python experts/factuality.py "The Earth orbits the sun once every 365 days."
```

### Python API
```python
from experts.factuality import predict, aggregate
```

## Tests
```bash
python experts/tests/factuality/quick_test.py
python experts/tests/factuality/test_factuality.py
python experts/tests/factuality/evaluate_factuality.py --limit 100
```

