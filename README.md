# Army of Safeguards

A modular, research-friendly collection of **AI safety “safeguards”** (specialized detectors) for flagging harmful or problematic content, plus an **aggregator** that runs multiple safeguards and returns a unified safety assessment.


## Badges

- **Python**: 3.9+
- **Status**: active development
- **License**: TBD (see [License](#license))

## Key features

- **Modular experts**: separate safeguards for factuality, sexual/sensitive content, toxicity, and jailbreak attempts
- **Unified API**: a single `evaluate_text(...)` call to run all safeguards together
- **CLI-friendly**: run individual experts or the aggregator from the command line
- **Benchmarking & evaluation**: scripts and documented results for each safeguard

## Project layout

```
ArmyOfSafeguards/
├── experts/                 # Expert safeguards (specialized detectors)
│   ├── factuality.py
│   ├── toxicity.py
│   ├── sexual.py
│   ├── jailbreak.py
│   └── __init__.py
├── docs/                    # Documentation
│   └── experts/             # Per-expert docs
├── training/                # Training scripts / notebooks — see training/README.md
│   ├── jailbreak/
│   ├── sexual/
│   ├── meta/                # Meta-aggregator data & expert-featurization scripts
│   └── teacher_dataset/     # Q-values + ShieldGemma/Granite teacher labels → meta training
├── meta_classifier/         # Learned logistic meta-model over expert outputs
├── docs/
│   ├── experts/             # Per-expert docs
│   ├── meta_training_labeling.md  # Meta training JSONL schema & labeling
│   └── experts_training_status.md # Which experts are pretrained vs trainable locally
├── aggregator/              # Unified interface for all safeguards
│   ├── aggregator.py
│   └── README.md
├── requirements.txt         # Shared dependencies
└── README.md
```

## Getting started

### Install

```bash
git clone https://github.com/SohamNagi/ArmyOfSafeguards.git
cd ArmyOfSafeguards

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Quick usage (CLI)

Run an individual safeguard:

```bash
python experts/factuality.py "The Earth is flat."
python experts/sexual.py "Your text to evaluate"
python experts/toxicity.py "Your text to evaluate"
python experts/jailbreak.py "Your text to evaluate"
```

Run the aggregator (all safeguards):

```bash
python aggregator/aggregator.py "Your text to evaluate here"
```

### Quick usage (Python)

Individual expert:

```python
from experts.toxicity import predict

result = predict("Hello, how are you?")
print(f"Label: {result['label']}, Confidence: {result['confidence']:.2%}")
```

Aggregator:

```python
from aggregator.aggregator import evaluate_text

result = evaluate_text("Your text here", threshold=0.7)
print(f"Is Safe: {result['is_safe']}")
print(f"Individual Results: {result['individual_results']}")
```

## Included safeguards

- **Factuality safeguard (Ajith)**
  - **Model**: `ajith-bondili/deberta-v3-factuality-small`
  - **Docs**: `docs/experts/factuality.md`
- **Sexual / sensitive-content safeguard (Jian)**
  - **Model**: `faketut/x-sensitive-deberta-binary`
  - **Docs**: `docs/experts/sexual.md`
- **Toxicity safeguard (Soham)**
  - **Model**: `SohamNagi/tiny-toxicity-classifier`
  - **Docs**: `docs/experts/toxicity.md`
- **Jailbreak safeguard (Tommy)**
  - **Model**: `tommypang04/finetuned-model-jailbrak`
  - **Docs**: `docs/experts/jailbreak.md`

## Meta aggregator training

- **Training index**: [training/README.md](training/README.md)
- **Expert / training status**: [docs/experts_training_status.md](docs/experts_training_status.md)
- **Labeling & schema**: [docs/meta_training_labeling.md](docs/meta_training_labeling.md)
- **Pipeline & commands**: [training/meta/README.md](training/meta/README.md)
- **Teacher-labeled data (Q₁…Q₄ + label)**: [training/teacher_dataset/README.md](training/teacher_dataset/README.md) — run `training/teacher_dataset/generate_teacher_labeled_dataset.py` with `--teacher shieldgemma` or `--teacher granite`, then `train_meta` on `--output-meta-jsonl`.
- Train a model: `python3 -m meta_classifier.train_meta --data training/meta/synthetic_meta_train.jsonl --n-folds 5 --calibrate temperature --out meta_classifier/artifacts/meta_lr.json`
- End-to-end shell example: `training/teacher_dataset/run_teacher_meta_pipeline.sh`
- Runtime: set `AOS_META_MODEL_PATH` to your artifact, or use the default path under `meta_classifier/artifacts/`.

## Testing & evaluation

Each safeguard includes runnable tests/evaluators. Example commands:

```bash
python experts/tests/factuality/quick_test.py
python experts/tests/sexual/quick_test.py
python experts/tests/toxicity/quick_test.py
python experts/tests/jailbreak/quick_test.py
```

### Reported results (from this repo)

**Factuality safeguard**

Model trained on TruthfulQA & FEVER; OOD datasets are most indicative of generalization.

| Dataset | Accuracy | F1-Score | Domain |
|---------|----------|----------|--------|
| VitaminC | 54.00% | 36.11% | General claims |
| Climate-FEVER | 81.00% | - | Climate-specific |
| LIAR | 81.00% | - | Political statements |

Sanity check on training-domain datasets:

| Dataset | Accuracy | F1-Score |
|---------|----------|----------|
| FEVER | 84.00% | 78.38% |
| TruthfulQA | 75.00% | - |

**Sexual / sensitive-content safeguard**

| Metric | Score |
|--------|-------|
| Accuracy | 82.6% |
| F1-Score | 82.9% |

**Toxicity safeguard (ToxiGen)**

| Metric | Score |
|--------|-------|
| Accuracy | 79.00% |
| Precision | 75.00% |
| Recall | 69.23% |
| F1-Score | 72.00% |

**Jailbreak safeguard**

| Metric | Score |
|--------|-------|
| Accuracy | 94.8248% |
| F1-Score | 65.7143% |

### Benchmarks & datasets

- **Individual safeguard datasets**
  - **Factuality**: TruthfulQA, FEVER, SciFact, VitaminC, Climate-FEVER
  - **Sexual**: CardiffNLP x_sensitive
  - **Toxicity**: ToxiGen, hate_speech18, civil_comments
  - **Jailbreak**: JBB-Behaviors
- **System benchmarks**
  - **Jailbreak & harmful-content robustness**: [HarmBench](https://huggingface.co/datasets/walledai/HarmBench), [JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors)
  - **Moderation / guardrail benchmarks**: [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix)
  - **Broader safety suites**: [HELM Safety](https://crfm.stanford.edu/helm/safety/latest/)

## Contributing

Contributions are welcome. The main extension point is adding new safeguards that follow the same minimal interface used across the repo.

- **Add a new safeguard**
  - Create a new expert module (e.g., `experts/my_guard.py` or a new folder if needed).
  - Implement `predict()` returning `{"label": str, "confidence": float}`.
  - Wire it into the aggregator so it participates in `evaluate_text(...)`.
  - Add tests and documentation under `docs/experts/`.

If you’re making a larger change (new aggregator strategy, new benchmark suite, refactors), please include a short write-up in the PR describing motivation and evaluation.

## Security

If you discover a vulnerability or an issue that could lead to unsafe behavior in downstream usage, please open an issue with a minimal reproduction and clearly label it **security**.

## License

This repository currently has **no finalized license text** in `README.md`. Add a standard OSS license file (e.g., MIT/Apache-2.0) and update this section accordingly.

## Team

- **Ajith**: Factuality safeguard
- **Soham**: Toxicity safeguard
- **Jian**: Sexual/sensitive-content safeguard
- **Tommy**: Jailbreak safeguard
