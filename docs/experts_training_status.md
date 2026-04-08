# Expert & training status

## Runtime experts (`experts/`)

Each expert is a **small HF sequence-classification model** (or boolean jailbreak head) loaded by ID in code. They are **already fine-tuned upstream**; this repo consumes them via `predict()`.

| Expert | Model ID (in code) | Local fine-tune in repo |
|--------|--------------------|-------------------------|
| Toxicity | `SohamNagi/tiny-toxicity-classifier` | Optional: replace model ID after your own training |
| Sexual | `faketut/x-sensitive-deberta-binary` | Notebooks under `training/sexual/` |
| Factuality | `ajith-bondili/deberta-v3-factuality-small` | Train externally; plug in new ID |
| Jailbreak | `tommypang04/finetuned-model-jailbrak` | `training/jailbreak/train.py` (DeBERTa on in-the-wild jailbreak data) |

**Nothing is “missing weights” at runtime** as long as Hugging Face can download these checkpoints. “Finishing training” usually means **your own** fine-tunes on domain data, or **distillation** from a stronger teacher (below).

## Teacher models (not part of the 4-expert bundle)

| Teacher | Role | Notes |
|---------|------|--------|
| **ShieldGemma** (`google/shieldgemma-2b`) | Strong moderation prior | Fits on one GPU more easily than 8B |
| **Granite Guardian** (`ibm-granite/granite-guardian-3.3-8b`) | Criterion-specific (e.g. jailbreak) | Heavier; may need vLLM + large VRAM |

Use them to **label** text and to train the **meta-classifier** (and optionally future student heads). See [training/teacher_dataset/README.md](../training/teacher_dataset/README.md) and the overview [training/README.md](../training/README.md).

## Meta-classifier

- **Input features**: per-expert \(P(\text{unsafe})\) (+ optional rules flag). See `meta_classifier/feature_builder.py`.
- **Training**: `python3 -m meta_classifier.train_meta --data <jsonl> ...`
- **Recommended data**: JSONL with `individual_results` + `label`, where `label` comes from a teacher or human. Generate with `training/teacher_dataset/generate_teacher_labeled_dataset.py`.

## Suggested “finish training” order

1. Generate teacher-labeled dataset: `q` vector + `label` (CSV/JSONL).
2. Train meta: `train_meta.py` on the JSONL that includes `individual_results` + `label` (see teacher script `--write-meta-jsonl`).
3. Optionally fine-tune individual experts on domain JSON using standard `Trainer` + your labels (teacher or human), then update `MODEL_ID` in each expert module.
