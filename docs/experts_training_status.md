# Expert & training status

## Runtime experts (`experts/`)

Each expert is a **small HF sequence-classification model** (or boolean jailbreak head) loaded by ID in code. They are **already fine-tuned upstream**; this repo consumes them via `predict()`.

| Expert | Model ID (in code) | Local fine-tune in repo |
|--------|--------------------|-------------------------|
| Toxicity | `SohamNagi/tiny-toxicity-classifier` | Optional: replace model ID after your own training |
| Sexual | `faketut/x-sensitive-deberta-binary` | Notebooks under `training/sexual/` |
| Factuality | `ajith-bondili/deberta-v3-factuality-small` | Train externally; plug in new ID |
| Jailbreak | `tommypang04/finetuned-model-jailbrak` | `training/jailbreak/train.py` (DeBERTa on in-the-wild jailbreak data) |

**Nothing is “missing weights” at runtime** as long as Hugging Face can download these checkpoints. “Finishing training” usually means **your own** fine-tunes on domain data, or **distillation** from a stronger external moderator (outside this repo).

## Meta-classifier

- **Input features**: per-expert \(P(\text{unsafe})\) (+ optional rules flag). See `meta_classifier/feature_builder.py`.
- **Training**: `python3 -m meta_classifier.train_meta --data <jsonl> ...`
- **Recommended data**: JSONL with `individual_results` + `label`, where `label` is human, dataset-native (see `training/meta/build_meta_from_hf_labels.py`), or from the in-repo **expert-Q heuristic** in `training/teacher_dataset/generate_teacher_labeled_dataset.py`.

## Suggested “finish training” order

1. Build a dataset with `q` / `individual_results` + `label` (CSV/JSONL as needed).
2. Train meta: `train_meta.py` on JSONL that includes `individual_results` + `label`.
3. Optionally fine-tune individual experts on domain JSON using standard `Trainer` + your labels, then update `MODEL_ID` in each expert module.
