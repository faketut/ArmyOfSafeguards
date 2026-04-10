# Expert & training status

## Runtime experts (`experts/`)

Each expert is a **small HF sequence-classification model** (or boolean jailbreak head) loaded by ID in code. They are **already fine-tuned upstream**; this repo consumes them via `predict()`.

| Expert | Model ID (default) | Local SFT in repo |
|--------|--------------------|-------------------|
| Toxicity | `SohamNagi/tiny-toxicity-classifier` | `training/toxicity/train.py` → set `AOS_TOXICITY_MODEL` |
| Sexual | `faketut/x-sensitive-deberta-binary` | `training/sexual/train.py` → set `AOS_SEXUAL_MODEL` |
| Factuality | `ajith-bondili/deberta-v3-factuality-small` | `training/factuality/train.py` → set `AOS_FACTUALITY_MODEL` |
| Jailbreak | `tommypang04/finetuned-model-jailbrak` | `training/jailbreak/train.py` → set `AOS_JAILBREAK_MODEL` |

**Nothing is “missing weights” at runtime** as long as Hugging Face can download these checkpoints. “Finishing training” usually means **your own** fine-tunes on domain data, or **distillation** from a stronger external moderator (outside this repo).

## Meta-classifier

- **Input features**: per-expert \(P(\text{unsafe})\) (+ optional rules flag). See `meta_classifier/feature_builder.py`.
- **Training**: `python3 -m meta_classifier.train_meta --data <jsonl> ...`
- **Recommended data**: JSONL with `individual_results` + `label`, where `label` is human, dataset-native (see `training/meta/build_meta_from_hf_labels.py`), or from the in-repo **expert-Q heuristic** in `training/teacher_dataset/generate_teacher_labeled_dataset.py`.

## Suggested “finish training” order

1. Build a dataset with `q` / `individual_results` + `label` (CSV/JSONL as needed).
2. Train meta: `train_meta.py` on JSONL that includes `individual_results` + `label`.
3. Optionally fine-tune individual experts on native curriculum JSONL (`training/experts/build_expert_sft_jsonl.py` + the matching `training/<expert>/train.py`), then point runtime at the artifact with the corresponding `AOS_*_MODEL` env var (see table above).
