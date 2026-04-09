# Training

Entry points for fine-tuning experts, building meta-aggregator data, and running end-to-end pipelines.

| Path | Purpose |
|------|---------|
| [meta/README.md](meta/README.md) | Meta JSONL schema helpers, synthetic data, `generate_expert_features.py`, `train_meta` commands |
| [teacher_dataset/README.md](teacher_dataset/README.md) | **Q-values** (per-expert P(unsafe)) + **ShieldGemma / Granite** teacher labels → CSV + meta-ready JSONL |
| [teacher_dataset/run_teacher_meta_pipeline.sh](teacher_dataset/run_teacher_meta_pipeline.sh) | One-shot: teacher-labeled JSONL → `meta_classifier/artifacts/meta_lr.json` |
| [jailbreak/train.py](jailbreak/train.py) | Optional DeBERTa jailbreak expert fine-tune |
| [toxicity/train.py](toxicity/train.py) | Fine-tune toxicity expert on teacher-labeled meta JSONL |
| `sexual/` | Notebooks for sensitive-content expert experiments |

**Docs:** [docs/experts_training_status.md](../docs/experts_training_status.md), [docs/meta_training_labeling.md](../docs/meta_training_labeling.md).

### Quick: teacher labels → trained meta

```bash
export LIMIT=500   # optional cap for iteration
./training/teacher_dataset/run_teacher_meta_pipeline.sh
```

With dry-run (no teacher GPU load; heuristic labels):

```bash
DRY_RUN=1 LIMIT=50 ./training/teacher_dataset/run_teacher_meta_pipeline.sh
```
