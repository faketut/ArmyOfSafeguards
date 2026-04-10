# Moved: expert-Q meta labeling

This directory used to hold the **expert-Q heuristic** pipeline (max expert P(unsafe) vs a threshold). Everything now lives under **`training/meta/`** so native-label meta and expert-Q meta stay in one place.

| Old | New |
|-----|-----|
| `generate_teacher_labeled_dataset.py` | [`../meta/generate_expert_q_meta_jsonl.py`](../meta/generate_expert_q_meta_jsonl.py) |
| `label_manifest.py` | [`../meta/label_manifest_expert_q.py`](../meta/label_manifest_expert_q.py) |
| `run_teacher_meta_pipeline.sh` | [`../meta/run_expert_q_meta_pipeline.sh`](../meta/run_expert_q_meta_pipeline.sh) |
| `manifest.example.json` | [`../meta/manifest.expert_q.example.json`](../meta/manifest.expert_q.example.json) |
| `expert_q_label.py` | [`../meta/expert_q_label.py`](../meta/expert_q_label.py) |

Details: [`../meta/README.md`](../meta/README.md) (section **Expert-Q heuristic meta**).
