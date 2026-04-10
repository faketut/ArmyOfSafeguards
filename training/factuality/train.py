from __future__ import annotations

"""
Fine-tune the factuality expert from native curriculum JSONL (see training/experts/build_expert_sft_jsonl.py).

Example:
  python training/experts/build_expert_sft_jsonl.py --expert factuality --out data/factuality_native.jsonl
  python training/factuality/train.py --data data/factuality_native.jsonl

Runtime:
  set AOS_FACTUALITY_MODEL to the saved directory (or HF repo id).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.common.sequence_classifier_train import main as train_main


def main() -> int:
    return train_main(
        output_dir="experts/artifacts/factuality_ft",
        domain="factuality",
        max_length=512,
        model_name="ajith-bondili/deberta-v3-factuality-small",
    )


if __name__ == "__main__":
    raise SystemExit(main())
