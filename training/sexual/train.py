from __future__ import annotations

"""
Fine-tune the sexual/sensitive-content expert from native curriculum JSONL
(see training/experts/build_expert_sft_jsonl.py).

Example:
  python training/experts/build_expert_sft_jsonl.py --expert sexual --out data/sexual_native.jsonl
  python training/sexual/train.py --data data/sexual_native.jsonl

Runtime:
  set AOS_SEXUAL_MODEL to the saved directory (or HF repo id).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.common.sequence_classifier_train import main as train_main


def main() -> int:
    return train_main(
        output_dir="experts/artifacts/sexual_ft",
        domain="sexual",
        max_length=128,
        model_name="faketut/x-sensitive-deberta-binary",
    )


if __name__ == "__main__":
    raise SystemExit(main())
