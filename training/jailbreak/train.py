from __future__ import annotations

"""
Fine-tune the jailbreak expert from native curriculum JSONL (see training/experts/build_expert_sft_jsonl.py).

Example:
  python training/experts/build_expert_sft_jsonl.py --expert jailbreak --out data/jailbreak_native.jsonl \\
    --target-train-pos-rate 0.3
  python training/jailbreak/train.py --data data/jailbreak_native.jsonl --domain jailbreak \\
    --output-dir experts/artifacts/jailbreak_ft --model-name FacebookAI/roberta-base

Runtime:
  set AOS_JAILBREAK_MODEL to the saved directory (or HF repo id).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.common.sequence_classifier_train import main as train_main


def main() -> int:
    return train_main(
        output_dir="experts/artifacts/jailbreak_ft",
        domain="jailbreak",
        max_length=384,
        model_name="FacebookAI/roberta-base",
    )


if __name__ == "__main__":
    raise SystemExit(main())
