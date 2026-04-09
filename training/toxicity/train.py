from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root import works when executed as a script from any cwd (e.g. Colab).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.common.sequence_classifier_train import main as train_main


def main() -> int:
    return train_main()


if __name__ == "__main__":
    raise SystemExit(main())
