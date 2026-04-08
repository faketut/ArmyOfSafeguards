from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python experts/__main__.py ...` from repo root or elsewhere.
sys.path.insert(0, str(Path(__file__).parent.parent))

from experts.registry import get_expert_registry, run_expert


def main() -> int:
    reg = get_expert_registry()

    parser = argparse.ArgumentParser(description="Run an expert safeguard by name")
    parser.add_argument("--expert", required=True, choices=sorted(reg.keys()), help="Expert to run")
    parser.add_argument("text", nargs="?", help="Text to evaluate")
    args = parser.parse_args()

    text = args.text or input("Enter text to evaluate: ")
    result = run_expert(args.expert, text)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

