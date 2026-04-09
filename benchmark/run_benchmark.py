"""
Backward-compatible entry point. Implementation lives in evaluation/run_benchmark.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    target = root / "evaluation" / "run_benchmark.py"
    if not target.is_file():
        print(f"error: expected benchmark runner at {target}", file=sys.stderr)
        return 1
    return int(subprocess.call([sys.executable, str(target), *sys.argv[1:]]))


if __name__ == "__main__":
    raise SystemExit(main())
