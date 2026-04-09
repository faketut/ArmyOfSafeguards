# Benchmark shim (legacy path)

The evaluation CLI has moved to **[`evaluation/`](../evaluation/README.md)**.

Preferred invocation from the repo root:

```bash
python evaluation/run_benchmark.py --benchmark JailbreakBench --limit 100
```

The script in this folder forwards to `evaluation/run_benchmark.py` so existing commands such as `python benchmark/run_benchmark.py ...` keep working.
