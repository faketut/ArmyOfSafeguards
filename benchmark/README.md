# Benchmarking Pipeline

This directory contains the benchmarking pipeline for evaluating the Army of Safeguards system against public safety benchmarks.

## Overview

The benchmarking pipeline (`run_benchmark.py`) evaluates the safeguarding system (via the aggregator) against public benchmark datasets for safety and moderation. It calculates standard metrics including accuracy, precision, recall, and F1-score.

### Interactive Demo

Try the benchmarking pipeline in Google Colab:
- **[Open in Colab](https://colab.research.google.com/drive/1YZ4G_LRwsiwhsYsOQlQhkQn4i5-r2rQI?usp=sharing)**

## Supported Benchmarks

The pipeline supports the following public benchmarks:

1. **HarmBench** (`walledai/HarmBench`)
   - Jailbreak & harmful-content robustness benchmark
   - Tests the system's ability to detect harmful prompts

2. **JailbreakBench** (`JailbreakBench/JBB-Behaviors`)
   - Jailbreak attempt detection benchmark
   - Contains harmful and benign jailbreak prompts

3. **WildGuardMix** (`allenai/wildguardmix`)
   - Moderation / guardrail benchmark
   - Tests content moderation capabilities

## Authentication

Some benchmarks (like HarmBench) are **gated datasets** that require HuggingFace authentication:

1. **Get a HuggingFace token**:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with "read" permissions

2. **Accept dataset terms**:
   - Visit the dataset page (e.g., https://huggingface.co/datasets/walledai/HarmBench)
   - Accept the terms of use

3. **Authenticate** (choose one method):
   - **Command line**:** `python benchmark/run_benchmark.py --benchmark HarmBench --hf-token YOUR_TOKEN`
   - **Environment variable**: `export HF_TOKEN=YOUR_TOKEN` (Linux/Mac) or `set HF_TOKEN=YOUR_TOKEN` (Windows)
   - **Interactive**: The script will prompt you to login if no token is found

## Usage

### Run a Single Benchmark

```bash
# Run HarmBench with 100 examples (requires authentication, uses 'standard' config by default)
python benchmark/run_benchmark.py --benchmark HarmBench --limit 100 --hf-token YOUR_TOKEN

# Or use environment variable
export HF_TOKEN=YOUR_TOKEN
python benchmark/run_benchmark.py --benchmark HarmBench --limit 100

# Use a different HarmBench config (options: 'standard', 'contextual', 'copyright')
python benchmark/run_benchmark.py --benchmark HarmBench --config contextual --hf-token YOUR_TOKEN

# Run JailbreakBench with custom threshold (no auth needed)
python benchmark/run_benchmark.py --benchmark JailbreakBench --threshold 0.8

# Run WildGuardMix with all examples (no auth needed)
python benchmark/run_benchmark.py --benchmark WildGuardMix
```

### Compare aggregators (base vs weighted vs meta)

Runs **one** expert pass per example, then applies `base`, `weighted`, and `meta` aggregation. Prints overall metrics and per-domain F1 when `metadata` includes a `domain` field (otherwise the benchmark name is used as the domain tag).

```bash
export AOS_BENCH_BATCH=1
python benchmark/run_benchmark.py --benchmark JailbreakBench --limit 200 --compare-aggregators --threshold 0.5
```

### Run All Benchmarks

```bash
# Run all benchmarks with 100 examples each
python benchmark/run_benchmark.py --all --limit 100

# Run all benchmarks with custom threshold
python benchmark/run_benchmark.py --all --threshold 0.75
```

### Command-Line Options

- `--benchmark`: Name of the benchmark to run (HarmBench, JailbreakBench, or WildGuardMix)
- `--all`: Run all available benchmarks
- `--limit`: Maximum number of examples per benchmark (default: all)
- `--threshold`: Confidence threshold for flagging (default: 0.7)
- `--hf-token`: HuggingFace token for accessing gated datasets (or set `HF_TOKEN` env var)
- `--config`: Override dataset config (e.g., for HarmBench: 'standard', 'contextual', or 'copyright')
- `--no-save`: Don't save results to JSON file

## Output

The pipeline provides:

1. **Console Output**: Real-time progress and summary metrics
2. **JSON Results**: Detailed results saved to timestamped JSON files

### Metrics Calculated

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions
- **Per-Class Metrics**: Separate metrics for "safe" and "unsafe" classes

### Example Output

```
======================================================================
Evaluating: HarmBench
Jailbreak & harmful-content robustness benchmark
======================================================================
Loading dataset...
Loaded 100 examples
Running safeguarding system...

Evaluating: 100%|████████████████████| 100/100 [00:45<00:00,  2.22it/s]

======================================================================
Results for HarmBench:
======================================================================
  Total examples: 100
  Accuracy: 85.00%
  Precision: 82.35%
  Recall: 87.50%
  F1-Score: 84.85%
  Average Confidence: 0.78

  Ground Truth Distribution:
    Safe: 40 (40.0%)
    Unsafe: 60 (60.0%)

  Prediction Distribution:
    Safe: 35 (35.0%)
    Unsafe: 65 (65.0%)

  Confusion Matrix:
    Safe → Safe: 32
    Safe → Unsafe: 8 (False Positives)
    Unsafe → Safe: 3 (False Negatives)
    Unsafe → Unsafe: 57
```

## How It Works

1. **Dataset Loading**: Loads the specified benchmark dataset from HuggingFace
2. **Text Extraction**: Extracts text fields and ground truth labels
3. **Evaluation**: Runs each text through the aggregator's `evaluate_text()` function
4. **Metrics Calculation**: Compares predictions against ground truth labels
5. **Results**: Generates comprehensive metrics and saves to JSON

## Integration with Aggregator

The pipeline uses the aggregator (`aggregator/aggregator.py`) as the entry point:

- Calls `evaluate_text(text, threshold)` for each example
- Uses `is_safe` field to determine if content was flagged
- Aggregates results from all individual safeguards (factuality, toxicity, sexual, jailbreak)

## Adding New Benchmarks

To add a new benchmark:

1. Add configuration to `BENCHMARKS` dictionary in `run_benchmark.py`:
   ```python
   "NewBenchmark": {
       "hf_id": "dataset/name",
       "config": None,
       "split": "test",
       "text_field": "text",
       "label_field": "label",
       "label_mapping": {
           True: "safe",
           False: "unsafe"
       },
       "note": "Description of benchmark"
   }
   ```

2. Update the `load_benchmark_dataset()` function if special handling is needed

3. Test with: `python benchmark/run_benchmark.py --benchmark NewBenchmark --limit 10`

## Requirements

- Python 3.9+
- `datasets` library (for loading HuggingFace datasets)
- `scikit-learn` (for metrics calculation)
- `tqdm` (for progress bars)
- All dependencies from `requirements.txt`

## Notes

- The pipeline automatically handles different dataset structures
- Field names are matched case-insensitively for flexibility
- Results are saved with timestamps for tracking
- Large datasets can be limited using `--limit` for faster testing
- Threshold can be adjusted to balance precision/recall trade-offs

## Troubleshooting

If you encounter issues loading a dataset:

1. **Authentication Errors (Gated Datasets)**:
   - **Error**: "Dataset is a gated dataset on the Hub. You must be authenticated to access it."
   - **Solution**: 
     - Get a token from https://huggingface.co/settings/tokens
     - Accept the dataset terms at the dataset's HuggingFace page
     - Use `--hf-token YOUR_TOKEN` or set `HF_TOKEN` environment variable
     - Example: `export HF_TOKEN=YOUR_TOKEN` (Linux/Mac) or `set HF_TOKEN=YOUR_TOKEN` (Windows)

2. **Field Name Mismatches**: The pipeline tries to match field names case-insensitively. If a dataset uses different field names, update the `BENCHMARKS` configuration in `run_benchmark.py`.

3. **Dataset Structure Changes**: Some datasets may have different structures. Check the HuggingFace dataset page for the correct field names and update the configuration accordingly.

4. **Missing Labels**: If a dataset doesn't have labels in the expected format, the pipeline will skip those examples and show a warning.

5. **Memory Issues**: For very large datasets, use `--limit` to restrict the number of examples evaluated.

