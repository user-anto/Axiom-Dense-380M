# Axiom CLI Arguments

## `train.py`

- `--dry-run`: Run only a few steps (`TrainConfig.dry_run_steps`) and exit. Useful to validate VRAM and throughput.
- `--resume {latest,none}`: Resume policy when no explicit checkpoint is provided. Default is `latest`.
- `--resume-from PATH`: Resume from a specific checkpoint file.
- `--resume-step N`: Resume from checkpoint step `N`; if exact step is missing, it falls back to nearest earlier step.
- `--strict-step`: With `--resume-step`, require an exact checkpoint step (no fallback).
- `--reset-optimizer`: Load model weights from checkpoint but start with a fresh optimizer state.
- `--save-on-interrupt` / `--no-save-on-interrupt`: Enable/disable emergency checkpoint save on Ctrl-C. Default is enabled.
- `--ckpt-interval N`: Override checkpoint save frequency (steps).
- `--ckpt-keep-last N`: Override checkpoint retention count; `<=0` keeps all checkpoints.

Notes:
- Training metrics are written every `TrainConfig.metrics_interval` steps (default: 500) to `TrainConfig.metrics_csv_path` (default: `train_metrics.csv`).
- Eval metrics are written every `TrainConfig.eval_interval` steps (default: 500) to `TrainConfig.eval_csv_path` (default: `eval.csv`).

## `cli.py`

- `--ckpt PATH`: Checkpoint path (`.pt`). If omitted, latest checkpoint from `--ckpt-dir` is used.
- `--ckpt-dir DIR`: Directory to search for latest checkpoint. Default: `checkpoints`.
- `--prompt TEXT`: Single non-interactive prompt. If omitted, runs interactive chat mode.
- `--system TEXT`: System instruction prepended to chat prompt.
- `--max-tokens N`: Max new tokens to generate. Default: `256`.
- `--temperature FLOAT`: Sampling temperature. Default: `0.8`.
- `--top-p FLOAT`: Nucleus sampling threshold. Default: `0.9`.
- `--rep-penalty FLOAT`: Repetition penalty. Default: `1.1`.
- `--no-repeat-ngram-size N`: Block repeated n-grams of size `N`. Default: `3`.

## `test.py`

- `--ckpt PATH`: Checkpoint to evaluate. If omitted, evaluates a fresh untrained model.
- `--out-csv PATH`: CSV file to append one evaluation row. Default: `eval_metrics.csv`.
- `--train-steps N`: Number of train batches used for train-loss/gradient metrics. Default: `20`.
- `--eval-steps N`: Number of validation batches used for eval loss/perplexity. Default: `100`.
- `--attention-probe-len N`: Token window length for attention entropy probe. Default: `256`.
- `--gpu-peak-tflops FLOAT`: Override GPU peak TFLOPS for MFU estimate. Default: `0.0` (auto/disabled fallback).
