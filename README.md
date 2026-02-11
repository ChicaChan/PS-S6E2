# PS-S6E2

Kaggle competition project for **Playground Series S6E2 - Predicting Heart Disease**.

## Structure

- `src/`: local training and blending scripts
- `tests/`: unit tests for core utilities and scripts
- `configs/`: experiment configs for rank push workflow
- `kaggle_kernel/ps_s6e2_gpu/`: cloud training kernel script and metadata

## Online-First Workflow (Low Local Compute)

This repository supports a **Kaggle-cloud-first** workflow. If local compute is weak, do not run local model training.

### 1) Local minimum smoke checks only

- `python -m pytest -q`
- `python src/train_ranker.py --help`
- `python src/blend_submissions.py --help`
- `python src/online_sprint.py --help`

### 2) Run training on Kaggle only

- Push kernel:
  - `kaggle kernels push -p kaggle_kernel/ps_s6e2_gpu`
- Download outputs:
  - `kaggle kernels output chicachan/ps-s6e2-optimized-hybrid-gpu -p kaggle_outputs/cloud_latest`

Expected cloud artifacts include:
- `candidate_scores_cloud.csv`
- `metrics_rank_push_cloud.json`
- `submission_*.csv`

### 3) Validate submission files locally (no training)

- `python src/online_sprint.py validate-submission --submission-path submissions/my_candidate.csv --sample-sub-path data/raw/sample_submission.csv`

### 4) Pick two daily submissions (稳健 + 低相关)

- `python src/online_sprint.py pick-daily --candidate-scores-path kaggle_outputs/cloud_latest/candidate_scores_cloud.csv --submission-dir kaggle_outputs/cloud_latest --top-k 2 --max-correlation 0.998 --output-path kaggle_outputs/cloud_latest/daily_pick.json`

This selects:
- one highest-CV candidate;
- one diverse candidate passing correlation gate (or CV fallback).

### 5) One-command daily automation

- `powershell -ExecutionPolicy Bypass -File scripts/daily_online_loop.ps1 -KernelRef chicachan/ps-s6e2-optimized-hybrid-gpu -CloudOutputDir kaggle_outputs/cloud_latest -SampleSubmissionPath data/raw/sample_submission.csv`

Optional flags:
- `-SkipDownload` (use existing local cloud outputs)
- `-TopK 2 -MaxCorrelation 0.998`
- `-DailyPickPath kaggle_outputs/cloud_latest/daily_pick.json`

## Notes

- Full training is designed for Kaggle cloud environments.
- Local runs should be used for smoke tests only.
