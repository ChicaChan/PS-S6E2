# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains local training and ensembling entrypoints:
  - `train_lgbm_baseline.py` for a fast LightGBM baseline.
  - `train_ranker.py` for multi-model CV and rank-push experiments.
  - `blend_submissions.py` for weighted and rank blending.
- `tests/` holds unit tests for public helpers and validation logic (`test_*.py`).
- `configs/` stores experiment configs (for example `configs/rank_push_v1.yaml`).
- `kaggle_kernel/ps_s6e2_gpu/` contains the Kaggle cloud script and `kernel-metadata.json`.
- Generated artifacts belong in ignored paths such as `submissions/`, `kaggle_outputs/`, and `data/raw/`.

## Build, Test, and Development Commands
- `python -m pytest -q` - run the full local unit test suite.
- `python src/train_lgbm_baseline.py --train-path data/raw/train.csv --test-path data/raw/test.csv --output-path submissions/lgbm_optimized.csv` - baseline local smoke run.
- `python src/train_ranker.py --config-path configs/rank_push_v1.yaml` - run config-driven rank-push CV workflow.
- `python src/blend_submissions.py --inputs submissions/a.csv submissions/b.csv --method weighted_mean --output-path submissions/blended.csv` - blend multiple submissions.
- `kaggle kernels push -p kaggle_kernel/ps_s6e2_gpu` - upload the cloud training kernel.

## Coding Style & Naming Conventions
- Use Python with 4-space indentation, explicit type hints, and small single-purpose functions.
- Follow existing naming: `snake_case` for functions and variables, `UPPER_CASE` for constants, descriptive argparse flags.
- Keep imports grouped (`stdlib`, third-party, local) and preserve `from __future__ import annotations` where used.
- Prefer clear validation errors (`ValueError` and `RuntimeError`) with actionable messages.

## Testing Guidelines
- Framework: `pytest` with tests under `tests/test_*.py` and test functions named `test_*`.
- Add tests for new public functions plus edge and error cases (shape mismatch, missing columns, invalid config, unknown labels).
- Target high coverage for touched code (80%+ for modified modules when practical).
- Run focused tests first, then the full suite before opening a PR.

## Commit & Pull Request Guidelines
- Current history is minimal; use concise, imperative commit subjects (for example: `Add rank-mean blend validation`).
- Keep commits scoped to one logical change and include tests in the same commit.
- PRs should include purpose, key file changes, exact validation commands run, and notable CV/LB metric impact.
- Link related issue or task IDs when available and include sample output paths for reproducibility.

## Security & Configuration Tips
- Do not commit Kaggle credentials or raw competition data.
- Keep secrets in local environment files (for example `%USERPROFILE%/.kaggle/kaggle.json`).
- Treat `data/raw/` as local-only input and only version code and config needed to reproduce experiments.
