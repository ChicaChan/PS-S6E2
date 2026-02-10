from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend Kaggle submission files")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of submission CSV files")
    parser.add_argument("--weights", default="", help="Comma-separated weights; defaults to equal weights")
    parser.add_argument("--method", choices=("weighted_mean", "rank_mean"), default="weighted_mean")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--target-col", default="Heart Disease")
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def parse_weights(raw: str, n_inputs: int) -> np.ndarray:
    if not raw:
        return np.ones(n_inputs, dtype=np.float64) / n_inputs

    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if len(parts) != n_inputs:
        raise ValueError(f"weights count mismatch: expected {n_inputs}, got {len(parts)}")

    values = np.array([float(item) for item in parts], dtype=np.float64)
    if np.any(values < 0):
        raise ValueError("weights must be non-negative")
    if np.allclose(values.sum(), 0.0):
        raise ValueError("weights sum cannot be zero")

    return values / values.sum()


def load_submission(path: Path, id_col: str, target_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = [id_col, target_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    if frame[required].isna().any().any():
        raise ValueError(f"{path} contains NaN in required columns")
    return frame[required].copy()


def validate_alignment(submissions: list[pd.DataFrame], id_col: str) -> None:
    if not submissions:
        raise ValueError("No submissions provided")

    base_ids = submissions[0][id_col].to_numpy()
    for idx, submission in enumerate(submissions[1:], start=1):
        if len(submission) != len(submissions[0]):
            raise ValueError(
                f"Row count mismatch between submission[0]={len(submissions[0])} and submission[{idx}]={len(submission)}"
            )
        if not np.array_equal(base_ids, submission[id_col].to_numpy()):
            raise ValueError(f"ID column mismatch between submission[0] and submission[{idx}]")


def blend_predictions(
    submissions: list[pd.DataFrame],
    weights: np.ndarray,
    method: str,
    target_col: str,
) -> np.ndarray:
    matrix = np.column_stack([submission[target_col].to_numpy(dtype=np.float64) for submission in submissions])

    if method == "weighted_mean":
        blended = matrix @ weights
    elif method == "rank_mean":
        ranks = np.column_stack(
            [pd.Series(matrix[:, idx]).rank(method="average", pct=True).to_numpy() for idx in range(matrix.shape[1])]
        )
        blended = ranks @ weights
    else:
        raise ValueError(f"Unsupported blend method: {method}")

    return np.clip(blended, 0.0, 1.0)


def main() -> None:
    args = parse_args()

    input_paths = [Path(item) for item in args.inputs]
    weights = parse_weights(args.weights, len(input_paths))

    submissions = [load_submission(path, args.id_col, args.target_col) for path in input_paths]
    validate_alignment(submissions, args.id_col)
    blended = blend_predictions(submissions, weights, args.method, args.target_col)

    output = pd.DataFrame(
        {
            args.id_col: submissions[0][args.id_col],
            args.target_col: blended,
        }
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Method: {args.method}")
    print(f"Weights: {[round(float(weight), 6) for weight in weights.tolist()]}")
    print(f"Saved blended submission: {output_path}")


if __name__ == "__main__":
    main()
