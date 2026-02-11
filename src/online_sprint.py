from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_CANDIDATE_COLUMNS = ("candidate", "cv_auc", "submission_file")


def load_candidate_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_CANDIDATE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Candidate scores missing required columns: {missing}")

    scores = frame[list(REQUIRED_CANDIDATE_COLUMNS)].copy()
    scores["cv_auc"] = pd.to_numeric(scores["cv_auc"], errors="coerce")
    if scores["cv_auc"].isna().any():
        raise ValueError("Candidate scores contain non-numeric cv_auc values")

    return scores.sort_values("cv_auc", ascending=False, kind="mergesort").reset_index(drop=True)


def load_prediction_column(path: Path, target_col: str) -> pd.Series:
    frame = pd.read_csv(path)
    if target_col not in frame.columns:
        raise ValueError(f"{path} missing target column: {target_col}")

    prediction = pd.to_numeric(frame[target_col], errors="coerce")
    if prediction.isna().any():
        raise ValueError(f"{path} has non-numeric or NaN values in {target_col}")

    return prediction.astype("float64")


def validate_submission_csv(
    submission_path: Path,
    sample_submission_path: Path,
    id_col: str,
    target_col: str,
) -> None:
    submission = pd.read_csv(submission_path)
    sample_submission = pd.read_csv(sample_submission_path)

    expected_columns = sample_submission.columns.tolist()
    got_columns = submission.columns.tolist()
    if got_columns != expected_columns:
        raise ValueError(f"Submission columns mismatch: expected {expected_columns}, got {got_columns}")

    if len(submission) != len(sample_submission):
        raise ValueError(f"Submission row mismatch: expected {len(sample_submission)}, got {len(submission)}")

    if submission.isna().any().any():
        raise ValueError("Submission contains NaN values")

    if id_col not in submission.columns or id_col not in sample_submission.columns:
        raise ValueError(f"Missing id column in submission or sample: {id_col}")

    if not np.array_equal(submission[id_col].to_numpy(), sample_submission[id_col].to_numpy()):
        raise ValueError("Submission id column order does not match sample submission")

    if target_col not in submission.columns:
        raise ValueError(f"Submission missing target column: {target_col}")

    prediction = pd.to_numeric(submission[target_col], errors="coerce")
    if prediction.isna().any():
        raise ValueError(f"Submission has non-numeric values in target column: {target_col}")
    if ((prediction < 0.0) | (prediction > 1.0)).any():
        raise ValueError("Submission predictions must be within [0, 1]")


def correlation(a: pd.Series, b: pd.Series) -> float:
    value = float(np.corrcoef(a.to_numpy(dtype=np.float64), b.to_numpy(dtype=np.float64))[0, 1])
    if np.isnan(value):
        return 1.0
    return value


def select_submission_candidates(
    candidate_scores: pd.DataFrame,
    submission_dir: Path,
    target_col: str,
    top_k: int = 2,
    max_correlation: float = 0.998,
) -> list[dict[str, object]]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    selected: list[dict[str, object]] = []
    cached_pred: dict[str, pd.Series] = {}

    for row in candidate_scores.sort_values("cv_auc", ascending=False, kind="mergesort").itertuples(index=False):
        candidate_name = str(row.candidate)
        submission_file = str(row.submission_file)
        file_path = submission_dir / submission_file

        if not file_path.exists():
            raise ValueError(f"Candidate submission file not found: {file_path}")

        if submission_file not in cached_pred:
            cached_pred[submission_file] = load_prediction_column(file_path, target_col)

        pred = cached_pred[submission_file]
        if not selected:
            selected.append(
                {
                    "candidate": candidate_name,
                    "submission_file": submission_file,
                    "cv_auc": float(row.cv_auc),
                    "selection_reason": "highest_cv_auc",
                    "max_corr_to_selected": None,
                }
            )
            if len(selected) == top_k:
                return selected
            continue

        current_correlations: list[float] = []
        for picked in selected:
            picked_file = str(picked["submission_file"])
            corr = correlation(pred, cached_pred[picked_file])
            current_correlations.append(corr)

        max_corr = max(current_correlations)
        if max_corr <= max_correlation:
            selected.append(
                {
                    "candidate": candidate_name,
                    "submission_file": submission_file,
                    "cv_auc": float(row.cv_auc),
                    "selection_reason": "diversity_gate_passed",
                    "max_corr_to_selected": float(max_corr),
                }
            )
            if len(selected) == top_k:
                return selected

    if len(selected) == top_k:
        return selected

    picked_files = {str(item["submission_file"]) for item in selected}
    for row in candidate_scores.sort_values("cv_auc", ascending=False, kind="mergesort").itertuples(index=False):
        submission_file = str(row.submission_file)
        if submission_file in picked_files:
            continue

        selected.append(
            {
                "candidate": str(row.candidate),
                "submission_file": submission_file,
                "cv_auc": float(row.cv_auc),
                "selection_reason": "fallback_high_cv",
                "max_corr_to_selected": None,
            }
        )
        if len(selected) == top_k:
            break

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online sprint helper for PS-S6E2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-submission")
    validate_parser.add_argument("--submission-path", required=True)
    validate_parser.add_argument("--sample-sub-path", required=True)
    validate_parser.add_argument("--id-col", default="id")
    validate_parser.add_argument("--target-col", default="Heart Disease")

    pick_parser = subparsers.add_parser("pick-daily")
    pick_parser.add_argument("--candidate-scores-path", required=True)
    pick_parser.add_argument("--submission-dir", required=True)
    pick_parser.add_argument("--target-col", default="Heart Disease")
    pick_parser.add_argument("--top-k", type=int, default=2)
    pick_parser.add_argument("--max-correlation", type=float, default=0.998)
    pick_parser.add_argument("--output-path", default="")

    return parser.parse_args()


def run_validate_submission(args: argparse.Namespace) -> None:
    validate_submission_csv(
        submission_path=Path(args.submission_path),
        sample_submission_path=Path(args.sample_sub_path),
        id_col=args.id_col,
        target_col=args.target_col,
    )
    print(f"Submission validation passed: {args.submission_path}")


def run_pick_daily(args: argparse.Namespace) -> None:
    candidate_scores = load_candidate_scores(Path(args.candidate_scores_path))
    selected = select_submission_candidates(
        candidate_scores=candidate_scores,
        submission_dir=Path(args.submission_dir),
        target_col=args.target_col,
        top_k=args.top_k,
        max_correlation=args.max_correlation,
    )

    payload = {
        "top_k": args.top_k,
        "max_correlation": args.max_correlation,
        "selected": selected,
    }

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved selection summary: {output_path}")

    print(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "validate-submission":
        run_validate_submission(args)
    elif args.command == "pick-daily":
        run_pick_daily(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
