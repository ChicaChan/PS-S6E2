from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from src.online_sprint import (
    load_candidate_scores,
    load_prediction_column,
    select_submission_candidates,
    validate_submission_csv,
)


def test_load_candidate_scores_sorts_by_cv_auc(tmp_path: Path):
    path = tmp_path / "candidate_scores_cloud.csv"
    pd.DataFrame(
        {
            "candidate": ["b", "a"],
            "cv_auc": [0.901, 0.905],
            "submission_file": ["b.csv", "a.csv"],
            "details": ["{}", "{}"],
        }
    ).to_csv(path, index=False)

    scores = load_candidate_scores(path)
    assert scores["candidate"].tolist() == ["a", "b"]
    assert scores["cv_auc"].tolist() == [0.905, 0.901]


def test_load_candidate_scores_missing_columns_raises(tmp_path: Path):
    path = tmp_path / "candidate_scores_cloud.csv"
    pd.DataFrame({"candidate": ["a"], "cv_auc": [0.9]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_candidate_scores(path)


def test_validate_submission_csv_happy_path(tmp_path: Path):
    sample = tmp_path / "sample_submission.csv"
    sub = tmp_path / "submission.csv"

    pd.DataFrame({"id": [1, 2], "Heart Disease": [0.0, 0.0]}).to_csv(sample, index=False)
    pd.DataFrame({"id": [1, 2], "Heart Disease": [0.1, 0.9]}).to_csv(sub, index=False)

    validate_submission_csv(
        submission_path=sub,
        sample_submission_path=sample,
        id_col="id",
        target_col="Heart Disease",
    )


def test_validate_submission_csv_out_of_range_raises(tmp_path: Path):
    sample = tmp_path / "sample_submission.csv"
    sub = tmp_path / "submission.csv"

    pd.DataFrame({"id": [1, 2], "Heart Disease": [0.0, 0.0]}).to_csv(sample, index=False)
    pd.DataFrame({"id": [1, 2], "Heart Disease": [0.1, 1.2]}).to_csv(sub, index=False)

    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        validate_submission_csv(
            submission_path=sub,
            sample_submission_path=sample,
            id_col="id",
            target_col="Heart Disease",
        )


def test_load_prediction_column_non_numeric_raises(tmp_path: Path):
    path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "Heart Disease": [0.1, "bad"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="non-numeric"):
        load_prediction_column(path, "Heart Disease")


def test_select_submission_candidates_prefers_diverse_predictions(tmp_path: Path):
    submissions = tmp_path / "submissions"
    submissions.mkdir(parents=True, exist_ok=True)

    ids = [1, 2, 3, 4]
    pd.DataFrame({"id": ids, "Heart Disease": [0.1, 0.2, 0.3, 0.4]}).to_csv(submissions / "a.csv", index=False)
    pd.DataFrame({"id": ids, "Heart Disease": [0.1001, 0.2001, 0.3001, 0.4001]}).to_csv(
        submissions / "b.csv", index=False
    )
    pd.DataFrame({"id": ids, "Heart Disease": [0.9, 0.2, 0.8, 0.1]}).to_csv(submissions / "c.csv", index=False)

    scores = pd.DataFrame(
        {
            "candidate": ["A", "B", "C"],
            "cv_auc": [0.905, 0.904, 0.903],
            "submission_file": ["a.csv", "b.csv", "c.csv"],
        }
    )

    selected = select_submission_candidates(
        candidate_scores=scores,
        submission_dir=submissions,
        target_col="Heart Disease",
        top_k=2,
        max_correlation=0.998,
    )

    assert [item["candidate"] for item in selected] == ["A", "C"]
    assert selected[0]["selection_reason"] == "highest_cv_auc"
    assert selected[1]["selection_reason"] == "diversity_gate_passed"


def test_select_submission_candidates_fallback_when_all_similar(tmp_path: Path):
    submissions = tmp_path / "submissions"
    submissions.mkdir(parents=True, exist_ok=True)

    ids = [1, 2, 3]
    for name in ("a.csv", "b.csv", "c.csv"):
        pd.DataFrame({"id": ids, "Heart Disease": [0.1, 0.5, 0.9]}).to_csv(submissions / name, index=False)

    scores = pd.DataFrame(
        {
            "candidate": ["A", "B", "C"],
            "cv_auc": [0.905, 0.904, 0.903],
            "submission_file": ["a.csv", "b.csv", "c.csv"],
        }
    )

    selected = select_submission_candidates(
        candidate_scores=scores,
        submission_dir=submissions,
        target_col="Heart Disease",
        top_k=2,
        max_correlation=0.5,
    )

    assert [item["candidate"] for item in selected] == ["A", "B"]
    assert selected[1]["selection_reason"] == "fallback_high_cv"


def test_select_submission_candidates_invalid_top_k_raises(tmp_path: Path):
    submissions = tmp_path / "submissions"
    submissions.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [1], "Heart Disease": [0.5]}).to_csv(submissions / "a.csv", index=False)

    scores = pd.DataFrame(
        {
            "candidate": ["A"],
            "cv_auc": [0.9],
            "submission_file": ["a.csv"],
        }
    )

    with pytest.raises(ValueError, match="top_k"):
        select_submission_candidates(
            candidate_scores=scores,
            submission_dir=submissions,
            target_col="Heart Disease",
            top_k=0,
        )
