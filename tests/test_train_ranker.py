from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from src.train_ranker import (
    REQUIRED_CONFIG_KEYS,
    build_feature_matrices,
    parse_seeds,
    validate_config,
    validate_submission_frame,
)


def test_parse_seeds_dedup_preserves_order():
    seeds = parse_seeds("42, 2024,42, 3407")
    assert seeds == [42, 2024, 3407]


def test_validate_config_missing_keys_raises():
    with pytest.raises(ValueError, match="Config missing required keys"):
        validate_config({"model": "lgbm"}, REQUIRED_CONFIG_KEYS)


def test_build_feature_matrices_base_plus_interactions_adds_columns():
    train_df = pd.DataFrame(
        {
            "id": [1, 2],
            "f1": [1.0, 2.0],
            "f2": [3.0, 4.0],
            "f3": [5.0, 6.0],
            "Heart Disease": ["Absence", "Presence"],
        }
    )
    test_df = pd.DataFrame(
        {
            "id": [10, 11],
            "f1": [7.0, 8.0],
            "f2": [9.0, 10.0],
            "f3": [11.0, 12.0],
        }
    )

    x_train, x_test, feature_cols = build_feature_matrices(
        train_df=train_df,
        test_df=test_df,
        target_col="Heart Disease",
        id_col="id",
        feature_set="base_plus_interactions",
    )

    assert "f1_mul_f2" in x_train.columns
    assert "f1_mul_f3" in x_train.columns
    assert "f2_mul_f3" in x_train.columns
    assert set(x_train.columns) == set(x_test.columns)
    assert "id" not in feature_cols
    assert "Heart Disease" not in feature_cols


def test_validate_submission_frame_errors_for_shape_and_columns():
    sample_sub = pd.DataFrame({"id": [1, 2], "Heart Disease": [0.0, 0.0]})
    bad_cols = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})
    with pytest.raises(ValueError, match="Submission columns mismatch"):
        validate_submission_frame(bad_cols, sample_sub)

    bad_rows = pd.DataFrame({"id": [1], "Heart Disease": [0.1]})
    with pytest.raises(ValueError, match="Submission row mismatch"):
        validate_submission_frame(bad_rows, sample_sub)
