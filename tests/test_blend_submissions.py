from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from src.blend_submissions import (
    build_quantile_candidates,
    blend_predictions,
    parse_weights,
    prune_correlated_columns,
    select_with_group_quota,
    validate_alignment,
)


def test_parse_weights_defaults_to_equal():
    weights = parse_weights("", 3)
    assert np.allclose(weights, np.array([1 / 3, 1 / 3, 1 / 3]))


def test_parse_weights_normalizes_sum():
    weights = parse_weights("2,1", 2)
    assert np.allclose(weights, np.array([2 / 3, 1 / 3]))


def test_parse_weights_mismatch_raises():
    with pytest.raises(ValueError, match="weights count mismatch"):
        parse_weights("1,2", 3)


def test_validate_alignment_detects_id_mismatch():
    a = pd.DataFrame({"id": [1, 2], "Heart Disease": [0.1, 0.2]})
    b = pd.DataFrame({"id": [1, 3], "Heart Disease": [0.3, 0.4]})
    with pytest.raises(ValueError, match="ID column mismatch"):
        validate_alignment([a, b], "id")


def test_blend_predictions_rank_mean_clipped_range():
    a = pd.DataFrame({"id": [1, 2, 3], "Heart Disease": [0.1, 0.9, 0.2]})
    b = pd.DataFrame({"id": [1, 2, 3], "Heart Disease": [0.3, 0.8, 0.4]})
    blended = blend_predictions([a, b], np.array([0.5, 0.5]), "rank_mean", "Heart Disease")
    assert blended.shape == (3,)
    assert np.all((blended >= 0.0) & (blended <= 1.0))


def test_blend_predictions_quantile_method():
    a = pd.DataFrame({"id": [1, 2, 3], "Heart Disease": [0.1, 0.6, 0.2]})
    b = pd.DataFrame({"id": [1, 2, 3], "Heart Disease": [0.3, 0.7, 0.4]})
    c = pd.DataFrame({"id": [1, 2, 3], "Heart Disease": [0.2, 0.8, 0.5]})

    blended = blend_predictions(
        [a, b, c],
        np.array([1 / 3, 1 / 3, 1 / 3]),
        "quantile",
        "Heart Disease",
        quantile=0.5,
    )
    assert np.allclose(blended, np.array([0.2, 0.7, 0.4]))


def test_prune_correlated_columns_removes_near_duplicate():
    matrix = np.column_stack(
        [
            np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
            np.array([0.1, 0.2000001, 0.3000001, 0.4], dtype=np.float64),
            np.array([0.9, 0.1, 0.8, 0.2], dtype=np.float64),
        ]
    )
    keep = prune_correlated_columns(matrix, 0.999)
    assert keep == [0, 2]


def test_blend_predictions_with_corr_prune_keeps_stability():
    a = pd.DataFrame({"id": [1, 2, 3], "Heart Disease": [0.1, 0.2, 0.3]})
    b = pd.DataFrame({"id": [1, 2, 3], "Heart Disease": [0.1, 0.2000001, 0.3000001]})
    c = pd.DataFrame({"id": [1, 2, 3], "Heart Disease": [0.9, 0.1, 0.8]})

    blended = blend_predictions(
        [a, b, c],
        np.array([0.4, 0.4, 0.2]),
        "weighted_mean",
        "Heart Disease",
        corr_prune_threshold=0.999,
    )
    assert blended.shape == (3,)
    assert np.all((blended >= 0.0) & (blended <= 1.0))


def test_select_with_group_quota_keeps_internal_and_external():
    matrix = np.column_stack(
        [
            np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
            np.array([0.1, 0.2000001, 0.3000001, 0.4000001], dtype=np.float64),
            np.array([0.9, 0.1, 0.8, 0.2], dtype=np.float64),
        ]
    )
    keep, details = select_with_group_quota(
        matrix,
        groups=["internal", "internal", "external"],
        max_corr=0.999,
        min_per_group={"internal": 1, "external": 1},
        order_scores=np.array([0.5, 0.4, 0.1], dtype=np.float64),
    )

    assert keep == [0, 2]
    assert details["min_per_group"] == {"internal": 1, "external": 1}


def test_build_quantile_candidates_outputs_multiple_quantiles():
    a = pd.DataFrame({"id": [1, 2, 3], "Heart Disease": [0.1, 0.6, 0.2]})
    b = pd.DataFrame({"id": [1, 2, 3], "Heart Disease": [0.3, 0.7, 0.4]})
    c = pd.DataFrame({"id": [1, 2, 3], "Heart Disease": [0.2, 0.8, 0.5]})

    results = build_quantile_candidates(
        submissions=[a, b, c],
        weights=np.array([1 / 3, 1 / 3, 1 / 3]),
        target_col="Heart Disease",
        quantiles=[0.25, 0.5],
        corr_prune_threshold=0.999,
        input_groups=["internal", "external", "external"],
        min_per_group={"internal": 1, "external": 1},
    )

    assert [item[0] for item in results] == [0.25, 0.5]
    assert results[0][1].shape == (3,)
    assert results[1][2]["method"] == "quantile"


def test_blend_predictions_group_quota_fallback_when_external_missing():
    a = pd.DataFrame({"id": [1, 2], "Heart Disease": [0.1, 0.9]})
    b = pd.DataFrame({"id": [1, 2], "Heart Disease": [0.10001, 0.90001]})

    with pytest.raises(ValueError, match="Group quota infeasible"):
        blend_predictions(
            submissions=[a, b],
            weights=np.array([0.5, 0.5]),
            method="weighted_mean",
            target_col="Heart Disease",
            corr_prune_threshold=0.999,
            input_groups=["internal", "internal"],
            min_per_group={"internal": 1, "external": 1},
        )
