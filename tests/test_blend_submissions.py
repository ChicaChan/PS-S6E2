from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from src.blend_submissions import (
    blend_predictions,
    parse_weights,
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
