from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import pytest

from src.train_lgbm_baseline import (
    TARGET_MAP,
    build_submission,
    encode_target,
    split_features,
)


def test_encode_target_success():
    y = pd.Series(["Absence", "Presence", "Absence"])
    encoded = encode_target(y)
    assert encoded.tolist() == [TARGET_MAP["Absence"], TARGET_MAP["Presence"], TARGET_MAP["Absence"]]
    assert str(encoded.dtype) == "int8"


def test_encode_target_unknown_label_raises():
    y = pd.Series(["Absence", "Unknown"])
    with pytest.raises(ValueError, match="Unknown target labels"):
        encode_target(y)


def test_split_features_drop_id():
    data = pd.DataFrame(
        {
            "id": [1, 2],
            "f1": [10, 20],
            "Heart Disease": ["Absence", "Presence"],
        }
    )
    X, y = split_features(data, target_col="Heart Disease", id_col="id", drop_id=True)
    assert list(X.columns) == ["f1"]
    assert y.tolist() == [0, 1]


def test_build_submission_success():
    test_data = pd.DataFrame({"id": [101, 102], "f1": [0.1, 0.2]})
    pred = pd.Series([0.3, 0.8])
    sub = build_submission(test_data, pred, id_col="id", target_col="Heart Disease")
    assert list(sub.columns) == ["id", "Heart Disease"]
    assert sub["Heart Disease"].tolist() == [0.3, 0.8]


def test_build_submission_errors():
    test_data = pd.DataFrame({"x": [1]})
    pred = pd.Series([0.5])
    with pytest.raises(ValueError, match="Missing id column"):
        build_submission(test_data, pred, id_col="id", target_col="Heart Disease")

    test_data_ok = pd.DataFrame({"id": [1, 2]})
    pred_short = pd.Series([0.2])
    with pytest.raises(ValueError, match="Prediction size does not match"):
        build_submission(test_data_ok, pred_short, id_col="id", target_col="Heart Disease")

