from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util
import sys as _sys

import numpy as np
import pandas as pd


def _load_cloud_module():
    module_path = ROOT / "kaggle_kernel" / "ps_s6e2_gpu" / "train_gpu_catboost.py"
    spec = importlib.util.spec_from_file_location("train_gpu_catboost_cloud", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_search_best_lgb_cat_xgb_weight_returns_valid_candidate():
    cloud = _load_cloud_module()

    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], dtype="int8")
    lgb_oof = np.array([0.15, 0.85, 0.2, 0.8, 0.25, 0.75, 0.3, 0.7], dtype=np.float64)
    cat_oof = np.array([0.1, 0.9, 0.18, 0.82, 0.22, 0.78, 0.29, 0.71], dtype=np.float64)
    xgb_oof = np.array([0.2, 0.8, 0.25, 0.75, 0.28, 0.72, 0.31, 0.69], dtype=np.float64)

    candidate = cloud.search_best_lgb_cat_xgb_weight(
        y=y,
        lgb_oof=lgb_oof,
        cat_oof=cat_oof,
        xgb_oof=xgb_oof,
        lgb_test=lgb_oof,
        cat_test=cat_oof,
        xgb_test=xgb_oof,
        step=0.25,
        min_weight=0.1,
    )

    assert candidate.name == "tuned_lgb_cat_xgb"
    weights = candidate.details["weights"]
    assert set(weights.keys()) == {"lgbm", "catboost", "xgb"}
    assert abs(weights["lgbm"] + weights["catboost"] + weights["xgb"] - 1.0) < 1e-10
    assert all(weight >= 0.1 for weight in weights.values())
    assert candidate.oof_pred.shape == lgb_oof.shape
    assert candidate.test_pred.shape == lgb_oof.shape


def test_search_best_lgb_cat_xgb_weight_invalid_min_weight_raises():
    cloud = _load_cloud_module()

    y = pd.Series([0, 1, 0, 1], dtype="int8")
    base = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float64)

    try:
        cloud.search_best_lgb_cat_xgb_weight(
            y=y,
            lgb_oof=base,
            cat_oof=base,
            xgb_oof=base,
            lgb_test=base,
            cat_test=base,
            xgb_test=base,
            step=0.25,
            min_weight=0.6,
        )
    except ValueError as exc:
        assert "min_weight" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid min_weight")


def test_search_best_lgb_cat_rank_weight_returns_named_candidate():
    cloud = _load_cloud_module()

    y = pd.Series([0, 1, 0, 1, 0, 1], dtype="int8")
    lgb = np.array([0.15, 0.85, 0.2, 0.8, 0.25, 0.75], dtype=np.float64)
    cat = np.array([0.12, 0.88, 0.21, 0.79, 0.23, 0.77], dtype=np.float64)

    candidate = cloud.search_best_lgb_cat_rank_weight(
        y=y,
        lgb_oof=lgb,
        cat_oof=cat,
        lgb_test=lgb,
        cat_test=cat,
        step=0.25,
    )

    assert candidate.name == "tuned_rank_lgb_cat"
    assert candidate.details["method"] == "rank_weighted"
    assert set(candidate.details["weights"].keys()) == {"lgbm", "catboost"}


def test_search_best_weight_rank_hybrid_prefers_better_source_when_step_one():
    cloud = _load_cloud_module()

    y = pd.Series([0, 1, 0, 1, 0, 1], dtype="int8")
    weighted = cloud.CandidateResult(
        name="weighted_source",
        oof_pred=np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6], dtype=np.float64),
        test_pred=np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6], dtype=np.float64),
        cv_auc=0.0,
        details={},
    )
    rank = cloud.CandidateResult(
        name="rank_source",
        oof_pred=np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7], dtype=np.float64),
        test_pred=np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7], dtype=np.float64),
        cv_auc=0.0,
        details={},
    )

    candidate = cloud.search_best_weight_rank_hybrid(
        y=y,
        weighted_candidate=weighted,
        rank_candidate=rank,
        alpha_step=1.0,
        name="hybrid",
    )

    assert candidate.name == "hybrid"
    assert candidate.details["method"] == "hybrid_weight_rank"
    assert candidate.details["weighted_alpha"] in (0.0, 1.0)
    assert candidate.details["rank_alpha"] in (0.0, 1.0)
