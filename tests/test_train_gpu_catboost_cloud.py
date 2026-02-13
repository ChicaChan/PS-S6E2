from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util
import sys as _sys

import numpy as np
import pandas as pd
import pytest


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


def test_build_target_stat_features_generates_non_empty_outputs():
    cloud = _load_cloud_module()

    x_train = pd.DataFrame(
        {
            "Sex": [0, 1, 0, 1, 0, 1, 0, 1],
            "Chest_pain_type": [1, 2, 1, 3, 2, 3, 1, 2],
            "BP": [120, 130, 125, 140, 118, 138, 122, 135],
        }
    )
    x_test = pd.DataFrame(
        {
            "Sex": [0, 1, 1],
            "Chest_pain_type": [2, 1, 3],
            "BP": [128, 132, 126],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], dtype="int8")

    stat_train, stat_test, meta = cloud.build_target_stat_features(
        x_train=x_train,
        x_test=x_test,
        y=y,
        cv_folds=4,
        cv_seed=42,
        smoothing=10.0,
        max_cardinality=10,
        top_k_interactions=2,
        bin_count=5,
    )

    assert stat_train.shape[0] == len(x_train)
    assert stat_test.shape[0] == len(x_test)
    assert stat_train.shape[1] >= 6
    assert "te_Sex" in stat_train.columns
    assert "freq_BP" in stat_train.columns
    assert not stat_train.isna().any().any()
    assert not stat_test.isna().any().any()
    assert meta["generated_feature_count"] == stat_train.shape[1]


def test_build_target_stat_features_invalid_fold_raises():
    cloud = _load_cloud_module()

    x_train = pd.DataFrame({"a": [0, 1, 0, 1]})
    x_test = pd.DataFrame({"a": [0, 1]})
    y = pd.Series([0, 1, 0, 1], dtype="int8")

    try:
        cloud.build_target_stat_features(
            x_train=x_train,
            x_test=x_test,
            y=y,
            cv_folds=1,
            cv_seed=42,
            smoothing=1.0,
            max_cardinality=10,
            top_k_interactions=1,
            bin_count=5,
        )
    except ValueError as exc:
        assert "target_stats_folds" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid target stats folds")


def test_build_corr_prune_quantile_candidate_prunes_duplicates():
    cloud = _load_cloud_module()

    y = pd.Series([0, 1, 0, 1, 0, 1], dtype="int8")
    candidate_a = cloud.CandidateResult(
        name="a",
        oof_pred=np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7], dtype=np.float64),
        test_pred=np.array([0.2, 0.8, 0.3], dtype=np.float64),
        cv_auc=0.0,
        details={},
    )
    candidate_b = cloud.CandidateResult(
        name="b",
        oof_pred=np.array([0.1, 0.9000001, 0.2, 0.7999999, 0.3, 0.7000001], dtype=np.float64),
        test_pred=np.array([0.2, 0.8, 0.3], dtype=np.float64),
        cv_auc=0.0,
        details={},
    )
    candidate_c = cloud.CandidateResult(
        name="c",
        oof_pred=np.array([0.3, 0.7, 0.4, 0.6, 0.5, 0.5], dtype=np.float64),
        test_pred=np.array([0.4, 0.6, 0.5], dtype=np.float64),
        cv_auc=0.0,
        details={},
    )

    blended = cloud.build_corr_prune_quantile_candidate(
        y=y,
        candidates=[candidate_a, candidate_b, candidate_c],
        quantile=0.5,
        max_corr=0.999,
        name="robust_median",
    )

    assert blended.name == "robust_median"
    assert blended.details["method"] == "corr_prune_quantile"
    assert len(blended.details["kept_candidates"]) <= 2
    assert blended.oof_pred.shape == y.shape
    assert blended.test_pred.shape == candidate_a.test_pred.shape



def test_parse_int_list_handles_unique_values():
    cloud = _load_cloud_module()
    assert cloud.parse_int_list("4,6,4,8") == [4, 6, 8]


def test_build_target_stat_features_invalid_bin_count_raises():
    cloud = _load_cloud_module()

    x_train = pd.DataFrame({"a": [0, 1, 0, 1]})
    x_test = pd.DataFrame({"a": [0, 1]})
    y = pd.Series([0, 1, 0, 1], dtype="int8")

    with pytest.raises(ValueError, match="target_stats_bin_count"):
        cloud.build_target_stat_features(
            x_train=x_train,
            x_test=x_test,
            y=y,
            cv_folds=2,
            cv_seed=42,
            smoothing=1.0,
            max_cardinality=10,
            top_k_interactions=1,
            bin_count=1,
        )



def test_build_target_stat_features_adds_binned_columns_for_high_cardinality():
    cloud = _load_cloud_module()

    x_train = pd.DataFrame(
        {
            "high_card": [10, 20, 30, 40, 50, 60, 70, 80],
            "low_card": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    x_test = pd.DataFrame(
        {
            "high_card": [15, 35, 55],
            "low_card": [0, 1, 1],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], dtype="int8")

    stat_train, _, meta = cloud.build_target_stat_features(
        x_train=x_train,
        x_test=x_test,
        y=y,
        cv_folds=4,
        cv_seed=42,
        smoothing=5.0,
        max_cardinality=2,
        top_k_interactions=1,
        bin_count=4,
    )

    assert any(column.startswith("te_high_card__bin") for column in stat_train.columns)
    assert len(meta.get("binned_columns", [])) >= 1


def test_select_candidates_with_diversity_limits_high_corr():
    cloud = _load_cloud_module()

    candidate_a = cloud.CandidateResult(
        name="a",
        oof_pred=np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float64),
        test_pred=np.array([0.1, 0.2], dtype=np.float64),
        cv_auc=0.90,
        details={"meta_gate": {"min_gain": 0.0001, "mean_gain": 0.0002}},
    )
    candidate_b = cloud.CandidateResult(
        name="b",
        oof_pred=np.array([0.1, 0.9000001, 0.2, 0.7999999], dtype=np.float64),
        test_pred=np.array([0.1, 0.2], dtype=np.float64),
        cv_auc=0.899,
        details={"meta_gate": {"min_gain": 0.00009, "mean_gain": 0.00019}},
    )
    candidate_c = cloud.CandidateResult(
        name="c",
        oof_pred=np.array([0.6, 0.2, 0.5, 0.3], dtype=np.float64),
        test_pred=np.array([0.9, 0.8], dtype=np.float64),
        cv_auc=0.898,
        details={"meta_gate": {"min_gain": 0.00008, "mean_gain": 0.00018}},
    )

    selected, trace = cloud.select_candidates_with_diversity(
        [candidate_a, candidate_b, candidate_c],
        max_corr=0.999,
        max_count=2,
    )

    assert [item.name for item in selected] == ["a", "c"]
    rejected = [item for item in trace if item["candidate"] == "b" and item["status"] == "rejected"]
    assert rejected


def test_build_robust_pool_only_keeps_base_blend_candidates():
    cloud = _load_cloud_module()

    candidates = [
        cloud.CandidateResult(
            name="safe_ts_i4",
            oof_pred=np.array([0.1, 0.9], dtype=np.float64),
            test_pred=np.array([0.2], dtype=np.float64),
            cv_auc=0.90,
            details={},
        ),
        cloud.CandidateResult(
            name="aggressive_ts_i4",
            oof_pred=np.array([0.12, 0.88], dtype=np.float64),
            test_pred=np.array([0.3], dtype=np.float64),
            cv_auc=0.91,
            details={},
        ),
        cloud.CandidateResult(
            name="robust_q0p5_ts_i4",
            oof_pred=np.array([0.14, 0.86], dtype=np.float64),
            test_pred=np.array([0.4], dtype=np.float64),
            cv_auc=0.89,
            details={},
        ),
        cloud.CandidateResult(
            name="global_robust_q0p33",
            oof_pred=np.array([0.15, 0.85], dtype=np.float64),
            test_pred=np.array([0.5], dtype=np.float64),
            cv_auc=0.88,
            details={},
        ),
    ]

    robust_pool = cloud.build_robust_pool(candidates)
    assert [item.name for item in robust_pool] == ["safe_ts_i4", "aggressive_ts_i4"]


def test_filter_selection_pool_excludes_global_candidates_by_default():
    cloud = _load_cloud_module()

    internal = cloud.CandidateResult(
        name="aggressive_ts_i4",
        oof_pred=np.array([0.1, 0.9], dtype=np.float64),
        test_pred=np.array([0.2], dtype=np.float64),
        cv_auc=0.91,
        details={"scope": "profile"},
    )
    global_candidate = cloud.CandidateResult(
        name="global_robust_q0p5",
        oof_pred=np.array([0.2, 0.8], dtype=np.float64),
        test_pred=np.array([0.3], dtype=np.float64),
        cv_auc=0.92,
        details={"scope": "global_robust"},
    )

    filtered = cloud.filter_selection_pool([internal, global_candidate], allow_global_best=False)
    assert [item.name for item in filtered] == ["aggressive_ts_i4"]

    kept = cloud.filter_selection_pool([internal, global_candidate], allow_global_best=True)
    assert [item.name for item in kept] == ["aggressive_ts_i4", "global_robust_q0p5"]
