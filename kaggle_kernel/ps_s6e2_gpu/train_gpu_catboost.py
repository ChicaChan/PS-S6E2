from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

TARGET_COL = "Heart Disease"
ID_COL = "id"
TARGET_MAP = {"Absence": 0, "Presence": 1}


@dataclass
class BranchConfig:
    seeds: list[int]
    params: dict[str, Any]


@dataclass
class BranchResult:
    name: str
    oof_pred: np.ndarray
    test_pred: np.ndarray
    cv_auc: float
    seed_aucs: list[float]
    meta: dict[str, Any]


@dataclass
class CandidateResult:
    name: str
    oof_pred: np.ndarray
    test_pred: np.ndarray
    cv_auc: float
    details: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PS-S6E2 cloud rank push trainer")
    parser.add_argument("--exp-name", default="cloud_round_v13")
    parser.add_argument("--preset", choices=("fast", "full"), default="full")
    parser.add_argument("--lgb-seeds", default="42,2024,3407")
    parser.add_argument("--cat-seeds", default="42,2024")
    parser.add_argument("--xgb-seeds", default="42")
    parser.add_argument("--cat-l2-grid", default="2.0,3.0,5.0")
    parser.add_argument("--disable-xgb", action="store_true")
    parser.add_argument("--blend-mode", choices=("weighted", "rank", "auto"), default="auto")
    parser.add_argument("--weight-step", type=float, default=0.025)
    parser.add_argument("--min-blend-weight", type=float, default=0.10)
    parser.add_argument("--meta-seed-a", type=int, default=73)
    parser.add_argument("--meta-seed-b", type=int, default=131)
    parser.add_argument("--meta-test-size", type=float, default=0.35)
    parser.add_argument("--meta-min-gain", type=float, default=0.00001)
    parser.add_argument("--disable-target-stats", action="store_true")
    parser.add_argument("--target-stats-folds", type=int, default=5)
    parser.add_argument("--target-stats-smoothing", type=float, default=20.0)
    parser.add_argument("--target-stats-max-cardinality", type=int, default=256)
    parser.add_argument("--target-stats-interactions", type=int, default=4)
    parser.add_argument("--corr-prune-threshold", type=float, default=0.9995)
    parser.add_argument("--train-path", default="/kaggle/input/playground-series-s6e2/train.csv")
    parser.add_argument("--test-path", default="/kaggle/input/playground-series-s6e2/test.csv")
    parser.add_argument("--sample-sub-path", default="/kaggle/input/playground-series-s6e2/sample_submission.csv")
    parser.add_argument("--output-dir", default="/kaggle/working")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=42)
    return parser.parse_args()


def parse_seed_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Seeds list cannot be empty")
    return list(dict.fromkeys(int(item) for item in values))


def parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Float list cannot be empty")
    return list(dict.fromkeys(float(item) for item in values))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def encode_target(target: pd.Series) -> pd.Series:
    encoded = target.map(TARGET_MAP)
    if encoded.isna().any():
        unknown = sorted(target[encoded.isna()].unique().tolist())
        raise ValueError(f"Unknown target labels: {unknown}")
    return encoded.astype("int8")


def sanitize_feature_names(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: col.replace(" ", "_") for col in frame.columns}
    return frame.rename(columns=renamed)


def build_feature_frames(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [col for col in train_df.columns if col not in (TARGET_COL, ID_COL)]
    x_train = sanitize_feature_names(train_df[feature_cols].copy())
    x_test = sanitize_feature_names(test_df[feature_cols].copy())
    return x_train, x_test


def _safe_abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    if np.isnan(value):
        return 0.0
    return abs(value)


def build_target_stat_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    cv_seed: int,
    smoothing: float,
    max_cardinality: int,
    top_k_interactions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if cv_folds < 2:
        raise ValueError(f"target_stats_folds must be >= 2, got {cv_folds}")
    if smoothing < 0.0:
        raise ValueError(f"target_stats_smoothing must be >= 0, got {smoothing}")
    if max_cardinality < 2:
        raise ValueError(f"target_stats_max_cardinality must be >= 2, got {max_cardinality}")
    if top_k_interactions < 0:
        raise ValueError(f"target_stats_interactions must be >= 0, got {top_k_interactions}")

    selected_cols = [
        col
        for col in x_train.columns
        if int(x_train[col].nunique(dropna=False)) <= max_cardinality
    ]
    if not selected_cols:
        empty_train = pd.DataFrame(index=x_train.index)
        empty_test = pd.DataFrame(index=x_test.index)
        return (
            empty_train,
            empty_test,
            {
                "enabled": True,
                "selected_columns": [],
                "interaction_features": [],
                "generated_feature_count": 0,
                "message": "no columns passed cardinality filter",
            },
        )

    global_mean = float(y.mean())
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cv_seed)

    train_stats: dict[str, np.ndarray] = {}
    test_stats: dict[str, np.ndarray] = {}

    for col in selected_cols:
        train_col = x_train[col]
        oof_te = np.full(len(x_train), global_mean, dtype=np.float64)

        for fold_train_idx, fold_valid_idx in splitter.split(x_train, y):
            fold_train_col = train_col.iloc[fold_train_idx]
            fold_train_target = y.iloc[fold_train_idx]

            group_sum = fold_train_target.groupby(fold_train_col).sum()
            group_count = fold_train_target.groupby(fold_train_col).count()
            mapping = ((group_sum + smoothing * global_mean) / (group_count + smoothing)).to_dict()

            oof_te[fold_valid_idx] = (
                train_col.iloc[fold_valid_idx].map(mapping).fillna(global_mean).to_numpy(dtype=np.float64)
            )

        full_group_sum = y.groupby(train_col).sum()
        full_group_count = y.groupby(train_col).count()
        full_mapping = ((full_group_sum + smoothing * global_mean) / (full_group_count + smoothing)).to_dict()
        test_te = x_test[col].map(full_mapping).fillna(global_mean).to_numpy(dtype=np.float64)

        freq_mapping = train_col.value_counts(normalize=True).to_dict()
        train_freq = train_col.map(freq_mapping).fillna(0.0).to_numpy(dtype=np.float64)
        test_freq = x_test[col].map(freq_mapping).fillna(0.0).to_numpy(dtype=np.float64)

        train_stats[f"te_{col}"] = oof_te
        test_stats[f"te_{col}"] = test_te
        train_stats[f"freq_{col}"] = train_freq
        test_stats[f"freq_{col}"] = test_freq

    stat_train = pd.DataFrame(train_stats, index=x_train.index)
    stat_test = pd.DataFrame(test_stats, index=x_test.index)

    interaction_features: list[str] = []
    if top_k_interactions > 0:
        ranked_cols = sorted(
            [f"te_{col}" for col in selected_cols],
            key=lambda col: _safe_abs_corr(stat_train[col].to_numpy(dtype=np.float64), y.to_numpy(dtype=np.float64)),
            reverse=True,
        )
        top_cols = ranked_cols[: max(2, min(len(ranked_cols), top_k_interactions))]
        pair_limit = min(top_k_interactions, len(top_cols) * (len(top_cols) - 1) // 2)
        for col_a, col_b in list(combinations(top_cols, 2))[:pair_limit]:
            feature_name = f"{col_a}_x_{col_b}"
            stat_train[feature_name] = stat_train[col_a] * stat_train[col_b]
            stat_test[feature_name] = stat_test[col_a] * stat_test[col_b]
            interaction_features.append(feature_name)

    meta = {
        "enabled": True,
        "selected_columns": selected_cols,
        "interaction_features": interaction_features,
        "generated_feature_count": int(stat_train.shape[1]),
        "smoothing": float(smoothing),
        "folds": int(cv_folds),
    }
    return stat_train, stat_test, meta


def build_branch_configs(args: argparse.Namespace) -> dict[str, BranchConfig]:
    if args.preset == "fast":
        lgb_params = {
            "objective": "binary",
            "learning_rate": 0.05,
            "n_estimators": 600,
            "num_leaves": 63,
            "min_child_samples": 30,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "n_jobs": -1,
            "verbose": -1,
        }
        cat_params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "learning_rate": 0.05,
            "iterations": 1400,
            "depth": 8,
            "random_strength": 0.1,
            "verbose": False,
            "allow_writing_files": False,
        }
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": 0.05,
            "n_estimators": 1000,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "tree_method": "hist",
        }
    else:
        lgb_params = {
            "objective": "binary",
            "learning_rate": 0.045,
            "n_estimators": 950,
            "num_leaves": 63,
            "min_child_samples": 30,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "n_jobs": -1,
            "verbose": -1,
        }
        cat_params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "learning_rate": 0.04,
            "iterations": 1900,
            "depth": 8,
            "random_strength": 0.1,
            "verbose": False,
            "allow_writing_files": False,
        }
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": 0.045,
            "n_estimators": 1500,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "tree_method": "hist",
        }

    return {
        "lgbm": BranchConfig(seeds=parse_seed_list(args.lgb_seeds), params=lgb_params),
        "catboost": BranchConfig(seeds=parse_seed_list(args.cat_seeds), params=cat_params),
        "xgb": BranchConfig(seeds=parse_seed_list(args.xgb_seeds), params=xgb_params),
    }


def train_lgbm_fold(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    seed: int,
    params: dict[str, Any],
):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("lightgbm is required for lgbm branch training") from exc

    config = dict(params)
    config["random_state"] = seed
    model = lgb.LGBMClassifier(**config)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=120, verbose=False)],
    )
    return model


def train_catboost_fold(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    seed: int,
    params: dict[str, Any],
    use_gpu: bool,
):
    from catboost import CatBoostClassifier

    config = dict(params)
    config["random_seed"] = seed
    if use_gpu:
        config["task_type"] = "GPU"

    model = CatBoostClassifier(**config)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        use_best_model=True,
        early_stopping_rounds=120,
        verbose=False,
    )
    return model


def train_xgb_fold(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    seed: int,
    params: dict[str, Any],
):
    from xgboost import XGBClassifier

    config = dict(params)
    config["random_state"] = seed
    model = XGBClassifier(**config)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    return model


def train_branch(
    name: str,
    cfg: BranchConfig,
    x: pd.DataFrame,
    y: pd.Series,
    x_test: pd.DataFrame,
    cv_folds: int,
    cv_seed: int,
    use_gpu_for_catboost: bool = True,
    meta: dict[str, Any] | None = None,
) -> BranchResult:
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cv_seed)
    seed_oof_preds: list[np.ndarray] = []
    seed_test_preds: list[np.ndarray] = []
    seed_aucs: list[float] = []

    for seed in cfg.seeds:
        oof_pred = np.zeros(len(x), dtype=np.float64)
        test_pred = np.zeros(len(x_test), dtype=np.float64)

        for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
            x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            if name == "lgbm":
                model = train_lgbm_fold(x_train, y_train, x_valid, y_valid, seed + fold_id, cfg.params)
            elif name == "catboost":
                model = train_catboost_fold(
                    x_train,
                    y_train,
                    x_valid,
                    y_valid,
                    seed + fold_id,
                    cfg.params,
                    use_gpu_for_catboost,
                )
            elif name == "xgb":
                model = train_xgb_fold(x_train, y_train, x_valid, y_valid, seed + fold_id, cfg.params)
            else:
                raise ValueError(f"Unsupported branch: {name}")

            valid_pred = model.predict_proba(x_valid)[:, 1]
            test_fold_pred = model.predict_proba(x_test)[:, 1]

            oof_pred[valid_idx] = valid_pred
            test_pred += test_fold_pred / cv_folds

            fold_auc = roc_auc_score(y_valid, valid_pred)
            print(f"[{name}] seed={seed} fold={fold_id} auc={fold_auc:.6f}")

        seed_auc = float(roc_auc_score(y, oof_pred))
        print(f"[{name}] seed={seed} cv_auc={seed_auc:.6f}")

        seed_aucs.append(seed_auc)
        seed_oof_preds.append(oof_pred)
        seed_test_preds.append(test_pred)

    blended_oof = np.mean(np.vstack(seed_oof_preds), axis=0)
    blended_test = np.mean(np.vstack(seed_test_preds), axis=0)
    cv_auc = float(roc_auc_score(y, blended_oof))

    return BranchResult(
        name=name,
        oof_pred=blended_oof,
        test_pred=blended_test,
        cv_auc=cv_auc,
        seed_aucs=seed_aucs,
        meta=meta or {},
    )


def weighted(preds: list[np.ndarray], weights: list[float]) -> np.ndarray:
    arr = np.vstack(preds)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return np.average(arr, axis=0, weights=w)


def rank_mean(preds: list[np.ndarray]) -> np.ndarray:
    ranked = [pd.Series(pred).rank(method="average", pct=True).to_numpy() for pred in preds]
    return np.mean(np.vstack(ranked), axis=0)


def rank_array(pred: np.ndarray) -> np.ndarray:
    return pd.Series(pred).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def search_best_lgb_cat_weight(
    y: pd.Series,
    lgb_oof: np.ndarray,
    cat_oof: np.ndarray,
    lgb_test: np.ndarray,
    cat_test: np.ndarray,
    step: float,
) -> CandidateResult:
    best_auc = -1.0
    best_lgb_weight = 0.5
    values = np.arange(0.25, 0.751, step)
    for lgb_weight in values:
        cat_weight = 1.0 - float(lgb_weight)
        pred = lgb_weight * lgb_oof + cat_weight * cat_oof
        auc = float(roc_auc_score(y, pred))
        if auc > best_auc:
            best_auc = auc
            best_lgb_weight = float(lgb_weight)

    tuned_oof = best_lgb_weight * lgb_oof + (1.0 - best_lgb_weight) * cat_oof
    tuned_test = best_lgb_weight * lgb_test + (1.0 - best_lgb_weight) * cat_test
    return CandidateResult(
        name="tuned_lgb_cat",
        oof_pred=tuned_oof,
        test_pred=tuned_test,
        cv_auc=float(roc_auc_score(y, tuned_oof)),
        details={"lgb_weight": best_lgb_weight, "cat_weight": 1.0 - best_lgb_weight},
    )


def search_best_lgb_cat_xgb_weight(
    y: pd.Series,
    lgb_oof: np.ndarray,
    cat_oof: np.ndarray,
    xgb_oof: np.ndarray,
    lgb_test: np.ndarray,
    cat_test: np.ndarray,
    xgb_test: np.ndarray,
    step: float,
    min_weight: float,
) -> CandidateResult:
    if not (0.0 < min_weight < 0.5):
        raise ValueError(f"min_weight must be in (0, 0.5), got {min_weight}")

    values = np.arange(min_weight, 1.0 - min_weight + 1e-9, step)
    best_auc = -1.0
    best_weights = (0.35, 0.55, 0.10)

    for lgb_weight in values:
        for cat_weight in values:
            xgb_weight = 1.0 - float(lgb_weight) - float(cat_weight)
            if xgb_weight < min_weight:
                continue

            pred = float(lgb_weight) * lgb_oof + float(cat_weight) * cat_oof + float(xgb_weight) * xgb_oof
            auc = float(roc_auc_score(y, pred))
            if auc > best_auc:
                best_auc = auc
                best_weights = (float(lgb_weight), float(cat_weight), float(xgb_weight))

    lgb_w, cat_w, xgb_w = best_weights
    tuned_oof = lgb_w * lgb_oof + cat_w * cat_oof + xgb_w * xgb_oof
    tuned_test = lgb_w * lgb_test + cat_w * cat_test + xgb_w * xgb_test
    return CandidateResult(
        name="tuned_lgb_cat_xgb",
        oof_pred=tuned_oof,
        test_pred=tuned_test,
        cv_auc=float(roc_auc_score(y, tuned_oof)),
        details={
            "weights": {"lgbm": lgb_w, "catboost": cat_w, "xgb": xgb_w},
            "search_step": step,
            "min_weight": min_weight,
        },
    )


def search_best_lgb_cat_rank_weight(
    y: pd.Series,
    lgb_oof: np.ndarray,
    cat_oof: np.ndarray,
    lgb_test: np.ndarray,
    cat_test: np.ndarray,
    step: float,
) -> CandidateResult:
    result = search_best_lgb_cat_weight(
        y=y,
        lgb_oof=rank_array(lgb_oof),
        cat_oof=rank_array(cat_oof),
        lgb_test=rank_array(lgb_test),
        cat_test=rank_array(cat_test),
        step=step,
    )
    return CandidateResult(
        name="tuned_rank_lgb_cat",
        oof_pred=result.oof_pred,
        test_pred=result.test_pred,
        cv_auc=result.cv_auc,
        details={
            "method": "rank_weighted",
            "weights": {"lgbm": result.details["lgb_weight"], "catboost": result.details["cat_weight"]},
            "search_step": step,
        },
    )


def search_best_lgb_cat_xgb_rank_weight(
    y: pd.Series,
    lgb_oof: np.ndarray,
    cat_oof: np.ndarray,
    xgb_oof: np.ndarray,
    lgb_test: np.ndarray,
    cat_test: np.ndarray,
    xgb_test: np.ndarray,
    step: float,
    min_weight: float,
) -> CandidateResult:
    result = search_best_lgb_cat_xgb_weight(
        y=y,
        lgb_oof=rank_array(lgb_oof),
        cat_oof=rank_array(cat_oof),
        xgb_oof=rank_array(xgb_oof),
        lgb_test=rank_array(lgb_test),
        cat_test=rank_array(cat_test),
        xgb_test=rank_array(xgb_test),
        step=step,
        min_weight=min_weight,
    )
    return CandidateResult(
        name="tuned_rank_lgb_cat_xgb",
        oof_pred=result.oof_pred,
        test_pred=result.test_pred,
        cv_auc=result.cv_auc,
        details={
            "method": "rank_weighted",
            "weights": result.details["weights"],
            "search_step": step,
            "min_weight": min_weight,
        },
    )


def search_best_weight_rank_hybrid(
    y: pd.Series,
    weighted_candidate: CandidateResult,
    rank_candidate: CandidateResult,
    alpha_step: float,
    name: str,
) -> CandidateResult:
    if not (0.0 < alpha_step <= 1.0):
        raise ValueError(f"alpha_step must be in (0, 1], got {alpha_step}")

    values = np.arange(0.0, 1.0 + 1e-9, alpha_step)
    best_alpha = 0.5
    best_auc = -1.0
    for alpha in values:
        pred = float(alpha) * weighted_candidate.oof_pred + (1.0 - float(alpha)) * rank_candidate.oof_pred
        auc = float(roc_auc_score(y, pred))
        if auc > best_auc:
            best_auc = auc
            best_alpha = float(alpha)

    blended_oof = best_alpha * weighted_candidate.oof_pred + (1.0 - best_alpha) * rank_candidate.oof_pred
    blended_test = best_alpha * weighted_candidate.test_pred + (1.0 - best_alpha) * rank_candidate.test_pred
    return CandidateResult(
        name=name,
        oof_pred=blended_oof,
        test_pred=blended_test,
        cv_auc=float(roc_auc_score(y, blended_oof)),
        details={
            "method": "hybrid_weight_rank",
            "weighted_source": weighted_candidate.name,
            "rank_source": rank_candidate.name,
            "weighted_alpha": best_alpha,
            "rank_alpha": 1.0 - best_alpha,
            "alpha_step": alpha_step,
        },
    )


def build_corr_prune_quantile_candidate(
    y: pd.Series,
    candidates: list[CandidateResult],
    quantile: float,
    max_corr: float,
    name: str,
) -> CandidateResult:
    if len(candidates) < 2:
        raise ValueError("At least two candidates are required for corr-prune quantile blend")
    if not (0.0 <= quantile <= 1.0):
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    if not (0.9 <= max_corr < 1.0):
        raise ValueError(f"max_corr must be in [0.9, 1.0), got {max_corr}")

    oof_matrix = np.column_stack([item.oof_pred for item in candidates])
    test_matrix = np.column_stack([item.test_pred for item in candidates])
    corr = np.corrcoef(oof_matrix.T)

    keep_indices: list[int] = []
    for idx in range(oof_matrix.shape[1]):
        if not keep_indices:
            keep_indices.append(idx)
            continue

        keep = True
        for kept_idx in keep_indices:
            value = float(corr[idx, kept_idx])
            if not np.isnan(value) and abs(value) >= max_corr:
                keep = False
                break
        if keep:
            keep_indices.append(idx)

    if not keep_indices:
        keep_indices = [0]

    pruned_oof = oof_matrix[:, keep_indices]
    pruned_test = test_matrix[:, keep_indices]
    blended_oof = np.quantile(pruned_oof, q=quantile, axis=1)
    blended_test = np.quantile(pruned_test, q=quantile, axis=1)

    return CandidateResult(
        name=name,
        oof_pred=blended_oof,
        test_pred=blended_test,
        cv_auc=float(roc_auc_score(y, blended_oof)),
        details={
            "method": "corr_prune_quantile",
            "quantile": float(quantile),
            "max_corr": float(max_corr),
            "n_input_candidates": len(candidates),
            "kept_candidates": [candidates[idx].name for idx in keep_indices],
        },
    )


def build_meta_splits(y: pd.Series, seed_a: int, seed_b: int, test_size: float) -> list[tuple[np.ndarray, np.ndarray]]:
    if not (0.1 <= test_size <= 0.5):
        raise ValueError(f"meta_test_size must be in [0.1, 0.5], got {test_size}")

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for seed in (seed_a, seed_b):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, valid_idx = next(splitter.split(np.zeros(len(y)), y))
        splits.append((train_idx, valid_idx))
    return splits


def attach_meta_stability(
    y: pd.Series,
    baseline_oof: np.ndarray,
    candidates: list[CandidateResult],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    for candidate in candidates:
        gains: list[float] = []
        candidate_aucs: list[float] = []
        baseline_aucs: list[float] = []

        for _, valid_idx in splits:
            y_valid = y.iloc[valid_idx]
            candidate_auc = float(roc_auc_score(y_valid, candidate.oof_pred[valid_idx]))
            baseline_auc = float(roc_auc_score(y_valid, baseline_oof[valid_idx]))
            gains.append(candidate_auc - baseline_auc)
            candidate_aucs.append(candidate_auc)
            baseline_aucs.append(baseline_auc)

        candidate.details["meta_gate"] = {
            "gains": gains,
            "candidate_aucs": candidate_aucs,
            "baseline_aucs": baseline_aucs,
            "mean_gain": float(np.mean(gains)),
            "min_gain": float(np.min(gains)),
            "max_gain": float(np.max(gains)),
        }


def filter_candidates_by_meta(
    candidates: list[CandidateResult],
    baseline_name: str,
    min_gain: float,
) -> list[CandidateResult]:
    baseline_candidates = [candidate for candidate in candidates if candidate.name == baseline_name]
    if not baseline_candidates:
        raise ValueError(f"Baseline candidate not found: {baseline_name}")

    selected: list[CandidateResult] = [baseline_candidates[0]]
    for candidate in candidates:
        if candidate.name == baseline_name:
            continue
        gate = candidate.details.get("meta_gate", {})
        gate_min = float(gate.get("min_gain", -1.0))
        gate_mean = float(gate.get("mean_gain", -1.0))
        if gate_min >= min_gain and gate_mean >= 0.0:
            selected.append(candidate)

    if len(selected) == 1 and len(candidates) > 1:
        fallback = max(
            [candidate for candidate in candidates if candidate.name != baseline_name],
            key=lambda item: float(item.details.get("meta_gate", {}).get("mean_gain", -99.0)),
        )
        selected.append(fallback)
        fallback.details["selection_fallback"] = "all_candidates_failed_meta_gate"

    return selected


def save_submission(sample_sub: pd.DataFrame, pred: np.ndarray, output_path: Path) -> None:
    submission = sample_sub.copy()
    submission[TARGET_COL] = pred
    submission.to_csv(output_path, index=False)
    print(f"Saved submission: {output_path}")


def choose_best(candidates: list[CandidateResult]) -> tuple[CandidateResult, str]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            float(item.details.get("meta_gate", {}).get("min_gain", -99.0)),
            float(item.details.get("meta_gate", {}).get("mean_gain", -99.0)),
            item.cv_auc,
        ),
        reverse=True,
    )
    best = ordered[0]
    gate = best.details.get("meta_gate", {})
    reason = (
        f"meta-first selection; min_gain={float(gate.get('min_gain', 0.0)):.6f}, "
        f"mean_gain={float(gate.get('mean_gain', 0.0)):.6f}, cv_auc={best.cv_auc:.6f}"
    )
    return best, reason


def main() -> None:
    args = parse_args()
    set_seed(args.cv_seed)
    started_at = time.perf_counter()

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    sample_sub = pd.read_csv(args.sample_sub_path)

    y = encode_target(train_df[TARGET_COL])
    base_x, base_x_test = build_feature_frames(train_df, test_df)

    target_stats_meta: dict[str, Any] = {
        "enabled": not args.disable_target_stats,
        "selected_columns": [],
        "interaction_features": [],
        "generated_feature_count": 0,
    }
    if args.disable_target_stats:
        x, x_test = base_x, base_x_test
    else:
        stat_train, stat_test, target_stats_meta = build_target_stat_features(
            x_train=base_x,
            x_test=base_x_test,
            y=y,
            cv_folds=args.target_stats_folds,
            cv_seed=args.cv_seed,
            smoothing=args.target_stats_smoothing,
            max_cardinality=args.target_stats_max_cardinality,
            top_k_interactions=args.target_stats_interactions,
        )
        x = pd.concat([base_x, stat_train], axis=1)
        x_test = pd.concat([base_x_test, stat_test], axis=1)
        print(
            "[target_stats] added "
            f"{target_stats_meta.get('generated_feature_count', 0)} features "
            f"from {len(target_stats_meta.get('selected_columns', []))} columns"
        )

    branch_cfg = build_branch_configs(args)

    lgbm_result = train_branch(
        "lgbm",
        branch_cfg["lgbm"],
        x,
        y,
        x_test,
        cv_folds=args.cv_folds,
        cv_seed=args.cv_seed,
    )

    cat_l2_grid = parse_float_list(args.cat_l2_grid)
    catboost_candidates: list[BranchResult] = []
    catboost_device = "gpu"
    for l2 in cat_l2_grid:
        cat_cfg = BranchConfig(
            seeds=branch_cfg["catboost"].seeds,
            params={**branch_cfg["catboost"].params, "l2_leaf_reg": float(l2)},
        )
        try:
            result = train_branch(
                "catboost",
                cat_cfg,
                x,
                y,
                x_test,
                cv_folds=args.cv_folds,
                cv_seed=args.cv_seed,
                use_gpu_for_catboost=True,
                meta={"l2_leaf_reg": float(l2), "device": "gpu"},
            )
        except Exception as exc:
            print(f"[catboost] gpu failed at l2={l2}, fallback to cpu: {exc}")
            catboost_device = "cpu"
            result = train_branch(
                "catboost",
                cat_cfg,
                x,
                y,
                x_test,
                cv_folds=args.cv_folds,
                cv_seed=args.cv_seed,
                use_gpu_for_catboost=False,
                meta={"l2_leaf_reg": float(l2), "device": "cpu"},
            )
        catboost_candidates.append(result)

    catboost_result = max(catboost_candidates, key=lambda item: item.cv_auc)
    print(
        f"[catboost] selected l2={catboost_result.meta.get('l2_leaf_reg')} "
        f"cv_auc={catboost_result.cv_auc:.6f}"
    )

    xgb_result: BranchResult | None = None
    if not args.disable_xgb:
        xgb_result = train_branch(
            "xgb",
            branch_cfg["xgb"],
            x,
            y,
            x_test,
            cv_folds=args.cv_folds,
            cv_seed=args.cv_seed,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_submission(sample_sub, lgbm_result.test_pred, output_dir / "submission_lgbm_3seed.csv")
    save_submission(sample_sub, catboost_result.test_pred, output_dir / "submission_catboost.csv")
    if xgb_result is not None:
        save_submission(sample_sub, xgb_result.test_pred, output_dir / "submission_xgb.csv")

    all_candidates: list[CandidateResult] = []

    safe_oof = weighted([lgbm_result.oof_pred, catboost_result.oof_pred], [0.5, 0.5])
    safe_test = weighted([lgbm_result.test_pred, catboost_result.test_pred], [0.5, 0.5])
    safe = CandidateResult(
        name="safe",
        oof_pred=safe_oof,
        test_pred=safe_test,
        cv_auc=float(roc_auc_score(y, safe_oof)),
        details={"weights": {"lgbm": 0.5, "catboost": 0.5}},
    )
    all_candidates.append(safe)

    tuned_lgb_cat = search_best_lgb_cat_weight(
        y=y,
        lgb_oof=lgbm_result.oof_pred,
        cat_oof=catboost_result.oof_pred,
        lgb_test=lgbm_result.test_pred,
        cat_test=catboost_result.test_pred,
        step=args.weight_step,
    )
    all_candidates.append(tuned_lgb_cat)

    aggressive = search_best_lgb_cat_rank_weight(
        y=y,
        lgb_oof=lgbm_result.oof_pred,
        cat_oof=catboost_result.oof_pred,
        lgb_test=lgbm_result.test_pred,
        cat_test=catboost_result.test_pred,
        step=args.weight_step,
    )
    aggressive.name = "aggressive"
    all_candidates.append(aggressive)

    if xgb_result is not None:
        if xgb_result.cv_auc >= min(lgbm_result.cv_auc, catboost_result.cv_auc) - 0.00015:
            balanced_oof = weighted(
                [lgbm_result.oof_pred, catboost_result.oof_pred, xgb_result.oof_pred],
                [0.35, 0.55, 0.10],
            )
            balanced_test = weighted(
                [lgbm_result.test_pred, catboost_result.test_pred, xgb_result.test_pred],
                [0.35, 0.55, 0.10],
            )
            all_candidates.append(
                CandidateResult(
                    name="balanced",
                    oof_pred=balanced_oof,
                    test_pred=balanced_test,
                    cv_auc=float(roc_auc_score(y, balanced_oof)),
                    details={"weights": {"lgbm": 0.35, "catboost": 0.55, "xgb": 0.10}},
                )
            )

            tuned_three = search_best_lgb_cat_xgb_weight(
                y=y,
                lgb_oof=lgbm_result.oof_pred,
                cat_oof=catboost_result.oof_pred,
                xgb_oof=xgb_result.oof_pred,
                lgb_test=lgbm_result.test_pred,
                cat_test=catboost_result.test_pred,
                xgb_test=xgb_result.test_pred,
                step=args.weight_step,
                min_weight=args.min_blend_weight,
            )
            all_candidates.append(tuned_three)
        else:
            print(
                "[xgb] skipped in candidate generation due to weak cv signal: "
                f"xgb={xgb_result.cv_auc:.6f}, lgb={lgbm_result.cv_auc:.6f}, cat={catboost_result.cv_auc:.6f}"
            )

    robust_pool_names = {"safe", "tuned_lgb_cat", "aggressive", "balanced", "tuned_lgb_cat_xgb"}
    robust_pool = [candidate for candidate in all_candidates if candidate.name in robust_pool_names]
    if len(robust_pool) >= 2:
        robust_median = build_corr_prune_quantile_candidate(
            y=y,
            candidates=robust_pool,
            quantile=0.5,
            max_corr=args.corr_prune_threshold,
            name="robust_median",
        )
        robust_p25 = build_corr_prune_quantile_candidate(
            y=y,
            candidates=robust_pool,
            quantile=0.25,
            max_corr=args.corr_prune_threshold,
            name="robust_p25",
        )
        all_candidates.extend([robust_median, robust_p25])

    if args.blend_mode == "weighted":
        all_candidates = [
            candidate
            for candidate in all_candidates
            if candidate.name in (
                "safe",
                "tuned_lgb_cat",
                "balanced",
                "tuned_lgb_cat_xgb",
                "robust_median",
                "robust_p25",
            )
        ]
    elif args.blend_mode == "rank":
        all_candidates = [candidate for candidate in all_candidates if candidate.name == "aggressive"]

    meta_splits = build_meta_splits(
        y=y,
        seed_a=args.meta_seed_a,
        seed_b=args.meta_seed_b,
        test_size=args.meta_test_size,
    )
    attach_meta_stability(y=y, baseline_oof=safe.oof_pred, candidates=all_candidates, splits=meta_splits)

    selected_candidates = filter_candidates_by_meta(
        candidates=all_candidates,
        baseline_name="safe",
        min_gain=args.meta_min_gain,
    )

    for candidate in all_candidates:
        save_submission(sample_sub, candidate.test_pred, output_dir / f"submission_{candidate.name}.csv")

    best_candidate, selected_reason = choose_best(selected_candidates)
    best_path = output_dir / f"submission_{best_candidate.name}.csv"
    pd.read_csv(best_path).to_csv(output_dir / "submission_best.csv", index=False)

    candidate_scores = pd.DataFrame(
        [
            {
                "candidate": candidate.name,
                "cv_auc": candidate.cv_auc,
                "submission_file": f"submission_{candidate.name}.csv",
                "meta_min_gain": float(candidate.details.get("meta_gate", {}).get("min_gain", 0.0)),
                "meta_mean_gain": float(candidate.details.get("meta_gate", {}).get("mean_gain", 0.0)),
                "details": json.dumps(candidate.details, ensure_ascii=False),
            }
            for candidate in sorted(all_candidates, key=lambda item: item.cv_auc, reverse=True)
        ]
    )
    candidate_scores.to_csv(output_dir / "candidate_scores_cloud.csv", index=False)

    metrics: dict[str, Any] = {
        "exp_name": args.exp_name,
        "preset": args.preset,
        "cv_folds": args.cv_folds,
        "cv_seed": args.cv_seed,
        "blend_mode": args.blend_mode,
        "disable_xgb": args.disable_xgb,
        "weight_step": args.weight_step,
        "min_blend_weight": args.min_blend_weight,
        "cat_l2_grid": cat_l2_grid,
        "meta_seed_a": args.meta_seed_a,
        "meta_seed_b": args.meta_seed_b,
        "meta_test_size": args.meta_test_size,
        "meta_min_gain": args.meta_min_gain,
        "corr_prune_threshold": args.corr_prune_threshold,
        "target_stats": target_stats_meta,
        "lgbm": {
            "cv_auc": lgbm_result.cv_auc,
            "seed_aucs": lgbm_result.seed_aucs,
            "seeds": branch_cfg["lgbm"].seeds,
            "params": branch_cfg["lgbm"].params,
        },
        "catboost": {
            "cv_auc": catboost_result.cv_auc,
            "seed_aucs": catboost_result.seed_aucs,
            "seeds": branch_cfg["catboost"].seeds,
            "params": {**branch_cfg["catboost"].params, "l2_leaf_reg": catboost_result.meta.get("l2_leaf_reg")},
            "device": catboost_device,
            "selected_l2_leaf_reg": catboost_result.meta.get("l2_leaf_reg"),
            "grid_results": [
                {
                    "l2_leaf_reg": item.meta.get("l2_leaf_reg"),
                    "cv_auc": item.cv_auc,
                }
                for item in sorted(catboost_candidates, key=lambda item: item.cv_auc, reverse=True)
            ],
        },
        "candidates": {candidate.name: candidate.cv_auc for candidate in all_candidates},
        "selected_candidates": [candidate.name for candidate in selected_candidates],
        "best_candidate": best_candidate.name,
        "best_candidate_auc": best_candidate.cv_auc,
        "best_submission_file": best_path.name,
        "selected_reason": selected_reason,
        "candidate_public_submit": "",
        "runtime_sec": float(time.perf_counter() - started_at),
    }
    if xgb_result is not None:
        metrics["xgb"] = {
            "cv_auc": xgb_result.cv_auc,
            "seed_aucs": xgb_result.seed_aucs,
            "seeds": branch_cfg["xgb"].seeds,
            "params": branch_cfg["xgb"].params,
        }

    metrics_path = output_dir / "metrics_rank_push_cloud.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Best candidate: {best_candidate.name} | AUC={best_candidate.cv_auc:.6f}")
    print(f"Selected reason: {selected_reason}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
