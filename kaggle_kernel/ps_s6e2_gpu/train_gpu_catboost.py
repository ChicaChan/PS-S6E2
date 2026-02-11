from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

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
    parser.add_argument("--exp-name", default="cloud_round_b")
    parser.add_argument("--preset", choices=("fast", "full"), default="full")
    parser.add_argument("--lgb-seeds", default="42,2024,3407,777")
    parser.add_argument("--cat-seeds", default="42")
    parser.add_argument("--xgb-seeds", default="42")
    parser.add_argument("--cat-l2-grid", default="2.0,3.0,5.0")
    parser.add_argument("--disable-xgb", action="store_true", default=False)
    parser.add_argument("--blend-mode", choices=("weighted", "rank", "auto"), default="auto")
    parser.add_argument("--weight-step", type=float, default=0.025)
    parser.add_argument("--hybrid-alpha-step", type=float, default=0.05)
    parser.add_argument("--min-blend-weight", type=float, default=0.05)
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


def build_branch_configs(args: argparse.Namespace) -> dict[str, BranchConfig]:
    if args.preset == "fast":
        lgb_params = {
            "objective": "binary",
            "learning_rate": 0.05,
            "n_estimators": 550,
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
            "iterations": 1300,
            "depth": 8,
            "random_strength": 0.1,
            "verbose": False,
            "allow_writing_files": False,
        }
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": 0.05,
            "n_estimators": 950,
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


def save_submission(sample_sub: pd.DataFrame, pred: np.ndarray, output_path: Path) -> None:
    submission = sample_sub.copy()
    submission[TARGET_COL] = pred
    submission.to_csv(output_path, index=False)
    print(f"Saved submission: {output_path}")


def choose_best(candidates: list[CandidateResult]) -> tuple[CandidateResult, str]:
    ordered = sorted(candidates, key=lambda item: item.cv_auc, reverse=True)
    top = ordered[0]
    if len(ordered) > 1:
        gap = ordered[0].cv_auc - ordered[1].cv_auc
        return top, f"max cv auc {top.cv_auc:.6f}; gap to #2 is {gap:.8f}"
    return top, f"single candidate {top.name} with cv auc {top.cv_auc:.6f}"


def main() -> None:
    args = parse_args()
    set_seed(args.cv_seed)
    started_at = time.perf_counter()

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    sample_sub = pd.read_csv(args.sample_sub_path)

    y = encode_target(train_df[TARGET_COL])
    x, x_test = build_feature_frames(train_df, test_df)
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

    candidates: list[CandidateResult] = []

    safe_oof = weighted([lgbm_result.oof_pred, catboost_result.oof_pred], [0.5, 0.5])
    safe_test = weighted([lgbm_result.test_pred, catboost_result.test_pred], [0.5, 0.5])
    candidates.append(
        CandidateResult(
            name="safe",
            oof_pred=safe_oof,
            test_pred=safe_test,
            cv_auc=float(roc_auc_score(y, safe_oof)),
            details={"weights": {"lgbm": 0.5, "catboost": 0.5}},
        )
    )

    if xgb_result is not None:
        balanced_oof = weighted(
            [lgbm_result.oof_pred, catboost_result.oof_pred, xgb_result.oof_pred],
            [0.35, 0.55, 0.10],
        )
        balanced_test = weighted(
            [lgbm_result.test_pred, catboost_result.test_pred, xgb_result.test_pred],
            [0.35, 0.55, 0.10],
        )
        aggressive_oof = rank_mean([lgbm_result.oof_pred, catboost_result.oof_pred, xgb_result.oof_pred])
        aggressive_test = rank_mean([lgbm_result.test_pred, catboost_result.test_pred, xgb_result.test_pred])
    else:
        balanced_oof = safe_oof.copy()
        balanced_test = safe_test.copy()
        aggressive_oof = rank_mean([lgbm_result.oof_pred, catboost_result.oof_pred])
        aggressive_test = rank_mean([lgbm_result.test_pred, catboost_result.test_pred])

    candidates.append(
        CandidateResult(
            name="balanced",
            oof_pred=balanced_oof,
            test_pred=balanced_test,
            cv_auc=float(roc_auc_score(y, balanced_oof)),
            details={"weights": {"lgbm": 0.35, "catboost": 0.55, "xgb": 0.10 if xgb_result is not None else 0.0}},
        )
    )
    candidates.append(
        CandidateResult(
            name="aggressive",
            oof_pred=aggressive_oof,
            test_pred=aggressive_test,
            cv_auc=float(roc_auc_score(y, aggressive_oof)),
            details={"method": "rank_mean"},
        )
    )

    tuned_two = search_best_lgb_cat_weight(
        y,
        lgbm_result.oof_pred,
        catboost_result.oof_pred,
        lgbm_result.test_pred,
        catboost_result.test_pred,
        step=args.weight_step,
    )
    candidates.append(tuned_two)

    tuned_two_rank = search_best_lgb_cat_rank_weight(
        y=y,
        lgb_oof=lgbm_result.oof_pred,
        cat_oof=catboost_result.oof_pred,
        lgb_test=lgbm_result.test_pred,
        cat_test=catboost_result.test_pred,
        step=args.weight_step,
    )
    candidates.append(tuned_two_rank)

    hybrid_two = search_best_weight_rank_hybrid(
        y=y,
        weighted_candidate=tuned_two,
        rank_candidate=tuned_two_rank,
        alpha_step=args.hybrid_alpha_step,
        name="hybrid_tuned_lgb_cat",
    )
    candidates.append(hybrid_two)

    if xgb_result is not None:
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
        candidates.append(tuned_three)

        tuned_three_rank = search_best_lgb_cat_xgb_rank_weight(
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
        candidates.append(tuned_three_rank)

        hybrid_three = search_best_weight_rank_hybrid(
            y=y,
            weighted_candidate=tuned_three,
            rank_candidate=tuned_three_rank,
            alpha_step=args.hybrid_alpha_step,
            name="hybrid_tuned_lgb_cat_xgb",
        )
        candidates.append(hybrid_three)

    if args.blend_mode == "weighted":
        candidates = [
            candidate
            for candidate in candidates
            if candidate.name in (
                "safe",
                "balanced",
                "tuned_lgb_cat",
                "tuned_lgb_cat_xgb",
                "hybrid_tuned_lgb_cat",
                "hybrid_tuned_lgb_cat_xgb",
            )
        ]
    elif args.blend_mode == "rank":
        candidates = [
            candidate
            for candidate in candidates
            if candidate.name in ("aggressive", "tuned_rank_lgb_cat", "tuned_rank_lgb_cat_xgb")
        ]

    for candidate in candidates:
        save_submission(sample_sub, candidate.test_pred, output_dir / f"submission_{candidate.name}.csv")

    best_candidate, selected_reason = choose_best(candidates)
    best_path = output_dir / f"submission_{best_candidate.name}.csv"
    pd.read_csv(best_path).to_csv(output_dir / "submission_best.csv", index=False)

    candidate_scores = pd.DataFrame(
        [
            {
                "candidate": candidate.name,
                "cv_auc": candidate.cv_auc,
                "submission_file": f"submission_{candidate.name}.csv",
                "details": json.dumps(candidate.details, ensure_ascii=False),
            }
            for candidate in sorted(candidates, key=lambda item: item.cv_auc, reverse=True)
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
        "hybrid_alpha_step": args.hybrid_alpha_step,
        "min_blend_weight": args.min_blend_weight,
        "cat_l2_grid": cat_l2_grid,
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
        "candidates": {candidate.name: candidate.cv_auc for candidate in candidates},
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
