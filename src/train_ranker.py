from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET_MAP = {"Absence": 0, "Presence": 1}
REQUIRED_CONFIG_KEYS = (
    "goal",
    "model",
    "cv_folds",
    "cv_seed",
    "seeds",
    "feature_set",
    "blend",
)

DEFAULT_MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "lgbm": {
        "objective": "binary",
        "learning_rate": 0.05,
        "n_estimators": 2500,
        "num_leaves": 63,
        "min_child_samples": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "n_jobs": -1,
    },
    "catboost": {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.03,
        "iterations": 3000,
        "depth": 8,
        "l2_leaf_reg": 3.0,
        "random_strength": 0.1,
        "verbose": False,
    },
    "xgb": {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.05,
        "n_estimators": 2500,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "min_child_weight": 1.0,
        "n_jobs": -1,
        "tree_method": "hist",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified rank push trainer for PS-S6E2")
    parser.add_argument("--train-path", default="data/raw/train.csv")
    parser.add_argument("--test-path", default="data/raw/test.csv")
    parser.add_argument("--sample-sub-path", default="data/raw/sample_submission.csv")
    parser.add_argument("--output-dir", default="kaggle_outputs/rank_push")
    parser.add_argument("--registry-path", default="kaggle_outputs/experiment_registry.csv")
    parser.add_argument("--config-path", default="")
    parser.add_argument("--model", choices=("lgbm", "catboost", "xgb"), default="lgbm")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=42)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--feature-set", choices=("base", "base_plus_interactions"), default="base")
    parser.add_argument("--target-col", default="Heart Disease")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--model-params-json", default="")
    parser.add_argument("--save-oof", action="store_true")
    parser.add_argument("--save-test-pred", action="store_true")
    parser.add_argument("--experiment-id", default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def encode_target(target: pd.Series) -> pd.Series:
    encoded = target.map(TARGET_MAP)
    if encoded.isna().any():
        unknown_labels = sorted(target[encoded.isna()].unique().tolist())
        raise ValueError(f"Unknown target labels: {unknown_labels}")
    return encoded.astype("int8")


def parse_seeds(raw: str | Iterable[int]) -> list[int]:
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if not parts:
            raise ValueError("--seeds cannot be empty")
        values = [int(part) for part in parts]
    else:
        values = [int(value) for value in raw]
    deduped = list(dict.fromkeys(values))
    if not deduped:
        raise ValueError("Parsed seeds list is empty")
    return deduped


def load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for --config-path. Install with: pip install pyyaml") from exc

    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ValueError(f"Config at {path} must be a YAML object")
    return content


def validate_config(config: dict[str, Any], required_keys: Iterable[str] = REQUIRED_CONFIG_KEYS) -> None:
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")


def apply_config_overrides(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    if not config:
        return args

    for key in ("model", "cv_folds", "cv_seed", "feature_set", "target_col", "id_col"):
        if key in config:
            setattr(args, key, config[key])

    if "seeds" in config:
        seeds = config["seeds"]
        if isinstance(seeds, list):
            args.seeds = ",".join(str(seed) for seed in seeds)
        else:
            args.seeds = str(seeds)

    return args


def parse_model_param_overrides(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--model-params-json must parse to a JSON object")
    return parsed


def build_feature_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    id_col: str,
    feature_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    feature_cols = [col for col in train_df.columns if col not in (target_col, id_col)]
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    added_cols: list[str] = []

    if feature_set == "base_plus_interactions":
        numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(X_train[col])]
        interaction_cols = numeric_cols[:3]
        for left_idx in range(len(interaction_cols)):
            for right_idx in range(left_idx + 1, len(interaction_cols)):
                left_col = interaction_cols[left_idx]
                right_col = interaction_cols[right_idx]
                new_col = f"{left_col}_mul_{right_col}"
                X_train[new_col] = X_train[left_col] * X_train[right_col]
                X_test[new_col] = X_test[left_col] * X_test[right_col]
                added_cols.append(new_col)

    return X_train, X_test, feature_cols + added_cols


def resolve_model_params(
    model_name: str,
    seed: int,
    config: dict[str, Any],
    cli_overrides: dict[str, Any],
) -> dict[str, Any]:
    params = dict(DEFAULT_MODEL_PARAMS[model_name])
    params["random_state"] = seed

    config_params = config.get("model_params", {}) if config else {}
    if isinstance(config_params, dict):
        model_specific = config_params.get(model_name, {})
        if isinstance(model_specific, dict):
            params.update(model_specific)

    params.update(cli_overrides)

    if model_name == "catboost":
        params.pop("random_state", None)
        params["random_seed"] = seed
    elif model_name == "xgb":
        params.pop("random_seed", None)

    return params


def make_model(model_name: str, params: dict[str, Any]) -> Any:
    if model_name == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(**params)
    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise RuntimeError("catboost is required for --model catboost") from exc
        return CatBoostClassifier(**params)
    if model_name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("xgboost is required for --model xgb") from exc
        return XGBClassifier(**params)
    raise ValueError(f"Unsupported model name: {model_name}")


def _predict_positive(model: Any, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    return np.asarray(proba)[:, 1]


def _best_iteration(model: Any) -> int:
    for attr_name in ("best_iteration_", "best_iteration", "tree_count_"):
        value = getattr(model, attr_name, None)
        if value is not None:
            return int(value)
    n_estimators = getattr(model, "n_estimators", None)
    if n_estimators is not None:
        return int(n_estimators)
    return -1


def fit_fold_model(
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> Any:
    if model_name == "lgbm":
        import lightgbm as lgb

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=120, verbose=False)],
        )
        return model

    if model_name == "catboost":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            use_best_model=True,
            early_stopping_rounds=120,
            verbose=False,
        )
        return model

    if model_name == "xgb":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
        return model

    raise ValueError(f"Unsupported model name: {model_name}")


def run_single_seed_cv(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    cv_folds: int,
    cv_seed: int,
    seed: int,
    config: dict[str, Any],
    cli_model_overrides: dict[str, Any],
) -> dict[str, Any]:
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cv_seed)
    oof_pred = np.zeros(len(X), dtype=np.float64)
    test_pred_acc = np.zeros(len(X_test), dtype=np.float64)
    fold_aucs: list[float] = []
    fold_best_iterations: list[int] = []

    for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        model_params = resolve_model_params(
            model_name=model_name,
            seed=seed + fold_id,
            config=config,
            cli_overrides=cli_model_overrides,
        )
        model = make_model(model_name=model_name, params=model_params)
        model = fit_fold_model(model_name, model, X_train, y_train, X_valid, y_valid)

        valid_pred = _predict_positive(model, X_valid)
        test_fold_pred = _predict_positive(model, X_test)

        oof_pred[valid_idx] = valid_pred
        test_pred_acc += test_fold_pred / cv_folds

        fold_auc = roc_auc_score(y_valid, valid_pred)
        fold_aucs.append(float(fold_auc))
        fold_best_iterations.append(_best_iteration(model))

    return {
        "seed": seed,
        "oof": oof_pred,
        "test": test_pred_acc,
        "fold_aucs": fold_aucs,
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "best_iterations": fold_best_iterations,
    }


def build_submission(test_df: pd.DataFrame, pred: np.ndarray, id_col: str, target_col: str) -> pd.DataFrame:
    if id_col not in test_df.columns:
        raise ValueError(f"Missing id column in test data: {id_col}")
    if len(test_df) != len(pred):
        raise ValueError("Prediction size does not match test rows")
    return pd.DataFrame({id_col: test_df[id_col], target_col: pred})


def validate_submission_frame(submission: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    expected_columns = sample_submission.columns.tolist()
    got_columns = submission.columns.tolist()
    if got_columns != expected_columns:
        raise ValueError(f"Submission columns mismatch: expected {expected_columns}, got {got_columns}")
    if len(submission) != len(sample_submission):
        raise ValueError(f"Submission row mismatch: expected {len(sample_submission)}, got {len(submission)}")
    if submission.isna().any().any():
        raise ValueError("Submission contains NaN values")


def append_experiment_registry(path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "exp_id",
        "model",
        "seeds",
        "features",
        "cv_auc",
        "cv_std",
        "public_lb",
        "delta_lb",
        "submit_file",
        "submit_time",
        "status",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row], columns=fieldnames)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    set_seed(args.cv_seed)

    config: dict[str, Any] = {}
    if args.config_path:
        config = load_yaml_config(Path(args.config_path))
        validate_config(config)
        args = apply_config_overrides(args, config)

    seeds = parse_seeds(args.seeds)
    cli_model_overrides = parse_model_param_overrides(args.model_params_json)

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    sample_sub = pd.read_csv(args.sample_sub_path)
    y = encode_target(train_df[args.target_col])

    X, X_test, feature_cols = build_feature_matrices(
        train_df=train_df,
        test_df=test_df,
        target_col=args.target_col,
        id_col=args.id_col,
        feature_set=args.feature_set,
    )

    started_at = time.perf_counter()
    seed_results: list[dict[str, Any]] = []
    oof_stack: list[np.ndarray] = []
    test_stack: list[np.ndarray] = []

    for seed in seeds:
        result = run_single_seed_cv(
            model_name=args.model,
            X=X,
            y=y,
            X_test=X_test,
            cv_folds=args.cv_folds,
            cv_seed=args.cv_seed,
            seed=seed,
            config=config,
            cli_model_overrides=cli_model_overrides,
        )
        seed_results.append(result)
        oof_stack.append(result["oof"])
        test_stack.append(result["test"])
        print(
            f"Seed {seed}: mean_auc={result['mean_auc']:.6f} std_auc={result['std_auc']:.6f} "
            f"fold_aucs={[round(v, 6) for v in result['fold_aucs']]}"
        )

    oof_blend = np.mean(oof_stack, axis=0)
    test_blend = np.mean(test_stack, axis=0)
    overall_cv_auc = float(roc_auc_score(y, oof_blend))
    overall_cv_std = float(np.std([item["mean_auc"] for item in seed_results]))
    total_runtime = float(time.perf_counter() - started_at)

    submission = build_submission(
        test_df=test_df,
        pred=test_blend,
        id_col=args.id_col,
        target_col=args.target_col,
    )
    validate_submission_frame(submission, sample_sub)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    submission_path = output_dir / f"submission_{args.model}_blend_{timestamp}.csv"
    metrics_path = output_dir / f"metrics_{args.model}_blend_{timestamp}.json"
    submission.to_csv(submission_path, index=False)

    if args.save_oof:
        oof_df = pd.DataFrame(
            {
                args.id_col: train_df[args.id_col],
                args.target_col: y,
                "prediction": oof_blend,
            }
        )
        oof_df.to_csv(output_dir / f"oof_{args.model}_blend_{timestamp}.csv", index=False)

    if args.save_test_pred:
        test_pred_df = pd.DataFrame({args.id_col: test_df[args.id_col], "prediction": test_blend})
        test_pred_df.to_csv(output_dir / f"test_pred_{args.model}_blend_{timestamp}.csv", index=False)

    metrics = {
        "overall_cv_auc": overall_cv_auc,
        "overall_cv_std": overall_cv_std,
        "runtime_sec": total_runtime,
        "model": args.model,
        "cv_folds": args.cv_folds,
        "cv_seed": args.cv_seed,
        "seeds": seeds,
        "feature_set": args.feature_set,
        "feature_count": len(feature_cols),
        "seed_results": [
            {
                "seed": result["seed"],
                "mean_auc": result["mean_auc"],
                "std_auc": result["std_auc"],
                "fold_aucs": result["fold_aucs"],
                "best_iterations": result["best_iterations"],
            }
            for result in seed_results
        ],
        "model_param_overrides": cli_model_overrides,
        "config_path": args.config_path,
        "submission_path": str(submission_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    experiment_id = args.experiment_id or f"{args.model}_{timestamp}"
    append_experiment_registry(
        path=Path(args.registry_path),
        row={
            "exp_id": experiment_id,
            "model": args.model,
            "seeds": "|".join(str(seed) for seed in seeds),
            "features": args.feature_set,
            "cv_auc": f"{overall_cv_auc:.6f}",
            "cv_std": f"{overall_cv_std:.6f}",
            "public_lb": "",
            "delta_lb": "",
            "submit_file": submission_path.name,
            "submit_time": datetime.now().isoformat(timespec="seconds"),
            "status": "local_completed",
            "notes": "auto-recorded by train_ranker",
        },
    )

    print(f"Overall CV AUC: {overall_cv_auc:.6f}")
    print(f"Saved submission: {submission_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Runtime (s): {total_runtime:.2f}")


if __name__ == "__main__":
    main()
