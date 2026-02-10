from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

TARGET_MAP = {"Absence": 0, "Presence": 1}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def encode_target(target: pd.Series) -> pd.Series:
    encoded = target.map(TARGET_MAP)
    if encoded.isna().any():
        unknown_labels = sorted(target[encoded.isna()].unique().tolist())
        raise ValueError(f"Unknown target labels: {unknown_labels}")
    return encoded.astype("int8")


def split_features(
    data: pd.DataFrame,
    target_col: str,
    id_col: str,
    drop_id: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    y = encode_target(data[target_col])
    X = data.drop(columns=[target_col]).copy()
    if drop_id and id_col in X.columns:
        X = X.drop(columns=[id_col])
    return X, y


def make_model(seed: int, n_estimators: int = 2000) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        n_estimators=n_estimators,
        num_leaves=63,
        min_child_samples=30,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        random_state=seed,
        n_jobs=-1,
    )


def train_validate(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> tuple[int, float, float]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, valid_idx = next(splitter.split(X, y))
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = make_model(seed=seed)
    started_at = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=120, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    valid_pred = model.predict_proba(X_valid)[:, 1]
    valid_auc = roc_auc_score(y_valid, valid_pred)
    best_iteration = model.best_iteration_ or model.n_estimators
    elapsed = time.perf_counter() - started_at
    return int(best_iteration), float(valid_auc), float(elapsed)


def train_full_and_predict(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    seed: int,
    n_estimators: int,
) -> tuple[pd.Series, float]:
    model = make_model(seed=seed, n_estimators=n_estimators)
    started_at = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - started_at
    pred = model.predict_proba(X_test)[:, 1]
    return pd.Series(pred, index=X_test.index, dtype="float64"), float(elapsed)


def build_submission(
    test_data: pd.DataFrame,
    prediction: pd.Series,
    id_col: str,
    target_col: str,
) -> pd.DataFrame:
    if id_col not in test_data.columns:
        raise ValueError(f"Missing id column in test data: {id_col}")
    if len(test_data) != len(prediction):
        raise ValueError("Prediction size does not match test rows")
    return pd.DataFrame({id_col: test_data[id_col], target_col: prediction.values})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train optimized LightGBM baseline for PS-S6E2")
    parser.add_argument("--train-path", default="data/raw/train.csv")
    parser.add_argument("--test-path", default="data/raw/test.csv")
    parser.add_argument("--output-path", default="submissions/lgbm_optimized.csv")
    parser.add_argument("--target-col", default="Heart Disease")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-id", action="store_true", help="Keep id as feature")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    X, y = split_features(
        data=train_df,
        target_col=args.target_col,
        id_col=args.id_col,
        drop_id=not args.keep_id,
    )
    X_test = test_df.copy()
    if not args.keep_id and args.id_col in X_test.columns:
        X_test = X_test.drop(columns=[args.id_col])

    best_iteration, valid_auc, val_time = train_validate(X=X, y=y, seed=args.seed)
    prediction, full_train_time = train_full_and_predict(
        X=X,
        y=y,
        X_test=X_test,
        seed=args.seed,
        n_estimators=best_iteration,
    )
    submission_df = build_submission(
        test_data=test_df,
        prediction=prediction,
        id_col=args.id_col,
        target_col=args.target_col,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    print(f"Validation AUC: {valid_auc:.6f}")
    print(f"Best iteration: {best_iteration}")
    print(f"Validation train time (s): {val_time:.2f}")
    print(f"Full train time (s): {full_train_time:.2f}")
    print(f"Saved submission: {output_path}")


if __name__ == "__main__":
    main()
