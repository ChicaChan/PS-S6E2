from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend Kaggle submission files")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of submission CSV files")
    parser.add_argument("--weights", default="", help="Comma-separated weights; defaults to equal weights")
    parser.add_argument("--input-groups", default="", help="Comma-separated group labels per input (e.g. internal,external)")
    parser.add_argument("--min-per-group", default="", help="Group quota mapping like internal:1,external:1")
    parser.add_argument("--method", choices=("weighted_mean", "rank_mean", "quantile"), default="weighted_mean")
    parser.add_argument("--quantile", type=float, default=0.5, help="Quantile used when method=quantile")
    parser.add_argument("--quantile-grid", default="", help="Comma-separated quantiles, generates multiple outputs")
    parser.add_argument(
        "--corr-prune-threshold",
        type=float,
        default=-1.0,
        help="Enable correlation pruning when in [0.9, 1.0); disabled if < 0",
    )
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--target-col", default="Heart Disease")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--metadata-path", default="", help="Optional JSON path for blend metadata")
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Float list cannot be empty")
    return list(dict.fromkeys(float(item) for item in values))


def parse_weights(raw: str, n_inputs: int) -> np.ndarray:
    if not raw:
        return np.ones(n_inputs, dtype=np.float64) / n_inputs

    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if len(parts) != n_inputs:
        raise ValueError(f"weights count mismatch: expected {n_inputs}, got {len(parts)}")

    values = np.array([float(item) for item in parts], dtype=np.float64)
    if np.any(values < 0):
        raise ValueError("weights must be non-negative")
    if np.allclose(values.sum(), 0.0):
        raise ValueError("weights sum cannot be zero")

    return values / values.sum()


def parse_input_groups(raw: str, n_inputs: int) -> list[str]:
    if not raw:
        return ["internal"] * n_inputs

    parts = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if len(parts) != n_inputs:
        raise ValueError(f"input_groups count mismatch: expected {n_inputs}, got {len(parts)}")
    return parts


def parse_group_minima(raw: str) -> dict[str, int]:
    if not raw:
        return {}

    minima: dict[str, int] = {}
    for item in [segment.strip() for segment in raw.split(",") if segment.strip()]:
        if ":" not in item:
            raise ValueError(f"Invalid min-per-group item: {item}")
        key, value = [token.strip().lower() for token in item.split(":", 1)]
        parsed_value = int(value)
        if parsed_value < 0:
            raise ValueError(f"Group minimum must be >= 0, got {parsed_value} for {key}")
        minima[key] = parsed_value
    return minima


def load_submission(path: Path, id_col: str, target_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = [id_col, target_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    values = pd.to_numeric(frame[target_col], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{path} has non-numeric values in {target_col}")

    selected = frame[required].copy()
    selected[target_col] = values.astype("float64")
    return selected


def validate_alignment(submissions: list[pd.DataFrame], id_col: str) -> None:
    if not submissions:
        raise ValueError("No submissions provided")

    base_ids = submissions[0][id_col].to_numpy()
    for idx, submission in enumerate(submissions[1:], start=1):
        if len(submission) != len(submissions[0]):
            raise ValueError(
                f"Row count mismatch between submission[0]={len(submissions[0])} and submission[{idx}]={len(submission)}"
            )
        if not np.array_equal(base_ids, submission[id_col].to_numpy()):
            raise ValueError(f"ID column mismatch between submission[0] and submission[{idx}]")


def prune_correlated_columns(matrix: np.ndarray, max_corr: float) -> list[int]:
    if not (0.9 <= max_corr < 1.0):
        raise ValueError(f"corr_prune_threshold must be in [0.9, 1.0), got {max_corr}")
    if matrix.shape[1] == 0:
        raise ValueError("Cannot prune empty prediction matrix")

    corr = np.corrcoef(matrix.T)
    keep_indices: list[int] = []
    for idx in range(matrix.shape[1]):
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
    return keep_indices


def select_with_group_quota(
    matrix: np.ndarray,
    groups: list[str],
    max_corr: float,
    min_per_group: dict[str, int],
    order_scores: np.ndarray | None = None,
) -> tuple[list[int], dict[str, Any]]:
    if matrix.shape[1] != len(groups):
        raise ValueError(f"group length mismatch: matrix has {matrix.shape[1]} columns, groups has {len(groups)}")
    if not (0.9 <= max_corr < 1.0):
        raise ValueError(f"corr_prune_threshold must be in [0.9, 1.0), got {max_corr}")

    available_by_group: dict[str, int] = {}
    for group in groups:
        available_by_group[group] = available_by_group.get(group, 0) + 1
    for group, minimum in min_per_group.items():
        if minimum > available_by_group.get(group, 0):
            raise ValueError(f"Group quota infeasible for {group}: requires {minimum}, available {available_by_group.get(group, 0)}")

    if order_scores is None:
        priority = list(range(matrix.shape[1]))
    else:
        if len(order_scores) != matrix.shape[1]:
            raise ValueError(
                f"order_scores length mismatch: expected {matrix.shape[1]}, got {len(order_scores)}"
            )
        priority = sorted(range(matrix.shape[1]), key=lambda idx: float(order_scores[idx]), reverse=True)

    corr = np.corrcoef(matrix.T)
    keep_indices: list[int] = []
    forced_indices: list[int] = []

    def can_keep(candidate_idx: int) -> bool:
        for kept_idx in keep_indices:
            value = float(corr[candidate_idx, kept_idx])
            if not np.isnan(value) and abs(value) >= max_corr:
                return False
        return True

    for group, minimum in min_per_group.items():
        needed = minimum
        group_candidates = [idx for idx in priority if groups[idx] == group]

        for idx in group_candidates:
            if idx in keep_indices:
                continue
            if can_keep(idx):
                keep_indices.append(idx)
                needed -= 1
            if needed == 0:
                break

        if needed > 0:
            for idx in group_candidates:
                if idx in keep_indices:
                    continue
                keep_indices.append(idx)
                forced_indices.append(idx)
                needed -= 1
                if needed == 0:
                    break

    for idx in priority:
        if idx in keep_indices:
            continue
        if can_keep(idx):
            keep_indices.append(idx)

    if not keep_indices:
        keep_indices = [priority[0]]

    details: dict[str, Any] = {
        "min_per_group": min_per_group,
        "forced_indices": forced_indices,
        "forced_groups": [groups[idx] for idx in forced_indices],
    }
    return keep_indices, details


def blend_predictions(
    submissions: list[pd.DataFrame],
    weights: np.ndarray,
    method: str,
    target_col: str,
    quantile: float = 0.5,
    corr_prune_threshold: float | None = None,
    input_groups: list[str] | None = None,
    min_per_group: dict[str, int] | None = None,
    return_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    matrix = np.column_stack([submission[target_col].to_numpy(dtype=np.float64) for submission in submissions])
    active_weights = weights.copy()
    active_groups = list(input_groups) if input_groups is not None else None

    details: dict[str, Any] = {
        "method": method,
        "n_inputs": int(matrix.shape[1]),
        "kept_indices": list(range(matrix.shape[1])),
        "corr_prune_threshold": corr_prune_threshold,
    }

    if corr_prune_threshold is not None:
        if active_groups is not None and min_per_group:
            keep_indices, quota_details = select_with_group_quota(
                matrix,
                groups=active_groups,
                max_corr=corr_prune_threshold,
                min_per_group=min_per_group,
                order_scores=active_weights,
            )
            details["quota_details"] = quota_details
        else:
            keep_indices = prune_correlated_columns(matrix, corr_prune_threshold)

        matrix = matrix[:, keep_indices]
        active_weights = active_weights[keep_indices]
        active_weights = active_weights / active_weights.sum()
        details["kept_indices"] = keep_indices
        details["n_kept"] = len(keep_indices)
        if active_groups is not None:
            active_groups = [active_groups[idx] for idx in keep_indices]
            details["kept_groups"] = active_groups

    if method == "weighted_mean":
        blended = matrix @ active_weights
    elif method == "rank_mean":
        ranks = np.column_stack(
            [pd.Series(matrix[:, idx]).rank(method="average", pct=True).to_numpy() for idx in range(matrix.shape[1])]
        )
        blended = ranks @ active_weights
    elif method == "quantile":
        if not (0.0 <= quantile <= 1.0):
            raise ValueError(f"quantile must be in [0, 1], got {quantile}")
        blended = np.quantile(matrix, q=quantile, axis=1)
        details["quantile"] = float(quantile)
    else:
        raise ValueError(f"Unsupported blend method: {method}")

    clipped = np.clip(blended, 0.0, 1.0)
    if return_details:
        return clipped, details
    return clipped


def format_quantile_label(value: float) -> str:
    normalized = f"{value:.4f}".rstrip("0").rstrip(".")
    return normalized.replace(".", "p")


def build_quantile_candidates(
    submissions: list[pd.DataFrame],
    weights: np.ndarray,
    target_col: str,
    quantiles: list[float],
    corr_prune_threshold: float | None,
    input_groups: list[str],
    min_per_group: dict[str, int],
) -> list[tuple[float, np.ndarray, dict[str, Any]]]:
    results: list[tuple[float, np.ndarray, dict[str, Any]]] = []
    for quantile in quantiles:
        blended, details = blend_predictions(
            submissions,
            weights,
            method="quantile",
            target_col=target_col,
            quantile=quantile,
            corr_prune_threshold=corr_prune_threshold,
            input_groups=input_groups,
            min_per_group=min_per_group,
            return_details=True,
        )
        results.append((quantile, blended, details))
    return results


def main() -> None:
    args = parse_args()

    input_paths = [Path(item) for item in args.inputs]
    weights = parse_weights(args.weights, len(input_paths))
    input_groups = parse_input_groups(args.input_groups, len(input_paths))
    min_per_group = parse_group_minima(args.min_per_group)
    if not min_per_group and {"internal", "external"}.issubset(set(input_groups)):
        min_per_group = {"internal": 1, "external": 1}

    submissions = [load_submission(path, args.id_col, args.target_col) for path in input_paths]
    validate_alignment(submissions, args.id_col)

    corr_prune_threshold = None
    if args.corr_prune_threshold >= 0.0:
        corr_prune_threshold = float(args.corr_prune_threshold)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_records: list[dict[str, Any]] = []
    outputs_written: list[Path] = []

    if args.method == "quantile":
        quantiles = parse_float_list(args.quantile_grid) if args.quantile_grid else [args.quantile]
        quantile_candidates = build_quantile_candidates(
            submissions=submissions,
            weights=weights,
            target_col=args.target_col,
            quantiles=quantiles,
            corr_prune_threshold=corr_prune_threshold,
            input_groups=input_groups,
            min_per_group=min_per_group,
        )

        for index, (quantile, blended, details) in enumerate(quantile_candidates):
            if len(quantile_candidates) == 1:
                candidate_path = output_path
            else:
                suffix = format_quantile_label(quantile)
                candidate_path = output_path.with_name(f"{output_path.stem}_q{suffix}{output_path.suffix}")

            output = pd.DataFrame(
                {
                    args.id_col: submissions[0][args.id_col],
                    args.target_col: blended,
                }
            )
            output.to_csv(candidate_path, index=False)
            outputs_written.append(candidate_path)
            metadata_records.append(
                {
                    "index": index,
                    "output_file": candidate_path.name,
                    "method": args.method,
                    "quantile": float(quantile),
                    "details": details,
                }
            )
    else:
        blended, details = blend_predictions(
            submissions,
            weights,
            args.method,
            args.target_col,
            quantile=args.quantile,
            corr_prune_threshold=corr_prune_threshold,
            input_groups=input_groups,
            min_per_group=min_per_group,
            return_details=True,
        )
        output = pd.DataFrame(
            {
                args.id_col: submissions[0][args.id_col],
                args.target_col: blended,
            }
        )
        output.to_csv(output_path, index=False)
        outputs_written.append(output_path)
        metadata_records.append(
            {
                "index": 0,
                "output_file": output_path.name,
                "method": args.method,
                "quantile": float(args.quantile) if args.method == "quantile" else None,
                "details": details,
            }
        )

    payload: dict[str, Any] = {
        "inputs": [str(path) for path in input_paths],
        "weights": [float(weight) for weight in weights.tolist()],
        "input_groups": input_groups,
        "min_per_group": min_per_group,
        "corr_prune_threshold": corr_prune_threshold,
        "method": args.method,
        "records": metadata_records,
    }

    if args.metadata_path:
        metadata_path = Path(args.metadata_path)
    elif len(outputs_written) > 1:
        metadata_path = output_path.with_suffix(".json")
    else:
        metadata_path = None

    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved blend metadata: {metadata_path}")

    print(f"Method: {args.method}")
    print(f"Weights: {[round(float(weight), 6) for weight in weights.tolist()]}")
    print(f"Input groups: {input_groups}")
    if min_per_group:
        print(f"Min per group: {min_per_group}")
    if args.method == "quantile":
        if args.quantile_grid:
            print(f"Quantile grid: {parse_float_list(args.quantile_grid)}")
        else:
            print(f"Quantile: {args.quantile:.4f}")
    if corr_prune_threshold is not None:
        print(f"Correlation prune threshold: {corr_prune_threshold:.6f}")
    for path in outputs_written:
        print(f"Saved blended submission: {path}")


if __name__ == "__main__":
    main()
