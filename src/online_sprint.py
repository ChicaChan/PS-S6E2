from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_CANDIDATE_COLUMNS = ("candidate", "cv_auc", "submission_file")
OPTIONAL_META_COLUMNS = ("meta_min_gain", "meta_mean_gain")
VALID_STRATEGIES = ("meta_first", "robust_first", "anchor_safe", "external_mix_first")


def load_candidate_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_CANDIDATE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Candidate scores missing required columns: {missing}")

    scores = frame.copy()
    scores["cv_auc"] = pd.to_numeric(scores["cv_auc"], errors="coerce")
    if scores["cv_auc"].isna().any():
        raise ValueError("Candidate scores contain non-numeric cv_auc values")

    for column in OPTIONAL_META_COLUMNS:
        if column in scores.columns:
            scores[column] = pd.to_numeric(scores[column], errors="coerce").fillna(-999.0)

    sort_keys = ["cv_auc"]
    ascending = [False]
    if "meta_min_gain" in scores.columns:
        sort_keys.insert(0, "meta_min_gain")
        ascending.insert(0, False)
    if "meta_mean_gain" in scores.columns:
        sort_keys.insert(1 if "meta_min_gain" in scores.columns else 0, "meta_mean_gain")
        ascending.insert(1 if "meta_min_gain" in scores.columns else 0, False)

    return scores.sort_values(sort_keys, ascending=ascending, kind="mergesort").reset_index(drop=True)


def infer_source_mix(candidate_name: str, source: str = "", source_mix: str = "") -> str:
    if source_mix:
        normalized = source_mix.strip().lower()
        if normalized:
            return normalized

    normalized_source = source.strip().lower()
    if normalized_source in {"external", "mixed", "external_mix", "hybrid"}:
        return normalized_source

    name = candidate_name.lower()
    if "external" in name or name.startswith("ext_"):
        return "external"
    if "mix" in name or "cross" in name or "blend" in name:
        return "mixed"
    if "global_robust" in name:
        return "mixed"
    return "internal"


def assign_expected_roles(selected: list[dict[str, object]]) -> None:
    for idx, item in enumerate(selected):
        if idx == 0:
            item["expected_role"] = "main"
        elif idx == 1:
            item["expected_role"] = "safe"
        else:
            item["expected_role"] = "explore"


def load_prediction_column(path: Path, target_col: str) -> pd.Series:
    frame = pd.read_csv(path)
    if target_col not in frame.columns:
        raise ValueError(f"{path} missing target column: {target_col}")

    prediction = pd.to_numeric(frame[target_col], errors="coerce")
    if prediction.isna().any():
        raise ValueError(f"{path} has non-numeric or NaN values in {target_col}")

    return prediction.astype("float64")


def validate_submission_csv(
    submission_path: Path,
    sample_submission_path: Path,
    id_col: str,
    target_col: str,
) -> None:
    submission = pd.read_csv(submission_path)
    sample_submission = pd.read_csv(sample_submission_path)

    expected_columns = sample_submission.columns.tolist()
    got_columns = submission.columns.tolist()
    if got_columns != expected_columns:
        raise ValueError(f"Submission columns mismatch: expected {expected_columns}, got {got_columns}")

    if len(submission) != len(sample_submission):
        raise ValueError(f"Submission row mismatch: expected {len(sample_submission)}, got {len(submission)}")

    if submission.isna().any().any():
        raise ValueError("Submission contains NaN values")

    if id_col not in submission.columns or id_col not in sample_submission.columns:
        raise ValueError(f"Missing id column in submission or sample: {id_col}")

    if not np.array_equal(submission[id_col].to_numpy(), sample_submission[id_col].to_numpy()):
        raise ValueError("Submission id column order does not match sample submission")

    if target_col not in submission.columns:
        raise ValueError(f"Submission missing target column: {target_col}")

    prediction = pd.to_numeric(submission[target_col], errors="coerce")
    if prediction.isna().any():
        raise ValueError(f"Submission has non-numeric values in target column: {target_col}")
    if ((prediction < 0.0) | (prediction > 1.0)).any():
        raise ValueError("Submission predictions must be within [0, 1]")


def correlation(a: pd.Series, b: pd.Series) -> float:
    value = float(np.corrcoef(a.to_numpy(dtype=np.float64), b.to_numpy(dtype=np.float64))[0, 1])
    if np.isnan(value):
        return 1.0
    return value


def prioritize_candidates(candidate_scores: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unsupported strategy: {strategy}")

    ordered = candidate_scores.copy()
    if strategy == "meta_first":
        return ordered.sort_values("cv_auc", ascending=False, kind="mergesort")

    names = ordered["candidate"].astype(str).str.lower()
    if strategy == "robust_first":
        ordered["_robust_priority"] = names.str.contains("robust|median|quantile").astype(int)
        ordered = ordered.sort_values(["_robust_priority", "cv_auc"], ascending=[False, False], kind="mergesort")
        return ordered.drop(columns=["_robust_priority"])

    if strategy == "external_mix_first":
        source = ordered.get("source", pd.Series(["" for _ in range(len(ordered))]))
        source_mix = ordered.get("source_mix", pd.Series(["" for _ in range(len(ordered))]))
        ordered["source_mix_resolved"] = [
            infer_source_mix(str(candidate), str(src), str(mix))
            for candidate, src, mix in zip(ordered["candidate"], source, source_mix)
        ]
        source_priority_map = {"mixed": 2, "external_mix": 2, "external": 1, "internal": 0}
        ordered["_source_priority"] = ordered["source_mix_resolved"].map(source_priority_map).fillna(0).astype(int)
        ordered["_robust_priority"] = names.str.contains("robust|median|quantile").astype(int)
        ordered = ordered.sort_values(
            ["_source_priority", "_robust_priority", "cv_auc"],
            ascending=[False, False, False],
            kind="mergesort",
        )
        return ordered.drop(columns=["_source_priority", "_robust_priority"])

    ordered["_anchor_safe_priority"] = names.str.contains("safe|robust|median").astype(int)
    ordered = ordered.sort_values(["_anchor_safe_priority", "cv_auc"], ascending=[False, False], kind="mergesort")
    return ordered.drop(columns=["_anchor_safe_priority"])


def select_submission_candidates(
    candidate_scores: pd.DataFrame,
    submission_dir: Path,
    target_col: str,
    top_k: int = 2,
    max_correlation: float = 0.998,
    min_meta_gain: float = -999.0,
    strategy: str = "meta_first",
) -> list[dict[str, object]]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    filtered = candidate_scores.copy()
    if "meta_min_gain" in filtered.columns:
        filtered = filtered[filtered["meta_min_gain"] >= min_meta_gain]
        if filtered.empty:
            filtered = candidate_scores.copy()

    ordered = prioritize_candidates(filtered, strategy)
    if "source_mix_resolved" not in ordered.columns:
        source = ordered.get("source", pd.Series(["" for _ in range(len(ordered))]))
        source_mix = ordered.get("source_mix", pd.Series(["" for _ in range(len(ordered))]))
        ordered["source_mix_resolved"] = [
            infer_source_mix(str(candidate), str(src), str(mix))
            for candidate, src, mix in zip(ordered["candidate"], source, source_mix)
        ]

    internal_best_pred: pd.Series | None = None
    internal_rows = ordered[ordered["source_mix_resolved"] == "internal"]
    if not internal_rows.empty:
        internal_row = internal_rows.sort_values("cv_auc", ascending=False, kind="mergesort").iloc[0]
        internal_path = submission_dir / str(internal_row["submission_file"])
        if internal_path.exists():
            internal_best_pred = load_prediction_column(internal_path, target_col)

    selected: list[dict[str, object]] = []
    cached_pred: dict[str, pd.Series] = {}

    for row in ordered.itertuples(index=False):
        candidate_name = str(row.candidate)
        submission_file = str(row.submission_file)
        file_path = submission_dir / submission_file

        if not file_path.exists():
            raise ValueError(f"Candidate submission file not found: {file_path}")

        if submission_file not in cached_pred:
            cached_pred[submission_file] = load_prediction_column(file_path, target_col)
        pred = cached_pred[submission_file]
        source_mix = str(getattr(row, "source_mix_resolved", "internal"))
        corr_to_internal_best: float | None = None
        if internal_best_pred is not None:
            corr_to_internal_best = float(correlation(pred, internal_best_pred))

        if not selected:
            selected.append(
                {
                    "candidate": candidate_name,
                    "submission_file": submission_file,
                    "cv_auc": float(row.cv_auc),
                    "selection_reason": "highest_priority",
                    "max_corr_to_selected": None,
                    "source_mix": source_mix,
                    "corr_to_internal_best": corr_to_internal_best,
                }
            )
            if len(selected) == top_k:
                assign_expected_roles(selected)
                return selected
            continue

        current_correlations: list[float] = []
        for picked in selected:
            picked_file = str(picked["submission_file"])
            corr = correlation(pred, cached_pred[picked_file])
            current_correlations.append(corr)

        max_corr = max(current_correlations)
        if max_corr <= max_correlation:
            selected.append(
                {
                    "candidate": candidate_name,
                    "submission_file": submission_file,
                    "cv_auc": float(row.cv_auc),
                    "selection_reason": "diversity_gate_passed",
                    "max_corr_to_selected": float(max_corr),
                    "source_mix": source_mix,
                    "corr_to_internal_best": corr_to_internal_best,
                }
            )
            if len(selected) == top_k:
                assign_expected_roles(selected)
                return selected

    if len(selected) == top_k:
        return selected

    picked_files = {str(item["submission_file"]) for item in selected}
    for row in ordered.itertuples(index=False):
        submission_file = str(row.submission_file)
        if submission_file in picked_files:
            continue

        selected.append(
            {
                "candidate": str(row.candidate),
                "submission_file": submission_file,
                "cv_auc": float(row.cv_auc),
                "selection_reason": "fallback_high_cv",
                "max_corr_to_selected": None,
                "source_mix": str(getattr(row, "source_mix_resolved", "internal")),
                "corr_to_internal_best": float(correlation(cached_pred[submission_file], internal_best_pred))
                if internal_best_pred is not None
                else None,
            }
        )
        if len(selected) == top_k:
            break

    assign_expected_roles(selected)
    return selected


def create_anchor_submission(
    submission_dir: Path,
    base_submission_file: str,
    anchor_path: Path,
    id_col: str,
    target_col: str,
    anchor_weight: float,
) -> Path:
    if not (0.0 <= anchor_weight <= 1.0):
        raise ValueError(f"anchor_weight must be in [0, 1], got {anchor_weight}")

    base_path = submission_dir / base_submission_file
    if not base_path.exists():
        raise ValueError(f"Base submission not found: {base_path}")
    if not anchor_path.exists():
        raise ValueError(f"Anchor submission not found: {anchor_path}")

    base_frame = pd.read_csv(base_path)
    anchor_frame = pd.read_csv(anchor_path)
    required = [id_col, target_col]
    for column in required:
        if column not in base_frame.columns or column not in anchor_frame.columns:
            raise ValueError(f"Missing required column {column} in base or anchor submission")

    if len(base_frame) != len(anchor_frame):
        raise ValueError("Anchor submission row count does not match base submission")
    if not np.array_equal(base_frame[id_col].to_numpy(), anchor_frame[id_col].to_numpy()):
        raise ValueError("Anchor submission id order mismatch")

    base_pred = pd.to_numeric(base_frame[target_col], errors="coerce")
    anchor_pred = pd.to_numeric(anchor_frame[target_col], errors="coerce")
    if base_pred.isna().any() or anchor_pred.isna().any():
        raise ValueError("Anchor or base submission contains non-numeric predictions")

    blended = anchor_weight * anchor_pred.to_numpy(dtype=np.float64) + (1.0 - anchor_weight) * base_pred.to_numpy(dtype=np.float64)
    output = base_frame.copy()
    output[target_col] = np.clip(blended, 0.0, 1.0)

    suffix = int(round(anchor_weight * 100.0))
    stem = Path(base_submission_file).stem
    anchor_name = f"submission_anchor_{suffix:02d}_{stem}.csv"
    output_path = submission_dir / anchor_name
    output.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online sprint helper for PS-S6E2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-submission")
    validate_parser.add_argument("--submission-path", required=True)
    validate_parser.add_argument("--sample-sub-path", required=True)
    validate_parser.add_argument("--id-col", default="id")
    validate_parser.add_argument("--target-col", default="Heart Disease")

    pick_parser = subparsers.add_parser("pick-daily")
    pick_parser.add_argument("--candidate-scores-path", required=True)
    pick_parser.add_argument("--submission-dir", required=True)
    pick_parser.add_argument("--target-col", default="Heart Disease")
    pick_parser.add_argument("--id-col", default="id")
    pick_parser.add_argument("--top-k", type=int, default=2)
    pick_parser.add_argument("--max-submit-count", type=int, default=3)
    pick_parser.add_argument("--max-correlation", type=float, default=0.998)
    pick_parser.add_argument("--strategy", choices=VALID_STRATEGIES, default="meta_first")
    pick_parser.add_argument("--anchor-path", default="")
    pick_parser.add_argument("--anchor-weight", type=float, default=0.95)
    pick_parser.add_argument("--output-path", default="")
    pick_parser.add_argument("--min-meta-gain", type=float, default=-999.0)

    return parser.parse_args()


def run_validate_submission(args: argparse.Namespace) -> None:
    validate_submission_csv(
        submission_path=Path(args.submission_path),
        sample_submission_path=Path(args.sample_sub_path),
        id_col=args.id_col,
        target_col=args.target_col,
    )
    print(f"Submission validation passed: {args.submission_path}")


def run_pick_daily(args: argparse.Namespace) -> None:
    candidate_scores = load_candidate_scores(Path(args.candidate_scores_path))
    target_k = max(args.top_k, args.max_submit_count)
    selected = select_submission_candidates(
        candidate_scores=candidate_scores,
        submission_dir=Path(args.submission_dir),
        target_col=args.target_col,
        top_k=target_k,
        max_correlation=args.max_correlation,
        min_meta_gain=args.min_meta_gain,
        strategy=args.strategy,
    )

    anchor_output = ""
    if args.anchor_path and selected and args.strategy == "anchor_safe" and len(selected) < args.max_submit_count:
        anchor_path = Path(args.anchor_path)
        output_path = create_anchor_submission(
            submission_dir=Path(args.submission_dir),
            base_submission_file=str(selected[0]["submission_file"]),
            anchor_path=anchor_path,
            id_col=args.id_col,
            target_col=args.target_col,
            anchor_weight=args.anchor_weight,
        )
        anchor_output = output_path.name
        selected.append(
            {
                "candidate": f"anchor_blend_{Path(selected[0]['candidate']).name}",
                "submission_file": output_path.name,
                "cv_auc": float(selected[0]["cv_auc"]),
                "selection_reason": "anchor_blend",
                "max_corr_to_selected": None,
                "source_mix": "mixed",
                "corr_to_internal_best": None,
            }
        )

    selected = selected[: args.max_submit_count]
    assign_expected_roles(selected)

    payload = {
        "top_k": args.top_k,
        "max_submit_count": args.max_submit_count,
        "max_correlation": args.max_correlation,
        "min_meta_gain": args.min_meta_gain,
        "strategy": args.strategy,
        "anchor_path": args.anchor_path,
        "anchor_weight": args.anchor_weight,
        "anchor_output": anchor_output,
        "selected": selected,
    }

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved selection summary: {output_path}")

    print(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "validate-submission":
        run_validate_submission(args)
    elif args.command == "pick-daily":
        run_pick_daily(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
