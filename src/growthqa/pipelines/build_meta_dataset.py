from __future__ import annotations
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
from growthqa.io.wide_loader import load_and_concat_wides, load_wide_csv
from growthqa.preprocess.blank_status import load_blank_status_map
from growthqa.preprocess.interpolate import build_raw_merged
from growthqa.preprocess.transform import preprocess_wide
from growthqa.features.meta import build_metadata_from_wide
from growthqa.preprocess.timegrid import get_sorted_time_columns
from growthqa.preprocess.truncation_augment import augment_raw_wide
from growthqa.preprocess.gap_augment import augment_raw_wide_with_gaps
from growthqa.config import (
    STEP_HOURS as TRAIN_STEP_HOURS,
    TMAX_HOURS as TRAIN_TMAX_HOURS,
    TRUNC_HORIZONS as TRAIN_TRUNC_HORIZONS,
    TRUNC_PER_CURVE as TRAIN_TRUNC_PER_CURVE,
    TRUNC_SEED as TRAIN_TRUNC_SEED,
    SMOOTH_METHOD as TRAIN_SMOOTH_METHOD,
    SMOOTH_WINDOW as TRAIN_SMOOTH_WINDOW,
    NORMALIZE as TRAIN_NORMALIZE,
    GAP_AUG_FRACTION as TRAIN_GAP_AUG_FRACTION,
    GAP_AUG_PER_CURVE as TRAIN_GAP_AUG_PER_CURVE,
    GAP_AUG_SEED as TRAIN_GAP_AUG_SEED,
    GAP_MIN_HOURS as TRAIN_GAP_MIN_HOURS,
    GAP_MAX_HOURS as TRAIN_GAP_MAX_HOURS,
    GAP_MIN_MISSING_FRAC as TRAIN_GAP_MIN_MISSING_FRAC,
    GAP_MAX_MISSING_FRAC as TRAIN_GAP_MAX_MISSING_FRAC,
)

RAW_ID_COLS = ["FileName", "Test Id", "Model Name", "Is_Valid"]
FINAL_ID_COLS = [
    "FileName",
    "Test Id",
    "Model Name",
    "base_curve_id",
    "aug_id",
    "Is_Valid",
    "tmax_original",
    "train_horizon",
    "is_censored",
    "too_sparse",
    "n_points_observed_raw",
    "max_gap_hours_raw",
    "missing_frac_on_grid_raw",
    "is_synthetic",
    "gap_augmented",
    "gap_pattern",
]


def _order_columns(df: pd.DataFrame, id_cols: List[str]) -> pd.DataFrame:
    present_ids = [c for c in id_cols if c in df.columns]
    tcols = tcols = get_sorted_time_columns(df)
    return df[present_ids + tcols].copy()


def run_merge_preprocess_meta(
    *,
    inputs: List[str],
    out_raw: Optional[str],
    out_final: Optional[str],
    out_meta: Optional[str],
    # interpolation/grid
    step: float = 0.5,
    min_points: int = 3,
    tmax_hours: Optional[float] = 16.0,
    # blank/baseline
    blank_subtracted: bool = False,
    clip_negatives: bool = False,
    global_blank: Optional[float] = None,
    blank_status_csv: Optional[str] = None,
    blank_default: str = "RAW",
    # smoothing + normalization
    smooth_method: str = "NONE", 
    smooth_window: int = 5,
    normalize: str = "NONE",
    # logging
    loglevel: str = "INFO",
    # augmentation
    augment_trunc: bool = False,
    trunc_horizons: Optional[List[float]] = None,
    trunc_per_curve: int = 3,
    trunc_seed: int = 123,
    augment_gaps: bool = False,
    gap_fraction: Optional[float] = None,
    gap_per_curve: Optional[int] = None,
    gap_seed: Optional[int] = None,
    gap_min_hours: Optional[float] = None,
    gap_max_hours: Optional[float] = None,
    gap_min_missing_frac: Optional[float] = None,
    gap_max_missing_frac: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logging.basicConfig(level=getattr(logging, loglevel.upper(), logging.INFO))
    log = logging.getLogger("merge_preprocess_meta")

    df_in = load_and_concat_wides(inputs)
    raw_merged_df = build_raw_merged(
        df_in,
        step_hours=step,
        min_points=min_points,
        tmax_hours=tmax_hours,
    )
    if augment_trunc:
        hs = trunc_horizons or TRAIN_TRUNC_HORIZONS
        truncated_raw_df = augment_raw_wide(
            df_in,
            candidate_horizons=hs,
            per_curve=trunc_per_curve,
            seed=trunc_seed,
            full_horizon=float(tmax_hours or TRAIN_TMAX_HOURS),
            step_hours=step,
        )
        log.info("Truncation augmentation (raw-first): per_curve=%s horizons=%s", trunc_per_curve, hs)
        augmented_raw_parts = [truncated_raw_df]
        if augment_gaps:
            gap_raw_df = augment_raw_wide_with_gaps(
                df_in,
                frac_curves_to_augment=gap_fraction if gap_fraction is not None else TRAIN_GAP_AUG_FRACTION,
                per_curve=gap_per_curve if gap_per_curve is not None else TRAIN_GAP_AUG_PER_CURVE,
                seed=gap_seed if gap_seed is not None else TRAIN_GAP_AUG_SEED,
                min_gap_hours=gap_min_hours if gap_min_hours is not None else TRAIN_GAP_MIN_HOURS,
                max_gap_hours=gap_max_hours if gap_max_hours is not None else TRAIN_GAP_MAX_HOURS,
                min_missing_frac=gap_min_missing_frac if gap_min_missing_frac is not None else TRAIN_GAP_MIN_MISSING_FRAC,
                max_missing_frac=gap_max_missing_frac if gap_max_missing_frac is not None else TRAIN_GAP_MAX_MISSING_FRAC,
            )
            if not gap_raw_df.empty:
                augmented_raw_parts.append(gap_raw_df)
            log.info("Gap augmentation (raw-first): %d gap-augmented rows generated", len(gap_raw_df))

        combined_raw_df = (
            pd.concat(augmented_raw_parts, ignore_index=True, sort=False)
            if len(augmented_raw_parts) > 1
            else augmented_raw_parts[0]
        )
        final_input_df = build_raw_merged(
            combined_raw_df,
            step_hours=step,
            min_points=min_points,
            tmax_hours=tmax_hours,
        )
    else:
        final_input_df = raw_merged_df


    blank_status_map = load_blank_status_map(blank_status_csv) if blank_status_csv else None

    final_merged_df = preprocess_wide(
        final_input_df,
        blank_subtracted=blank_subtracted,
        clip_negatives=clip_negatives,
        global_blank=global_blank,
        blank_status_map=blank_status_map,
        blank_default=blank_default,
        smooth_method=smooth_method,
        smooth_window=smooth_window,
        normalize_mode=str(normalize),
    )

    meta_df = build_metadata_from_wide(final_merged_df)
    if out_raw:
        Path(out_raw).parent.mkdir(parents=True, exist_ok=True)
        _order_columns(raw_merged_df, RAW_ID_COLS).to_csv(out_raw, index=False)
    if out_final:
        Path(out_final).parent.mkdir(parents=True, exist_ok=True)
        _order_columns(final_merged_df, FINAL_ID_COLS).to_csv(out_final, index=False)
    if out_meta:
        Path(out_meta).parent.mkdir(parents=True, exist_ok=True)
        meta_df.to_csv(out_meta, index=False)

    return raw_merged_df, final_merged_df, meta_df


def _tagged_temp_csv(path: str, source_type: str, tmp_dir: Path) -> str:
    df = load_wide_csv(path)
    df["source_type"] = source_type
    df["is_synthetic"] = int(source_type == "synthetic")
    out = tmp_dir / f"{source_type}_{Path(path).stem}.csv"
    df.to_csv(out, index=False)
    return str(out)


def build_training_meta(
    *,
    synthetic_csv: str,
    lab_csv: Optional[str] = None,
    out_dir: str,
) -> dict:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    has_lab = bool(lab_csv)

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        inputs = [_tagged_temp_csv(synthetic_csv, "synthetic", tdp)]
        if has_lab:
            inputs.append(_tagged_temp_csv(lab_csv, "lab", tdp))

        raw, final, meta = run_merge_preprocess_meta(
            inputs=inputs,
            out_raw=str(out_dir_p / "raw_merged.csv") if has_lab else None,
            out_final=str(out_dir_p / "final_merged.csv"),
            out_meta=str(out_dir_p / "training_meta.csv"),
            step=TRAIN_STEP_HOURS,
            tmax_hours=TRAIN_TMAX_HOURS,
            blank_subtracted=False,
            smooth_method=TRAIN_SMOOTH_METHOD,
            smooth_window=TRAIN_SMOOTH_WINDOW,
            normalize=TRAIN_NORMALIZE,
            # rich_meta=False,
            augment_trunc=True,
            trunc_horizons=TRAIN_TRUNC_HORIZONS,
            trunc_per_curve=TRAIN_TRUNC_PER_CURVE,
            trunc_seed=TRAIN_TRUNC_SEED,
            augment_gaps=True,
            loglevel="ERROR",
        )

    return {
        "raw_merged_rows": int(len(raw)) if has_lab else None,
        "final_merged_rows": int(len(final)),
        "meta_rows": int(len(meta)),
        "meta_feature_columns": [c for c in meta.columns],
        "out_dir": str(out_dir_p),
        "lab_data_included": has_lab,
    }