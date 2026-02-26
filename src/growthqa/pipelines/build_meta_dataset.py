from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from growthqa.io.wide_loader import load_and_concat_wides
from growthqa.preprocess.blank_status import load_blank_status_map
from growthqa.preprocess.interpolate import build_raw_merged
from growthqa.preprocess.transform import preprocess_wide
from growthqa.features.meta import build_metadata_from_wide


def run_merge_preprocess_meta(
    *,
    inputs: List[str],
    out_raw: Optional[str],
    out_final: Optional[str],
    out_meta: Optional[str],
    # interpolation/grid
    step: float = 0.5,
    min_points: int = 3,
    low_res_threshold: int = 7,
    tmax_hours: Optional[float] = 16.0,
    auto_tmax: bool = False,
    auto_tmax_coverage: float = 0.8,
    # blank/baseline
    blank_subtracted: bool = False,
    clip_negatives: bool = False,
    global_blank: Optional[float] = None,
    blank_status_csv: Optional[str] = None,
    blank_default: str = "RAW",  # RAW or ALREADY
    # smoothing + normalization
    smooth_method: str = "NONE",  # NONE, RAW, LWS, SGF
    smooth_window: int = 5,
    normalize: str = "NONE",  # NONE, MAX, MINMAX
    # logging
    loglevel: str = "INFO",
    # audit label
    add_audit_meta_label: bool = False,
    rich_meta: bool = False,
    # augmentation (NEW)
    augment_trunc: bool = False,
    trunc_horizons: Optional[List[float]] = None,
    trunc_per_curve: int = 3,
    trunc_seed: int = 123,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    inputs (wide CSVs) -> raw_merged.csv -> final_merged.csv -> meta.csv
    Returns: (raw_merged_df, final_merged_df, meta_df)
    """
    logging.basicConfig(level=getattr(logging, loglevel.upper(), logging.INFO))
    log = logging.getLogger("merge_preprocess_meta")

    df_in = load_and_concat_wides(inputs)

    # Interpolate to common grid (no extrapolation) with fixed tmax if provided
    raw_merged_df = build_raw_merged(
    df_in,
    step_hours=step,
    min_points=min_points,
    low_res_threshold=low_res_threshold,
    tmax_hours=tmax_hours,
    auto_tmax=auto_tmax,
    auto_tmax_coverage=auto_tmax_coverage,
    )


    # (NEW) truncation augmentation BEFORE preprocessing/meta
    if augment_trunc:
        from growthqa.preprocess.truncation_augment import augment_df

        hs = trunc_horizons or [8, 10, 12, 14.75, 16]
        raw_merged_df = augment_df(
            raw_merged_df,
            candidate_horizons=hs,
            per_curve=trunc_per_curve,
            seed=trunc_seed,
            full_horizon=float(tmax_hours or 16.0),
        )
        log.info("Applied truncation augmentation: per_curve=%s horizons=%s", trunc_per_curve, hs)

    # Blank status map (optional)
    blank_status_map = None
    if blank_status_csv:
        blank_status_map = load_blank_status_map(blank_status_csv)

    # Preprocess: blank/smooth/normalize on observed region
    final_merged_df, _had_outliers = preprocess_wide(
        raw_merged_df,
        # min_points=min_points,
        # low_res_threshold=low_res_threshold,
        blank_subtracted=blank_subtracted,
        clip_negatives=clip_negatives,
        global_blank=global_blank,
        blank_status_map=blank_status_map,
        blank_default=blank_default,
        smooth_method=smooth_method,
        smooth_window=smooth_window,
        normalize_mode=str(normalize),
    )

    meta_df = build_metadata_from_wide(final_merged_df, rich_meta=bool(rich_meta))

    del add_audit_meta_label
    if "meta_label" in meta_df.columns:
        meta_df = meta_df.drop(columns=["meta_label"])

    # Save outputs
    if out_raw:
        Path(out_raw).parent.mkdir(parents=True, exist_ok=True)
        raw_merged_df.to_csv(out_raw, index=False)
    if out_final:
        Path(out_final).parent.mkdir(parents=True, exist_ok=True)
        final_merged_df.to_csv(out_final, index=False)
    if out_meta:
        Path(out_meta).parent.mkdir(parents=True, exist_ok=True)
        meta_df.to_csv(out_meta, index=False)

    return raw_merged_df, final_merged_df, meta_df
