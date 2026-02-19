# # src/growthqa/pipelines/build_meta_dataset.py
# from __future__ import annotations

# import logging
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import pandas as pd
# from growthqa.io.wide_loader import load_and_concat_wides
# from growthqa.preprocess.blank_status import load_blank_status_map
# from growthqa.preprocess.interpolate import build_raw_merged
# from growthqa.preprocess.transform import preprocess_wide
# from growthqa.features.meta import build_metadata_from_wide
# from growthqa.features.meta_label import add_meta_label

# def run_merge_preprocess_meta(
#     *,
#     inputs: List[str],
#     out_raw: Optional[str],
#     out_final: Optional[str],
#     out_meta: Optional[str],
#     # interpolation/grid
#     step: float = 0.25,
#     min_points: int = 3,
#     low_res_threshold: int = 7,
#     tmax_hours: Optional[float] = None,
#     auto_tmax: bool = False,
#     auto_tmax_coverage: float = 0.8,
#     # blank/baseline
#     blank_subtracted: bool = False,
#     clip_negatives: bool = False,
#     global_blank: Optional[float] = None,
#     blank_status_csv: Optional[str] = None,
#     blank_default: str = "RAW",  # RAW or ALREADY
#     # smoothing + normalization
#     smooth_method: str = "NONE",  # NONE, RAW, LWS, SGF
#     smooth_window: int = 5,
#     normalize: str = "NONE",  # NONE, MAX, MINMAX
#     # logging
#     loglevel: str = "INFO",
#     # audit label
#     add_audit_meta_label: bool = True,
#     # augmentation
#     augment_trunc: bool = False,
#     trunc_horizons: Optional[List[float]] = None,
#     trunc_per_curve: int = 3,
#     trunc_seed: int = 123,
# ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """
#     Orchestrator for:
#       inputs (wide CSVs) -> raw_merged.csv -> final_merged.csv -> meta.csv

#     Returns:
#       (raw_merged_df, final_merged_df, meta_df)

#     Notes:
#       - This is the library replacement of the old merge_meta.py "main()".
#       - CLI and UI should call THIS, not internal modules directly.
#     """

#     # ---- logging ----
#     try:
#         level = getattr(logging, str(loglevel).upper())
#     except Exception:
#         level = logging.INFO
#     logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

#     # ---- normalize some string params ----
#     blank_default = str(blank_default).upper().strip()
#     smooth_method = str(smooth_method).upper().strip()
#     normalize = str(normalize).upper().strip()

#     # ---- ensure output dirs exist ----
#     if out_raw:
#         Path(out_raw).parent.mkdir(parents=True, exist_ok=True)
#     if out_final:
#         Path(out_final).parent.mkdir(parents=True, exist_ok=True)
#     if out_meta:
#         Path(out_meta).parent.mkdir(parents=True, exist_ok=True)

#     # ---- 1) load all wide inputs ----
#     df_all = load_and_concat_wides(inputs)
#     logging.info(f"Loaded wide inputs: rows={len(df_all)} cols={df_all.shape[1]}")

#     # ---- 2) blank status map (optional) ----
#     blank_status_map: Dict[str, Dict[str, object]] = load_blank_status_map(blank_status_csv)
#     logging.info(f"Blank-status map entries: {len(blank_status_map)}")

#     # ---- 3) Stage 1: interpolation only -> raw_merged ----
#     raw_merged = build_raw_merged(
#         df_all_wide=df_all,
#         step_hours=float(step),
#         min_points=int(min_points),
#         tmax_hours=tmax_hours,
#         auto_tmax=bool(auto_tmax),
#         auto_tmax_coverage=float(auto_tmax_coverage),
#         low_res_threshold=int(low_res_threshold),
#     )
#     if augment_trunc:
#         from growthqa.preprocess.augment import make_truncated_variants
#         hs = trunc_horizons or [8, 10, 12, 14.75, 16]
#         raw_merged_df = make_truncated_variants(
#             raw_merged_df,
#             horizons=hs,
#             per_curve=trunc_per_curve,
#             seed=trunc_seed,
#             id_col="Test Id",
#     )

#     if out_raw:
#         raw_merged.to_csv(out_raw, index=False)
#         logging.info(f"Wrote raw merged: {out_raw}  rows={len(raw_merged)} cols={raw_merged.shape[1]}")

#     # ---- 4) Stage 2: preprocessing -> final_merged ----
#     final_merged, _had_outliers = preprocess_wide(
#         raw_wide=raw_merged,
#         blank_subtracted=bool(blank_subtracted),
#         global_blank=global_blank,
#         smooth_method=str(smooth_method),
#         smooth_window=int(smooth_window),
#         clip_negatives=bool(clip_negatives),
#         blank_status_map=blank_status_map,
#         blank_default=blank_default,
#         normalize_mode=str(normalize),
#     )
#     if out_final:
#         final_merged.to_csv(out_final, index=False)
#         logging.info(f"Wrote final merged: {out_final}  rows={len(final_merged)} cols={final_merged.shape[1]}")

#     # ---- 5) Stage 3: meta-features ----
#     meta = build_metadata_from_wide(final_merged)

#     # ---- 6) audit-only meta_label ----
#     if add_audit_meta_label:
#         meta = add_meta_label(meta)

#         # optional logging summary
#         if "meta_label" in meta.columns:
#             dist = meta["meta_label"].value_counts(dropna=False).to_dict()
#             logging.info(f"meta_label distribution: {dist}")

#             # quick audit crosstab if Is_Valid exists
#             if "Is_Valid" in meta.columns:
#                 try:
#                     audit = pd.crosstab(meta["Is_Valid"], meta["meta_label"])
#                     logging.info("Audit crosstab Is_Valid vs meta_label:\n" + audit.to_string())
#                 except Exception:
#                     pass

#     if out_meta:
#         meta.to_csv(out_meta, index=False)
#         logging.info(f"Wrote meta: {out_meta}  rows={len(meta)} cols={meta.shape[1]}")

#     return raw_merged, final_merged, meta
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
