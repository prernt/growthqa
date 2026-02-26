from __future__ import annotations

from typing import Iterable
import re

import numpy as np
import pandas as pd

from growthqa.preprocess.timegrid import parse_time_from_header


# ============================================================
# Meta features (unchanged)
# ============================================================
AUDIT_META_FEATURES: list[str] = [
    "raw_observed_tmax",
    "observed_tmax",
    "n_points_observed",
    "max_gap_hours",
    "missing_frac_on_grid",
    "auc_per_hour",
    "net_change_per_hour",
    "max_slope",
    "lag_time_est",
    "dip_fraction",
    "largest_drop_frac",
    "monotonicity_fraction",
    "roughness",
    "final_to_peak_ratio",
]

# ============================================================
# NEW Stage-2 evidence columns (thesis-friendly)
# ============================================================
AUDIT_LATE_FEATURES: list[str] = [
    "has_late_data",
    "late_window_start",
    "late_n_points",
    "late_span_hours",
    "data_quality",
    "growth_z_like",
    "artifact_score",
    "decision_confidence",
    "Stage 2 Label",
    "Label Reason",
]

# Optional extra diagnostics (keep out of biologist-facing CSV)
AUDIT_LATE_DEBUG_FEATURES: list[str] = [
    "late_slope",
    "late_delta",
    "noise_level",
    "late_growth_detected",
    "artifact_detected",
]

def _col_as_series(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)

def _sorted_time_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if parse_time_from_header(str(c)) is not None]
    return sorted(
        cols,
        key=lambda c: parse_time_from_header(str(c)) if parse_time_from_header(str(c)) is not None else float("inf"),
    )


def _test_id_encodes_conc(s: object) -> bool:
    if s is None:
        return False
    return re.search(r"\[(?:\s*Conc\s*=\s*)?([0-9]+(?:\.[0-9]+)?)\s*\]", str(s), flags=re.IGNORECASE) is not None


def _with_curve_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps your existing curve_id/curve_key logic style.
    """
    out = df.copy()
    if "Test Id" not in out.columns:
        return out

    out["Test Id"] = out["Test Id"].astype(str)

    if "Concentration" in out.columns:
        out["Concentration"] = pd.to_numeric(out["Concentration"], errors="coerce")
        conc_txt = out["Concentration"].map(lambda v: f"{float(v):g}" if np.isfinite(v) else "")
        has_conc = out["Concentration"].notna()
        enc = out["Test Id"].map(_test_id_encodes_conc)

        out["curve_id"] = out["Test Id"]
        out["curve_key"] = out["Test Id"]
        use_append = has_conc & (~enc)
        out.loc[use_append, "curve_key"] = out.loc[use_append, "Test Id"] + "||" + conc_txt.loc[use_append]
    else:
        out["curve_id"] = out["Test Id"]
        out["curve_key"] = out["Test Id"]

    return out


def build_classifier_audit_df(
    *,
    wide_original_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    mode: str,
    review_df: pd.DataFrame | None = None,
    processed_wide_df: pd.DataFrame | None = None,
    include_debug_features: bool = False,
) -> pd.DataFrame:
    """
    Builds the classifier audit CSV dataframe.

    Assumptions:
      - infer_df contains the Stage-2 evidence columns from the new pipeline.
      - infer_df contains: Pred Label, Pred Confidence, Predicted S1 Label, Stage 2 Label, Label Reason, Reviewed (MANUAL)
    """
    if "Test Id" not in wide_original_df.columns:
        raise ValueError("wide_original_df must contain 'Test Id'.")

    # Time columns from original
    time_cols = _sorted_time_cols(wide_original_df)

    # Normalize IDs
    wide0 = _with_curve_id(wide_original_df)
    infer0 = _with_curve_id(infer_df)
    meta0 = _with_curve_id(meta_df)

    # Base columns expected (be tolerant)
    base_infer_cols = [
        "Test Id",
        "Concentration",
        "curve_key",
        "curve_id",
        "Pred Label",
        "Pred Confidence",
        "Predicted S1 Label",
        "S1 Confidence Valid",
        "Stage 2 Label",
        "Label Reason",
    ]

    if mode.upper() == "MANUAL":
        base_infer_cols.append("Reviewed")

    # Join infer onto wide (for time-series)
    df = wide0.merge(infer0, on="curve_key", how="left", suffixes=("", "_infer"))

    # Join meta features (optional)
    meta_cols = [c for c in AUDIT_META_FEATURES if c in meta0.columns]
    if meta_cols:
        df = df.merge(meta0[["curve_key"] + meta_cols], on="curve_key", how="left", suffixes=("", "_meta"))

    # Stage-2 cols
    late_cols = [c for c in AUDIT_LATE_FEATURES if c in df.columns]
    if include_debug_features:
        late_cols += [c for c in AUDIT_LATE_DEBUG_FEATURES if c in df.columns]
    
    proc_added_cols: list[str] = []
    if isinstance(processed_wide_df, pd.DataFrame) and not processed_wide_df.empty:
        proc0 = _with_curve_id(processed_wide_df)
        proc_time_cols = _sorted_time_cols(proc0)
        if proc_time_cols:
            # merge on curve_key if present in proc0
            if "curve_key" in proc0.columns:
                proc_keys = ["curve_key"]
            else:
                # fallback to Test Id if curve_key not available
                proc_keys = ["Test Id"] if "Test Id" in proc0.columns else []

            if proc_keys:
                rename_map: dict[str, str] = {}
                for c in proc_time_cols:
                    new_c = c if str(c).startswith("P_") else f"P_{c}"
                    if new_c in df.columns:
                        new_c = f"{new_c}_processed"
                    rename_map[c] = new_c
                    proc_added_cols.append(new_c)

                proc_slice = proc0[proc_keys + proc_time_cols].rename(columns=rename_map)
                df = df.merge(proc_slice, on=proc_keys, how="left")
    if "Reviewed" not in df.columns:
        df["Reviewed"] = False

    # Merge manual review updates if provided (MANUAL mode)
    if isinstance(review_df, pd.DataFrame) and not review_df.empty:
        review0 = _with_curve_id(review_df)
        # Prefer curve_key merge if possible
        if "curve_key" in df.columns and "curve_key" in review0.columns:
            rk = ["curve_key"]
        else:
            rk = ["Test Id"] if ("Test Id" in df.columns and "Test Id" in review0.columns) else []

        if rk:
            # Accept either 'true_label' or 'True Label' in review_df
            review_label_col = "true_label" if "true_label" in review0.columns else ("True Label" if "True Label" in review0.columns else None)
            review_cols = [c for c in ["Reviewed", review_label_col] if c and c in review0.columns]
            if review_cols:
                df = df.merge(review0[rk + review_cols], on=rk, how="left", suffixes=("", "_review"))

                # Apply reviewed flag update
                if "Reviewed_review" in df.columns:
                    df["Reviewed"] = _col_as_series(df, "Reviewed_review").combine_first(_col_as_series(df, "Reviewed", False))
                    df.drop(columns=["Reviewed_review"], inplace=True)

                # Apply label update into a canonical internal column
                if review_label_col is not None:
                    col_name = f"{review_label_col}_review"
                    if col_name in df.columns:
                        df["true_label"] = _col_as_series(df, col_name).combine_first(_col_as_series(df, "true_label", np.nan))
                        df.drop(columns=[col_name], inplace=True)

    # True Label:
    # - MANUAL: should reflect human label when available (true_label), else Pred Label
    # - AUTO: mirrors Pred Label (no ground truth yet)
    if "Pred Label" not in df.columns:
        df["Pred Label"] = _col_as_series(df, "final_label", _col_as_series(df, "pred_label", ""))

    df["True Label"] = _col_as_series(df, "true_label", df["Pred Label"])

    # normalize Reviewed to boolean
    df["Reviewed"] = _col_as_series(df, "Reviewed", False).fillna(False).astype(bool)

    # Authoritative sparse override for audit/export semantics.
    sparse_mask = pd.to_numeric(_col_as_series(df, "too_sparse", 0), errors="coerce").fillna(0).astype(int).eq(1)
    if sparse_mask.any():
        df.loc[sparse_mask, "Pred Label"] = "Unsure"
        df.loc[sparse_mask, "True Label"] = "Unsure"
        if "Label Reason" in df.columns:
            df.loc[sparse_mask, "Label Reason"] = "TOO_SPARSE_OVERRIDE"

    ordered: list[str] = []

        # ----------------------------------------------------
    # Curve ID column cleanup (clean debug philosophy)
    # ----------------------------------------------------
    if "curve_id" in df.columns and "curve_key" in df.columns:
        same_mask = (
            df["curve_id"].astype(str).fillna("") 
            == df["curve_key"].astype(str).fillna("")
        )

        if same_mask.all():
            # They are identical everywhere → keep only curve_id
            df = df.drop(columns=["curve_key"])
        else:
            # They differ somewhere → keep both (debug needed)
            pass
    for must in ["Test Id", "Concentration", "curve_key", "Pred Label", "Pred Confidence",
                 "Predicted S1 Label", "S1 Confidence Valid", "S1 Confidence Invalid",
                 "Stage 2 Label", "Label Reason", "True Label", "Reviewed"]:
        if must in df.columns and must not in ordered:
            ordered.insert(min(len(ordered), 3) if must in {"curve_key"} else len(ordered), must)
    
    ordered += [c for c in base_infer_cols if c in df.columns]
    ordered += [c for c in meta_cols if c in df.columns]
    ordered += [c for c in late_cols if c in df.columns]
    ordered += [c for c in time_cols if c in df.columns]

 # De-duplicate while preserving order
    seen = set()
    ordered_unique = []
    for c in ordered:
        if c not in seen:
            ordered_unique.append(c)
            seen.add(c)

    return df[ordered_unique].copy()
