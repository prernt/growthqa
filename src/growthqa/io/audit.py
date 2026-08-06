from __future__ import annotations
import re
import numpy as np
import pandas as pd
from growthqa.preprocess.timegrid import get_sorted_time_columns
from growthqa.config import MAX_GAP_HOURS_OVERRIDE, MISSING_FRAC_OVERRIDE
from app.utils import resolve_display_label


AUDIT_META_FEATURES: list[str] = [
    "raw_observed_tmax",
    "observed_tmax",
    "n_points_observed",
    "max_gap_hours",
    "missing_frac_on_grid",
    "too_sparse",
    "grid_resolution_mismatch",
    "auc_per_hour",
    "net_change_per_hour",
    "max_slope",
    "lag_time_est",
    "largest_drop_frac",
    "monotonicity_fraction",
    "roughness",
    "final_OD",
    "initial_OD",              
    "growth_phase_duration",   
    "multi_phase_flag",        
    "noise_residual_std",     
    "noise_residual_std_is_fallback",
    "n_features_missing",
    "feature_missing_frac",

]


AUDIT_LATE_FEATURES: list[str] = [
    "has_late_data",
    "late_n_points",
    "min_late_points_required",
    "late_span_hours",
    "data_quality",
    "growth_z_like",
    "artifact_score",
    "decision_confidence",
    "Stage 2 Label",
    "Label Reason",
]

AUDIT_LATE_DEBUG_FEATURES: list[str] = [
    "late_slope",
    "late_delta",
    "noise_level",
    "decline_score",
    "late_growth_detected",
    "artifact_detected",
]

def _col_as_series(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _test_id_encodes_conc(s: object) -> bool:
    if s is None:
        return False
    return re.search(r"\[(?:\s*Conc\s*=\s*)?([0-9]+(?:\.[0-9]+)?)\s*\]", str(s), flags=re.IGNORECASE) is not None


def _with_curve_id(df: pd.DataFrame, *, concentration_lookup: dict | None = None) -> pd.DataFrame:
    out = df.copy()
    if "Test Id" not in out.columns:
        return out

    out["Test Id"] = out["Test Id"].astype(str)

    if "Concentration" in out.columns:
        out["Concentration"] = pd.to_numeric(out["Concentration"], errors="coerce")
    elif concentration_lookup:
        out["Concentration"] = out["Test Id"].map(concentration_lookup)
        out["Concentration"] = pd.to_numeric(out["Concentration"], errors="coerce")

    if "Concentration" in out.columns:
        conc_txt = out["Concentration"].map(lambda v: f"{float(v):g}" if np.isfinite(v) else "")
        has_conc = out["Concentration"].notna()
        enc = out["Test Id"].map(_test_id_encodes_conc)
        out["curve_key"] = out["Test Id"]
        use_append = has_conc & (~enc)
        out.loc[use_append, "curve_key"] = out.loc[use_append, "Test Id"] + "||" + conc_txt.loc[use_append]
    else:
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
    if "Test Id" not in wide_original_df.columns:
        raise ValueError("wide_original_df must contain 'Test Id'.")
    time_cols = get_sorted_time_columns(wide_original_df)
    wide0 = _with_curve_id(wide_original_df)
    conc_lookup = None
    if "Concentration" in wide0.columns:
        conc_lookup = (
            wide0[["Test Id", "Concentration"]]
            .drop_duplicates("Test Id")
            .set_index("Test Id")["Concentration"]
            .to_dict()
        )
    infer0 = _with_curve_id(infer_df, concentration_lookup=conc_lookup)
    meta0 = _with_curve_id(meta_df, concentration_lookup=conc_lookup)
    base_infer_cols = [
        "Test Id",
        "Concentration",
        "curve_key",
        "Pred Label",
        "Pred Confidence",
        "S1 Confidence Valid",
        "Stage 2 Label",
        "Label Reason",
        "Final Label (S1+S2)",
        "True Label",
    ]
    if mode.upper() == "MANUAL":
        base_infer_cols.append("Reviewed")
    df = wide0.merge(infer0, on="curve_key", how="left", suffixes=("", "_infer"))
    meta_cols = [c for c in AUDIT_META_FEATURES if c in meta0.columns]
    if meta_cols:
        df = df.merge(meta0[["curve_key"] + meta_cols], on="curve_key", how="left", suffixes=("", "_meta"))
    late_cols = [c for c in AUDIT_LATE_FEATURES if c in df.columns]
    if include_debug_features:
        late_cols += [c for c in AUDIT_LATE_DEBUG_FEATURES if c in df.columns]
    
    proc_added_cols: list[str] = []
    if isinstance(processed_wide_df, pd.DataFrame) and not processed_wide_df.empty:
        proc0 = _with_curve_id(processed_wide_df)
        proc_time_cols = get_sorted_time_columns(proc0)
        if proc_time_cols:
            if "curve_key" in proc0.columns:
                proc_keys = ["curve_key"]
            else:
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
    if isinstance(review_df, pd.DataFrame) and not review_df.empty:
        review0 = _with_curve_id(review_df, concentration_lookup=conc_lookup)
        if "curve_key" in df.columns and "curve_key" in review0.columns:
            rk = ["curve_key"]
        else:
            rk = ["Test Id"] if ("Test Id" in df.columns and "Test Id" in review0.columns) else []

        if rk:
            review_label_col = "true_label" if "true_label" in review0.columns else ("True Label" if "True Label" in review0.columns else None)
            review_cols = [c for c in ["Reviewed", review_label_col] if c and c in review0.columns]
            if review_cols:
                df = df.merge(review0[rk + review_cols], on=rk, how="left", suffixes=("", "_review"))
                if "Reviewed_review" in df.columns:
                    df["Reviewed"] = _col_as_series(df, "Reviewed_review").combine_first(_col_as_series(df, "Reviewed", False))
                    df.drop(columns=["Reviewed_review"], inplace=True)
                elif "Reviewed" in df.columns:
                    df["Reviewed"] = _col_as_series(df, "Reviewed", False).fillna(False).astype(bool)
                if review_label_col is not None:
                    col_name = f"{review_label_col}_review"
                    merged_label_col = col_name if col_name in df.columns else review_label_col
                    if merged_label_col in df.columns:
                        df["true_label"] = _col_as_series(df, merged_label_col).combine_first(_col_as_series(df, "true_label", np.nan))
                        if merged_label_col != review_label_col:
                            df.drop(columns=[merged_label_col], inplace=True)

    if "Pred Label" not in df.columns:
        df["Pred Label"] = _col_as_series(df, "final_label", _col_as_series(df, "pred_label", ""))

    if "Final Label (S1+S2)" not in df.columns:
        df["Final Label (S1+S2)"] = _col_as_series(df, "final_label", _col_as_series(df, "Pred Label", ""))

    if "true_label" not in df.columns:
        df["true_label"] = df["Final Label (S1+S2)"]

    df["True Label"] = df.apply(lambda r: resolve_display_label(r, fallback=str(r.get("Pred Label", ""))), axis=1)

    if mode.upper() == "MANUAL":
        df["Reviewed"] = _col_as_series(df, "Reviewed", False).fillna(False).astype(bool)
    else:
        df["Reviewed"] = "Disabled"
    ood_gap_mask = (
        pd.to_numeric(_col_as_series(df, "max_gap_hours", np.nan), errors="coerce") > MAX_GAP_HOURS_OVERRIDE
    ) | (
        pd.to_numeric(_col_as_series(df, "missing_frac_on_grid", np.nan), errors="coerce") > MISSING_FRAC_OVERRIDE
    )
    ood_gap_mask = ood_gap_mask.fillna(False)
    if ood_gap_mask.any():
        df.loc[ood_gap_mask, "Pred Label"] = "Unsure"
        df.loc[ood_gap_mask, "True Label"] = "Unsure"
        if "final_label" in df.columns:
            df.loc[ood_gap_mask, "final_label"] = "Unsure"
        if "Final Label (S1+S2)" in df.columns:
            df.loc[ood_gap_mask, "Final Label (S1+S2)"] = "Unsure"
        if "pred_label" in df.columns:
            df.loc[ood_gap_mask, "pred_label"] = "Unsure"
        if "Label Reason" in df.columns:
            df.loc[ood_gap_mask, "Label Reason"] = "OUT_OF_DISTRIBUTION_GAP_OVERRIDE"
    sparse_mask = pd.to_numeric(_col_as_series(df, "too_sparse", 0), errors="coerce").fillna(0).astype(int).eq(1)
    if sparse_mask.any():
        df.loc[sparse_mask, "Pred Label"] = "Unsure"
        df.loc[sparse_mask, "True Label"] = "Unsure"
        if "final_label" in df.columns:
            df.loc[sparse_mask, "final_label"] = "Unsure"
        if "Final Label (S1+S2)" in df.columns:
            df.loc[sparse_mask, "Final Label (S1+S2)"] = "Unsure"
        if "pred_label" in df.columns:
            df.loc[sparse_mask, "pred_label"] = "Unsure"
        if "Label Reason" in df.columns:
            df.loc[sparse_mask, "Label Reason"] = "TOO_SPARSE_OVERRIDE"

    ordered: list[str] = []

    if "curve_id" in df.columns and "curve_key" in df.columns:
        same_mask = (
            df["curve_id"].astype(str).fillna("") == df["curve_key"].astype(str).fillna("")
        )

        if same_mask.all():
            df = df.drop(columns=["curve_key"])
        else:
            pass

    for must in ["Test Id", "Concentration", "curve_key", "Pred Label", "Pred Confidence",
                 "S1 Confidence Valid", "S1 Confidence Invalid",
                 "Stage 2 Label", "Label Reason", "Final Label (S1+S2)", "True Label", "Reviewed"]:
        if must in df.columns and must not in ordered:
            ordered.insert(min(len(ordered), 3) if must == "curve_key" else len(ordered), must)

    ordered += [c for c in base_infer_cols if c in df.columns]
    ordered += [c for c in meta_cols if c in df.columns]
    ordered += [c for c in late_cols if c in df.columns]
    ordered += [c for c in time_cols if c in df.columns]
    seen = set()
    ordered_unique = []
    for c in ordered:
        if c not in seen:
            ordered_unique.append(c)
            seen.add(c)

    return df[ordered_unique].copy()