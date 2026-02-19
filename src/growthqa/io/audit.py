from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from growthqa.preprocess.timegrid import parse_time_from_header
import re


AUDIT_META_FEATURES: list[str] = [
    "early_observed_tmax",
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

AUDIT_LATE_FEATURES: list[str] = [
    "has_late_data",
    "late_min_points",
    "late_window_start",
    "late_tmax",
    "late_n_points",
    "late_too_sparse",
    "late_slope",
    "late_delta",
    "late_max_increase",
    "late_growth_detected",
    "plateau_detected",
    "decline_detected",
    "drift_detected",
    "noise_detected",
    "sigma_noise",
    "late_linearity_r2",
]


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
    out = df.copy()
    if "Test Id" not in out.columns:
        return out
    if "Concentration" in out.columns:
        conc_num = pd.to_numeric(out["Concentration"], errors="coerce")
        out["Concentration"] = conc_num
        conc_txt = conc_num.map(lambda v: "" if pd.isna(v) else f"{float(v):g}")
        out["curve_id"] = out["Test Id"].astype(str) + "|" + conc_txt.astype(str)
        enc = out["Test Id"].map(_test_id_encodes_conc)

        out["curve_key"] = out["Test Id"].astype(str)
        has_conc = conc_num.notna()

        # Only append "||conc" when Test Id does NOT already encode it
        use_append = has_conc & (~enc)
        out.loc[use_append, "curve_key"] = out.loc[use_append, "Test Id"].astype(str) + "||" + conc_txt.loc[use_append].astype(str)

        # out["curve_key"] = out["Test Id"].astype(str)
        # has_conc = conc_num.notna()
        # out.loc[has_conc, "curve_key"] = out.loc[has_conc, "Test Id"].astype(str) + "||" + conc_txt.loc[has_conc].astype(str)
        return out

    out["Concentration"] = np.nan
    dup_rank = out.groupby("Test Id").cumcount()
    out["curve_id"] = out["Test Id"].astype(str)
    out["curve_key"] = out["Test Id"].astype(str)
    out.loc[dup_rank > 0, "curve_id"] = out.loc[dup_rank > 0, "curve_id"] + "|" + dup_rank.loc[dup_rank > 0].astype(str)
    return out


def _pick_merge_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    if "curve_key" in left.columns and "curve_key" in right.columns:
        return ["curve_key"]
    if all(c in left.columns and c in right.columns for c in ["Test Id", "Concentration"]):
        l_has_conc = pd.to_numeric(left["Concentration"], errors="coerce").notna().any()
        r_has_conc = pd.to_numeric(right["Concentration"], errors="coerce").notna().any()
        if l_has_conc and r_has_conc:
            return ["Test Id", "Concentration"]
    return ["curve_id"] if "curve_id" in left.columns and "curve_id" in right.columns else ["Test Id"]


def _normalize_label(v: object) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip().lower()
    if s in {"valid", "true", "1"}:
        return "Valid"
    if s in {"invalid", "false", "0"}:
        return "Invalid"
    if s in {"unsure", "unknown"}:
        return "Unsure"
    return str(v).strip()


def _unique_cols(cols: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def build_classifier_audit_df(
    wide_original_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    mode: str,
    review_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    mode_u = str(mode).strip().upper()
    manual_mode = mode_u == "MANUAL"

    wide = _with_curve_id(wide_original_df if isinstance(wide_original_df, pd.DataFrame) else pd.DataFrame())
    infer = _with_curve_id(infer_df if isinstance(infer_df, pd.DataFrame) else pd.DataFrame())
    meta = _with_curve_id(meta_df if isinstance(meta_df, pd.DataFrame) else pd.DataFrame())

    if wide.empty:
        return pd.DataFrame()

    time_cols = _sorted_time_cols(wide)
    keys = _pick_merge_keys(wide, infer)
    df = wide.copy()
    base_infer_cols = [
    "Pred Label",
    "Pred Confidence",
    "Predicted S1 Label",
    "S1 Confidence Valid",
    "S1 Confidence Invalid",
    "Stage 2 Label",
    "Label Reason",
    "true_label",
    "Reviewed",
    "pred_label",
    "pred_confidence",
    "confidence_valid",
    "confidence_invalid",
    "final_label",
    "final_reason",
    ]

    stage2_infer_cols = [
        # Late-window availability + features produced by infer_labels.py after Stage-2 merge
        "has_late_data",
        "late_min_points",
        "raw_observed_tmax",
        "late_window_start",
        "late_tmax",
        "late_n_points",
        "late_too_sparse",
        "late_slope",
        "late_delta",
        "late_max_increase",
        "late_growth_detected",
        "plateau_detected",
        "decline_detected",
        "drift_detected",
        "noise_detected",
    ]

    infer_cols = [c for c in (base_infer_cols + stage2_infer_cols) if c in infer.columns]

    if infer_cols:
        df = df.merge(infer[keys + infer_cols], on=keys, how="left")

    if not meta.empty:
        meta_cols = [c for c in _unique_cols(AUDIT_META_FEATURES) if c in meta.columns]
        if meta_cols:
            df = df.merge(meta[keys + meta_cols], on=keys, how="left", suffixes=("", "_meta"))

    # Normalize inference output columns
    if "Pred Label" not in df.columns:
        df["Pred Label"] = df.get("final_label", df.get("pred_label", ""))
    if "Pred Confidence" not in df.columns:
        df["Pred Confidence"] = pd.to_numeric(df.get("pred_confidence", np.nan), errors="coerce")
    if "Predicted S1 Label" not in df.columns:
        df["Predicted S1 Label"] = df.get("pred_label", "")
    if "S1 Confidence Valid" not in df.columns:
        df["S1 Confidence Valid"] = pd.to_numeric(df.get("confidence_valid", np.nan), errors="coerce")
    if "S1 Confidence Invalid" not in df.columns:
        df["S1 Confidence Invalid"] = pd.to_numeric(df.get("confidence_invalid", np.nan), errors="coerce")
    if "Stage 2 Label" not in df.columns:
        df["Stage 2 Label"] = np.nan
    if "Label Reason" not in df.columns:
        df["Label Reason"] = np.nan

    for c in AUDIT_LATE_FEATURES:
        if c not in df.columns:
            df[c] = np.nan

    df["Pred Label"] = df["Pred Label"].apply(_normalize_label)
    df["Predicted S1 Label"] = df["Predicted S1 Label"].apply(_normalize_label)
    df["Pred Confidence"] = pd.to_numeric(df["Pred Confidence"], errors="coerce")
    df["S1 Confidence Valid"] = pd.to_numeric(df["S1 Confidence Valid"], errors="coerce")
    df["S1 Confidence Invalid"] = pd.to_numeric(df["S1 Confidence Invalid"], errors="coerce")
    # Do not fabricate late-data state in audit; reflect inference output as-is.
    df["has_late_data"] = df["has_late_data"].astype("boolean")

    no_late_mask = df["has_late_data"].eq(False).fillna(False)
    df.loc[no_late_mask, "Stage 2 Label"] = np.nan
    df.loc[no_late_mask, "Label Reason"] = np.nan
    for c in [
        "late_window_start",
        "late_tmax",
        "late_n_points",
        "late_too_sparse",
        "late_slope",
        "late_delta",
        "late_max_increase",
        "late_growth_detected",
        "sigma_noise",
        "late_linearity_r2",
    ]:
        df.loc[no_late_mask, c] = np.nan

    # Review / true label columns
    if manual_mode and isinstance(review_df, pd.DataFrame) and not review_df.empty:
        review = _with_curve_id(review_df)
        rk = _pick_merge_keys(df, review)
        review_cols = [c for c in ["true_label", "Reviewed"] if c in review.columns]
        if review_cols:
            df = df.merge(review[rk + review_cols], on=rk, how="left", suffixes=("", "_review"))
            if "true_label_review" in df.columns:
                base_true = df["true_label"] if "true_label" in df.columns else pd.Series(index=df.index, dtype=object)
                df["true_label"] = df["true_label_review"].combine_first(base_true)
                df.drop(columns=["true_label_review"], inplace=True)
            if "Reviewed_review" in df.columns:
                base_reviewed = df["Reviewed"] if "Reviewed" in df.columns else pd.Series(index=df.index, dtype=object)
                df["Reviewed"] = df["Reviewed_review"].combine_first(base_reviewed)
                df.drop(columns=["Reviewed_review"], inplace=True)

    df["True Label"] = df.get("true_label", df["Pred Label"]).apply(_normalize_label)
    if manual_mode:
        df["Reviewed"] = df.get("Reviewed", False).fillna(False).astype(bool)

    ordered = [
        "Test Id",
        "Concentration",
        "curve_key",
        "curve_id",
        "Pred Label",
        "Pred Confidence",
        "Predicted S1 Label",
        "S1 Confidence Valid",
        "S1 Confidence Invalid",
        "Stage 2 Label",
        "Label Reason",
        "True Label",
    ]
    if manual_mode:
        ordered.append("Reviewed")
    ordered += [c for c in AUDIT_META_FEATURES if c in df.columns]
    ordered += [c for c in AUDIT_LATE_FEATURES if c in df.columns]
    ordered += [c for c in time_cols if c in df.columns]

    ordered = _unique_cols([c for c in ordered if c in df.columns])
    return df[ordered].copy()
