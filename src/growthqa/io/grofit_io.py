from __future__ import annotations

import numpy as np
import pandas as pd

from growthqa.preprocess.timegrid import parse_time_from_header


def _sorted_time_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if parse_time_from_header(str(c)) is not None]
    return sorted(
        cols,
        key=lambda c: parse_time_from_header(str(c)) if parse_time_from_header(str(c)) is not None else float("inf"),
    )


def _with_curve_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Test Id" not in out.columns:
        return out
    if "curve_id" in out.columns:
        return out
    if "Concentration" in out.columns:
        conc_num = pd.to_numeric(out["Concentration"], errors="coerce")
        out["Concentration"] = conc_num
        conc_txt = conc_num.map(lambda v: "" if pd.isna(v) else f"{float(v):g}")
        out["curve_id"] = out["Test Id"].astype(str) + "|" + conc_txt.astype(str)
    else:
        out["Concentration"] = np.nan
        out["curve_id"] = out["Test Id"].astype(str)
    return out


def build_grofit_input_df(
    wide_original_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    raw_extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    wide = _with_curve_id(wide_original_df if isinstance(wide_original_df, pd.DataFrame) else pd.DataFrame())
    audit = _with_curve_id(audit_df if isinstance(audit_df, pd.DataFrame) else pd.DataFrame())
    if wide.empty:
        return pd.DataFrame()

    time_cols = _sorted_time_cols(wide)
    merge_keys = ["Test Id", "Concentration"] if ("Concentration" in wide.columns and "Concentration" in audit.columns) else ["curve_id"]

    # Pull classifier result columns from audit into grofit input.
    # "Pred Label"  = pipeline's final prediction (always present after v17).
    # "Reviewed"    = duplicated from Pred Label by default; overwritten in MANUAL mode.
    # "True Label"  = in MANUAL mode the user-reviewed label; in AUTO mode same as Pred Label.
    label_cols_wanted = ["Pred Label", "True Label", "Reviewed"]
    label_cols = [c for c in label_cols_wanted if c in audit.columns]

    out = wide.copy()
    if label_cols:
        out = out.merge(audit[merge_keys + label_cols].drop_duplicates(subset=merge_keys),
                        on=merge_keys, how="left")
    if "Pred Label" not in out.columns:
        out["Pred Label"] = np.nan
    if "True Label" not in out.columns:
        # True Label mirrors Pred Label; MANUAL mode review will overwrite it.
        out["True Label"] = out["Pred Label"]
    if "Reviewed" not in out.columns:
        out["Reviewed"] = False

    if raw_extra_cols is None:
        reserved = {"Test Id", "Concentration", "curve_id", "Pred Label", "True Label", "Reviewed", *time_cols}
        raw_extra_cols = [c for c in wide.columns if c not in reserved]
    extra_cols = [c for c in raw_extra_cols if c in out.columns]

    ordered = [c for c in ["Test Id", "Concentration", "Pred Label", "True Label", "Reviewed", "curve_id"]
               if c in out.columns]
    ordered += [c for c in time_cols if c in out.columns]
    ordered += [c for c in extra_cols if c not in ordered]
    return out[ordered].copy()