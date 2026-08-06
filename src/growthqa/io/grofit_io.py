from __future__ import annotations

import numpy as np
import pandas as pd

from growthqa.preprocess.timegrid import get_sorted_time_columns

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
    # raw_extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    wide = _with_curve_id(wide_original_df if isinstance(wide_original_df, pd.DataFrame) else pd.DataFrame())
    audit = _with_curve_id(audit_df if isinstance(audit_df, pd.DataFrame) else pd.DataFrame())
    if wide.empty:
        return pd.DataFrame()

    # time_cols = _sorted_time_cols(wide)
    time_cols = get_sorted_time_columns(wide)
    merge_keys = ["Test Id", "Concentration"] if ("Concentration" in wide.columns and "Concentration" in audit.columns) else ["curve_id"]

    out = wide.copy()
    if "True Label" in audit.columns:
        out = out.merge(audit[merge_keys + ["True Label"]].drop_duplicates(subset=merge_keys),
                        on=merge_keys, how="left")
    else:
        out["True Label"] = np.nan

    ordered = [c for c in ["Test Id", "Concentration", "True Label"] if c in out.columns]
    ordered += [c for c in time_cols if c in out.columns]
    return out[ordered].copy()