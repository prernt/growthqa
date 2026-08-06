from __future__ import annotations
import re
import pandas as pd
from growthqa.preprocess.timegrid import get_time_columns, parse_time_from_header

_CONC_ID_RE = re.compile(
    r"\[(?:\s*Conc\s*=\s*)?([0-9]+(?:\.[0-9]+)?)\s*\]",
    flags=re.IGNORECASE,
)

_CONC_COL_CANDIDATES = [
    "concentration", "Concentration", "conc", "Conc",
    "dose", "Dose", "drug_conc", "DrugConc",
]


def extract_conc_from_curve_id(curve_id: str) -> float | None:
    if curve_id is None:
        return None
    m = _CONC_ID_RE.search(str(curve_id))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None
    

def find_concentration_col(df: pd.DataFrame) -> str | None:
    for c in _CONC_COL_CANDIDATES:
        if c in df.columns:
            return c
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in _CONC_COL_CANDIDATES:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def wide_to_grofit_tidy(
    wide_df: pd.DataFrame, *, file_tag: str, test_id_col: str = "Test Id",
)-> pd.DataFrame:
    if test_id_col not in wide_df.columns:
        raise ValueError(f"Expected '{test_id_col}' column in wide input.")
    time_cols = get_time_columns(wide_df)
    if not time_cols:
        raise ValueError("No time columns found (expected T#.## (h) headers).")

    conc_col = find_concentration_col(wide_df)
    id_vars = [test_id_col] + ([conc_col] if conc_col else [])

    tidy = wide_df.melt(id_vars=id_vars, value_vars=time_cols, var_name="_tl", value_name="y")
    tidy["time"] = tidy["_tl"].map(lambda s: float(parse_time_from_header(str(s))))
    tidy["test_id"] = str(file_tag)
    tidy["curve_id"] = tidy[test_id_col].astype(str)
    tidy.drop(columns=["_tl"], inplace=True)

    if conc_col is None:
        tidy["concentration"] = tidy[test_id_col].astype(str).map(extract_conc_from_curve_id)
    else:
        tidy["concentration"] = pd.to_numeric(tidy[conc_col], errors="coerce")
    tidy["concentration"] = pd.to_numeric(tidy["concentration"], errors="coerce").fillna(0.0)
    tidy["y"] = pd.to_numeric(tidy["y"], errors="coerce")
    tidy = tidy.dropna(subset=["time", "y"])
    return tidy[["test_id", "curve_id", "concentration", "time", "y"]]