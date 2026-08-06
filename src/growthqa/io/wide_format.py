from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from growthqa.io.parsers import (
    convert_grofit_v_wide_to_long,
    convert_simple_wide_to_long,
    parse_excel_any,
    parse_time_table_any
)
from growthqa.io.time_parse import make_time_header


def long_to_wide_preserve_times(df_long: pd.DataFrame, file_tag: str, add_prefix: bool = True) -> pd.DataFrame:
    df = df_long.copy()
    if "Is_Valid" not in df.columns:
        df["Is_Valid"] = True
    else:
        df["Is_Valid"] = df["Is_Valid"].fillna(True)

    if add_prefix:
        df["Test Id"] = df["orig_TestId"].astype(str).map(lambda s: f"{file_tag}_{s}")
    else:
        df["Test Id"] = df["orig_TestId"].astype(str)

    group_cols = ["FileName", "Test Id", "Model Name", "Is_Valid", "time_h"]
    if "Concentration" in df.columns:
        group_cols.insert(3, "Concentration")
    df = df.groupby(group_cols, as_index=False, dropna=False)["OD"].mean()

    wide = df.pivot_table(
        index=[c for c in group_cols if c != "time_h"],
        columns="time_h",
        values="OD",
        aggfunc="mean"
    )

    times = np.array(wide.columns.tolist(), dtype=float)
    times_sorted = np.sort(times)
    col_map = {t: make_time_header(float(t)) for t in times_sorted}
    wide = wide.reindex(columns=times_sorted)
    wide.columns = [col_map[t] for t in wide.columns]
    wide = wide.reset_index()
    return wide


def parse_any_file_to_long(path: str) -> pd.DataFrame:
    p = Path(path)
    ext = p.suffix.lower()

    if ext in {".xlsx", ".xls"}:
        return parse_excel_any(path)

    if ext in {".csv"}:
        df = pd.read_csv(path)
        out = parse_time_table_any(path, df=df)
        if out is not None:
            return out
        long = convert_simple_wide_to_long(df, p.stem)
        if long is not None:
            return long
        if any(isinstance(c, str) and re.match(r"^V\d+$", c.strip()) for c in df.columns):
            return convert_grofit_v_wide_to_long(df, p.stem)


        raise ValueError(f"CSV format not recognized (not a time-table and not synthetic-wide): {path}")

    raise ValueError(f"Unsupported file extension: {ext} ({path})")