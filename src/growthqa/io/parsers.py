from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from .long_format import standardize_long
from .time_parse import (
    infer_and_convert_numeric_time_to_hours,
    parse_time_any_to_hours,
    split_condition_and_well,
)


def parse_plate_xlsx(path: str, sheet_name=0) -> Optional[pd.DataFrame]:
    """
    Plate-reader export style (your described format):
      - 'Reading' column (1..60)
      - 'avg. time [s]' column = time in seconds
      - many curve columns like '... (A01)', '... (A02)' etc

    Produces long with:
      FileName, orig_TestId (A01), Model Name (condition text), Is_Valid, time_h, OD
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    if df.empty or df.shape[1] < 3:
        return None

    # --- locate time column robustly ---
    time_col = None
    hinted_unit = None
    for c in df.columns:
        cl = str(c).lower()
        if "avg" in cl and "time" in cl:
            time_col = c
            if "[s" in cl or "sec" in cl:
                hinted_unit = "s"
            elif "min" in cl:
                hinted_unit = "m"
            elif "[h" in cl or "hour" in cl or "hr" in cl:
                hinted_unit = "h"
            break
    # fallback: any column containing 'time'
    if time_col is None:
        for c in df.columns:
            cl = str(c).lower()
            if "time" in cl or "tiempo" in cl:
                time_col = c
                if "[s" in cl or "sec" in cl:
                    hinted_unit = "s"
                elif "min" in cl:
                    hinted_unit = "m"
                elif "[h" in cl or "hour" in cl or "hr" in cl:
                    hinted_unit = "h"
                break
    if time_col is None:
        return None

    # --- ignore non-curve columns ---
    ignore = {time_col}
    for c in df.columns:
        cl = str(c).lower()
        if "reading" in cl:   # Reading 1..60
            ignore.add(c)

    # --- identify curve columns: numeric-ish columns other than ignore ---
    candidate = [c for c in df.columns if c not in ignore]
    curve_cols = []
    for c in candidate:
        s = pd.to_numeric(df[c], errors="coerce")
        # at least 2 finite values => treat as curve
        if np.isfinite(s).sum() >= 2:
            curve_cols.append(c)

    if not curve_cols:
        return None

    # --- time in seconds -> hours ---
    # Your file is explicitly seconds; obey that.
    if hinted_unit == "s" or hinted_unit is None:
        t_h = pd.to_numeric(df[time_col], errors="coerce") / 3600.0
    elif hinted_unit == "m":
        t_h = pd.to_numeric(df[time_col], errors="coerce") / 60.0
    else:
        t_h = pd.to_numeric(df[time_col], errors="coerce")

    base = pd.DataFrame({"time_h": t_h})

    # melt OD columns
    tmp = pd.concat([base, df[curve_cols]], axis=1)
    long = tmp.melt(id_vars=["time_h"], var_name="curve_col", value_name="OD")
    long["OD"] = pd.to_numeric(long["OD"], errors="coerce")
    long = long.dropna(subset=["time_h", "OD"]).copy()

    # create meaningful identifiers
    cond_well = long["curve_col"].map(split_condition_and_well)
    long["Model Name"] = cond_well.map(lambda x: x[0])    # condition string
    long["orig_TestId"] = cond_well.map(lambda x: x[1])   # A01, A02, ...

    long["FileName"] = Path(path).stem
    long["Is_Valid"] = True

    return standardize_long(long[["FileName","orig_TestId","Model Name","Is_Valid","time_h","OD"]], Path(path).stem)


def convert_simple_wide_to_long(df: pd.DataFrame, file_stem: str) -> Optional[pd.DataFrame]:
    """
    Handle simple wide with Test Id + T#.0 (h) columns (CSV or Excel).
    """
    tcols, mode = detect_wide_time_columns(df)
    if mode != "synthetic":
        return None

    conc_col = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"concentration", "conc", "dose", "drug_conc"}:
            conc_col = c
            break

    out = df.copy()
    # Ensure required metadata columns exist independently.
    if "FileName" not in out.columns:
        out["FileName"] = file_stem
    if "Test Id" not in out.columns:
        out["Test Id"] = out.index.astype(str)
    if "Model Name" not in out.columns:
        out["Model Name"] = out["Test Id"]
    if "Is_Valid" not in out.columns:
        out["Is_Valid"] = True
    else:
        # Blank labels in uploaded lab files should default to valid during inference upload parsing.
        out["Is_Valid"] = out["Is_Valid"].fillna(True)

    if conc_col is not None and conc_col != "Concentration":
        out = out.rename(columns={conc_col: "Concentration"})

    id_vars = ["FileName", "Test Id", "Model Name", "Is_Valid"]
    if "Concentration" in out.columns:
        id_vars.append("Concentration")

    long = out.melt(
        id_vars=id_vars,
        value_vars=tcols,
        var_name="time_col",
        value_name="OD",
    )
    long["time_h"] = pd.to_numeric(long["time_col"].str.extract(r"^T(\d+(\.\d+)?)")[0], errors="coerce")
    long = long.dropna(subset=["time_h", "OD"]).copy()
    long = long.rename(columns={"Test Id": "orig_TestId"})
    return standardize_long(long, file_stem)


def parse_time_table_any(path: str, sheet_name=0) -> Optional[pd.DataFrame]:
    """
    Simple table:
      time column + multiple curve columns.
    Works for CSV and Excel.
    """
    if str(path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        df = pd.read_csv(path)

    if df.empty or df.shape[1] < 2:
        return None

    # choose time col by name, else first col
    time_col = None
    for c in df.columns:
        if isinstance(c, str) and ("time" in c.lower() or "tiempo" in c.lower()):
            time_col = c
            break
    if time_col is None:
        time_col = df.columns[0]

    t_raw = df[time_col].map(parse_time_any_to_hours)

    # If mostly numeric, infer unit
    if df[time_col].apply(lambda x: isinstance(x, (int, float, np.integer, np.floating))).mean() > 0.7:
        # try detect unit from header text
        hinted = None
        header = str(time_col).lower()
        if "sec" in header or "[s" in header:
            hinted = "s"
        elif "min" in header or "[m" in header:
            hinted = "m"
        elif "hour" in header or "hr" in header or "[h" in header:
            hinted = "h"
        t_raw = infer_and_convert_numeric_time_to_hours(df[time_col], hinted)

    if pd.to_numeric(t_raw, errors="coerce").isna().all():
        return None

    value_cols = [c for c in df.columns if c != time_col]
    # keep columns with enough numeric values
    keep = []
    for c in value_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if np.isfinite(s).sum() >= 2:
            keep.append(c)
    if not keep:
        return None

    long = pd.DataFrame({"time_h": pd.to_numeric(t_raw, errors="coerce")})
    long = pd.concat([long, df[keep]], axis=1).melt(id_vars=["time_h"], var_name="cond", value_name="OD")
    long["OD"] = pd.to_numeric(long["OD"], errors="coerce")
    long = long.dropna(subset=["time_h", "OD"]).copy()

    long["FileName"] = Path(path).stem
    long["orig_TestId"] = long["cond"].astype(str)
    long["Model Name"] = long["cond"].astype(str)
    long["Is_Valid"] = True
    return standardize_long(long, Path(path).stem)

def parse_excel_block_style(path: str, sheet_name=0) -> Optional[pd.DataFrame]:
    """
    Block-style Excel with repeated 'Tiempo'/'Time' markers across columns.
    """
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    if raw.empty:
        return None

    header_row = None
    for i in range(min(40, len(raw))):
        row = raw.iloc[i].astype(str).str.lower()
        hits = row.str.contains("tiempo|time", regex=True, na=False).sum()
        if hits >= 2:
            header_row = i
            break
    if header_row is None:
        return None

    hdr = raw.iloc[header_row].tolist()
    data = raw.iloc[header_row + 1:].copy()

    starts = []
    for j, v in enumerate(hdr):
        if isinstance(v, str) and ("tiempo" in v.lower() or "time" in v.lower()):
            starts.append(j)
    if not starts:
        return None
    starts.append(len(hdr))

    outs = []
    for bi in range(len(starts) - 1):
        s, e = starts[bi], starts[bi + 1]
        sub = data.iloc[:, s:e].copy()

        # drop empty columns
        nonempty = ~sub.isna().all(axis=0)
        sub = sub.loc[:, nonempty]
        if sub.shape[1] < 2:
            continue

        t_col = sub.iloc[:, 0].map(parse_time_any_to_hours)
        # if numeric-only, infer (often hours already here)
        if sub.iloc[:, 0].apply(lambda x: isinstance(x, (int, float, np.integer, np.floating))).mean() > 0.7:
            t_col = infer_and_convert_numeric_time_to_hours(sub.iloc[:, 0], None)

        tmp = pd.concat([pd.to_numeric(t_col, errors="coerce").rename("time_h"), sub.iloc[:, 1:]], axis=1)
        tmp.columns = ["time_h"] + [f"c{ci}" for ci in range(1, tmp.shape[1])]
        long = tmp.melt(id_vars=["time_h"], var_name="cond", value_name="OD")
        long["OD"] = pd.to_numeric(long["OD"], errors="coerce")
        long = long.dropna(subset=["time_h", "OD"]).copy()

        long["FileName"] = f"{Path(path).stem}__sheet{sheet_name}__b{bi}"
        long["orig_TestId"] = long["cond"].astype(str)
        long["Model Name"] = long["cond"].astype(str)
        long["Is_Valid"] = True
        outs.append(standardize_long(long, Path(path).stem))

    if not outs:
        return None
    return pd.concat(outs, ignore_index=True)


def convert_grofit_v_wide_to_long(df: pd.DataFrame, file_stem: str) -> pd.DataFrame:
    """
    grofit-like wide:
      - V3 = time 0.00 h
      - V4 = 0.25 h
      - V5 = 0.50 h ...
    plus an ID column (often 'TestId' or similar) and optional Is_Valid / Model Name
    """

    # find V-columns
    vcols = [c for c in df.columns if isinstance(c, str) and re.match(r"^V\d+$", c.strip())]
    if not vcols:
        raise ValueError("No V-columns found for grofit format.")

    # sort by numeric suffix
    vnums = sorted([(int(c[1:]), c) for c in vcols], key=lambda x: x[0])
    vcols_sorted = [c for _, c in vnums]

    # Map V3->0.00, V4->0.25, V5->0.50 ...
    # (If your file starts at V4 instead, change base_v to 4)
    base_v = 4
    def v_to_hours(vname: str) -> float:
        k = int(vname[1:])
        return (k - base_v) * 0.25

    # Ensure metadata columns exist (best-effort)
    out = df.copy()
    if "FileName" not in out.columns:
        out["FileName"] = file_stem

    # Try common id names (with special handling for Test_Id + V3)
    if "Test_Id" in out.columns and "V3" in out.columns:
        out["orig_TestId"] = (
            out["Test_Id"].astype(str).str.strip()
            + "_C"
            + out["V3"].astype(str).str.strip()
        )

    elif "Test Id" in out.columns:
        out["orig_TestId"] = out["Test Id"].astype(str)

    elif "TestId" in out.columns:
        out["orig_TestId"] = out["TestId"].astype(str)

    elif "ID" in out.columns:
        out["orig_TestId"] = out["ID"].astype(str)

    else:
        # fallback
        out["orig_TestId"] = out.index.astype(str)


    if "Model Name" not in out.columns:
        out["Model Name"] = ""

    if "Is_Valid" not in out.columns:
        # you said grofit curves are valid by default
        out["Is_Valid"] = True

    # wide -> long
    long = out.melt(
        id_vars=["FileName", "orig_TestId", "Model Name", "Is_Valid"],
        value_vars=vcols_sorted,
        var_name="vcol",
        value_name="OD"
    )
    long["time_h"] = long["vcol"].map(v_to_hours)
    long = long.drop(columns=["vcol"])

    return standardize_long(long, file_stem)

def parse_excel_any(path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    collected = []
    for sh in xls.sheet_names:
        # Try simple wide (Test Id + T#.0 (h)) first
        try:
            df_sheet = pd.read_excel(path, sheet_name=sh)
            out = convert_simple_wide_to_long(df_sheet, Path(path).stem)
            if out is not None and not out.empty:
                collected.append(out)
                continue
        except Exception:
            pass
        # 1) plate export
        try:
            out = parse_plate_xlsx(path, sheet_name=sh)
            if out is not None and not out.empty:
                collected.append(out)
                continue
        except Exception:
            pass
        # 2) simple time table
        try:
            out = parse_time_table_any(path, sheet_name=sh)
            if out is not None and not out.empty:
                collected.append(out)
                continue
        except Exception:
            pass
        # 3) block style
        try:
            out = parse_excel_block_style(path, sheet_name=sh)
            if out is not None and not out.empty:
                collected.append(out)
                continue
        except Exception:
            pass

    if not collected:
        raise ValueError(f"Could not parse Excel file: {path}")
    return pd.concat(collected, ignore_index=True)

def detect_wide_time_columns(df: pd.DataFrame) -> Tuple[List[str], Optional[str]]:
    """
    Detect wide time columns:
      - synthetic style: 'T0.25 (h)' etc
      - grofit-ish: V3/V4... and a separate mapping is needed (not done here unless time is encoded)
    Returns (time_cols, mode)
    """
    cols = list(df.columns)
    tcols = [c for c in cols if isinstance(c, str) and re.match(r"^T\d+(\.\d+)?\s*\(h\)$", c.strip())]
    if tcols:
        return tcols, "synthetic"
    return [], None
