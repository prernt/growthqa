from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

@dataclass(frozen=True)
class GrofitAdapterConfig:
    # required output schema for grofit pipeline
    out_test_id: str = "test_id"          # experiment/group
    out_curve_id: str = "curve_id"        # curve replicate / well
    out_conc: str = "concentration"       # dose (float)
    out_time: str = "time"               # time (float)
    out_y: str = "y"                     # observation (float)
    out_is_valid: str = "is_valid"       # boolean

    # common incoming column aliases
    time_col_aliases: Tuple[str, ...] = ("Time (h)", "time", "Time", "t", "T")
    test_id_aliases: Tuple[str, ...] = ("test_id", "Test Id", "TestID", "testid")
    curve_id_aliases: Tuple[str, ...] = ("curve_id", "Curve Id", "CurveID", "well", "Well")
    conc_aliases: Tuple[str, ...] = ("concentration", "Concentration", "conc", "Conc", "dose", "Dose")

    # prediction/meta output aliases
    pred_test_id_aliases: Tuple[str, ...] = ("Test Id", "test_id", "TestID")
    pred_label_aliases: Tuple[str, ...] = ("Predicted Label", "pred_label", "label", "Label", "is_valid", "valid")
    pred_pvalid_aliases: Tuple[str, ...] = (
        "Confidence (Valid)", "confidence_valid", "p_valid", "pvalid", "prob_valid"
    )


_TIME_PATTERNS = (
    r"^T(?P<t>\d+(\.\d+)?)\s*\(h\)\s*$",       # T0.50 (h)
    r"^T(?P<t>\d+(\.\d+)?)\s*\(hours?\)\s*$",
    r"^(?P<t>\d+(\.\d+)?)\s*h$",
)
_TIME_REGEXES = [re.compile(p, re.IGNORECASE) for p in _TIME_PATTERNS]


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------

def _first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower_map = {str(c).lower(): c for c in cols}
    for c in candidates:
        lc = str(c).lower()
        if lc in lower_map:
            return lower_map[lc]
    return None


def _infer_time_cols(df: pd.DataFrame) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for col in df.columns:
        s = str(col).strip()
        for rx in _TIME_REGEXES:
            m = rx.match(s)
            if m:
                out.append((col, float(m.group("t"))))
                break
    out.sort(key=lambda x: x[1])
    return out


def _coerce_is_valid_from_label(x) -> Optional[bool]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        if x == 1:
            return True
        if x == 0:
            return False

    s = str(x).strip().lower()
    if s in ("valid", "v", "true", "yes", "1"):
        return True
    if s in ("invalid", "i", "false", "no", "0"):
        return False
    if s in ("unsure", "unknown", "na", "nan"):
        return False
    return None


# ------------------------------------------------------------
# Key logic for your project outputs:
# predictions use Test Id like "testSample1_BY4741"
# ------------------------------------------------------------

def split_pred_test_id(pred_test_id: Union[str, int, float], delim: str = "_") -> Tuple[str, str]:
    """
    Split prediction 'Test Id' into (test_id, curve_id).

    For your outputs:
      "testSample1_BY4741" -> ("testSample1", "BY4741")

    Rule:
      split on last delimiter (safer if test_id contains underscores).
    """
    s = str(pred_test_id)
    if delim not in s:
        return s, s  # fallback: can't split -> treat both same
    left, right = s.rsplit(delim, 1)
    return left, right


def predictions_to_curve_map(
    predictions_df: pd.DataFrame,
    *,
    config: GrofitAdapterConfig = GrofitAdapterConfig(),
    delim: str = "_",
    prefer_label: bool = True,
    valid_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Convert predictions/meta_debug file into a mapping table:
      test_id, curve_id, is_valid

    Supports:
      - Predicted Label (Valid/Invalid/Unsure)
      - or Confidence (Valid) / p_valid with threshold
    """
    df = predictions_df.copy()

    pred_tid_col = _first_existing_col(df, config.pred_test_id_aliases)
    if pred_tid_col is None:
        raise ValueError("predictions_df: could not find prediction Test Id column.")

    label_col = _first_existing_col(df, config.pred_label_aliases)
    pvalid_col = _first_existing_col(df, config.pred_pvalid_aliases)

    # split into test_id + curve_id
    split = df[pred_tid_col].apply(lambda v: split_pred_test_id(v, delim=delim))
    df["_test_id"] = split.apply(lambda x: x[0])
    df["_curve_id"] = split.apply(lambda x: x[1])

    # compute is_valid
    is_valid = None
    if prefer_label and label_col is not None:
        is_valid = df[label_col].map(_coerce_is_valid_from_label)
    elif pvalid_col is not None:
        pv = pd.to_numeric(df[pvalid_col], errors="coerce")
        is_valid = pv >= float(valid_threshold)

    if is_valid is None:
        raise ValueError("predictions_df: need either Predicted Label or Confidence(Valid)/p_valid column.")

    out = pd.DataFrame(
        {
            config.out_test_id: df["_test_id"].astype(str),
            config.out_curve_id: df["_curve_id"].astype(str),
            config.out_is_valid: pd.Series(is_valid).fillna(False).astype(bool),
        }
    ).drop_duplicates(subset=[config.out_test_id, config.out_curve_id])

    return out


# ------------------------------------------------------------
# Adapters for the two input formats you use
# ------------------------------------------------------------

def input_time_well_to_tidy(
    df: pd.DataFrame,
    *,
    config: GrofitAdapterConfig = GrofitAdapterConfig(),
    file_test_id: Optional[str] = None,
    concentration_map: Optional[Dict[str, float]] = None,
    drop_na_y: bool = True,
) -> pd.DataFrame:
    """
    For testSample1.csv shape:
      Time (h) | BY4741 | trk1trk2 | ...

    Produces tidy:
      test_id, curve_id, concentration, time, y
    """
    src = df.copy()
    time_col = _first_existing_col(src, config.time_col_aliases)
    if time_col is None:
        raise ValueError("Could not infer time column (expected e.g. 'Time (h)').")

    curve_cols = [c for c in src.columns if c != time_col]
    if not curve_cols:
        raise ValueError("No curve columns found besides time.")

    tidy = src.melt(
        id_vars=[time_col],
        value_vars=curve_cols,
        var_name=config.out_curve_id,
        value_name=config.out_y,
    )

    tidy[config.out_time] = pd.to_numeric(tidy[time_col], errors="coerce")
    tidy[config.out_y] = pd.to_numeric(tidy[config.out_y], errors="coerce")
    tidy[config.out_curve_id] = tidy[config.out_curve_id].astype(str)

    # test_id: use passed file_test_id, else "test_1"
    tidy[config.out_test_id] = str(file_test_id) if file_test_id is not None else "test_1"

    # concentration: from map if available, else 0
    if concentration_map is None:
        tidy[config.out_conc] = 0.0
    else:
        tidy[config.out_conc] = tidy[config.out_curve_id].map(concentration_map).astype(float)

    if drop_na_y:
        tidy = tidy[tidy[config.out_y].notna() & tidy[config.out_time].notna()].copy()

    return tidy[[config.out_test_id, config.out_curve_id, config.out_conc, config.out_time, config.out_y]]


def input_wide_curve_rows_to_tidy(
    df: pd.DataFrame,
    *,
    config: GrofitAdapterConfig = GrofitAdapterConfig(),
    file_test_id: Optional[str] = None,
    test_id_col: Optional[str] = None,
    concentration_col: Optional[str] = None,
    drop_na_y: bool = True,
) -> pd.DataFrame:
    """
    For __testSample1.csv shape:
      Test Id | T0.00 (h) | T0.50 (h) | ... | T7.00 (h)

    One row = one curve_id (from Test Id column in that file).
    test_id (experiment) can be provided via file_test_id.
    """
    src = df.copy()

    test_id_col = test_id_col or _first_existing_col(src, config.test_id_aliases)
    if test_id_col is None:
        raise ValueError("Expected a 'Test Id' column in wide-per-curve input.")

    concentration_col = concentration_col or _first_existing_col(src, config.conc_aliases)
    if concentration_col is None:
        src[config.out_conc] = 0.0
        concentration_col = config.out_conc

    time_cols = _infer_time_cols(src)
    if not time_cols:
        raise ValueError("No time columns detected (expected columns like 'T0.50 (h)').")

    # melt into tidy
    id_vars = [test_id_col, concentration_col]
    tidy = src.melt(
        id_vars=id_vars,
        value_vars=[c for c, _t in time_cols],
        var_name="_time_label",
        value_name=config.out_y,
    )
    time_map = {c: t for c, t in time_cols}
    tidy[config.out_time] = tidy["_time_label"].map(time_map).astype(float)
    tidy[config.out_y] = pd.to_numeric(tidy[config.out_y], errors="coerce")

    # curve_id is the row's Test Id
    tidy[config.out_curve_id] = tidy[test_id_col].astype(str)

    # experiment/test group id
    tidy[config.out_test_id] = str(file_test_id) if file_test_id is not None else "test_1"

    tidy = tidy.rename(columns={concentration_col: config.out_conc})

    if drop_na_y:
        tidy = tidy[tidy[config.out_y].notna() & tidy[config.out_time].notna()].copy()

    return tidy[[config.out_test_id, config.out_curve_id, config.out_conc, config.out_time, config.out_y]]


# ------------------------------------------------------------
# Attach validity to tidy curves (using your prediction outputs)
# ------------------------------------------------------------

def attach_validity_from_predictions(
    tidy_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    *,
    config: GrofitAdapterConfig = GrofitAdapterConfig(),
    delim: str = "_",
    prefer_label: bool = True,
    valid_threshold: float = 0.5,
    default_if_missing: bool = False,
) -> pd.DataFrame:
    """
    Join is_valid onto tidy curves using your prediction outputs.

    - predictions_df has Test Id like "testSample1_BY4741"
    - tidy_df has test_id = "testSample1" (file_test_id) and curve_id = "BY4741"
    """
    out = tidy_df.copy()

    required = [config.out_test_id, config.out_curve_id, config.out_conc, config.out_time, config.out_y]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"tidy_df missing required columns: {missing}")

    pred_map = predictions_to_curve_map(
        predictions_df,
        config=config,
        delim=delim,
        prefer_label=prefer_label,
        valid_threshold=valid_threshold,
    )

    out = out.merge(pred_map, on=[config.out_test_id, config.out_curve_id], how="left")
    out[config.out_is_valid] = out[config.out_is_valid].fillna(bool(default_if_missing)).astype(bool)
    return out


# ------------------------------------------------------------
# One-call convenience for your exact file patterns
# ------------------------------------------------------------

def build_tidy_for_grofit(
    input_df: pd.DataFrame,
    *,
    predictions_df: Optional[pd.DataFrame] = None,
    config: GrofitAdapterConfig = GrofitAdapterConfig(),
    file_test_id: Optional[str] = None,
    delim: str = "_",
) -> pd.DataFrame:
    """
    Auto-detect and convert:
      - if has 'Time (h)' + multiple curve columns -> time-well adapter
      - else if has 'Test Id' + T*. (h) columns -> wide-per-curve adapter

    Then optionally attach validity from predictions.
    """
    cols = set(map(str, input_df.columns))

    time_col = _first_existing_col(input_df, config.time_col_aliases)
    has_time_well = time_col is not None and len(cols) >= 2 and "T0.00 (h)" not in cols

    has_wide_rows = _first_existing_col(input_df, config.test_id_aliases) is not None and len(_infer_time_cols(input_df)) > 0

    if has_time_well and not has_wide_rows:
        tidy = input_time_well_to_tidy(input_df, config=config, file_test_id=file_test_id)
    elif has_wide_rows:
        tidy = input_wide_curve_rows_to_tidy(input_df, config=config, file_test_id=file_test_id)
    else:
        raise ValueError("Could not detect input format. Provide a known format.")

    if predictions_df is not None:
        if file_test_id is None:
            raise ValueError("file_test_id is required to attach predictions (needed to match 'testSample1_*').")
        tidy = attach_validity_from_predictions(
            tidy,
            predictions_df,
            config=config,
            delim=delim,
        )
    else:
        tidy[config.out_is_valid] = True

    return tidy
