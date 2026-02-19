from __future__ import annotations

from typing import List

import pandas as pd

from growthqa.preprocess.truncation_augment import (
    augment_df,
    make_base_curve_id as _make_base_curve_id_row,
)


def build_base_curve_id(df: pd.DataFrame) -> pd.Series:
    return df.apply(_make_base_curve_id_row, axis=1)


def ensure_aug_columns(
    df: pd.DataFrame,
    *,
    default_horizon: float = 16.0,
) -> pd.DataFrame:
    out = df.copy()
    if "base_curve_id" not in out.columns:
        out["base_curve_id"] = build_base_curve_id(out)
    if "train_horizon" not in out.columns:
        out["train_horizon"] = float(default_horizon)
    if "is_censored" not in out.columns:
        out["is_censored"] = (pd.to_numeric(out["train_horizon"], errors="coerce") < float(default_horizon)).astype(int)
    if "aug_id" not in out.columns:
        out["aug_id"] = out["base_curve_id"].astype(str)
    return out


def make_truncated_variants(
    wide: pd.DataFrame,
    horizons: List[float],
    per_curve: int = 3,
    seed: int = 123,
    id_col: str = "Test Id",
    full_horizon: float = 16.0,
) -> pd.DataFrame:
    del id_col  # kept for backward compatibility
    out = augment_df(
        wide,
        candidate_horizons=list(horizons),
        per_curve=int(per_curve),
        seed=int(seed),
        full_horizon=float(full_horizon),
    )
    return ensure_aug_columns(out, default_horizon=float(full_horizon))
