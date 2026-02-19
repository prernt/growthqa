# src/growthqa/grofit/interactive.py
from __future__ import annotations
import pandas as pd
from typing import Callable, Optional

# user_filter_fn gets a dataframe of curve fits and returns a list/series of curve_ids to exclude
UserFilterFn = Callable[[pd.DataFrame], pd.Series]

def apply_user_exclusion(
    fits_df: pd.DataFrame,
    user_filter_fn: Optional[UserFilterFn] = None,
) -> pd.DataFrame:
    """
    Optional interactive exclusion step described in paper.
    In automatic mode: no exclusion.
    """
    if user_filter_fn is None:
        return fits_df
    exclude_mask = user_filter_fn(fits_df)
    if exclude_mask is None:
        return fits_df
    # exclude_mask can be boolean series aligned with fits_df
    return fits_df.loc[~exclude_mask].copy()
