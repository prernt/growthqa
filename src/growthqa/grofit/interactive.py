from __future__ import annotations
import pandas as pd
from typing import Callable, Optional

UserFilterFn = Callable[[pd.DataFrame], pd.Series]

def apply_user_exclusion(
    fits_df: pd.DataFrame,
    user_filter_fn: Optional[UserFilterFn] = None,
) -> pd.DataFrame:
    if user_filter_fn is None:
        return fits_df
    exclude_mask = user_filter_fn(fits_df)
    if exclude_mask is None:
        return fits_df
    return fits_df.loc[~exclude_mask].copy()
