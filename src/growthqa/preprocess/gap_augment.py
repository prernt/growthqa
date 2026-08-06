from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from growthqa.preprocess.timegrid import get_sorted_time_columns
from growthqa.preprocess.truncation_augment import (
    _time_values,
    _sanitize_token,
    _index_to_letters,
    _conc_token,
    _stable_seed,
    compute_tmax_original,
    make_base_curve_id,
)

def augment_raw_wide_with_gaps(
    df_wide_raw: pd.DataFrame,
    *,
    frac_curves_to_augment: float = 0.30,
    per_curve: int = 2,
    min_gap_hours: float = 2.0,
    max_gap_hours: float = 6.0,
    min_missing_frac: float = 0.40,
    max_missing_frac: float = 0.80,
    seed: int = 456,
    min_real_points_to_augment: int = 4,
) -> pd.DataFrame:
    time_cols = get_sorted_time_columns(df_wide_raw)
    if not time_cols:
        out = df_wide_raw.iloc[0:0].copy()
        out["tmax_original"] = pd.Series(dtype=float)
        return out

    rng_master = np.random.default_rng(seed)
    n_curves = len(df_wide_raw)
    n_to_augment = int(round(float(frac_curves_to_augment) * n_curves))
    if n_to_augment <= 0 or n_curves == 0:
        out = df_wide_raw.iloc[0:0].copy()
        out["tmax_original"] = pd.Series(dtype=float)
        return out

    all_idx = df_wide_raw.index.to_numpy()
    chosen_idx = rng_master.choice(all_idx, size=min(n_to_augment, n_curves), replace=False)

    used_base_ids: set[str] = set()
    missing_conc_counts: dict[str, int] = {}
    rows: List[pd.Series] = []

    t_all = _time_values(time_cols)

    for idx in chosen_idx:
        row = df_wide_raw.loc[idx]

        tmax_original = compute_tmax_original(row, time_cols)
        if not np.isfinite(tmax_original) or tmax_original <= 0:
            continue

        y_all = pd.to_numeric(row[time_cols], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(t_all) & np.isfinite(y_all)
        if int(finite.sum()) < int(min_real_points_to_augment):
            continue

        base_test = _sanitize_token(row.get("Test Id")) or make_base_curve_id(row, fallback_index=int(idx))
        conc = _conc_token(row.get("Concentration", np.nan))
        if conc:
            suffix = conc
        else:
            c = missing_conc_counts.get(base_test, 0) + 1
            missing_conc_counts[base_test] = c
            suffix = _index_to_letters(c)
        base_curve_id = f"{base_test}_{suffix}"
        if base_curve_id in used_base_ids:
            k = 2
            while f"{base_curve_id}_{k}" in used_base_ids:
                k += 1
            base_curve_id = f"{base_curve_id}_{k}"
        used_base_ids.add(base_curve_id)

        row_seed = _stable_seed(seed, base_curve_id, tmax_original)
        rng = np.random.default_rng(row_seed)

        for rep in range(int(per_curve)):
            pattern = "block" if rng.random() < 0.5 else "scattered"
            out = row.copy()

            if pattern == "block":
                real_times = np.sort(t_all[finite])
                gap_hours = float(rng.uniform(min_gap_hours, max_gap_hours))
                start_lo = float(real_times[0])
                start_hi = float(max(real_times[0], real_times[-1] - gap_hours))
                start = float(rng.uniform(start_lo, start_hi)) if start_hi > start_lo else start_lo
                end = start + gap_hours
                for c, tv in zip(time_cols, t_all):
                    if np.isfinite(tv) and start <= tv <= end:
                        out[c] = np.nan
            else:
                target_frac = float(rng.uniform(min_missing_frac, max_missing_frac))
                finite_cols = [c for c, f in zip(time_cols, finite) if f]
                n_to_drop = int(round(target_frac * len(finite_cols)))
                max_droppable = max(len(finite_cols) - int(min_real_points_to_augment), 0)
                n_to_drop = min(n_to_drop, max_droppable)
                if n_to_drop > 0:
                    drop_cols = rng.choice(np.array(finite_cols, dtype=object), size=n_to_drop, replace=False)
                    for c in drop_cols:
                        out[c] = np.nan

            out["base_curve_id"] = base_curve_id
            out["train_horizon"] = float(tmax_original)
            out["tmax_original"] = float(tmax_original)
            out["is_censored"] = 0
            out["aug_id"] = f"{base_curve_id}_GAP{rep + 1}{pattern[0].upper()}"
            out["gap_augmented"] = 1
            out["gap_pattern"] = pattern
            rows.append(out)

    if not rows:
        out = df_wide_raw.iloc[0:0].copy()
        out["tmax_original"] = pd.Series(dtype=float)
        return out
    return pd.DataFrame(rows).reset_index(drop=True)
