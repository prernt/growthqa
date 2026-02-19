from __future__ import annotations

import hashlib
import re
from typing import Iterable, List

import numpy as np
import pandas as pd

from growthqa.preprocess.timegrid import parse_time_from_header

_FULL_HORIZON_DEFAULT = 16.0


def _time_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if parse_time_from_header(str(c)) is not None]
    return sorted(cols, key=lambda c: float(parse_time_from_header(str(c)) or 0.0))


def _time_values(time_cols: Iterable[str]) -> np.ndarray:
    return np.array([parse_time_from_header(str(c)) for c in time_cols], dtype=float)


def _fmt_h(h: float) -> str:
    s = f"{float(h):.2f}"
    return s.rstrip("0").rstrip(".")


def _sanitize_token(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return ""
    s = re.sub(r"[^0-9A-Za-z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _index_to_letters(index: int) -> str:
    n = int(index)
    if n < 1:
        n = 1
    out = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out.append(chr(ord("A") + rem))
    return "".join(reversed(out))


def _conc_token(value: object) -> str:
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(v):
        s = f"{float(v):g}"
        return s.replace(".", "p").replace("-", "m")
    return _sanitize_token(value)


def _stable_seed(seed: int, base_curve_id: str, tmax_original: float) -> int:
    token = f"{int(seed)}|{base_curve_id}|{float(tmax_original):.6f}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) & 0xFFFFFFFF


def compute_tmax_original(row: pd.Series, time_cols: List[str]) -> float:
    if not time_cols:
        return np.nan
    tvals = _time_values(time_cols)
    yvals = pd.to_numeric(row[time_cols], errors="coerce").to_numpy(dtype=float)
    obs = np.isfinite(tvals) & np.isfinite(yvals)
    if not np.any(obs):
        return np.nan
    return float(np.nanmax(tvals[obs]))


def make_base_curve_id(row: pd.Series, fallback_index: int | None = None) -> str:
    test_id = _sanitize_token(row.get("Test Id"))
    if test_id:
        return test_id
    fname = _sanitize_token(row.get("FileName"))
    if fname:
        return fname
    if fallback_index is None:
        return "curve"
    return f"curve_{int(fallback_index) + 1}"


def make_aug_id(base_curve_id: str, h: float) -> str:
    hid = f"H{_fmt_h(float(h))}"
    return f"{base_curve_id}_{hid}"


def _align_horizon_to_grid(h: float, grid_times: np.ndarray) -> float:
    valid = grid_times[np.isfinite(grid_times)]
    if valid.size == 0:
        return np.nan
    candidates = valid[valid <= float(h) + 1e-9]
    if candidates.size == 0:
        return np.nan
    return float(np.max(candidates))


def sample_valid_horizons(
    tmax_original: float,
    candidate_horizons: List[float],
    *,
    per_curve: int,
    seed: int,
    base_curve_id: str,
    grid_times: np.ndarray,
) -> List[float]:
    if not np.isfinite(tmax_original):
        return []

    aligned: list[float] = []
    for h in candidate_horizons:
        h_aligned = _align_horizon_to_grid(float(min(float(h), float(tmax_original))), grid_times)
        if np.isfinite(h_aligned) and h_aligned <= float(tmax_original) + 1e-9:
            aligned.append(float(h_aligned))
    valid = sorted(set(aligned))
    if not valid:
        return []

    k = max(2, min(3, int(per_curve)))
    k = min(k, len(valid))

    rng = np.random.default_rng(_stable_seed(seed, base_curve_id, tmax_original))
    if k == len(valid):
        picked = valid
    else:
        idx = rng.choice(len(valid), size=k, replace=False)
        picked = [valid[int(i)] for i in sorted(idx)]
    return sorted(float(h) for h in picked)


def apply_truncation(row: pd.Series, time_cols: List[str], h: float) -> pd.Series:
    out = row.copy()
    for c in time_cols:
        t = parse_time_from_header(str(c))
        if t is not None and float(t) > float(h) + 1e-9:
            out[c] = np.nan
    return out


def augment_df(
    df_wide: pd.DataFrame,
    candidate_horizons: List[float],
    *,
    per_curve: int = 3,
    seed: int = 123,
    full_horizon: float = _FULL_HORIZON_DEFAULT,
) -> pd.DataFrame:
    time_cols = _time_cols(df_wide)
    grid_times = _time_values(time_cols)
    if not time_cols:
        out = df_wide.copy()
        out["tmax_original"] = np.nan
        return out

    used_base_ids: set[str] = set()
    missing_conc_counts: dict[str, int] = {}
    rows: list[pd.Series] = []
    candidates = sorted(float(h) for h in candidate_horizons)

    for idx, row in df_wide.iterrows():
        test_id = _sanitize_token(row.get("Test Id"))
        if not test_id:
            continue

        tmax_original = compute_tmax_original(row, time_cols)
        if not np.isfinite(tmax_original):
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

        sampled = sample_valid_horizons(
            tmax_original=float(tmax_original),
            candidate_horizons=candidates,
            per_curve=per_curve,
            seed=seed,
            base_curve_id=base_curve_id,
            grid_times=grid_times,
        )
        if len(sampled) < 2:
            continue

        for h in sampled:
            r = apply_truncation(row, time_cols, h)
            r["base_curve_id"] = base_curve_id
            r["train_horizon"] = float(h)
            r["tmax_original"] = float(tmax_original)
            r["is_censored"] = int(float(h) < float(full_horizon))
            r["aug_id"] = make_aug_id(base_curve_id, float(h))
            rows.append(r)

    if not rows:
        out = df_wide.iloc[0:0].copy()
        out["tmax_original"] = pd.Series(dtype=float)
        return out
    return pd.DataFrame(rows).reset_index(drop=True)
