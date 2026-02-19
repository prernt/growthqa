from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from growthqa.preprocess.timegrid import parse_time_from_header


S2_REASON_LATE_GROWTH = "S2_LATE_GROWTH_DETECTED"
S2_REASON_CORROBORATION = "S2_PLATEAU_DECLINE_CORROBORATION"
S2_REASON_ARTIFACT = "S2_POSSIBLE_ARTIFACT"
S2_REASON_DRIFT_NOISE = "S2_DRIFT_OR_NOISE"
S2_REASON_NO_SUPPORT = "S2_NO_LATE_SUPPORT"
S2_REASON_NO_CHANGE = "S2_NO_CHANGE"
S2_REASON_TOO_SPARSE_LATE = "S2_TOO_SPARSE_LATE"
S2_REASON_ARTIFACT_EVAPORATION = "S2_REASON_ARTIFACT_EVAPORATION"


@dataclass
class Stage2Config:
    # Late-window definition
    stage2_start: float = 16.0
    late_min_points: int = 5

    # Late growth detection
    abs_threshold: float = 0.05
    rel_threshold: float = 0.08
    slope_threshold: float = 0.01
    diff_threshold: float = 0.005

    # Plateau detection
    plateau_slope_eps: float = 0.005
    plateau_band: float = 0.03
    growth_min_amplitude: float = 0.1

    # Decline detection
    decline_slope_threshold: float = 0.01
    decline_drop_threshold: float = 0.08
    min_decline_span: float = 1.0

    # Drift/noise detection
    drift_amplitude_max: float = 0.05
    drift_monotonicity_min: float = 0.85
    drift_roughness_max: float = 0.02
    noise_roughness_min: float = 0.05
    noise_signchange_min: int = 6
    spike_threshold: float = 0.1
    evaporation_r2_threshold: float = 0.95
    evaporation_quadratic_term_threshold: float = 1e-3

    # Final label safeguard
    conf_downgrade_threshold: float = 0.65

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _normalize_label(label: object) -> str:
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return ""
    s = str(label).strip().lower()
    if s in {"valid", "true", "1"}:
        return "Valid"
    if s in {"invalid", "false", "0"}:
        return "Invalid"
    if s in {"unsure", "unknown"}:
        return "Unsure"
    return str(label).strip()


def _winsorize(y: np.ndarray, lo_q: float = 0.05, hi_q: float = 0.95) -> np.ndarray:
    if y.size == 0:
        return y
    lo = float(np.nanquantile(y, lo_q))
    hi = float(np.nanquantile(y, hi_q))
    return np.clip(y, lo, hi)


def _robust_slope(t: np.ndarray, y: np.ndarray) -> float:
    if t.size < 2:
        return float(np.nan)
    dt = np.diff(t)
    dy = np.diff(y)
    mask = np.isfinite(dt) & np.isfinite(dy) & (np.abs(dt) > 1e-12)
    if not np.any(mask):
        return float(np.nan)
    return float(np.nanmedian(dy[mask] / dt[mask]))


def _roughness(y: np.ndarray) -> float:
    if y.size < 3:
        return float(np.nan)
    d1 = np.diff(y)
    return float(np.nanmedian(np.abs(np.diff(d1))))


def _monotonicity_fraction(diffs: np.ndarray, eps: float) -> float:
    if diffs.size == 0:
        return float(np.nan)
    sig = diffs[np.abs(diffs) > eps]
    if sig.size == 0:
        return 1.0
    pos = float(np.mean(sig > 0))
    neg = float(np.mean(sig < 0))
    return max(pos, neg)


def _sign_changes(diffs: np.ndarray, eps: float) -> int:
    if diffs.size == 0:
        return 0
    sig = diffs[np.abs(diffs) > eps]
    if sig.size < 2:
        return 0
    s = np.sign(sig)
    return int(np.sum(s[1:] * s[:-1] < 0))


def _sustained_positive(diffs: np.ndarray, thr: float) -> bool:
    if diffs.size == 0:
        return False
    k = int(max(2, min(3, diffs.size)))
    run = 0
    for d in diffs:
        if d > thr:
            run += 1
            if run >= k:
                return True
        else:
            run = 0
    frac_pos = float(np.mean(diffs > thr))
    return frac_pos >= 0.7


def _extract_curve_arrays(wide_row: pd.Series, time_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    t = np.array([parse_time_from_header(str(c)) for c in time_cols], dtype=float)
    y = pd.to_numeric(wide_row[time_cols], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if t.size:
        order = np.argsort(t)
        t = t[order]
        y = y[order]
    return t, y


def compute_has_late_data_from_raw(
    wide_row: pd.Series,
    time_cols: list[str],
    stage2_start: float = 16.0,
) -> tuple[bool, float]:
    """
    Determine late-horizon availability strictly from raw wide time headers.
    This intentionally ignores any early-pass truncation/resampling so Stage-2
    cannot be accidentally disabled when the experiment extends past 16h.
    """
    times = np.array([parse_time_from_header(str(c)) for c in time_cols], dtype=float)
    times = times[np.isfinite(times)]
    if times.size == 0:
        return False, float(np.nan)
    raw_observed_tmax = float(np.nanmax(times))
    return bool(raw_observed_tmax > float(stage2_start)), raw_observed_tmax


def get_time_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if parse_time_from_header(str(c)) is not None]


def compute_raw_tmax_and_has_late(
    row: pd.Series,
    time_cols: list[str],
    stage2_start: float = 16.0,
) -> tuple[float, bool]:
    has_late_data, raw_tmax = compute_has_late_data_from_raw(row, time_cols, stage2_start=stage2_start)
    return raw_tmax, has_late_data


def compute_late_features(wide_row: pd.Series, time_cols: list[str], cfg: Stage2Config) -> dict[str, object]:
    out: dict[str, object] = {
        "has_late_data": False,
        "late_window_start": float(cfg.stage2_start),
        "late_tmax": np.nan,
        "late_n_points": 0,
        "late_slope": np.nan,
        "late_delta": np.nan,
        "late_max_increase": np.nan,
        "late_growth_detected": False,
        "plateau_detected": False,
        "decline_detected": False,
        "drift_detected": False,
        "noise_detected": False,
        "sigma_noise": np.nan,
        "late_linearity_r2": np.nan,
        "raw_observed_tmax": np.nan,
        "late_too_sparse": False,
        # Internal context used by decision policy.
        "_artifact_strong": False,
        "_early_weak": False,
        "_evaporation_artifact": False,
    }
    if not time_cols:
        return out

    has_late_data_raw, raw_observed_tmax = compute_has_late_data_from_raw(
        wide_row, time_cols, stage2_start=cfg.stage2_start
    )
    times_raw = np.array([parse_time_from_header(str(c)) for c in time_cols], dtype=float)
    vals_raw = pd.to_numeric(wide_row[time_cols], errors="coerce").to_numpy(dtype=float)
    n_late_raw = int(np.sum(np.isfinite(times_raw) & (times_raw > float(cfg.stage2_start)) & np.isfinite(vals_raw)))
    out["raw_observed_tmax"] = raw_observed_tmax
    if has_late_data_raw:
        out["has_late_data"] = True
        out["late_tmax"] = raw_observed_tmax
        out["late_n_points"] = n_late_raw
        out["late_window_start"] = float(cfg.stage2_start)
    else:
        out["late_n_points"] = 0

    t, y = _extract_curve_arrays(wide_row, time_cols)
    if t.size < 2:
        return out

    early = t <= float(cfg.stage2_start)
    late = t > float(cfg.stage2_start)

    if np.any(early):
        y_early = y[early]
        t_early = t[early]
        early_max = float(np.nanmax(y_early))
        early_min = float(np.nanmin(y_early))
        early_last = float(y_early[-1])
    else:
        y_early = np.array([], dtype=float)
        t_early = np.array([], dtype=float)
        early_max = float(np.nanmax(y))
        early_min = float(np.nanmin(y))
        early_last = float(y[0])

    overall_peak = float(np.nanmax(y))
    peak_idx = int(np.nanargmax(y))
    peak_time = float(t[peak_idx])
    early_amp = float(early_max - early_min) if np.isfinite(early_max) and np.isfinite(early_min) else np.nan
    out["_early_weak"] = bool(np.isfinite(early_amp) and early_amp < cfg.growth_min_amplitude)

    if not has_late_data_raw:
        return out

    t_late = t[late]
    y_late = y[late]

    if n_late_raw < int(cfg.late_min_points) or t_late.size < 2:
        # Stage-2 unreliable: keep feature values as NaN/False.
        out["late_too_sparse"] = True
        return out

    sigma_noise = 0.0
    if y_early.size >= 3:
        n_early = int(y_early.size)
        w = min(5, n_early if (n_early % 2 == 1) else (n_early - 1))
        if w >= 3:
            try:
                poly = min(2, w - 1)
                y_early_s = savgol_filter(y_early, window_length=w, polyorder=poly, mode="interp")
                sigma_noise = float(np.nanstd(y_early - y_early_s))
            except Exception:
                sigma_noise = 0.0
    out["sigma_noise"] = float(sigma_noise)

    dt_early = np.diff(t_early) if t_early.size >= 2 else np.array([], dtype=float)
    dt_early = dt_early[np.isfinite(dt_early) & (dt_early > 1e-12)]
    delta_time = float(np.nanmedian(dt_early)) if dt_early.size else 1.0
    dynamic_abs_threshold = max(float(cfg.abs_threshold), 3.0 * float(sigma_noise))
    dynamic_slope_threshold = max(float(cfg.slope_threshold), 3.0 * float(sigma_noise) / max(delta_time, 1e-12))

    y_lw = _winsorize(y_late)
    out["late_slope"] = _robust_slope(t_late, y_lw)
    out["late_delta"] = float(y_lw[-1] - y_lw[0]) if y_lw.size else np.nan
    out["late_max_increase"] = float(np.nanmax(y_lw) - np.nanmin(y_lw)) if y_lw.size else np.nan

    diffs = np.diff(y_lw)
    growth_min_delta = max(dynamic_abs_threshold, float(cfg.rel_threshold) * max(1e-6, early_max))
    sustained = _sustained_positive(diffs, float(cfg.diff_threshold))
    out["late_growth_detected"] = bool(
        np.isfinite(out["late_delta"])
        and np.isfinite(out["late_slope"])
        and (float(out["late_delta"]) >= growth_min_delta)
        and (float(out["late_slope"]) >= dynamic_slope_threshold)
        and sustained
    )
    if out["late_growth_detected"]:
        r2_linear = np.nan
        quad_term = np.nan
        try:
            lin_coef = np.polyfit(t_late, y_lw, 1)
            y_lin = np.polyval(lin_coef, t_late)
            ss_res_lin = float(np.sum((y_lw - y_lin) ** 2))
            ss_tot = float(np.sum((y_lw - np.mean(y_lw)) ** 2))
            r2_linear = 1.0 - (ss_res_lin / max(ss_tot, 1e-12))
        except Exception:
            r2_linear = np.nan
        try:
            quad_coef = np.polyfit(t_late, y_lw, 2)
            quad_term = float(quad_coef[0])
        except Exception:
            quad_term = np.nan
        out["late_linearity_r2"] = float(r2_linear) if np.isfinite(r2_linear) else np.nan
        out["_evaporation_artifact"] = bool(
            np.isfinite(r2_linear)
            and (float(r2_linear) > float(cfg.evaporation_r2_threshold))
            and np.isfinite(quad_term)
            and (abs(float(quad_term)) < float(cfg.evaporation_quadratic_term_threshold))
        )

    # Plateau corroboration: near-flat late slope and last values close to late max,
    # only meaningful when prior rise amplitude exists.
    late_max = float(np.nanmax(y_lw))
    tail_n = int(max(3, min(5, y_lw.size)))
    tail_median = float(np.nanmedian(y_lw[-tail_n:]))
    growth_before = bool((overall_peak - early_min) >= float(cfg.growth_min_amplitude))
    out["plateau_detected"] = bool(
        growth_before
        and np.isfinite(out["late_slope"])
        and (abs(float(out["late_slope"])) <= float(cfg.plateau_slope_eps))
        and ((late_max - tail_median) <= float(cfg.plateau_band))
    )

    # Decline corroboration: peak not at very end + sustained negative late slope + peak-to-last drop.
    last_span = float(np.nanmax(t_late) - peak_time)
    last_w = int(max(3, min(6, y_lw.size)))
    slope_last = _robust_slope(t_late[-last_w:], y_lw[-last_w:])
    drop_peak_to_last = float(overall_peak - y_lw[-1])
    out["decline_detected"] = bool(
        (last_span >= float(cfg.min_decline_span))
        and np.isfinite(slope_last)
        and (float(slope_last) <= -float(cfg.decline_slope_threshold))
        and (drop_peak_to_last >= float(cfg.decline_drop_threshold))
    )

    rough = _roughness(y_lw)
    mono_frac = _monotonicity_fraction(diffs, float(cfg.diff_threshold))
    sign_changes = _sign_changes(diffs, float(cfg.diff_threshold))
    max_abs_diff = float(np.nanmax(np.abs(diffs))) if diffs.size else 0.0
    directionality = float(max(np.mean(diffs > 0), np.mean(diffs < 0))) if diffs.size else 1.0

    out["drift_detected"] = bool(
        np.isfinite(out["late_max_increase"])
        and (float(out["late_max_increase"]) <= float(cfg.drift_amplitude_max))
        and np.isfinite(mono_frac)
        and (mono_frac >= float(cfg.drift_monotonicity_min))
        and np.isfinite(rough)
        and (rough <= float(cfg.drift_roughness_max))
        and not out["late_growth_detected"]
    )

    out["noise_detected"] = bool(
        (np.isfinite(rough) and rough >= float(cfg.noise_roughness_min))
        or (sign_changes >= int(cfg.noise_signchange_min))
        or (max_abs_diff >= float(cfg.spike_threshold) and directionality < 0.7)
    )

    out["_artifact_strong"] = bool(
        (out["drift_detected"] and np.isfinite(out["late_max_increase"]) and float(out["late_max_increase"]) <= float(cfg.drift_amplitude_max) * 0.75)
        or (out["noise_detected"] and sign_changes >= int(cfg.noise_signchange_min) + 2)
    )
    return out


def compute_stage2_decision(
    stage1_label: object,
    stage1_conf: float | int | None,
    late_features: Mapping[str, object],
    cfg: Stage2Config,
) -> tuple[str, str]:
    s1 = _normalize_label(stage1_label)
    s1_conf_num = pd.to_numeric(pd.Series([stage1_conf]), errors="coerce").iloc[0]
    s1_conf_val = float(s1_conf_num) if np.isfinite(s1_conf_num) else np.nan

    has_late = bool(late_features.get("has_late_data", False))
    n_late = pd.to_numeric(pd.Series([late_features.get("late_n_points", np.nan)]), errors="coerce").iloc[0]
    late_too_sparse = bool(late_features.get("late_too_sparse", False))
    if (not has_late) or (not np.isfinite(n_late)) or int(n_late) < int(cfg.late_min_points):
        if has_late:
            return "Unsure", S2_REASON_TOO_SPARSE_LATE
        return "", ""
    if late_too_sparse:
        return "Unsure", S2_REASON_TOO_SPARSE_LATE

    late_growth = bool(late_features.get("late_growth_detected", False))
    evaporation_artifact = bool(late_features.get("_evaporation_artifact", False))
    plateau = bool(late_features.get("plateau_detected", False))
    decline = bool(late_features.get("decline_detected", False))
    drift = bool(late_features.get("drift_detected", False))
    noise = bool(late_features.get("noise_detected", False))
    early_weak = bool(late_features.get("_early_weak", False))
    artifact_strong = bool(late_features.get("_artifact_strong", False))

    if late_growth and evaporation_artifact:
        return "Invalid", S2_REASON_ARTIFACT_EVAPORATION

    if s1 == "Invalid":
        if late_growth:
            return "Unsure", S2_REASON_LATE_GROWTH
        if drift or noise:
            if artifact_strong and early_weak:
                return "Invalid", S2_REASON_DRIFT_NOISE
            return "Unsure", S2_REASON_ARTIFACT
        return "Invalid", S2_REASON_NO_SUPPORT

    if s1 == "Valid":
        if plateau or decline:
            return "Valid", S2_REASON_CORROBORATION
        if drift or noise:
            return "Unsure", S2_REASON_ARTIFACT
        return "Valid", S2_REASON_NO_CHANGE

    # Stage-1 unsure/unknown fallback.
    if late_growth and (plateau or decline):
        return "Valid", S2_REASON_CORROBORATION
    if drift or noise:
        return "Unsure", S2_REASON_ARTIFACT
    return "Unsure", S2_REASON_NO_CHANGE


def compute_stage2_label(
    stage1_label: object,
    stage1_conf: float | int | None,
    late_features: Mapping[str, object],
    cfg: Stage2Config,
) -> tuple[str, str]:
    return compute_stage2_decision(stage1_label, stage1_conf, late_features, cfg)


def compose_final_label(
    stage1_label: object,
    stage1_conf: float | int | None,
    stage2_label: object,
    stage2_reason: object,
    cfg: Stage2Config,
) -> str:
    s1 = _normalize_label(stage1_label)
    s2 = _normalize_label(stage2_label)
    reason = str(stage2_reason).strip() if stage2_reason is not None else ""
    s1_conf_num = pd.to_numeric(pd.Series([stage1_conf]), errors="coerce").iloc[0]
    s1_conf_val = float(s1_conf_num) if np.isfinite(s1_conf_num) else np.nan

    if s2 == "":
        return s1 if s1 else "Unsure"
    if s2 == "Valid":
        return "Valid"
    if s2 == "Unsure":
        if s1 == "Valid":
            if np.isfinite(s1_conf_val) and s1_conf_val >= float(cfg.conf_downgrade_threshold):
                return "Valid"
        return "Unsure"
    if s2 == "Invalid":
        if s1 == "Invalid":
            return "Invalid"
        if s1 == "Valid":
            strong_artifact = reason == S2_REASON_DRIFT_NOISE
            if strong_artifact and np.isfinite(s1_conf_val) and s1_conf_val < float(cfg.conf_downgrade_threshold):
                return "Invalid"
            if np.isfinite(s1_conf_val) and s1_conf_val < float(cfg.conf_downgrade_threshold):
                return "Unsure"
            return "Valid"
        return "Invalid"
    return s1 if s1 else "Unsure"
