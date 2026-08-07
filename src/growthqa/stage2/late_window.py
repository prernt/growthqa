#Stage 2 evidence-based checker for growth curves, focusing on late-window data to assess growth, artifacts and decline.
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import theilslopes
from growthqa.preprocess.timegrid import parse_time_from_header

@dataclass
class Stage2ConfigEvidence:
    """
    Evidence-based Stage-2 config.
    Philosophy:
      - Stage-2 is a CHECKER, not a re-classifier.
      - It uses late-window raw data only and produces evidence scores.
    """
    stage2_start: float = 16.0
    min_late_points_floor: int = 3
    min_late_points_ceiling: int = 10
    min_late_hours_anchor: float = 2.5
    min_late_points_fallback_rate_per_hour: float = 2.0
    quality_threshold: float = 0.30
    late_window_reference_step_hours: float = 1.0
    late_window_max_missing_frac: float = 0.85
    # Evidence thresholds
    growth_z_threshold: float = 2.0          
    artifact_score_threshold: float = 0.70  
    decline_score_threshold: float = 0.70    
    artifact_cv_low: float = 0.05           
    artifact_cv_high: float = 0.20          
    artifact_osc_noise_mult: float = 2.0     
    artifact_evap_slope: float = -0.005      
    min_noise_level: float = 0.005          
    eps_dt: float = 1e-9
    

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

@dataclass
class EvidenceScores:
    growth_z_like: float        
    artifact_score: float       
    data_quality: float         
    confidence: float          
    decline_score: float = 0.0 
    late_slope: float = np.nan
    late_delta: float = np.nan
    noise_level: float = np.nan
    n_late_points: int = 0
    late_span_hours: float = np.nan
    late_coverage_ok: bool = False  
    min_late_points_required: int = 0  


def _mad_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return float(1.4826 * mad)


def _bounded(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def compute_noise_baseline_from_diffs(
    y_early: np.ndarray,
    y_late: np.ndarray,
    cfg: Stage2ConfigEvidence,
) -> float:
    y_early = np.asarray(y_early, dtype=float)
    y_late = np.asarray(y_late, dtype=float)
    diffs = None
    if np.isfinite(y_early).sum() >= 6:
        ye = y_early[np.isfinite(y_early)]
        diffs = np.diff(ye)
    elif np.isfinite(y_late).sum() >= 6:
        yl = y_late[np.isfinite(y_late)]
        diffs = np.diff(yl)
    sigma = _mad_std(diffs) if diffs is not None else 0.0
    sigma = max(float(cfg.min_noise_level), float(sigma))
    return float(sigma)


def compute_growth_evidence_z_like(
    t_late: np.ndarray,
    y_late: np.ndarray,
    noise_level_od: float,
    cfg: Stage2ConfigEvidence,
) -> tuple[float, float, float]:
    t = np.asarray(t_late, dtype=float)
    y = np.asarray(y_late, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]

    if t.size < 2:
        return 0.0, np.nan, np.nan

    idx = np.argsort(t)
    t, y = t[idx], y[idx]
    try:
        slope, intercept, _, _ = theilslopes(y, t)
        slope = float(slope)
    except Exception:
        # fallback: simple slope
        denom = float(t[-1] - t[0])
        slope = float((y[-1] - y[0]) / max(denom, cfg.eps_dt))
    delta = float(y[-1] - y[0])
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > cfg.eps_dt)]
    dt_med = float(np.nanmedian(dt)) if dt.size > 0 else 1.0
    noise_per_hour = float(noise_level_od / max(dt_med, cfg.eps_dt))
    z_like = float(abs(slope) / max(noise_per_hour, 1e-12))
    z_like = _bounded(z_like, 0.0, 50.0)

    return z_like, slope, delta


def compute_artifact_score(
    t_late: np.ndarray,
    y_late: np.ndarray,
    noise_level_od: float,
    cfg: Stage2ConfigEvidence,
) -> float:
    t = np.asarray(t_late, dtype=float)
    y = np.asarray(y_late, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]

    if t.size < 3:
        return 0.5

    idx = np.argsort(t)
    t, y = t[idx], y[idx]

    indicators: list[float] = []
    mu = float(np.nanmean(y))
    sd = float(np.nanstd(y))
    cv = sd / max(abs(mu), 1e-9)
    cv_score = _bounded(
        (cv - cfg.artifact_cv_low) / (cfg.artifact_cv_high - cfg.artifact_cv_low),
        0.0, 1.0,
    )

    indicators.append(cv_score)

    diffs = np.diff(y)
    if diffs.size >= 2 and np.isfinite(noise_level_od) and noise_level_od > 0:
        # Only consider "meaningful" diffs (above a noise-scaled threshold).
        eps = float(cfg.artifact_osc_noise_mult * noise_level_od)
        sig = diffs[np.abs(diffs) > eps]
        if sig.size >= 2:
            s = np.sign(sig)
            sc = int(np.sum(np.diff(s) != 0))
            osc_score = _bounded(sc / max(sig.size - 1, 1), 0.0, 1.0)
        else:
            osc_score = 0.0
    else:
        osc_score = 0.0
    indicators.append(osc_score)

    evap_score = 0.0
    if t.size >= 4:
        try:
            slope, intercept, r_value, _, _ = stats.linregress(t, y)
            r2 = float(r_value * r_value)
            # strong linear decrease yields higher score
            if slope < cfg.artifact_evap_slope:
                evap_score = _bounded(r2, 0.0, 1.0)
        except Exception:
            evap_score = 0.0
    indicators.append(evap_score)

    score = float(np.nanmean(indicators))
    score = _bounded(score, 0.0, 1.0)
    return score


def _dynamic_min_late_points(t_early: np.ndarray, cfg: Stage2ConfigEvidence) -> int:
    t_early = np.asarray(t_early, dtype=float)
    t_early = t_early[np.isfinite(t_early)]

    if t_early.size >= 2:
        t_early = np.sort(t_early)
        early_span_hours = float(t_early[-1] - t_early[0])
        if early_span_hours > 0:
            early_rate_per_hour = (t_early.size - 1) / early_span_hours
        else:
            early_rate_per_hour = cfg.min_late_points_fallback_rate_per_hour
    else:
        early_rate_per_hour = cfg.min_late_points_fallback_rate_per_hour

    raw = int(round(early_rate_per_hour * cfg.min_late_hours_anchor))
    return int(np.clip(raw, cfg.min_late_points_floor, cfg.min_late_points_ceiling))


def compute_data_quality(
    t_late: np.ndarray,
    y_late: np.ndarray,
    cfg: Stage2ConfigEvidence,
    min_late_points_dynamic: int,
) -> float:
    t = np.asarray(t_late, dtype=float)
    y = np.asarray(y_late, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]

    if t.size < min_late_points_dynamic:
        return 0.0

    idx = np.argsort(t)
    t, y = t[idx], y[idx]

    size_quality = min(1.0, t.size / max(2 * min_late_points_dynamic, 1))
    span = float(t[-1] - t[0])
    span_quality = min(1.0, span / 4.0)  # prefer >=4h late span
    finite_quality = 1.0  
    q = float(np.mean([size_quality, span_quality, finite_quality]))
    return _bounded(q, 0.0, 1.0)


def compute_evidence_scores(
    wide_row: pd.Series,
    time_cols: list[str],
    cfg: Stage2ConfigEvidence,
) -> EvidenceScores:
    t_all = np.array([parse_time_from_header(str(c)) for c in time_cols], dtype=float)
    y_all = pd.to_numeric(wide_row[time_cols], errors="coerce").to_numpy(dtype=float)

    m = np.isfinite(t_all) & np.isfinite(y_all)
    t_all, y_all = t_all[m], y_all[m]

    if t_all.size < cfg.min_late_points_floor:
        return EvidenceScores(
            growth_z_like=0.0,
            artifact_score=0.5,
            decline_score=0.0,
            data_quality=0.0,
            confidence=0.0,
            n_late_points=0,
            min_late_points_required=cfg.min_late_points_floor,

        )

    idx = np.argsort(t_all)
    t_all, y_all = t_all[idx], y_all[idx]

    early_mask = t_all <= float(cfg.stage2_start)
    late_mask = t_all > float(cfg.stage2_start)

    y_early = y_all[early_mask]
    t_late = t_all[late_mask]
    y_late = y_all[late_mask]

    n_late = int(np.isfinite(y_late).sum())
    span = float(np.nanmax(t_late) - np.nanmin(t_late)) if t_late.size > 0 else np.nan
    min_late_points_dynamic = _dynamic_min_late_points(t_all[early_mask], cfg)
    if np.isfinite(span) and span > 0:
        expected_late_pts = int(round(span / float(cfg.late_window_reference_step_hours))) + 1
        late_missing_frac = float(max(0, expected_late_pts - n_late) / expected_late_pts) if expected_late_pts > 0 else np.nan
        density_ok = np.isfinite(late_missing_frac) and (late_missing_frac <= float(cfg.late_window_max_missing_frac))
    else:
        density_ok = True

    late_coverage_ok = bool(n_late >= min_late_points_dynamic and density_ok)

    if n_late < min_late_points_dynamic or t_late.size < min_late_points_dynamic or not density_ok:
        return EvidenceScores(
            growth_z_like=0.0,
            artifact_score=0.5,
            decline_score=0.0,
            data_quality=0.0,
            confidence=0.0,
            n_late_points=n_late,
            late_span_hours=float(span) if np.isfinite(span) else np.nan,
            late_coverage_ok=late_coverage_ok,
            min_late_points_required=min_late_points_dynamic,

        )

    noise_level = compute_noise_baseline_from_diffs(y_early, y_late, cfg)
    z_like, slope, delta = compute_growth_evidence_z_like(t_late, y_late, noise_level, cfg)
    artifact_score = compute_artifact_score(t_late, y_late, noise_level, cfg)
    decline_score = compute_decline_score(t_late, y_late, cfg)
    data_quality = compute_data_quality(t_late, y_late, cfg, min_late_points_dynamic)
    span = float(np.nanmax(t_late) - np.nanmin(t_late)) if t_late.size > 0 else np.nan
    evidence_strength = _bounded(z_like / 4.0, 0.0, 1.0)
    artifact_penalty = 1.0 - _bounded(artifact_score, 0.0, 1.0)
    confidence = float(data_quality * evidence_strength * artifact_penalty)
    confidence = _bounded(confidence, 0.0, 1.0)

    return EvidenceScores(
        growth_z_like=float(z_like),
        artifact_score=float(artifact_score),
        decline_score=float(decline_score),
        data_quality=float(data_quality),
        confidence=float(confidence),
        late_slope=float(slope) if np.isfinite(slope) else np.nan,
        late_delta=float(delta) if np.isfinite(delta) else np.nan,
        noise_level=float(noise_level) if np.isfinite(noise_level) else np.nan,
        n_late_points=int(n_late),
        late_span_hours=float(span) if np.isfinite(span) else np.nan,
        late_coverage_ok=True,
        min_late_points_required=int(min_late_points_dynamic),
    )


def compute_decline_score(
    t_late: np.ndarray,
    y_late: np.ndarray,
    cfg: Stage2ConfigEvidence,
) -> float:
    t = np.asarray(t_late, dtype=float)
    y = np.asarray(y_late, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]

    if t.size < 4:
        return 0.0

    idx = np.argsort(t)
    t, y = t[idx], y[idx]

    try:
        slope, intercept, r_value, _, _ = stats.linregress(t, y)
        r2 = float(r_value * r_value)
    except Exception:
        return 0.0

    if slope >= cfg.artifact_evap_slope:
        return 0.0

    return _bounded(r2, 0.0, 1.0)


def compute_stage2_checker_status(
    stage1_label: str,
    stage1_confidence: float,
    evidence: EvidenceScores,
    cfg: Stage2ConfigEvidence,
) -> tuple[str, str, dict[str, Any]]:
    s1 = str(stage1_label or "").strip()
    ed = {
        "growth_z_like": float(evidence.growth_z_like),
        "artifact_score": float(evidence.artifact_score),
        "data_quality": float(evidence.data_quality),
        "decision_confidence": float(evidence.confidence),
        "decline_score": float(evidence.decline_score),
        "late_slope": float(evidence.late_slope) if np.isfinite(evidence.late_slope) else np.nan,
        "late_delta": float(evidence.late_delta) if np.isfinite(evidence.late_delta) else np.nan,
        "noise_level": float(evidence.noise_level) if np.isfinite(evidence.noise_level) else np.nan,
        "late_n_points": int(evidence.n_late_points),
        "min_late_points_required": int(evidence.min_late_points_required),
        "late_span_hours": float(evidence.late_span_hours) if np.isfinite(evidence.late_span_hours) else np.nan,
    }

    if float(evidence.data_quality) < float(cfg.quality_threshold):
        return "Insufficient", "S2_INSUFFICIENT_DATA_QUALITY", ed

    strong_growth = float(evidence.growth_z_like) >= float(cfg.growth_z_threshold)
    strong_artifact = float(evidence.artifact_score) >= float(cfg.artifact_score_threshold)

    if s1 == "Invalid":
        if strong_growth and (not strong_artifact):
            return "Contradiction", "S2_CONTRADICTORY_LATE_GROWTH", ed
        return "Corroborated", "S2_CORROBORATES_INVALID", ed

    if s1 == "Valid":
        strong_decline = float(evidence.decline_score) >= float(cfg.decline_score_threshold)
        if strong_artifact:
            return "Contradiction", "S2_ARTIFACT_DETECTED", ed
        if strong_decline:
            return "Contradiction", "S2_LATE_DECLINE_DETECTED", ed

        if strong_growth:
            return "Corroborated", "S2_CONTINUED_GROWTH", ed
        return "Corroborated", "S2_STABLE_OR_PLATEAU", ed

    return "Insufficient", "S2_STAGE1_MISSING_OR_UNKNOWN", ed