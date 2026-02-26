# src/growthqa/stage2/late_window.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import theilslopes

from growthqa.preprocess.timegrid import parse_time_from_header


# ============================================================
# Revised Stage-2 (Evidence-based) + 4 defensibility tweaks
#   Tweak 1: noise baseline from robust MAD of first differences
#   Tweak 2: growth evidence = standardized slope (z-like, unit-consistent)
#   Tweak 3: artifact is a SCORE (0..1), not a "probability"
#   Tweak 4: Stage-2 outputs checker status only:
#            Corroborated / Contradiction / Insufficient
# ============================================================


# ----------------------------
# Config (simple + defensible)
# ----------------------------
@dataclass
class Stage2ConfigEvidence:
    """
    Evidence-based Stage-2 config (thesis-friendly).

    Philosophy:
      - Stage-2 is a CHECKER, not a re-classifier.
      - It uses late-window raw data only and produces evidence scores.
    """
    stage2_start: float = 16.0

    # Quality gate
    min_late_points: int = 5
    quality_threshold: float = 0.30

    # Evidence thresholds
    growth_z_threshold: float = 2.0          # "2-sigma" style threshold (z-like)
    artifact_score_threshold: float = 0.70   # high artifact severity
    unsure_margin: float = 0.10              # closeness margin around thresholds (optional)

    # Small numeric safeties
    min_noise_level: float = 0.005           # OD units (robust floor)
    eps_dt: float = 1e-9

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


# ----------------------------
# Evidence Scores
# ----------------------------
@dataclass
class EvidenceScores:
    """
    Clean evidence quantification for Stage-2.

    IMPORTANT (defensibility):
      - growth_z_like is a standardized effect size (unit-consistent), not a literal z-score
      - artifact_score is a bounded score in [0,1], not a calibrated probability
    """
    growth_z_like: float        # standardized slope evidence (z-like)
    artifact_score: float       # [0,1] severity score (NOT probability)
    data_quality: float         # [0,1]
    confidence: float           # [0,1] overall decision confidence (simple mapping)

    # Supporting metrics (for audit/debug)
    late_slope: float = np.nan
    late_delta: float = np.nan
    noise_level: float = np.nan
    n_late_points: int = 0
    late_span_hours: float = np.nan

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "growth_z_like": float(self.growth_z_like),
            "artifact_score": float(self.artifact_score),
            "data_quality": float(self.data_quality),
            "confidence": float(self.confidence),
            "late_slope": float(self.late_slope) if np.isfinite(self.late_slope) else np.nan,
            "late_delta": float(self.late_delta) if np.isfinite(self.late_delta) else np.nan,
            "noise_level": float(self.noise_level) if np.isfinite(self.noise_level) else np.nan,
            "n_late_points": int(self.n_late_points),
            "late_span_hours": float(self.late_span_hours) if np.isfinite(self.late_span_hours) else np.nan,
        }


# ----------------------------
# Helpers (robust statistics)
# ----------------------------
def _mad_std(x: np.ndarray) -> float:
    """
    Robust std estimate using MAD: sigma ~= 1.4826 * MAD.
    Returns 0.0 if not enough finite values.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return float(1.4826 * mad)


def _bounded(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _safe_float(x: Any, default: float = np.nan) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float(default)


# ----------------------------
# Tweak 1: Noise baseline
# ----------------------------
def compute_noise_baseline_from_diffs(
    y_early: np.ndarray,
    y_late: np.ndarray,
    cfg: Stage2ConfigEvidence,
) -> float:
    """
    Robust noise estimate in OD units derived from FIRST DIFFERENCES.

    Why this is defensible:
      - It measures short-term variability, not absolute level.
      - It is robust (MAD), stable, and unit-consistent with slope standardization.

    Strategy:
      - Prefer early diffs if early exists and has enough points.
      - Fallback to late diffs if early is insufficient.
      - Apply minimum floor to avoid exploding standardized scores.
    """
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


# ----------------------------
# Tweak 2: Growth evidence (z-like)
# ----------------------------
def compute_growth_evidence_z_like(
    t_late: np.ndarray,
    y_late: np.ndarray,
    noise_level_od: float,
    cfg: Stage2ConfigEvidence,
) -> tuple[float, float, float]:
    """
    Computes:
      - z_like: |TheilSenSlope| / (noise_per_hour)
      - slope:  robust slope estimate
      - delta:  endpoint change

    Key point:
      - noise_level_od is OD noise at the increment scale (from diffs)
      - convert to OD/hour via median dt
      - results are dimensionally consistent and easy to defend

    Returns: (z_like, slope, delta)
    """
    t = np.asarray(t_late, dtype=float)
    y = np.asarray(y_late, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]

    if t.size < 2:
        return 0.0, np.nan, np.nan

    # Sort by time
    idx = np.argsort(t)
    t, y = t[idx], y[idx]

    # Robust slope
    try:
        slope, intercept, _, _ = theilslopes(y, t)
        slope = float(slope)
    except Exception:
        # fallback: simple slope
        denom = float(t[-1] - t[0])
        slope = float((y[-1] - y[0]) / max(denom, cfg.eps_dt))

    # Delta
    delta = float(y[-1] - y[0])

    # Convert OD noise to OD/hour
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > cfg.eps_dt)]
    dt_med = float(np.nanmedian(dt)) if dt.size > 0 else 1.0
    noise_per_hour = float(noise_level_od / max(dt_med, cfg.eps_dt))

    z_like = float(abs(slope) / max(noise_per_hour, 1e-12))
    # Bound to keep stable and interpretable
    z_like = _bounded(z_like, 0.0, 50.0)

    return z_like, slope, delta


# ----------------------------
# Tweak 3: Artifact SCORE (not probability)
# ----------------------------
def compute_artifact_score(
    t_late: np.ndarray,
    y_late: np.ndarray,
    noise_level_od: float,
    cfg: Stage2ConfigEvidence,
) -> float:
    """
    Returns artifact_score in [0,1] (severity score).

    Indicators (simple + defensible):
      1) Excessive relative variability (CV-like)
      2) High oscillation rate AFTER noise-thresholding on diffs
      3) Evaporation-like linear decline (soft score using R^2)

    NOTE:
      - This is a SCORE, not a calibrated probability.
    """
    t = np.asarray(t_late, dtype=float)
    y = np.asarray(y_late, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]

    if t.size < 3:
        return 0.5

    # Sort
    idx = np.argsort(t)
    t, y = t[idx], y[idx]

    indicators: list[float] = []

    # (1) Relative variability indicator
    mu = float(np.nanmean(y))
    sd = float(np.nanstd(y))
    cv = sd / max(abs(mu), 1e-9)
    # Map CV to 0..1 (CV ~0.05 low, CV >=0.20 high)
    cv_score = _bounded((cv - 0.05) / (0.20 - 0.05), 0.0, 1.0)
    indicators.append(cv_score)

    # (2) Oscillation indicator (noise-thresholded sign changes)
    diffs = np.diff(y)
    if diffs.size >= 2 and np.isfinite(noise_level_od) and noise_level_od > 0:
        # Only consider "meaningful" diffs
        eps = float(2.0 * noise_level_od)
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

    # (3) Evaporation-like linear decline (soft)
    evap_score = 0.0
    if t.size >= 4:
        try:
            slope, intercept, r_value, _, _ = stats.linregress(t, y)
            r2 = float(r_value * r_value)
            # strong linear decrease yields higher score
            if slope < -0.005:
                evap_score = _bounded(r2, 0.0, 1.0)
        except Exception:
            evap_score = 0.0
    indicators.append(evap_score)

    # Combine: mean keeps it simple; max tends to be too aggressive.
    score = float(np.nanmean(indicators))
    score = _bounded(score, 0.0, 1.0)
    return score


# ----------------------------
# Data quality score
# ----------------------------
def compute_data_quality(
    t_late: np.ndarray,
    y_late: np.ndarray,
    cfg: Stage2ConfigEvidence,
) -> float:
    """
    Quality score in [0,1].

    Components (simple + defensible):
      - size adequacy
      - span adequacy
      - finite ratio
    """
    t = np.asarray(t_late, dtype=float)
    y = np.asarray(y_late, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]

    if t.size < cfg.min_late_points:
        return 0.0

    # Sort
    idx = np.argsort(t)
    t, y = t[idx], y[idx]

    size_quality = min(1.0, t.size / max(2 * cfg.min_late_points, 1))
    span = float(t[-1] - t[0])
    span_quality = min(1.0, span / 4.0)  # prefer >=4h late span
    finite_quality = 1.0  # already filtered finite

    q = float(np.mean([size_quality, span_quality, finite_quality]))
    return _bounded(q, 0.0, 1.0)


# ----------------------------
# Evidence computation (main)
# ----------------------------
def compute_evidence_scores(
    wide_row: pd.Series,
    time_cols: list[str],
    cfg: Stage2ConfigEvidence,
) -> EvidenceScores:
    """
    Computes evidence scores from a single canonical-wide row.
    Uses raw values; does not normalize/smooth/interpolate.
    """
    t_all = np.array([parse_time_from_header(str(c)) for c in time_cols], dtype=float)
    y_all = pd.to_numeric(wide_row[time_cols], errors="coerce").to_numpy(dtype=float)

    m = np.isfinite(t_all) & np.isfinite(y_all)
    t_all, y_all = t_all[m], y_all[m]

    if t_all.size < cfg.min_late_points:
        return EvidenceScores(
            growth_z_like=0.0,
            artifact_score=0.5,
            data_quality=0.0,
            confidence=0.0,
            n_late_points=0,
        )

    # Sort
    idx = np.argsort(t_all)
    t_all, y_all = t_all[idx], y_all[idx]

    early_mask = t_all <= float(cfg.stage2_start)
    late_mask = t_all > float(cfg.stage2_start)

    y_early = y_all[early_mask]
    t_late = t_all[late_mask]
    y_late = y_all[late_mask]

    n_late = int(np.isfinite(y_late).sum())
    if n_late < cfg.min_late_points or t_late.size < cfg.min_late_points:
        return EvidenceScores(
            growth_z_like=0.0,
            artifact_score=0.5,
            data_quality=0.0,
            confidence=0.0,
            n_late_points=n_late,
        )

    # Compute components
    noise_level = compute_noise_baseline_from_diffs(y_early, y_late, cfg)
    z_like, slope, delta = compute_growth_evidence_z_like(t_late, y_late, noise_level, cfg)
    artifact_score = compute_artifact_score(t_late, y_late, noise_level, cfg)
    data_quality = compute_data_quality(t_late, y_late, cfg)

    span = float(np.nanmax(t_late) - np.nanmin(t_late)) if t_late.size > 0 else np.nan

    # Confidence: simple and honest mapping (quality * evidence strength)
    # Normalize z-like (2.0 ≈ threshold) => evidence_strength in [0,1]
    evidence_strength = _bounded(z_like / 4.0, 0.0, 1.0)
    # Penalize if artifact is high
    artifact_penalty = 1.0 - _bounded(artifact_score, 0.0, 1.0)
    confidence = float(data_quality * evidence_strength * artifact_penalty)
    confidence = _bounded(confidence, 0.0, 1.0)

    return EvidenceScores(
        growth_z_like=float(z_like),
        artifact_score=float(artifact_score),
        data_quality=float(data_quality),
        confidence=float(confidence),
        late_slope=float(slope) if np.isfinite(slope) else np.nan,
        late_delta=float(delta) if np.isfinite(delta) else np.nan,
        noise_level=float(noise_level) if np.isfinite(noise_level) else np.nan,
        n_late_points=int(n_late),
        late_span_hours=float(span) if np.isfinite(span) else np.nan,
    )


# ----------------------------
# Tweak 4: Checker-only decision
# ----------------------------
def compute_stage2_checker_status(
    stage1_label: str,
    stage1_confidence: float,
    evidence: EvidenceScores,
    cfg: Stage2ConfigEvidence,
) -> tuple[str, str, dict[str, Any]]:
    """
    Returns:
      status: one of {"Corroborated", "Contradiction", "Insufficient"}
      reason: string code
      evidence_dict: scalar evidence payload for audit/debug

    This does NOT output Valid/Invalid. Stage-2 is a checker only.
    """

    s1 = str(stage1_label or "").strip()
    s1c = _safe_float(stage1_confidence, default=np.nan)

    ed = {
        "growth_z_like": float(evidence.growth_z_like),
        "artifact_score": float(evidence.artifact_score),
        "data_quality": float(evidence.data_quality),
        "decision_confidence": float(evidence.confidence),
        "late_slope": float(evidence.late_slope) if np.isfinite(evidence.late_slope) else np.nan,
        "late_delta": float(evidence.late_delta) if np.isfinite(evidence.late_delta) else np.nan,
        "noise_level": float(evidence.noise_level) if np.isfinite(evidence.noise_level) else np.nan,
        "late_n_points": int(evidence.n_late_points),
        "late_span_hours": float(evidence.late_span_hours) if np.isfinite(evidence.late_span_hours) else np.nan,
    }

    # 1) Quality gate
    if float(evidence.data_quality) < float(cfg.quality_threshold):
        return "Insufficient", "S2_INSUFFICIENT_DATA_QUALITY", ed

    # 2) Evidence flags
    strong_growth = float(evidence.growth_z_like) >= float(cfg.growth_z_threshold)
    strong_artifact = float(evidence.artifact_score) >= float(cfg.artifact_score_threshold)

    # 3) Checker logic
    # If Stage-1 says Invalid, late growth without artifact is a contradiction (delayed growth scenario)
    if s1 == "Invalid":
        if strong_growth and (not strong_artifact):
            return "Contradiction", "S2_CONTRADICTORY_LATE_GROWTH", ed
        return "Corroborated", "S2_CORROBORATES_INVALID", ed

    # If Stage-1 says Valid, strong artifact is a contradiction (Stage-1 likely overly confident)
    if s1 == "Valid":
        if strong_artifact:
            return "Contradiction", "S2_ARTIFACT_DETECTED", ed
        # Whether growth continues or plateaus, Stage-2 corroborates validity
        if strong_growth:
            return "Corroborated", "S2_CONTINUED_GROWTH", ed
        return "Corroborated", "S2_STABLE_OR_PLATEAU", ed

    # Unknown Stage-1 label
    return "Insufficient", "S2_STAGE1_MISSING_OR_UNKNOWN", ed


# ============================================================
# Optional: Legacy-ish wrapper (scalar-only outputs)
#   You said you'll handle integration later; this wrapper can
#   help you keep a similar "late features" table shape.
# ============================================================
def compute_late_features(
    wide_row: pd.Series,
    time_cols: list[str],
    cfg: Stage2ConfigEvidence | None = None,
) -> dict[str, object]:
    """
    Scalar-only late features output (safe for CSV/UI).
    """
    if cfg is None:
        cfg = Stage2ConfigEvidence()

    ev = compute_evidence_scores(wide_row, time_cols, cfg)

    has_late = int(ev.n_late_points) >= int(cfg.min_late_points)
    out: dict[str, object] = {
        "has_late_data": bool(has_late),
        "late_n_points": int(ev.n_late_points),
        "late_span_hours": float(ev.late_span_hours) if np.isfinite(ev.late_span_hours) else np.nan,

        # Core evidence (thesis-friendly)
        "growth_z_like": float(ev.growth_z_like),
        "artifact_score": float(ev.artifact_score),
        "data_quality": float(ev.data_quality),
        "decision_confidence": float(ev.confidence),

        # Supporting metrics
        "late_slope": float(ev.late_slope) if np.isfinite(ev.late_slope) else np.nan,
        "late_delta": float(ev.late_delta) if np.isfinite(ev.late_delta) else np.nan,
        "noise_level": float(ev.noise_level) if np.isfinite(ev.noise_level) else np.nan,

        # Thresholded flags (useful but still defensible)
        "late_growth_detected": bool(ev.growth_z_like >= cfg.growth_z_threshold),
        "artifact_detected": bool(ev.artifact_score >= cfg.artifact_score_threshold),
        "late_window_start": float(cfg.stage2_start),
    }
    return out