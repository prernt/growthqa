from __future__ import annotations

import numpy as np
import pandas as pd

from growthqa.stage2.late_window import (
    S2_REASON_ARTIFACT_EVAPORATION,
    S2_REASON_CORROBORATION,
    S2_REASON_DRIFT_NOISE,
    S2_REASON_LATE_GROWTH,
    S2_REASON_ARTIFACT,
    S2_REASON_TOO_SPARSE_LATE,
    Stage2Config,
    compute_late_features,
    compute_stage2_decision,
)


def _row_from_series(times: np.ndarray, y: np.ndarray) -> tuple[pd.Series, list[str]]:
    cols = [f"T{float(t):.1f} (h)" for t in times]
    return pd.Series({c: v for c, v in zip(cols, y)}), cols


def test_growth_then_plateau():
    cfg = Stage2Config()
    t = np.arange(0.0, 25.0, 1.0)
    y = 0.05 + 0.95 * (1.0 - np.exp(-t / 5.0))
    y[t > 18] = 0.98
    row, cols = _row_from_series(t, y)
    feats = compute_late_features(row, cols, cfg)
    s2_label, reason = compute_stage2_decision("Valid", 0.72, feats, cfg)
    assert feats["plateau_detected"] is True
    assert s2_label == "Valid"
    assert reason == S2_REASON_CORROBORATION


def test_growth_then_decline():
    cfg = Stage2Config()
    t = np.arange(0.0, 25.0, 1.0)
    y = 0.05 + 0.9 * (1.0 - np.exp(-t / 5.0))
    late = t >= 18
    y[late] = y[late] - 0.03 * (t[late] - 18)
    row, cols = _row_from_series(t, y)
    feats = compute_late_features(row, cols, cfg)
    s2_label, reason = compute_stage2_decision("Valid", 0.7, feats, cfg)
    assert feats["decline_detected"] is True
    assert s2_label == "Valid"
    assert reason == S2_REASON_CORROBORATION


def test_drift_only_pattern():
    cfg = Stage2Config()
    t = np.arange(0.0, 25.0, 1.0)
    y = np.full_like(t, 0.08, dtype=float)
    y[t > 16] = 0.08 + 0.002 * (t[t > 16] - 16)  # tiny monotonic drift
    row, cols = _row_from_series(t, y)
    feats = compute_late_features(row, cols, cfg)
    s2_label, reason = compute_stage2_decision("Invalid", 0.82, feats, cfg)
    assert feats["drift_detected"] is True
    assert s2_label == "Invalid"
    assert reason == S2_REASON_DRIFT_NOISE


def test_noisy_oscillation_pattern():
    cfg = Stage2Config()
    t = np.arange(0.0, 25.0, 1.0)
    y = 0.08 + 0.15 * (1.0 - np.exp(-t / 8.0))
    late = t > 16
    y[late] = y[late] + 0.08 * np.sin(2.4 * (t[late] - 16))
    row, cols = _row_from_series(t, y)
    feats = compute_late_features(row, cols, cfg)
    s2_label, reason = compute_stage2_decision("Valid", 0.58, feats, cfg)
    assert feats["noise_detected"] is True
    assert s2_label == "Unsure"
    assert reason == S2_REASON_ARTIFACT


def test_late_growth_only_pattern():
    cfg = Stage2Config()
    t = np.arange(0.0, 25.0, 1.0)
    y = np.full_like(t, 0.08, dtype=float)
    late = t > 16
    y[late] = 0.08 + 0.03 * (t[late] - 16)  # strong late rise
    row, cols = _row_from_series(t, y)
    feats = compute_late_features(row, cols, cfg)
    s2_label, reason = compute_stage2_decision("Invalid", 0.76, feats, cfg)
    assert feats["late_growth_detected"] is True
    assert s2_label == "Invalid"
    assert reason == S2_REASON_ARTIFACT_EVAPORATION


def test_late_growth_nonlinear_stays_unsure():
    cfg = Stage2Config()
    t = np.arange(0.0, 25.0, 1.0)
    y = np.full_like(t, 0.08, dtype=float)
    late = t > 16
    z = t[late] - 16.0
    y[late] = 0.08 + 0.012 * z + 0.003 * z * z
    row, cols = _row_from_series(t, y)
    feats = compute_late_features(row, cols, cfg)
    s2_label, reason = compute_stage2_decision("Invalid", 0.76, feats, cfg)
    assert feats["late_growth_detected"] is True
    assert s2_label == "Unsure"
    assert reason == S2_REASON_LATE_GROWTH


def test_late_too_sparse_gives_unsure_reason():
    cfg = Stage2Config(late_min_points=5)
    t = np.array([0, 4, 8, 12, 16, 22.5, 25.5, 29.5, 31.5], dtype=float)
    y = np.array([0.05, 0.06, 0.08, 0.10, 0.11, 0.12, 0.13, 0.135, 0.14], dtype=float)
    row, cols = _row_from_series(t, y)
    feats = compute_late_features(row, cols, cfg)
    s2_label, reason = compute_stage2_decision("Invalid", 0.8, feats, cfg)
    assert feats["has_late_data"] is True
    assert float(feats["raw_observed_tmax"]) == 31.5
    assert int(feats["late_n_points"]) == 4
    assert feats["late_too_sparse"] is True
    assert s2_label == "Unsure"
    assert reason == S2_REASON_TOO_SPARSE_LATE
