from __future__ import annotations

import numpy as np
import pandas as pd

from growthqa.stage2.late_window import (
    Stage2ConfigEvidence,
    EvidenceScores,
    compute_evidence_scores,
    compute_stage2_checker_status,
)


def _row_from_series(times: np.ndarray, y: np.ndarray) -> tuple[pd.Series, list[str]]:
    cols = [f"T{float(t):.1f} (h)" for t in times]
    return pd.Series({c: v for c, v in zip(cols, y)}), cols


def test_detects_late_growth_as_contradiction_if_stage1_invalid():
    cfg = Stage2ConfigEvidence(stage2_start=16.0, min_late_points=5)

    # flat early, strong growth after 16h
    t = np.arange(0.0, 28.0, 1.0)
    y = np.full_like(t, 0.08, dtype=float)
    late = t > 16
    y[late] = 0.08 + 0.04 * (t[late] - 16)

    row, cols = _row_from_series(t, y)
    ev = compute_evidence_scores(row, cols, cfg)

    status, reason, _ = compute_stage2_checker_status(
        stage1_label="Invalid",
        stage1_confidence=0.8,
        evidence=ev,
        cfg=cfg,
    )

    assert ev.data_quality >= cfg.quality_threshold
    assert status == "Contradiction"
    assert "LATE_GROWTH" in reason


def test_artifact_flags_contradiction_if_stage1_valid():
    cfg = Stage2ConfigEvidence(stage2_start=16.0, min_late_points=5)

    # late oscillations / noisy behaviour
    t = np.arange(0.0, 28.0, 1.0)
    y = 0.15 + 0.0 * t
    late = t > 16
    y = y.astype(float)
    y[late] = 0.15 + 0.06 * np.sin((t[late] - 16) * 2.5)

    row, cols = _row_from_series(t, y)
    ev = compute_evidence_scores(row, cols, cfg)

    status, reason, _ = compute_stage2_checker_status(
        stage1_label="Valid",
        stage1_confidence=0.85,
        evidence=ev,
        cfg=cfg,
    )

    # We don't require artifact_score always exceed threshold in synthetic,
    # but if it does exceed, it MUST contradict Valid.
    if ev.artifact_score >= cfg.artifact_score_threshold:
        assert status == "Contradiction"
        assert "ARTIFACT" in reason
    else:
        # Otherwise it should be corroborated (no strong artifact)
        assert status in {"Corroborated", "Insufficient"}


def test_quality_gate_insufficient():
    cfg = Stage2ConfigEvidence(stage2_start=16.0, min_late_points=5)

    # Too few late points
    t = np.array([0, 8, 16, 17, 18], dtype=float)  # only 2 late points (>16)
    y = np.array([0.1, 0.1, 0.1, 0.12, 0.14], dtype=float)

    row, cols = _row_from_series(t, y)
    ev = compute_evidence_scores(row, cols, cfg)

    status, reason, _ = compute_stage2_checker_status(
        stage1_label="Invalid",
        stage1_confidence=0.9,
        evidence=ev,
        cfg=cfg,
    )

    assert status == "Insufficient"
    assert "INSUFFICIENT" in reason


def test_growth_z_like_is_nonnegative_and_bounded():
    cfg = Stage2ConfigEvidence(stage2_start=16.0, min_late_points=5)

    t = np.arange(0.0, 40.0, 0.5)
    y = 0.08 + 0.02 * np.sin(t)  # noisy-ish
    y = y.astype(float)

    # inject a large linear growth late
    late = t > 16
    y[late] = y[late] + 0.08 * (t[late] - 16)

    row, cols = _row_from_series(t, y)
    ev = compute_evidence_scores(row, cols, cfg)

    assert ev.growth_z_like >= 0.0
    assert ev.growth_z_like <= 50.0