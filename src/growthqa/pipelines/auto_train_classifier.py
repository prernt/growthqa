from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from growthqa.classifier.train_from_meta import train_from_meta_csv
from growthqa.classifier.train_from_meta import NOTEBOOK_STAGE1_CUSTOM_FEATURES
from growthqa.preprocess.timegrid import parse_time_from_header
from growthqa.pipelines.build_meta_dataset import run_merge_preprocess_meta
from growthqa.synthetic.timeseries_curve_data import build_synthetic_training_set

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class TimeProfile:
    tmax_hours: float
    step_hours: float


def _robust_step_from_times(times: np.ndarray) -> float:
    """Infer representative sampling interval (mode-of-diffs with median fallback)."""
    t = np.array(times, dtype=float)
    t = np.unique(t[np.isfinite(t)])
    if t.size < 2:
        return 0.25
    t.sort()
    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 0.25

    # rounding stabilizes the mode (millihours)
    rd = np.round(diffs, 3)
    vals, counts = np.unique(rd, return_counts=True)
    mode = float(vals[np.argmax(counts)])
    return mode if mode > 0 else float(np.nanmedian(diffs))


def infer_time_profile_from_wide_csv(path: str | Path) -> TimeProfile:
    """
    Supports two wide formats:

    A) time-in-header:
       columns like T0 (h), T0.25 (h), ...

    B) explicit time column:
       first column like 'Time (h)' / 'time' / 't', with numeric values (0, 0.5, 1, ...)
       and remaining columns are curves.
    """
    path = Path(path)

    # Read small sample for header-based parsing
    df_head = pd.read_csv(path, nrows=5)

    # --- A) time in header columns ---
    times = []
    for c in df_head.columns:
        tt = parse_time_from_header(str(c))
        if tt is not None:
            times.append(float(tt))

    if times:
        times = np.array(sorted(set(times)), dtype=float)
        tmax = float(np.nanmax(times))
        step = float(_robust_step_from_times(times))
        return TimeProfile(tmax_hours=tmax, step_hours=step)

    # --- B) explicit time column (like 'Time (h)') ---
    # Read only the first column fully (cheap) to infer timepoints
    full = pd.read_csv(path, usecols=[0])
    time_col = full.columns[0]
    t = pd.to_numeric(full[time_col], errors="coerce").dropna().to_numpy(dtype=float)

    t = np.unique(t)
    if t.size < 2:
        raise ValueError(
            f"No time columns detected in header AND time column '{time_col}' has <2 valid points in: {path}"
        )

    t.sort()
    tmax = float(np.nanmax(t))
    step = float(_robust_step_from_times(t))
    return TimeProfile(tmax_hours=tmax, step_hours=step)

def _score_lab_candidate(desired: TimeProfile, candidate: TimeProfile) -> float:
    """Lower is better. Strong penalty if candidate shorter than desired."""
    short_penalty = 1000.0 if candidate.tmax_hours + 1e-9 < desired.tmax_hours else 0.0
    return (
        short_penalty
        + abs(candidate.tmax_hours - desired.tmax_hours) * 10.0
        + abs(candidate.step_hours - desired.step_hours) * 100.0
    )


def select_lab_file(
    *,
    train_data_dir: str | Path,
    desired: TimeProfile,
    namespace_regex: str = r".+\.csv$",
) -> Path:
    """
    Select lab file from data/train_data/.

    Rules:
    - Consider .csv files excluding: meta/raw/final/syn
    - File must look like wide format (has T.. (h) columns)
    - Score by closeness of (tmax, step), penalize too-short tmax
    """
    train_data_dir = Path(train_data_dir)
    rx = re.compile(namespace_regex, flags=re.I)

    excluded = {"meta.csv", "raw_merged.csv", "final_merged.csv", "syn.csv"}
    candidates = [p for p in train_data_dir.glob("*.csv") if p.name not in excluded and rx.match(p.name)]

    if not candidates:
        raise FileNotFoundError(
            f"No lab data files found in {train_data_dir}. "
            "Expected 1+ .csv besides meta/raw/final/syn."
        )

    best, best_score = None, float("inf")
    for p in candidates:
        try:
            prof = infer_time_profile_from_wide_csv(p)
        except Exception:
            continue
        sc = _score_lab_candidate(desired, prof)
        if sc < best_score:
            best, best_score = p, sc

    if best is None:
        raise ValueError("No lab CSV candidate looked like a wide file with 'T.. (h)' columns.")
    return best


def generate_synthetic_wide_csv(
    *,
    out_syn_csv: str | Path,
    tmax_hours: float,
    step_hours: float,
    seed: int = 123,
    n_reps: int = 10,
    script_path: str | Path | None = None,
) -> Path:
    """
    Generate syn.csv using the existing generator script (subprocess).
    Writes to data/train_data/syn.csv
    """
    out_syn_csv = Path(out_syn_csv)
    out_syn_csv.parent.mkdir(parents=True, exist_ok=True)

    if script_path is None:
        script_path = ROOT / "scripts" / "timeseries_curve_data.py"
    script_path = Path(script_path)
    if not script_path.exists():
        raise FileNotFoundError(f"Synthetic generator script not found: {script_path}")

    out_dir = out_syn_csv.parent
    file_stem = "syn"
    expected = out_dir / f"timeseries_wide_{file_stem}.csv"
    if expected.exists():
        expected.unlink(missing_ok=True)

    cmd = [
        sys.executable, str(script_path),
        "--seed", str(seed),
        "--n-reps", str(n_reps),
        "--max-time", str(float(tmax_hours)),
        "--time-step", str(float(step_hours)),
        "--time-unit", "h",
        "--output-dir", str(out_dir),
        "--file-stem", file_stem,
    ]
    subprocess.check_call(cmd)

    if not expected.exists():
        raise FileNotFoundError(f"Synthetic generator did not write expected file: {expected}")

    if out_syn_csv.exists():
        out_syn_csv.unlink()
    expected.replace(out_syn_csv)
    return out_syn_csv


def build_training_meta_from_uploaded_file(
    *,
    uploaded_wide_path: str | Path,
    train_data_dir: str | Path = ROOT / "data" / "train_data",
    blank_status_csv: str | Path | None = None,
    blank_default: str = "RAW",
    step_override: float | None = None,
    tmax_override: float | None = None,
) -> dict:
    """upload -> bucketed synthetic + compatible labs -> raw/final/meta"""
    train_data_dir = Path(train_data_dir)
    train_data_dir.mkdir(parents=True, exist_ok=True)

    desired = infer_time_profile_from_wide_csv(uploaded_wide_path)
    if step_override is not None:
        desired = TimeProfile(desired.tmax_hours, float(step_override))
    if tmax_override is not None:
        desired = TimeProfile(float(tmax_override), desired.step_hours)

    synth_payload = build_synthetic_training_set(
        train_data_dir=train_data_dir,
        out_syn_csv=train_data_dir / "syn.csv",
        seed=123,
        time_unit="h",
    )
    syn_path = Path(synth_payload["syn_path"])
    lab_inputs = [str(p) for p in synth_payload.get("lab_inputs", [])]
    inputs = [str(syn_path)] + lab_inputs

    out_raw = train_data_dir / "raw_merged.csv"
    out_final = train_data_dir / "final_merged.csv"
    out_meta = train_data_dir / "meta.csv"

    run_merge_preprocess_meta(
        inputs=inputs,
        out_raw=str(out_raw),
        out_final=str(out_final),
        out_meta=str(out_meta),
        step=float(step_override) if step_override is not None else 0.5,
        tmax_hours=float(tmax_override) if tmax_override is not None else 16.0,
        auto_tmax=False,
        augment_trunc=True,
        trunc_horizons=[8.0, 10.0, 12.0, 14.75, 16.0],
        trunc_per_curve=3,
        trunc_seed=123,
        blank_status_csv=str(blank_status_csv) if blank_status_csv else None,
        blank_default=str(blank_default),
    )

    return {
        "desired_tmax_hours": desired.tmax_hours,
        "desired_step_hours": desired.step_hours,
        "syn_path": str(syn_path),
        "lab_path": lab_inputs[0] if lab_inputs else None,
        "lab_paths": lab_inputs,
        "raw_merged_path": str(out_raw),
        "final_merged_path": str(out_final),
        "meta_path": str(out_meta),
        "bucket_summary": synth_payload.get("bucket_summaries", []),
    }


def auto_train_classifier_from_uploaded_file(
    *,
    uploaded_wide_path: str | Path,
    train_data_dir: str | Path = ROOT / "data" / "train_data",
    models_out_dir: str | Path = ROOT / "classifier_output" / "saved_models_selected",
    blank_status_csv: str | Path | None = None,
    blank_default: str = "RAW",
    step_override: float | None = None,
    tmax_override: float | None = None,
) -> dict:
    """upload -> syn+lab -> meta.csv -> train -> saved joblibs"""
    models_out_dir = Path(models_out_dir)
    if models_out_dir.exists():
        shutil.rmtree(models_out_dir)
    models_out_dir.mkdir(parents=True, exist_ok=True)

    prep = build_training_meta_from_uploaded_file(
        uploaded_wide_path=uploaded_wide_path,
        train_data_dir=train_data_dir,
        blank_status_csv=blank_status_csv,
        blank_default=blank_default,
        step_override=step_override,
        tmax_override=tmax_override,
    )

    train_out = train_from_meta_csv(
        meta_csv=prep["meta_path"],
        art_dir=models_out_dir,
        selected_features=NOTEBOOK_STAGE1_CUSTOM_FEATURES,
    )

    return {**prep, **train_out}


def train_classifier_from_meta_file(
    *,
    meta_csv_path: str | Path,
    models_out_dir: str | Path = ROOT / "classifier_output" / "saved_models_selected",
    selected_features: list[str] | None = None,
) -> dict:
    """
    Train classifier directly from an existing meta.csv and refresh output models directory.
    """
    models_out_dir = Path(models_out_dir)
    if models_out_dir.exists():
        shutil.rmtree(models_out_dir)
    models_out_dir.mkdir(parents=True, exist_ok=True)

    return train_from_meta_csv(
        meta_csv=meta_csv_path,
        art_dir=models_out_dir,
        selected_features=selected_features,
    )
