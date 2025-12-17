#!/usr/bin/env python3
"""
timeseries_curve_data.py
---------------------------------
Synthetic growth-curve generator that writes a single **wide** CSV:

  timeseries_wide_<file-stem>.csv
    Columns:
      FileName, Test Id, Model Name, Is_Valid,
      T1 = 0 (<unit>), T2 = <t2> (<unit>), ...

Also writes run_info.xlsx exactly as before.

Notes:
  - Models: Logistic, Gompertz, ModifiedGompertz, Richards, Diauxic, Flat
  - Optional missing values & negative outlier injection
"""

import argparse
import logging
import os
from datetime import datetime
import numpy as np
import pandas as pd

# --------------------------
# 1) Model definitions
# --------------------------
def logistic(t, A, mu, lam):
    return A / (1.0 + np.exp((4.0 * mu / A) * (lam - t) + 2.0))

def gompertz(t, A, mu, lam):
    return A * np.exp(-np.exp((mu * np.e / A) * (lam - t) + 1.0))

def modified_gompertz(t, A, mu, lam, alpha, tshift):
    part1 = A * np.exp(-np.exp((mu * np.e / A) * (lam - t) + 1.0))
    part2 = A * np.exp(alpha * (t - tshift))
    return part1 + part2

def richards(t, A, mu, lam, nu):
    return A * (1.0 + nu * np.exp((mu * (1.0 + nu) / A) * (lam - t))) ** (-1.0 / nu)

def diauxic(t, A1, mu1, lam1, A2, mu2, lam2):
    return logistic(t, A1, mu1, lam1) + logistic(t, A2, mu2, lam2)

def flat_line(t, baseline):
    return np.full_like(t, baseline)

# model name -> (fn, param names)
MODEL_SPECS = {
    "Logistic": (logistic, ["A", "mu", "lam"]),
    "Gompertz": (gompertz, ["A", "mu", "lam"]),
    "ModifiedGompertz": (modified_gompertz, ["A", "mu", "lam", "alpha", "tshift"]),
    "Richards": (richards, ["A", "mu", "lam", "nu"]),
    "Diauxic": (diauxic, ["A1", "mu1", "lam1", "A2", "mu2", "lam2"]),
    "Flat": (flat_line, ["baseline"]),
}

GOOD_MODELS = {"Logistic", "Gompertz", "ModifiedGompertz", "Richards"}

# --------------------------
# 2) Noise / corruption
# --------------------------
def inject_missing(y: pd.Series, frac: float, rng: np.random.Generator):
    if frac <= 0: return y
    n = len(y)
    k = max(1, int(round(frac * n)))
    idx = rng.choice(n, size=min(k, n), replace=False)
    y.iloc[idx] = np.nan
    return y

def inject_negative_outliers(y: pd.Series, frac: float, scale_min: float, scale_max: float, rng: np.random.Generator):
    if frac <= 0: return y
    n = len(y)
    k = max(1, int(round(frac * n)))
    idx = rng.choice(n, size=min(k, n), replace=False)
    sub = rng.uniform(scale_min, scale_max, size=len(idx))
    y.iloc[idx] = (y.iloc[idx].values - sub).clip(min=0.0)
    return y

def make_obvious_invalid(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Very obvious invalid curve:
    - big drop to near-zero in the middle or tail.
    """
    y = np.asarray(y, dtype=float).copy()
    n = len(y)
    if n < 4:
        return y

    # Choose a cut in middle third
    cut = rng.integers(n // 3, 2 * n // 3)
    drop_factor = rng.uniform(0.0, 0.2)  # drop to 0–20% of value
    y[cut:] = y[cut] * drop_factor
    return y


def make_subtle_invalid(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Subtle invalid curve:
    - local dip segment (e.g. bubble / misreading) but not totally crazy.
    """
    y = np.asarray(y, dtype=float).copy()
    n = len(y)
    if n < 6:
        return y

    center = rng.integers(n // 4, 3 * n // 4)
    width = rng.integers(2, min(6, n - center))
    max_y = np.nanmax(y) if np.any(np.isfinite(y)) else 1.0
    drop = rng.uniform(0.1, 0.4) * max_y
    y[center:center + width] = np.clip(y[center:center + width] - drop, 0.0, None)
    return y


def make_near_real_invalid(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Very close to realistic but technically invalid curve:
    - tail is suppressed / slightly declining so it never really stabilises.
    """
    y = np.asarray(y, dtype=float).copy()
    n = len(y)
    if n < 6:
        return y

    start_tail = int(0.6 * n)
    factor = rng.uniform(0.4, 0.8)  # reduce plateau
    y[start_tail:] = y[start_tail:] * factor

    # optional very mild downward trend
    trend = np.linspace(0.0, rng.uniform(0.05, 0.15) * np.nanmax(y), n - start_tail)
    y[start_tail:] = np.clip(y[start_tail:] - trend, 0.0, None)
    return y

# --------------------------
# 3) Run-info writer (kept identical in spirit)
# --------------------------
def write_run_info_xlsx(output_dir, file_stem, args, wide_path, stats):
    """
    Appends run metadata to run_info.xlsx without overwriting prior runs.

    Sheets:
      - RUNS   (cumulative log; one row per run; appended)
      - INFO   (latest run snapshot; refreshed each execution)
      - PARAMS (latest run args;   refreshed each execution)
    """
    import os
    from datetime import datetime
    from openpyxl import Workbook, load_workbook

    xlsx_path = os.path.join(output_dir, "run_info.xlsx")
    os.makedirs(output_dir, exist_ok=True)

    # 1) Open or create workbook
    if os.path.exists(xlsx_path):
        wb = load_workbook(xlsx_path)
    else:
        wb = Workbook()

    # 2) Ensure RUNS sheet exists with header
    if "RUNS" in wb.sheetnames:
        ws_runs = wb["RUNS"]
        # If the sheet exists but is empty, write header
        if ws_runs.max_row == 1 and ws_runs.max_column == 1 and ws_runs["A1"].value is None:
            ws_runs.append([
                "Timestamp", "Output Dir", "Wide CSV", "Seed",
                "points_per_curve", "n_curves", "n_rows",
                "file_stem", "time_unit", "max_time", "time_step",
                "n_reps", "noise_level",
                "pct_missing_curves", "missing_frac_per_curve",
                "pct_outlier_curves", "outlier_frac_per_curve",
                "outlier_scale_min", "outlier_scale_max"
            ])
    else:
        ws_runs = wb.create_sheet("RUNS")
        ws_runs.append([
            "Timestamp", "Output Dir", "Wide CSV", "Seed",
            "points_per_curve", "n_curves", "n_rows",
            "file_stem", "time_unit", "max_time", "time_step",
            "n_reps", "noise_level",
            "pct_missing_curves", "missing_frac_per_curve",
            "pct_outlier_curves", "outlier_frac_per_curve",
            "outlier_scale_min", "outlier_scale_max"
        ])

    # 3) Append current run to RUNS
    ts = datetime.now().isoformat(timespec="seconds")
    ws_runs.append([
        ts,
        os.path.abspath(output_dir),
        os.path.abspath(wide_path),
        args.seed,
        stats.get("points_per_curve", 0),
        stats.get("n_curves", 0),
        stats.get("n_rows", 0),
        file_stem,
        getattr(args, "time_unit", None),
        getattr(args, "max_time", None),
        getattr(args, "time_step", None),
        getattr(args, "n_reps", None),
        getattr(args, "noise_level", None),
        getattr(args, "pct_missing_curves", None),
        getattr(args, "missing_frac_per_curve", None),
        getattr(args, "pct_outlier_curves", None),
        getattr(args, "outlier_frac_per_curve", None),
        getattr(args, "outlier_scale_min", None),
        getattr(args, "outlier_scale_max", None),
    ])

    # 4) Refresh INFO (latest snapshot)
    if "INFO" in wb.sheetnames:
        del wb["INFO"]
    ws_info = wb.create_sheet("INFO")
    a1_text = (
        f"Output: {os.path.abspath(output_dir)} | "
        f"File: {os.path.basename(wide_path)} | "
        f"Seed: {args.seed} | "
        f"Timestamp: {ts}"
    )
    ws_info["A1"] = a1_text
    ws_info["A3"] = "points_per_curve"; ws_info["B3"] = stats.get("points_per_curve", 0)
    ws_info["A4"] = "n_curves";         ws_info["B4"] = stats.get("n_curves", 0)
    ws_info["A5"] = "n_rows";           ws_info["B5"] = stats.get("n_rows", 0)

    # 5) Refresh PARAMS (latest args)
    if "PARAMS" in wb.sheetnames:
        del wb["PARAMS"]
    ws_params = wb.create_sheet("PARAMS")
    ws_params.append(["arg", "value"])
    for k, v in sorted(vars(args).items()):
        ws_params.append([k, str(v)])

    # 6) Save (in-place; preserves all old content and appended RUNS)
    # Remove the default 'Sheet' if it's still there and empty
    if "Sheet" in wb.sheetnames and wb["Sheet"].max_row == 1 and wb["Sheet"]["A1"].value is None:
        del wb["Sheet"]
    wb.save(xlsx_path)

# --------------------------
# 4) Main
# --------------------------
def main():
    p = argparse.ArgumentParser(description="Synthetic Growth Curve Generator (WIDE CSV only)")
    p.add_argument("--seed",                  type=int,   default=123)
    p.add_argument("--n-reps",                type=int,   default=10, help="replicates per model")
    p.add_argument("--max-time",              type=float, default=24.0, help="max time")
    p.add_argument("--time-step",             type=float, default=0.5, help="time step")
    p.add_argument("--time-unit",             type=str,   default="h", choices=["s","m","h"], help="unit label for header")
    p.add_argument("--noise-level",           type=float, default=0.05, help="Gaussian noise stdev")
    p.add_argument("--pct-missing-curves",    type=float, default=0.1,  help="fraction of curves with missing values")
    p.add_argument("--missing-frac-per-curve",type=float, default=0.1,  help="fraction of points set to NaN in selected curves")
    p.add_argument("--pct-outlier-curves",    type=float, default=0.05, help="fraction of curves with negative outliers")
    p.add_argument("--outlier-frac-per-curve",type=float, default=0.05, help="fraction of points made outliers in selected curves")
    p.add_argument("--outlier-scale-min",     type=float, default=0.1)
    p.add_argument("--outlier-scale-max",     type=float, default=0.3)
    p.add_argument("--output-dir",            type=str,   default="./dataNew")
    p.add_argument("--file-stem",             type=str,   default="timedata")
        # NEW: curve-quality and invalid patterns
    p.add_argument(
        "--pct-high-quality-valid",
        type=float,
        default=0.3,
        help="Fraction of GOOD model curves made very realistic (low noise, no missing/outliers)."
    )
    p.add_argument(
        "--pct-obvious-invalid-curves",
        type=float,
        default=0.1,
        help="Fraction of GOOD model curves corrupted into *obviously* invalid shapes."
    )
    p.add_argument(
        "--pct-subtle-invalid-curves",
        type=float,
        default=0.1,
        help="Fraction of GOOD model curves corrupted into *subtle* invalid shapes."
    )
    p.add_argument(
        "--pct-nearreal-invalid-curves",
        type=float,
        default=0.05,
        help="Fraction of GOOD model curves corrupted into almost-realistic but invalid shapes."
    )

    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    rng = np.random.default_rng(args.seed)

    # time grid
    time_points = np.arange(0, args.max_time + args.time_step/2, args.time_step)
    n_t = len(time_points)

    rows = []
    curve_id = 0
    test_id_prefix = args.file_stem[-2:]

    for model_name, (fn, params) in MODEL_SPECS.items():
        for rep in range(args.n_reps):
            curve_id += 1

            # sample model parameters
            if model_name == "Flat":
                pars = {"baseline": rng.uniform(0.0, 0.3)}
                y = fn(time_points, **pars)
            elif model_name == "Diauxic":
                pars = {
                    "A1": rng.uniform(0.3, 1.0),
                    "mu1": rng.uniform(0.2, 1.2),
                    "lam1": rng.uniform(0.0, 5.0),
                    "A2": rng.uniform(0.3, 1.0),
                    "mu2": rng.uniform(0.2, 1.2),
                    "lam2": rng.uniform(5.0, 15.0),
                }
                y = fn(time_points, **pars)
            else:
                pars = {"A": rng.uniform(0.5, 2.0), "mu": rng.uniform(0.2, 1.5), "lam": rng.uniform(0.0, 10.0)}
                if model_name == "ModifiedGompertz":
                    pars.update({"alpha": rng.uniform(0.0, 0.3), "tshift": rng.uniform(5.0, 12.0)})
                if model_name == "Richards":
                    pars.update({"nu": rng.uniform(0.5, 2.0)})
                y = fn(time_points, **pars)

                        # --- NEW: decide baseline validity from model type ---
            is_valid_base = model_name in GOOD_MODELS

            # --- NEW: realistic valid curves (low noise, no missing/outliers) ---
            high_quality = False
            noise_std = args.noise_level
            if is_valid_base and rng.random() < args.pct_high_quality_valid:
                high_quality = True
                noise_std = args.noise_level * 0.3  # much lower noise

            # add noise / clamp
            y = (y + rng.normal(0, noise_std, size=y.shape)).clip(min=0.0)

            # generic missing/outlier corruption only if NOT high-quality
            if not high_quality:
                if rng.random() < args.pct_missing_curves:
                    y = inject_missing(pd.Series(y), args.missing_frac_per_curve, rng).values
                if rng.random() < args.pct_outlier_curves:
                    y = inject_negative_outliers(
                        pd.Series(y), args.outlier_frac_per_curve,
                        args.outlier_scale_min, args.outlier_scale_max, rng
                    ).values

            # --- NEW: synthetic invalid curves on top of GOOD models ---
            is_valid = is_valid_base
            corruption_tag = None

            if is_valid_base:
                r = rng.random()
                if r < args.pct_obvious_invalid_curves:
                    y = make_obvious_invalid(y, rng)
                    is_valid = False
                    corruption_tag = "Invalid_Obvious"
                elif r < args.pct_obvious_invalid_curves + args.pct_subtle_invalid_curves:
                    y = make_subtle_invalid(y, rng)
                    is_valid = False
                    corruption_tag = "Invalid_Subtle"
                elif r < (
                    args.pct_obvious_invalid_curves
                    + args.pct_subtle_invalid_curves
                    + args.pct_nearreal_invalid_curves
                ):
                    y = make_near_real_invalid(y, rng)
                    is_valid = False
                    corruption_tag = "Invalid_NearReal"

            # For non-GOOD models (Flat, Diauxic) keep them invalid
            if not is_valid_base:
                is_valid = False

            # build row
            model_label = model_name
            if high_quality and is_valid:
                model_label = f"{model_label}_HQ"
            if corruption_tag is not None:
                model_label = f"{model_label}_{corruption_tag}"

            base = {
                "FileName": args.file_stem,
                "Test Id": f"{test_id_prefix}_{curve_id}",
                "Model Name": model_label,
                "Is_Valid": bool(is_valid),
            }

            # time columns as OD values
            for i, t in enumerate(time_points, start=1):
                base[f"T{np.round(t, 6)} ({args.time_unit})"] = float(y[i-1])
            rows.append(base)

    df_wide = pd.DataFrame(rows)

    # output
    os.makedirs(args.output_dir, exist_ok=True)
    wide_path = os.path.join(args.output_dir, f"timeseries_wide_{args.file_stem}.csv")
    df_wide.to_csv(wide_path, index=False)

    # run-info stats (akin to long form, but derived from wide)
    stats = {
        "points_per_curve": int(n_t),
        "n_curves": int(len(df_wide)),
        "n_rows": int(len(df_wide)),  # one row per curve
    }
    write_run_info_xlsx(args.output_dir, args.file_stem, args, wide_path, stats)

    logging.info(f"Wrote wide CSV to {wide_path}")
    logging.info(f"Wrote run_info.xlsx to {os.path.join(args.output_dir, 'run_info.xlsx')}")

if __name__ == "__main__":
    main()
