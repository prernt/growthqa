# src/growthqa/cli/synth_cli.py
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def add_synth_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "synth",
        help="Generate synthetic wide growth curves (wraps timeseries_curve_data.py).",
    )

    # Match timeseries_curve_data.py args 1:1
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-reps", type=int, default=10, help="Replicates per model")
    p.add_argument("--max-time", type=float, default=16, help="Max time")
    p.add_argument("--time-step", type=float, default=0.25, help="Time step")
    p.add_argument("--time-unit", type=str, default="h", choices=["s", "m", "h"], help="Unit label for header")

    p.add_argument("--noise-level", type=float, default=0.05, help="Gaussian noise stdev")
    p.add_argument("--pct-missing-curves", type=float, default=0.1, help="Fraction of curves with missing values")
    p.add_argument("--missing-frac-per-curve", type=float, default=0.1, help="Fraction of points set to NaN in selected curves")
    p.add_argument("--pct-outlier-curves", type=float, default=0.05, help="Fraction of curves with negative outliers")
    p.add_argument("--outlier-frac-per-curve", type=float, default=0.05, help="Fraction of points made outliers in selected curves")
    p.add_argument("--outlier-scale-min", type=float, default=0.1)
    p.add_argument("--outlier-scale-max", type=float, default=0.3)

    p.add_argument("--output-dir", type=str, default="./dataNew")
    p.add_argument("--file-stem", type=str, default="timedata")

    # NEW invalid/quality controls already present in your script
    p.add_argument(
        "--pct-high-quality-valid",
        type=float,
        default=0.3,
        help="Fraction of GOOD model curves made very realistic (low noise, no missing/outliers).",
    )
    p.add_argument(
        "--pct-obvious-invalid-curves",
        type=float,
        default=0.1,
        help="Fraction of GOOD model curves corrupted into obviously invalid shapes.",
    )
    p.add_argument(
        "--pct-subtle-invalid-curves",
        type=float,
        default=0.1,
        help="Fraction of GOOD model curves corrupted into subtle invalid shapes.",
    )
    p.add_argument(
        "--pct-nearreal-invalid-curves",
        type=float,
        default=0.05,
        help="Fraction of GOOD model curves corrupted into near-realistic but invalid shapes.",
    )

    # Script location override (important once you move the script under src/)
    p.add_argument(
        "--script-path",
        type=str,
        default=None,
        help="Optional path to timeseries_curve_data.py. If omitted, uses the packaged location.",
    )

    p.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    # Prefer packaged script under src/growthqa/synthetic/
    if args.script_path:
        script_path = Path(args.script_path).expanduser().resolve()
    else:
        # Expected final location after you move the script:
        # src/growthqa/synthetic/timeseries_curve_data.py
        script_path = (Path(__file__).resolve().parents[1] / "synthetic" / "timeseries_curve_data.py").resolve()

    if not script_path.exists():
        raise FileNotFoundError(
            f"timeseries_curve_data.py not found at: {script_path}\n"
            "Fix by either:\n"
            "  (1) moving it to src/growthqa/synthetic/timeseries_curve_data.py, or\n"
            "  (2) running with --script-path /path/to/timeseries_curve_data.py"
        )

    cmd = [
        sys.executable, str(script_path),

        "--seed", str(args.seed),
        "--n-reps", str(args.n_reps),
        "--max-time", str(args.max_time),
        "--time-step", str(args.time_step),
        "--time-unit", str(args.time_unit),

        "--noise-level", str(args.noise_level),
        "--pct-missing-curves", str(args.pct_missing_curves),
        "--missing-frac-per-curve", str(args.missing_frac_per_curve),
        "--pct-outlier-curves", str(args.pct_outlier_curves),
        "--outlier-frac-per-curve", str(args.outlier_frac_per_curve),
        "--outlier-scale-min", str(args.outlier_scale_min),
        "--outlier-scale-max", str(args.outlier_scale_max),

        "--output-dir", str(args.output_dir),
        "--file-stem", str(args.file_stem),

        "--pct-high-quality-valid", str(args.pct_high_quality_valid),
        "--pct-obvious-invalid-curves", str(args.pct_obvious_invalid_curves),
        "--pct-subtle-invalid-curves", str(args.pct_subtle_invalid_curves),
        "--pct-nearreal-invalid-curves", str(args.pct_nearreal_invalid_curves),
    ]

    # run
    subprocess.check_call(cmd)
    return 0
