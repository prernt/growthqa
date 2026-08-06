# src/growthqa/cli/synth_cli.py
from __future__ import annotations

import argparse
import logging
import os

import growthqa.synthetic.timeseries_curve_data as timeseries_curve_data

def add_synth_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "synth",
        help="Generate the synthetic wide growth-curve dataset.",
    )
    p.add_argument("--seed", type=int, default=123, help="Random seed.")
    p.add_argument("--max-time", type=float, default=16.0, help="Maximum time in hours.")
    p.add_argument("--time-step", type=float, default=0.5, help="Sampling interval in hours.")
    p.add_argument("--noise-level", type=float, default=0.05, help="Gaussian measurement-noise stdev.")
    p.add_argument("--output-dir", type=str, default="./gen_data", help="Output directory.")
    p.add_argument("--file-stem", type=str, default="syn", help="File stem for the output CSV.")
    p.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df, summary = timeseries_curve_data.generate_synthetic_wide_df(
        tmax_hours=args.max_time,
        step_hours=args.time_step,
        seed=args.seed,
        noise_level=args.noise_level,
        file_stem=args.file_stem,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    wide_path = os.path.join(args.output_dir, f"timeseries_wide_{args.file_stem}.csv")
    df.to_csv(wide_path, index=False)
    timeseries_curve_data.write_run_info_xlsx(args.output_dir, wide_path, summary)

    logging.info(
        "Wrote %d curves (%d valid, %d invalid) to %s",
        summary["n_curves"], summary["n_valid"], summary["n_invalid"], wide_path,
    )
    return 0