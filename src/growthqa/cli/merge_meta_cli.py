# src/growthqa/cli/merge_meta_cli.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from growthqa.pipelines.build_meta_dataset import run_merge_preprocess_meta

def add_merge_meta_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """
    Mirrors the original merge_meta.py CLI:
      python merge_meta.py --step 0.25 --auto-tmax --auto-tmax-coverage 0.8 ...
    but routed through the modular pipeline function run_merge_preprocess_meta().
    """
    p = subparsers.add_parser(
        "merge-meta",
        help="Merge already-converted wide growth curves, preprocess, and build rich meta-features.",
    )

    # positional: inputs
    p.add_argument("inputs", nargs="+", help="Input CSV(s), already converted to wide format.")

    # interpolation / grid
    p.add_argument("--step", type=float, default=0.25, help="Common grid step (hours). Default 0.25.")
    p.add_argument("--min-points", type=int, default=3, help="Min finite points required to interpolate. Default 3.")
    p.add_argument(
        "--low-res-threshold",
        type=int,
        default=7,
        help="Curves with min_points..(low_res_threshold-1) points flagged low_resolution. Default 7.",
    )
    p.add_argument("--tmax-hours", type=float, default=None, help="Optional cap on grid max time (hours).")
    p.add_argument(
        "--auto-tmax",
        action="store_true",
        default=True,
        help="If set, choose tmax so >=coverage fraction of curves reach it.",
    )
    p.add_argument(
        "--auto-tmax-coverage",
        type=float,
        default=0.8,
        help="Coverage for auto-tmax. Default 0.8.",
    )

    # blank subtraction / baseline
    p.add_argument(
        "--blank-subtracted",
        action="store_true",
        default=False,
        help="Apply blank/baseline subtraction (recommended if raw data).",
    )
    p.add_argument(
        "--clip-negatives",
        action="store_true",
        default=False,
        help="After blank subtraction, clip negative OD values to 0.",
    )
    p.add_argument(
        "--global-blank",
        type=float,
        default=None,
        help="If set, subtract this constant blank OD. Otherwise subtract robust early baseline.",
    )
    p.add_argument(
        "--blank-status-csv",
        type=str,
        default=None,
        help="CSV mapping FileName -> already_blank_subtracted (RAW/ALREADY) and optional blank_value.",
    )
    p.add_argument(
        "--blank-default",
        type=str,
        default="RAW",
        choices=["RAW", "ALREADY"],
        help="Default blank status if FileName not in blank-status-csv.",
    )

    # smoothing
    p.add_argument(
        "--smooth-method",
        type=str,
        default="SGF",
        choices=["NONE", "RAW", "LWS", "SGF"],
        help="Smoothing method: NONE, RAW (rolling), LWS (LOWESS), SGF (Savitzky–Golay).",
    )
    p.add_argument("--smooth-window", type=int, default=5, help="Window size (points) for smoothing. Default 5.")

    # normalization
    p.add_argument(
        "--normalize",
        type=str,
        default="MINMAX",
        choices=["NONE", "MAX", "MINMAX"],
        help="Within-curve normalization. Default NONE.",
    )

    # outputs (required like original script)
    p.add_argument("--out-raw", required=True, help="Output raw_merged.csv")
    p.add_argument("--out-final", required=True, help="Output final_merged.csv")
    p.add_argument("--out-meta", required=True, help="Output meta.csv")

    # logging
    p.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )

    p.set_defaults(_fn=_run_merge_meta)


def _run_merge_meta(args: argparse.Namespace) -> int:
    # ensure output dirs exist
    Path(args.out_raw).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_final).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_meta).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=getattr(logging, args.loglevel), format="%(levelname)s: %(message)s")

    run_merge_preprocess_meta(
        inputs=args.inputs,
        out_raw=args.out_raw,
        out_final=args.out_final,
        out_meta=args.out_meta,
        step=float(args.step),
        min_points=int(args.min_points),
        low_res_threshold=int(args.low_res_threshold),
        tmax_hours=args.tmax_hours,
        auto_tmax=bool(args.auto_tmax),
        auto_tmax_coverage=float(args.auto_tmax_coverage),
        blank_subtracted=bool(args.blank_subtracted),
        clip_negatives=bool(args.clip_negatives),
        global_blank=args.global_blank,
        blank_status_csv=args.blank_status_csv,
        blank_default=str(args.blank_default).upper(),
        smooth_method=str(args.smooth_method).upper(),
        smooth_window=int(args.smooth_window),
        normalize=str(args.normalize).upper(),
        loglevel=str(args.loglevel).upper(),
    )
    return 0
