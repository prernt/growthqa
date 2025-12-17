# src/growthqa/cli/convert_cli.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from growthqa.io.wide_format import long_to_wide_preserve_times, parse_any_file_to_long


def add_convert_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """
    CLI wrapper around the converter (file_convertor.py logic).

    Mirrors the original script behavior:
      - Convert 1..N files to wide CSV
      - Keep original timepoints (no resampling)
      - Convert time to hours
      - Output: wide_like_synthetic__<stem>.csv
    """
    p = subparsers.add_parser(
        "convert",
        help="Convert lab csv/xlsx into canonical wide CSV(s) with Txx.xx (h) columns.",
    )

    p.add_argument("inputs", nargs="+", help="Input file(s): .csv, .xlsx")
    p.add_argument("--outdir", required=True, help="Output directory for converted wide CSVs")

    p.add_argument(
        "--prefix-testid",
        action="store_true",
        default=False,
        help="Prefix Test Id with per-file tag (recommended to avoid collisions later).",
    )
    p.add_argument(
        "--file-tag-mode",
        default="stem2",
        choices=["stem2", "stem", "none"],
        help="How to build per-file tag used for Test Id prefix. stem2=last 2 chars of stem.",
    )

    p.set_defaults(_fn=_run_convert)


def _run_convert(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for inp in args.inputs:
        p = Path(inp)
        stem = p.stem

        if args.file_tag_mode == "stem2":
            tag = stem[-2:] if len(stem) >= 2 else stem
        elif args.file_tag_mode == "stem":
            tag = stem
        else:
            tag = ""

        # file -> standardized long
        long_df = parse_any_file_to_long(str(p))

        # long -> wide (preserve original timepoints)
        wide_df = long_to_wide_preserve_times(
            long_df,
            file_tag=tag,
            add_prefix=bool(args.prefix_testid) and (tag != ""),
        )

        out_path = outdir / f"wide_like_synthetic__{stem}.csv"
        wide_df.to_csv(out_path, index=False)

        n_time_cols = max(0, wide_df.shape[1] - 4)  # FileName/Test Id/Model Name/Is_Valid = 4
        print(f"[OK] {inp} -> {out_path}  (rows={len(wide_df)}, time_cols={n_time_cols})")

    return 0
