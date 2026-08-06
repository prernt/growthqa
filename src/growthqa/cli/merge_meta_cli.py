from __future__ import annotations
import argparse
import json
from growthqa.pipelines.build_meta_dataset import build_training_meta


def add_merge_meta_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "build-train-meta",
        help="Build final_merged.csv and training_meta.csv (plus raw_merged.csv if --lab is given) for classifier training.",
    )
    p.add_argument("--synthetic", required=True, help="Synthetic wide CSV (e.g. timeseries_wide_SD1.csv).")
    p.add_argument("--lab", required=False, default=None, help="Optional laboratory wide CSV (e.g. lab_14.75h_0.25.csv).")
    p.add_argument("--out-dir", required=True, help="Directory for the output files.")
    p.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    info = build_training_meta(
        synthetic_csv=args.synthetic,
        lab_csv=args.lab,
        out_dir=args.out_dir,
    )
    print(json.dumps(info, indent=2))
    return 0