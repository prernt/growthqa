# src/growthqa/cli/main.py
import argparse

from growthqa.cli.merge_meta_cli import add_merge_meta_subcommand
from growthqa.cli.synth_cli import add_synth_subcommand


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="growthqa", description="GrowthQA: synthetic data generation + training metadata")
    sub = parser.add_subparsers(dest="command", required=True)

    add_synth_subcommand(sub)
    add_merge_meta_subcommand(sub)

    args = parser.parse_args(argv)
    return args._fn(args)  # each subcommand sets a handler


if __name__ == "__main__":
    raise SystemExit(main())
