# src/growthqa/cli/grofit_cli.py
from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path

from growthqa.grofit.pipeline import run_grofit_pipeline

def main():
    ap = argparse.ArgumentParser(description="Run grofit-like pipeline in Python (growthqa).")
    ap.add_argument("--curves", required=True, help="Input tidy CSV with columns: test_id,curve_id,concentration,time,y,is_valid")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--response", default="mu", choices=["A", "mu", "lag", "integral"])
    ap.add_argument("--have-atleast", type=int, default=6)
    ap.add_argument("--gc-boot", type=int, default=200)
    ap.add_argument("--dr-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    curves_df = pd.read_csv(args.curves)
    res = run_grofit_pipeline(
        curves_df=curves_df,
        response_var=args.response,
        have_atleast=args.have_atleast,
        gc_boot_B=args.gc_boot,
        dr_boot_B=args.dr_boot,
        random_state=args.seed,
        export_dir=Path(args.outdir),
    )
    _ = res.get("zip_path")

if __name__ == "__main__":
    main()
