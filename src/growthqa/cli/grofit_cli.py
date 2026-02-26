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
    ap.add_argument("--dr_x_transform", default="log10", choices=["none","log1p","log10","log"])
    ap.add_argument("--dr_y_transform", default="none", choices=["none","log1p"],
                    help="DR y transform. 'log1p' matches Grofit log.y.dr (ln(y+1)).")
    ap.add_argument("--smooth_gc", type=float, default=None,
                    help="Grofit-like smooth.gc spar in (0,1]. None => auto/CV. WARNING: changes mu/EC50.")
    ap.add_argument("--smooth_dr", type=float, default=None,
                    help="Grofit-like smooth.dr spar in (0,1]. None => auto/CV. WARNING: changes EC50.")


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
        smooth_gc=args.smooth_gc,
        smooth_dr=args.smooth_dr,
        dr_x_transform=None if args.dr_x_transform=="none" else args.dr_x_transform,
        dr_y_transform=None if args.dr_y_transform=="none" else args.dr_y_transform,
        export_dir=Path(args.outdir),
    )
    _ = res.get("zip_path")

if __name__ == "__main__":
    main()
