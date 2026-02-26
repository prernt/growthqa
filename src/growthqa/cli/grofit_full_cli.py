from __future__ import annotations

import argparse
from pathlib import Path

from growthqa.grofit.full_pipeline import run_full_input_to_grofit, FullGrofitPipelineConfig


def main():
    ap = argparse.ArgumentParser(
        description="End-to-end pipeline: input -> growthqa classifier -> tidy -> grofit -> outputs"
    )
    ap.add_argument("--input", required=True, help="Input CSV/XLSX (either time+wells OR wide-per-curve)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--model", required=True, help="Path to saved joblib model pipeline")
    ap.add_argument("--file-test-id", default=None, help="Optional explicit test_id prefix (e.g., testSample1)")

    # grofit knobs (common)
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
    ap.add_argument("--dr_s", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    

    args = ap.parse_args()

    cfg = FullGrofitPipelineConfig(
        response_var=args.response,
        have_atleast=args.have_atleast,
        gc_boot_B=args.gc_boot,
        dr_boot_B=args.dr_boot,
        spline_auto_cv=True,
        spline_s=None,
        dr_x_transform=None if args.dr_x_transform=="none" else args.dr_x_transform,
        dr_y_transform=None if args.dr_y_transform=="none" else args.dr_y_transform,
        smooth_gc=args.smooth_gc,
        smooth_dr=args.smooth_dr,
        dr_s=args.dr_s,
        random_state=args.seed,
    )

    out = run_full_input_to_grofit(
        input_path=Path(args.input),
        outdir=Path(args.outdir),
        model_joblib_path=Path(args.model),
        file_test_id=args.file_test_id,
        cfg=cfg,
    )

    print("Wrote outputs:")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
