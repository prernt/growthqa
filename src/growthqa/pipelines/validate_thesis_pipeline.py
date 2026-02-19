from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from growthqa.classifier.train_from_meta import _group_split, build_model_matrix, detect_label_col
from growthqa.pipelines.build_meta_dataset import run_merge_preprocess_meta
from growthqa.preprocess.timegrid import parse_time_from_header


def _time_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if parse_time_from_header(str(c)) is not None]


def run_validation(inputs: list[str]) -> None:
    with tempfile.TemporaryDirectory() as td:
        out_raw = Path(td) / "raw.csv"
        out_final = Path(td) / "final.csv"
        out_meta = Path(td) / "meta.csv"

        raw, final, meta = run_merge_preprocess_meta(
            inputs=inputs,
            out_raw=str(out_raw),
            out_final=str(out_final),
            out_meta=str(out_meta),
            step=0.5,
            tmax_hours=16.0,
            auto_tmax=False,
            augment_trunc=True,
            trunc_horizons=[8.0, 10.0, 12.0, 14.75, 16.0],
            trunc_per_curve=3,
            trunc_seed=123,
        )

        assert out_meta.exists(), "meta.csv was not created"
        required = [
            "FileName",
            "Test Id",
            "Is_Valid",
            "base_curve_id",
            "aug_id",
            "source_type",
            "train_horizon",
            "observed_tmax",
            "is_censored",
            "n_points_observed",
            "max_gap_hours",
            "median_dt_hours",
            "missing_frac_on_grid",
            "low_resolution",
            "too_sparse",
            "initial_OD",
            "final_OD",
            "max_OD",
            "min_OD",
            "range_OD",
            "auc",
            "auc_per_hour",
            "net_change_per_hour",
        ]
        missing = [c for c in required if c not in meta.columns]
        assert not missing, f"meta.csv missing required columns: {missing}"

        tcols_raw = _time_cols(raw)
        tvals_raw = np.array([parse_time_from_header(c) for c in tcols_raw], dtype=float)
        h10 = raw[pd.to_numeric(raw["train_horizon"], errors="coerce") == 10.0]
        if not h10.empty:
            mask = tvals_raw > 10.0
            assert h10.loc[:, np.array(tcols_raw)[mask]].isna().all().all(), "Rows with train_horizon=10 have non-NaN points >10h"

        tcols_final = _time_cols(final)
        tvals_final = np.array([parse_time_from_header(c) for c in tcols_final], dtype=float)
        final_indexed = final.set_index("aug_id", drop=False)
        for _, r in meta.head(200).iterrows():
            aid = r.get("aug_id")
            if aid not in final_indexed.index:
                continue
            fr = final_indexed.loc[aid]
            if isinstance(fr, pd.DataFrame):
                fr = fr.iloc[0]
            y = pd.to_numeric(fr[tcols_final], errors="coerce").to_numpy(dtype=float)
            obs = np.isfinite(y) & np.isfinite(tvals_final)
            if not np.any(obs):
                continue
            t = tvals_final[obs]
            od = y[obs]
            tmax = float(np.max(t))
            od_final = float(od[np.argmax(t)])
            auc = float(np.trapezoid(od, t)) if hasattr(np, "trapezoid") else float(np.trapz(od, t))
            assert np.isclose(float(r["observed_tmax"]), tmax, atol=1e-6, equal_nan=True), "observed_tmax mismatch"
            assert np.isclose(float(r["final_OD"]), od_final, atol=1e-6, equal_nan=True), "final_OD not computed at observed_tmax"
            assert np.isclose(float(r["auc"]), auc, atol=1e-6, equal_nan=True), "auc not integrated on observed range"

        label_col = detect_label_col(meta)
        X, y, groups, _, _ = build_model_matrix(meta, label_col)
        tr, va, te = _group_split(X, y, groups)
        g_tr = set(groups.iloc[tr].tolist())
        g_va = set(groups.iloc[va].tolist())
        g_te = set(groups.iloc[te].tolist())
        assert not (g_tr & g_te), "base_curve_id leakage between train and test"
        assert not (g_va & g_te), "base_curve_id leakage between val and test"

        print("Validation passed.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="Wide input CSVs")
    args = p.parse_args()
    run_validation(args.inputs)


if __name__ == "__main__":
    main()

