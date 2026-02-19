# src/growthqa/grofit/pipeline.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Literal
from pathlib import Path

from .gc_fit_model import gc_fit_model
from .gc_fit_spline import gc_fit_spline
from .gc_boot_spline import gc_boot_spline, BootstrapMethod
from .dr_fit_spline import dr_fit_spline
from .dr_fit_model import dr_fit_model
from .dr_boot_spline import dr_boot_spline
from .interactive import apply_user_exclusion, UserFilterFn
from .export import export_results_zip

ResponseVar = Literal["A", "mu", "lag", "integral"]
FitOpt = Literal["m", "s", "b"]
DrFitMethod = Literal["auto", "spline", "4pl"]


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _response_col_map(response_var: ResponseVar) -> Dict[str, str]:
    if response_var == "mu":
        return {"param": "mu.model", "spline": "mu.spline"}
    if response_var == "lag":
        return {"param": "lambda.model", "spline": "lambda.spline"}
    if response_var == "A":
        return {"param": "A.para", "spline": "A.nonpara"}
    return {"param": "Integral.model", "spline": "integral.spline"}


def run_grofit_pipeline(
    curves_df: pd.DataFrame,
    response_var: ResponseVar = "mu",
    have_atleast: int = 6,
    gc_boot_B: int = 200,
    dr_boot_B: int = 300,
    spline_auto_cv: bool = True,
    spline_s: Optional[float] = None,
    dr_x_transform: Optional[str] = "log1p",
    dr_s: Optional[float] = None,
    random_state: Optional[int] = 42,
    fit_opt: FitOpt = "b",
    bootstrap_method: BootstrapMethod = "pairs",
    validity_col: str = "is_valid_final",
    dr_fit_method: DrFitMethod = "auto",
    user_filter_fn: Optional[UserFilterFn] = None,
    export_dir: Optional[str | Path] = None,
    export_zip_name: str = "grofit_outputs.zip",
) -> Dict[str, Any]:
    """
    Fully automatic grofit-like pipeline.

    Returns:
      - gc_fit: curve-level results (gcFit.csv schema)
      - dr_fit: dose-response results (drFit.csv schema)
      - gc_boot: curve-level bootstrap (gcBoot.csv schema) or empty
      - dr_boot: dose-response bootstrap (drBoot.csv schema) or empty
      - zip_bytes/zip_path if export_dir is provided
    """
    _ensure_columns(curves_df, ["test_id", "curve_id", "concentration", "time", "y"])

    df = curves_df.copy()
    if validity_col in df.columns:
        valid_series = df[validity_col]
        valid_values = pd.Series(valid_series, index=df.index).fillna(False).astype(bool).to_numpy()
    elif "is_valid" in df.columns:
        valid_series = df["is_valid"]
        valid_values = pd.Series(valid_series, index=df.index).fillna(False).astype(bool).to_numpy()
    else:
        # Manual mode can intentionally pass a non-existent validity column to include all curves.
        valid_values = np.ones(len(df), dtype=bool)
    df["is_valid_final"] = valid_values

    curve_index = (
        df.groupby(["test_id", "curve_id", "concentration"], dropna=False)["is_valid_final"]
        .first()
        .reset_index()
    )

    gc_fit_rows = []
    gc_boot_rows = []

    for _, row in curve_index.iterrows():
        test_id = row["test_id"]
        curve_id = row["curve_id"]
        conc = row["concentration"]
        is_valid_final = bool(row["is_valid_final"])

        g = df[(df["test_id"] == test_id) & (df["curve_id"] == curve_id) & (df["concentration"] == conc)]
        t = g["time"].to_numpy()
        y = g["y"].to_numpy()

        pfit = None
        sfit = None
        boot = None

        if is_valid_final:
            if fit_opt in {"m", "b"}:
                pfit = gc_fit_model(t, y)
            if fit_opt in {"s", "b"}:
                sfit = gc_fit_spline(t, y, s=spline_s, auto_cv=spline_auto_cv)
            if gc_boot_B > 0 and fit_opt in {"s", "b"}:
                boot = gc_boot_spline(
                    t,
                    y,
                    B=gc_boot_B,
                    ci=0.95,
                    random_state=None if random_state is None else (random_state + int(hash(curve_id) % 10000)),
                    spline_s=spline_s,
                    auto_cv=spline_auto_cv,
                    bootstrap_method=bootstrap_method,
                )

        def _val_or_nan(fit, attr):
            if fit is None or not fit.success:
                return np.nan
            return getattr(fit, attr)

        gc_fit_rows.append(
            {
                "test.id": test_id,
                "add.id": curve_id,
                "concentration": conc,
                "reliability": bool(is_valid_final),
                "use.model": "" if (pfit is None or not pfit.success) else (pfit.model or ""),
                "log.x": 0,
                "log.y": 0,
                "nboot.fit": int(gc_boot_B) if (gc_boot_B > 0 and is_valid_final and fit_opt in {"s", "b"}) else 0,
                "mu.model": _val_or_nan(pfit, "mu"),
                "lambda.model": _val_or_nan(pfit, "lag"),
                "A.para": _val_or_nan(pfit, "A"),
                "Integral.model": _val_or_nan(pfit, "integral"),
                "mu.spline": _val_or_nan(sfit, "mu"),
                "lambda.spline": _val_or_nan(sfit, "lag"),
                "A.nonpara": _val_or_nan(sfit, "A"),
                "integral.spline": _val_or_nan(sfit, "integral"),
            }
        )

        if gc_boot_B > 0:
            def _boot_val(key: str, stat: str) -> float:
                if not boot or not boot.get("success"):
                    return np.nan
                return float(boot[key].get(stat, np.nan))

            gc_boot_rows.append(
                {
                    "test.id": test_id,
                    "add.id": curve_id,
                    "concentration": conc,
                    "mu.bt": _boot_val("mu", "mean"),
                    "sdmu.bt": _boot_val("mu", "sd"),
                    "ci95.mu.bt.lo": _boot_val("mu", "lo"),
                    "ci95.mu.bt.up": _boot_val("mu", "hi"),
                    "lambda.bt": _boot_val("lag", "mean"),
                    "sdlambda.bt": _boot_val("lag", "sd"),
                    "ci95.lambda.bt.lo": _boot_val("lag", "lo"),
                    "ci95.lambda.bt.up": _boot_val("lag", "hi"),
                    "A.bt": _boot_val("A", "mean"),
                    "sdA.bt": _boot_val("A", "sd"),
                    "ci95.A.bt.lo": _boot_val("A", "lo"),
                    "ci95.A.bt.up": _boot_val("A", "hi"),
                    "integral.bt": _boot_val("integral", "mean"),
                    "sdIntegral.bt": _boot_val("integral", "sd"),
                    "ci95.integral.bt.lo": _boot_val("integral", "lo"),
                    "ci95.integral.bt.up": _boot_val("integral", "hi"),
                }
            )

    gc_fit_cols = [
        "test.id",
        "add.id",
        "concentration",
        "reliability",
        "use.model",
        "log.x",
        "log.y",
        "nboot.fit",
        "mu.model",
        "lambda.model",
        "A.para",
        "Integral.model",
        "mu.spline",
        "lambda.spline",
        "A.nonpara",
        "integral.spline",
    ]
    gc_fit = pd.DataFrame(gc_fit_rows, columns=gc_fit_cols)

    gc_boot_cols = [
        "test.id",
        "add.id",
        "concentration",
        "mu.bt",
        "sdmu.bt",
        "ci95.mu.bt.lo",
        "ci95.mu.bt.up",
        "lambda.bt",
        "sdlambda.bt",
        "ci95.lambda.bt.lo",
        "ci95.lambda.bt.up",
        "A.bt",
        "sdA.bt",
        "ci95.A.bt.lo",
        "ci95.A.bt.up",
        "integral.bt",
        "sdIntegral.bt",
        "ci95.integral.bt.lo",
        "ci95.integral.bt.up",
    ]
    gc_boot = pd.DataFrame(gc_boot_rows, columns=gc_boot_cols)

    # Apply interactive/user exclusion before dose-response
    dr_source = gc_fit.copy()
    dr_source = dr_source[dr_source["reliability"] == True].copy()
    dr_source = apply_user_exclusion(dr_source, user_filter_fn)

    resp_cols = _response_col_map(response_var)
    preferred_cols = []
    if fit_opt == "m":
        preferred_cols = [resp_cols["param"]]
    elif fit_opt == "s":
        preferred_cols = [resp_cols["spline"]]
    else:
        preferred_cols = [resp_cols["spline"], resp_cols["param"]]

    dr_rows = []
    dr_boot_rows = []
    dr_audit_rows = []
    log_x = 1 if dr_x_transform in {"log1p", "log"} else 0

    all_test_ids = curve_index["test_id"].drop_duplicates().tolist()
    for test_id in all_test_ids:
        g = dr_source[dr_source["test.id"] == test_id].copy()
        if g.empty:
            dr_rows.append(
                {
                    "name": test_id,
                    "log.x": log_x,
                    "log.y": 0,
                    "Samples": int(dr_boot_B) if dr_boot_B > 0 else 0,
                    "EC50": np.nan,
                    "yEC50": np.nan,
                    "EC50.orig": np.nan,
                    "yEC50.orig": np.nan,
                }
            )
            continue
        resp_col = None
        for cand in preferred_cols:
            if cand in g.columns:
                cand_num = pd.to_numeric(g[cand], errors="coerce")
                if np.isfinite(cand_num.to_numpy(dtype=float)).any():
                    resp_col = cand
                    break

        if resp_col is None:
            dr_rows.append(
                {
                    "name": test_id,
                    "log.x": log_x,
                    "log.y": 0,
                    "Samples": int(dr_boot_B) if dr_boot_B > 0 else 0,
                    "EC50": np.nan,
                    "yEC50": np.nan,
                    "EC50.orig": np.nan,
                    "yEC50.orig": np.nan,
                }
            )
            continue

        resp_num = pd.to_numeric(g[resp_col], errors="coerce")
        gg = g[np.isfinite(resp_num.to_numpy(dtype=float))].copy()
        gg[resp_col] = pd.to_numeric(gg[resp_col], errors="coerce")
        n_points = int(gg["concentration"].nunique())

        if n_points < int(have_atleast):
            dr_rows.append(
                {
                    "name": test_id,
                    "log.x": log_x,
                    "log.y": 0,
                    "Samples": int(dr_boot_B) if dr_boot_B > 0 else 0,
                    "EC50": np.nan,
                    "yEC50": np.nan,
                    "EC50.orig": np.nan,
                    "yEC50.orig": np.nan,
                }
            )
            continue

        conc = gg["concentration"].to_numpy(dtype=float)
        resp = gg[resp_col].to_numpy(dtype=float)

        spline_fit = dr_fit_spline(
            conc,
            resp,
            x_transform=dr_x_transform,
            s=dr_s,
            auto_cv=(dr_s is None),
            enforce_monotonic=True,
            fallback_to_4pl=(dr_fit_method != "spline"),
        )
        model_fit = dr_fit_model(conc, resp) if dr_fit_method in {"auto", "4pl"} else {"success": False}

        chosen = spline_fit
        chosen_name = str(spline_fit.get("method", "spline"))
        if dr_fit_method == "4pl":
            chosen = model_fit
            chosen_name = "4pl"
        elif dr_fit_method == "auto":
            spline_ok = bool(spline_fit.get("success"))
            model_ok = bool(model_fit.get("success"))
            spline_aic = pd.to_numeric(pd.Series([spline_fit.get("aic", np.nan)]), errors="coerce").iloc[0]
            model_aic = pd.to_numeric(pd.Series([model_fit.get("aic", np.nan)]), errors="coerce").iloc[0]
            if model_ok and (not spline_ok):
                chosen = model_fit
                chosen_name = "4pl"
            elif spline_ok and (not model_ok):
                chosen = spline_fit
                chosen_name = str(spline_fit.get("method", "spline"))
            elif model_ok and spline_ok:
                if np.isfinite(model_aic) and np.isfinite(spline_aic) and float(model_aic) < float(spline_aic):
                    chosen = model_fit
                    chosen_name = "4pl"
                else:
                    chosen = spline_fit
                    chosen_name = str(spline_fit.get("method", "spline"))
        ec50 = chosen.get("ec50", np.nan)
        y_ec50 = chosen.get("y_ec50", np.nan)

        dr_rows.append(
            {
                "name": test_id,
                "log.x": log_x,
                "log.y": 0,
                "Samples": int(dr_boot_B) if dr_boot_B > 0 else 0,
                "EC50": ec50,
                "yEC50": y_ec50,
                "EC50.orig": ec50,
                "yEC50.orig": y_ec50,
            }
        )
        dr_audit_rows.append(
            {
                "name": test_id,
                "dr_model": chosen_name,
                "dr_monotonic": bool(chosen.get("dr_monotonic", True)),
                "ec50_status": chosen.get("ec50_status", "OK"),
                "aic_spline": spline_fit.get("aic", np.nan),
                "aic_4pl": model_fit.get("aic", np.nan),
            }
        )

        min_boot_points = max(6, int(have_atleast))
        if dr_boot_B > 0 and n_points >= min_boot_points:
            dr_boot = dr_boot_spline(
                conc,
                resp,
                B=dr_boot_B,
                ci=0.95,
                random_state=None if random_state is None else (random_state + int(hash(test_id) % 10000)),
                x_transform=dr_x_transform,
                s=dr_s,
            )
            if dr_boot.get("success"):
                dr_boot_rows.append(
                    {
                        "name": test_id,
                        "meanEC50": dr_boot.get("ec50_mean", np.nan),
                        "sdEC50": dr_boot.get("ec50_sd", np.nan),
                        "ci95EC50.lo": dr_boot.get("ec50_lo", np.nan),
                        "ci95EC50.up": dr_boot.get("ec50_hi", np.nan),
                    }
                )

    dr_fit_cols = [
        "name",
        "log.x",
        "log.y",
        "Samples",
        "EC50",
        "yEC50",
        "EC50.orig",
        "yEC50.orig",
    ]
    dr_fit = pd.DataFrame(dr_rows, columns=dr_fit_cols)

    dr_boot_cols = [
        "name",
        "meanEC50",
        "sdEC50",
        "ci95EC50.lo",
        "ci95EC50.up",
    ]
    dr_boot = pd.DataFrame(dr_boot_rows, columns=dr_boot_cols)
    dr_audit_cols = ["name", "dr_model", "dr_monotonic", "ec50_status", "aic_spline", "aic_4pl"]
    dr_audit = pd.DataFrame(dr_audit_rows, columns=dr_audit_cols)

    export_payload = {}
    if export_dir is not None:
        export_payload = export_results_zip(
            gc_fit=gc_fit,
            dr_fit=dr_fit,
            gc_boot=gc_boot if gc_boot_B > 0 else None,
            dr_boot=dr_boot if dr_boot_B > 0 else None,
            out_dir=Path(export_dir),
            zip_name=export_zip_name,
        )

    return {
        "gc_fit": gc_fit,
        "dr_fit": dr_fit,
        "gc_boot": gc_boot,
        "dr_boot": dr_boot,
        "dr_audit": dr_audit,
        **export_payload,
    }
