from __future__ import annotations
import hashlib
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Literal
from pathlib import Path
from growthqa.grofit.gc_fit_model import gc_fit_model
from growthqa.grofit.gc_fit_spline import gc_fit_spline
from growthqa.grofit.gc_boot_spline import gc_boot_spline, BootstrapMethod
from growthqa.grofit.dr_fit_spline import dr_fit_spline
from growthqa.grofit.dr_fit_model import dr_fit_model
from growthqa.grofit.dr_boot_spline import dr_boot_spline
from growthqa.grofit.interactive import apply_user_exclusion, UserFilterFn
from growthqa.grofit.export import export_results_zip
from growthqa.grofit.gc_fit_spline import GC_MIN_DF

ResponseVar= Literal["A", "mu", "lambda", "integral"]
FitOpt= Literal["m", "s", "b"]
DrFitMethod= Literal["auto", "spline", "4pl"]
GC_FIT_COLS: list[str]= [
    "test.id",
    "add.id",
    "concentration",
    "reliability", 
    "used.model", 
    "nboot.fit", 
    "n.obs", 
    "mu.model",
    "lambda.model",
    "A.model",
    "Integral.model",
    "mu.se", 
    "lambda.se",
    "A.se",
    "mu.spline",
    "lambda.spline",
    "A.spline",
    "integral.spline",
    "mu.low", 
    "mu.high",
    "lambda.low", 
    "lambda.high",
    "A.low", 
    "A.high",
    "Integral.low", 
    "Integral.high",
    "fit.status.spline", 
    "fit.status.model",
    "fail.reason.spline", 
    "fail.reason.model",
]

GC_BOOT_COLS: list[str]= [
    "test.id", "add.id", "concentration",
    "mu.bt",
    "sd.mu.bt",
    "ci90.mu.bt.lo", 
    "ci90.mu.bt.up",
    "ci95.mu.bt.lo", 
    "ci95.mu.bt.up",
    "lambda.bt", 
    "sd.lambda.bt",
    "ci90.lambda.bt.lo", 
    "ci90.lambda.bt.up",
    "ci95.lambda.bt.lo", 
    "ci95.lambda.bt.up",
    "A.bt", 
    "sd.A.bt",
    "ci90.A.bt.lo", 
    "ci90.A.bt.up",
    "ci95.A.bt.lo", 
    "ci95.A.bt.up",
    "integral.bt", 
    "sd.integral.bt",
    "ci90.integral.bt.lo",
    "ci90.integral.bt.up",
    "ci95.integral.bt.lo",
    "ci95.integral.bt.up",
]

GC_AUDIT_COLS: list[str]= [
    "test.id", "add.id", "concentration",
    "aic.logistic",
    "aic.gompertz",
    "aic.richards",
    "aic.modified_gompertz",
    "status.logistic",
    "status.gompertz",
    "status.richards",
    "status.modified_gompertz",
    "aic.model", 
    "bic.model", 
    "aic.second", 
    "delta.aic.second", 
    "smooth.used", 
    "df.effective", 
    "smoothing.method", 
    "lag.method.spline", 
    "lag.method.model", 
    "y0.baseline.spline", 
]

DR_FIT_COLS: list[str]= [
    "name",
    "log.x",
    "log.y",
    "Samples", 
    "n.conc", 
    "EC50", 
    "meanEC50", 
    "sdEC50", 
    "yEC50",
    "EC50.orig", 
    "yEC50.orig",
    "EC50.low", 
    "EC50.high", 
    "EC50.orig.low",
    "EC50.orig.high",
    "fit.status", 
    "fail.reason", 
    "dr.method", 
]

DR_BOOT_COLS: list[str]= [
    "name",
    "Samples", 
    "meanEC50", 
    "sdEC50", 
    "EC50.low", 
    "EC50.high", 
    "EC50.orig.low",
    "EC50.orig.high",
    "EC50.ec50_crossing_rate", 
]

DR_AUDIT_COLS: list[str]= [
    "name",
    "aic.spline", 
    "aic.4pl", 
    "delta.aic.dr", 
    "dr.monotonic", 
    "ec50.status", 
    "x_transform_norm", 
    "conc_span_orders_of_magnitude", 
    "log_transform_advisable", 
    "boot.method", 
    "ec50_crossing_rate",
    "ec50_fit_space",
]


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing= [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _stable_curve_seed(random_state: int, key: object) -> int:
    digest= hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:8]
    return int(random_state) + (int(digest, 16) % 10_000)


def _response_col_map(response_var: ResponseVar) -> dict[str,str]:
    if response_var== "mu":
        return {"param": "mu.model", "spline": "mu.spline"}
    if response_var== "lambda":
        return {"param": "lambda.model", "spline": "lambda.spline"}
    if response_var== "A":
        return {"param": "A.model", "spline": "A.spline"}
    return {"param": "Integral.model", "spline": "integral.spline"}


def _coerce_se(val: Any) -> float:
    if val is None:
        return np.nan
    try:
        return float(val)
    except (TypeError,ValueError):
        return np.nan


def run_grofit_pipeline(
    curves_df: pd.DataFrame,
    response_var: ResponseVar= "mu",
    have_atleast: int= 6,
    gc_boot_B: int= 200,
    dr_boot_B: int= 300,
    spline_auto_cv: bool= True,
    spline_s: Optional[float]= None,
    smooth_gc: Optional[float]= None,
    smooth_dr: Optional[float]= None,
    dr_x_transform: Optional[str]= None,
    dr_y_transform: Optional[str]= None,
    dr_s: Optional[float]= None,
    random_state: Optional[int]= 42,
    fit_opt: FitOpt= "b",
    bootstrap_method: BootstrapMethod= "pairs",
    validity_col: str= "is_valid_final",
    dr_fit_method: DrFitMethod= "auto",
    user_filter_fn: Optional[UserFilterFn]= None,
    export_dir: Optional[str | Path]= None,
    export_zip_name: str= "grofit_outputs.zip",
) -> Dict[str,Any]:
    _ensure_columns(curves_df,["test_id","curve_id","concentration","time","y"])

    df= curves_df.copy()
    if validity_col in df.columns:
        valid_values= (
            pd.Series(df[validity_col], index=df.index)
            .fillna(False).astype(bool).to_numpy()
        )
    elif "is_valid" in df.columns:
        valid_values= (
            pd.Series(df["is_valid"], index=df.index)
            .fillna(False).astype(bool).to_numpy()
        )
    else:
        # Manual mode: include all curves
        valid_values= np.ones(len(df), dtype=bool)
    df["is_valid_final"]= valid_values

    curve_index= (
        df.groupby(["test_id", "curve_id", "concentration"], dropna=False)["is_valid_final"]
        .first()
        .reset_index()
    )

    gc_fit_rows: list[dict]= []
    gc_boot_rows: list[dict]= []
    gc_audit_rows: list[dict]= []

    for _, row in curve_index.iterrows():
        test_id= row["test_id"]
        curve_id= row["curve_id"]
        conc= row["concentration"]
        is_valid_final= bool(row["is_valid_final"])

        g= df[
            (df["test_id"]== test_id)
            & (df["curve_id"]== curve_id)
            & (df["concentration"]== conc)
        ]
        t= g["time"].to_numpy()
        y= g["y"].to_numpy()
        n_obs= int((np.isfinite(t) & np.isfinite(y)).sum())

        pfit= None
        sfit= None
        boot= None

        if is_valid_final:
            if fit_opt in {"m", "b"}:
                pfit= gc_fit_model(t, y)
            if fit_opt in {"s", "b"}:
                sfit= gc_fit_spline(t, y, lam=spline_s, 
                                     auto_cv=(spline_auto_cv and spline_s is None and smooth_gc is None),
                                     smooth=smooth_gc 
                                     )
            if gc_boot_B > 0 and fit_opt in {"s", "b"}:
                boot= gc_boot_spline(
                    t, y,
                    B=gc_boot_B,
                    ci=0.95,
                    random_state=(
                        None if random_state is None
                        else _stable_curve_seed(random_state, curve_id)
                    ),
                    spline_s=spline_s,
                    auto_cv=(spline_auto_cv and spline_s is None and smooth_gc is None),
                    smooth=smooth_gc,
                    bootstrap_method=bootstrap_method,
                )
        def _actual_nboot(boot_result, requested_B, valid_final, opt) -> int: 
            if not (requested_B > 0 and valid_final and opt in {"s", "b"}):
                return 0
            if boot_result is None or not bool(boot_result.get("success")):
                return 0
            ns= [int(boot_result.get(k, {}).get("n", 0))
                for k in ("mu", "lambda", "A", "integral")]
            return min(ns) if ns else 0


        def _resolve_smoothing_method(fit) -> float | str:
            if fit is None:
                return np.nan
            method= (getattr(fit, "extra", {}) or {}).get("lam_method", np.nan)
            if method != "gcv_bounded":
                return method
            df_eff= getattr(fit, "df_effective", np.nan)
            if not np.isfinite(df_eff):
                return method
            return "gcv_bounded_floor" if df_eff >= (GC_MIN_DF - 1e-6) else "gcv_bounded_unreachable"
        
        def _v(fit, attr: str) -> float:
            if fit is None or not fit.success:
                return np.nan
            return getattr(fit, attr, np.nan)

        def _ex(fit, key: str, default: Any= np.nan) -> Any:
            if fit is None or not fit.success:
                return default
            return (getattr(fit, "extra", None) or {}).get(key, default)
        mu_se= _coerce_se(getattr(pfit, "mu_se", None) if pfit else None)
        A_se= _coerce_se(getattr(pfit, "A_se", None) if pfit else None)
        lambda_se= _coerce_se(getattr(pfit, "lag_se", None) if pfit else None)
        ex= (getattr(pfit, "extra", None) or {}) if pfit else {}
        gc_fit_rows.append({
            "test.id": test_id,
            "add.id": curve_id,
            "concentration": conc,
            "reliability": bool(is_valid_final),
            "used.model": ("" if (pfit is None or not pfit.success)
                               else (pfit.model or "")),
            "nboot.fit": _actual_nboot(boot, gc_boot_B, is_valid_final, fit_opt),
            "n.obs": n_obs,
            "mu.model": _v(pfit, "mu"),
            "lambda.model": _v(pfit, "lag"),
            "A.model": _v(pfit, "A"),
            "Integral.model": _v(pfit, "integral"),
            "mu.se": mu_se,
            "lambda.se": lambda_se,
            "A.se": A_se,
            "mu.spline": _v(sfit, "mu"),
            "lambda.spline": _v(sfit, "lag"),
            "A.spline": _v(sfit, "A"),
            "integral.spline": _v(sfit, "integral"),
            "mu.low": np.nan,
            "mu.high": np.nan,
            "lambda.low": np.nan,
            "lambda.high": np.nan,
            "A.low": np.nan,
            "A.high": np.nan,
            "Integral.low": np.nan,
            "Integral.high": np.nan,
            "fit.status.spline": getattr(sfit, "fit_status", "not_run") if sfit else "not_run",
            "fit.status.model": getattr(pfit, "fit_status", "not_run") if pfit else "not_run",
            "fail.reason.spline": getattr(sfit, "fail_reason", None) if sfit else None,
            "fail.reason.model": getattr(pfit, "fail_reason", None) if pfit else None,
        })
        _aic_by_model= {row["model"]: row.get("aic", np.nan) for row in ex.get("aic_table", [])}
        _status_by_model= {row["model"]: row.get("status", "unknown") for row in ex.get("aic_table", [])}

        gc_audit_rows.append({
            "test.id": test_id,
            "add.id": curve_id,
            "concentration": conc,
            "aic.logistic": _aic_by_model.get("logistic", np.nan),
            "aic.gompertz": _aic_by_model.get("gompertz", np.nan),
            "aic.richards": _aic_by_model.get("richards", np.nan),
            "aic.modified_gompertz": _aic_by_model.get("modified_gompertz", np.nan),
            "status.logistic": _status_by_model.get("logistic", "unknown"),
            "status.gompertz": _status_by_model.get("gompertz", "unknown"),
            "status.richards": _status_by_model.get("richards", "unknown"),
            "status.modified_gompertz": _status_by_model.get("modified_gompertz", "unknown"),
            "aic.model": _v(pfit, "aic"),
            "bic.model": _v(pfit, "bic"),
            "aic.second": ex.get("aic_second", np.nan),
            "delta.aic.second": ex.get("delta_aic_second", np.nan),
            "smooth.used": _v(sfit, "smooth_used"),
            "df.effective": _v(sfit, "df_effective"),
            "smoothing.method": _resolve_smoothing_method(sfit),
            "lag.method.spline": getattr(sfit, "lag_method", np.nan) if sfit else np.nan,
            "lag.method.model": getattr(pfit, "lag_method", np.nan) if pfit else np.nan,
            "y0.baseline.spline":_v(sfit, "y0_baseline"),
             })

        if gc_boot_B > 0 and boot is not None and bool(boot.get("success")):
            def _bt(key: str, stat: str) -> float:
                return float(boot[key].get(stat, np.nan))

            gc_boot_rows.append({
                "test.id": test_id,
                "add.id": curve_id,
                "concentration":conc,
                "mu.bt": _bt("mu", "mean"),
                "sd.mu.bt": _bt("mu", "sd"),
                "ci90.mu.bt.lo": _bt("mu", "lo90"),
                "ci90.mu.bt.up": _bt("mu", "hi90"),
                "ci95.mu.bt.lo": _bt("mu", "lo"),
                "ci95.mu.bt.up": _bt("mu", "hi"),
                "lambda.bt": _bt("lambda", "mean"),
                "sd.lambda.bt": _bt("lambda", "sd"),
                "ci90.lambda.bt.lo": _bt("lambda", "lo90"),
                "ci90.lambda.bt.up": _bt("lambda", "hi90"),
                "ci95.lambda.bt.lo": _bt("lambda", "lo"),
                "ci95.lambda.bt.up": _bt("lambda", "hi"),
                "A.bt": _bt("A", "mean"),
                "sd.A.bt": _bt("A", "sd"),
                "ci90.A.bt.lo": _bt("A", "lo90"),
                "ci90.A.bt.up": _bt("A", "hi90"),
                "ci95.A.bt.lo": _bt("A", "lo"),
                "ci95.A.bt.up": _bt("A", "hi"),
                "integral.bt": _bt("integral", "mean"),
                "sd.integral.bt": _bt("integral", "sd"),
                "ci90.integral.bt.lo": _bt("integral", "lo90"),
                "ci90.integral.bt.up": _bt("integral", "hi90"),
                "ci95.integral.bt.lo": _bt("integral", "lo"),
                "ci95.integral.bt.up": _bt("integral", "hi"),
            })

    gc_fit= pd.DataFrame(gc_fit_rows, columns=GC_FIT_COLS)
    gc_boot= pd.DataFrame(gc_boot_rows, columns=GC_BOOT_COLS)
    gc_audit= pd.DataFrame(gc_audit_rows, columns=GC_AUDIT_COLS)
    if gc_boot_B > 0 and not gc_boot.empty:
        boot_idx= gc_boot.set_index(["test.id", "add.id", "concentration"])
        for i, row in gc_fit.iterrows():
            key= (row["test.id"], row["add.id"], row["concentration"])
            if key in boot_idx.index:
                br= boot_idx.loc[key]
                gc_fit.at[i, "mu.low"]= br.get("ci95.mu.bt.lo", np.nan)
                gc_fit.at[i, "mu.high"]= br.get("ci95.mu.bt.up", np.nan)
                gc_fit.at[i, "lambda.low"]= br.get("ci95.lambda.bt.lo", np.nan)
                gc_fit.at[i, "lambda.high"]= br.get("ci95.lambda.bt.up", np.nan)
                gc_fit.at[i, "A.low"]= br.get("ci95.A.bt.lo", np.nan)
                gc_fit.at[i, "A.high"]= br.get("ci95.A.bt.up", np.nan)
                gc_fit.at[i, "Integral.low"]= br.get("ci95.integral.bt.lo", np.nan)
                gc_fit.at[i, "Integral.high"]= br.get("ci95.integral.bt.up", np.nan)

    dr_source= gc_fit[gc_fit["reliability"]== True].copy()
    dr_source= apply_user_exclusion(dr_source, user_filter_fn)

    resp_cols_map= _response_col_map(response_var)
    if fit_opt== "m":
        preferred_cols= [resp_cols_map["param"]]
    elif fit_opt== "s":
        preferred_cols= [resp_cols_map["spline"]]
    else:
        preferred_cols= [resp_cols_map["spline"], resp_cols_map["param"]]

    dr_rows: list[dict]= []
    dr_boot_rows: list[dict]= []
    dr_audit_rows: list[dict]= []
    log_x= 1 if str(dr_x_transform).strip().lower() in {"log1p", "log", "log10"} else 0
    log_y_dr= 1 if str(dr_y_transform or "").strip().lower() in {"log1p", "log10", "log"} else 0
    metric= str(response_var)
    for test_id in curve_index["test_id"].drop_duplicates():
        g= dr_source[dr_source["test.id"]== test_id].copy()
        def _dr_failed(reason: str, n_conc: int= 0) -> dict:
            return {
                "name": test_id,
                "log.x": log_x,
                "log.y": 0,
                "Samples": 0,
                "n.conc": n_conc,
                "EC50": np.nan,
                "yEC50": np.nan,
                "EC50.orig": np.nan,
                "yEC50.orig": np.nan,
                "EC50.low": np.nan,
                "EC50.high": np.nan,
                "EC50.orig.low": np.nan,
                "EC50.orig.high":np.nan,
                "fit.status": "failed",
                "fail.reason": reason,
                "dr.method": f"{metric}-none",
            }
        if g.empty:
            dr_rows.append(_dr_failed("no_valid_curves"))
            continue
        resp_col= None
        for cand in preferred_cols:
            if cand in g.columns:
                cand_vals= pd.to_numeric(g[cand], errors="coerce").to_numpy(dtype=float)
                if np.isfinite(cand_vals).any():
                    resp_col= cand
                    break

        if resp_col is None:
            dr_rows.append(_dr_failed("no_response_column"))
            continue

        gg= g[np.isfinite(
            pd.to_numeric(g[resp_col], errors="coerce").to_numpy(dtype=float)
        )].copy()
        gg[resp_col]= pd.to_numeric(gg[resp_col], errors="coerce")
        n_conc= int(gg["concentration"].nunique())

        if n_conc < int(have_atleast):
            dr_rows.append(_dr_failed(f"too_few_conc_points_{n_conc}", n_conc=n_conc))
            continue

        conc_arr= gg["concentration"].to_numpy(dtype=float)
        resp_arr= gg[resp_col].to_numpy(dtype=float)
        log_y= 1 if str(dr_y_transform or "").strip().lower() in {"log1p", "log10", "log"} else 0
        resp_for_fit= resp_arr
        spline_fit= dr_fit_spline(
            conc_arr, resp_for_fit,
            x_transform=dr_x_transform,
            lam=dr_s,
            smooth=smooth_dr,
            y_transform=dr_y_transform,
            auto_cv=(dr_s is None and smooth_dr is None),
            enforce_monotonic=True,
            fallback_to_4pl=(dr_fit_method != "spline"),
        )
        model_fit= (
            dr_fit_model(conc_arr, resp_for_fit)
            if dr_fit_method in {"auto", "4pl"}
            else {"success": False}
        )
        chosen= spline_fit
        chosen_name= str(spline_fit.get("method", "spline"))

        if dr_fit_method== "4pl":
            chosen, chosen_name= model_fit, "4pl"
        elif dr_fit_method== "auto":
            spline_ok= bool(spline_fit.get("success"))
            model_ok= bool(model_fit.get("success"))
            spline_aic= pd.to_numeric(
                pd.Series([spline_fit.get("aic", np.nan)]), errors="coerce"
            ).iloc[0]
            model_aic= pd.to_numeric(
                pd.Series([model_fit.get("aic", np.nan)]), errors="coerce"
            ).iloc[0]
            if model_ok and not spline_ok:
                chosen, chosen_name= model_fit, "4pl"
            elif spline_ok and not model_ok:
                chosen, chosen_name= spline_fit, str(spline_fit.get("method", "spline"))
            elif model_ok and spline_ok:
                if (np.isfinite(model_aic) and np.isfinite(spline_aic)
                        and float(model_aic) < float(spline_aic)):
                    chosen, chosen_name= model_fit, "4pl"
                else:
                    chosen, chosen_name= spline_fit, str(spline_fit.get("method", "spline"))
        dr_samples_used= 0
        ec50= chosen.get("ec50", np.nan)
        y_ec50= float(chosen.get("y_ec50")) if np.isfinite(chosen.get("y_ec50", np.nan)) else np.nan
        _y_ec50_orig= chosen.get("y_ec50_orig", np.nan)
        y_ec50_orig= (
            float(_y_ec50_orig) if np.isfinite(_y_ec50_orig) else y_ec50
        )
        _fit_used_raw= str(chosen_name).strip().lower() in {"4pl", "4pl_fallback"}
        log_y_row= 0 if _fit_used_raw else log_y
        log_x_row= 0 if _fit_used_raw else log_x
        _spline_is_fallback= str(spline_fit.get("method", "")).strip().lower()== "4pl_fallback"
        aic_s= np.nan if _spline_is_fallback else spline_fit.get("aic", np.nan)
        aic_m= model_fit.get("aic", np.nan)
        method_label= (
        "Spline" if str(chosen_name).lower().startswith(("spline", "mono", "smooth"))
        else ("4pl" if str(chosen_name).lower()== "4pl" else str(chosen_name))
        )
        dr_method_tag= f"{metric}-{method_label}"

        delta_aic_dr= (
            abs(float(aic_s) - float(aic_m))
            if (np.isfinite(aic_s) and np.isfinite(aic_m))
            else np.nan
        )
        dr_rows.append({
            "name": test_id,
            "log.x": log_x_row,
            "log.y": log_y_row,
            "Samples": dr_samples_used,
            "n.conc": n_conc,
            "EC50": chosen.get("ec50_x_transformed", ec50),
            "meanEC50": np.nan, 
            "sdEC50": np.nan, 
            "yEC50": float(chosen.get("y_ec50", np.nan)),
            "EC50.orig": ec50,
            "yEC50.orig": y_ec50_orig,
            "EC50.low": np.nan,
            "EC50.high": np.nan,
            "EC50.orig.low": np.nan,
            "EC50.orig.high": np.nan,
            "fit.status": "ok" if chosen.get("success") else "failed",
            "fail.reason": chosen.get("fail_reason", None),
            "dr.method": dr_method_tag,
        })
        
        _conc_finite= conc_arr[np.isfinite(conc_arr) & (conc_arr > 0)]
        _conc_span_orders= (
            float(np.log10(np.nanmax(_conc_finite) / np.nanmin(_conc_finite)))
            if _conc_finite.size >= 2 else np.nan)

        dr_audit_rows.append({
            "name": test_id,
            "aic.spline": aic_s,
            "aic.4pl": aic_m,
            "delta.aic.dr": delta_aic_dr,
            "dr.monotonic": bool(chosen.get("dr_monotonic", True)),
            "ec50.status": chosen.get("ec50_status", "OK"),
            "x_transform_norm": chosen.get("x_transform_norm", dr_x_transform or "none"),
            "ec50_fit_space": chosen.get("ec50_fit_space",
                    "raw_concentration" if chosen_name== "4pl" else "transformed"),
            "conc_span_orders_of_magnitude": _conc_span_orders,
            "log_transform_advisable": bool(
                np.isfinite(_conc_span_orders) and _conc_span_orders >= 2.0
                and (dr_x_transform or "none").lower()== "none"
            ),
            "boot.method": None,
            "ec50_crossing_rate": np.nan,
        })
        _dr_audit_row= dr_audit_rows[-1]

        if dr_boot_B > 0 and n_conc >= max(6, int(have_atleast)):
            dr_boot_result= dr_boot_spline(
                conc_arr, resp_arr,
                B=dr_boot_B,
                ci=0.95,
                random_state=(
                    None if random_state is None
                    else _stable_curve_seed(random_state, test_id)
                ),
                x_transform=dr_x_transform,
                lam=dr_s,
                fit_method=("4pl" if _fit_used_raw else "spline"),
            )

            _dr_audit_row["ec50_crossing_rate"]= dr_boot_result.get("ec50_crossing_rate", np.nan)
            _dr_audit_row["boot.method"]= dr_boot_result.get("fit_method", None)
            dr_samples_used= int(dr_boot_result.get("ec50_samples_n", 0) or 0)
            if dr_boot_result.get("success"):
                ec50_orig_lo= dr_boot_result.get("ec50_lo", np.nan)
                ec50_orig_hi= dr_boot_result.get("ec50_hi", np.nan)
                _xt= "" if _fit_used_raw else str(dr_x_transform or "").strip().lower()
                def _fwd(v: float) -> float:
                    if not np.isfinite(v) or v <= 0:
                        return np.nan
                    if _xt== "log10":
                        return float(np.log10(v))
                    if _xt== "log":
                        return float(np.log(v))
                    if _xt== "log1p":
                        return float(np.log1p(v))
                    return float(v)
                
                dr_boot_rows.append({
                    "name": test_id,
                    "Samples": dr_samples_used,
                    "meanEC50": dr_boot_result.get("ec50_mean", np.nan),
                    "sdEC50": dr_boot_result.get("ec50_sd", np.nan),
                    "EC50.low": _fwd(ec50_orig_lo), 
                    "EC50.high": _fwd(ec50_orig_hi),
                    "EC50.orig.low": ec50_orig_lo, 
                    "EC50.orig.high": ec50_orig_hi,
                    "EC50.crossing_rate": dr_boot_result.get("ec50_crossing_rate", np.nan),
                })

    dr_fit= pd.DataFrame(dr_rows, columns=DR_FIT_COLS)
    dr_boot= pd.DataFrame(dr_boot_rows, columns=DR_BOOT_COLS)
    dr_audit= pd.DataFrame(dr_audit_rows, columns=DR_AUDIT_COLS)
    if dr_boot_B > 0 and not dr_boot.empty:
        boot_dr_idx= dr_boot.set_index("name")
        for i, row in dr_fit.iterrows():
            nm= row["name"]
            if nm in boot_dr_idx.index:
                br= boot_dr_idx.loc[nm]
                dr_fit.at[i, "Samples"]= br.get("Samples", 0)
                dr_fit.at[i, "meanEC50"]= br.get("meanEC50", np.nan)
                dr_fit.at[i, "sdEC50"]= br.get("sdEC50", np.nan)
                dr_fit.at[i, "EC50.low"]= br.get("EC50.low", np.nan)
                dr_fit.at[i, "EC50.high"]= br.get("EC50.high", np.nan)
                dr_fit.at[i, "EC50.orig.low"]= br.get("EC50.orig.low", np.nan)
                dr_fit.at[i, "EC50.orig.high"]= br.get("EC50.orig.high", np.nan)

    export_payload: dict= {}
    if export_dir is not None:
        
        run_info= {
             "fit.opt": fit_opt,
             "smooth.gc": smooth_gc,
             "smooth.dr": smooth_dr,
             "nboot.gc": int(gc_boot_B),
             "nboot.dr": int(dr_boot_B),
             "have.atleast": int(have_atleast),
             "parameter": response_var,
             "log.x.dr": 1 if (dr_x_transform or "none") != "none" else 0,
             "log.y.dr": log_y_dr,
             "dr_x_transform": dr_x_transform,
             "dr_y_transform": dr_y_transform,
             "gc_lam_raw": spline_s,
             "dr_lam_raw": dr_s,
             "spline_auto_cv": bool(spline_auto_cv),
             "dr_fit_method": dr_fit_method,
             "bootstrap_method": bootstrap_method,
         }

        export_payload= export_results_zip(
            gc_fit=gc_fit,
            dr_fit=dr_fit,
            gc_boot=gc_boot if gc_boot_B > 0 else None,
            dr_boot=dr_boot if dr_boot_B > 0 else None,
            out_dir=Path(export_dir),
            zip_name=export_zip_name,
            run_info=run_info
        )

    return {
        "gc_fit": gc_fit,
        "gc_boot": gc_boot,
        "gc_audit": gc_audit,
        "dr_fit": dr_fit,
        "dr_boot": dr_boot,
        "dr_audit": dr_audit,
        **export_payload,
    }