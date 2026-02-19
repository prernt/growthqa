from __future__ import annotations

from typing import Optional, Iterable
import numpy as np
import pandas as pd

from growthqa.preprocess.timegrid import parse_time_from_header
from growthqa.grofit.gc_fit_spline import gc_fit_spline
from growthqa.grofit.gc_fit_model import gc_fit_model
from growthqa.grofit.dr_fit_spline import dr_fit_spline
from growthqa.grofit.parametric_models import get_model_specs, extract_grofit_params_from_curve


def _time_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if parse_time_from_header(str(c)) is not None]


def _extract_series(row: pd.Series, time_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    t = np.array([parse_time_from_header(c) for c in time_cols], dtype=float)
    y = pd.to_numeric(row[time_cols], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    return t[mask], y[mask]


def _spline_payload(
    t: np.ndarray,
    y: np.ndarray,
    *,
    spline_s: Optional[float],
    auto_cv: bool,
) -> dict:
    fit = gc_fit_spline(t, y, s=spline_s, auto_cv=auto_cv)
    if not fit.success:
        return {"ran": False}
    s_used = None
    if fit.extra:
        s_used = fit.extra.get("s")
    if s_used is None:
        s_used = spline_s if spline_s is not None else 0.0

    # Rebuild spline to get grid + curve
    from scipy.interpolate import UnivariateSpline

    order = np.argsort(t)
    t_sorted = t[order]
    y_sorted = y[order]
    k_spline = min(3, len(t_sorted) - 1)
    sp = UnivariateSpline(t_sorted, y_sorted, k=k_spline, s=float(s_used))
    t_grid = np.linspace(float(np.min(t_sorted)), float(np.max(t_sorted)), 400)
    y_grid = sp(t_grid)
    dy = sp.derivative(1)(t_grid)
    idx = int(np.nanargmax(dy))
    t_mu = float(t_grid[idx])
    y_mu = float(np.interp(t_mu, t_grid, y_grid))
    y0 = float(np.nanpercentile(y_grid, 5))
    return {
        "ran": True,
        "t_grid": t_grid,
        "y_hat": y_grid,
        "dy_dt": dy,
        "params": {
            "mu": float(fit.mu) if fit.mu is not None else np.nan,
            "lambda": float(fit.lag) if fit.lag is not None else np.nan,
            "A": float(fit.A) if fit.A is not None else np.nan,
            "integral": float(fit.integral) if fit.integral is not None else np.nan,
            "y0": y0,
            "t_mu": t_mu,
            "y_mu": y_mu,
        },
    }


def _param_payload(t: np.ndarray, y: np.ndarray) -> dict:
    fit = gc_fit_model(t, y)
    if not fit.success or fit.model is None:
        return {"ran": False}

    specs = get_model_specs(t, y)
    spec = specs.get(fit.model)
    if spec is None:
        return {"ran": False}

    t_grid = np.linspace(float(np.min(t)), float(np.max(t)), 400)
    params = fit.params if fit.params is not None else None
    if params is None:
        return {"ran": False}
    y_hat = spec.func(t_grid, *params)

    y0 = float((fit.extra or {}).get("y0", np.nan))
    A_param = float((fit.extra or {}).get("A_param", np.nan))
    derived = extract_grofit_params_from_curve(
        model_name=fit.model,
        t=t,
        y0=y0 if np.isfinite(y0) else float(np.nanmin(y)),
        A=A_param if np.isfinite(A_param) else float(np.nanmax(y) - np.nanmin(y)),
        fitted_func=lambda tt: spec.func(tt, *params),
        t_min=float(np.min(t)),
        t_max=float(np.max(t)),
    )
    return {
        "ran": True,
        "model_name": fit.model,
        "t_grid": t_grid,
        "y_hat": y_hat,
        "params": {
            "mu": float(derived.get("mu", np.nan)),
            "lambda": float(derived.get("lag", np.nan)),
            "A": float(derived.get("A", np.nan)),
            "integral": float(derived.get("integral", np.nan)),
        },
        "aic": float(fit.aic) if fit.aic is not None else np.nan,
        "passed_sanity": True,
    }


def build_curve_payloads(
    *,
    curves_df: pd.DataFrame,
    raw_wide: pd.DataFrame,
    proc_wide: pd.DataFrame,
    labels_df: pd.DataFrame,
    gc_boot: Optional[pd.DataFrame],
    spline_s: Optional[float],
    spline_auto_cv: bool,
    include_bootstrap: bool,
    test_id: Optional[str] = None,
    curve_ids: Optional[Iterable[str]] = None,
) -> dict[str, dict]:
    payloads: dict[str, dict] = {}
    raw_time_cols = _time_cols(raw_wide)
    proc_time_cols = _time_cols(proc_wide)

    label_map = labels_df.set_index("Test Id")
    curve_ids_set = set(curve_ids) if curve_ids is not None else None
    boot_df = gc_boot if isinstance(gc_boot, pd.DataFrame) and not gc_boot.empty else None

    for curve_id, grp in curves_df.groupby("curve_id"):
        if curve_ids_set is not None and str(curve_id) not in curve_ids_set:
            continue
        # raw/proc row
        raw_row = raw_wide.loc[raw_wide["Test Id"].astype(str) == str(curve_id)]
        proc_row = proc_wide.loc[proc_wide["Test Id"].astype(str) == str(curve_id)]
        if raw_row.empty or proc_row.empty:
            continue
        raw_row = raw_row.iloc[0]
        proc_row = proc_row.iloc[0]

        t_raw, y_raw = _extract_series(raw_row, raw_time_cols)
        t_proc, y_proc = _extract_series(proc_row, proc_time_cols)

        # fit data from tidy
        t_fit = grp["time"].to_numpy(dtype=float)
        y_fit = grp["y"].to_numpy(dtype=float)

        labels = {"pred": "", "final": "", "reviewed": False}
        if str(curve_id) in label_map.index:
            row = label_map.loc[str(curve_id)]
            labels["pred"] = str(row.get("pred_label", ""))
            labels["final"] = str(row.get("final_label", labels["pred"]))
            labels["reviewed"] = bool(row.get("Reviewed", False))

        spline = _spline_payload(t_fit, y_fit, spline_s=spline_s, auto_cv=spline_auto_cv)
        label_for_model = labels["final"] or labels["pred"]
        parametric = _param_payload(t_fit, y_fit) if str(label_for_model).lower() in {"valid", "true", "1"} else {"ran": False}
        bootstrap = {"ran": False}
        if include_bootstrap and boot_df is not None:
            boot_match = boot_df[boot_df["add.id"].astype(str) == str(curve_id)]
            if test_id is not None and "test.id" in boot_df.columns:
                boot_match = boot_match[boot_match["test.id"].astype(str) == str(test_id)]
            if not boot_match.empty:
                brow = boot_match.iloc[0]
                bootstrap = {
                    "ran": True,
                    "ci": {
                        "mu": [brow.get("ci95.mu.bt.lo"), brow.get("ci95.mu.bt.up")],
                        "lambda": [brow.get("ci95.lambda.bt.lo"), brow.get("ci95.lambda.bt.up")],
                        "A": [brow.get("ci95.A.bt.lo"), brow.get("ci95.A.bt.up")],
                        "integral": [brow.get("ci95.integral.bt.lo"), brow.get("ci95.integral.bt.up")],
                    },
                    "n": int(brow.get("nboot.fit", np.nan)) if "nboot.fit" in brow else None,
                }

        payloads[str(curve_id)] = {
            "curve_id": str(curve_id),
            "t_raw": t_raw,
            "y_raw": y_raw,
            "t_proc": t_proc,
            "y_proc": y_proc,
            "labels": labels,
            "spline": spline,
            "parametric": parametric,
            "bootstrap": bootstrap,
        }
    return payloads


def build_dr_payload(
    *,
    gc_fit: pd.DataFrame,
    labels_df: pd.DataFrame,
    dr_boot: Optional[pd.DataFrame],
    test_id: Optional[str],
    response_metric: str,
    label_source: str,
    include_unsure: bool,
    include_invalid: bool,
    dr_s: Optional[float],
    dr_x_transform: Optional[str],
    show_bootstrap: bool,
) -> dict:
    metric_map = {
        "mu": ("mu.spline", "mu.model"),
        "A": ("A.nonpara", "A.para"),
        "lambda": ("lambda.spline", "lambda.model"),
        "integral": ("integral.spline", "Integral.model"),
    }
    spline_col, model_col = metric_map.get(response_metric, ("mu.spline", "mu.model"))

    labels = labels_df.set_index("Test Id")
    rows = []
    excluded = 0
    for _, r in gc_fit.iterrows():
        curve_id = str(r.get("add.id"))
        if curve_id not in labels.index:
            continue
        if label_source == "final":
            lbl = labels.loc[curve_id].get("final_label", labels.loc[curve_id].get("pred_label", ""))
        else:
            lbl = labels.loc[curve_id].get("pred_label", "")
        lbl_norm = str(lbl).strip().lower()
        if lbl_norm == "unsure" and not include_unsure:
            excluded += 1
            continue
        if lbl_norm in {"invalid", "false", "0"} and not include_invalid:
            excluded += 1
            continue
        y_val = pd.to_numeric(pd.Series([r.get(spline_col)]), errors="coerce").iloc[0]
        if pd.isna(y_val) or not np.isfinite(float(y_val)):
            y_val = pd.to_numeric(pd.Series([r.get(model_col)]), errors="coerce").iloc[0]
        rows.append(
            {"curve_id": curve_id, "concentration": r.get("concentration"), "y": y_val}
        )

    if not rows:
        return {
            "metric": response_metric,
            "x_conc": [],
            "y_metric": [],
            "fit": {"x_grid": [], "y_hat": []},
            "bootstrap": {"ran": False},
            "n_points": 0,
            "excluded": excluded,
        }

    df = pd.DataFrame(rows)
    df = df[np.isfinite(df["y"].to_numpy(dtype=float))]
    x = df["concentration"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    fit = dr_fit_spline(x, y, x_transform=dr_x_transform, s=dr_s, auto_cv=(dr_s is None))
    x_grid = fit.get("x_grid")
    y_hat = fit.get("y_hat")
    if x_grid is None or y_hat is None:
        x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 400)
        try:
            from scipy.interpolate import UnivariateSpline

            if dr_x_transform == "log1p":
                xt = np.log1p(x)
                xt_grid = np.log1p(x_grid)
            else:
                xt = x
                xt_grid = x_grid
            order = np.argsort(xt)
            xt = xt[order]
            yy = y[order]
            k_spline = min(3, len(xt) - 1)
            s_used = fit.get("s") if fit.get("success") else (dr_s if dr_s is not None else 0.0)
            sp = UnivariateSpline(xt, yy, k=k_spline, s=float(s_used))
            y_hat = sp(xt_grid)
        except Exception:
            y_hat = np.interp(x_grid, np.sort(x), y[np.argsort(x)])

    dr_payload = {
        "metric": response_metric,
        "x_conc": x,
        "y_metric": y,
        "labels_used": label_source,
        "fit": {
            "x_grid": x_grid,
            "y_hat": y_hat,
            "ec50": fit.get("ec50"),
            "y_mid": fit.get("y_ec50"),
            "dr_monotonic": fit.get("dr_monotonic"),
            "ec50_status": fit.get("ec50_status"),
            "dr_method": fit.get("method"),
        },
        "bootstrap": {"ran": False},
        "n_points": int(len(x)),
        "excluded": excluded,
    }

    if show_bootstrap and isinstance(dr_boot, pd.DataFrame) and not dr_boot.empty:
        boot_match = dr_boot
        if test_id is not None and "name" in dr_boot.columns:
            boot_match = boot_match[boot_match["name"].astype(str) == str(test_id)]
        if not boot_match.empty:
            brow = boot_match.iloc[0]
            dr_payload["bootstrap"] = {
                "ran": True,
                "ec50_ci": [brow.get("ci95EC50.lo"), brow.get("ci95EC50.up")],
                "n": brow.get("Samples") if "Samples" in brow else None,
                "y_hat_q025": None,
                "y_hat_q975": None,
            }
    return dr_payload
