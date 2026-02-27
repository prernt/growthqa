# apps/streamlit_app.py
from __future__ import annotations



import tempfile
import sys
import importlib
import shutil
import io
import zipfile
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sklearn
import json
import platform
import re
import importlib

ROOT = Path(__file__).resolve().parents[1]

# Ensure repository paths are on sys.path for local imports
for cand in {
    ROOT,
    ROOT / "src",
    Path.cwd(),
    Path.cwd() / "src",
}:
    if cand.exists():
        sp = str(cand)
        if sp not in sys.path:
            sys.path.insert(0, sp)

from growthqa.grofit.pipeline import run_grofit_pipeline
from growthqa.classifier.train_from_meta import NOTEBOOK_STAGE1_CUSTOM_FEATURES
import growthqa.pipelines.auto_train_classifier as _auto_train_mod
from growthqa.preprocess.timegrid import parse_time_from_header
from growthqa.io.wide_format import long_to_wide_preserve_times, parse_any_file_to_long
from growthqa.io.audit import build_classifier_audit_df
from growthqa.io.grofit_io import build_grofit_input_df as build_grofit_input_artifact
from growthqa.pipelines.build_meta_dataset import run_merge_preprocess_meta
# compute_late_growth_features: inline fallback so the app does not depend on
# a specific version of growthqa.features.meta.  The canonical Stage-2 logic is
# in growthqa.stage2.late_window; this lightweight version is only used by the
# in-app _late_growth_map_from_wide pre-scan.
try:
    from growthqa.features.meta import compute_late_growth_features
except ImportError:
    def compute_late_growth_features(
        times: np.ndarray, ods: np.ndarray, start: float = 16.0
    ) -> dict:
        """Fallback: simple late-growth detector used only for UI pre-scan."""
        import numpy as _np
        t = _np.array(times, dtype=float)
        y = _np.array(ods, dtype=float)
        ok = _np.isfinite(t) & _np.isfinite(y)
        t, y = t[ok], y[ok]
        out = {"has_late_data": 0, "late_growth_detected": 0,
               "late_slope": float("nan"), "late_delta": float("nan"),
               "late_max_increase": float("nan"), "late_n_points": 0}
        if t.size == 0:
            return out
        t, y = t[_np.argsort(t)], y[_np.argsort(t)]
        late = t > float(start)
        if not _np.any(late):
            return out
        t_l, y_l = t[late], y[late]
        out["has_late_data"] = 1
        out["late_n_points"] = int(t_l.size)
        if t_l.size >= 2:
            dt, dy = _np.diff(t_l), _np.diff(y_l)
            good = dt > 1e-12
            if _np.any(good):
                out["late_slope"] = float(_np.nanmedian(dy[good] / dt[good]))
        ref = (t >= float(start)) & (t <= float(start + 2.0))
        if _np.any(ref):
            out["late_delta"] = float(_np.nanmedian(y_l[-max(1, int(_np.ceil(0.2 * y_l.size))):])
                                      - _np.nanmedian(y[ref]))
        od0 = float(_np.interp(float(start), t, y)) if t.size >= 2 else float("nan")
        if _np.isfinite(od0):
            out["late_max_increase"] = float(_np.nanmax(y_l) - od0)
        slope_ok = _np.isfinite(out["late_slope"])  and out["late_slope"]      > 0.01
        delta_ok = _np.isfinite(out["late_delta"])  and out["late_delta"]      > 0.03
        inc_ok   = _np.isfinite(out["late_max_increase"]) and out["late_max_increase"] > 0.05
        out["late_growth_detected"] = int(slope_ok or delta_ok or inc_ok)
        return out
from growthqa.viz.payloads import build_curve_payloads, build_dr_payload
from growthqa.pipelines.infer_labels import run_label_inference_from_uploaded_wide




# -------------------------
# User paths (as requested)
# -------------------------
MODEL_DIR = ROOT / "classifier_output" / "saved_models_selected"
TRAIN_META = ROOT / "data" / "train_data" / "meta.csv"
# TRAIN_META = ROOT / "data" / "output" / "metaNoGrofit.csv"


# =========================
# Settings (UI-minimal)
# =========================
@dataclass
class InferenceSettings:
    # Blank handling
    input_is_raw: bool = False               # UI checkbox
    global_blank: float | None = None        # optional UI field when raw

    # Fixed pipeline defaults (not exposed in UI)
    step: float = 0.5
    min_points: int = 3
    low_res_threshold: int = 7
    auto_tmax: bool = False
    auto_tmax_coverage: float = 0.8
    tmax_hours: float | None = 16.0

    # Not exposed anymore; forced OFF
    clip_negatives: bool = False
    smooth_method: str = "SGF"   # Savitzky-Golay smoothing as default
    smooth_window: int = 5
    normalize: str = "MINMAX"       # scale curves to max=1


@dataclass
class GrofitOptions:
    response_var: str = "mu"
    have_atleast: int = 6
    fit_opt: str = "b"
    gc_boot_B: int = 200
    dr_boot_B: int = 300
    spline_auto_cv: bool = True
    spline_s: float | None = None          # legacy: manual spline smoothing param
    dr_s: float | None = None              # legacy: manual DR smoothing param
    smooth_gc: float | None = None         # Grofit-R-style smooth.gc spar ∈ (0,1]
    smooth_dr: float | None = None         # Grofit-R-style smooth.dr spar ∈ (0,1]
    dr_x_transform: str | None = None
    dr_y_transform: str | None = None
    bootstrap_method: str = None


# =========================
# Helpers
# =========================

def assert_runtime_matches_model(model_path: str):
    mp = Path(model_path)
    manifest = mp.with_suffix(".manifest.json")
    if not manifest.exists():
        return  # allow old models, but best practice is always to have this

    m = json.loads(manifest.read_text(encoding="utf-8"))

    problems = []
    if m.get("python_version") != platform.python_version():
        problems.append(f"Python {platform.python_version()} != trained {m.get('python_version')}")
    if m.get("sklearn_version") != sklearn.__version__:
        problems.append(f"sklearn {sklearn.__version__} != trained {m.get('sklearn_version')}")
    if m.get("numpy_version") != np.__version__:
        problems.append(f"numpy {np.__version__} != trained {m.get('numpy_version')}")
    if m.get("joblib_version") != joblib.__version__:
        problems.append(f"joblib {joblib.__version__} != trained {m.get('joblib_version')}")

    if problems:
        # On mismatches (common when deploying on a newer Python), log a warning but continue
        warn_msg = (
            "Model/runtime version mismatch detected:\n"
            + "\n".join(["- " + p for p in problems])
            + "\nProceeding anyway; retrain or regenerate models to silence this warning."
        )
        print(warn_msg, file=sys.stderr)

def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _to_numeric_scalar(x) -> float:
    return float(pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0])


def _normalize_bootstrap_method(v: object) -> str:
    s = str(v).strip().lower()
    return s if s in {"pairs", "residual"} else "pairs"


def _isfinite_scalar(x) -> bool:
    try:
        return bool(np.isfinite(_to_numeric_scalar(x)))
    except Exception:
        return False


def _st_pills_multi(
    label: str,
    options: list[str],
    default: list[str] | None = None,
    key: str | None = None,
) -> list[str]:
    """
    Backward-compatible multi-select pills.
    Uses st.pills when available; falls back to st.multiselect on older Streamlit.
    """
    default_vals = [str(v) for v in (default or []) if str(v) in {str(o) for o in options}]
    if hasattr(st, "pills"):
        selected = st.pills(
            label,
            options=options,
            default=default_vals,
            selection_mode="multi",
            key=key,
        )
        if selected is None:
            return []
        return [str(v) for v in selected]
    selected = st.multiselect(label, options=options, default=default_vals, key=key)
    return [str(v) for v in selected]


def has_trained_models(model_dir: Path) -> bool:
    p = Path(model_dir)
    if not p.exists():
        return False
    return any(p.glob("*.joblib"))


def train_classifier_from_meta_file(
    *,
    meta_csv_path: str | Path,
    models_out_dir: str | Path,
    selected_features: list[str] | None = None,
) -> dict:
    """
    Compatibility wrapper: use module function when available; otherwise fallback.
    """
    if hasattr(_auto_train_mod, "train_classifier_from_meta_file"):
        return _auto_train_mod.train_classifier_from_meta_file(
            meta_csv_path=meta_csv_path,
            models_out_dir=models_out_dir,
            selected_features=selected_features,
        )

    # Fallback for stale/older loaded module versions in long-running Streamlit process.
    from growthqa.classifier.train_from_meta import train_from_meta_csv
    import shutil

    out_dir = Path(models_out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return train_from_meta_csv(
        meta_csv=meta_csv_path,
        art_dir=out_dir,
        selected_features=selected_features,
    )


def make_plot(df_one: pd.Series, time_cols: list[str], title: str):
    xs, ys = [], []
    for c in time_cols:
        try:
            t = float(c.split("(")[0].strip()[1:].strip())
        except Exception:
            continue
        xs.append(t)
        ys.append(df_one.get(c, np.nan))

    order = np.argsort(xs) if len(xs) else []
    xs = np.array(xs)[order] if len(xs) else np.array([])
    ys = np.array(ys)[order] if len(ys) else np.array([])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers"))
    fig.update_layout(
        title=title,
        xaxis_title="Time (hours)",
        yaxis_title="Relative OD (normalized)",
        height=420,
        margin=dict(l=30, r=10, t=50, b=40),
    )
    return fig


def make_overlay_plot(
    raw_row: pd.Series,
    raw_time_cols: list[str],
    proc_row: pd.Series,
    proc_time_cols: list[str],
    *,
    title: str,
    input_is_raw: bool,
    global_blank: float | None,
):
    def _series_from_row(row: pd.Series, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for c in cols:
            try:
                t = float(c.split("(")[0].strip()[1:].strip())
            except Exception:
                continue
            xs.append(t)
            ys.append(row.get(c, np.nan))
        order = np.argsort(xs) if len(xs) else []
        xs = np.array(xs)[order] if len(xs) else np.array([])
        ys = np.array(ys)[order] if len(ys) else np.array([])
        return xs, ys

    raw_x, raw_y = _series_from_row(raw_row, raw_time_cols)
    if input_is_raw and global_blank is not None:
        raw_y = raw_y - float(global_blank)

    proc_x, proc_y = _series_from_row(proc_row, proc_time_cols)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=raw_x, y=raw_y, mode="lines+markers", name="Raw"))
    fig.add_trace(go.Scatter(x=proc_x, y=proc_y, mode="lines+markers", name="Processed"))
    fig.update_layout(
        title=title,
        xaxis_title="Time (hours)",
        yaxis_title="Relative OD",
        height=700,
        margin=dict(l=30, r=10, t=50, b=40),
    )
    return fig


# def _generate_fast_bootstrap_bands(t, y, lam_s, *, t_grid: np.ndarray):
#     from scipy.interpolate import make_smoothing_spline

#     t, y = np.asarray(t, float), np.asarray(y, float)
#     order = np.argsort(t)
#     t, y = t[order], y[order]

#     try:
#         lam_s = float(lam_s) if lam_s is not None else None
#         sp_base = make_smoothing_spline(t, y, lam=lam_s)
#         y_fit = sp_base(t)
#         resid = y - y_fit
#         t_grid = np.linspace(np.min(t), np.max(t), 400)

#         y_boots = []
#         rng = np.random.default_rng()
#         for _ in range(B):
#             y_b = y_fit + rng.choice(resid, size=len(resid), replace=True)
#             try:
#                 sp_b = make_smoothing_spline(t, y_b, lam=lam_s)
#                 y_boots.append(sp_b(t_grid))
#             except Exception:
#                 pass
#         if not y_boots:
#             return None, None
#         y_boots = np.array(y_boots)
#         return np.percentile(y_boots, 2.5, axis=0), np.percentile(y_boots, 97.5, axis=0)
#     except Exception:
#         return None, None

def _generate_fast_bootstrap_bands(t, y, lam_s, *, t_grid: np.ndarray) -> tuple:
    """
    Residual bootstrap on the smoothing spline.

    Parameters
    ----------
    t, y    : the tidy fit arrays (same data/scale that produced the orange spline).
    lam_s   : the locked smoothing lambda from _spline_payload (spline["lam"]).
    t_grid  : the EXACT grid from spline["t_grid"] so the band aligns with the curve.

    Returns (y_q025, y_q975) both on t_grid, or (None, None) on failure.
    """
    from scipy.interpolate import make_smoothing_spline
    from growthqa.grofit.gc_fit_spline import _dedupe_sorted_xy

    t, y = np.asarray(t, float), np.asarray(y, float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if len(t) < 6:
        return None, None

    try:
        lam_s = float(lam_s) if lam_s is not None else None

        order = np.argsort(t)
        t_s, y_s = t[order], y[order]
        # Deduplicate exactly as gc_fit_spline does so we match the orange curve
        t_u, y_u = _dedupe_sorted_xy(t_s, y_s)

        sp_base = make_smoothing_spline(t_u, y_u, lam=lam_s)
        y_base  = sp_base(t_u)
        resid   = y_u - y_base

        y_boots = []
        rng = np.random.default_rng(42)
        for _ in range(200):
            y_b = y_base + rng.choice(resid, size=len(resid), replace=True)
            try:
                sp_b = make_smoothing_spline(t_u, y_b, lam=lam_s)
                y_boots.append(sp_b(t_grid))
            except Exception:
                pass
        if len(y_boots) < 10:
            return None, None
        arr = np.array(y_boots)
        return np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0)
    except Exception:
        return None, None


def make_overlay_plot_payload(
    payload: dict,
    *,
    title: str,
    show_spline: bool,
    show_model: bool,
    show_bootstrap: bool,
) -> go.Figure:
    fig = go.Figure()
    spline_color = "#f28e2b"   # orange
    param_color = "#2ca02c"    # green
    tangent_color = "#8b4513"  # brown
    bootstrap_color = "rgba(128,0,128,0.24)"  # purple

    t_raw = payload.get("t_raw", np.array([]))
    y_raw = payload.get("y_raw", np.array([]))
    t_proc = payload.get("t_proc", np.array([]))
    y_proc = payload.get("y_proc", np.array([]))

    if len(t_raw) and len(y_raw):
        fig.add_trace(go.Scatter(x=t_raw, y=y_raw, mode="markers", name="Raw"))
    if len(t_proc) and len(y_proc):
        fig.add_trace(go.Scatter(x=t_proc, y=y_proc, mode="lines", name="Processed"))

    spline = payload.get("spline", {})
    if show_spline and spline.get("ran"):
        fig.add_trace(
            go.Scatter(
                x=spline["t_grid"],
                y=spline["y_hat"],
                mode="lines",
                name="Spline Fit",
                line=dict(dash="dash", color=spline_color),
            )
        )

        params = spline.get("params", {})
        t_mu = _to_numeric_scalar(params.get("t_mu"))
        y_mu = _to_numeric_scalar(params.get("y_mu"))
        mu = _to_numeric_scalar(params.get("mu"))
        lam = _to_numeric_scalar(params.get("lambda"))
        y0 = _to_numeric_scalar(params.get("y0"))
        A = _to_numeric_scalar(params.get("A"))
        if np.isfinite(t_mu) and np.isfinite(y_mu):
            fig.add_trace(
                go.Scatter(
                    x=[t_mu],
                    y=[y_mu],
                    mode="markers+text",
                    name="μ point",
                    text=["μ"],
                    textposition="top center",
                    marker=dict(size=14, color=spline_color, line=dict(width=1, color="#5a3a16")),
                )
            )
        # Tangent at max growth using point-slope form anchored at (t_mu, y_mu)
        if np.isfinite(mu) and np.isfinite(lam) and abs(float(mu)) > 1e-12:
            if np.isfinite(y0) and np.isfinite(A) and A > 0 and np.isfinite(t_mu) and np.isfinite(y_mu):
                x_base = float(lam)
                x_plat = float(lam + A / mu)

                # Add 15% visual padding to the line length so it extends cleanly
                x_span = x_plat - x_base
                x_min_plot = x_base - 0.15 * x_span
                x_max_plot = x_plat + 0.15 * x_span

                x_line = np.array([x_min_plot, x_max_plot], dtype=float)

                # MATHEMATICAL FIX: Anchor the equation directly to the plotted dot
                y_line = mu * (x_line - t_mu) + y_mu

                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name="Tangent (mu)",
                        line=dict(color=tangent_color, width=2),
                    )
                )
        if np.isfinite(lam):
            fig.add_vline(x=float(lam), line_dash="dot", line_color="#7a6a5f")
        if np.isfinite(y0):
            fig.add_hline(y=float(y0), line_dash="dot", line_color="#7a6a5f")
        if np.isfinite(y0) and np.isfinite(A) and np.isfinite(t_mu):
            fig.add_shape(
                type="line",
                x0=float(t_mu),
                x1=float(t_mu),
                y0=float(y0),
                y1=float(y0 + A),
                line=dict(color="#7a6a5f", width=2),
            )
            fig.add_annotation(
                x=float(t_mu),
                y=float(y0 + A),
                text=f"A={A:.3g}",
                showarrow=True,
                arrowhead=2,
            )

    parametric = payload.get("parametric", {})
    if show_model and parametric.get("ran") and parametric.get("passed_sanity", True):
        fig.add_trace(
            go.Scatter(
                x=parametric["t_grid"],
                y=parametric["y_hat"],
                mode="lines",
                name=f"Model ({parametric.get('model_name','')})",
                line=dict(dash="dashdot", color=param_color),
            )
        )

    bootstrap = payload.get("bootstrap", {})
    if show_bootstrap and bootstrap.get("ran") and bootstrap.get("y_hat_q025") is not None:
        fig.add_trace(
            go.Scatter(
                x=spline.get("t_grid"),
                y=bootstrap.get("y_hat_q975"),
                line=dict(width=2.5, color="rgba(128,0,128,0.95)"),
                showlegend=False,
                hoverinfo="skip",
                name="Boot CI",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=spline.get("t_grid"),
                y=bootstrap.get("y_hat_q025"),
                line=dict(width=2.5, color="rgba(128,0,128,0.95)"),
                fill="tonexty",
                fillcolor="rgba(128,0,128,0.36)",
                showlegend=True,
                name="Bootstrap band",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (hours)",
        yaxis_title="Relative OD",
        height=520,
        margin=dict(l=30, r=10, t=50, b=40),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, bgcolor="rgba(255, 255, 255, 0.5)"),
    )
    return fig


def make_dr_plot(payload: dict, *, show_bootstrap: bool) -> go.Figure:
    fig = go.Figure()
    x = np.asarray(payload.get("x_conc", []), dtype=float)
    y = np.asarray(payload.get("y_metric", []), dtype=float)
    if len(x) and len(y):
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data"))

    fit = payload.get("fit", {})
    x_grid = fit.get("x_grid", [])
    y_hat = fit.get("y_hat", [])
    if len(x_grid) and len(y_hat):
        fig.add_trace(go.Scatter(x=x_grid, y=y_hat, mode="lines", name="DR Spline"))

    ec50 = _to_numeric_scalar(fit.get("ec50"))
    y_mid = _to_numeric_scalar(fit.get("y_mid"))
    if np.isfinite(ec50):
        fig.add_vline(x=float(ec50), line_dash="dot", line_color="#7a6a5f")
    if np.isfinite(y_mid):
        fig.add_hline(y=float(y_mid), line_dash="dot", line_color="#7a6a5f")

    boot = payload.get("bootstrap", {})
    if show_bootstrap and boot.get("ran") and boot.get("y_hat_q025") is not None:
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=boot.get("y_hat_q975"),
                line=dict(width=2.5, color="rgba(128,0,128,0.95)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=boot.get("y_hat_q025"),
                line=dict(width=2.5, color="rgba(128,0,128,0.95)"),
                fill="tonexty",
                fillcolor="rgba(128,0,128,0.36)",
                showlegend=True,
                name="Bootstrap band",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        xaxis_title="Concentration",
        yaxis_title=payload.get("metric", "response"),
        height=420,
        margin=dict(l=30, r=10, t=30, b=40),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, bgcolor="rgba(255, 255, 255, 0.5)"),
    )
    return fig


def load_model_pipeline(model_path: str):
    _install_legacy_sklearn_pickle_aliases()
    assert_runtime_matches_model(model_path)
    return joblib.load(model_path)


def _install_legacy_sklearn_pickle_aliases() -> None:
    legacy_mod = "sklearn.ensemble._hist_gradient_boosting.loss"
    if legacy_mod in sys.modules:
        return
    try:
        new_mod = importlib.import_module("sklearn._loss.loss")
    except Exception:
        return
    sys.modules[legacy_mod] = new_mod


def discover_models(model_dir: str) -> dict[str, Path]:
    p = Path(model_dir)
    if not p.exists():
        return {}
    models = {}
    for f in sorted(p.glob("*.joblib")):
        models[f.stem] = f
    return models


def predict_hard_with_confidence(pipeline, meta_df: pd.DataFrame):
    """
    No thresholds, no 'Unsure'.
    - pred_label = pipeline.predict(...)
    - confidence = max(predict_proba) if available else NaN
    - also return p_valid if we can identify 'valid' class, else NaN
    """
    non_features = {"FileName", "Test Id"}
    X = meta_df.drop(columns=[c for c in meta_df.columns if c in non_features], errors="ignore")

    # Align columns to what the pipeline was trained on: drop unseen, add missing as NaN
    expected_features = getattr(pipeline, "feature_names_in_", None)
    if expected_features is not None:
        expected = [str(c) for c in expected_features]
        X = X.reindex(columns=expected, fill_value=np.nan)

    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="coerce")

    pred_label = pipeline.predict(X)

    conf = np.full(len(X), np.nan, dtype=float)
    p_valid = np.full(len(X), np.nan, dtype=float)

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        conf = np.max(proba, axis=1).astype(float)

        classes = getattr(pipeline, "classes_", None)
        if classes is not None and len(classes) == proba.shape[1]:
            cls_list = [str(c).strip().lower() for c in classes]
            if "valid" in cls_list:
                p_valid = proba[:, cls_list.index("valid")].astype(float)
            elif "true" in cls_list:
                p_valid = proba[:, cls_list.index("true")].astype(float)
            elif "1" in cls_list:
                p_valid = proba[:, cls_list.index("1")].astype(float)

    # normalize labels a bit for display
    pred_label_norm = []
    for v in pred_label:
        s = str(v).strip()
        if s.lower() in {"1", "true"}:
            pred_label_norm.append("Valid")
        elif s.lower() in {"0", "false"}:
            pred_label_norm.append("Invalid")
        elif s.lower() == "valid":
            pred_label_norm.append("Valid")
        elif s.lower() == "invalid":
            pred_label_norm.append("Invalid")
        else:
            pred_label_norm.append(s)
    pred_label_norm = np.array(pred_label_norm, dtype=object)

    return pred_label_norm, conf, p_valid


def _labels_to_prob_valid(labels: np.ndarray) -> np.ndarray:
    lbl = np.char.lower(labels.astype(str))
    prob = np.full(lbl.shape, np.nan, dtype=float)
    prob[np.isin(lbl, ["valid", "true", "1"])] = 1.0
    prob[np.isin(lbl, ["invalid", "false", "0"])] = 0.0
    return prob


def _label_is_valid(label: object) -> bool:
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return False
    s = str(label).strip().lower()
    return s in {"valid", "true", "1"}


def _late_growth_map_from_wide(wide_df: pd.DataFrame) -> dict[str, int]:
    tcols = [c for c in wide_df.columns if parse_time_from_header(str(c)) is not None]
    if not tcols:
        return {}
    times = np.array([parse_time_from_header(str(c)) for c in tcols], dtype=float)
    out = {}
    for _, row in wide_df.iterrows():
        ods = pd.to_numeric(row[tcols], errors="coerce").to_numpy(dtype=float)
        late = compute_late_growth_features(times, ods, start=16.0)
        out[str(row.get("Test Id"))] = int(late.get("late_growth_detected", 0))
    return out


def _assign_final_reason_labels(out_df: pd.DataFrame, wide_df: pd.DataFrame) -> pd.DataFrame:
    late_growth = _late_growth_map_from_wide(wide_df)
    final_labels: list[str] = []
    final_reasons: list[str] = []

    for _, row in out_df.iterrows():
        pred = str(row.get("pred_label", "")).strip().lower()
        conf_valid = pd.to_numeric(pd.Series([row.get("confidence_valid", np.nan)]), errors="coerce").iloc[0]
        too_sparse = bool(row.get("too_sparse", False))
        long_gaps = bool(row.get("too_sparse", False))
        gap = pd.to_numeric(pd.Series([row.get("max_gap_hours", np.nan)]), errors="coerce").iloc[0]
        miss = pd.to_numeric(pd.Series([row.get("missing_frac_on_grid", np.nan)]), errors="coerce").iloc[0]
        if (np.isfinite(gap) and gap > 2.0) or (np.isfinite(miss) and miss > 0.30):
            long_gaps = True

        if too_sparse:
            final_labels.append("UNSURE")
            final_reasons.append("UNSURE_TOO_SPARSE")
            continue
        if long_gaps:
            final_labels.append("UNSURE")
            final_reasons.append("UNSURE_LONG_GAPS")
            continue
        if np.isfinite(conf_valid) and min(conf_valid, 1.0 - conf_valid) > 0.40:
            final_labels.append("UNSURE")
            final_reasons.append("UNSURE_LOW_CONFIDENCE")
            continue

        if pred in {"valid", "true", "1"}:
            final_labels.append("VALID")
            final_reasons.append("OK_STAGE1_VALID")
            continue

        tid = str(row.get("Test Id"))
        if int(late_growth.get(tid, 0)) == 1:
            final_labels.append("UNSURE")
            final_reasons.append("UNSURE_LATE_GROWTH_AFTER_16H")
        else:
            final_labels.append("INVALID")
            final_reasons.append("OK_STAGE1_INVALID_NO_LATE_GROWTH")

    out = out_df.copy()
    out["final_label"] = final_labels
    out["final_reason"] = final_reasons
    return out


def show_friendly_error(exc: Exception):
    st.markdown(
        """
        <div style="padding:12px;border-radius:8px;border:2px solid #c00;background:#ffecec;color:#600;font-size:17px;font-weight:700;">
        Run failed. Please write what you were doing and send the issue details/logs to <a href="mailto:theprerna@uni-koblenz.de">theprerna@uni-koblenz.de</a> for improvement and feedback.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Error: {type(exc).__name__}: {exc}")


def line_with_tooltip(label: str, value: object, tooltip: str):
    safe_val = "" if value is None else value
    icon = (
        f'<sup style="margin-left:6px;color:#888;" title="{tooltip}">🛈</sup>'
        if tooltip
        else ""
    )
    st.markdown(
        f'<div title="{tooltip}"><strong>{label}:</strong> {safe_val}{icon}</div>',
        unsafe_allow_html=True,
    )

def _extract_conc_from_curve_id(curve_id: str) -> float | None:
    """
    Extract concentration from well header patterns like:
      - A01[Conc=0.1]
      - A01[0.1]
    Works even after prefixing, e.g.:
      - myfile_A01[Conc=0.1]
      - myfile_A02[0.1]

    Returns float concentration if found, else None.
    """
    if curve_id is None:
        return None
    s = str(curve_id)

    # Look for "[Conc=...]" or "[...]" numeric
    m = re.search(r"\[(?:\s*Conc\s*=\s*)?([0-9]+(?:\.[0-9]+)?)\s*\]", s, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _strip_bracketed_suffix(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"\s*\[.*?\]\s*", "", str(value)).strip()


def _make_curve_key(test_id: str, concentration: object) -> str:
    conc = "" if concentration is None or pd.isna(concentration) else str(concentration)
    return f"{test_id}|{conc}"


def _init_review_df(out_df: pd.DataFrame, wide_df: pd.DataFrame) -> pd.DataFrame:
    df = out_df.copy()
    if "is_valid_pred" not in df.columns:
        df["is_valid_pred"] = df["pred_label"].map(_label_is_valid)
    if "final_label" not in df.columns:
        df["final_label"] = df["pred_label"].astype(str)
    if "true_label" not in df.columns:
        df["true_label"] = df["final_label"].astype(str)
    if "is_valid_true" not in df.columns:
        df["is_valid_true"] = df["true_label"].map(_label_is_valid).astype(bool)
    if "Reviewed" not in df.columns:
        df["Reviewed"] = False
    # Backward compatibility for existing UI pieces.
    df["is_valid_final"] = df["is_valid_true"].astype(bool)

    if "Concentration" in df.columns:
        conc = df["Concentration"]
    elif "Concentration" in wide_df.columns:
        conc_map = wide_df.set_index("Test Id")["Concentration"]
        conc = df["Test Id"].map(conc_map)
    else:
        conc = df["Test Id"].astype(str).map(_extract_conc_from_curve_id)
    df["Concentration"] = pd.to_numeric(conc, errors="coerce")

    df["CurveKey"] = df.apply(lambda r: _make_curve_key(str(r["Test Id"]), r["Concentration"]), axis=1)
    return df




def run_training():
    """
    Run the training script in-process so artifacts are produced with the same environment
    as the running Streamlit app. Uses repo-relative defaults and clears old artifacts first.
    """
    train_mod = importlib.import_module("growthqa.classifier.train_from_meta")
    meta_path = ROOT / "data" / "train_data" / "meta.csv"
    # meta_path = ROOT / "data" / "output" / "metaNoGrofit.csv"
    art_dir = ROOT / "classifier_output" / "saved_models_selected"
    lockfile_out = ROOT / "classifier_output" / "requirements_lock.txt"

    # Clear classifier_output to ensure only fresh artifacts remain
    clf_root = art_dir.parent
    if clf_root.exists():
        for child in clf_root.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
    clf_root.mkdir(parents=True, exist_ok=True)

    train_mod.TRAIN_META_CSV = str(meta_path)
    train_mod.ART_DIR = str(art_dir)
    train_mod.LOCKFILE_OUT = str(lockfile_out)
    train_mod.main()


def make_sample_wide_csv_bytes() -> bytes:
    """
    Grofit-ready wide-format sample (one row per curve).
    Includes a 'concentration' column so dose-response can work.
    """
    data = {
        "Test Id": ["A01", "A02", "A03", "A04", "A05", "A06"],
        "concentration": [0.0, 0.1, 0.3, 1.0, 3.0, 10.0],
        "T0.0 (h)":  [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        "T1.0 (h)":  [0.07, 0.07, 0.068, 0.065, 0.060, 0.055],
        "T2.0 (h)":  [0.12, 0.115, 0.105, 0.095, 0.075, 0.060],
        "T3.0 (h)":  [0.22, 0.205, 0.185, 0.155, 0.095, 0.070],
        "T4.0 (h)":  [0.35, 0.32, 0.28, 0.22, 0.115, 0.080],
        "T5.0 (h)":  [0.48, 0.43, 0.36, 0.26, 0.130, 0.090],
        "T6.0 (h)":  [0.60, 0.52, 0.41, 0.29, 0.140, 0.095],
        "T7.0 (h)":  [0.70, 0.58, 0.44, 0.31, 0.145, 0.098],
        "T8.0 (h)":  [0.78, 0.62, 0.46, 0.32, 0.148, 0.100],
    }
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode("utf-8")


def make_sample_long_csv_bytes() -> bytes:
    """
    Long-ish sample: Time column + wells as columns.

    The well column headers encode concentration using either of these accepted formats:
      - A01[Conc=0.1]
      - A01[0.1]

    (Both should be acceptable in the first row header of Excel/CSV.)
    """
    times = [0, 0.5, 1, 1.5, 2, 3, 3.5, 4, 4.5, 5, 6, 7, 8]

    # Same doses as sample_wide
    # Mix both header styles on purpose to show both are acceptable
    cols = [
        ("A01[Conc=0.0]", 0.0),
        ("A02[0.1]", 0.1),
        ("A03[Conc=0.3]", 0.3),
        ("A04[1.0]", 1.0),
        ("A05[Conc=3.0]", 3.0),
        ("A06[10.0]", 10.0),
    ]

    # Simple synthetic growth-like series with inhibition at higher concentrations
    # (values are just for demonstration)
    base = np.array([0.05, 0.055, 0.065, 0.080, 0.110, 0.200, 0.270, 0.340, 0.410, 0.480, 0.600, 0.700, 0.780])

    def inhibited_series(strength: float) -> np.ndarray:
        # strength in [0..1], larger => more inhibition
        # keep baseline similar, dampen later growth
        damp = 1.0 - strength
        y = base.copy()
        y[4:] = y[4:] * damp + 0.05 * (1.0 - damp)
        return y

    # map conc -> inhibition strength (toy)
    conc_to_strength = {
        0.0: 0.00,
        0.1: 0.05,
        0.3: 0.15,
        1.0: 0.35,
        3.0: 0.60,
        10.0: 0.80,
    }

    data = {"Time (h)": times}
    for col_name, conc in cols:
        data[col_name] = inhibited_series(conc_to_strength[conc]).tolist()

    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode("utf-8")

def _find_concentration_col(df: pd.DataFrame) -> str | None:
    """
    Try to find a concentration/dose column in the canonical wide input.
    If none exists, Grofit can still run curve fitting; dose-response will be meaningless (all conc=0).
    """
    candidates = [
        "concentration", "Concentration", "conc", "Conc", "dose", "Dose", "drug_conc", "DrugConc"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive fallback
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def wide_original_to_grofit_tidy(
    wide_original: pd.DataFrame,
    *,
    file_tag: str,
    test_id_col: str = "Test Id",
) -> pd.DataFrame:
    """
    Convert canonical wide input (with prefixed Test Ids) into tidy format required by grofit pipeline:
      test_id, curve_id, concentration, time, y

    IMPORTANT: curve_id stays as the FULL prefixed Test Id (e.g., testSample1_BY4741),
    because classifier outputs use that same ID. We do NOT split IDs for Streamlit.
    """
    if test_id_col not in wide_original.columns:
        raise ValueError(f"Expected '{test_id_col}' in canonical wide input.")

    # Identify time columns from your existing helper parse_time_from_header()
    time_cols = [c for c in wide_original.columns if parse_time_from_header(str(c)) is not None]
    if not time_cols:
        raise ValueError("No time columns detected in wide_original (expected T#.## (h) headers).")

    conc_col = _find_concentration_col(wide_original)

    # Melt to tidy
    id_vars = [test_id_col]
    if conc_col is not None:
        id_vars.append(conc_col)

    tidy = wide_original.melt(
        id_vars=id_vars,
        value_vars=time_cols,
        var_name="_time_label",
        value_name="y",
    )

    # Parse time to float
    tidy["time"] = tidy["_time_label"].map(lambda s: float(parse_time_from_header(str(s))))
    tidy.drop(columns=["_time_label"], inplace=True)

    # Rename and add required columns
    tidy["test_id"] = str(file_tag)
    tidy["curve_id"] = tidy[test_id_col].astype(str)

    if conc_col is None:
        tidy["concentration"] = tidy[test_id_col].astype(str).map(_extract_conc_from_curve_id)
        tidy["concentration"] = pd.to_numeric(tidy["concentration"], errors="coerce").fillna(0.0)
    else:
        tidy["concentration"] = pd.to_numeric(tidy[conc_col], errors="coerce").fillna(0.0)

    tidy["y"] = pd.to_numeric(tidy["y"], errors="coerce")
    tidy = tidy.dropna(subset=["time", "y"])

    return tidy[["test_id", "curve_id", "concentration", "time", "y"]]


# def attach_is_valid_to_tidy(
#     tidy: pd.DataFrame,
#     out_df: pd.DataFrame,
#     *,
#     id_col_predictions: str = "Test Id",
#     label_col: str = "pred_label",
# ) -> pd.DataFrame:
#     """
#     Attach is_valid based on classifier out_df by FULL Test Id match.
#     (No splitting, because your canonical wide input uses add_prefix=True.)
#     """
#     if id_col_predictions not in out_df.columns:
#         raise ValueError(f"Expected '{id_col_predictions}' in classifier output out_df.")
#     if label_col not in out_df.columns:
#         raise ValueError(f"Expected '{label_col}' in classifier output out_df.")

#     # Valid only if label is exactly "Valid"
#     label_map = (
#         out_df[[id_col_predictions, label_col]]
#         .drop_duplicates(subset=[id_col_predictions])
#         .set_index(id_col_predictions)[label_col]
#         .astype(str)
#     )

#     tidy = tidy.copy()
#     tidy["is_valid"] = tidy["curve_id"].map(lambda cid: str(label_map.get(cid, "Invalid")) == "Valid")

#     # Exclude "Unsure" automatically (treat as invalid)
#     # If you want to include it in the future, change this rule.
#     return tidy


def _normalize_label(v: object) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip().lower()
    if s in {"valid", "true", "1"}:
        return "Valid"
    if s in {"invalid", "false", "0"}:
        return "Invalid"
    if s in {"unsure", "unknown"}:
        return "Unsure"
    return str(v).strip()


def _build_classifier_output(
    *,
    wide_df: pd.DataFrame,
    out_df: pd.DataFrame,
    review_df: pd.DataFrame | None,
    manual_review_mode: bool,
    meta_df: pd.DataFrame | None = None,
    processed_wide_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    kwargs = dict(
        wide_original_df=wide_df,
        infer_df=out_df,
        meta_df=meta_df if isinstance(meta_df, pd.DataFrame) else out_df,
        mode="MANUAL" if manual_review_mode else "AUTO",
        review_df=review_df,
    )
    if processed_wide_df is not None:
        try:
            return build_classifier_audit_df(
                **kwargs,
                processed_wide_df=processed_wide_df,
            )
        except TypeError:
            # Backward compatibility: older installed growthqa builds may not
            # yet expose the processed_wide_df argument.
            pass
    return build_classifier_audit_df(**kwargs)


def _build_grofit_input_df(
    *,
    wide_df: pd.DataFrame,
    out_df: pd.DataFrame,
    review_df: pd.DataFrame | None,
    manual_review_mode: bool,
    meta_df: pd.DataFrame | None = None,
    audit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if not isinstance(audit_df, pd.DataFrame):
        audit_df = _build_classifier_output(
            wide_df=wide_df,
            out_df=out_df,
            review_df=review_df,
            manual_review_mode=manual_review_mode,
            meta_df=meta_df,
        )
    return build_grofit_input_artifact(
        wide_original_df=wide_df,
        audit_df=audit_df,
    )


def _build_export_zip(
    *,
    wide_df: pd.DataFrame,
    out_df: pd.DataFrame,
    review_df: pd.DataFrame | None,
    gc_fit: pd.DataFrame,
    gc_boot: pd.DataFrame,
    dr_fit: pd.DataFrame,
    dr_boot: pd.DataFrame,
    proc_wide_df: pd.DataFrame | None,
    grofit_opts: GrofitOptions,
    settings: InferenceSettings,
    mode_label: str,
    file_stem: str,
    predicting_model: str,
    auto_bootstrap_scope: str | None = None,
    auto_preferred_model: str | None = None,
    auto_response_metric: str | None = None,
    auto_dr_bootstrap: str | None = None,
    selected_gc_bootstrap: str | None = None,
    selected_preferred_fit: str | None = None,
    selected_response_metric: str | None = None,
    selected_dr_bootstrap: str | None = None,
    export_label_filter: str = "Valid",
    export_dr_include_unsure: bool = False,
    export_dr_include_invalid: bool = False,
    audit_df: pd.DataFrame | None = None,
    grofit_df: pd.DataFrame | None = None,
    grofit_tidy_all: pd.DataFrame | None = None,
    stage2_config: dict | None = None,
) -> tuple[str, bytes]:
    date_tag = datetime.now().strftime("%m.%d.%y")
    zip_name = f"{mode_label}_{date_tag}_{file_stem}.zip"

    classifier_df = audit_df if isinstance(audit_df, pd.DataFrame) else _build_classifier_output(
        wide_df=wide_df,
        out_df=out_df,
        review_df=review_df,
        manual_review_mode=(mode_label == "MANUAL"),
        processed_wide_df=proc_wide_df,
    )
    grofit_input_df = grofit_df if isinstance(grofit_df, pd.DataFrame) else _build_grofit_input_df(
        wide_df=wide_df,
        out_df=out_df,
        review_df=review_df,
        manual_review_mode=(mode_label == "MANUAL"),
        audit_df=classifier_df,
    )

    # Load feature list from saved manifest if available (for reproducibility in thesis)
    _feature_list = None
    try:
        import json as _json
        _feat_path = Path(MODEL_DIR) / "stage1_features.json"
        if _feat_path.exists():
            _feature_list = _json.loads(_feat_path.read_text(encoding="utf-8"))
    except Exception:
        pass

    grofit_opts_info = dict(grofit_opts.__dict__)
    grofit_opts_info["bootstrap_method"] = _normalize_bootstrap_method(grofit_opts_info.get("bootstrap_method"))

    run_info = {
        "mode": mode_label,
        "timestamp": datetime.now().isoformat(),
        "file_stem": file_stem,
        "predicting_model": predicting_model,
        "grofit_options": grofit_opts_info,
        "inference_settings": settings.__dict__,
        "stage2_thresholds": stage2_config,
        "stage1_feature_list": _feature_list,
        "auto_options": {
            "bootstrap_scope": auto_bootstrap_scope,
            "preferred_model": auto_preferred_model,
            "response_metric": auto_response_metric,
            "dr_bootstrap": auto_dr_bootstrap,
            "export_label_filter": export_label_filter,
            "export_dr_include_unsure": bool(export_dr_include_unsure),
            "export_dr_include_invalid": bool(export_dr_include_invalid),
        }
        if mode_label == "AUTO"
        else None,
        "pipeline_filters": {
            "gc_bootstrap": selected_gc_bootstrap,
            "preferred_fit": selected_preferred_fit,
            "response_metric": selected_response_metric,
            "dr_bootstrap": selected_dr_bootstrap,
            "export_label_filter": export_label_filter,
            "export_dr_include_unsure": bool(export_dr_include_unsure),
            "export_dr_include_invalid": bool(export_dr_include_invalid),
        },
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    }

    def _df_bytes(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8")

    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        allowed_ids: list[str] = []
        mode_upper = str(mode_label or "").strip().upper()
        if mode_upper == "AUTO":
            label_candidates = ["Pred Label", "pred_label", "True Label"]
        else:
            label_candidates = ["True Label", "Pred Label", "pred_label"]
        if isinstance(classifier_df, pd.DataFrame) and not classifier_df.empty and "Test Id" in classifier_df.columns:
            label_col = next((c for c in label_candidates if c in classifier_df.columns), None)
            if label_col is not None:
                labels = classifier_df[label_col].map(_normalize_label)
                f = str(export_label_filter or "Valid").strip().lower()
                if f == "all":
                    mask = labels.isin(["Valid", "Invalid", "Unsure"])
                elif f == "invalid":
                    mask = labels == "Invalid"
                elif f == "unsure":
                    mask = labels == "Unsure"
                else:
                    mask = labels == "Valid"
                allowed_ids = classifier_df.loc[mask, "Test Id"].astype(str).drop_duplicates().tolist()

        gc_fit_out = gc_fit.copy()
        gc_boot_out = gc_boot.copy()
        if allowed_ids and isinstance(gc_fit_out, pd.DataFrame) and not gc_fit_out.empty and "add.id" in gc_fit_out.columns:
            gc_fit_out = gc_fit_out[gc_fit_out["add.id"].astype(str).isin(allowed_ids)].copy()
        elif isinstance(gc_fit_out, pd.DataFrame) and not gc_fit_out.empty and "add.id" in gc_fit_out.columns:
            gc_fit_out = gc_fit_out.iloc[0:0].copy()
        if allowed_ids and isinstance(gc_boot_out, pd.DataFrame) and not gc_boot_out.empty and "add.id" in gc_boot_out.columns:
            gc_boot_out = gc_boot_out[gc_boot_out["add.id"].astype(str).isin(allowed_ids)].copy()
        elif isinstance(gc_boot_out, pd.DataFrame) and not gc_boot_out.empty and "add.id" in gc_boot_out.columns:
            gc_boot_out = gc_boot_out.iloc[0:0].copy()

        # Recompute DR export tables from selected labels:
        # Valid is always included; optionally include Unsure/Invalid.
        dr_fit_out = dr_fit.copy()
        dr_boot_out = dr_boot.copy()
        dr_allowed_ids: list[str] = []
        if isinstance(classifier_df, pd.DataFrame) and not classifier_df.empty and "Test Id" in classifier_df.columns:
            dr_label_col = next((c for c in label_candidates if c in classifier_df.columns), None)
            if dr_label_col is not None:
                _lbl = classifier_df[dr_label_col].map(_normalize_label).astype(str)
                keep_mask = _lbl.eq("Valid")
                if export_dr_include_unsure:
                    keep_mask = keep_mask | _lbl.eq("Unsure")
                if export_dr_include_invalid:
                    keep_mask = keep_mask | _lbl.eq("Invalid")
                dr_allowed_ids = classifier_df.loc[keep_mask, "Test Id"].astype(str).drop_duplicates().tolist()

        if isinstance(grofit_tidy_all, pd.DataFrame) and not grofit_tidy_all.empty:
            dr_curves = grofit_tidy_all
            if dr_allowed_ids:
                dr_curves = dr_curves[dr_curves["curve_id"].astype(str).isin(dr_allowed_ids)].copy()
            else:
                dr_curves = dr_curves.iloc[0:0].copy()
            if not dr_curves.empty:
                try:
                    dr_res = run_grofit_pipeline(
                        curves_df=dr_curves,
                        response_var=grofit_opts.response_var,
                        have_atleast=grofit_opts.have_atleast,
                        gc_boot_B=0,
                        dr_boot_B=grofit_opts.dr_boot_B,
                        spline_auto_cv=grofit_opts.spline_auto_cv,
                        spline_s=grofit_opts.spline_s,
                        smooth_gc=grofit_opts.smooth_gc,
                        smooth_dr=grofit_opts.smooth_dr,
                        dr_x_transform=grofit_opts.dr_x_transform,
                        dr_y_transform=grofit_opts.dr_y_transform,
                        dr_s=grofit_opts.dr_s,
                        fit_opt=grofit_opts.fit_opt,
                        bootstrap_method=_normalize_bootstrap_method(grofit_opts.bootstrap_method),
                        validity_col="__all__",
                        random_state=42,
                        export_dir=None,
                    )
                    dr_fit_out = dr_res.get("dr_fit", pd.DataFrame())
                    dr_boot_out = dr_res.get("dr_boot", pd.DataFrame())
                except Exception:
                    dr_fit_out = pd.DataFrame()
                    dr_boot_out = pd.DataFrame()
            else:
                dr_fit_out = pd.DataFrame()
                dr_boot_out = pd.DataFrame()

        zf.writestr("gcFit.csv", _df_bytes(gc_fit_out))
        if isinstance(gc_boot_out, pd.DataFrame) and not gc_boot_out.empty:
            zf.writestr("gcBoot.csv", _df_bytes(gc_boot_out))
        if isinstance(dr_fit_out, pd.DataFrame) and not dr_fit_out.empty:
            zf.writestr("drFit.csv", _df_bytes(dr_fit_out))
        if isinstance(dr_boot_out, pd.DataFrame) and not dr_boot_out.empty:
            zf.writestr("drBoot.csv", _df_bytes(dr_boot_out))
        # Classifier performance metrics for thesis (if trained in this session)
        try:
            _perf_path = Path(MODEL_DIR) / "classifier_performance_latest.csv"
            if _perf_path.exists():
                zf.writestr("classifier_performance.csv", _perf_path.read_bytes())
        except Exception:
            pass

        n_plot_files = 0

        valid_ids: list[str] = allowed_ids
        if valid_ids:
            fit_source_wide = proc_wide_df if isinstance(proc_wide_df, pd.DataFrame) and not proc_wide_df.empty else wide_df
            curves_tidy = (
                grofit_tidy_all.copy()
                if isinstance(grofit_tidy_all, pd.DataFrame) and not grofit_tidy_all.empty
                else pd.DataFrame()
            )
            if curves_tidy.empty:
                try:
                    curves_tidy = wide_original_to_grofit_tidy(fit_source_wide, file_tag=file_stem)
                except Exception:
                    curves_tidy = pd.DataFrame()

            labels_for_payload = pd.DataFrame({"Test Id": valid_ids})
            if isinstance(classifier_df, pd.DataFrame) and not classifier_df.empty and "Test Id" in classifier_df.columns:
                label_map = classifier_df.drop_duplicates(subset=["Test Id"]).set_index("Test Id")
                if "Pred Label" in label_map.columns:
                    labels_for_payload["pred_label"] = labels_for_payload["Test Id"].map(label_map["Pred Label"]).map(_normalize_label)
                else:
                    labels_for_payload["pred_label"] = "Valid"
                if "True Label" in label_map.columns:
                    labels_for_payload["final_label"] = labels_for_payload["Test Id"].map(label_map["True Label"]).map(_normalize_label)
                else:
                    labels_for_payload["final_label"] = labels_for_payload["pred_label"]
                labels_for_payload["Reviewed"] = False
            else:
                labels_for_payload["pred_label"] = "Valid"
                labels_for_payload["final_label"] = "Valid"
                labels_for_payload["Reviewed"] = False

            if isinstance(curves_tidy, pd.DataFrame) and not curves_tidy.empty:
                payloads = build_curve_payloads(
                    curves_df=curves_tidy,
                    raw_wide=wide_df,
                    proc_wide=fit_source_wide,
                    labels_df=labels_for_payload,
                    gc_boot=gc_boot if isinstance(gc_boot, pd.DataFrame) else None,
                    spline_s=grofit_opts.spline_s,
                    smooth_gc=grofit_opts.smooth_gc,
                    spline_auto_cv=grofit_opts.spline_auto_cv,
                    include_bootstrap=bool(isinstance(gc_boot, pd.DataFrame) and not gc_boot.empty),
                    test_id=file_stem,
                    curve_ids=valid_ids,
                )

                for tid in valid_ids:
                    payload = payloads.get(str(tid))
                    if not payload:
                        continue
                    plot_payload = dict(payload)
                    # Export requirement: fit-only view (no raw/processed traces).
                    plot_payload["t_raw"] = np.array([], dtype=float)
                    plot_payload["y_raw"] = np.array([], dtype=float)
                    plot_payload["t_proc"] = np.array([], dtype=float)
                    plot_payload["y_proc"] = np.array([], dtype=float)

                    fig = make_overlay_plot_payload(
                        plot_payload,
                        title=str(tid),
                        show_spline=True,
                        show_model=True,
                        show_bootstrap=bool(plot_payload.get("bootstrap", {}).get("ran", False)),
                    )
                    safe_tid = re.sub(r"[^A-Za-z0-9._-]+", "_", str(tid)).strip("_") or "curve"
                    zf.writestr(f"plots/{safe_tid}.html", fig.to_html(full_html=True, include_plotlyjs="cdn"))
                    n_plot_files += 1

        if isinstance(dr_fit_out, pd.DataFrame) and not dr_fit_out.empty and isinstance(gc_fit_out, pd.DataFrame) and not gc_fit_out.empty:
            labels_for_dr = pd.DataFrame()
            if isinstance(classifier_df, pd.DataFrame) and not classifier_df.empty and "Test Id" in classifier_df.columns:
                labels_for_dr["Test Id"] = classifier_df["Test Id"].astype(str)
                if "True Label" in classifier_df.columns:
                    labels_for_dr["final_label"] = classifier_df["True Label"].map(_normalize_label)
                elif "Pred Label" in classifier_df.columns:
                    labels_for_dr["final_label"] = classifier_df["Pred Label"].map(_normalize_label)
                else:
                    labels_for_dr["final_label"] = ""
                labels_for_dr["pred_label"] = labels_for_dr["final_label"]
                labels_for_dr["Reviewed"] = False

            dr_test_id = None
            if "name" in dr_fit_out.columns:
                dr_names = dr_fit_out["name"].dropna().astype(str).tolist()
                dr_test_id = dr_names[0] if dr_names else None

            if not labels_for_dr.empty:
                dr_payload = build_dr_payload(
                    gc_fit=gc_fit_out,
                    labels_df=labels_for_dr,
                    dr_boot=dr_boot_out if isinstance(dr_boot_out, pd.DataFrame) else None,
                    test_id=dr_test_id,
                    response_metric=str(grofit_opts.response_var),
                    label_source="final",
                    include_unsure=bool(export_dr_include_unsure),
                    include_invalid=bool(export_dr_include_invalid),
                    dr_s=grofit_opts.dr_s,
                    smooth_dr=grofit_opts.smooth_dr,
                    dr_x_transform=grofit_opts.dr_x_transform,
                    dr_y_transform=grofit_opts.dr_y_transform,
                    show_bootstrap=bool(isinstance(dr_boot_out, pd.DataFrame) and not dr_boot_out.empty),
                )
                dr_fig = make_dr_plot(dr_payload, show_bootstrap=bool(isinstance(dr_boot_out, pd.DataFrame) and not dr_boot_out.empty))
                zf.writestr("plots/Dose_Response.html", dr_fig.to_html(full_html=True, include_plotlyjs="cdn"))
                n_plot_files += 1

        if n_plot_files == 0:
            zf.writestr("plots/README.txt", "No plot assets could be generated for this run.")

    return zip_name, bio.getvalue()


def _grofit_tidy_to_wide_for_download(grofit_tidy_all: pd.DataFrame) -> pd.DataFrame:
    """
    Convert grofit tidy (long) to wide format with one row per curve_id and time columns as headers.
    """
    if grofit_tidy_all is None or grofit_tidy_all.empty:
        return pd.DataFrame()

    d = grofit_tidy_all.copy()
    d["time"] = pd.to_numeric(d["time"], errors="coerce")
    d["y"] = pd.to_numeric(d["y"], errors="coerce")
    d = d.dropna(subset=["time"])
    d["time_col"] = d["time"].map(lambda t: f"T{float(t):.2f} (h)")

    wide = (
        d.pivot_table(
            index="curve_id",
            columns="time_col",
            values="y",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"curve_id": "Test Id"})
    )

    time_cols = [c for c in wide.columns if c.startswith("T") and "(h)" in c]
    time_cols = sorted(time_cols, key=lambda c: parse_time_from_header(c) if parse_time_from_header(c) is not None else float("inf"))
    meta_cols = [c for c in ["test_id", "concentration", "is_valid_true", "pred_label", "final_label", "pred_confidence"] if c in d.columns]
    if meta_cols:
        meta = d[["curve_id"] + meta_cols].drop_duplicates(subset=["curve_id"]).rename(columns={"curve_id": "Test Id"})
        wide = meta.merge(wide, on="Test Id", how="right")
        ordered = ["Test Id"] + [c for c in meta_cols if c in wide.columns] + [c for c in time_cols if c in wide.columns]
        wide = wide[ordered]
    else:
        wide = wide[["Test Id"] + [c for c in time_cols if c in wide.columns]]

    return wide


def render_results(results: dict):
    final_merged = results["final_merged"]
    out_df = results["out_df"]
    settings = results["settings"]
    time_cols_final = results["time_cols_final"]
    file_stem = results["file_stem"]
    wide_original = results.get("wide_original")
    time_cols_original = results.get("time_cols_original", [])
    predicting_model = results.get("predicting_model", "Unknown")
    review_df = results.get("review_df")
    manual_review_mode = results.get("manual_review_mode", False)
    grofit_opts = results.get("grofit_opts", st.session_state.get("grofit_opts", GrofitOptions()))
    grofit_opts.bootstrap_method = _normalize_bootstrap_method(getattr(grofit_opts, "bootstrap_method", None))
    meta_df = results.get("meta_df", out_df)

    if review_df is not None:
        if st.session_state.get("review_df") is None:
            st.session_state["review_df"] = review_df.copy()
        review_df = st.session_state.get("review_df", review_df)

    gc_fit = results.get("gc_fit", pd.DataFrame())
    dr_fit = results.get("dr_fit", pd.DataFrame())
    gc_boot = results.get("gc_boot", pd.DataFrame())
    dr_boot = results.get("dr_boot", pd.DataFrame())
    gc_audit = results.get("gc_audit", pd.DataFrame())
    dr_audit = results.get("dr_audit", pd.DataFrame())
    grofit_tidy_all = results.get("grofit_tidy_all", pd.DataFrame())
    zip_bytes = results.get("zip_bytes", b"")
    grofit_ran = results.get("grofit_ran", False)
    wide_for_artifacts = wide_original if isinstance(wide_original, pd.DataFrame) else final_merged
    audit_df = results.get("audit_df")
    if not isinstance(audit_df, pd.DataFrame) or audit_df.empty:
        audit_df = _build_classifier_output(
            wide_df=wide_for_artifacts,
            out_df=out_df,
            review_df=review_df if manual_review_mode else None,
            manual_review_mode=manual_review_mode,
            meta_df=meta_df if isinstance(meta_df, pd.DataFrame) else out_df,
            processed_wide_df=final_merged if isinstance(final_merged, pd.DataFrame) else None,
        )
    grofit_df = results.get("grofit_df")
    if not isinstance(grofit_df, pd.DataFrame) or grofit_df.empty:
        grofit_df = _build_grofit_input_df(
            wide_df=wide_for_artifacts,
            out_df=out_df,
            review_df=review_df if manual_review_mode else None,
            manual_review_mode=manual_review_mode,
            meta_df=meta_df if isinstance(meta_df, pd.DataFrame) else out_df,
            audit_df=audit_df,
        )
    results["audit_df"] = audit_df
    results["grofit_df"] = grofit_df
    st.session_state["last_run_results"] = results

    def _get_concentration_map() -> dict:
        if wide_original is not None and "Concentration" in wide_original.columns:
            return wide_original.set_index("Test Id")["Concentration"].to_dict()
        if "Concentration" in final_merged.columns:
            return final_merged.set_index("Test Id")["Concentration"].to_dict()
        if review_df is not None and "Concentration" in review_df.columns:
            return review_df.set_index("Test Id")["Concentration"].to_dict()
        return {tid: _extract_conc_from_curve_id(tid) for tid in out_df["Test Id"].astype(str).tolist()}

    conc_map = _get_concentration_map()

    def _label_with_conc(tid: str) -> str:
        conc = conc_map.get(tid, "")
        conc_label = "" if pd.isna(conc) else str(conc)
        return f"{tid} + {conc_label}" if conc_label != "" else tid

    if manual_review_mode and isinstance(review_df, pd.DataFrame) and not review_df.empty:
        curve_df = review_df.copy()
        merge_cols = [
            c
            for c in [
                "pred_label",
                "Pred Label",
                "pred_confidence",
                "Pred Confidence",
                "confidence_valid",
                "confidence_invalid",
                "too_sparse",
                "low_resolution",
                "had_outliers",
            ]
            if c in out_df.columns
        ]
        if merge_cols:
            curve_df = curve_df.merge(out_df[["Test Id"] + merge_cols], on="Test Id", how="left")
    else:
        curve_df = out_df.copy()
        if "Concentration" not in curve_df.columns:
            curve_df["Concentration"] = curve_df["Test Id"].map(conc_map)

    pred_col = next(
        (
            c
            for c in ["pred_label", "Pred Label", "final_label", "true_label", "Predicted S1 Label"]
            if c in curve_df.columns
        ),
        None,
    )
    if pred_col is None:
        pred_col = "_pred_label_fallback"
        curve_df[pred_col] = ""
    curve_df["_filter_label"] = curve_df[pred_col].map(_normalize_label)

    st.markdown("---")
    st.markdown(f"### {'MANUAL MODE' if manual_review_mode else 'AUTO MODE'}")
    # st.markdown("#### Summary of Curves")

    total_curves = int(len(out_df))
    final_norm = out_df["final_label"].astype(str).str.upper() if "final_label" in out_df.columns else pd.Series([], dtype=str)
    valid_count = int((final_norm == "VALID").sum())
    invalid_count = int((final_norm == "INVALID").sum())
    unsure_count = int((final_norm == "UNSURE").sum()) if "final_label" in out_df.columns else 0

    reviewed_count = 0
    correct_count = 0
    if manual_review_mode and isinstance(review_df, pd.DataFrame) and not review_df.empty:
        reviewed_count = int(review_df["Reviewed"].sum())
        correct_count = int((review_df["Reviewed"] & (review_df["is_valid_true"] == review_df["is_valid_pred"])).sum())
        incorrect_count = int((review_df["Reviewed"] & (review_df["is_valid_true"] != review_df["is_valid_pred"])).sum())
        reviewed_acc = (correct_count / reviewed_count) if reviewed_count > 0 else 0.0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("No. of curves", total_curves)
        c2.metric("Valid", valid_count)
        c3.metric("Invalid", invalid_count)
        c4.metric("Unsure", unsure_count)
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("No. of curves", total_curves)
        c2.metric("Valid", valid_count)
        c3.metric("Invalid", invalid_count)
        c4.metric("Unsure", unsure_count)

    st.markdown("---")
    st.markdown("#### Review Section")

    # --- Right panel header numbers (keep same logic) ---
    total_reviewed_label = f"{reviewed_count}/{total_curves}"
    correct_reviewed_label = f"{correct_count}/{reviewed_count}" if reviewed_count > 0 else "0/0"

    # ============================
    # ROW 1: Dropdowns (left) + Metrics header (right)
    # ============================
    hdr_left, hdr_right = st.columns([2.2, 1.0], gap="large")

    with hdr_left:
        filter_col, select_col = st.columns([1, 2])

        with filter_col:
            filter_choice = st.selectbox(
                "Filter",
                options=["All", "Valid", "Invalid", "Unsure"],
                key="curve_filter",
            )

        # Build filtered options list (same logic as before)
        if filter_choice != "All":
            filtered = curve_df[curve_df["_filter_label"] == filter_choice].copy()
        else:
            filtered = curve_df.copy()

        base_order = (
            wide_original["Test Id"].astype(str).tolist()
            if wide_original is not None and "Test Id" in wide_original.columns
            else curve_df["Test Id"].astype(str).tolist()
        )
        allowed = set(filtered["Test Id"].astype(str).tolist())
        options = [tid for tid in base_order if tid in allowed]
        if not options:
            st.warning("No curves match the current filter.")
            return

        pending = st.session_state.pop("pending_curve_select", None)
        if pending in options:
            st.session_state["curve_select"] = pending
            st.session_state["selected_test_id"] = pending

        if st.session_state.get("selected_test_id") not in options:
            st.session_state["selected_test_id"] = options[0]

        with select_col:
            current_sel = st.session_state.get("curve_select", st.session_state.get("selected_test_id"))
            index_val = options.index(str(current_sel)) if str(current_sel) in options else 0
            chosen = st.selectbox(
                "Select Test Id + Conc",
                options,
                index=index_val,
                key="curve_select",
                format_func=_label_with_conc,
            )

        st.session_state["selected_test_id"] = chosen

    with hdr_right:
        # Align header + subheader with the left dropdown title + dropdown bar
        st.markdown("<div class='metrics-header'><strong>Metrics</strong></div>", unsafe_allow_html=True)
        if manual_review_mode:
            st.markdown(
                f"<div class='metrics-subheader'>Total Reviewed: {total_reviewed_label} | "
                f"Correctly Predicted: {correct_reviewed_label}</div>",
                unsafe_allow_html=True,
            )

    # helper for navigation rerun (same as before)
    def _nav_set(new_idx: int):
        new_val = options[new_idx]
        st.session_state["selected_test_id"] = new_val
        st.session_state["pending_curve_select"] = new_val
        st.rerun()

    row = curve_df.loc[curve_df["Test Id"].astype(str) == str(chosen)].iloc[0]
    out_row = out_df.loc[out_df["Test Id"].astype(str) == str(chosen)].iloc[0] if not out_df.empty else row
    pred_label = _normalize_label(out_row.get("Pred Label", row.get(pred_col, "")))
    if not pred_label:
        bool_val = row.get("is_valid_pred")
        if isinstance(bool_val, (bool, np.bool_)):
            pred_label = "Valid" if bool_val else "Invalid"
        elif isinstance(bool_val, (int, float, np.integer, np.floating)) and pd.notna(bool_val):
            pred_label = "Valid" if int(bool_val) == 1 else "Invalid"
    pred_conf_display = pd.to_numeric(
        pd.Series([out_row.get("Pred Confidence", out_row.get("pred_confidence", np.nan))]),
        errors="coerce",
    ).iloc[0]

    if manual_review_mode:
        final_label = _normalize_label(row.get("true_label", row.get("final_label", pred_label)))
        if not final_label:
            final_label_bool = bool(row.get("is_valid_true", _label_is_valid(pred_label)))
            final_label = "Valid" if final_label_bool else "Invalid"
    else:
        final_label = pred_label

    labels_df = review_df if (manual_review_mode and isinstance(review_df, pd.DataFrame)) else out_df.copy()
    if isinstance(labels_df, pd.DataFrame):
        if manual_review_mode and "true_label" in labels_df.columns:
            labels_df = labels_df.copy()
            labels_df["final_label"] = labels_df["true_label"]
        if "final_label" not in labels_df.columns and "pred_label" in labels_df.columns:
            labels_df = labels_df.copy()
            labels_df["final_label"] = labels_df["pred_label"]
        if "Reviewed" not in labels_df.columns:
            labels_df = labels_df.copy()
            labels_df["Reviewed"] = False

    zip_ready = bool(grofit_ran and zip_bytes)
    metric_font = "1.1rem"
    fit_row = None
    active_test_id = str(file_stem)
    if isinstance(gc_fit, pd.DataFrame) and not gc_fit.empty and "add.id" in gc_fit.columns:
        fit_match = gc_fit[gc_fit["add.id"].astype(str) == str(chosen)].copy()
        if not fit_match.empty:
            if "test.id" in fit_match.columns:
                preferred_test_ids = [str(file_stem)]
                if "FileName" in out_row and str(out_row.get("FileName", "")).strip():
                    preferred_test_ids.insert(0, str(out_row.get("FileName")).strip())
                picked = None
                for tid in preferred_test_ids:
                    exact = fit_match[fit_match["test.id"].astype(str) == tid]
                    if not exact.empty:
                        picked = exact.iloc[0]
                        active_test_id = tid
                        break
                if picked is None:
                    picked = fit_match.iloc[0]
                    active_test_id = str(picked.get("test.id", file_stem))
                fit_row = picked
            else:
                fit_row = fit_match.iloc[0]

    boot_row = None
    if isinstance(gc_boot, pd.DataFrame) and not gc_boot.empty and "add.id" in gc_boot.columns:
        boot_match = gc_boot[gc_boot["add.id"].astype(str) == str(chosen)].copy()
        if not boot_match.empty:
            if "test.id" in boot_match.columns:
                exact_boot = boot_match[boot_match["test.id"].astype(str) == str(active_test_id)]
                boot_row = exact_boot.iloc[0] if not exact_boot.empty else boot_match.iloc[0]
            else:
                boot_row = boot_match.iloc[0]

    def _fmt_val(val: object) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "NA"
        if isinstance(val, bool):
            return "True" if val else "False"
        return str(val)

    def _fmt_metric(val: object) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "NA"
        if isinstance(val, (int, float, np.floating)):
            return f"{float(val):.4f}"
        return str(val)

    def _render_row(label: str, value: str | None = None) -> None:
        r_label, r_value = st.columns([1.2, 1.8], gap="small")
        r_label.markdown(
            f"<div style='font-size:{metric_font};font-weight:600;'>{label}</div>",
            unsafe_allow_html=True,
        )
        if value is None:
            r_value.markdown(f"<div style='font-size:{metric_font};'>NA</div>", unsafe_allow_html=True)
        else:
            r_value.markdown(f"<div style='font-size:{metric_font};'>{value}</div>", unsafe_allow_html=True)

    def _render_select_row(label: str, options_list: list[str], index_val: int, key_val: str) -> str:
        r_label, r_value, r_spacer = st.columns([1.2, 0.78, 1.02], gap="small")
        r_label.markdown(
            f"<div style='font-size:{metric_font};font-weight:600;'>{label}</div>",
            unsafe_allow_html=True,
        )
        sel = r_value.selectbox(
            label,
            options=options_list,
            index=index_val,
            key=key_val,
            label_visibility="collapsed",
        )
        r_spacer.markdown("")
        return sel

    @st.dialog("Configure & Run Grofit")
    def _configure_and_run_grofit_dialog():
        st.write("Review your pipeline settings before executing Grofit.")
        c1, c2 = st.columns(2)
        with c1:
            gc_bootstrap_targets = st.selectbox(
                "GC Bootstrap",
                options=["False", "True"],
                index=["False", "True"].index(
                    st.session_state.get("exit_gc_bootstrap", "False")
                    if st.session_state.get("exit_gc_bootstrap", "False") in {"False", "True"}
                    else "False"
                ),
            )
            response_metric = st.selectbox(
                "Response Metric",
                options=["mu", "A", "lag", "integral"],
                index=["mu", "A", "lag", "integral"].index(
                    st.session_state.get("exit_response_metric", "mu")
                    if st.session_state.get("exit_response_metric", "mu") in {"mu", "A", "lag", "integral"}
                    else "mu"
                ),
            )
        with c2:
            preferred_model = st.selectbox(
                "Preferred Fit",
                options=["Best Model", "Spline", "Parametric"],
                index=["Best Model", "Spline", "Parametric"].index(
                    st.session_state.get("exit_preferred_model", "Best Model")
                    if st.session_state.get("exit_preferred_model", "Best Model") in {"Best Model", "Spline", "Parametric"}
                    else "Best Model"
                ),
            )
            dr_bootstrap = st.selectbox(
                "DR Bootstrap",
                options=["True", "False"],
                index=["True", "False"].index(
                    st.session_state.get("exit_dr_bootstrap", "True")
                    if st.session_state.get("exit_dr_bootstrap", "True") in {"True", "False"}
                    else "True"
                ),
            )
        st.divider()
        if st.button("Confirm & Run Pipeline", type="primary", use_container_width=True):
            st.session_state["exit_gc_bootstrap"] = gc_bootstrap_targets
            st.session_state["exit_response_metric"] = response_metric
            st.session_state["exit_preferred_model"] = preferred_model
            st.session_state["exit_dr_bootstrap"] = dr_bootstrap
            st.session_state["trigger_grofit_run"] = True
            st.rerun()

    curve_payload = None
    if (
        isinstance(grofit_tidy_all, pd.DataFrame)
        and not grofit_tidy_all.empty
        and isinstance(labels_df, pd.DataFrame)
        and wide_original is not None
        and not final_merged.empty
    ):
        payloads = build_curve_payloads(
            curves_df=grofit_tidy_all,
            raw_wide=wide_original,
            proc_wide=final_merged,
            labels_df=labels_df,
            gc_boot=gc_boot,
            spline_s=grofit_opts.spline_s,
            smooth_gc=grofit_opts.smooth_gc,
            spline_auto_cv=grofit_opts.spline_auto_cv,
            include_bootstrap=bool(isinstance(gc_boot, pd.DataFrame) and not gc_boot.empty),
            test_id=active_test_id,
            curve_ids=[str(chosen)],
        )
        curve_payload = payloads.get(str(chosen))


    tab_curve, tab_dr = st.tabs(["Single Curve Review", "Dose Response Analysis"])

    with tab_curve:
        left, right = st.columns([2.2, 1.0], gap="large")

        with left:
            st.markdown('<div class="curve-title">Curve Viewer</div>', unsafe_allow_html=True)
            # 1. Determine if Bootstrap bands are available (from bulk export or on-demand session state)
            bulk_boot_available = bool(isinstance(gc_boot, pd.DataFrame) and not gc_boot.empty)
            session_boot_key = f"boot_arrays_{chosen}"
            boot_arrays_ready = bulk_boot_available or (session_boot_key in st.session_state)

            # 2. Render Overlay Pills
            overlay_options = ["Spline Fit", "Parametric Model"]
            if manual_review_mode:
                overlay_options.append("Bootstrap CI")

            overlay_selected = _st_pills_multi(
                " ",
                options=overlay_options,
                default=["Spline Fit", "Parametric Model"],
                key=f"curve_overlay_pills_{chosen}",
            )

            # 3. Inject Session State Bootstrap into Payload if needed
            if not bulk_boot_available and (session_boot_key in st.session_state) and curve_payload:
                y_lo, y_hi = st.session_state[session_boot_key]
                curve_payload.setdefault("bootstrap", {})
                curve_payload["bootstrap"]["ran"] = True
                curve_payload["bootstrap"]["y_hat_q025"] = y_lo
                curve_payload["bootstrap"]["y_hat_q975"] = y_hi

            # 4. On-demand Bootstrap generation: in MANUAL mode, selecting Bootstrap CI triggers computation.
            wants_bootstrap = manual_review_mode and ("Bootstrap CI" in overlay_selected)
            if wants_bootstrap and (not boot_arrays_ready) and curve_payload:
                with st.spinner("Running fast in-memory bootstrap for UI..."):
                    spline_pl = curve_payload.get("spline", {})
                    # Use the tidy fit arrays (same data/scale as the orange spline),
                    # the locked lambda from _spline_payload, and the exact t_grid so
                    # the band aligns pixel-perfectly with the displayed orange curve.
                    t_for_boot = curve_payload.get("t_fit")
                    y_for_boot = curve_payload.get("y_fit")
                    lam_s      = spline_pl.get("lam")           # set by _spline_payload
                    t_grid     = spline_pl.get("t_grid")
                    if (
                        t_for_boot is not None and y_for_boot is not None
                        and t_grid is not None and len(t_for_boot) >= 6
                    ):
                        y_lo, y_hi = _generate_fast_bootstrap_bands(
                            t_for_boot, y_for_boot, lam_s, t_grid=t_grid
                        )
                        if y_lo is not None:
                            st.session_state[session_boot_key] = (y_lo, y_hi)
                    st.rerun()

            show_spline = "Spline Fit" in overlay_selected
            show_model = "Parametric Model" in overlay_selected
            show_bootstrap = manual_review_mode and ("Bootstrap CI" in overlay_selected) and boot_arrays_ready

            if wide_original is not None and not final_merged.empty:
                raw_row = wide_original.loc[wide_original["Test Id"].astype(str) == str(chosen)]
                proc_row = final_merged.loc[final_merged["Test Id"].astype(str) == str(chosen)]
                if not raw_row.empty and not proc_row.empty:
                    if curve_payload:
                        fig = make_overlay_plot_payload(
                            curve_payload,
                            title=f"{chosen}",
                            show_spline=show_spline,
                            show_model=show_model,
                            show_bootstrap=show_bootstrap,
                        )
                    else:
                        fig = make_overlay_plot(
                            raw_row.iloc[0],
                            time_cols_original,
                            proc_row.iloc[0],
                            time_cols_final,
                            title=f"{chosen}",
                            input_is_raw=settings.input_is_raw,
                            global_blank=settings.global_blank,
                        )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)

                    nav_left, nav_right = st.columns(2)
                    if nav_left.button("Previous Curve", key="prev_curve_overlay", use_container_width=True):
                        idx = options.index(str(chosen))
                        _nav_set((idx - 1) % len(options))
                    if nav_right.button("Next Curve", key="next_curve_overlay", use_container_width=True):
                        idx = options.index(str(chosen))
                        _nav_set((idx + 1) % len(options))

                else:
                    st.warning("Could not find curve data for overlay plot.")
            else:
                st.warning("No time columns found for plotting.")

        with right:
            st.markdown('<div class="metrics-panel">', unsafe_allow_html=True)

            def _fmt_ci_metric(val: object, ci: list | None) -> str:
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return "NA"
                if isinstance(val, (int, float, np.floating)):
                    base = f"{float(val):.4f}"
                else:
                    base = str(val)
                if ci and len(ci) == 2:
                    lo = pd.to_numeric(pd.Series([ci[0]]), errors="coerce").iloc[0]
                    hi = pd.to_numeric(pd.Series([ci[1]]), errors="coerce").iloc[0]
                    if np.isfinite(lo) and np.isfinite(hi):
                        return f"{base} ({float(lo):.4f}-{float(hi):.4f})"
                return base

            with st.container(border=True):
                label_color = {"valid": "#2e7d32", "invalid": "#c62828", "unsure": "#ef6c00"}.get(pred_label.strip().lower(), "#5f4337")
                st.markdown(
                    (
                        "<div style='border:1px solid #8f8f8f;border-radius:8px;padding:8px 10px;margin:4px 0 10px 0;'>"
                        "<div style='font-size:1.0rem;font-weight:500;'>Predicted Label</div>"
                        f"<div style='font-size:1.2rem;font-weight:700;color:{label_color};'>{_fmt_val(pred_label)}</div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

                if manual_review_mode and isinstance(review_df, pd.DataFrame) and not review_df.empty:
                    curve_key = row.get("CurveKey")
                    if not curve_key:
                        curve_key = _make_curve_key(str(row["Test Id"]), row.get("Concentration"))
                    label_key = f"true_label_{curve_key}"
                    reviewed_key = f"reviewed_{curve_key}"

                    if label_key not in st.session_state:
                        true_norm = _normalize_label(row.get("true_label", row.get("final_label", pred_label)))
                        st.session_state[label_key] = true_norm if true_norm in {"Valid", "Invalid", "Unsure"} else "Unsure"

                    if reviewed_key not in st.session_state:
                        st.session_state[reviewed_key] = "True" if bool(row.get("Reviewed", False)) else "False"
                    elif isinstance(st.session_state[reviewed_key], bool):
                        st.session_state[reviewed_key] = "True" if st.session_state[reviewed_key] else "False"

                    new_final = _render_select_row(
                        "True Label",
                        ["Valid", "Invalid", "Unsure"],
                        ["Valid", "Invalid", "Unsure"].index(st.session_state[label_key]),
                        label_key,
                    )
                    reviewed_val = _render_select_row(
                        "Reviewed",
                        ["False", "True"],
                        1 if st.session_state[reviewed_key] == "True" else 0,
                        reviewed_key,
                    )
                    reviewed_bool = reviewed_val == "True"
                    if new_final != _normalize_label(row.get("true_label", row.get("final_label", pred_label))) or reviewed_bool != bool(row.get("Reviewed", False)):
                        review_df.loc[review_df["CurveKey"] == curve_key, "true_label"] = new_final
                        review_df.loc[review_df["CurveKey"] == curve_key, "is_valid_true"] = (new_final == "Valid")
                        review_df.loc[review_df["CurveKey"] == curve_key, "is_valid_final"] = review_df.loc[review_df["CurveKey"] == curve_key, "is_valid_true"]
                        review_df.loc[review_df["CurveKey"] == curve_key, "Reviewed"] = reviewed_bool
                        st.session_state["review_df"] = review_df
                        updated_audit = _build_classifier_output(
                            wide_df=wide_for_artifacts,
                            out_df=out_df,
                            review_df=review_df,
                            manual_review_mode=True,
                            meta_df=meta_df if isinstance(meta_df, pd.DataFrame) else out_df,
                            processed_wide_df=final_merged if isinstance(final_merged, pd.DataFrame) else None,
                        )
                        updated_grofit = _build_grofit_input_df(
                            wide_df=wide_for_artifacts,
                            out_df=out_df,
                            review_df=review_df,
                            manual_review_mode=True,
                            meta_df=meta_df if isinstance(meta_df, pd.DataFrame) else out_df,
                            audit_df=updated_audit,
                        )
                        results["audit_df"] = updated_audit
                        results["grofit_df"] = updated_grofit
                        results["review_df"] = review_df
                        st.session_state["last_run_results"] = results
                        st.rerun()

                pred_conf_text = "" if pred_label.strip().lower() == "unsure" else _fmt_metric(pred_conf_display)
                _render_row(f"Confidence ({pred_label})", pred_conf_text)

            if curve_payload:
                spline_params = curve_payload.get("spline", {}).get("params", {})
                parametric = curve_payload.get("parametric", {})
                param_params = parametric.get("params", {})
                boot_ci = curve_payload.get("bootstrap", {}).get("ci", {})

                metric_rows = [
                    {"Metric": "mu", "Spline": _fmt_ci_metric(spline_params.get("mu"), boot_ci.get("mu")), "Model": _fmt_metric(param_params.get("mu"))},
                    {"Metric": "lambda", "Spline": _fmt_ci_metric(spline_params.get("lambda"), boot_ci.get("lambda")), "Model": _fmt_metric(param_params.get("lambda"))},
                    {"Metric": "A", "Spline": _fmt_ci_metric(spline_params.get("A"), boot_ci.get("A")), "Model": _fmt_metric(param_params.get("A"))},
                    {"Metric": "Integral", "Spline": _fmt_ci_metric(spline_params.get("integral"), boot_ci.get("integral")), "Model": _fmt_metric(param_params.get("integral"))},
                ]
                params_df = pd.DataFrame(metric_rows).rename(columns={"Model": str(parametric.get("model_name") or "Model")})
                st.dataframe(params_df, hide_index=True, use_container_width=True)
            if manual_review_mode and st.button("Run Final Pipeline & Export", type="primary", use_container_width=True, key="run_pipeline_dialog_btn"):
                _configure_and_run_grofit_dialog()

            _render_row("Too Sparse", _fmt_val(bool(out_row.get("too_sparse")) if "too_sparse" in out_row else "NA"))
            _render_row("Low Resolution", _fmt_val(bool(out_row.get("low_resolution")) if "low_resolution" in out_row else "NA"))
            _render_row(
                "Blank subtraction mode",
                _fmt_val("RAW (applied)" if settings.input_is_raw else "ALREADY BLANK SUBTRACTED (so not applied)"),
            )

            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("About the Processed plot ...", expanded=False):
                st.markdown(
                    "- Timepoints are extracted from column headers like `T1.0 (h)` and sorted numerically.\n"
                    "- Values come from the preprocessed table after interpolation, smoothing, and blank handling.\n"
                    "- Y-axis is Relative OD.\n"
                    "- Plot shows Raw vs Processed curves."
                )
            st.markdown("---")
            st.markdown("#### Debug Downloads")
            zip_name = results.get("zip_name") or f"outputs_{file_stem}.zip"
            st.download_button(
                "Download Results.zip",
                data=zip_bytes,
                file_name=zip_name,
                mime="application/zip",
                disabled=not zip_ready,
                use_container_width=True,
                help="This contains gc_fit, gc_boot, dr_fit, dr_boot, and a plots folder")
            audit_run_info = {
                "mode": "MANUAL" if manual_review_mode else "AUTO",
                "timestamp": datetime.now().isoformat(),
                "file_stem": file_stem,
                "predicting_model": predicting_model,
                "grofit_options": grofit_opts.__dict__ if grofit_opts is not None else None,
                "settings": settings.__dict__ if settings is not None else None,
                "pipeline_filters": {
                    "gc_bootstrap": (
                        st.session_state.get("exit_gc_bootstrap", "False")
                        if manual_review_mode
                        else st.session_state.get("auto_bootstrap_scope", "False")
                    ),
                    "preferred_fit": (
                        st.session_state.get("exit_preferred_model", "Best Model")
                        if manual_review_mode
                        else st.session_state.get("auto_preferred_model", "Best Model")
                    ),
                    "response_metric": (
                        st.session_state.get("exit_response_metric", "mu")
                        if manual_review_mode
                        else st.session_state.get("auto_response_metric", "mu")
                    ),
                    "dr_bootstrap": (
                        st.session_state.get("exit_dr_bootstrap", "False")
                        if manual_review_mode
                        else st.session_state.get("auto_dr_bootstrap", "False")
                    ),
                    "export_label_filter": results.get("export_label_filter", "Valid"),
                    "export_dr_include_unsure": bool(results.get("export_dr_include_unsure", False)),
                    "export_dr_include_invalid": bool(results.get("export_dr_include_invalid", False)),
                },
                "stage2_thresholds": results.get("stage2_config"),
                "versions": {
                    "python": sys.version,
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "sklearn": sklearn.__version__,
                },
            }
            audit_zip_bio = io.BytesIO()
            with zipfile.ZipFile(audit_zip_bio, "w", compression=zipfile.ZIP_DEFLATED) as az:
                az.writestr("Classifier Audit.csv", audit_df.to_csv(index=False))
                if isinstance(gc_audit, pd.DataFrame) and not gc_audit.empty:
                    az.writestr("GC Audit.csv", gc_audit.to_csv(index=False))
                if isinstance(dr_audit, pd.DataFrame) and not dr_audit.empty:
                    az.writestr("DR Audit.csv", dr_audit.to_csv(index=False))
                if isinstance(grofit_df, pd.DataFrame) and not grofit_df.empty:
                    grofit_out = grofit_df.drop(columns=["FileName", "Model Name", "Is_Valid"], errors="ignore")
                    az.writestr("Grofit.csv", grofit_out.to_csv(index=False))
                az.writestr("run_info.json", json.dumps(audit_run_info, indent=2))
            st.download_button(
                "Download Auditing.zip",
                data=audit_zip_bio.getvalue(),
                file_name="Auditing.zip",
                mime="application/zip",
                use_container_width=True,
                help="Contains classifier audit, optional GC/DR audits, and run metadata.",
            )

    with tab_dr:
        dr_available = isinstance(dr_fit, pd.DataFrame) and not dr_fit.empty
        if not dr_available:
            st.info("Dose-response results are not available yet. Run the pipeline first.")
        else:
            left_dr, right_dr = st.columns([2.2, 1.0], gap="large")

            with right_dr:
                st.markdown("#### Configuration & Metrics")
                with st.container(border=True):
                    st.markdown("**Filters**")
                    response_metric = st.selectbox(
                        "Metric to display",
                        options=["mu", "A", "lambda", "integral"],
                        index=0,
                        key="dr_metric_tab",
                    )
                    if manual_review_mode:
                        label_source_ui = st.selectbox(
                            "Label source",
                            options=["True", "Predicted"],
                            index=0,
                            key="dr_label_source_tab_manual",
                        )
                        label_source = "final" if label_source_ui == "True" else "pred"
                    else:
                        label_source_ui = st.selectbox(
                            "Label source",
                            options=["Predicted"],
                            index=0,
                            key="dr_label_source_tab_auto",
                        )
                        label_source = "pred"
                    f_col1, f_col2 = st.columns(2)
                    include_unsure = f_col1.checkbox("Include 'Unsure' Curves ⚠️ ", value=False, key="dr_inc_unsure_tab")
                    include_invalid = f_col2.checkbox("Include 'Invalid' Curves ⚠️ ", value=False, key="dr_inc_invalid_tab")

                dr_pill_key = f"dr_overlay_pills_tab_{chosen}"
                dr_bootstrap_selected = bool(
                    manual_review_mode and ("Bootstrap CI" in st.session_state.get(dr_pill_key, []))
                )

                tab_dr_payload = build_dr_payload(
                    gc_fit=gc_fit,
                    labels_df=labels_df,
                    dr_boot=dr_boot,
                    test_id=file_stem,
                    response_metric=response_metric,
                    label_source=label_source,
                    include_unsure=include_unsure,
                    include_invalid=include_invalid,
                    dr_s=grofit_opts.dr_s,
                    smooth_dr=grofit_opts.smooth_dr,
                    dr_x_transform=grofit_opts.dr_x_transform,
                    dr_y_transform=grofit_opts.dr_y_transform,
                    # Compute bootstrap payload when either:
                    # 1) MANUAL pill requests a band, or
                    # 2) drBoot exists (to populate CI metrics).
                    show_bootstrap=bool(
                        dr_bootstrap_selected
                        or (isinstance(dr_boot, pd.DataFrame) and not dr_boot.empty)
                    ),
                )

                fit_payload = tab_dr_payload.get("fit", {})
                ec50 = _to_numeric_scalar(fit_payload.get("ec50"))
                y_mid = _to_numeric_scalar(fit_payload.get("y_mid"))
                dr_method = fit_payload.get("dr_method")
                ec50_status = fit_payload.get("ec50_status")
                boot = tab_dr_payload.get("bootstrap", {})
                ec50_ci = boot.get("ec50_ci") if boot.get("ran") else None
                ec50_lo = _to_numeric_scalar(ec50_ci[0]) if ec50_ci and len(ec50_ci) == 2 else np.nan
                ec50_hi = _to_numeric_scalar(ec50_ci[1]) if ec50_ci and len(ec50_ci) == 2 else np.nan
                n_points = int(tab_dr_payload.get("n_points", 0))
                excluded = int(tab_dr_payload.get("excluded", 0))
                ci_na_reason = ""
                if not (np.isfinite(ec50_lo) and np.isfinite(ec50_hi)):
                    dr_boot_toggle_on = (
                        str(st.session_state.get("exit_dr_bootstrap", "False")).strip().lower() == "true"
                        if manual_review_mode
                        else str(st.session_state.get("auto_dr_bootstrap", "False")).strip().lower() == "true"
                    )
                    min_points_req = max(6, int(grofit_opts.have_atleast))
                    if not dr_boot_toggle_on:
                        ci_na_reason = "DR bootstrap disabled"
                    elif n_points < min_points_req:
                        ci_na_reason = f"insufficient points (< {min_points_req})"
                    elif not isinstance(dr_boot, pd.DataFrame) or dr_boot.empty or "name" not in dr_boot.columns:
                        ci_na_reason = "no matching drBoot row"
                    else:
                        _m = dr_boot["name"].astype(str) == str(file_stem)
                        if int(_m.sum()) == 0:
                            ci_na_reason = "no matching drBoot row"
                        else:
                            ci_na_reason = "EC50 CI missing in drBoot row"

                with st.container(border=True):
                    st.markdown("**Results**")
                    st.metric("EC50", f"{ec50:.4g}" if np.isfinite(ec50) else "NA")
                    st.metric(
                        "95% CI",
                        f"{ec50_lo:.4g} - {ec50_hi:.4g}" if np.isfinite(ec50_lo) and np.isfinite(ec50_hi) else "NA",
                    )
                    if ci_na_reason:
                        st.caption(f"95% CI status: {ci_na_reason}")
                    if dr_method:
                        st.markdown(f"**Fit Method:** `{dr_method}`")
                    if ec50_status:
                        st.markdown(f"**EC50 Status:** `{ec50_status}`")
                    x_tf = str(grofit_opts.dr_x_transform or "OFF")
                    y_tf = str(grofit_opts.dr_y_transform or "OFF")
                    st.caption(
                        f"Axis transforms: X = `{x_tf}`, Y = `{y_tf}`. "
                        "Transforms reshape the fitted DR curve; EC50 is reported back in original concentration units."
                    )
                    st.divider()
                    st.metric("Points Used", n_points)
                    if excluded > 0:
                        st.caption(f"Excluded points: {excluded}")

            with left_dr:
                st.markdown("#### Dose-Response Curve")
                if manual_review_mode:
                    show_dr_boot = _st_pills_multi(
                        "",
                        ["Bootstrap CI"],
                        default=st.session_state.get(dr_pill_key, []),
                        key=dr_pill_key,
                    )
                    show_dr_bootstrap = "Bootstrap CI" in show_dr_boot
                else:
                    show_dr_bootstrap = False
                dr_fig = make_dr_plot(tab_dr_payload, show_bootstrap=show_dr_bootstrap)
                dr_fig.update_layout(height=600)
                st.plotly_chart(dr_fig, use_container_width=True)

    if st.session_state.get("trigger_grofit_run"):
        st.session_state["trigger_grofit_run"] = False
        if manual_review_mode:
            gc_bootstrap = st.session_state.get("exit_gc_bootstrap", "False")
            preferred_model = st.session_state.get("exit_preferred_model", "Best Model")
            response_metric = st.session_state.get("exit_response_metric", "mu")
            dr_bootstrap = st.session_state.get("exit_dr_bootstrap", "True")
            with st.spinner("Preparing Grofit raw file (raw OD + final labels) and running Grofit..."):
                grofit_source_df = results.get("grofit_df")
                if not isinstance(grofit_source_df, pd.DataFrame) or grofit_source_df.empty:
                    grofit_source_df = _build_grofit_input_df(
                        wide_df=wide_original,
                        out_df=out_df,
                        review_df=review_df,
                        manual_review_mode=True,
                        meta_df=results.get("meta_df", out_df),
                        audit_df=results.get("audit_df"),
                    )
                grofit_tidy_all = wide_original_to_grofit_tidy(grofit_source_df, file_tag=file_stem)
                true_map = grofit_source_df.set_index("Test Id")["True Label"].to_dict() if "True Label" in grofit_source_df.columns else {}
                grofit_tidy_all["true_label"] = grofit_tidy_all["curve_id"].map(true_map).apply(_normalize_label)
                grofit_tidy_all["is_valid_true"] = grofit_tidy_all["true_label"].map(_label_is_valid).fillna(False).astype(bool)

                preferred_fit_map = {
                    "Best Model": "Both (param + spline)",
                    "Spline": "Spline only",
                    "Parametric": "Parametric only",
                }
                fit_opt_map = {
                    "Both (param + spline)": "b",
                    "Parametric only": "m",
                    "Spline only": "s",
                }
                fit_mode_label = preferred_fit_map.get(preferred_model, "Both (param + spline)")
                fit_opt = fit_opt_map.get(fit_mode_label, "b")

                gc_boot_B = 0 if gc_bootstrap == "False" else grofit_opts.gc_boot_B
                dr_boot_B = grofit_opts.dr_boot_B if dr_bootstrap == "True" else 0
                validity_col = "__all__"
                grofit_opts_effective = GrofitOptions(**grofit_opts.__dict__)
                grofit_opts_effective.response_var = response_metric
                grofit_opts_effective.fit_opt = fit_opt
                grofit_opts_effective.gc_boot_B = int(gc_boot_B)
                grofit_opts_effective.dr_boot_B = int(dr_boot_B)
                grofit_opts_effective.bootstrap_method = _normalize_bootstrap_method(
                    grofit_opts_effective.bootstrap_method
                )

                gc_fit = pd.DataFrame()
                dr_fit = pd.DataFrame()
                gc_boot = pd.DataFrame()
                dr_boot = pd.DataFrame()
                gc_audit = pd.DataFrame()
                dr_audit = pd.DataFrame()
                zip_bytes = b""
                zip_name = ""

                curves_for_run = grofit_tidy_all

                if not grofit_tidy_all.empty:
                    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as grofit_td:
                        grofit_res = run_grofit_pipeline(
                            curves_df=curves_for_run,
                            response_var=response_metric,
                            have_atleast=grofit_opts.have_atleast,
                            gc_boot_B=gc_boot_B,
                            dr_boot_B=dr_boot_B,
                            spline_auto_cv=grofit_opts.spline_auto_cv,
                            spline_s=grofit_opts.spline_s,
                            smooth_gc=grofit_opts.smooth_gc,
                            smooth_dr=grofit_opts.smooth_dr,
                            dr_x_transform=grofit_opts.dr_x_transform,
                            dr_y_transform=grofit_opts.dr_y_transform,
                            dr_s=grofit_opts.dr_s,
                            fit_opt=fit_opt,
                            bootstrap_method=_normalize_bootstrap_method(grofit_opts.bootstrap_method),
                            validity_col=validity_col,
                            random_state=42,
                            export_dir=Path(grofit_td),
                        )
                        gc_fit = grofit_res.get("gc_fit", pd.DataFrame())
                        dr_fit = grofit_res.get("dr_fit", pd.DataFrame())
                        gc_boot = grofit_res.get("gc_boot", pd.DataFrame())
                        dr_boot = grofit_res.get("dr_boot", pd.DataFrame())
                        gc_audit = grofit_res.get("gc_audit", pd.DataFrame())
                        dr_audit = grofit_res.get("dr_audit", pd.DataFrame())
                        zip_name, zip_bytes = _build_export_zip(
                            wide_df=wide_original,
                            out_df=out_df,
                            review_df=review_df,
                            gc_fit=gc_fit,
                            gc_boot=gc_boot,
                            dr_fit=dr_fit,
                            dr_boot=dr_boot,
                            proc_wide_df=final_merged,
                            grofit_opts=grofit_opts_effective,
                            settings=settings,
                            mode_label="MANUAL",
                            file_stem=file_stem,
                            predicting_model=predicting_model,
                            selected_gc_bootstrap=str(gc_bootstrap),
                            selected_preferred_fit=str(preferred_model),
                            selected_response_metric=str(response_metric),
                            selected_dr_bootstrap=str(dr_bootstrap),
                            export_label_filter=results.get("export_label_filter", "Valid"),
                            export_dr_include_unsure=bool(results.get("export_dr_include_unsure", False)),
                            export_dr_include_invalid=bool(results.get("export_dr_include_invalid", False)),
                            audit_df=results.get("audit_df"),
                            grofit_df=results.get("grofit_df"),
                            grofit_tidy_all=grofit_tidy_all,
                            stage2_config=results.get("stage2_config"),
                        )

            results["grofit_tidy_all"] = grofit_tidy_all
            # Persist effective runtime options so run_info/audit reflects what was executed.
            results["grofit_opts"] = grofit_opts_effective
            results["gc_fit"] = gc_fit
            results["dr_fit"] = dr_fit
            results["gc_boot"] = gc_boot
            results["dr_boot"] = dr_boot
            results["gc_audit"] = gc_audit
            results["dr_audit"] = dr_audit
            results["zip_bytes"] = zip_bytes
            results["zip_name"] = zip_name if "zip_name" in locals() else ""
            results["grofit_ran"] = True
            st.session_state["last_run_results"] = results
            st.rerun()

    st.markdown("---")


# =========================
# UI
# =========================
st.set_page_config(page_title="GrowthQC - Curve Validator", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Source+Code+Pro:wght@400;600&display=swap');
    :root {
        --ink: #2b1f1a;
        --muted: #7a6a5f;
        --panel: #fffaf4;
        --line: #d9c8b8;
        --blue: #c26d3a;
        --blue-dark: #9a4f27;
        --bg1: #f7efe5;
        --bg2: #eadfce;
    }
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(120deg, var(--bg1), var(--bg2));
        font-family: 'Space Grotesk', system-ui, sans-serif;
        color: var(--ink);
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --ink: #ffffff;
            --muted: #d2c4b6;
            --panel: #2a1f1a;
            --line: #4a3a32;
            --blue: #d07a45;
            --blue-dark: #b06133;
            --bg1: #1a1411;
            --bg2: #2a201a;
        }
        html, body, [data-testid="stAppViewContainer"] {
            color: var(--ink);
        }
    }
    [data-testid="stAppViewContainer"] .stMarkdown h1,
    [data-testid="stAppViewContainer"] .stMarkdown h2,
    [data-testid="stAppViewContainer"] .stMarkdown h3 {
        letter-spacing: 0.2px;
    }
    .ui-card {
        background: var(--panel);
        border: 2px solid #111;
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 6px 20px rgba(16,24,40,0.06);
        margin-top: 0;
    }
    .metrics-title {
        font-size: 6rem;
        font-weight: 700;
        margin: 0;
        width: 100%;
    }
    .total-reviewed {
        font-size: 1.6rem;
        font-weight: 500;
        margin: 0;
        text-align: center;
    }
    .metrics-panel {
        height: 520px;          /* SAME as the plot height */
        overflow-y: auto;       /* scroll if content is longer */
        margin-top: -542px;      /* ensure table starts immediately */
        padding-top: 0;
    }
    .metrics-header {
        margin-bottom: -5px;
        margin-top: -15px;
        font-size: 1.7rem;      /* Metrics heading */
        font-weight: 700;
    }
    .metrics-subheader {
        margin-top: 0px;
        margin-bottom: 4px;
        text-align: left;
        opacity: 0.95;
        font-size: 1.5rem;      /* Metrics heading */
        font-weight: 500;
    }
    # .metrics-top {
    #     margin: 0 0 6px 0;
    # }
    .curve-title {
        font-weight: 600;
        margin: 0 0 6px 0;
    }
    .boot-btn button {
        width: 100%;
    }
    .boot-btn.added button {
        background: #4f8b5a !important;
        border-color: #2f6a3a !important;
    }
    .adv-wrap [data-testid="stExpander"] summary {
        justify-content: center;
        text-align: center;
    }
    .ui-row-title {
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 6px;
    }
    .ui-inline-note {
        font-size: 12px;
        color: var(--muted);
    }
    .ui-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: #f1e2d3;
        color: #6e3b1c;
        font-size: 12px;
        font-weight: 600;
    }
    .stButton > button {
        background: var(--blue);
        color: #fff;
        border-radius: 8px;
        border: 2px solid var(--blue-dark);
        padding: 0.65rem 1.1rem;
        font-weight: 600;
        box-shadow: 0 2px 0 rgba(0,0,0,0.15);
    }
    .stButton > button:disabled {
        opacity: 0.55;
    }
    .stDownloadButton > button {
        border-radius: 8px;
        border: 2px solid var(--blue-dark);
    }
    .stRadio > div, .stSelectbox > div, .stTextInput > div {
        border-radius: 8px;
    }
    .stDataFrame, .stDataEditor {
        background: var(--panel);
    }
    .sample-card {
        box-shadow: none;
    }
    .sample-card [data-testid="stTextInput"] {
        display: none;
    }
    .review-actions .stButton > button,
    .review-actions .stDownloadButton > button {
        height: 44px;
        min-height: 44px;
        padding: 0 12px;
        width: 190px;
    }
    .review-actions .stButton,
    .review-actions .stDownloadButton {
        display: flex;
        justify-content: center;
    }
    .review-actions [data-testid="stCheckbox"] label{
        background: var(--blue) !important;
        color: #fff !important;
        border: 2px solid var(--blue-dark) !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        width: 190px !important;
        box-shadow: 0 2px 0 rgba(0,0,0,0.15) !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }
    .review-actions [data-testid="stCheckbox"] input{
        accent-color: #fff !important;
    }
    /* #7A Download = blue */
    .action-buttons .stDownloadButton > button{
        background: var(--blue) !important;
        color: #fff !important;
        border: 2px solid var(--blue-dark) !important;
        box-shadow: 0 2px 0 rgba(0,0,0,0.15) !important;
        border-radius: 8px !important;
        height: 44px !important;
    }

    /* Keep uploader button styled separately */
    .action-buttons [data-testid="stFileUploader"] button{
        background: var(--blue) !important;
        color: #fff !important;
        border: 2px solid var(--blue-dark) !important;
        box-shadow: 0 2px 0 rgba(0,0,0,0.15) !important;
        border-radius: 8px !important;
        height: 44px !important;
        width: 100% !important;
        font-weight: 600 !important;
    }
    /* Hide the dropzone “card” look + instructions */
    .action-buttons [data-testid="stFileUploader"] section{
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
    }
    .action-buttons [data-testid="stFileUploaderDropzone"]{
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
    }
    .action-buttons [data-testid="stFileUploaderDropzoneInstructions"],
    .action-buttons [data-testid="stFileUploader"] svg,
    .action-buttons [data-testid="stFileUploader"] small,
    .action-buttons [data-testid="stFileUploader"] ul{
        display: none !important;
    }

    /* Replace “Browse files” text */
    .action-buttons [data-testid="stFileUploader"] button span{
        display: none !important;
    }
    .action-buttons [data-testid="stFileUploader"] button::before{
        content: "#7B Upload True Labels";
    }
    .review-actions .stDownloadButton > button {
        background: var(--blue);
        color: #fff;
        border: 2px solid var(--blue-dark);
        box-shadow: 0 2px 0 rgba(0,0,0,0.15);
    }
    .review-actions [data-testid="stFileUploader"] section {
        padding: 0;
        border: none;
        background: transparent;
    }
    .review-actions [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
    .review-actions [data-testid="stFileUploader"] svg,
    .review-actions [data-testid="stFileUploader"] small {
        display: none !important;
    }
    .adv-wrap {
        margin-top: 34px;
        max-width: 520px;
    }
    /* --- NEW: Blue "button-style" checkbox for Reviewed --- */
    .reviewed-blue [data-testid="stCheckbox"] label {
        background: var(--blue) !important;
        color: #fff !important;
        border: 2px solid var(--blue-dark) !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        width: 190px !important;
        box-shadow: 0 2px 0 rgba(0,0,0,0.15) !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }
    .reviewed-blue [data-testid="stCheckbox"] input {
        accent-color: #fff !important;
    }

    /* --- NEW: Make ONLY #7A download look like your primary blue buttons --- */
    .action-buttons .btn-blue .stDownloadButton > button {
        background: var(--blue) !important;
        color: #fff !important;
        border: 2px solid var(--blue-dark) !important;
        box-shadow: 0 2px 0 rgba(0,0,0,0.15) !important;
    }

    /* --- NEW: Make #7B uploader look like a single normal button --- */
    .action-buttons .upload-btn [data-testid="stFileUploader"] section {
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
    }
    .action-buttons .upload-btn [data-testid="stFileUploaderDropzone"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
    }
    .action-buttons .upload-btn [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
    .action-buttons .upload-btn [data-testid="stFileUploader"] svg,
    .action-buttons .upload-btn [data-testid="stFileUploader"] small,
    .action-buttons .upload-btn [data-testid="stFileUploader"] ul {
        display: none !important;
    }

    /* Style the internal "Browse files" button to match your blue buttons */
    .action-buttons .upload-btn [data-testid="stFileUploader"] button {
        background: var(--blue) !important;
        color: #fff !important;
        border-radius: 8px !important;
        border: 2px solid var(--blue-dark) !important;
        height: 44px !important;
        width: 100% !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 0 rgba(0,0,0,0.15) !important;
    }

    /* Replace the button text "Browse files" with "#7B Upload True Labels" */
    .action-buttons .upload-btn [data-testid="stFileUploader"] button span {
        display: none !important;
    }
    .action-buttons .upload-btn [data-testid="stFileUploader"] button::before {
        content: "#7B Upload True Labels";
    }
    /* ---------- #7B: Target ONLY the manual uploader by its input id (key) ---------- */
    div[data-testid="stFileUploader"]:has(input[id="manual_review_upload"]) section{
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    }
    div[data-testid="stFileUploader"]:has(input[id="manual_review_upload"]) [data-testid="stFileUploaderDropzone"]{
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
    }
    div[data-testid="stFileUploader"]:has(input[id="manual_review_upload"]) [data-testid="stFileUploaderDropzoneInstructions"],
    div[data-testid="stFileUploader"]:has(input[id="manual_review_upload"]) svg,
    div[data-testid="stFileUploader"]:has(input[id="manual_review_upload"]) small,
    div[data-testid="stFileUploader"]:has(input[id="manual_review_upload"]) ul{
    display: none !important;
    }
    div[data-testid="stFileUploader"]:has(input[id="manual_review_upload"]) button{
    background: var(--blue) !important;
    color: #fff !important;
    border: 2px solid var(--blue-dark) !important;
    border-radius: 8px !important;
    height: 44px !important;
    width: 100% !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 0 rgba(0,0,0,0.15) !important;
    }
    /* Replace internal button text */
    div[data-testid="stFileUploader"]:has(input[id="manual_review_upload"]) button span{
    display: none !important;
    }
    div[data-testid="stFileUploader"]:has(input[id="manual_review_upload"]) button::before{
    content: "#7B Upload True Labels";
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("## GrowthQC - Bacterial Growth Curve Validator")
st.caption("Upload -> preprocess -> predict -> review -> grofit -> download")

# Show any training status messages (persist across reruns)
train_status = st.session_state.pop("train_status", None)
if train_status:
    status_kind, status_msg = train_status
    if status_kind == "success":
        st.success(status_msg)
    else:
        st.error(status_msg)

# -------------------------
# Top controls (no sidebar)
# -------------------------
top_left, top_right = st.columns([3.2, 1.6], gap="large")
with top_left:
    uploaded = st.file_uploader(
        "Upload file (Excel .xlsx or CSV .csv) in your defined format. CSV Preferred",
        type=["xlsx", "csv"],
        accept_multiple_files=False,
    )
    left_group = st.container()
    with left_group:
        models_ready = has_trained_models(MODEL_DIR)
        run_disabled = (uploaded is None) or (not models_ready)
        mode_col, run_col, train_col = st.columns([2.2, 0.8, 1.3], gap="small")
        with mode_col:
            st.markdown('<div class="ui-row-title">Mode</div>', unsafe_allow_html=True)
            mode = st.radio(
                "Run mode",
                options=["Auto Mode", "Manual Mode"],
                horizontal=True,
                label_visibility="collapsed",
            )
        with run_col:
            st.markdown('<div class="ui-row-title">Run</div>', unsafe_allow_html=True)
            run = st.button("Run pipeline", type="primary", disabled=run_disabled, use_container_width=True)
        with train_col:
            st.markdown('<div class="ui-row-title">Classifier</div>', unsafe_allow_html=True)
            train_refresh = st.button(
                "Train / Refresh Classifier",
                use_container_width=True,
            )
        if not models_ready:
            st.info("No trained classifier found. Click 'Train / Refresh Classifier' to train models.")
        auto_mode = mode == "Auto Mode"
        manual_review_mode = mode == "Manual Mode"

        auto_bootstrap_scope = "False"
        auto_preferred_model = "Best Model"
        auto_response_metric = "mu"
        auto_dr_bootstrap = "True"
        if auto_mode:
            st.markdown('<div class="ui-row-title">Input File Upload</div>', unsafe_allow_html=True)
            auto_c1, auto_c2, auto_c3, auto_c4 = st.columns(4)
            with auto_c1:
                auto_bootstrap_scope = st.selectbox(
                    "GC Bootstrap",
                    options=["False", "True"],
                    index=0,
                    key="auto_bootstrap_scope",
                )
            with auto_c2:
                auto_preferred_model = st.selectbox(
                    "Preferred Fit",
                    options=["Best Model", "Spline", "Parametric"],
                    index=0,
                    key="auto_preferred_model",
                )
            with auto_c3:
                auto_response_metric = st.selectbox(
                    "Response Metric",
                    options=["mu", "A", "lag", "integral"],
                    index=0,
                    key="auto_response_metric",
                )
            with auto_c4:
                auto_dr_bootstrap = st.selectbox(
                    "DR Bootstrap",
                    options=["True", "False"],
                    index=0,
                    key="auto_dr_bootstrap",
                )
                	

        with st.expander("Filter Options for Pipeline & Export", expanded=False):
            response_var = auto_response_metric if auto_mode else "mu"

            # Model Selection (Stage 1 Classifier)
            available_models = discover_models(MODEL_DIR)

            def _label_from_stem(stem: str) -> str:
                s = stem.lower()
                if "hgb" in s or "hist" in s:
                    return "HGB"
                if "rf" in s or "random" in s:
                    return "RF"
                if "lr" in s or "logreg" in s or "logistic" in s:
                    return "LR"
                return stem

            model_label_map: dict[str, Path] = {}
            for stem, p in available_models.items():
                label = _label_from_stem(stem)
                if label in model_label_map:
                    label = f"{label}-{stem}"
                model_label_map[label] = p

            model_options: list[str] = []
            model_name: str | None = None
            if model_label_map:
                model_options = ["Average"] + sorted(model_label_map.keys())
                model_name = st.selectbox("Choose trained model or ensemble", options=model_options, index=0)
                if model_name == "Average":
                    st.caption(f"Loaded ensemble of {len(model_label_map)} models: {', '.join(sorted(model_label_map.keys()))}")
                else:
                    st.caption("Loaded from: `/classifier_output/saved_models_selected`")
            else:
                st.warning("No trained model found. Click Train/Refresh Classifier.")
            
            # st.markdown("**Blank Handling**")
            input_is_raw = st.checkbox("Input is raw (apply blank subtraction)", value=False)
            global_blank_str = ""
            if input_is_raw:
                global_blank_str = st.text_input(
                    "Global blank value (optional)",
                    value="",
                    help="Leave blank to calculate dynamically from the first few points of each curve.",
                )

            # Pipeline Mathematical Thresholds
            st.markdown("**Grofit Parameters**")
            have_atleast = st.number_input("Min. points required for Dose-Response", min_value=1, value=6, step=1)

            c1, c2 = st.columns(2)
            with c1:
                gc_boot_B = st.number_input(
                    "GC Bootstrap Iterations",
                    min_value=0,
                    value=200,
                    step=50,
                    help="Number of resamples for single curve confidence intervals.",
                )
            with c2:
                dr_boot_B = st.number_input(
                    "DR Bootstrap Iterations",
                    min_value=0,
                    value=300,
                    step=50,
                    help="Number of resamples for dose-response confidence intervals.",
                )

            # dr_x_transform = st.selectbox("Dose-Response X-Axis Transform", options=["OFF","log10", "log1p"], index=0)
            c3, c4 = st.columns(2)
            with c3:
                 dr_x_transform = st.selectbox(
                    "Dose-Response X-Axis Transform",
                    options=["OFF", "log10", "log1p"],
                    index=0,
                    help="Transform applied to concentration axis for DR fit/plots.",
                 )
            with c4:
                dr_y_transform = st.selectbox(
                    "Dose-Response Y-Axis Transform",
                    options=["OFF", "log10", "log1p"],
                    index=0,
                    help="Transform applied to response metric axis for DR fit/plots.",
                )
            export_label_filter = st.selectbox(
                "Export Curve Labels",
                options=["Valid", "Invalid", "Unsure", "All"],
                index=0,
                help="Filter which curve labels are exported into Results.zip.",
            )
            _exp_label_norm = str(export_label_filter).strip().lower()
            dr_unsure_enabled = _exp_label_norm in {"all", "unsure"}
            dr_invalid_enabled = _exp_label_norm in {"all", "invalid"}
            if not dr_unsure_enabled:
                st.session_state["export_dr_include_unsure"] = False
            if not dr_invalid_enabled:
                st.session_state["export_dr_include_invalid"] = False
            export_dr_c1, export_dr_c2 = st.columns(2)
            with export_dr_c1:
                export_dr_include_unsure = st.checkbox(
                    "DR Export: Include Unsure",
                    value=bool(st.session_state.get("export_dr_include_unsure", False)),
                    key="export_dr_include_unsure",
                    disabled=not dr_unsure_enabled,
                )
            with export_dr_c2:
                export_dr_include_invalid = st.checkbox(
                    "DR Export: Include Invalid",
                    value=bool(st.session_state.get("export_dr_include_invalid", False)),
                    key="export_dr_include_invalid",
                    disabled=not dr_invalid_enabled,
                )

            st.markdown("**Advanced (Parity / Debug)**")
            override_smoothing = st.checkbox(
                "Override smoothing (smooth.gc / smooth.dr)",
                value=False,
                help="Default is AUTO (recommended). Changing these affects fitted parameters and parity.",
            )
            smooth_c1, smooth_c2 = st.columns(2)
            with smooth_c1:
                smooth_gc = st.number_input(
                    "smooth.gc (GC spline smoothing)",
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    disabled=not override_smoothing,
                    help="Only used when override is ON. Keep OFF for automated pipeline.",
                )
            with smooth_c2:
                smooth_dr = st.number_input(
                    "smooth.dr (DR spline smoothing)",
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    disabled=not override_smoothing,
                    help="Only used when override is ON. Keep OFF for automated pipeline.",
                )
            # Hardcoded Mathematical Defaults (Hidden from Biologists to prevent errors)
            preferred_fit_map = {
                "Best Model": "b",
                "Spline": "s",
                "Parametric": "m",
            }
            dr_boot_B_effective = 0 if (auto_mode and auto_dr_bootstrap == "False") else int(dr_boot_B)

            grofit_opts = GrofitOptions(
                response_var=response_var,
                have_atleast=int(have_atleast),
                fit_opt=preferred_fit_map.get(auto_preferred_model if auto_mode else "Best Model", "b"),
                gc_boot_B=int(gc_boot_B),
                dr_boot_B=dr_boot_B_effective,
                # spline_auto_cv=True,
                # spline_s=None,  # Mathematically forced to GCV in backend
                # dr_s=None,  # Mathematically forced to DR constraints in backend
                # dr_x_transform=None if dr_x_transform == "none" else dr_x_transform,
                spline_auto_cv=(not override_smoothing),
                spline_s=(None if (not override_smoothing) else float(smooth_gc) if float(smooth_gc) > 0 else None),
                smooth_gc=(None if (not override_smoothing) else float(smooth_gc) if float(smooth_gc) > 0 else None),
                dr_s=(None if (not override_smoothing) else float(smooth_dr) if float(smooth_dr) > 0 else None),
                smooth_dr=(None if (not override_smoothing) else float(smooth_dr) if float(smooth_dr) > 0 else None),
                dr_x_transform=None if dr_x_transform == "OFF" else dr_x_transform,
                dr_y_transform=None if dr_y_transform == "OFF" else dr_y_transform,
            )
            grofit_opts.bootstrap_method = _normalize_bootstrap_method(grofit_opts.bootstrap_method)
        st.session_state["grofit_opts"] = grofit_opts
        settings = InferenceSettings(
            input_is_raw=bool(input_is_raw),
            global_blank=_safe_float(global_blank_str, None) if input_is_raw else None,
        )
with top_right:
    # st.markdown('<div class="ui-card sample-card">', unsafe_allow_html=True)
    st.markdown('<div class="ui-row-title">Sample formats</div>', unsafe_allow_html=True)
    st.caption(
        "For the long-format sample, well headers may encode concentration like "
        "`A01[Conc=0.1]` or `A01[0.1]` (both are acceptable in the first row header). "
        "For dose-response, the wide-format sample includes a 'concentration' column."
    )
    st.download_button(
        "Download sample (Wide CSV)",
        data=make_sample_wide_csv_bytes(),
        file_name="sample_wide.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        "Download sample (Long CSV)",
        data=make_sample_long_csv_bytes(),
        file_name="sample_long.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Auto-training debug summary (if present)
auto_debug = st.session_state.get("train_auto_debug")
auto_summary = st.session_state.get("train_auto_summary")
if auto_debug or auto_summary:
    with st.expander("Auto-training debug info", expanded=False):
        if auto_debug:
            st.code(json.dumps(auto_debug, indent=2), language="json")
        if auto_summary:
            st.code(json.dumps(auto_summary, indent=2), language="json")
results = st.session_state.get("last_run_results")

# If the user clears the upload (clicks X), drop cached results
if uploaded is None and results and not run:
    st.session_state.pop("last_run_results", None)
    results = None

if train_refresh:
    if not TRAIN_META.exists():
        st.error(f"Training meta.csv not found at: {TRAIN_META}")
    else:
        try:
            # Show exact meta rows used for this training run in the UI.
            meta_rows = int(pd.read_csv(TRAIN_META).shape[0])
            st.info(f"Training classifier with `{TRAIN_META.name}` ({meta_rows} rows).")
            with st.spinner("Training classifier from meta.csv..."):
                train_out = train_classifier_from_meta_file(
                    meta_csv_path=str(TRAIN_META),
                    models_out_dir=str(MODEL_DIR),
                    selected_features=NOTEBOOK_STAGE1_CUSTOM_FEATURES,
                )
            model_files = sorted([p.name for p in MODEL_DIR.glob("*.joblib")])
            split_sizes = train_out.get("split_sizes", {}) if isinstance(train_out, dict) else {}
            split_text = ""
            if split_sizes:
                split_text = (
                    f" | split sizes: train={split_sizes.get('train', 'NA')}, "
                    f"val={split_sizes.get('val', 'NA')}, test={split_sizes.get('test', 'NA')}"
                )
            st.success(
                "Training complete. Models refreshed in classifier_output/saved_models_selected. "
                f"Meta rows used: {meta_rows}. "
                f"Saved {len(model_files)} model file(s): {', '.join(model_files)}"
                f"{split_text}"
            )
            st.rerun()
        except Exception as e:
            show_friendly_error(e)

if run:
    if uploaded is None:
        st.warning("Please upload a file before running validation.")
    else:
        try:
            suffix = Path(uploaded.name).suffix.lower()
            upload_bytes = uploaded.getvalue()
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
                tmp_in = Path(td) / f"upload{suffix}"
                tmp_in.write_bytes(upload_bytes)
                with st.spinner("Converting uploaded file to canonical wide format..."):
                    try:
                        long_df = parse_any_file_to_long(str(tmp_in))
                    except Exception as e:
                        st.warning(
                            "Could not read the uploaded file. Please ensure it matches the sample formats provided "
                            "below (wide: Test Id + T#.0 (h) columns; long: Time (h) with wells as columns)."
                        )
                        show_friendly_error(e)
                        st.stop()
                    file_tag = Path(uploaded.name).stem
                    wide_df = long_to_wide_preserve_times(long_df, file_tag=file_tag, add_prefix=True)
                    if "Concentration" not in wide_df.columns:
                        conc = wide_df["Test Id"].astype(str).map(_extract_conc_from_curve_id)
                        conc = pd.to_numeric(conc, errors="coerce")
                        if conc.notna().any():
                            wide_df["Concentration"] = conc.fillna(0.0)
                    time_cols_original = [c for c in wide_df.columns if parse_time_from_header(str(c)) is not None]

            available_models = discover_models(MODEL_DIR)
            if not available_models:
                st.error("No trained model found. Click Train/Refresh Classifier.")
                st.stop()

            with st.spinner("Running Stage-1 model inference + Stage-2 late-growth logic..."):
                infer_res = run_label_inference_from_uploaded_wide(
                    wide_df=wide_df,
                    settings=settings,
                    model_dir=str(MODEL_DIR),
                    model_name=model_name or "Average",
                    stage2_start=16.0,
                    unsure_conf_threshold=None,
                )
            predicting_model = model_name or "Average"
            raw_merged = infer_res["raw_merged_df"]
            final_merged = infer_res["final_merged_df"]
            out_df = infer_res["out_df"].copy()
            time_cols_final = [c for c in final_merged.columns if isinstance(c, str) and c.strip().startswith("T") and "(h)" in c]
            review_df = _init_review_df(out_df, wide_df)
            st.session_state["review_df"] = review_df.copy()

            audit_df = _build_classifier_output(
                wide_df=wide_df,
                out_df=out_df,
                review_df=review_df if manual_review_mode else None,
                manual_review_mode=manual_review_mode,
                meta_df=infer_res.get("meta_df", out_df),
                processed_wide_df=final_merged,
            )
            grofit_df = _build_grofit_input_df(
                wide_df=wide_df,
                out_df=out_df,
                review_df=review_df if manual_review_mode else None,
                manual_review_mode=manual_review_mode,
                meta_df=infer_res.get("meta_df", out_df),
                audit_df=audit_df,
            )
            grofit_tidy_all = wide_original_to_grofit_tidy(grofit_df, file_tag=file_tag)
            true_map = grofit_df.set_index("Test Id")["True Label"].to_dict() if "True Label" in grofit_df.columns else {}
            pred_map = out_df.set_index("Test Id")["Pred Label"].to_dict() if "Pred Label" in out_df.columns else {}
            conf_map = out_df.set_index("Test Id")["Pred Confidence"].to_dict() if "Pred Confidence" in out_df.columns else {}
            grofit_tidy_all["true_label"] = grofit_tidy_all["curve_id"].map(true_map).apply(_normalize_label)
            grofit_tidy_all["is_valid_true"] = grofit_tidy_all["true_label"].map(_label_is_valid).fillna(False).astype(bool)
            grofit_tidy_all["pred_label"] = grofit_tidy_all["curve_id"].map(pred_map)
            grofit_tidy_all["pred_confidence"] = pd.to_numeric(grofit_tidy_all["curve_id"].map(conf_map), errors="coerce")
            gc_fit = pd.DataFrame()
            dr_fit = pd.DataFrame()
            gc_boot = pd.DataFrame()
            dr_boot = pd.DataFrame()
            gc_audit = pd.DataFrame()
            dr_audit = pd.DataFrame()
            zip_bytes = b""
            grofit_ran = False

            if manual_review_mode:
                run_grofit = not grofit_tidy_all.empty
                if run_grofit:
                    st.info("Manual Mode: running curve fitting (spline + model) to populate metrics.")
                    with st.spinner("Running curve fitting for metrics (manual mode)..."):
                        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as grofit_td:
                            grofit_res = run_grofit_pipeline(
                                curves_df=grofit_tidy_all,
                                response_var=grofit_opts.response_var,
                                have_atleast=grofit_opts.have_atleast,
                                gc_boot_B=0,
                                dr_boot_B=0,
                                spline_auto_cv=grofit_opts.spline_auto_cv,
                                spline_s=grofit_opts.spline_s,
                                smooth_gc=grofit_opts.smooth_gc,
                                smooth_dr=grofit_opts.smooth_dr,
                                dr_x_transform=grofit_opts.dr_x_transform,
                                dr_y_transform=grofit_opts.dr_y_transform,
                                dr_s=grofit_opts.dr_s,
                                fit_opt=grofit_opts.fit_opt,
                                bootstrap_method=_normalize_bootstrap_method(grofit_opts.bootstrap_method),
                                validity_col="__all__",
                                random_state=42,
                                export_dir=Path(grofit_td),
                            )
                            gc_fit = grofit_res.get("gc_fit", pd.DataFrame())
                            dr_fit = grofit_res.get("dr_fit", pd.DataFrame())
                            gc_boot = grofit_res.get("gc_boot", pd.DataFrame())
                            dr_boot = grofit_res.get("dr_boot", pd.DataFrame())
                            gc_audit = grofit_res.get("gc_audit", pd.DataFrame())
                            dr_audit = grofit_res.get("dr_audit", pd.DataFrame())
                else:
                    st.info("Manual Mode: no curves found for fitting.")
                st.info("You can still review labels and run the manual 'RUN GROFIT' action for full export.")
            else:
                st.info("Auto Mode: running Grofit pipeline with bootstrap and dose-response settings.")
                bootstrap_curve_scope = auto_bootstrap_scope if auto_mode else "Only Valid Curves"
                gc_boot_B_auto = 0 if str(bootstrap_curve_scope) == "False" else int(grofit_opts.gc_boot_B)
                grofit_opts_effective_auto = GrofitOptions(**grofit_opts.__dict__)
                grofit_opts_effective_auto.gc_boot_B = int(gc_boot_B_auto)
                grofit_opts_effective_auto.bootstrap_method = _normalize_bootstrap_method(
                    grofit_opts_effective_auto.bootstrap_method
                )
                # Restore previous behavior: in AUTO mode, run Grofit whenever curves exist.
                # Curve-level reliability/point checks are handled inside Grofit pipeline.
                validity_col = "__all__"
                run_grofit = not grofit_tidy_all.empty

                if run_grofit:
                    st.info(
                        f"Grofit settings: GC_Fit Bootstrap={bootstrap_curve_scope}, "
                        f"Preferred Fit={auto_preferred_model}, Response Metric={auto_response_metric}, "
                        f"DR Bootstrap={auto_dr_bootstrap}"
                    )
                    with st.spinner("Running Grofit (parametric + spline fits, bootstrap, dose-response)..."):
                        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as grofit_td:
                            grofit_res = run_grofit_pipeline(
                                curves_df=grofit_tidy_all,
                                response_var=grofit_opts_effective_auto.response_var,
                                have_atleast=grofit_opts_effective_auto.have_atleast,
                                gc_boot_B=grofit_opts_effective_auto.gc_boot_B,
                                dr_boot_B=grofit_opts_effective_auto.dr_boot_B,
                                spline_auto_cv=grofit_opts_effective_auto.spline_auto_cv,
                                spline_s=grofit_opts_effective_auto.spline_s,
                                smooth_gc=grofit_opts_effective_auto.smooth_gc,
                                smooth_dr=grofit_opts_effective_auto.smooth_dr,
                                dr_x_transform=grofit_opts_effective_auto.dr_x_transform,
                                dr_y_transform=grofit_opts_effective_auto.dr_y_transform,
                                dr_s=grofit_opts_effective_auto.dr_s,
                                fit_opt=grofit_opts_effective_auto.fit_opt,
                                bootstrap_method=grofit_opts_effective_auto.bootstrap_method,
                                validity_col=validity_col,
                                random_state=42,
                                export_dir=Path(grofit_td),
                            )
                            gc_fit = grofit_res.get("gc_fit", pd.DataFrame())
                            dr_fit = grofit_res.get("dr_fit", pd.DataFrame())
                            gc_boot = grofit_res.get("gc_boot", pd.DataFrame())
                            dr_boot = grofit_res.get("dr_boot", pd.DataFrame())
                            gc_audit = grofit_res.get("gc_audit", pd.DataFrame())
                            dr_audit = grofit_res.get("dr_audit", pd.DataFrame())
                            zip_name, zip_bytes = _build_export_zip(
                                wide_df=wide_df,
                                out_df=out_df,
                                review_df=review_df if manual_review_mode else None,
                                gc_fit=gc_fit,
                                gc_boot=gc_boot,
                                dr_fit=dr_fit,
                                dr_boot=dr_boot,
                                proc_wide_df=final_merged,
                                audit_df=audit_df,
                                grofit_df=grofit_df,
                                grofit_opts=grofit_opts_effective_auto,
                                settings=settings,
                                mode_label="AUTO",
                                file_stem=file_tag,
                                predicting_model=predicting_model,
                                auto_bootstrap_scope=auto_bootstrap_scope,
                                auto_preferred_model=auto_preferred_model,
                                auto_response_metric=auto_response_metric,
                                auto_dr_bootstrap=auto_dr_bootstrap,
                                selected_gc_bootstrap=str(auto_bootstrap_scope),
                                selected_preferred_fit=str(auto_preferred_model),
                                selected_response_metric=str(auto_response_metric),
                                selected_dr_bootstrap=str(auto_dr_bootstrap),
                                export_label_filter=export_label_filter,
                                export_dr_include_unsure=bool(export_dr_include_unsure),
                                export_dr_include_invalid=bool(export_dr_include_invalid),
                                stage2_config=infer_res.get("stage2_config"),
                                grofit_tidy_all=grofit_tidy_all,
                            )
                else:
                    st.info("No valid curves found (or all valid curves had insufficient points). Skipping Grofit.")
                grofit_ran = True
                if not run_grofit:
                    zip_name = ""

            st.session_state["last_run_results"] = {
                "final_merged": final_merged,
                "raw_merged": raw_merged,
                "meta_df": infer_res.get("meta_df"),
                "stage2_config": infer_res.get("stage2_config"),
                "out_df": out_df,
                "time_cols_final": time_cols_final,
                "settings": settings,
                "file_stem": Path(uploaded.name).stem,
                "wide_original": wide_df,
                "time_cols_original": time_cols_original,
                "predicting_model": predicting_model,
                "grofit_tidy_all": grofit_tidy_all,
                "gc_fit": gc_fit,
                "dr_fit": dr_fit,
                "gc_boot": gc_boot,
                "dr_boot": dr_boot,
                "gc_audit": gc_audit,
                "dr_audit": dr_audit,
                "audit_df": audit_df,
                "grofit_df": grofit_df,
                "zip_bytes": zip_bytes,
                "zip_name": zip_name if "zip_name" in locals() else "",
                "review_df": review_df,
                "grofit_opts": (
                    grofit_opts_effective_auto
                    if (not manual_review_mode and "grofit_opts_effective_auto" in locals())
                    else grofit_opts
                ),
                "manual_review_mode": manual_review_mode,
                "grofit_ran": grofit_ran,
                "export_label_filter": export_label_filter,
                "export_dr_include_unsure": bool(export_dr_include_unsure),
                "export_dr_include_invalid": bool(export_dr_include_invalid),
            }
            # default curve selection to first Test Id
            if not out_df.empty:
                st.session_state["selected_test_id"] = str(out_df["Test Id"].iloc[0])
            results = st.session_state["last_run_results"]
            render_results(results)

        except Exception as e:
            show_friendly_error(e)
            st.session_state.pop("last_run_results", None)

# On non-run interactions, reuse previous successful results to avoid forcing re-run
if not run and results:
    try:
        with st.spinner("Using cached results."):
            render_results(results)
    except Exception as e:
        show_friendly_error(e)
