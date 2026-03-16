# app/core/plots.py
"""
All Plotly figure-building functions.
No Streamlit dependency — returns go.Figure objects only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils import to_numeric_scalar


# ---------------------------------------------------------------------------
# Bootstrap band helper
# ---------------------------------------------------------------------------

def generate_fast_bootstrap_bands(
    t, y, lam_s, *, t_grid: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Residual bootstrap on the smoothing spline, aligning exactly with
    the orange spline curve already displayed.

    Returns ``(y_q025, y_q975)`` on *t_grid*, or ``(None, None)`` on failure.
    """
    from scipy.interpolate import make_smoothing_spline
    from growthqa.grofit.gc_fit_spline import _dedupe_sorted_xy

    t = np.asarray(t, float)
    y = np.asarray(y, float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if len(t) < 6:
        return None, None
    try:
        lam_s  = float(lam_s) if lam_s is not None else None
        order  = np.argsort(t)
        t_u, y_u = _dedupe_sorted_xy(t[order], y[order])
        sp_base  = make_smoothing_spline(t_u, y_u, lam=lam_s)
        y_base   = sp_base(t_u)
        resid    = y_u - y_base
        boots, rng = [], np.random.default_rng(42)
        for _ in range(200):
            y_b = y_base + rng.choice(resid, size=len(resid), replace=True)
            try:
                boots.append(make_smoothing_spline(t_u, y_b, lam=lam_s)(t_grid))
            except Exception:
                pass
        if len(boots) < 10:
            return None, None
        arr = np.array(boots)
        return np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Simple single-series plot (used in training/debug views)
# ---------------------------------------------------------------------------

def make_simple_plot(df_one: pd.Series, time_cols: list[str], title: str) -> go.Figure:
    xs, ys = [], []
    for c in time_cols:
        try:
            xs.append(float(c.split("(")[0].strip()[1:].strip()))
            ys.append(df_one.get(c, np.nan))
        except Exception:
            continue
    order = np.argsort(xs) if xs else []
    xs = np.array(xs)[order] if xs else np.array([])
    ys = np.array(ys)[order] if ys else np.array([])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers"))
    fig.update_layout(
        title=title, xaxis_title="Time (hours)",
        yaxis_title="Relative OD (normalized)",
        height=420, margin=dict(l=30, r=10, t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Raw vs processed overlay (no payload dict, direct series)
# ---------------------------------------------------------------------------

def make_overlay_plot(
    raw_row: pd.Series, raw_time_cols: list[str],
    proc_row: pd.Series, proc_time_cols: list[str],
    *, title: str, input_is_raw: bool, global_blank: float | None,
) -> go.Figure:

    def _xy(row, cols):
        xs, ys = [], []
        for c in cols:
            try:
                xs.append(float(c.split("(")[0].strip()[1:].strip()))
                ys.append(row.get(c, np.nan))
            except Exception:
                continue
        order = np.argsort(xs) if xs else []
        return (np.array(xs)[order] if xs else np.array([]),
                np.array(ys)[order] if ys else np.array([]))

    raw_x, raw_y   = _xy(raw_row, raw_time_cols)
    proc_x, proc_y = _xy(proc_row, proc_time_cols)
    if input_is_raw and global_blank is not None:
        raw_y = raw_y - float(global_blank)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=raw_x,  y=raw_y,  mode="lines+markers", name="Raw"))
    fig.add_trace(go.Scatter(x=proc_x, y=proc_y, mode="lines+markers", name="Processed"))
    fig.update_layout(
        title=title, xaxis_title="Time (hours)", yaxis_title="Relative OD",
        height=700, margin=dict(l=30, r=10, t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Growth-curve overlay using Grofit payload dict
# ---------------------------------------------------------------------------

def make_overlay_plot_payload(
    payload: dict, *, title: str,
    show_spline: bool, show_model: bool, show_bootstrap: bool,
) -> go.Figure:
    fig = go.Figure()
    SPLINE_COL  = "#f28e2b"                   # orange
    PARAM_COL   = "#2ca02c"                   # green
    TANGENT_COL = "#8b4513"                   # brown
    BOOT_LINE   = "rgba(128,0,128,0.95)"
    BOOT_FILL   = "rgba(128,0,128,0.36)"

    t_raw  = payload.get("t_raw",  np.array([]))
    y_raw  = payload.get("y_raw",  np.array([]))
    t_proc = payload.get("t_proc", np.array([]))
    y_proc = payload.get("y_proc", np.array([]))

    if len(t_raw)  and len(y_raw):
        fig.add_trace(go.Scatter(x=t_raw,  y=y_raw,  mode="markers", name="Raw"))
    if len(t_proc) and len(y_proc):
        fig.add_trace(go.Scatter(x=t_proc, y=y_proc, mode="lines",   name="Processed"))

    spline = payload.get("spline", {})
    if show_spline and spline.get("ran"):
        fig.add_trace(go.Scatter(
            x=spline["t_grid"], y=spline["y_hat"],
            mode="lines", name="Spline Fit",
            line=dict(dash="dash", color=SPLINE_COL),
        ))
        p    = spline.get("params", {})
        t_mu = to_numeric_scalar(p.get("t_mu"))
        y_mu = to_numeric_scalar(p.get("y_mu"))
        mu   = to_numeric_scalar(p.get("mu"))
        lam  = to_numeric_scalar(p.get("lambda"))
        y0   = to_numeric_scalar(p.get("y0"))
        A    = to_numeric_scalar(p.get("A"))

        if np.isfinite(t_mu) and np.isfinite(y_mu):
            fig.add_trace(go.Scatter(
                x=[t_mu], y=[y_mu], mode="markers+text", name="μ point",
                text=["μ"], textposition="top center",
                marker=dict(size=14, color=SPLINE_COL, line=dict(width=1, color="#5a3a16")),
            ))

        if np.isfinite(mu) and np.isfinite(lam) and abs(float(mu)) > 1e-12:
            if np.isfinite(y0) and np.isfinite(A) and A > 0 and np.isfinite(t_mu) and np.isfinite(y_mu):
                x_span = float(A / mu)
                x_line = np.array([float(lam) - 0.15 * x_span, float(lam + A / mu) + 0.15 * x_span])
                fig.add_trace(go.Scatter(
                    x=x_line, y=mu * (x_line - t_mu) + y_mu,
                    mode="lines", name="Tangent (mu)",
                    line=dict(color=TANGENT_COL, width=2),
                ))

        if np.isfinite(lam): fig.add_vline(x=float(lam), line_dash="dot", line_color="#7a6a5f")
        if np.isfinite(y0):  fig.add_hline(y=float(y0),  line_dash="dot", line_color="#7a6a5f")
        if np.isfinite(y0) and np.isfinite(A) and np.isfinite(t_mu):
            fig.add_shape(type="line",
                          x0=float(t_mu), x1=float(t_mu),
                          y0=float(y0),   y1=float(y0 + A),
                          line=dict(color="#7a6a5f", width=2))
            fig.add_annotation(x=float(t_mu), y=float(y0 + A),
                                text=f"A={A:.3g}", showarrow=True, arrowhead=2)

    parametric = payload.get("parametric", {})
    if show_model and parametric.get("ran") and parametric.get("passed_sanity", True):
        fig.add_trace(go.Scatter(
            x=parametric["t_grid"], y=parametric["y_hat"],
            mode="lines", name=f"Model ({parametric.get('model_name', '')})",
            line=dict(dash="dashdot", color=PARAM_COL),
        ))

    bootstrap = payload.get("bootstrap", {})
    if show_bootstrap and bootstrap.get("ran") and bootstrap.get("y_hat_q025") is not None:
        tg = spline.get("t_grid")
        fig.add_trace(go.Scatter(x=tg, y=bootstrap["y_hat_q975"],
                                 line=dict(width=2.5, color=BOOT_LINE),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=tg, y=bootstrap["y_hat_q025"],
                                 line=dict(width=2.5, color=BOOT_LINE),
                                 fill="tonexty", fillcolor=BOOT_FILL,
                                 showlegend=True, name="Bootstrap band", hoverinfo="skip"))

    fig.update_layout(
        title=title, xaxis_title="Time (hours)", yaxis_title="Relative OD",
        height=520, margin=dict(l=30, r=10, t=50, b=40),
        legend=dict(orientation="h", yanchor="top", y=-0.15,
                    xanchor="center", x=0.5, bgcolor="rgba(255,255,255,0.5)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Dose-response plot
# ---------------------------------------------------------------------------

def make_dr_plot(payload: dict, *, show_bootstrap: bool) -> go.Figure:
    fig = go.Figure()
    x = np.asarray(payload.get("x_conc",   []), dtype=float)
    y = np.asarray(payload.get("y_metric", []), dtype=float)
    if len(x) and len(y):
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data"))

    fit    = payload.get("fit", {})
    x_grid = fit.get("x_grid", [])
    y_hat  = fit.get("y_hat",  [])
    if len(x_grid) and len(y_hat):
        fig.add_trace(go.Scatter(x=x_grid, y=y_hat, mode="lines", name="DR Spline"))

    ec50  = to_numeric_scalar(fit.get("ec50"))
    y_mid = to_numeric_scalar(fit.get("y_mid"))
    if np.isfinite(ec50):  fig.add_vline(x=float(ec50),  line_dash="dot", line_color="#7a6a5f")
    if np.isfinite(y_mid): fig.add_hline(y=float(y_mid), line_dash="dot", line_color="#7a6a5f")

    boot = payload.get("bootstrap", {})
    if show_bootstrap and boot.get("ran") and boot.get("y_hat_q025") is not None:
        fig.add_trace(go.Scatter(x=x_grid, y=boot["y_hat_q975"],
                                 line=dict(width=2.5, color="rgba(128,0,128,0.95)"),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x_grid, y=boot["y_hat_q025"],
                                 line=dict(width=2.5, color="rgba(128,0,128,0.95)"),
                                 fill="tonexty", fillcolor="rgba(128,0,128,0.36)",
                                 showlegend=True, name="Bootstrap band", hoverinfo="skip"))

    fig.update_layout(
        xaxis_title="Concentration",
        yaxis_title=payload.get("metric", "response"),
        height=420, margin=dict(l=30, r=10, t=30, b=40),
        legend=dict(orientation="h", yanchor="top", y=-0.15,
                    xanchor="center", x=0.5, bgcolor="rgba(255,255,255,0.5)"),
    )
    return fig
