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
from growthqa.features.meta import compute_late_growth_features
from growthqa.viz.payloads import build_curve_payloads, build_dr_payload
from growthqa.pipelines.infer_labels import run_label_inference_from_uploaded_wide




# -------------------------
# User paths (as requested)
# -------------------------
MODEL_DIR = ROOT / "classifier_output" / "saved_models_selected"
TRAIN_META = ROOT / "data" / "train_data" / "meta.csv"


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
    spline_s: float | None = None
    dr_s: float | None = None
    dr_x_transform: str | None = "log1p"
    bootstrap_method: str = "pairs"


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


def _isfinite_scalar(x) -> bool:
    try:
        return bool(np.isfinite(_to_numeric_scalar(x)))
    except Exception:
        return False


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


def make_overlay_plot_payload(
    payload: dict,
    *,
    title: str,
    show_spline: bool,
    show_model: bool,
    show_bootstrap: bool,
) -> go.Figure:
    fig = go.Figure()

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
                line=dict(dash="dash"),
            )
        )

        params = spline.get("params", {})
        t_mu = _to_numeric_scalar(params.get("t_mu"))
        y_mu = _to_numeric_scalar(params.get("y_mu"))
        if np.isfinite(t_mu) and np.isfinite(y_mu):
            fig.add_trace(
                go.Scatter(
                    x=[t_mu],
                    y=[y_mu],
                    mode="markers+text",
                    name="μ point",
                    text=["μ"],
                    textposition="top center",
                )
            )
        lam = _to_numeric_scalar(params.get("lambda"))
        if np.isfinite(lam):
            fig.add_vline(x=float(lam), line_dash="dot", line_color="#7a6a5f")
        y0 = _to_numeric_scalar(params.get("y0"))
        A = _to_numeric_scalar(params.get("A"))
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
                line=dict(dash="dashdot"),
            )
        )

    bootstrap = payload.get("bootstrap", {})
    if show_bootstrap and bootstrap.get("ran") and bootstrap.get("y_hat_q025") is not None:
        fig.add_trace(
            go.Scatter(
                x=spline.get("t_grid"),
                y=bootstrap.get("y_hat_q975"),
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name="Boot CI",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=spline.get("t_grid"),
                y=bootstrap.get("y_hat_q025"),
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(194,109,58,0.15)",
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=boot.get("y_hat_q025"),
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(194,109,58,0.15)",
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
) -> pd.DataFrame:
    return build_classifier_audit_df(
        wide_original_df=wide_df,
        infer_df=out_df,
        meta_df=meta_df if isinstance(meta_df, pd.DataFrame) else out_df,
        mode="MANUAL" if manual_review_mode else "AUTO",
        review_df=review_df,
    )


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
    grofit_opts: GrofitOptions,
    settings: InferenceSettings,
    mode_label: str,
    file_stem: str,
    predicting_model: str,
    auto_bootstrap_scope: str | None = None,
    auto_preferred_model: str | None = None,
    auto_response_metric: str | None = None,
    auto_dr_bootstrap: str | None = None,
    audit_df: pd.DataFrame | None = None,
    grofit_df: pd.DataFrame | None = None,
    stage2_config: dict | None = None,
) -> tuple[str, bytes]:
    date_tag = datetime.now().strftime("%m.%d.%y")
    zip_name = f"{mode_label}_{date_tag}_{file_stem}.zip"

    classifier_df = audit_df if isinstance(audit_df, pd.DataFrame) else _build_classifier_output(
        wide_df=wide_df,
        out_df=out_df,
        review_df=review_df,
        manual_review_mode=(mode_label == "MANUAL"),
    )
    grofit_input_df = grofit_df if isinstance(grofit_df, pd.DataFrame) else _build_grofit_input_df(
        wide_df=wide_df,
        out_df=out_df,
        review_df=review_df,
        manual_review_mode=(mode_label == "MANUAL"),
        audit_df=classifier_df,
    )

    run_info = {
        "mode": mode_label,
        "timestamp": datetime.now().isoformat(),
        "file_stem": file_stem,
        "predicting_model": predicting_model,
        "grofit_options": grofit_opts.__dict__,
        "settings": settings.__dict__,
        "stage2_thresholds": stage2_config,
        "auto_options": {
            "bootstrap_scope": auto_bootstrap_scope,
            "preferred_model": auto_preferred_model,
            "response_metric": auto_response_metric,
            "dr_bootstrap": auto_dr_bootstrap,
        }
        if mode_label == "AUTO"
        else None,
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
        zf.writestr("Grofit.csv", _df_bytes(grofit_input_df))
        zf.writestr("gcFit.csv", _df_bytes(gc_fit))
        zf.writestr("gcBoot.csv", _df_bytes(gc_boot))
        if isinstance(dr_fit, pd.DataFrame) and not dr_fit.empty:
            zf.writestr("drFit.csv", _df_bytes(dr_fit))
        if isinstance(dr_boot, pd.DataFrame) and not dr_boot.empty:
            zf.writestr("drBoot.csv", _df_bytes(dr_boot))
        zf.writestr("run_info.json", json.dumps(run_info, indent=2))
        zf.writestr("plots/README.txt", "Plots were not generated in this run.")

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
    meta_df = results.get("meta_df", out_df)

    if review_df is not None:
        if st.session_state.get("review_df") is None:
            st.session_state["review_df"] = review_df.copy()
        review_df = st.session_state.get("review_df", review_df)

    gc_fit = results.get("gc_fit", pd.DataFrame())
    dr_fit = results.get("dr_fit", pd.DataFrame())
    gc_boot = results.get("gc_boot", pd.DataFrame())
    dr_boot = results.get("dr_boot", pd.DataFrame())
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
            for c in ["pred_label", "pred_confidence", "confidence_valid", "confidence_invalid", "too_sparse", "low_resolution", "had_outliers"]
            if c in out_df.columns
        ]
        if merge_cols:
            curve_df = curve_df.merge(out_df[["Test Id"] + merge_cols], on="Test Id", how="left")
    else:
        curve_df = out_df.copy()
        if "Concentration" not in curve_df.columns:
            curve_df["Concentration"] = curve_df["Test Id"].map(conc_map)

    pred_col = "pred_label"

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

    # st.markdown("---")
    # st.markdown("#### Review Section")

    # top_left, top_right = st.columns([2.2, 1.0], gap="large")
    # with top_left:
    #     filter_col, select_col = st.columns([1, 2])
    #     with filter_col:
    #         filter_choice = st.selectbox("Filter", options=["All", "Valid", "Invalid", "Unsure"], key="curve_filter")
    # with top_right:
    #     st.markdown("", unsafe_allow_html=True)
    # if filter_choice != "All":
    #     filtered = curve_df[curve_df[pred_col] == filter_choice].copy()
    # else:
    #     filtered = curve_df.copy()

    # base_order = (
    #     wide_original["Test Id"].astype(str).tolist()
    #     if wide_original is not None and "Test Id" in wide_original.columns
    #     else curve_df["Test Id"].astype(str).tolist()
    # )
    # allowed = set(filtered["Test Id"].astype(str).tolist())
    # options = [tid for tid in base_order if tid in allowed]
    # if not options:
    #     st.warning("No curves match the current filter.")
    #     return

    # pending = st.session_state.pop("pending_curve_select", None)
    # if pending in options:
    #     st.session_state["curve_select"] = pending
    #     st.session_state["selected_test_id"] = pending

    # if st.session_state.get("selected_test_id") not in options:
    #     st.session_state["selected_test_id"] = options[0]

    # with select_col:
    #     current_sel = st.session_state.get("curve_select", st.session_state.get("selected_test_id"))
    #     index_val = options.index(str(current_sel)) if str(current_sel) in options else 0
    #     chosen = st.selectbox(
    #         "Select Test Id + Conc",
    #         options,
    #         index=index_val,
    #         key="curve_select",
    #         format_func=_label_with_conc,
    #     )

    # st.session_state["selected_test_id"] = chosen

    # def _nav_set(new_idx: int):
    #     new_val = options[new_idx]
    #     st.session_state["selected_test_id"] = new_val
    #     st.session_state["pending_curve_select"] = new_val
    #     st.rerun()

    # row = curve_df.loc[curve_df["Test Id"].astype(str) == str(chosen)].iloc[0]
    # out_row = out_df.loc[out_df["Test Id"].astype(str) == str(chosen)].iloc[0] if not out_df.empty else row
    # pred_label = _normalize_label(row.get(pred_col, ""))
    # if manual_review_mode:
    #     final_label = _normalize_label(row.get("final_label", pred_label))
    #     if not final_label:
    #         final_label_bool = bool(row.get("is_valid_final", _label_is_valid(pred_label)))
    #         final_label = "Valid" if final_label_bool else "Invalid"
    # else:
    #     final_label = pred_label

    # zip_ready = bool(grofit_ran and zip_bytes)

    # left, right = st.columns([2.2, 1.0], gap="small")
    # with left:
    #     st.markdown('<div class="curve-title">Curve Overlay</div>', unsafe_allow_html=True)
    #     if wide_original is not None and not final_merged.empty:
    #         raw_row = wide_original.loc[wide_original["Test Id"].astype(str) == str(chosen)]
    #         proc_row = final_merged.loc[final_merged["Test Id"].astype(str) == str(chosen)]
    #         if not raw_row.empty and not proc_row.empty:
    #             fig = make_overlay_plot(
    #                 raw_row.iloc[0],
    #                 time_cols_original,
    #                 proc_row.iloc[0],
    #                 time_cols_final,
    #                 title=f"{chosen}",
    #                 input_is_raw=settings.input_is_raw,
    #                 global_blank=settings.global_blank,
    #             )
    #             fig.update_layout(height=520)
    #             st.plotly_chart(fig, use_container_width=True)
    #             if manual_review_mode:
    #                 nav_left, nav_right = st.columns(2)
    #                 if nav_left.button("Previous Curve", key="prev_curve_overlay", use_container_width=True):
    #                     idx = options.index(str(chosen))
    #                     _nav_set((idx - 1) % len(options))
    #                 if nav_right.button("Next Curve", key="next_curve_overlay", use_container_width=True):
    #                     idx = options.index(str(chosen))
    #                     _nav_set((idx + 1) % len(options))
    #         else:
    #             st.warning("Could not find curve data for overlay plot.")
    #     else:
    #         st.warning("No time columns found for plotting.")

    #     with st.expander("How this plot is built?", expanded=False):
    #         st.markdown(
    #             "- Timepoints are extracted from column headers like `T1.0 (h)` and sorted numerically.\n"
    #             "- Values come from the preprocessed table after interpolation, smoothing, and blank handling.\n"
    #             "- Y-axis is Relative OD (normalized).\n"
    #             "- Plot shows the model-ready curve; it should closely match the cleaned measurement trace used for prediction."
    #         )
    #     if manual_review_mode:
    #         exit_left, exit_right = st.columns([1.2, 1.0])
    #         if exit_right.button("Exit Review Mode", key="exit_review_mode", use_container_width=True):
    #             st.session_state["show_exit_review"] = True
    #         if st.session_state.get("show_exit_review"):
    #             exit_c1, exit_c2 = st.columns(2)
    #             exit_c3, exit_c4 = st.columns(2)
    #             gc_bootstrap = exit_c1.selectbox(
    #                 "GC Bootstrap",
    #                 options=["None", "Only Valid Curves"],
    #                 index=0,
    #                 key="exit_gc_bootstrap",
    #             )
    #             preferred_model = exit_c2.selectbox(
    #                 "Preferred Model",
    #                 options=["Best Model", "Spline", "Parametric"],
    #                 index=0,
    #                 key="exit_preferred_model",
    #             )
    #             response_metric = exit_c3.selectbox(
    #                 "Response Metric",
    #                 options=["mu", "A", "lag", "integral"],
    #                 index=0,
    #                 key="exit_response_metric",
    #             )
    #             dr_bootstrap = exit_c4.selectbox(
    #                 "DR Bootstrap",
    #                 options=["True", "False"],
    #                 index=0,
    #                 key="exit_dr_bootstrap",
    #             )
    #             run_left, run_right = st.columns([1.2, 1.0])
    #             if run_right.button("RUN", key="exit_review_run", use_container_width=True):
    #                 with st.spinner("Preparing Grofit raw file (raw OD + final labels) and running Grofit..."):
    #                     validity_map = review_df.set_index("Test Id")["is_valid_final"].to_dict()
    #                     grofit_tidy_all = wide_original_to_grofit_tidy(wide_original, file_tag=file_stem)
    #                     grofit_tidy_all["is_valid_final"] = grofit_tidy_all["curve_id"].map(validity_map).fillna(False).astype(bool)
    #                     gc_fit = pd.DataFrame()
    #                     dr_fit = pd.DataFrame()
    #                     gc_boot = pd.DataFrame()
    #                     dr_boot = pd.DataFrame()
    #                     zip_bytes = b""

    #                     preferred_fit_map = {
    #                         "Best Model": "Both (param + spline)",
    #                         "Spline": "Spline only",
    #                         "Parametric": "Parametric only",
    #                     }
    #                     fit_opt_map = {
    #                         "Both (param + spline)": "b",
    #                         "Parametric only": "m",
    #                         "Spline only": "s",
    #                     }
    #                     fit_mode_label = preferred_fit_map.get(preferred_model, "Both (param + spline)")
    #                     fit_opt = fit_opt_map.get(fit_mode_label, "b")
    #                     gc_boot_B = 0 if gc_bootstrap == "None" else grofit_opts.gc_boot_B
    #                     dr_boot_B = grofit_opts.dr_boot_B if dr_bootstrap == "True" else 0

    #                     if grofit_tidy_all["is_valid_final"].any():
    #                         with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as grofit_td:
    #                             grofit_res = run_grofit_pipeline(
    #                                 curves_df=grofit_tidy_all,
    #                                 response_var=response_metric,
    #                                 have_atleast=grofit_opts.have_atleast,
    #                                 gc_boot_B=gc_boot_B,
    #                                 dr_boot_B=dr_boot_B,
    #                                 spline_auto_cv=grofit_opts.spline_auto_cv,
    #                                 spline_s=grofit_opts.spline_s,
    #                                 dr_x_transform=grofit_opts.dr_x_transform,
    #                                 dr_s=grofit_opts.dr_s,
    #                                 fit_opt=fit_opt,
    #                                 bootstrap_method=grofit_opts.bootstrap_method,
    #                                 random_state=42,
    #                                 export_dir=Path(grofit_td),
    #                             )
    #                             gc_fit = grofit_res.get("gc_fit", pd.DataFrame())
    #                             dr_fit = grofit_res.get("dr_fit", pd.DataFrame())
    #                             gc_boot = grofit_res.get("gc_boot", pd.DataFrame())
    #                             dr_boot = grofit_res.get("dr_boot", pd.DataFrame())
    #                             zip_bytes = grofit_res.get("zip_bytes", b"")
    #                     else:
    #                         st.info("No valid curves found (or all valid curves had insufficient points). Skipping Grofit.")

    #                 results["grofit_tidy_all"] = grofit_tidy_all
    #                 results["gc_fit"] = gc_fit
    #                 results["dr_fit"] = dr_fit
    #                 results["gc_boot"] = gc_boot
    #                 results["dr_boot"] = dr_boot
    #                 results["zip_bytes"] = zip_bytes
    #                 results["grofit_ran"] = True
    #                 st.session_state["last_run_results"] = results
    #                 st.rerun()

    # with right:
    #     total_reviewed_label = f"{reviewed_count}/{total_curves}"
    #     correct_reviewed_label = f"{correct_count}/{reviewed_count}" if reviewed_count > 0 else "0/0"
    #     metric_font = "1.1rem"
    #     st.markdown('<div class="metrics-panel">', unsafe_allow_html=True)
    #     def _fmt_val(val: object) -> str:
    #         if val is None or (isinstance(val, float) and pd.isna(val)):
    #             return "NA"
    #         if isinstance(val, bool):
    #             return "True" if val else "False"
    #         return str(val)

    #     def _fmt_metric(val: object) -> str:
    #         if val is None or (isinstance(val, float) and pd.isna(val)):
    #             return "NA"
    #         if isinstance(val, (int, float, np.floating)):
    #             return f"{float(val):.4f}"
    #         return str(val)

    #     def _render_row(label: str, value: str | None = None) -> None:
    #         r_label, r_value = st.columns([1.2, 1.8], gap="small")
    #         r_label.markdown(
    #             f"<div style='font-size:{metric_font};font-weight:600;'>{label}</div>",
    #             unsafe_allow_html=True,
    #         )
    #         if value is None:
    #             r_value.markdown(f"<div style='font-size:{metric_font};'>NA</div>", unsafe_allow_html=True)
    #         else:
    #             r_value.markdown(
    #                 f"<div style='font-size:{metric_font};'>{value}</div>",
    #                 unsafe_allow_html=True,
    #             )

    #     def _render_select_row(label: str, options_list: list[str], index_val: int, key_val: str) -> str:
    #         r_label, r_value, r_spacer = st.columns([1.2, 0.78, 1.02], gap="small")
    #         r_label.markdown(
    #             f"<div style='font-size:{metric_font};font-weight:600;'>{label}</div>",
    #             unsafe_allow_html=True,
    #         )
    #         sel = r_value.selectbox(
    #             label,
    #             options=options_list,
    #             index=index_val,
    #             key=key_val,
    #             label_visibility="collapsed",
    #         )
    #         r_spacer.markdown("")
    #         return sel

    #     if manual_review_mode:
    #         st.markdown(
    #             (
    #                 "<div class='metrics-top'>"
    #                 f"<div class='total-reviewed'>"
    #                 f"Total Reviewed: {total_reviewed_label} | Correctly Predicted: {correct_reviewed_label}"
    #                 "</div>"
    #                 "</div>"
    #                 "<div class='metrics-title'>Metrics</div>"
    #             ),
    #             unsafe_allow_html=True,
    #         )
    #     _render_row("Predicted Label", _fmt_val(pred_label))
    #     if manual_review_mode and isinstance(review_df, pd.DataFrame) and not review_df.empty:
    #         curve_key = row.get("CurveKey")
    #         if not curve_key:
    #             curve_key = _make_curve_key(str(row["Test Id"]), row.get("Concentration"))
    #         label_key = f"final_label_{curve_key}"
    #         reviewed_key = f"reviewed_{curve_key}"
    #         if label_key not in st.session_state:
    #             pred_norm = _normalize_label(pred_label)
    #             st.session_state[label_key] = pred_norm if pred_norm in {"Valid", "Invalid", "Unsure"} else "Unsure"
    #         if reviewed_key not in st.session_state:
    #             st.session_state[reviewed_key] = "True" if bool(row.get("Reviewed", False)) else "False"
    #         elif isinstance(st.session_state[reviewed_key], bool):
    #             st.session_state[reviewed_key] = "True" if st.session_state[reviewed_key] else "False"

    #         new_final = _render_select_row(
    #             "True Label",
    #             ["Valid", "Invalid", "Unsure"],
    #             ["Valid", "Invalid", "Unsure"].index(st.session_state[label_key]),
    #             label_key,
    #         )
    #         reviewed_val = _render_select_row(
    #             "Reviewed",
    #             ["False", "True"],
    #             1 if st.session_state[reviewed_key] == "True" else 0,
    #             reviewed_key,
    #         )
    #         reviewed_bool = reviewed_val == "True"

    #         if new_final != _normalize_label(row.get("final_label", pred_label)) or reviewed_bool != bool(row.get("Reviewed", False)):
    #             review_df.loc[review_df["CurveKey"] == curve_key, "final_label"] = new_final
    #             review_df.loc[review_df["CurveKey"] == curve_key, "is_valid_final"] = (new_final == "Valid")
    #             review_df.loc[review_df["CurveKey"] == curve_key, "Reviewed"] = reviewed_bool
    #             st.session_state["review_df"] = review_df
    #             st.rerun()
    #     fit_row = None
    #     if isinstance(gc_fit, pd.DataFrame) and not gc_fit.empty:
    #         fit_match = gc_fit[
    #             (gc_fit["test.id"].astype(str) == str(file_stem))
    #             & (gc_fit["add.id"].astype(str) == str(chosen))
    #         ]
    #         if not fit_match.empty:
    #             fit_row = fit_match.iloc[0]

    #     boot_row = None
    #     if isinstance(gc_boot, pd.DataFrame) and not gc_boot.empty:
    #         boot_match = gc_boot[
    #             (gc_boot["test.id"].astype(str) == str(file_stem))
    #             & (gc_boot["add.id"].astype(str) == str(chosen))
    #         ]
    #         if not boot_match.empty:
    #             boot_row = boot_match.iloc[0]

    #     _render_row("Confidence (Valid)", _fmt_metric(out_row.get("confidence_valid", "NA")))
    #     _render_row("mu.spline", _fmt_metric(fit_row.get("mu.spline") if fit_row is not None else None))
    #     _render_row("Use.model", _fmt_metric(fit_row.get("use.model") if fit_row is not None else None))
    #     _render_row("lambda.spline", _fmt_metric(fit_row.get("lambda.spline") if fit_row is not None else None))
    #     _render_row("A.nonpara", _fmt_metric(fit_row.get("A.nonpara") if fit_row is not None else None))
    #     _render_row("integral.spline", _fmt_metric(fit_row.get("integral.spline") if fit_row is not None else None))
    #     _render_row("mu.model", _fmt_metric(fit_row.get("mu.model") if fit_row is not None else None))
    #     _render_row("lambda.model", _fmt_metric(fit_row.get("lambda.model") if fit_row is not None else None))
    #     _render_row("A.para", _fmt_metric(fit_row.get("A.para") if fit_row is not None else None))
    #     _render_row("Integral.model", _fmt_metric(fit_row.get("Integral.model") if fit_row is not None else None))
    #     _render_row("mu.bt", _fmt_metric(boot_row.get("mu.bt") if boot_row is not None else None))
    #     _render_row("lambda.bt", _fmt_metric(boot_row.get("lambda.bt") if boot_row is not None else None))
    #     _render_row("A.bt", _fmt_metric(boot_row.get("A.bt") if boot_row is not None else None))
    #     _render_row("integral.bt", _fmt_metric(boot_row.get("integral.bt") if boot_row is not None else None))
    #     _render_row("Too Sparse", _fmt_val(bool(out_row.get("too_sparse")) if "too_sparse" in out_row else "NA"))
    #     _render_row("Low Resolution", _fmt_val(bool(out_row.get("low_resolution")) if "low_resolution" in out_row else "NA"))
    #     _render_row(
    #         "Blank subtraction mode",
    #         _fmt_val("RAW (applied)" if settings.input_is_raw else "ALREADY BLANK SUBTRACTED (so not applied)"),
    #     )
    #     st.markdown("</div>", unsafe_allow_html=True)

    #     st.markdown("---")
    #     if manual_review_mode and isinstance(review_df, pd.DataFrame) and not review_df.empty:
    #         st.markdown('<div class="review-actions">', unsafe_allow_html=True)
    #         curve_key = row.get("CurveKey")
    #         if not curve_key:
    #             curve_key = _make_curve_key(str(row["Test Id"]), row.get("Concentration"))

    #         review_c1, review_c2, review_c3 = st.columns(3)
    #         # Reset button intentionally disabled.
    #         if False:
    #             with review_c1:
    #                 if st.button("Reset", key=f"reset_{curve_key}", use_container_width=True):
    #                     review_df.loc[review_df["CurveKey"] == curve_key, "is_valid_final"] = review_df.loc[
    #                         review_df["CurveKey"] == curve_key, "is_valid_pred"
    #                     ].astype(bool)
    #                     review_df.loc[review_df["CurveKey"] == curve_key, "final_label"] = review_df.loc[
    #                         review_df["CurveKey"] == curve_key, "pred_label"
    #                     ].astype(str)
    #                     review_df.loc[review_df["CurveKey"] == curve_key, "Reviewed"] = False
    #                     st.session_state["review_df"] = review_df
    #                     st.rerun()
    #         # Show Random Curve intentionally disabled.
    #         if False:
    #             if review_c2.button("Show Random Curve", key="rand_curve", use_container_width=True):
    #                 _nav_set(np.random.randint(0, len(options)))

    #         st.markdown('<div class="action-buttons">', unsafe_allow_html=True)
    #         # File uploader intentionally disabled.
    #         if False:
    #             b_col = st.columns(1)[0]
    #             with b_col:
    #                 st.markdown('<div class="upload-btn">', unsafe_allow_html=True)
    #                 # Upload action: apply a batch of true labels to the current review set.
    #                 review_upload = st.file_uploader(
    #                     "Upload True Labels",                 # label hidden anyway
    #                     type=["csv", "xlsx"],
    #                     accept_multiple_files=False,
    #                     key="manual_review_upload",
    #                     label_visibility="collapsed",         # hide the label line completely
    #                     help="Upload a file with Raw Input Data and a 'True Label' column.",
    #                 )
    #                 st.markdown("</div>", unsafe_allow_html=True)
    #                 if review_upload is not None:
    #                     last_name = st.session_state.get("manual_review_upload_name")
    #                     if last_name != review_upload.name:
    #                         try:
    #                             if review_upload.name.lower().endswith(".csv"):
    #                                 review_in = pd.read_csv(review_upload)
    #                             else:
    #                                 review_in = pd.read_excel(review_upload)
    #                             cols_lower = {str(c).strip().lower(): c for c in review_in.columns}
    #                             test_col = cols_lower.get("test id") or cols_lower.get("test_id") or cols_lower.get("testid")
    #                             true_col = cols_lower.get("true label") or cols_lower.get("true_label") or cols_lower.get("final label") or cols_lower.get("final_label")
    #                             conc_col = cols_lower.get("concentration") or cols_lower.get("conc")
    #                             if not test_col or not true_col:
    #                                 st.error("Uploaded file must contain 'Test Id' and 'True Label' columns.")
    #                             else:
    #                                 for _, r in review_in.iterrows():
    #                                     tid = str(r.get(test_col, "")).strip()
    #                                     if not tid:
    #                                         continue
    #                                     true_lbl = _normalize_label(r.get(true_col, ""))
    #                                     true_bool = _label_is_valid(true_lbl)
    #                                     if conc_col and conc_col in r and "Concentration" in review_df.columns:
    #                                         ck = _make_curve_key(tid, r.get(conc_col))
    #                                         mask = review_df["CurveKey"] == ck
    #                                     else:
    #                                         mask = review_df["Test Id"].astype(str) == tid
    #                                     if not mask.any():
    #                                         continue
    #                                     if true_lbl:
    #                                         review_df.loc[mask, "final_label"] = true_lbl
    #                                         review_df.loc[mask, "is_valid_final"] = true_bool
    #                                         review_df.loc[mask, "Reviewed"] = True
    #                                     else:
    #                                         review_df.loc[mask, "Reviewed"] = False
    #                                         review_df.loc[mask, "final_label"] = review_df.loc[mask, "pred_label"].astype(str)
    #                                         review_df.loc[mask, "is_valid_final"] = review_df.loc[mask, "is_valid_pred"]

    #                                 st.session_state["review_df"] = review_df
    #                                 st.session_state["manual_review_upload_name"] = review_upload.name
    #                                 st.success("Manual review file applied.")
    #                                 st.rerun()
    #                         except Exception as e:
    #                             st.error(f"Failed to process upload: {e}")
    #         st.markdown("</div>", unsafe_allow_html=True)

    #         st.markdown("</div>", unsafe_allow_html=True)
    #     else:
    #         nav_c1, nav_c2, nav_c3 = st.columns(3)
    #         if nav_c1.button("Show Previous", key="prev_curve_auto"):
    #             idx = options.index(str(chosen))
    #             _nav_set((idx - 1) % len(options))
    #         if nav_c2.button("Show Random Curve", key="rand_curve_auto"):
    #             _nav_set(np.random.randint(0, len(options)))
    #         if nav_c3.button("Show Next", key="next_curve_auto"):
    #             idx = options.index(str(chosen))
    #             _nav_set((idx + 1) % len(options))

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
            filtered = curve_df[curve_df[pred_col] == filter_choice].copy()
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
            spline_auto_cv=grofit_opts.spline_auto_cv,
            include_bootstrap=bool(isinstance(gc_boot, pd.DataFrame) and not gc_boot.empty),
            test_id=active_test_id,
            curve_ids=[str(chosen)],
        )
        curve_payload = payloads.get(str(chosen))


    # ============================
    # ROW 2: Curve overlay (left) + Metrics table content (right)
    # ============================
    left, right = st.columns([2.2, 1.0], gap="large")

    # Precompute DR payload for split view
    dr_available = isinstance(dr_fit, pd.DataFrame) and not dr_fit.empty
    dr_payload = None
    if dr_available:
        label_source = "final" if manual_review_mode else "pred"
        dr_payload = build_dr_payload(
            gc_fit=gc_fit,
            labels_df=labels_df,
            dr_boot=dr_boot,
            test_id=active_test_id,
            response_metric=grofit_opts.response_var if grofit_opts.response_var in {"mu", "A", "lambda", "integral"} else "mu",
            label_source=label_source,
            include_unsure=False,
            include_invalid=False,
            dr_s=grofit_opts.dr_s,
            dr_x_transform=grofit_opts.dr_x_transform,
            show_bootstrap=bool(isinstance(dr_boot, pd.DataFrame) and not dr_boot.empty),
        )
        if not dr_payload or not dr_payload.get("n_points"):
            dr_available = False

    with left:
        st.markdown('<div class="curve-title">Curve Viewer</div>', unsafe_allow_html=True)
        show_c1, show_c2, show_c3 = st.columns(3)
        show_spline = show_c1.checkbox("Show spline", value=True, key="show_spline_overlay")
        show_model = show_c2.checkbox("Show model", value=True, key="show_model_overlay")
        boot_available = bool(isinstance(gc_boot, pd.DataFrame) and not gc_boot.empty)
        show_bootstrap = show_c3.checkbox(
            "Show bootstrap",
            value=False,
            key="show_boot_overlay",
            disabled=not boot_available,
        )

        if wide_original is not None and not final_merged.empty:
            raw_row = wide_original.loc[wide_original["Test Id"].astype(str) == str(chosen)]
            proc_row = final_merged.loc[final_merged["Test Id"].astype(str) == str(chosen)]
            if not raw_row.empty and not proc_row.empty:
                if dr_available and dr_payload:
                    ov_col, dr_col = st.columns(2)
                    with ov_col:
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
                        fig.update_layout(height=580)
                        st.plotly_chart(fig, use_container_width=True)
                    with dr_col:
                        dr_fig = make_dr_plot(
                            dr_payload,
                            show_bootstrap=bool(isinstance(dr_boot, pd.DataFrame) and not dr_boot.empty),
                        )
                        dr_fig.update_layout(height=580)
                        st.plotly_chart(dr_fig, use_container_width=True)
                else:
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
                    fig.update_layout(height=520)
                    st.plotly_chart(fig, use_container_width=True)

                nav_left, nav_right = st.columns(2)
                if nav_left.button("Previous Curve", key="prev_curve_overlay", use_container_width=True):
                    idx = options.index(str(chosen))
                    _nav_set((idx - 1) % len(options))
                if nav_right.button("Next Curve", key="next_curve_overlay", use_container_width=True):
                    idx = options.index(str(chosen))
                    _nav_set((idx + 1) % len(options))

                if manual_review_mode:
                    action_left, action_right = st.columns(2)
                    boot_list = st.session_state.get("bootstrap_curve_ids", [])
                    is_added = str(chosen) in boot_list
                    st.markdown(
                        '<div class="boot-btn added">' if is_added else '<div class="boot-btn">',
                        unsafe_allow_html=True,
                    )
                    if action_left.button(
                        "Remove This Curve From Bootstrap" if is_added else "Add This Curve To Bootstrap",
                        key="add_boot_curve",
                        use_container_width=True,
                    ):
                        if is_added:
                            boot_list = [c for c in boot_list if c != str(chosen)]
                        else:
                            boot_list.append(str(chosen))
                        st.session_state["bootstrap_curve_ids"] = boot_list
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                    run_exit_grofit = False
                    if st.session_state.get("show_exit_review"):
                        if action_right.button("RUN GROFIT", key="exit_review_mode", use_container_width=True):
                            run_exit_grofit = True
                    else:
                        if action_right.button("Exit Review Mode", key="exit_review_mode", use_container_width=True):
                            st.session_state["show_exit_review"] = True
                            st.rerun()
                    if st.session_state.get("show_exit_review"):
                        boot_list = st.session_state.get("bootstrap_curve_ids", [])
                        boot_label = f"Selected Curves ({len(boot_list)})"
                        boot_options = ["None", "Only Valid Curves"]
                        if boot_list:
                            boot_options.append(boot_label)
                        exit_c1, exit_c2 = st.columns(2)
                        exit_c3, exit_c4 = st.columns(2)
                        gc_bootstrap = exit_c1.selectbox(
                            "GC Bootstrap",
                            options=boot_options,
                            index=0,
                            key="exit_gc_bootstrap",
                        )
                        preferred_model = exit_c2.selectbox(
                            "Preferred Model",
                            options=["Best Model", "Spline", "Parametric"],
                            index=0,
                            key="exit_preferred_model",
                        )
                        response_metric = exit_c3.selectbox(
                            "Response Metric",
                            options=["mu", "A", "lag", "integral"],
                            index=0,
                            key="exit_response_metric",
                        )
                        dr_bootstrap = exit_c4.selectbox(
                            "DR Bootstrap",
                            options=["True", "False"],
                            index=0,
                            key="exit_dr_bootstrap",
                        )
                        cancel_left, cancel_right = st.columns([1.2, 1.0])
                        if cancel_right.button("Cancel", key="exit_review_cancel", use_container_width=True):
                            st.session_state["show_exit_review"] = False
                            st.rerun()
                    if run_exit_grofit:
                        gc_bootstrap = st.session_state.get("exit_gc_bootstrap", "None")
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

                            gc_boot_B = 0 if gc_bootstrap == "None" else grofit_opts.gc_boot_B
                            dr_boot_B = grofit_opts.dr_boot_B if dr_bootstrap == "True" else 0
                            validity_col = "__all__"

                            gc_fit = pd.DataFrame()
                            dr_fit = pd.DataFrame()
                            gc_boot = pd.DataFrame()
                            dr_boot = pd.DataFrame()
                            zip_bytes = b""
                            zip_name = ""

                            grofit_tidy_all_all = grofit_tidy_all.copy()
                            grofit_tidy_all_all["is_valid_true"] = True
                            if not grofit_tidy_all.empty:
                                with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as grofit_td:
                                    grofit_res = run_grofit_pipeline(
                                        curves_df=grofit_tidy_all_all if (boot_list and gc_bootstrap == boot_label) else grofit_tidy_all,
                                        response_var=response_metric,
                                        have_atleast=grofit_opts.have_atleast,
                                        gc_boot_B=gc_boot_B,
                                        dr_boot_B=dr_boot_B,
                                        spline_auto_cv=grofit_opts.spline_auto_cv,
                                        spline_s=grofit_opts.spline_s,
                                        dr_x_transform=grofit_opts.dr_x_transform,
                                        dr_s=grofit_opts.dr_s,
                                        fit_opt=fit_opt,
                                        bootstrap_method=grofit_opts.bootstrap_method,
                                        validity_col=validity_col,
                                        random_state=42,
                                        export_dir=Path(grofit_td),
                                    )
                                    gc_fit = grofit_res.get("gc_fit", pd.DataFrame())
                                    dr_fit = grofit_res.get("dr_fit", pd.DataFrame())
                                    gc_boot = grofit_res.get("gc_boot", pd.DataFrame())
                                    dr_boot = grofit_res.get("dr_boot", pd.DataFrame())
                                    zip_name, zip_bytes = _build_export_zip(
                                        wide_df=wide_original,
                                        out_df=out_df,
                                        review_df=review_df,
                                        gc_fit=gc_fit,
                                        gc_boot=gc_boot,
                                        dr_fit=dr_fit,
                                        dr_boot=dr_boot,
                                        grofit_opts=grofit_opts,
                                        settings=settings,
                                        mode_label="MANUAL",
                                        file_stem=file_stem,
                                        predicting_model=predicting_model,
                                        audit_df=results.get("audit_df"),
                                        grofit_df=results.get("grofit_df"),
                                        stage2_config=results.get("stage2_config"),
                                    )

                            if boot_list and gc_bootstrap == boot_label and not gc_boot.empty:
                                gc_boot = gc_boot[gc_boot["add.id"].astype(str).isin(boot_list)].copy()

                        results["grofit_tidy_all"] = grofit_tidy_all
                        results["gc_fit"] = gc_fit
                        results["dr_fit"] = dr_fit
                        results["gc_boot"] = gc_boot
                        results["dr_boot"] = dr_boot
                        results["zip_bytes"] = zip_bytes
                        results["zip_name"] = zip_name if "zip_name" in locals() else ""
                        results["grofit_ran"] = True
                        st.session_state["last_run_results"] = results
                        st.session_state["show_exit_review"] = False
                        st.rerun()
            else:
                st.warning("Could not find curve data for overlay plot.")
        else:
            st.warning("No time columns found for plotting.")

    with right:
        # IMPORTANT: metrics content height matches plot height due to CSS height:520px
        st.markdown('<div class="metrics-panel">', unsafe_allow_html=True)

        # (Keep your existing _fmt_val, _fmt_metric, _render_row, _render_select_row exactly the same)
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

        if curve_payload:
            spline_params = curve_payload.get("spline", {}).get("params", {})
            parametric = curve_payload.get("parametric", {})
            param_params = parametric.get("params", {})
            boot_ci = curve_payload.get("bootstrap", {}).get("ci", {})
            has_boot_vals = boot_row is not None and any(
                pd.notna(boot_row.get(k, np.nan)) for k in ["mu.bt", "lambda.bt", "A.bt", "integral.bt"]
            )
            model_name = str(parametric.get("model_name") or "Model")

            rows_html = [
                (
                    "&mu;",
                    _fmt_ci_metric(spline_params.get("mu"), boot_ci.get("mu")),
                    _fmt_metric(param_params.get("mu")),
                    _fmt_metric(boot_row.get("mu.bt")) if has_boot_vals else None,
                ),
                (
                    "&lambda;",
                    _fmt_ci_metric(spline_params.get("lambda"), boot_ci.get("lambda")),
                    _fmt_metric(param_params.get("lambda")),
                    _fmt_metric(boot_row.get("lambda.bt")) if has_boot_vals else None,
                ),
                (
                    "A",
                    _fmt_ci_metric(spline_params.get("A"), boot_ci.get("A")),
                    _fmt_metric(param_params.get("A")),
                    _fmt_metric(boot_row.get("A.bt")) if has_boot_vals else None,
                ),
                (
                    "Integral",
                    _fmt_ci_metric(spline_params.get("integral"), boot_ci.get("integral")),
                    _fmt_metric(param_params.get("integral")),
                    _fmt_metric(boot_row.get("integral.bt")) if has_boot_vals else None,
                ),
            ]

            header_boot = "<th style='border:1px solid #8f8f8f;padding:4px 6px;text-align:left;'>Bootstrap</th>" if has_boot_vals else ""
            body = []
            for metric, spline_v, model_v, boot_v in rows_html:
                boot_cell = f"<td style='border:1px solid #8f8f8f;padding:3px 6px;'>{boot_v}</td>" if has_boot_vals else ""
                body.append(
                    "<tr>"
                    f"<td style='border:1px solid #8f8f8f;padding:3px 6px;'>{metric}</td>"
                    f"<td style='border:1px solid #8f8f8f;padding:3px 6px;'>{spline_v}</td>"
                    f"<td style='border:1px solid #8f8f8f;padding:3px 6px;'>{model_v}</td>"
                    f"{boot_cell}"
                    "</tr>"
                )

            st.markdown(
                (
                    "<table style='width:100%;border-collapse:collapse;border:1px solid #8f8f8f;margin:70px 0 20px 0;'>"
                    "<thead><tr>"
                    "<th style='border:1px solid #8f8f8f;padding:4px 6px;text-align:left;width:20%;'>Metric</th>"
                    "<th style='border:1px solid #8f8f8f;padding:4px 6px;text-align:left;'>Spline</th>"
                    f"<th style='border:1px solid #8f8f8f;padding:4px 6px;text-align:left;'>{model_name}</th>"
                    f"{header_boot}"
                    "</tr></thead>"
                    "<tbody>"
                    + "".join(body)
                    + "</tbody></table>"
                ),
                unsafe_allow_html=True,
            )
        _render_row("Predicted Label", _fmt_val(pred_label))

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
                # Keep compatibility for components still reading final columns from review_df.
                review_df.loc[review_df["CurveKey"] == curve_key, "is_valid_final"] = review_df.loc[review_df["CurveKey"] == curve_key, "is_valid_true"]
                review_df.loc[review_df["CurveKey"] == curve_key, "Reviewed"] = reviewed_bool
                st.session_state["review_df"] = review_df
                updated_audit = _build_classifier_output(
                    wide_df=wide_for_artifacts,
                    out_df=out_df,
                    review_df=review_df,
                    manual_review_mode=True,
                    meta_df=meta_df if isinstance(meta_df, pd.DataFrame) else out_df,
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

        # keep the rest of your existing metrics rows as-is:
        pred_conf_text = "" if pred_label.strip().lower() == "unsure" else _fmt_metric(pred_conf_display)
        _render_row(f"Confidence ({pred_label})", pred_conf_text)
        _render_row("Too Sparse", _fmt_val(bool(out_row.get("too_sparse")) if "too_sparse" in out_row else "NA"))
        _render_row("Low Resolution", _fmt_val(bool(out_row.get("low_resolution")) if "low_resolution" in out_row else "NA"))
        _render_row(
            "Blank subtraction mode",
            _fmt_val("RAW (applied)" if settings.input_is_raw else "ALREADY BLANK SUBTRACTED (so not applied)"),
        )

        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("How this plot is built?", expanded=False):
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
            "Download zip file",
            data=zip_bytes,
            file_name=zip_name,
            mime="application/zip",
            disabled=not zip_ready,
            use_container_width=True,
            help="Enabled only after the pipeline (Grofit) completes.",
        )
        st.download_button(
            "Download Classifier Auditing.csv",
            data=audit_df.to_csv(index=False).encode("utf-8"),
            file_name="Classifier Auditing.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    if not dr_available:
        with st.expander("Dose Response", expanded=False):
            if not isinstance(gc_fit, pd.DataFrame) or gc_fit.empty:
                st.info("Dose-response results are not available yet. Run the pipeline first.")
            else:
                dr_c1, dr_c2, dr_c3 = st.columns([1.2, 1.2, 1.2])
                response_metric = dr_c1.selectbox(
                    "Metric",
                    options=["mu", "A", "lambda", "integral"],
                    index=0,
                    key="dr_metric_select",
                )

                if manual_review_mode:
                    label_source = dr_c2.selectbox(
                        "Label source",
                        options=["final", "pred"],
                        index=0,
                        key="dr_label_source_select",
                    )
                else:
                    label_source = "pred"
                    dr_c2.markdown("**Label source:** predicted")

                show_bootstrap = dr_c3.checkbox(
                    "Show bootstrap CI",
                    value=bool(isinstance(dr_boot, pd.DataFrame) and not dr_boot.empty),
                    key="dr_show_bootstrap",
                    disabled=not bool(isinstance(dr_boot, pd.DataFrame) and not dr_boot.empty),
                )

                filt_c1, filt_c2 = st.columns(2)
                include_unsure = filt_c1.checkbox("Include unsure", value=False, key="dr_inc_unsure")
                include_invalid = filt_c2.checkbox("Include invalid", value=False, key="dr_inc_invalid")

                dr_payload = build_dr_payload(
                    gc_fit=gc_fit,
                    labels_df=labels_df,
                    dr_boot=dr_boot,
                    test_id=file_stem,
                    response_metric=response_metric,
                    label_source=label_source,
                    include_unsure=include_unsure,
                    include_invalid=include_invalid,
                    dr_s=grofit_opts.dr_s,
                    dr_x_transform=grofit_opts.dr_x_transform,
                    show_bootstrap=show_bootstrap,
                )
                dr_fig = make_dr_plot(dr_payload, show_bootstrap=show_bootstrap)
                st.plotly_chart(dr_fig, use_container_width=True)

                ec50 = _to_numeric_scalar(dr_payload.get("fit", {}).get("ec50"))
                y_mid = _to_numeric_scalar(dr_payload.get("fit", {}).get("y_mid"))
                boot = dr_payload.get("bootstrap", {})
                ec50_ci = boot.get("ec50_ci") if boot.get("ran") else None
                ec50_lo = _to_numeric_scalar(ec50_ci[0]) if ec50_ci and len(ec50_ci) == 2 else np.nan
                ec50_hi = _to_numeric_scalar(ec50_ci[1]) if ec50_ci and len(ec50_ci) == 2 else np.nan

                info_c1, info_c2, info_c3 = st.columns(3)
                info_c1.metric("EC50", f"{ec50:.4g}" if np.isfinite(ec50) else "NA")
                info_c2.metric(
                    "EC50 CI",
                    f"{ec50_lo:.4g}-{ec50_hi:.4g}" if np.isfinite(ec50_lo) and np.isfinite(ec50_hi) else "NA",
                )
                info_c3.metric("Points used", int(dr_payload.get("n_points", 0)))
                if dr_payload.get("excluded", 0) > 0:
                    st.caption(f"Excluded due to labels: {dr_payload.get('excluded')}")

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

        auto_bootstrap_scope = "Only Valid Curves"
        auto_preferred_model = "Best Model"
        auto_response_metric = "mu"
        auto_dr_bootstrap = "True"
        if auto_mode:
            st.markdown('<div class="ui-row-title">Input File Upload</div>', unsafe_allow_html=True)
            auto_c1, auto_c2, auto_c3, auto_c4 = st.columns(4)
            with auto_c1:
                auto_bootstrap_scope = st.selectbox(
                    "GC_Fit Bootstrap",
                    options=["None", "Only Valid Curves", "All Curves"],
                    index=0,
                    key="auto_bootstrap_scope",
                )
            with auto_c2:
                auto_preferred_model = st.selectbox(
                    "Preferred Model",
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

        with st.expander("Filter & Pipeline Options", expanded=False):
            response_var = auto_response_metric if auto_mode else "mu"
            preferred_fit_map = {
                "Best Model": "Both (param + spline)",
                "Spline": "Spline only",
                "Parametric": "Parametric only",
            }
            fit_mode_options = ["Both (param + spline)", "Parametric only", "Spline only"]
            fit_mode_default = preferred_fit_map.get(auto_preferred_model, "Both (param + spline)")
            fit_mode_index = fit_mode_options.index(fit_mode_default) if auto_mode else 0

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

            fit_opt_label = st.selectbox(
                "Fit mode",
                options=fit_mode_options,
                index=fit_mode_index,
                disabled=auto_mode,
            )
            have_atleast = st.number_input("Min points for dose-response", min_value=1, value=6, step=1)
            gc_boot_B = st.number_input("GC bootstrap samples", min_value=0, value=200, step=50)
            dr_boot_B = st.number_input("DR bootstrap samples", min_value=0, value=300, step=50)
            bootstrap_method = st.selectbox("GC bootstrap method", options=["pairs", "residual"], index=0)
            spline_s_str = st.text_input("GC spline smoothing (blank = auto CV)", value="")
            dr_s_str = st.text_input("DR spline smoothing (blank = auto CV)", value="")
            dr_x_transform = st.selectbox("DR x transform", options=["log1p", "none"], index=0)
            if auto_mode:
                st.caption("Fit mode is set by Auto Mode > Preferred Model.")

            input_is_raw = st.checkbox("Input is raw (apply blank subtraction)", value=False)
            global_blank_str = ""
            if input_is_raw:
                global_blank_str = st.text_input("Global blank value (optional)", value="")

            fit_opt_map = {
                "Both (param + spline)": "b",
                "Parametric only": "m",
                "Spline only": "s",
            }
            dr_boot_B_effective = 0 if (auto_mode and auto_dr_bootstrap == "False") else int(dr_boot_B)
            grofit_opts = GrofitOptions(
                response_var=response_var,
                have_atleast=int(have_atleast),
                fit_opt=fit_opt_map.get(fit_opt_label, "b"),
                gc_boot_B=int(gc_boot_B),
                dr_boot_B=dr_boot_B_effective,
                spline_auto_cv=True,
                spline_s=_safe_float(spline_s_str, None),
                dr_s=_safe_float(dr_s_str, None),
                dr_x_transform=None if dr_x_transform == "none" else dr_x_transform,
                bootstrap_method=bootstrap_method,
            )
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
            with st.spinner("Training classifier from meta.csv..."):
                train_classifier_from_meta_file(
                    meta_csv_path=str(TRAIN_META),
                    models_out_dir=str(MODEL_DIR),
                    selected_features=NOTEBOOK_STAGE1_CUSTOM_FEATURES,
                )
            model_files = sorted([p.name for p in MODEL_DIR.glob("*.joblib")])
            st.success(
                "Training complete. Models refreshed in classifier_output/saved_models_selected. "
                f"Saved {len(model_files)} model file(s): {', '.join(model_files)}"
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
                                dr_x_transform=grofit_opts.dr_x_transform,
                                dr_s=grofit_opts.dr_s,
                                fit_opt=grofit_opts.fit_opt,
                                bootstrap_method=grofit_opts.bootstrap_method,
                                validity_col="__all__",
                                random_state=42,
                                export_dir=Path(grofit_td),
                            )
                            gc_fit = grofit_res.get("gc_fit", pd.DataFrame())
                            dr_fit = grofit_res.get("dr_fit", pd.DataFrame())
                            gc_boot = grofit_res.get("gc_boot", pd.DataFrame())
                            dr_boot = grofit_res.get("dr_boot", pd.DataFrame())
                else:
                    st.info("Manual Mode: no curves found for fitting.")
                st.info("You can still review labels and run the manual 'RUN GROFIT' action for full export.")
            else:
                st.info("Auto Mode: running Grofit pipeline with bootstrap and dose-response settings.")
                bootstrap_curve_scope = auto_bootstrap_scope if auto_mode else "Only Valid Curves"
                label_map = audit_df.set_index("Test Id")["True Label"].astype(str).str.strip().str.lower().to_dict()
                include_default = grofit_tidy_all["curve_id"].astype(str).map(
                    lambda cid: label_map.get(cid, "") in {"valid", "unsure"}
                )
                include_valid_only = grofit_tidy_all["curve_id"].astype(str).map(
                    lambda cid: label_map.get(cid, "") in {"valid"}
                )
                grofit_tidy_all["include_auto_default"] = include_default.fillna(False).astype(bool)
                grofit_tidy_all["include_auto_valid_only"] = include_valid_only.fillna(False).astype(bool)
                validity_col = "include_auto_valid_only" if bootstrap_curve_scope == "Only Valid Curves" else "include_auto_default"
                run_grofit = (not grofit_tidy_all.empty) and grofit_tidy_all[validity_col].any()

                if run_grofit:
                    st.info(
                        f"Grofit settings: GC_Fit Bootstrap={bootstrap_curve_scope}, "
                        f"Preferred Model={auto_preferred_model}, Response Metric={auto_response_metric}, "
                        f"DR Bootstrap={auto_dr_bootstrap}"
                    )
                    with st.spinner("Running Grofit (parametric + spline fits, bootstrap, dose-response)..."):
                        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as grofit_td:
                            grofit_res = run_grofit_pipeline(
                                curves_df=grofit_tidy_all,
                                response_var=grofit_opts.response_var,
                                have_atleast=grofit_opts.have_atleast,
                                gc_boot_B=grofit_opts.gc_boot_B,
                                dr_boot_B=grofit_opts.dr_boot_B,
                                spline_auto_cv=grofit_opts.spline_auto_cv,
                                spline_s=grofit_opts.spline_s,
                                dr_x_transform=grofit_opts.dr_x_transform,
                                dr_s=grofit_opts.dr_s,
                                fit_opt=grofit_opts.fit_opt,
                                bootstrap_method=grofit_opts.bootstrap_method,
                                validity_col=validity_col,
                                random_state=42,
                                export_dir=Path(grofit_td),
                            )
                            gc_fit = grofit_res.get("gc_fit", pd.DataFrame())
                            dr_fit = grofit_res.get("dr_fit", pd.DataFrame())
                            gc_boot = grofit_res.get("gc_boot", pd.DataFrame())
                            dr_boot = grofit_res.get("dr_boot", pd.DataFrame())
                            zip_name, zip_bytes = _build_export_zip(
                                wide_df=wide_df,
                                out_df=out_df,
                                review_df=review_df if manual_review_mode else None,
                                gc_fit=gc_fit,
                                gc_boot=gc_boot,
                                dr_fit=dr_fit,
                                dr_boot=dr_boot,
                                audit_df=audit_df,
                                grofit_df=grofit_df,
                                grofit_opts=grofit_opts,
                                settings=settings,
                                mode_label="AUTO",
                                file_stem=file_tag,
                                predicting_model=predicting_model,
                                auto_bootstrap_scope=auto_bootstrap_scope,
                                auto_preferred_model=auto_preferred_model,
                                auto_response_metric=auto_response_metric,
                                auto_dr_bootstrap=auto_dr_bootstrap,
                                stage2_config=infer_res.get("stage2_config"),
                            )
                else:
                    if use_all_curves:
                        st.info("No curves found to run Grofit. Skipping Grofit.")
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
                "audit_df": audit_df,
                "grofit_df": grofit_df,
                "zip_bytes": zip_bytes,
                "zip_name": zip_name if "zip_name" in locals() else "",
                "review_df": review_df,
                "grofit_opts": grofit_opts,
                "manual_review_mode": manual_review_mode,
                "grofit_ran": grofit_ran,
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
