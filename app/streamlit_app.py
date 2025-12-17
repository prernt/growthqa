# apps/streamlit_app.py
from __future__ import annotations

import io
import tempfile
import sys
import importlib
import shutil
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

from growthqa.preprocess.timegrid import parse_time_from_header
from growthqa.io.wide_format import long_to_wide_preserve_times, parse_any_file_to_long
from growthqa.pipelines.build_meta_dataset import run_merge_preprocess_meta




# -------------------------
# User paths (as requested)
# -------------------------
MODEL_DIR = ROOT / "classifier_output" / "saved_models_selected"


# =========================
# Settings (UI-minimal)
# =========================
@dataclass
class InferenceSettings:
    # Blank handling
    input_is_raw: bool = False               # UI checkbox
    global_blank: float | None = None        # optional UI field when raw

    # Fixed pipeline defaults (not exposed in UI)
    step: float = 0.25
    min_points: int = 3
    low_res_threshold: int = 7
    auto_tmax: bool = True
    auto_tmax_coverage: float = 0.8
    tmax_hours: float | None = None

    # Not exposed anymore; forced OFF
    clip_negatives: bool = False
    smooth_method: str = "SGF"   # Savitzky-Golay smoothing as default
    smooth_window: int = 5
    normalize: str = "MAX"       # scale curves to max=1


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
        yaxis_title="OD",
        height=420,
        margin=dict(l=30, r=10, t=50, b=40),
    )
    return fig


def load_model_pipeline(model_path: str):
    assert_runtime_matches_model(model_path)
    return joblib.load(model_path)


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


def to_labeled_excel_bytes(df: pd.DataFrame, sheet_name: str = "Labeled"):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    return bio.getvalue()


def make_sample_wide_csv_bytes() -> bytes:
    """Provide a minimal wide-format sample matching the requested layout."""
    data = {
        "Test Id": ["A01", "A02"],
        "T0.0 (h)": [0.05, 0.06],
        "T1.0 (h)": [0.08, 0.07],
        "T2.0 (h)": [0.15, 0.10],
        "T3.0 (h)": [0.30, 0.20],
        "T4.0 (h)": [0.45, 0.35],
    }
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode("utf-8")


def make_sample_long_csv_bytes() -> bytes:
    """Provide a minimal sample matching the requested format (Time column + wells as columns)."""
    data = {
        "Time (h)": [0, 0.5, 1, 1.5, 2, 3, 3.5, 4, 4.5, 5],
        "A01": [0.05, 0.06, 0.08, 0.07, 0.15, 0.10, 0.30, 0.20, 0.45, 0.35],
        "A02": [0.05, 0.06, 0.08, 0.07, 0.15, 0.10, 0.30, 0.20, 0.45, 0.35],
        "A03": [0.05, 0.06, 0.08, 0.07, 0.15, 0.10, 0.30, 0.20, 0.45, 0.35],
    }
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode("utf-8")


def render_results(results: dict):
    final_merged = results["final_merged"]
    out_df = results["out_df"]
    settings = results["settings"]
    time_cols_final = results["time_cols_final"]
    file_stem = results["file_stem"]
    wide_original = results.get("wide_original")
    time_cols_original = results.get("time_cols_original", [])
    predicting_model = results.get("predicting_model", "Unknown")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Curves", len(out_df))
    with c2:
        st.metric("Valid", int((out_df["pred_label"] == "Valid").sum()))
    with c3:
        st.metric("Invalid", int((out_df["pred_label"] == "Invalid").sum()))

    st.subheader("Predictions table")
    pretty_cols = ["Test Id", "pred_label", "pred_confidence"]
    pretty = out_df[pretty_cols].rename(
        columns={"pred_label": "Predicted label", "pred_confidence": "Predicted confidence"}
    )
    st.data_editor(
        pretty,
        use_container_width=True,
        hide_index=True,
        disabled=True,
        key="pred_table",
    )
    selection_state = st.session_state.get("pred_table", {})
    selected_rows = selection_state.get("selection", {}).get("rows", []) if isinstance(selection_state, dict) else []
    if selected_rows:
        idx = selected_rows[0]
        st.session_state["selected_test_id"] = str(pretty.iloc[idx]["Test Id"])
    elif "selected_test_id" not in st.session_state and not pretty.empty:
        st.session_state["selected_test_id"] = str(pretty.iloc[0]["Test Id"])

    # Main download: original columns (non-time), predicted label, then original timepoints
    if wide_original is not None:
        non_time_cols = [
            c
            for c in wide_original.columns
            if c not in time_cols_original and c not in {"FileName", "Model Name", "Is_Valid"}
        ]
        ordered_non_time = ["Test Id"] + [c for c in non_time_cols if c != "Test Id"]

        preds_map = out_df.set_index("Test Id")["pred_label"]
        download_df = wide_original.copy()
        download_df.pop("Is_Valid", None)
        download_df["Predicted Label"] = download_df["Test Id"].map(preds_map)
        download_df["Predicting Model"] = predicting_model

        final_cols = ordered_non_time + ["Predicted Label"] + time_cols_original + ["PredictingModel"]
        download_df = download_df[[c for c in final_cols if c in download_df.columns]]

        st.markdown("**Download predictions with original columns**")
        c_csv, c_xlsx = st.columns(2)
        with c_csv:
            st.download_button(
                label="Download CSV",
                data=download_df.to_csv(index=False).encode("utf-8"),
                file_name=f"predictions_{file_stem}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c_xlsx:
            st.download_button(
                label="Download Excel",
                data=to_labeled_excel_bytes(download_df, sheet_name="Predictions"),
                file_name=f"predictions_{file_stem}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    st.subheader("Interactive curve viewer")

    if not time_cols_final:
        st.warning("No time columns found in file for plotting.")
    elif "Test Id" not in final_merged.columns:
        st.warning("File is missing 'Test Id' column; cannot show curve dropdown.")
    else:
        join_cols = ["Test Id", "pred_label", "pred_confidence"]
        if "p_valid" in out_df.columns:
            join_cols.append("p_valid")

        preds_map = out_df[join_cols].set_index("Test Id")
        final_view = final_merged.merge(preds_map, left_on="Test Id", right_index=True, how="left")

        options = sorted(final_view["Test Id"].astype(str).tolist())
        if not options:
            st.warning("No curves available for plotting.")
        else:
            default_selection = st.session_state.get("selected_test_id")
            default_idx = options.index(str(default_selection)) if default_selection in options else 0
            chosen = st.selectbox("Select curve (Test Id)", options, index=default_idx, key="curve_select")
            st.session_state["selected_test_id"] = chosen

            row = final_view.loc[final_view["Test Id"].astype(str) == str(chosen)].iloc[0]

            left, right = st.columns([2, 1])
            with right:
                st.markdown("### Model decision")
                st.write(f"**Model:** {predicting_model}")
                line_with_tooltip(
                    "Label",
                    row.get("pred_label", "NA"),
                    "Predicted validity class for this curve.",
                )
                line_with_tooltip(
                    "Confidence",
                    row.get("pred_confidence", np.nan),
                    "Highest class probability; closer to 1 means higher certainty.",
                )
                if "p_valid" in final_view.columns:
                    line_with_tooltip(
                        "p_valid",
                        row.get("p_valid", np.nan),
                        "Model-reported probability for the 'valid' class when available.",
                    )

                flag_help = {
                    "too_sparse": "True if the time-series had too few points after preprocessing.",
                    "low_resolution": "True if sampling cadence was too coarse to meet the minimum point threshold.",
                    "had_outliers": "True if statistical outlier removal was applied to this curve.",
                }
                for flag, nice in [
                    ("too_sparse", "Too sparse"),
                    ("low_resolution", "Low resolution"),
                    ("had_outliers", "Outliers removed"),
                ]:
                    if flag in final_view.columns:
                        line_with_tooltip(nice, bool(row.get(flag)), flag_help.get(flag, ""))

                st.markdown("---")
                st.write("**Blank subtraction mode:**")
                st.write("RAW (applied)" if settings.input_is_raw else "ALREADY BLANK SUBTRACTED (so not applied)")
                if settings.input_is_raw and settings.global_blank is not None:
                    st.write(f"**Global blank used:** {settings.global_blank}")
                elif settings.input_is_raw:
                    st.write("**Blank used:** default blank estimation")

            with left:
                st.plotly_chart(make_plot(row, time_cols_final, title=f"Test Id: {chosen}"), use_container_width=True)

        # Full-width aligned expanders
        col_info, col_dl = st.columns([2, 1])
        with col_info:
            with st.expander("How this plot is built", expanded=False):
                st.markdown(
                    "- Timepoints are extracted from column headers like `T1.0 (h)` and sorted numerically.\n"
                    "- Values come from the preprocessed table after interpolation, smoothing, and blank handling.\n"
                    "- Plot shows the model-ready curve; it should closely match the cleaned measurement trace used for prediction."
                )
        with col_dl:
            with st.expander("Downloads (debug)", expanded=False):
                # Meta features subset (no raw/final merged)
                base_meta_cols = [
                    "Test Id",
                    "pred_label",
                    "True_A",
                    "True_mu",
                    "True_lam",
                    "initial_OD",
                    "max_slope",
                    "plateau_OD",
                    "auc",
                ]
                meta_debug = out_df[[c for c in base_meta_cols if c in out_df.columns]].rename(columns={"pred_label": "Predicted Label"})
                feature_plotting = final_merged.drop(columns=["Is_Valid"], errors="ignore")
                if not meta_debug.empty:
                    meta_debug["PredictingModel"] = predicting_model
                    c_meta, c_feat = st.columns(2)
                    with c_meta:
                        st.download_button(
                            label="Download meta_features_debug.csv",
                            data=meta_debug.to_csv(index=False).encode("utf-8"),
                            file_name=f"meta_features_debug_{file_stem}.csv",
                            mime="text/csv",
                            help="Meta-feature table used for prediction (engineered features + predicted label).",
                        )
                    with c_feat:
                        st.download_button(
                            label="Download feature_plotting_debug.csv",
                            data=feature_plotting.to_csv(index=False).encode("utf-8"),
                            file_name=f"feature_plotting_{file_stem}.csv",
                            mime="text/csv",
                            help="Cleaned time-series used for plotting/prediction after preprocessing.",
                        )


# =========================
# UI
# =========================
st.set_page_config(page_title="GrowthQC - Curve Validator", layout="wide")
st.title("GrowthQC - Bacterial Growth Curve Validator")
st.caption("Upload -> preprocess -> meta-features -> predict labels -> download + interactive curve viewer")

# Show any training status messages (persist across reruns)
train_status = st.session_state.pop("train_status", None)
if train_status:
    status_kind, status_msg = train_status
    if status_kind == "success":
        st.success(status_msg)
    else:
        st.error(status_msg)

with st.sidebar:
    st.header("Model selection")

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
            model_path = model_label_map[model_name]
            st.caption(f"Loaded from: `/classifier_output/saved_models_selected`")
    else:
        st.warning(
            f"No .joblib models found.\nRun training below",
            icon="⚠️",
        )

    st.markdown("---")
    st.header("Blank subtraction settings")

    input_is_raw = st.checkbox('Input is raw (apply blank subtraction)', value=False)

    global_blank_str = ""
    if input_is_raw:
        global_blank_str = st.text_input("Global blank value (optional)", value="")

    settings = InferenceSettings(
        input_is_raw=bool(input_is_raw),
        global_blank=_safe_float(global_blank_str, None) if input_is_raw else None,
    )

    st.markdown("---")
    with st.expander("Training (advanced)", expanded=False):
        st.caption("Run training here to regenerate models with the current environment. Uses the bundled training data.")
        enable_train = st.checkbox("Enable training (this may take time)", value=False)
        train_clicked = st.button(
            "Run training now",
            disabled=not enable_train,
            use_container_width=True,
            help="Only run if no models are available.",
        )

        if train_clicked:
            meta_path = ROOT / "data" / "train_data" / "meta.csv"
            if not meta_path.exists():
                st.error("Training data not found at the default location.")
            else:
                try:
                    with st.spinner("Training classifiers..."):
                        run_training()
                    st.session_state["train_status"] = ("success", "Training complete. Models refreshed.")
                    st.rerun()  # refresh model dropdown with new artifacts
                except Exception as e:
                    st.session_state["train_status"] = ("error", f"Training failed: {e}")
                    st.rerun()

st.markdown("---")

# If no models are available, stop before main workflow; training expander above can generate them.
if not model_label_map:
    st.info("No models available yet. Run training from the sidebar (advanced).", icon="ℹ️")
    st.stop()

uploaded = st.file_uploader(
    "Upload file (Excel .xlsx or CSV .csv) in your defined format",
    type=["xlsx", "csv"],
    accept_multiple_files=False,
)

with st.expander("Download sample input files", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label="Download sample_wide.csv",
            data=make_sample_wide_csv_bytes(),
            file_name="sample_wide.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            label="Download sample_long.csv",
            data=make_sample_long_csv_bytes(),
            file_name="sample_long.csv",
            mime="text/csv",
            use_container_width=True,
        )

run = st.button("Run validation", type="primary", disabled=(uploaded is None))

# Retrieve any existing run so the UI does not reset on interactions
results = st.session_state.get("last_run_results")

# If the user clears the upload (clicks X), drop cached results
if uploaded is None and results and not run:
    st.session_state.pop("last_run_results", None)
    results = None

if run:
    if uploaded is None:
        st.warning("Please upload a file before running validation.")
    else:
        try:
            if model_name == "Average":
                with st.spinner("Loading model pipelines..."):
                    pipelines = {label: load_model_pipeline(str(path)) for label, path in model_label_map.items()}
            else:
                with st.spinner("Loading model pipeline..."):
                    pipelines = {model_name: load_model_pipeline(str(model_label_map[model_name]))}

            suffix = Path(uploaded.name).suffix.lower()
            # ignore_cleanup_errors avoids Windows temp cleanup issues like NotADirectoryError
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
                tmp_in = Path(td) / f"upload{suffix}"
                tmp_in.write_bytes(uploaded.getvalue())

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
                    time_cols_original = [c for c in wide_df.columns if parse_time_from_header(str(c)) is not None]

                tmp_wide_csv = Path(td) / "wide_input.csv"
                wide_df.to_csv(tmp_wide_csv, index=False)

                out_raw = Path(td) / "raw_merged.csv"
                out_final = Path(td) / "final_merged.csv"
                out_meta = Path(td) / "meta.csv"

                # Backend blank logic as requested:
                # - If input_is_raw checked: treat files as RAW, apply blank subtraction
                # - Else: treat files as ALREADY, do not apply blank subtraction
                blank_default = "RAW" if settings.input_is_raw else "ALREADY"

                # Important: keep blank_status_csv hidden/disabled
                blank_status_csv = None

                # Important: drive blank subtraction solely via (blank_subtracted + blank_default)
                # We set blank_subtracted=True always so the per-file status gate works.
                # (When blank_default="ALREADY", apply_blank becomes False in preprocess_wide.)
                with st.spinner("Running interpolation + preprocessing + meta-features..."):
                    raw_merged, final_merged, meta = run_merge_preprocess_meta(
                        inputs=[str(tmp_wide_csv)],
                        out_raw=str(out_raw),
                        out_final=str(out_final),
                        out_meta=str(out_meta),

                        step=settings.step,
                        min_points=settings.min_points,
                        low_res_threshold=settings.low_res_threshold,
                        tmax_hours=settings.tmax_hours,
                        auto_tmax=settings.auto_tmax,
                        auto_tmax_coverage=settings.auto_tmax_coverage,

                        blank_subtracted=True,
                        clip_negatives=settings.clip_negatives,
                        global_blank=settings.global_blank,
                        blank_status_csv=blank_status_csv,
                        blank_default=blank_default,

                        smooth_method=settings.smooth_method,
                        smooth_window=settings.smooth_window,
                        normalize=settings.normalize,
                        loglevel="ERROR",
                    )

                with st.spinner("Predicting labels..."):
                    if model_name == "Average":
                        predicting_model = "Average"
                        per_model_preds = []
                        for lbl, pipe in pipelines.items():
                            plabel, pconf, pvalid = predict_hard_with_confidence(pipe, meta)
                            per_model_preds.append((lbl, plabel, pconf, pvalid))

                        valid_probs_list = []
                        for _, plabel, _, pvalid in per_model_preds:
                            if np.any(np.isfinite(pvalid)):
                                valid_probs_list.append(pvalid)
                            else:
                                valid_probs_list.append(_labels_to_prob_valid(plabel))
                        valid_probs = np.vstack(valid_probs_list)
                        avg_valid = np.nanmean(valid_probs, axis=0)
                        final_prob = np.where(np.isnan(avg_valid), 0.5, avg_valid)
                        final_label = np.where(final_prob >= 0.5, "Valid", "Invalid")
                        final_conf = np.maximum(final_prob, 1 - final_prob)
                    else:
                        predicting_model = model_name
                        lbl = next(iter(pipelines.values()))
                        final_label, final_conf, p_valid = predict_hard_with_confidence(lbl, meta)
                        final_prob = p_valid if np.any(np.isfinite(p_valid)) else _labels_to_prob_valid(final_label)

                out_df = meta.copy()
                out_df["pred_label"] = final_label
                out_df["pred_confidence"] = np.round(final_conf, 4)

                time_cols_final = [c for c in final_merged.columns if isinstance(c, str) and c.strip().startswith("T") and "(h)" in c]

                st.session_state["last_run_results"] = {
                    "final_merged": final_merged,
                    "raw_merged": raw_merged,
                    "out_df": out_df,
                    "time_cols_final": time_cols_final,
                    "settings": settings,
                    "file_stem": Path(uploaded.name).stem,
                    "wide_original": wide_df,
                    "time_cols_original": time_cols_original,
                    "predicting_model": predicting_model,
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
