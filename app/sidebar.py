# app/ui/sidebar.py
"""
Top-of-page controls: file upload, mode radio, run/train buttons,
auto-mode selectors, filter expander, and sample download buttons.

Returns a plain dict of every value needed by the main pipeline.
"""
from __future__ import annotations
from pathlib import Path

import streamlit as st

from config import GrofitOptions, InferenceSettings, MODEL_DIR, TRAIN_META
from utils import make_sample_wide_csv_bytes, make_sample_long_csv_bytes, safe_float
from model_io import has_trained_models, discover_models, label_from_stem


def render_top_controls() -> dict:
    """
    Render all top-of-page controls and return a dict with keys:

        uploaded, run, train_refresh, auto_mode, manual_review_mode,
        auto_bootstrap_scope, auto_preferred_model, auto_response_metric,
        auto_dr_bootstrap, model_name, model_label_map, settings, grofit_opts,
        export_label_filter, export_dr_include_unsure, export_dr_include_invalid
    """
    top_left, top_right = st.columns([3.2, 1.6], gap="large")

    # ------------------------------------------------------------------ left
    with top_left:
        uploaded = st.file_uploader(
            "Upload file (Excel .xlsx or CSV .csv) in your defined format. CSV Preferred",
            type=["xlsx", "csv"], accept_multiple_files=False,
        )

        models_ready = has_trained_models(MODEL_DIR)
        run_disabled = (uploaded is None) or (not models_ready)

        mode_col, run_col, train_col = st.columns([2.2, 0.8, 1.3], gap="small")
        with mode_col:
            st.markdown('<div class="ui-row-title">Mode</div>', unsafe_allow_html=True)
            mode = st.radio("Run mode", options=["Auto Mode", "Manual Mode"],
                            horizontal=True, label_visibility="collapsed")
        with run_col:
            st.markdown('<div class="ui-row-title">Run</div>', unsafe_allow_html=True)
            run = st.button("Run pipeline", type="primary",
                            disabled=run_disabled, use_container_width=True)
        with train_col:
            st.markdown('<div class="ui-row-title">Classifier</div>', unsafe_allow_html=True)
            train_refresh = st.button("Train / Refresh Classifier", use_container_width=True)

        if not models_ready:
            st.info("No trained classifier found. Click 'Train / Refresh Classifier' to train models.")

        auto_mode          = (mode == "Auto Mode")
        manual_review_mode = (mode == "Manual Mode")

        # Auto-mode quick selectors
        auto_bootstrap_scope = "False"
        auto_preferred_model = "Best Model"
        auto_response_metric = "mu"
        auto_dr_bootstrap    = "True"
        if auto_mode:
            st.markdown('<div class="ui-row-title">Pipeline Options</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                auto_bootstrap_scope = st.selectbox(
                    "GC Bootstrap", ["False", "True"], index=0, key="auto_bootstrap_scope")
            with c2:
                auto_preferred_model = st.selectbox(
                    "Preferred Fit", ["Best Model", "Spline", "Parametric"], index=0, key="auto_preferred_model")
            with c3:
                auto_response_metric = st.selectbox(
                    "Response Metric", ["mu", "A", "lag", "integral"], index=0, key="auto_response_metric")
            with c4:
                auto_dr_bootstrap = st.selectbox(
                    "DR Bootstrap", ["True", "False"], index=0, key="auto_dr_bootstrap")

        # ---- Filter / options expander ----
        with st.expander("Filter Options for Pipeline & Export", expanded=False):
            response_var = auto_response_metric if auto_mode else "mu"

            # Model selector
            available_models = discover_models(MODEL_DIR)
            model_label_map: dict[str, Path] = {}
            for stem, p in available_models.items():
                lbl = label_from_stem(stem)
                if lbl in model_label_map:
                    lbl = f"{lbl}-{stem}"
                model_label_map[lbl] = p

            model_name: str | None = None
            if model_label_map:
                opts = ["Average"] + sorted(model_label_map.keys())
                model_name = st.selectbox("Choose trained model or ensemble", options=opts, index=0)
                if model_name == "Average":
                    st.caption(f"Ensemble of {len(model_label_map)} models: "
                               f"{', '.join(sorted(model_label_map.keys()))}")
                else:
                    st.caption("Loaded from: `/classifier_output/saved_models_selected`")
            else:
                st.warning("No trained model found. Click Train/Refresh Classifier.")

            # Blank handling
            input_is_raw = st.checkbox("Input is raw (apply blank subtraction)", value=False)
            global_blank_str = ""
            if input_is_raw:
                global_blank_str = st.text_input(
                    "Global blank value (optional)", value="",
                    help="Leave blank to calculate dynamically from the first few points of each curve.")

            # Grofit parameters
            st.markdown("**Grofit Parameters**")
            have_atleast = st.number_input("Min. points required for Dose-Response",
                                           min_value=1, value=6, step=1)
            b1, b2 = st.columns(2)
            with b1:
                gc_boot_B = st.number_input("GC Bootstrap Iterations", min_value=0,
                                            value=200, step=50,
                                            help="Resamples for single-curve CIs.")
            with b2:
                dr_boot_B = st.number_input("DR Bootstrap Iterations", min_value=0,
                                            value=300, step=50,
                                            help="Resamples for dose-response CIs.")

            t1, t2 = st.columns(2)
            with t1:
                dr_x_transform = st.selectbox(
                    "Dose-Response X-Axis Transform", ["OFF", "log10", "log1p"], index=0,
                    help="Transform applied to concentration axis for DR fit/plots.")
            with t2:
                dr_y_transform = st.selectbox(
                    "Dose-Response Y-Axis Transform", ["OFF", "log10", "log1p"], index=0,
                    help="Transform applied to response metric axis for DR fit/plots.")

            export_label_filter = st.selectbox(
                "Export Curve Labels", ["Valid", "Invalid", "Unsure", "All"], index=0,
                help="Filter which curve labels go into Results.zip.")
            _exp = str(export_label_filter).strip().lower()
            dr_unsure_ok  = _exp in {"all", "unsure"}
            dr_invalid_ok = _exp in {"all", "invalid"}
            if not dr_unsure_ok:  st.session_state["export_dr_include_unsure"]  = False
            if not dr_invalid_ok: st.session_state["export_dr_include_invalid"] = False

            e1, e2 = st.columns(2)
            with e1:
                export_dr_include_unsure = st.checkbox(
                    "DR Export: Include Unsure",
                    value=bool(st.session_state.get("export_dr_include_unsure", False)),
                    key="export_dr_include_unsure", disabled=not dr_unsure_ok)
            with e2:
                export_dr_include_invalid = st.checkbox(
                    "DR Export: Include Invalid",
                    value=bool(st.session_state.get("export_dr_include_invalid", False)),
                    key="export_dr_include_invalid", disabled=not dr_invalid_ok)

            # Advanced smoothing override
            st.markdown("**Advanced**")
            override_smoothing = st.checkbox(
                "Override smoothing (smooth.gc / smooth.dr)", value=False,
                help="Default is AUTO (recommended).")
            s1, s2 = st.columns(2)
            with s1:
                smooth_gc_val = st.number_input("smooth.gc", min_value=0.0, value=0.0,
                                                step=0.1, disabled=not override_smoothing)
            with s2:
                smooth_dr_val = st.number_input("smooth.dr", min_value=0.0, value=0.0,
                                                step=0.1, disabled=not override_smoothing)

            fit_map = {"Best Model": "b", "Spline": "s", "Parametric": "m"}
            dr_boot_B_eff = 0 if (auto_mode and auto_dr_bootstrap == "False") else int(dr_boot_B)
            _sm_gc = None if not override_smoothing else (float(smooth_gc_val) or None)
            _sm_dr = None if not override_smoothing else (float(smooth_dr_val) or None)

            grofit_opts = GrofitOptions(
                response_var   = response_var,
                have_atleast   = int(have_atleast),
                fit_opt        = fit_map.get(auto_preferred_model if auto_mode else "Best Model", "b"),
                gc_boot_B      = int(gc_boot_B),
                dr_boot_B      = dr_boot_B_eff,
                spline_auto_cv = not override_smoothing,
                spline_s       = _sm_gc,
                smooth_gc      = _sm_gc,
                dr_s           = _sm_dr,
                smooth_dr      = _sm_dr,
                dr_x_transform = None if dr_x_transform == "OFF" else dr_x_transform,
                dr_y_transform = None if dr_y_transform == "OFF" else dr_y_transform,
            )
            from utils import normalize_bootstrap_method
            grofit_opts.bootstrap_method = normalize_bootstrap_method(grofit_opts.bootstrap_method)

        st.session_state["grofit_opts"] = grofit_opts
        settings = InferenceSettings(
            input_is_raw = bool(input_is_raw),
            global_blank = safe_float(global_blank_str, None) if input_is_raw else None,
        )

    # ----------------------------------------------------------------- right
    with top_right:
        st.markdown('<div class="ui-row-title">Sample formats</div>', unsafe_allow_html=True)
        st.caption(
            "Well headers may encode concentration like `A01[Conc=0.1]` or `A01[0.1]`. "
            "Wide-format includes a `concentration` column for dose-response."
        )
        st.download_button("Download sample (Wide CSV)", data=make_sample_wide_csv_bytes(),
                           file_name="sample_wide.csv", mime="text/csv",
                           use_container_width=True)
        st.download_button("Download sample (Long CSV)", data=make_sample_long_csv_bytes(),
                           file_name="sample_long.csv", mime="text/csv",
                           use_container_width=True)

    return dict(
        uploaded=uploaded, run=run, train_refresh=train_refresh,
        auto_mode=auto_mode, manual_review_mode=manual_review_mode,
        auto_bootstrap_scope=auto_bootstrap_scope,
        auto_preferred_model=auto_preferred_model,
        auto_response_metric=auto_response_metric,
        auto_dr_bootstrap=auto_dr_bootstrap,
        model_name=model_name,
        model_label_map=model_label_map if 'model_label_map' in dir() else {},
        settings=settings,
        grofit_opts=grofit_opts,
        export_label_filter=export_label_filter if 'export_label_filter' in dir() else "Valid",
        export_dr_include_unsure=export_dr_include_unsure if 'export_dr_include_unsure' in dir() else False,
        export_dr_include_invalid=export_dr_include_invalid if 'export_dr_include_invalid' in dir() else False,
    )
