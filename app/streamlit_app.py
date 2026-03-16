# apps/streamlit_app.py
"""
GrowthQC Streamlit entry point.
All logic lives in app/; this file only wires things together.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from utils import (
    extract_conc_from_curve_id,
    normalize_label,
    label_is_valid,
    normalize_bootstrap_method,
)
from model_io import (
    has_trained_models,
    train_classifier_from_meta_file,
)
from data import (
    build_classifier_output,
    build_grofit_input_df,
    build_export_zip,
    init_review_df,
    wide_to_grofit_tidy,
)
from styles import inject_css
from components import show_friendly_error
from sidebar import render_top_controls
from results import render_results

# growthqa pipeline imports
from growthqa.grofit.pipeline import run_grofit_pipeline
from growthqa.classifier.train_from_meta import NOTEBOOK_STAGE1_CUSTOM_FEATURES
from growthqa.preprocess.timegrid import parse_time_from_header
from growthqa.io.wide_format import long_to_wide_preserve_times, parse_any_file_to_long
from growthqa.pipelines.infer_labels import run_label_inference_from_uploaded_wide

# ==========================================================================
# Page config + CSS
# ==========================================================================
st.set_page_config(page_title="GrowthQC - Curve Validator", layout="wide")
inject_css()

st.markdown("## GrowthQC - Bacterial Growth Curve Validator")
st.caption("Upload → preprocess → predict → review → grofit → download")

# ==========================================================================
# Training status banner (persisted via session state)
# ==========================================================================
train_status = st.session_state.pop("train_status", None)
if train_status:
    kind, msg = train_status
    (st.success if kind == "success" else st.error)(msg)

# ==========================================================================
# Top controls
# ==========================================================================
ctrl = render_top_controls()
uploaded             = ctrl["uploaded"]
run                  = ctrl["run"]
train_refresh        = ctrl["train_refresh"]
auto_mode            = ctrl["auto_mode"]
manual_review_mode   = ctrl["manual_review_mode"]
auto_bootstrap_scope = ctrl["auto_bootstrap_scope"]
auto_preferred_model = ctrl["auto_preferred_model"]
auto_response_metric = ctrl["auto_response_metric"]
auto_dr_bootstrap    = ctrl["auto_dr_bootstrap"]
model_name           = ctrl["model_name"]
settings             = ctrl["settings"]
grofit_opts          = ctrl["grofit_opts"]
export_label_filter      = ctrl["export_label_filter"]
export_dr_include_unsure  = ctrl["export_dr_include_unsure"]
export_dr_include_invalid = ctrl["export_dr_include_invalid"]

# Auto-training debug info
auto_debug   = st.session_state.get("train_auto_debug")
auto_summary = st.session_state.get("train_auto_summary")
if auto_debug or auto_summary:
    with st.expander("Auto-training debug info", expanded=False):
        if auto_debug:   st.code(json.dumps(auto_debug,   indent=2), language="json")
        if auto_summary: st.code(json.dumps(auto_summary, indent=2), language="json")

# ==========================================================================
# Cached results
# ==========================================================================
results = st.session_state.get("last_run_results")
if uploaded is None and results and not run:
    st.session_state.pop("last_run_results", None)
    results = None

# ==========================================================================
# Train / Refresh Classifier
# ==========================================================================
if train_refresh:
    if not config.TRAIN_META.exists():
        st.error(f"Training meta.csv not found at: {config.TRAIN_META}")
    else:
        try:
            meta_rows = int(pd.read_csv(config.TRAIN_META).shape[0])
            st.info(
                f"Training classifier with `{config.TRAIN_META.name}` ({meta_rows} rows).\n"
                f"Path: `{config.TRAIN_META}`"
            )
            with st.spinner("Training classifier from meta.csv..."):
                train_out = train_classifier_from_meta_file(
                    meta_csv_path=str(config.TRAIN_META),
                    models_out_dir=str(config.MODEL_DIR),
                    selected_features=NOTEBOOK_STAGE1_CUSTOM_FEATURES,
                )
            model_files  = sorted(p.name for p in config.MODEL_DIR.glob("*.joblib"))
            split_sizes  = train_out.get("split_sizes", {}) if isinstance(train_out, dict) else {}
            split_text   = (
                f" | split sizes: train={split_sizes.get('train','NA')}, "
                f"val={split_sizes.get('val','NA')}, test={split_sizes.get('test','NA')}"
                if split_sizes else ""
            )
            st.success(
                "Training complete. Models refreshed in classifier_output/saved_models_selected. "
                f"Meta rows used: {meta_rows}. "
                f"Saved {len(model_files)} model file(s): {', '.join(model_files)}"
                f"{split_text}"
            )
            st.rerun()
        except Exception as e:
            if isinstance(e, ValueError) and "Selected training features are missing from meta.csv" in str(e):
                try:
                    loaded_cols = pd.read_csv(config.TRAIN_META, nrows=0).columns.tolist()
                    st.caption(f"Loaded training meta.csv: `{config.TRAIN_META}`")
                    st.caption(
                        "Columns detected in loaded file: "
                        + ", ".join(map(str, loaded_cols))
                    )
                except Exception:
                    pass
            show_friendly_error(e)

# ==========================================================================
# Main pipeline run
# ==========================================================================
if run:
    if uploaded is None:
        st.warning("Please upload a file before running validation.")
    else:
        try:
            suffix       = Path(uploaded.name).suffix.lower()
            upload_bytes = uploaded.getvalue()

            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
                tmp_in = Path(td) / f"upload{suffix}"
                tmp_in.write_bytes(upload_bytes)
                with st.spinner("Converting uploaded file to canonical wide format..."):
                    try:
                        long_df = parse_any_file_to_long(str(tmp_in))
                    except Exception as e:
                        st.warning(
                            "Could not read the uploaded file. Please ensure it matches the "
                            "sample formats (wide: Test Id + T#.0 (h) columns; "
                            "long: Time (h) with wells as columns)."
                        )
                        show_friendly_error(e)
                        st.stop()

                    file_tag = Path(uploaded.name).stem
                    wide_df  = long_to_wide_preserve_times(long_df, file_tag=file_tag, add_prefix=True)

                    if "Concentration" not in wide_df.columns:
                        conc = pd.to_numeric(
                            wide_df["Test Id"].astype(str).map(extract_conc_from_curve_id),
                            errors="coerce",
                        )
                        if conc.notna().any():
                            wide_df["Concentration"] = conc.fillna(0.0)

                    time_cols_original = [
                        c for c in wide_df.columns if parse_time_from_header(str(c)) is not None
                    ]

            if not has_trained_models(config.MODEL_DIR):
                st.error("No trained model found. Click Train/Refresh Classifier.")
                st.stop()

            with st.spinner("Running Stage-1 model inference + Stage-2 late-growth logic..."):
                infer_res = run_label_inference_from_uploaded_wide(
                    wide_df=wide_df, settings=settings,
                    model_dir=str(config.MODEL_DIR),
                    model_name=model_name or "Average",
                    stage2_start=16.0,
                    unsure_conf_threshold=None,
                )

            predicting_model = model_name or "Average"
            raw_merged   = infer_res["raw_merged_df"]
            final_merged = infer_res["final_merged_df"]
            out_df       = infer_res["out_df"].copy()
            time_cols_final = [
                c for c in final_merged.columns
                if isinstance(c, str) and c.strip().startswith("T") and "(h)" in c
            ]
            review_df = init_review_df(out_df, wide_df)
            st.session_state["review_df"] = review_df.copy()

            audit_df = build_classifier_output(
                wide_df=wide_df, out_df=out_df,
                review_df=review_df if manual_review_mode else None,
                manual_review_mode=manual_review_mode,
                meta_df=infer_res.get("meta_df", out_df),
                processed_wide_df=final_merged,
            )
            grofit_df = build_grofit_input_df(
                wide_df=wide_df, out_df=out_df,
                review_df=review_df if manual_review_mode else None,
                manual_review_mode=manual_review_mode,
                meta_df=infer_res.get("meta_df", out_df),
                audit_df=audit_df,
            )
            grofit_tidy_all = wide_to_grofit_tidy(grofit_df, file_tag=file_tag)

            true_map = (grofit_df.set_index("Test Id")["True Label"].to_dict()
                        if "True Label" in grofit_df.columns else {})
            pred_map = (out_df.set_index("Test Id")["Pred Label"].to_dict()
                        if "Pred Label" in out_df.columns else {})
            conf_map = (out_df.set_index("Test Id")["Pred Confidence"].to_dict()
                        if "Pred Confidence" in out_df.columns else {})
            grofit_tidy_all["true_label"]      = grofit_tidy_all["curve_id"].map(true_map).apply(normalize_label)
            grofit_tidy_all["is_valid_true"]   = grofit_tidy_all["true_label"].map(label_is_valid).fillna(False).astype(bool)
            grofit_tidy_all["pred_label"]      = grofit_tidy_all["curve_id"].map(pred_map)
            grofit_tidy_all["pred_confidence"] = pd.to_numeric(
                grofit_tidy_all["curve_id"].map(conf_map), errors="coerce")

            gc_fit = dr_fit = gc_boot = dr_boot = pd.DataFrame()
            gc_audit = dr_audit = pd.DataFrame()
            zip_bytes = b""; zip_name = ""; grofit_ran = False

            # ---- MANUAL MODE: initial curve fitting (no bootstrap) ----
            if manual_review_mode:
                if not grofit_tidy_all.empty:
                    st.info("Manual Mode: running curve fitting (spline + model) to populate metrics.")
                    with st.spinner("Running curve fitting for metrics (manual mode)..."):
                        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
                            res = run_grofit_pipeline(
                                curves_df=grofit_tidy_all,
                                response_var=grofit_opts.response_var,
                                have_atleast=grofit_opts.have_atleast,
                                gc_boot_B=0, dr_boot_B=0,
                                spline_auto_cv=grofit_opts.spline_auto_cv,
                                spline_s=grofit_opts.spline_s,
                                smooth_gc=grofit_opts.smooth_gc,
                                smooth_dr=grofit_opts.smooth_dr,
                                dr_x_transform=grofit_opts.dr_x_transform,
                                dr_y_transform=grofit_opts.dr_y_transform,
                                dr_s=grofit_opts.dr_s,
                                fit_opt=grofit_opts.fit_opt,
                                bootstrap_method=normalize_bootstrap_method(grofit_opts.bootstrap_method),
                                validity_col="__all__", random_state=42, export_dir=Path(td),
                            )
                            gc_fit  = res.get("gc_fit",  pd.DataFrame())
                            dr_fit  = res.get("dr_fit",  pd.DataFrame())
                            gc_boot = res.get("gc_boot", pd.DataFrame())
                            dr_boot = res.get("dr_boot", pd.DataFrame())
                            gc_audit = res.get("gc_audit", pd.DataFrame())
                            dr_audit = res.get("dr_audit", pd.DataFrame())
                else:
                    st.info("Manual Mode: no curves found for fitting.")
                st.info("You can still review labels and run the manual 'RUN GROFIT' action for full export.")

            # ---- AUTO MODE: full Grofit + export ----
            else:
                st.info("Auto Mode: running Grofit pipeline with bootstrap and dose-response settings.")
                gc_boot_B_auto = 0 if str(auto_bootstrap_scope) == "False" else grofit_opts.gc_boot_B
                eff_auto = type(grofit_opts)(**grofit_opts.__dict__)
                eff_auto.gc_boot_B        = int(gc_boot_B_auto)
                eff_auto.bootstrap_method = normalize_bootstrap_method(eff_auto.bootstrap_method)

                if not grofit_tidy_all.empty:
                    st.info(
                        f"Grofit settings: GC_Fit Bootstrap={auto_bootstrap_scope}, "
                        f"Preferred Fit={auto_preferred_model}, "
                        f"Response Metric={auto_response_metric}, "
                        f"DR Bootstrap={auto_dr_bootstrap}"
                    )
                    with st.spinner("Running Grofit (parametric + spline fits, bootstrap, dose-response)..."):
                        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
                            res = run_grofit_pipeline(
                                curves_df=grofit_tidy_all,
                                response_var=eff_auto.response_var,
                                have_atleast=eff_auto.have_atleast,
                                gc_boot_B=eff_auto.gc_boot_B,
                                dr_boot_B=eff_auto.dr_boot_B,
                                spline_auto_cv=eff_auto.spline_auto_cv,
                                spline_s=eff_auto.spline_s,
                                smooth_gc=eff_auto.smooth_gc,
                                smooth_dr=eff_auto.smooth_dr,
                                dr_x_transform=eff_auto.dr_x_transform,
                                dr_y_transform=eff_auto.dr_y_transform,
                                dr_s=eff_auto.dr_s,
                                fit_opt=eff_auto.fit_opt,
                                bootstrap_method=eff_auto.bootstrap_method,
                                validity_col="__all__", random_state=42, export_dir=Path(td),
                            )
                            gc_fit   = res.get("gc_fit",  pd.DataFrame())
                            dr_fit   = res.get("dr_fit",  pd.DataFrame())
                            gc_boot  = res.get("gc_boot", pd.DataFrame())
                            dr_boot  = res.get("dr_boot", pd.DataFrame())
                            gc_audit = res.get("gc_audit", pd.DataFrame())
                            dr_audit = res.get("dr_audit", pd.DataFrame())
                            zip_name, zip_bytes = build_export_zip(
                                wide_df=wide_df, out_df=out_df,
                                review_df=None,
                                gc_fit=gc_fit, gc_boot=gc_boot,
                                dr_fit=dr_fit, dr_boot=dr_boot,
                                proc_wide_df=final_merged,
                                audit_df=audit_df, grofit_df=grofit_df,
                                grofit_opts=eff_auto, settings=settings,
                                mode_label="AUTO", file_stem=file_tag,
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
                    st.info("No valid curves found (or all had insufficient points). Skipping Grofit.")
                grofit_ran = True
                if grofit_tidy_all.empty:
                    zip_name = ""

            # ---- persist results ----
            st.session_state["last_run_results"] = {
                "final_merged":      final_merged,
                "raw_merged":        raw_merged,
                "meta_df":           infer_res.get("meta_df"),
                "stage2_config":     infer_res.get("stage2_config"),
                "out_df":            out_df,
                "time_cols_final":   time_cols_final,
                "settings":          settings,
                "file_stem":         Path(uploaded.name).stem,
                "wide_original":     wide_df,
                "time_cols_original": time_cols_original,
                "predicting_model":  predicting_model,
                "grofit_tidy_all":   grofit_tidy_all,
                "gc_fit":  gc_fit,  "dr_fit":  dr_fit,
                "gc_boot": gc_boot, "dr_boot": dr_boot,
                "gc_audit": gc_audit, "dr_audit": dr_audit,
                "audit_df":  audit_df,
                "grofit_df": grofit_df,
                "zip_bytes": zip_bytes,
                "zip_name":  zip_name if "zip_name" in dir() else "",
                "review_df": review_df,
                "grofit_opts": eff_auto if (not manual_review_mode and "eff_auto" in dir()) else grofit_opts,
                "manual_review_mode": manual_review_mode,
                "grofit_ran":         grofit_ran,
                "export_label_filter":       export_label_filter,
                "export_dr_include_unsure":  bool(export_dr_include_unsure),
                "export_dr_include_invalid": bool(export_dr_include_invalid),
            }
            if not out_df.empty:
                st.session_state["selected_test_id"] = str(out_df["Test Id"].iloc[0])
            results = st.session_state["last_run_results"]
            render_results(results)

        except Exception as e:
            show_friendly_error(e)
            st.session_state.pop("last_run_results", None)

# ==========================================================================
# Non-run interactions: redisplay cached results
# ==========================================================================
if not run and results:
    try:
        with st.spinner("Using cached results."):
            render_results(results)
    except Exception as e:
        show_friendly_error(e)
