# app/ui/results.py
"""
render_results(results) — the full results section:
  summary metrics, single-curve review tab, dose-response tab,
  debug downloads, and the manual Grofit-run trigger.
"""
from __future__ import annotations
import io
import json
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from config import GrofitOptions
from utils import (
    extract_conc_from_curve_id,
    make_curve_key,
    normalize_bootstrap_method,
    normalize_label,
    label_is_valid,
    to_numeric_scalar,
)
from plots import (
    generate_fast_bootstrap_bands,
    make_overlay_plot,
    make_overlay_plot_payload,
    make_dr_plot,
)
from data import (
    build_classifier_output,
    build_grofit_input_df,
    build_export_zip,
    wide_to_grofit_tidy,
)
from components import (
    st_pills_multi,
    render_metric_row,
    render_select_row,
)

import sklearn


# ---------------------------------------------------------------------------
# Helpers local to this module
# ---------------------------------------------------------------------------

def _fmt_val(val: object) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)): return "NA"
    if isinstance(val, bool): return "True" if val else "False"
    return str(val)


def _fmt_metric(val: object) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)): return "NA"
    if isinstance(val, (int, float, np.floating)): return f"{float(val):.4f}"
    return str(val)


def _fmt_ci_metric(val: object, ci: list | None) -> str:
    base = _fmt_metric(val)
    if ci and len(ci) == 2:
        lo = pd.to_numeric(pd.Series([ci[0]]), errors="coerce").iloc[0]
        hi = pd.to_numeric(pd.Series([ci[1]]), errors="coerce").iloc[0]
        if np.isfinite(lo) and np.isfinite(hi):
            return f"{base} ({float(lo):.4f}-{float(hi):.4f})"
    return base


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_results(results: dict) -> None:  # noqa: C901 (long but cohesive)
    from growthqa.viz.payloads import build_curve_payloads, build_dr_payload
    from growthqa.grofit.pipeline import run_grofit_pipeline

    # ---- unpack results ----
    final_merged       = results["final_merged"]
    out_df             = results["out_df"]
    settings           = results["settings"]
    time_cols_final    = results["time_cols_final"]
    file_stem          = results["file_stem"]
    wide_original      = results.get("wide_original")
    time_cols_original = results.get("time_cols_original", [])
    predicting_model   = results.get("predicting_model", "Unknown")
    review_df          = results.get("review_df")
    manual_review_mode = results.get("manual_review_mode", False)
    grofit_opts        = results.get("grofit_opts",
                                     st.session_state.get("grofit_opts", GrofitOptions()))
    grofit_opts.bootstrap_method = normalize_bootstrap_method(
        getattr(grofit_opts, "bootstrap_method", None))
    meta_df         = results.get("meta_df", out_df)
    gc_fit          = results.get("gc_fit",          pd.DataFrame())
    dr_fit          = results.get("dr_fit",          pd.DataFrame())
    gc_boot         = results.get("gc_boot",         pd.DataFrame())
    dr_boot         = results.get("dr_boot",         pd.DataFrame())
    gc_audit        = results.get("gc_audit",        pd.DataFrame())
    dr_audit        = results.get("dr_audit",        pd.DataFrame())
    grofit_tidy_all = results.get("grofit_tidy_all", pd.DataFrame())
    zip_bytes       = results.get("zip_bytes", b"")
    grofit_ran      = results.get("grofit_ran", False)
    wide_for_art    = wide_original if isinstance(wide_original, pd.DataFrame) else final_merged

    # sync review_df with session state
    if review_df is not None:
        if st.session_state.get("review_df") is None:
            st.session_state["review_df"] = review_df.copy()
        review_df = st.session_state.get("review_df", review_df)

    # build/cache audit + grofit input dfs
    audit_df = results.get("audit_df")
    if not isinstance(audit_df, pd.DataFrame) or audit_df.empty:
        audit_df = build_classifier_output(
            wide_df=wide_for_art, out_df=out_df,
            review_df=review_df if manual_review_mode else None,
            manual_review_mode=manual_review_mode,
            meta_df=meta_df if isinstance(meta_df, pd.DataFrame) else out_df,
            processed_wide_df=final_merged if isinstance(final_merged, pd.DataFrame) else None,
        )
    grofit_df = results.get("grofit_df")
    if not isinstance(grofit_df, pd.DataFrame) or grofit_df.empty:
        grofit_df = build_grofit_input_df(
            wide_df=wide_for_art, out_df=out_df,
            review_df=review_df if manual_review_mode else None,
            manual_review_mode=manual_review_mode,
            meta_df=meta_df if isinstance(meta_df, pd.DataFrame) else out_df,
            audit_df=audit_df,
        )
    results["audit_df"] = audit_df
    results["grofit_df"] = grofit_df
    st.session_state["last_run_results"] = results

    # ---- concentration map ----
    def _conc_map() -> dict:
        if wide_original is not None and "Concentration" in wide_original.columns:
            return wide_original.set_index("Test Id")["Concentration"].to_dict()
        if "Concentration" in final_merged.columns:
            return final_merged.set_index("Test Id")["Concentration"].to_dict()
        if review_df is not None and "Concentration" in review_df.columns:
            return review_df.set_index("Test Id")["Concentration"].to_dict()
        return {tid: extract_conc_from_curve_id(tid)
                for tid in out_df["Test Id"].astype(str).tolist()}

    conc_map = _conc_map()

    def _label_with_conc(tid: str) -> str:
        conc = conc_map.get(tid, "")
        return f"{tid} + {conc}" if conc and not pd.isna(conc) else tid

    # ---- curve_df (for filter/select) ----
    if manual_review_mode and isinstance(review_df, pd.DataFrame) and not review_df.empty:
        curve_df = review_df.copy()
        merge_cols = [c for c in ["pred_label", "Pred Label", "pred_confidence",
                                   "Pred Confidence", "confidence_valid",
                                   "confidence_invalid", "too_sparse",
                                   "low_resolution", "had_outliers"]
                      if c in out_df.columns]
        if merge_cols:
            curve_df = curve_df.merge(out_df[["Test Id"] + merge_cols],
                                      on="Test Id", how="left")
    else:
        curve_df = out_df.copy()
        if "Concentration" not in curve_df.columns:
            curve_df["Concentration"] = curve_df["Test Id"].map(conc_map)

    pred_col = next(
        (c for c in ["pred_label", "Pred Label", "final_label",
                     "true_label", "Predicted S1 Label"] if c in curve_df.columns),
        None,
    )
    if pred_col is None:
        pred_col = "_pred_label_fallback"
        curve_df[pred_col] = ""
    curve_df["_filter_label"] = curve_df[pred_col].map(normalize_label)

    # ---- summary metrics ----
    st.markdown("---")
    st.markdown(f"### {'MANUAL MODE' if manual_review_mode else 'AUTO MODE'}")
    total_curves  = int(len(out_df))
    final_norm    = (out_df["final_label"].astype(str).str.upper()
                     if "final_label" in out_df.columns else pd.Series([], dtype=str))
    valid_count   = int((final_norm == "VALID").sum())
    invalid_count = int((final_norm == "INVALID").sum())
    unsure_count  = int((final_norm == "UNSURE").sum()) if "final_label" in out_df.columns else 0
    reviewed_count = correct_count = 0
    if manual_review_mode and isinstance(review_df, pd.DataFrame) and not review_df.empty:
        reviewed_count = int(review_df["Reviewed"].sum())
        correct_count  = int((review_df["Reviewed"] &
                               (review_df["is_valid_true"] == review_df["is_valid_pred"])).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("No. of curves", total_curves)
    c2.metric("Valid",   valid_count)
    c3.metric("Invalid", invalid_count)
    c4.metric("Unsure",  unsure_count)

    st.markdown("---")
    st.markdown("#### Review Section")

    total_reviewed_label  = f"{reviewed_count}/{total_curves}"
    correct_reviewed_label = (f"{correct_count}/{reviewed_count}"
                               if reviewed_count > 0 else "0/0")

    # ---- header row: filter + select ----
    hdr_left, hdr_right = st.columns([2.2, 1.0], gap="large")
    with hdr_left:
        filter_col, select_col = st.columns([1, 2])
        with filter_col:
            filter_choice = st.selectbox("Filter",
                                         options=["All", "Valid", "Invalid", "Unsure"],
                                         key="curve_filter")
        filtered = (curve_df[curve_df["_filter_label"] == filter_choice].copy()
                    if filter_choice != "All" else curve_df.copy())
        base_order = (wide_original["Test Id"].astype(str).tolist()
                      if wide_original is not None and "Test Id" in wide_original.columns
                      else curve_df["Test Id"].astype(str).tolist())
        allowed_set = set(filtered["Test Id"].astype(str).tolist())
        options = [tid for tid in base_order if tid in allowed_set]
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
            current = st.session_state.get("curve_select",
                                            st.session_state.get("selected_test_id"))
            idx = options.index(str(current)) if str(current) in options else 0
            chosen = st.selectbox("Select Test Id + Conc", options, index=idx,
                                  key="curve_select", format_func=_label_with_conc)
        st.session_state["selected_test_id"] = chosen

    with hdr_right:
        st.markdown("<div class='metrics-header'><strong>Metrics</strong></div>",
                    unsafe_allow_html=True)
        if manual_review_mode:
            st.markdown(
                f"<div class='metrics-subheader'>Total Reviewed: {total_reviewed_label} | "
                f"Correctly Predicted: {correct_reviewed_label}</div>",
                unsafe_allow_html=True,
            )

    def _nav_set(new_idx: int) -> None:
        v = options[new_idx]
        st.session_state["selected_test_id"] = v
        st.session_state["pending_curve_select"] = v
        st.rerun()

    # ---- row / out_row lookups ----
    row     = curve_df.loc[curve_df["Test Id"].astype(str) == str(chosen)].iloc[0]
    out_row = (out_df.loc[out_df["Test Id"].astype(str) == str(chosen)].iloc[0]
               if not out_df.empty else row)

    pred_label = normalize_label(out_row.get("Pred Label", row.get(pred_col, "")))
    if not pred_label:
        bv = row.get("is_valid_pred")
        if isinstance(bv, (bool, np.bool_)):
            pred_label = "Valid" if bv else "Invalid"
        elif isinstance(bv, (int, float, np.integer, np.floating)) and pd.notna(bv):
            pred_label = "Valid" if int(bv) == 1 else "Invalid"

    pred_conf_display = pd.to_numeric(
        pd.Series([out_row.get("Pred Confidence", out_row.get("pred_confidence", np.nan))]),
        errors="coerce",
    ).iloc[0]

    final_label = (normalize_label(row.get("true_label", row.get("final_label", pred_label)))
                   if manual_review_mode else pred_label)
    if not final_label:
        final_label = "Valid" if bool(row.get("is_valid_true", label_is_valid(pred_label))) else "Invalid"

    labels_df = (review_df if manual_review_mode and isinstance(review_df, pd.DataFrame)
                 else out_df.copy())
    if isinstance(labels_df, pd.DataFrame):
        labels_df = labels_df.copy()
        if manual_review_mode and "true_label" in labels_df.columns:
            labels_df["final_label"] = labels_df["true_label"]
        if "final_label" not in labels_df.columns and "pred_label" in labels_df.columns:
            labels_df["final_label"] = labels_df["pred_label"]
        if "Reviewed" not in labels_df.columns:
            labels_df["Reviewed"] = False

    zip_ready = bool(grofit_ran and zip_bytes)

    # ---- resolve best gc_fit / gc_boot rows for chosen curve ----
    active_test_id = str(file_stem)
    fit_row = boot_row = None
    if isinstance(gc_fit, pd.DataFrame) and not gc_fit.empty and "add.id" in gc_fit.columns:
        fm = gc_fit[gc_fit["add.id"].astype(str) == str(chosen)]
        if not fm.empty:
            if "test.id" in fm.columns:
                for candidate in [str(file_stem),
                                   str(out_row.get("FileName", "")).strip()]:
                    exact = fm[fm["test.id"].astype(str) == candidate]
                    if not exact.empty:
                        fit_row = exact.iloc[0]
                        active_test_id = candidate
                        break
                if fit_row is None:
                    fit_row = fm.iloc[0]
                    active_test_id = str(fit_row.get("test.id", file_stem))
            else:
                fit_row = fm.iloc[0]
    if isinstance(gc_boot, pd.DataFrame) and not gc_boot.empty and "add.id" in gc_boot.columns:
        bm = gc_boot[gc_boot["add.id"].astype(str) == str(chosen)]
        if not bm.empty:
            if "test.id" in bm.columns:
                exact = bm[bm["test.id"].astype(str) == str(active_test_id)]
                boot_row = exact.iloc[0] if not exact.empty else bm.iloc[0]
            else:
                boot_row = bm.iloc[0]

    # ---- build curve payload ----
    curve_payload = None
    if (isinstance(grofit_tidy_all, pd.DataFrame) and not grofit_tidy_all.empty
            and isinstance(labels_df, pd.DataFrame)
            and wide_original is not None and not final_merged.empty):
        payloads = build_curve_payloads(
            curves_df=grofit_tidy_all, raw_wide=wide_original,
            proc_wide=final_merged, labels_df=labels_df,
            gc_boot=gc_boot,
            spline_s=grofit_opts.spline_s, smooth_gc=grofit_opts.smooth_gc,
            spline_auto_cv=grofit_opts.spline_auto_cv,
            include_bootstrap=bool(isinstance(gc_boot, pd.DataFrame) and not gc_boot.empty),
            test_id=active_test_id, curve_ids=[str(chosen)],
        )
        curve_payload = payloads.get(str(chosen))

    # ===================================================================
    tab_curve, tab_dr = st.tabs(["Single Curve Review", "Dose Response Analysis"])

    # -------------------------------------------------------------------
    # TAB 1 – Single curve review
    # -------------------------------------------------------------------
    with tab_curve:
        left, right = st.columns([2.2, 1.0], gap="large")

        with left:
            st.markdown('<div class="curve-title">Curve Viewer</div>', unsafe_allow_html=True)
            bulk_boot  = bool(isinstance(gc_boot, pd.DataFrame) and not gc_boot.empty)
            sess_key   = f"boot_arrays_{chosen}"
            boot_ready = bulk_boot or (sess_key in st.session_state)

            overlay_opts = ["Spline Fit", "Parametric Model"]
            if manual_review_mode:
                overlay_opts.append("Bootstrap CI")
            curve_overlay_store = st.session_state.setdefault("curve_overlay_store", {})
            default_overlay_sel = curve_overlay_store.get(
                str(chosen), ["Spline Fit", "Parametric Model"]
            )
            overlay_sel = st_pills_multi(" ", options=overlay_opts,
                                         default=default_overlay_sel,
                                         key=f"curve_overlay_pills_{chosen}")
            curve_overlay_store[str(chosen)] = list(overlay_sel)

            # inject session-state bootstrap if needed
            if not bulk_boot and (sess_key in st.session_state) and curve_payload:
                y_lo, y_hi = st.session_state[sess_key]
                curve_payload.setdefault("bootstrap", {})
                curve_payload["bootstrap"].update(ran=True, y_hat_q025=y_lo, y_hat_q975=y_hi)

            # on-demand bootstrap in manual mode
            wants_boot = manual_review_mode and "Bootstrap CI" in overlay_sel
            if wants_boot and not boot_ready and curve_payload:
                with st.spinner("Running fast in-memory bootstrap for UI..."):
                    spl = curve_payload.get("spline", {})
                    t_b = curve_payload.get("t_fit")
                    y_b = curve_payload.get("y_fit")
                    tg  = spl.get("t_grid")
                    lam = spl.get("lam")
                    if t_b is not None and y_b is not None and tg is not None and len(t_b) >= 6:
                        y_lo, y_hi = generate_fast_bootstrap_bands(t_b, y_b, lam, t_grid=tg)
                        if y_lo is not None:
                            st.session_state[sess_key] = (y_lo, y_hi)
                    st.rerun()

            show_spline    = "Spline Fit"      in overlay_sel
            show_model     = "Parametric Model" in overlay_sel
            show_bootstrap = manual_review_mode and "Bootstrap CI" in overlay_sel and boot_ready

            if wide_original is not None and not final_merged.empty:
                raw_r  = wide_original.loc[wide_original["Test Id"].astype(str) == str(chosen)]
                proc_r = final_merged.loc[final_merged["Test Id"].astype(str) == str(chosen)]
                if not raw_r.empty and not proc_r.empty:
                    fig = (make_overlay_plot_payload(
                               curve_payload, title=str(chosen),
                               show_spline=show_spline, show_model=show_model,
                               show_bootstrap=show_bootstrap)
                           if curve_payload
                           else make_overlay_plot(
                               raw_r.iloc[0], time_cols_original,
                               proc_r.iloc[0], time_cols_final,
                               title=str(chosen),
                               input_is_raw=settings.input_is_raw,
                               global_blank=settings.global_blank))
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)

                    nav_l, nav_r = st.columns(2)
                    if nav_l.button("Previous Curve", key="prev_curve", use_container_width=True):
                        _nav_set((options.index(str(chosen)) - 1) % len(options))
                    if nav_r.button("Next Curve",     key="next_curve", use_container_width=True):
                        _nav_set((options.index(str(chosen)) + 1) % len(options))
                else:
                    st.warning("Could not find curve data for overlay plot.")
            else:
                st.warning("No time columns found for plotting.")

        with right:
            st.markdown('<div class="metrics-panel">', unsafe_allow_html=True)

            # -- Grofit dialog (manual mode) --
            @st.dialog("Configure & Run Grofit")
            def _grofit_dialog():
                st.write("Review your pipeline settings before executing Grofit.")
                d1, d2 = st.columns(2)
                with d1:
                    _safe_idx = lambda opts, key, default: (
                        opts.index(st.session_state.get(key, default))
                        if st.session_state.get(key, default) in opts else 0
                    )
                    gc_bs = st.selectbox("GC Bootstrap", ["False", "True"],
                                         index=_safe_idx(["False","True"], "exit_gc_bootstrap", "False"))
                    rm    = st.selectbox("Response Metric", ["mu","A","lag","integral"],
                                         index=_safe_idx(["mu","A","lag","integral"], "exit_response_metric", "mu"))
                with d2:
                    pf    = st.selectbox("Preferred Fit", ["Best Model","Spline","Parametric"],
                                         index=_safe_idx(["Best Model","Spline","Parametric"], "exit_preferred_model", "Best Model"))
                    dr_bs = st.selectbox("DR Bootstrap", ["True","False"],
                                         index=_safe_idx(["True","False"], "exit_dr_bootstrap", "True"))
                st.divider()
                if st.button("Confirm & Run Pipeline", type="primary", use_container_width=True):
                    st.session_state.update(
                        exit_gc_bootstrap=gc_bs, exit_response_metric=rm,
                        exit_preferred_model=pf, exit_dr_bootstrap=dr_bs,
                        trigger_grofit_run=True,
                    )
                    st.rerun()

            # -- predicted label badge --
            with st.container(border=True):
                lc = {"valid":"#2e7d32","invalid":"#c62828","unsure":"#ef6c00"}.get(
                    pred_label.strip().lower(), "#5f4337")
                st.markdown(
                    f"<div style='border:1px solid #8f8f8f;border-radius:8px;padding:8px 10px;"
                    f"margin:4px 0 10px 0;'>"
                    f"<div style='font-size:1.0rem;font-weight:500;'>Predicted Label</div>"
                    f"<div style='font-size:1.2rem;font-weight:700;color:{lc};'>{pred_label}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # -- manual review controls --
                if manual_review_mode and isinstance(review_df, pd.DataFrame) and not review_df.empty:
                    curve_key = row.get("CurveKey") or make_curve_key(
                        str(row["Test Id"]), row.get("Concentration"))
                    lbl_key = f"true_label_{curve_key}"
                    rev_key = f"reviewed_{curve_key}"

                    if lbl_key not in st.session_state:
                        tn = normalize_label(row.get("true_label", row.get("final_label", pred_label)))
                        st.session_state[lbl_key] = tn if tn in {"Valid","Invalid","Unsure"} else "Unsure"
                    if rev_key not in st.session_state:
                        st.session_state[rev_key] = "True" if bool(row.get("Reviewed", False)) else "False"
                    elif isinstance(st.session_state[rev_key], bool):
                        st.session_state[rev_key] = "True" if st.session_state[rev_key] else "False"

                    new_final    = render_select_row("True Label", ["Valid","Invalid","Unsure"],
                                                     ["Valid","Invalid","Unsure"].index(st.session_state[lbl_key]),
                                                     lbl_key)
                    reviewed_val = render_select_row("Reviewed", ["False","True"],
                                                     1 if st.session_state[rev_key]=="True" else 0,
                                                     rev_key)
                    reviewed_bool = reviewed_val == "True"

                    old_final    = normalize_label(row.get("true_label", row.get("final_label", pred_label)))
                    old_reviewed = bool(row.get("Reviewed", False))
                    if new_final != old_final or reviewed_bool != old_reviewed:
                        review_df.loc[review_df["CurveKey"] == curve_key, "true_label"]    = new_final
                        review_df.loc[review_df["CurveKey"] == curve_key, "is_valid_true"] = (new_final == "Valid")
                        review_df.loc[review_df["CurveKey"] == curve_key, "is_valid_final"] = (new_final == "Valid")
                        review_df.loc[review_df["CurveKey"] == curve_key, "Reviewed"]      = reviewed_bool
                        st.session_state["review_df"] = review_df
                        upd_audit  = build_classifier_output(
                            wide_df=wide_for_art, out_df=out_df, review_df=review_df,
                            manual_review_mode=True,
                            meta_df=meta_df if isinstance(meta_df, pd.DataFrame) else out_df,
                            processed_wide_df=final_merged if isinstance(final_merged, pd.DataFrame) else None,
                        )
                        upd_grofit = build_grofit_input_df(
                            wide_df=wide_for_art, out_df=out_df, review_df=review_df,
                            manual_review_mode=True,
                            meta_df=meta_df if isinstance(meta_df, pd.DataFrame) else out_df,
                            audit_df=upd_audit,
                        )
                        results["audit_df"]  = upd_audit
                        results["grofit_df"] = upd_grofit
                        results["review_df"] = review_df
                        st.session_state["last_run_results"] = results
                        st.rerun()

                conf_text = "" if pred_label.strip().lower() == "unsure" else _fmt_metric(pred_conf_display)
                render_metric_row(f"Confidence ({pred_label})", conf_text)

            # -- grofit metrics table --
            if curve_payload:
                sp_p   = curve_payload.get("spline",     {}).get("params", {})
                par    = curve_payload.get("parametric", {})
                pa_p   = par.get("params", {})
                bci    = curve_payload.get("bootstrap",  {}).get("ci", {})
                rows   = [
                    {"Metric":"mu",       "Spline":_fmt_ci_metric(sp_p.get("mu"),       bci.get("mu")),
                                          "Model": _fmt_metric(pa_p.get("mu"))},
                    {"Metric":"lambda",   "Spline":_fmt_ci_metric(sp_p.get("lambda"),   bci.get("lambda")),
                                          "Model": _fmt_metric(pa_p.get("lambda"))},
                    {"Metric":"A",        "Spline":_fmt_ci_metric(sp_p.get("A"),        bci.get("A")),
                                          "Model": _fmt_metric(pa_p.get("A"))},
                    {"Metric":"Integral", "Spline":_fmt_ci_metric(sp_p.get("integral"), bci.get("integral")),
                                          "Model": _fmt_metric(pa_p.get("integral"))},
                ]
                params_df = (pd.DataFrame(rows)
                             .rename(columns={"Model": str(par.get("model_name") or "Model")}))
                st.dataframe(params_df, hide_index=True, use_container_width=True)

            if manual_review_mode and st.button("Run Final Pipeline & Export", type="primary",
                                                use_container_width=True, key="run_pipeline_dialog_btn"):
                _grofit_dialog()

            render_metric_row("Too Sparse",
                              _fmt_val(bool(out_row.get("too_sparse")) if "too_sparse" in out_row else "NA"))
            render_metric_row("Low Resolution",
                              _fmt_val(bool(out_row.get("low_resolution")) if "low_resolution" in out_row else "NA"))
            render_metric_row("Blank subtraction mode",
                              _fmt_val("RAW (applied)" if settings.input_is_raw
                                       else "ALREADY BLANK SUBTRACTED (so not applied)"))
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("About the Processed plot ...", expanded=False):
                st.markdown(
                    "- Timepoints extracted from column headers like `T1.0 (h)` and sorted numerically.\n"
                    "- Values from the preprocessed table after interpolation, smoothing, and blank handling.\n"
                    "- Y-axis: Relative OD.\n"
                    "- Plot shows Raw vs Processed curves."
                )

            # -- downloads --
            st.markdown("---")
            st.markdown("#### Debug Downloads")
            zip_name = results.get("zip_name") or f"outputs_{file_stem}.zip"
            st.download_button("Download Results.zip", data=zip_bytes, file_name=zip_name,
                               mime="application/zip", disabled=not zip_ready,
                               use_container_width=True,
                               help="Contains gc_fit, gc_boot, dr_fit, dr_boot, and a plots folder")

            # auditing zip
            audit_run_info = {
                "mode": "MANUAL" if manual_review_mode else "AUTO",
                "timestamp": datetime.now().isoformat(),
                "file_stem": file_stem, "predicting_model": predicting_model,
                "grofit_options": grofit_opts.__dict__ if grofit_opts else None,
                "settings": settings.__dict__ if settings else None,
                "pipeline_filters": {
                    "gc_bootstrap": (st.session_state.get("exit_gc_bootstrap","False")
                                     if manual_review_mode
                                     else st.session_state.get("auto_bootstrap_scope","False")),
                    "preferred_fit": (st.session_state.get("exit_preferred_model","Best Model")
                                      if manual_review_mode
                                      else st.session_state.get("auto_preferred_model","Best Model")),
                    "response_metric": (st.session_state.get("exit_response_metric","mu")
                                        if manual_review_mode
                                        else st.session_state.get("auto_response_metric","mu")),
                    "dr_bootstrap": (st.session_state.get("exit_dr_bootstrap","False")
                                     if manual_review_mode
                                     else st.session_state.get("auto_dr_bootstrap","False")),
                    "export_label_filter": results.get("export_label_filter","Valid"),
                    "export_dr_include_unsure":  bool(results.get("export_dr_include_unsure",False)),
                    "export_dr_include_invalid": bool(results.get("export_dr_include_invalid",False)),
                },
                "stage2_thresholds": results.get("stage2_config"),
                "versions": {"python": sys.version, "numpy": np.__version__,
                             "pandas": pd.__version__, "sklearn": sklearn.__version__},
            }
            az_bio = io.BytesIO()
            with zipfile.ZipFile(az_bio, "w", compression=zipfile.ZIP_DEFLATED) as az:
                az.writestr("Classifier Audit.csv", audit_df.to_csv(index=False))
                if isinstance(gc_audit, pd.DataFrame) and not gc_audit.empty:
                    az.writestr("GC Audit.csv", gc_audit.to_csv(index=False))
                if isinstance(dr_audit, pd.DataFrame) and not dr_audit.empty:
                    az.writestr("DR Audit.csv", dr_audit.to_csv(index=False))
                if isinstance(grofit_df, pd.DataFrame) and not grofit_df.empty:
                    gout = grofit_df.drop(columns=["FileName","Model Name","Is_Valid"], errors="ignore")
                    az.writestr("Grofit.csv", gout.to_csv(index=False))
                az.writestr("run_info.json", json.dumps(audit_run_info, indent=2))
            st.download_button("Download Auditing.zip", data=az_bio.getvalue(),
                               file_name="Auditing.zip", mime="application/zip",
                               use_container_width=True,
                               help="Contains classifier audit, GC/DR audits, and run metadata.")

    # -------------------------------------------------------------------
    # TAB 2 – Dose response
    # -------------------------------------------------------------------
    with tab_dr:
        if not (isinstance(dr_fit, pd.DataFrame) and not dr_fit.empty):
            st.info("Dose-response results are not available yet. Run the pipeline first.")
        else:
            left_dr, right_dr = st.columns([2.2, 1.0], gap="large")
            with right_dr:
                st.markdown("#### Configuration & Metrics")
                with st.container(border=True):
                    st.markdown("**Filters**")
                    response_metric = st.selectbox("Metric to display",
                                                   ["mu","A","lambda","integral"],
                                                   index=0, key="dr_metric_tab")
                    if manual_review_mode:
                        ls_ui = st.selectbox("Label source", ["True","Predicted"],
                                             index=0, key="dr_label_source_tab_manual")
                        label_source = "final" if ls_ui == "True" else "pred"
                    else:
                        st.selectbox("Label source", ["Predicted"], index=0,
                                     key="dr_label_source_tab_auto")
                        label_source = "pred"
                    f1, f2 = st.columns(2)
                    include_unsure  = f1.checkbox("Include 'Unsure' Curves ⚠️", value=False,
                                                  key="dr_inc_unsure_tab")
                    include_invalid = f2.checkbox("Include 'Invalid' Curves ⚠️", value=False,
                                                  key="dr_inc_invalid_tab")

                dr_pill_key = f"dr_overlay_pills_tab_{chosen}"
                dr_bs_sel   = bool(manual_review_mode and
                                   "Bootstrap CI" in st.session_state.get(dr_pill_key, []))
                tab_dr_payload = build_dr_payload(
                    gc_fit=gc_fit, labels_df=labels_df, dr_boot=dr_boot,
                    test_id=file_stem, response_metric=response_metric,
                    label_source=label_source,
                    include_unsure=include_unsure, include_invalid=include_invalid,
                    dr_s=grofit_opts.dr_s, smooth_dr=grofit_opts.smooth_dr,
                    dr_x_transform=grofit_opts.dr_x_transform,
                    dr_y_transform=grofit_opts.dr_y_transform,
                    show_bootstrap=bool(dr_bs_sel or
                                        (isinstance(dr_boot, pd.DataFrame) and not dr_boot.empty)),
                )
                fit_pl  = tab_dr_payload.get("fit", {})
                ec50    = to_numeric_scalar(fit_pl.get("ec50"))
                y_mid   = to_numeric_scalar(fit_pl.get("y_mid"))
                dr_method = fit_pl.get("dr_method")
                ec50_status = fit_pl.get("ec50_status")
                boot    = tab_dr_payload.get("bootstrap", {})
                ec50_ci = boot.get("ec50_ci") if boot.get("ran") else None
                ec50_lo = to_numeric_scalar(ec50_ci[0]) if ec50_ci and len(ec50_ci)==2 else np.nan
                ec50_hi = to_numeric_scalar(ec50_ci[1]) if ec50_ci and len(ec50_ci)==2 else np.nan
                n_pts   = int(tab_dr_payload.get("n_points", 0))
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
                    elif n_pts < min_points_req:
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
                    st.metric("95% CI",
                              f"{ec50_lo:.4g} - {ec50_hi:.4g}"
                              if np.isfinite(ec50_lo) and np.isfinite(ec50_hi) else "NA")
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
                    st.metric("Points Used", n_pts)
                    if excluded > 0:
                        st.caption(f"Excluded points: {excluded}")

            with left_dr:
                st.markdown("#### Dose-Response Curve")
                if manual_review_mode:
                    dr_overlay_store = st.session_state.setdefault("dr_overlay_store", {})
                    dr_default = dr_overlay_store.get(str(chosen), [])
                    dr_boot_pills = st_pills_multi(" ", ["Bootstrap CI"],
                                                   default=dr_default, key=dr_pill_key)
                    show_dr_boot  = "Bootstrap CI" in dr_boot_pills
                    dr_overlay_store[str(chosen)] = list(dr_boot_pills)
                else:
                    show_dr_boot = False
                dr_fig = make_dr_plot(tab_dr_payload, show_bootstrap=show_dr_boot)
                dr_fig.update_layout(height=600)
                st.plotly_chart(dr_fig, use_container_width=True)

    # -------------------------------------------------------------------
    # Manual Grofit trigger (fires after dialog confirm)
    # -------------------------------------------------------------------
    if st.session_state.get("trigger_grofit_run"):
        st.session_state["trigger_grofit_run"] = False
        if manual_review_mode:
            gc_bs   = st.session_state.get("exit_gc_bootstrap",   "False")
            pf      = st.session_state.get("exit_preferred_model", "Best Model")
            rm      = st.session_state.get("exit_response_metric", "mu")
            dr_bs   = st.session_state.get("exit_dr_bootstrap",    "True")

            with st.spinner("Preparing Grofit input and running Grofit..."):
                grofit_src = results.get("grofit_df")
                if not isinstance(grofit_src, pd.DataFrame) or grofit_src.empty:
                    grofit_src = build_grofit_input_df(
                        wide_df=wide_original, out_df=out_df, review_df=review_df,
                        manual_review_mode=True,
                        meta_df=results.get("meta_df", out_df),
                        audit_df=results.get("audit_df"),
                    )
                gt_all  = wide_to_grofit_tidy(grofit_src, file_tag=file_stem)
                true_m  = (grofit_src.set_index("Test Id")["True Label"].to_dict()
                           if "True Label" in grofit_src.columns else {})
                gt_all["true_label"]    = gt_all["curve_id"].map(true_m).apply(normalize_label)
                gt_all["is_valid_true"] = gt_all["true_label"].map(label_is_valid).fillna(False).astype(bool)

                fit_map = {"Best Model":"Both (param + spline)","Spline":"Spline only","Parametric":"Parametric only"}
                opt_map = {"Both (param + spline)":"b","Spline only":"s","Parametric only":"m"}
                fit_opt = opt_map.get(fit_map.get(pf,"Both (param + spline)"), "b")
                gc_B    = 0 if gc_bs == "False" else grofit_opts.gc_boot_B
                dr_B    = grofit_opts.dr_boot_B if dr_bs == "True" else 0

                eff = GrofitOptions(**grofit_opts.__dict__)
                eff.response_var      = rm
                eff.fit_opt           = fit_opt
                eff.gc_boot_B         = int(gc_B)
                eff.dr_boot_B         = int(dr_B)
                eff.bootstrap_method  = normalize_bootstrap_method(eff.bootstrap_method)

                gc_fit2 = dr_fit2 = gc_boot2 = dr_boot2 = pd.DataFrame()
                gc_aud2 = dr_aud2 = pd.DataFrame()
                zip_bytes2 = b""; zip_name2 = ""

                if not gt_all.empty:
                    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
                        res2 = run_grofit_pipeline(
                            curves_df=gt_all, response_var=rm,
                            have_atleast=grofit_opts.have_atleast,
                            gc_boot_B=gc_B, dr_boot_B=dr_B,
                            spline_auto_cv=grofit_opts.spline_auto_cv,
                            spline_s=grofit_opts.spline_s, smooth_gc=grofit_opts.smooth_gc,
                            smooth_dr=grofit_opts.smooth_dr,
                            dr_x_transform=grofit_opts.dr_x_transform,
                            dr_y_transform=grofit_opts.dr_y_transform,
                            dr_s=grofit_opts.dr_s, fit_opt=fit_opt,
                            bootstrap_method=normalize_bootstrap_method(grofit_opts.bootstrap_method),
                            validity_col="__all__", random_state=42, export_dir=Path(td),
                        )
                        gc_fit2  = res2.get("gc_fit",  pd.DataFrame())
                        dr_fit2  = res2.get("dr_fit",  pd.DataFrame())
                        gc_boot2 = res2.get("gc_boot", pd.DataFrame())
                        dr_boot2 = res2.get("dr_boot", pd.DataFrame())
                        gc_aud2  = res2.get("gc_audit", pd.DataFrame())
                        dr_aud2  = res2.get("dr_audit", pd.DataFrame())
                        zip_name2, zip_bytes2 = build_export_zip(
                            wide_df=wide_original, out_df=out_df, review_df=review_df,
                            gc_fit=gc_fit2, gc_boot=gc_boot2,
                            dr_fit=dr_fit2, dr_boot=dr_boot2,
                            proc_wide_df=final_merged, grofit_opts=eff,
                            settings=settings, mode_label="MANUAL", file_stem=file_stem,
                            predicting_model=predicting_model,
                            selected_gc_bootstrap=str(gc_bs), selected_preferred_fit=str(pf),
                            selected_response_metric=str(rm), selected_dr_bootstrap=str(dr_bs),
                            export_label_filter=results.get("export_label_filter","Valid"),
                            export_dr_include_unsure=bool(results.get("export_dr_include_unsure",False)),
                            export_dr_include_invalid=bool(results.get("export_dr_include_invalid",False)),
                            audit_df=results.get("audit_df"), grofit_df=results.get("grofit_df"),
                            grofit_tidy_all=gt_all, stage2_config=results.get("stage2_config"),
                        )

            results.update(
                grofit_tidy_all=gt_all, grofit_opts=eff,
                gc_fit=gc_fit2, dr_fit=dr_fit2,
                gc_boot=gc_boot2, dr_boot=dr_boot2,
                gc_audit=gc_aud2, dr_audit=dr_aud2,
                zip_bytes=zip_bytes2,
                zip_name=zip_name2 if zip_name2 else "",
                grofit_ran=True,
            )
            st.session_state["last_run_results"] = results
            st.rerun()

    st.markdown("---")
