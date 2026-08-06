"""
Data-wrangling helpers: wide-tidy conversion, late-growth detection,
final-label assignment, review-df initialisation, classifier/Grofit
artifact builders and export-ZIP construction.
"""
from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from config import GrofitOptions, InferenceSettings, MODEL_DIR
from utils import (
    extract_conc_from_curve_id,
    label_is_valid,
    make_curve_key,
    normalize_bootstrap_method,
    normalize_label,
    resolve_display_label,
    
)
from growthqa.preprocess.timegrid import parse_time_from_header, get_time_columns
from growthqa.io.audit import build_classifier_audit_df
from growthqa.io.grofit_io import build_grofit_input_df as _build_grofit_artifact
from growthqa.grofit.pipeline import run_grofit_pipeline
from growthqa.viz.payloads import build_curve_payloads, build_dr_payload
from plots import make_overlay_plot_payload, make_dr_plot
from growthqa.io.tidy import find_concentration_col, wide_to_grofit_tidy


def init_review_df(out_df: pd.DataFrame, wide_df: pd.DataFrame) -> pd.DataFrame:
    df= out_df.copy()
    if "is_valid_pred"  not in df.columns: df["is_valid_pred"]= df["pred_label"].map(label_is_valid)
    if "final_label"    not in df.columns: df["final_label"]= df["pred_label"].astype(str)
    if "true_label"     not in df.columns: df["true_label"]= df["final_label"].astype(str)
    if "is_valid_true"  not in df.columns: df["is_valid_true"]= df["true_label"].map(label_is_valid).astype(bool)
    if "Reviewed"       not in df.columns: df["Reviewed"]= False
    df["is_valid_final"]= df["is_valid_true"].astype(bool)

    if "Concentration" in df.columns:
        conc= df["Concentration"]
    elif "Concentration" in wide_df.columns:
        conc= df["Test Id"].map(wide_df.set_index("Test Id")["Concentration"])
    else:
        conc= df["Test Id"].astype(str).map(extract_conc_from_curve_id)
    df["Concentration"]= pd.to_numeric(conc, errors="coerce")
    df["CurveKey"]= df.apply(
        lambda r: make_curve_key(str(r["Test Id"]), r["Concentration"]), axis=1
    )
    return df

def build_run_info(
    *,
    mode_label: str, file_stem: str, predicting_model: str,
    grofit_opts: GrofitOptions, settings: InferenceSettings,
    stage2_config: dict | None,
    user_selections: dict,
    classifier_features: list[str] | None= None,
) -> dict:
    preprocessing= dict(settings.__dict__) if settings else {}
    stage2_full= dict(stage2_config) if stage2_config else {}

    return {
        "run": {
            "mode": mode_label,
            "timestamp": datetime.now().isoformat(),
            "file_stem": file_stem,
            "predicting_model": predicting_model,
        },
        "preprocessing": {
            k: preprocessing[k] for k in [
                  "min_points", "tmax_hours",
             ] if k in preprocessing
        },
        "stage2_thresholds": {
            k: stage2_full[k] for k in [
                "stage2_start",
                "min_late_points_floor", "min_late_points_ceiling",
                "min_late_hours_anchor", "min_late_points_fallback_rate_per_hour",
                "late_window_reference_step_hours", "late_window_max_missing_frac",
                "quality_threshold", "growth_z_threshold",
                "artifact_score_threshold", "decline_score_threshold",
            ] if k in stage2_full
        },
        "grofit_config": {
            **grofit_opts.__dict__,
            "bootstrap_method": normalize_bootstrap_method(grofit_opts.bootstrap_method),
        } if grofit_opts else None,
        "classifier": {
            "selected_features": classifier_features,
            "n_features": len(classifier_features) if classifier_features is not None else None,
        },
        "user_selections": user_selections,
    }

def build_classifier_output(
    *, wide_df: pd.DataFrame, out_df: pd.DataFrame,
    review_df: pd.DataFrame | None, manual_review_mode: bool,
    meta_df: pd.DataFrame | None= None,
    processed_wide_df: pd.DataFrame | None= None,
) -> pd.DataFrame:
    return build_classifier_audit_df(
        wide_original_df=wide_df,
        infer_df=out_df,
        meta_df=meta_df if isinstance(meta_df, pd.DataFrame) else out_df,
        mode="MANUAL" if manual_review_mode else "AUTO",
        review_df=review_df,
        processed_wide_df=processed_wide_df,

    )
 
def build_grofit_input_df(
    *, wide_df: pd.DataFrame, out_df: pd.DataFrame,
    review_df: pd.DataFrame | None, manual_review_mode: bool,
    meta_df: pd.DataFrame | None= None,
    audit_df: pd.DataFrame | None= None,
) -> pd.DataFrame:
    if not isinstance(audit_df, pd.DataFrame):
        audit_df= build_classifier_output(
            wide_df=wide_df, out_df=out_df, review_df=review_df,
            manual_review_mode=manual_review_mode, meta_df=meta_df,
        )
    return _build_grofit_artifact(wide_original_df=wide_df, audit_df=audit_df)

def build_export_zip(
    *, wide_df: pd.DataFrame, out_df: pd.DataFrame,
    review_df: pd.DataFrame | None,
    gc_fit: pd.DataFrame, gc_boot: pd.DataFrame,
    dr_fit: pd.DataFrame, dr_boot: pd.DataFrame,
    gc_audit: pd.DataFrame | None= None,
    dr_audit: pd.DataFrame | None= None,
    proc_wide_df: pd.DataFrame | None= None,
    grofit_opts: GrofitOptions= None, settings: InferenceSettings= None,
    mode_label: str= "AUTO", file_stem: str= "", predicting_model: str= "",

    # optional selections
    auto_bootstrap_scope:    str | None= None,
    auto_preferred_model:    str | None= None,
    auto_response_metric:    str | None= None,
    auto_dr_bootstrap:       str | None= None,
    selected_gc_bootstrap:   str | None= None,
    selected_preferred_fit:  str | None= None,
    selected_response_metric: str | None= None,
    selected_dr_bootstrap:   str | None= None,
    export_label_filter:     str= "Valid",
    export_dr_include_unsure:  bool= False,
    export_dr_include_invalid: bool= False,
    audit_df:        pd.DataFrame | None= None,
    grofit_df:       pd.DataFrame | None= None,
    grofit_tidy_all: pd.DataFrame | None= None,
    stage2_config:   dict | None= None,
) -> tuple[str, bytes]:

    # zip_name= f"{mode_label}_{datetime.now().strftime('%m.%d.%y')}_{file_stem}.zip"
    date_tag= datetime.now().strftime("%d.%m.%y")
    results_name= f"{mode_label}_{date_tag}_{file_stem}.zip"
    auditing_name= f"{mode_label}_{date_tag}_{file_stem}_Auditing.zip"

    classifier_df= audit_df if isinstance(audit_df, pd.DataFrame) else build_classifier_output(
        wide_df=wide_df, out_df=out_df, review_df=review_df,
        manual_review_mode=(mode_label== "MANUAL"), processed_wide_df=proc_wide_df,
    )
    grofit_input_df= grofit_df if isinstance(grofit_df, pd.DataFrame) else build_grofit_input_df(
        wide_df=wide_df, out_df=out_df, review_df=review_df,
        manual_review_mode=(mode_label== "MANUAL"), audit_df=classifier_df,
    )

    _feature_list= None
    try:
        fp= Path(MODEL_DIR) / "stage1_features.json"
        if fp.exists():
            _feature_list= json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        pass


    run_info= build_run_info(
        mode_label=mode_label, file_stem=file_stem, predicting_model=predicting_model,
        grofit_opts=grofit_opts, settings=settings, stage2_config=stage2_config,
        classifier_features=_feature_list,
        user_selections={
            "gc_bootstrap": selected_gc_bootstrap if mode_label== "MANUAL" else auto_bootstrap_scope,
            "preferred_fit": selected_preferred_fit if mode_label== "MANUAL" else auto_preferred_model,
            "response_metric": selected_response_metric if mode_label== "MANUAL" else auto_response_metric,
            "dr_bootstrap": selected_dr_bootstrap if mode_label== "MANUAL" else auto_dr_bootstrap,
            "export_label_filter": export_label_filter,
            "export_dr_include_unsure": bool(export_dr_include_unsure),
            "export_dr_include_invalid": bool(export_dr_include_invalid),
        },
    )


    def _csv(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8")

    label_candidates=  ["True Label", "Final Label (S1+S2)"]
    allowed_ids: list[str]= []
    if isinstance(classifier_df, pd.DataFrame) and not classifier_df.empty and "Test Id" in classifier_df.columns:
        lcol= next((c for c in label_candidates if c in classifier_df.columns), None)
        if lcol:
            labels= classifier_df[lcol].map(normalize_label)
            f= str(export_label_filter or "Valid").strip().lower()
            if f== "all":     
                mask= labels.isin(["Valid", "Invalid", "Unsure"])
            elif f== "invalid": 
                mask= labels== "Invalid"
            elif f== "unsure":  
                mask= labels== "Unsure"
            else:                
                mask= labels== "Valid"
            allowed_ids= classifier_df.loc[mask, "Test Id"].astype(str).drop_duplicates().tolist()

    gc_fit_out= gc_fit.copy()
    gc_boot_out= gc_boot.copy()
    gc_audit_out= gc_audit.copy() if isinstance(gc_audit, pd.DataFrame) else pd.DataFrame()
    for tbl in (gc_fit_out, gc_boot_out, gc_audit_out):

        if "add.id" not in tbl.columns or tbl.empty:
            continue
        if allowed_ids:
            tbl.drop(tbl.index[~tbl["add.id"].astype(str).isin(allowed_ids)], inplace=True)
        else:
            tbl.drop(tbl.index, inplace=True)

    dr_fit_out, dr_boot_out= dr_fit.copy(), dr_boot.copy()
    dr_audit_out= dr_audit.copy() if isinstance(dr_audit, pd.DataFrame) else pd.DataFrame()

    dr_allowed_ids: list[str]= []
    if isinstance(classifier_df, pd.DataFrame) and not classifier_df.empty and "Test Id" in classifier_df.columns:
        lcol= next((c for c in label_candidates if c in classifier_df.columns), None)
        if lcol:
            _lbl= classifier_df[lcol].map(normalize_label).astype(str)
            keep= _lbl.eq("Valid")
            if export_dr_include_unsure:  
                keep |= _lbl.eq("Unsure")
            if export_dr_include_invalid: 
                keep |= _lbl.eq("Invalid")
            dr_allowed_ids= classifier_df.loc[keep, "Test Id"].astype(str).drop_duplicates().tolist()

    if isinstance(grofit_tidy_all, pd.DataFrame) and not grofit_tidy_all.empty:
        dr_curves= grofit_tidy_all[
            grofit_tidy_all["curve_id"].astype(str).isin(dr_allowed_ids)
        ].copy() if dr_allowed_ids else grofit_tidy_all.iloc[0:0].copy()
        if not dr_curves.empty:
            try:
                dr_res= run_grofit_pipeline(
                    curves_df=dr_curves, response_var=grofit_opts.response_var,
                    have_atleast=grofit_opts.have_atleast, gc_boot_B=0,
                    dr_boot_B=grofit_opts.dr_boot_B,
                    spline_auto_cv=grofit_opts.spline_auto_cv, spline_s=grofit_opts.spline_s,
                    smooth_gc=grofit_opts.smooth_gc, smooth_dr=grofit_opts.smooth_dr,
                    dr_x_transform=grofit_opts.dr_x_transform, dr_y_transform=grofit_opts.dr_y_transform,
                    dr_s=grofit_opts.dr_s, fit_opt=grofit_opts.fit_opt,
                    bootstrap_method=normalize_bootstrap_method(grofit_opts.bootstrap_method),
                    validity_col="__all__", random_state=42, export_dir=None,
                )
                dr_fit_out= dr_res.get("dr_fit",  pd.DataFrame())
                dr_boot_out= dr_res.get("dr_boot", pd.DataFrame())
                dr_audit_out= dr_res.get("dr_audit", pd.DataFrame())

            except Exception  as e:
                print(
                    f"DR export failed for this run ({type(e).__name__}: {e}); "
                    "drFit/drBoot/drAudit will be empty in the export. "
                    "Continuing with GC results only.",
                    file=sys.stderr,
                )
                dr_fit_out= dr_boot_out= dr_audit_out= pd.DataFrame()
        else:
            dr_audit_out= pd.DataFrame()


    results_bio= io.BytesIO()
    with zipfile.ZipFile(results_bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("gcFit.csv", _csv(gc_fit_out))

        if isinstance(gc_boot_out, pd.DataFrame) and not gc_boot_out.empty:
            zf.writestr("gcBoot.csv", _csv(gc_boot_out))
        if isinstance(dr_fit_out,  pd.DataFrame) and not dr_fit_out.empty:
            zf.writestr("drFit.csv",  _csv(dr_fit_out))
       
        n_plots= 0
        if allowed_ids and isinstance(grofit_tidy_all, pd.DataFrame) and not grofit_tidy_all.empty:
            fit_wide= proc_wide_df if isinstance(proc_wide_df, pd.DataFrame) and not proc_wide_df.empty else wide_df
            lp= pd.DataFrame({"Test Id": allowed_ids})
            if isinstance(classifier_df, pd.DataFrame) and not classifier_df.empty and "Test Id" in classifier_df.columns:
                lm= classifier_df.drop_duplicates("Test Id").set_index("Test Id")
                if "Pred Label" in lm.columns:
                    lp["pred_label"]= lp["Test Id"].map(lm["Pred Label"]).map(normalize_label)
                if "True Label" in lm.columns:
                    lp["final_label"]= lp["Test Id"].map(lm["True Label"]).map(normalize_label)
                else:
                    lp["final_label"]= lp.get("pred_label", "Valid")
                lp["Reviewed"]= False

            payloads= build_curve_payloads(
                curves_df=grofit_tidy_all, raw_wide=wide_df, proc_wide=fit_wide,
                labels_df=lp, gc_boot=gc_boot if isinstance(gc_boot, pd.DataFrame) else None,
                spline_s=grofit_opts.spline_s, smooth_gc=grofit_opts.smooth_gc,
                spline_auto_cv=grofit_opts.spline_auto_cv,
                include_bootstrap=bool(isinstance(gc_boot, pd.DataFrame) and not gc_boot.empty),
                test_id=file_stem, curve_ids=allowed_ids,
            )
            for tid in allowed_ids:
                pl= payloads.get(str(tid))
                if not pl:
                    continue
                pl= dict(pl)
                for k in ("t_raw", "y_raw", "t_proc", "y_proc"):
                    pl[k]= np.array([], dtype=float)
                fig= make_overlay_plot_payload(
                    pl, title=str(tid), show_spline=True, show_model=True,
                    show_bootstrap=bool(pl.get("bootstrap", {}).get("ran", False)),
                )
                safe_tid= re.sub(r"[^A-Za-z0-9._-]+", "_", str(tid)).strip("_") or "curve"
                zf.writestr(f"plots/{safe_tid}.html", fig.to_html(full_html=True, include_plotlyjs="cdn"))
                n_plots += 1

        if (isinstance(dr_fit_out, pd.DataFrame) and not dr_fit_out.empty
                and isinstance(gc_fit_out, pd.DataFrame) and not gc_fit_out.empty
                and isinstance(classifier_df, pd.DataFrame) and not classifier_df.empty
                and "Test Id" in classifier_df.columns):
            ldr= pd.DataFrame()
            ldr["Test Id"]= classifier_df["Test Id"].astype(str)
            ldr["final_label"]= classifier_df.apply(
                lambda r: resolve_display_label(r, fallback=""), axis=1,
            )
            ldr["pred_label"]= ldr["final_label"]

            ldr["Reviewed"]= False
            dr_test_id= None
            if "name" in dr_fit_out.columns:
                names= dr_fit_out["name"].dropna().astype(str).tolist()
                dr_test_id= names[0] if names else None
            if not ldr.empty:
                dr_payload= build_dr_payload(
                    gc_fit=gc_fit_out, labels_df=ldr,
                    dr_boot=dr_boot_out if isinstance(dr_boot_out, pd.DataFrame) else None,
                    test_id=dr_test_id, response_metric=str(grofit_opts.response_var),
                    label_source="final",
                    include_unsure=bool(export_dr_include_unsure),
                    include_invalid=bool(export_dr_include_invalid),
                    dr_s=grofit_opts.dr_s, smooth_dr=grofit_opts.smooth_dr,
                    dr_x_transform=grofit_opts.dr_x_transform,
                    dr_y_transform=grofit_opts.dr_y_transform,
                    show_bootstrap=bool(isinstance(dr_boot_out, pd.DataFrame) and not dr_boot_out.empty),
                )
                dr_fig= make_dr_plot(
                    dr_payload,
                    show_bootstrap=bool(isinstance(dr_boot_out, pd.DataFrame) and not dr_boot_out.empty),
                )
                zf.writestr("plots/Dose_Response.html", dr_fig.to_html(full_html=True, include_plotlyjs="cdn"))
                n_plots += 1

        if n_plots== 0:
            zf.writestr("plots/README.txt", "No plot assets could be generated for this run.")

    auditing_bio= io.BytesIO()
    with zipfile.ZipFile(auditing_bio, "w", compression=zipfile.ZIP_DEFLATED) as az:
        az.writestr("run_info.json", json.dumps(run_info, indent=2))
        az.writestr("classifier_audit.csv", _csv(classifier_df))
        az.writestr("grofit_input.csv", _csv(grofit_input_df))
        if isinstance(gc_audit_out, pd.DataFrame) and not gc_audit_out.empty:
            az.writestr("gc_audit.csv", _csv(gc_audit_out))
        if isinstance(dr_audit_out, pd.DataFrame) and not dr_audit_out.empty:
            az.writestr("dr_audit.csv", _csv(dr_audit_out))
        try:
            perf_candidates= sorted(
                Path(MODEL_DIR).glob("train_results_selected_*.csv"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if perf_candidates:
                az.writestr("classifier_performance.csv", perf_candidates[0].read_bytes())
        except Exception:
            pass

    return results_name, results_bio.getvalue(), auditing_name, auditing_bio.getvalue()
