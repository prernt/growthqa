# app/pipeline/data.py
"""
Data-wrangling helpers: wide ↔ tidy conversion, late-growth detection,
final-label assignment, review-df initialisation, classifier/Grofit
artefact builders, and export-ZIP construction.
No Streamlit dependency.
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
)

# growthqa imports
from growthqa.preprocess.timegrid import parse_time_from_header
from growthqa.io.audit import build_classifier_audit_df
from growthqa.io.grofit_io import build_grofit_input_df as _build_grofit_artifact

try:
    from growthqa.features.meta import compute_late_growth_features
except ImportError:
    def compute_late_growth_features(
        times: np.ndarray, ods: np.ndarray, start: float = 16.0
    ) -> dict:
        """Fallback late-growth detector (used only for UI pre-scan)."""
        t = np.array(times, dtype=float); y = np.array(ods, dtype=float)
        ok = np.isfinite(t) & np.isfinite(y); t, y = t[ok], y[ok]
        out = {"has_late_data": 0, "late_growth_detected": 0,
               "late_slope": float("nan"), "late_delta": float("nan"),
               "late_max_increase": float("nan"), "late_n_points": 0}
        if t.size == 0: return out
        t, y = t[np.argsort(t)], y[np.argsort(t)]
        late = t > float(start)
        if not np.any(late): return out
        t_l, y_l = t[late], y[late]
        out["has_late_data"] = 1; out["late_n_points"] = int(t_l.size)
        if t_l.size >= 2:
            dt, dy = np.diff(t_l), np.diff(y_l); good = dt > 1e-12
            if np.any(good): out["late_slope"] = float(np.nanmedian(dy[good] / dt[good]))
        ref = (t >= float(start)) & (t <= float(start + 2.0))
        if np.any(ref):
            out["late_delta"] = float(
                np.nanmedian(y_l[-max(1, int(np.ceil(0.2 * y_l.size))):]) - np.nanmedian(y[ref])
            )
        od0 = float(np.interp(float(start), t, y)) if t.size >= 2 else float("nan")
        if np.isfinite(od0): out["late_max_increase"] = float(np.nanmax(y_l) - od0)
        out["late_growth_detected"] = int(
            (np.isfinite(out["late_slope"]) and out["late_slope"] > 0.01) or
            (np.isfinite(out["late_delta"]) and out["late_delta"] > 0.03) or
            (np.isfinite(out["late_max_increase"]) and out["late_max_increase"] > 0.05)
        )
        return out


# ---------------------------------------------------------------------------
# Concentration-column detection
# ---------------------------------------------------------------------------

def find_concentration_col(df: pd.DataFrame) -> str | None:
    candidates = ["concentration", "Concentration", "conc", "Conc",
                   "dose", "Dose", "drug_conc", "DrugConc"]
    for c in candidates:
        if c in df.columns: return c
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map: return lower_map[c.lower()]
    return None


# ---------------------------------------------------------------------------
# Wide → Grofit tidy
# ---------------------------------------------------------------------------

def wide_to_grofit_tidy(
    wide_df: pd.DataFrame, *, file_tag: str, test_id_col: str = "Test Id",
) -> pd.DataFrame:
    """
    Convert canonical wide input (one row per curve, time-column headers)
    to the tidy format required by the Grofit pipeline:
    ``test_id, curve_id, concentration, time, y``.

    ``curve_id`` is kept as the full prefixed Test Id so it matches
    classifier outputs without splitting.
    """
    if test_id_col not in wide_df.columns:
        raise ValueError(f"Expected '{test_id_col}' column in wide input.")
    time_cols = [c for c in wide_df.columns if parse_time_from_header(str(c)) is not None]
    if not time_cols:
        raise ValueError("No time columns found (expected T#.## (h) headers).")
    conc_col = find_concentration_col(wide_df)
    id_vars  = [test_id_col] + ([conc_col] if conc_col else [])

    tidy = wide_df.melt(id_vars=id_vars, value_vars=time_cols,
                        var_name="_tl", value_name="y")
    tidy["time"]     = tidy["_tl"].map(lambda s: float(parse_time_from_header(str(s))))
    tidy["test_id"]  = str(file_tag)
    tidy["curve_id"] = tidy[test_id_col].astype(str)
    tidy.drop(columns=["_tl"], inplace=True)

    if conc_col is None:
        tidy["concentration"] = (
            tidy[test_id_col].astype(str).map(extract_conc_from_curve_id)
        )
    else:
        tidy["concentration"] = pd.to_numeric(tidy[conc_col], errors="coerce")
    tidy["concentration"] = pd.to_numeric(tidy["concentration"], errors="coerce").fillna(0.0)
    tidy["y"] = pd.to_numeric(tidy["y"], errors="coerce")
    tidy = tidy.dropna(subset=["time", "y"])
    return tidy[["test_id", "curve_id", "concentration", "time", "y"]]


def grofit_tidy_to_wide(grofit_tidy: pd.DataFrame) -> pd.DataFrame:
    """Convert Grofit tidy (long) back to wide with one row per curve_id."""
    if grofit_tidy is None or grofit_tidy.empty:
        return pd.DataFrame()
    d = grofit_tidy.copy()
    d["time"] = pd.to_numeric(d["time"], errors="coerce")
    d["y"]    = pd.to_numeric(d["y"],    errors="coerce")
    d = d.dropna(subset=["time"])
    d["time_col"] = d["time"].map(lambda t: f"T{float(t):.2f} (h)")
    wide = (
        d.pivot_table(index="curve_id", columns="time_col", values="y", aggfunc="first")
         .reset_index().rename(columns={"curve_id": "Test Id"})
    )
    time_cols = sorted(
        [c for c in wide.columns if c.startswith("T") and "(h)" in c],
        key=lambda c: parse_time_from_header(c) if parse_time_from_header(c) is not None else float("inf"),
    )
    meta_cols = [c for c in ["test_id", "concentration", "is_valid_true",
                              "pred_label", "final_label", "pred_confidence"] if c in d.columns]
    if meta_cols:
        meta = (d[["curve_id"] + meta_cols].drop_duplicates("curve_id")
                .rename(columns={"curve_id": "Test Id"}))
        wide = meta.merge(wide, on="Test Id", how="right")
        wide = wide[["Test Id"] + [c for c in meta_cols if c in wide.columns]
                    + [c for c in time_cols if c in wide.columns]]
    else:
        wide = wide[["Test Id"] + [c for c in time_cols if c in wide.columns]]
    return wide


# ---------------------------------------------------------------------------
# Late-growth map
# ---------------------------------------------------------------------------

def late_growth_map_from_wide(wide_df: pd.DataFrame) -> dict[str, int]:
    tcols = [c for c in wide_df.columns if parse_time_from_header(str(c)) is not None]
    if not tcols:
        return {}
    times = np.array([parse_time_from_header(str(c)) for c in tcols], dtype=float)
    out: dict[str, int] = {}
    for _, row in wide_df.iterrows():
        ods  = pd.to_numeric(row[tcols], errors="coerce").to_numpy(dtype=float)
        late = compute_late_growth_features(times, ods, start=16.0)
        out[str(row.get("Test Id"))] = int(late.get("late_growth_detected", 0))
    return out


# ---------------------------------------------------------------------------
# Final label + reason assignment
# ---------------------------------------------------------------------------

def assign_final_reason_labels(
    out_df: pd.DataFrame, wide_df: pd.DataFrame,
) -> pd.DataFrame:
    late_growth   = late_growth_map_from_wide(wide_df)
    final_labels, final_reasons = [], []
    for _, row in out_df.iterrows():
        pred       = str(row.get("pred_label", "")).strip().lower()
        conf_valid = pd.to_numeric(pd.Series([row.get("confidence_valid", np.nan)]), errors="coerce").iloc[0]
        too_sparse = bool(row.get("too_sparse", False))
        gap  = pd.to_numeric(pd.Series([row.get("max_gap_hours",        np.nan)]), errors="coerce").iloc[0]
        miss = pd.to_numeric(pd.Series([row.get("missing_frac_on_grid", np.nan)]), errors="coerce").iloc[0]
        long_gaps = too_sparse or (np.isfinite(gap) and gap > 2.0) or (np.isfinite(miss) and miss > 0.30)

        if too_sparse:
            final_labels.append("UNSURE"); final_reasons.append("UNSURE_TOO_SPARSE"); continue
        if long_gaps:
            final_labels.append("UNSURE"); final_reasons.append("UNSURE_LONG_GAPS"); continue
        if np.isfinite(conf_valid) and min(conf_valid, 1.0 - conf_valid) > 0.40:
            final_labels.append("UNSURE"); final_reasons.append("UNSURE_LOW_CONFIDENCE"); continue
        if pred in {"valid", "true", "1"}:
            final_labels.append("VALID"); final_reasons.append("OK_STAGE1_VALID"); continue
        tid = str(row.get("Test Id"))
        if int(late_growth.get(tid, 0)) == 1:
            final_labels.append("UNSURE");  final_reasons.append("UNSURE_LATE_GROWTH_AFTER_16H")
        else:
            final_labels.append("INVALID"); final_reasons.append("OK_STAGE1_INVALID_NO_LATE_GROWTH")

    out = out_df.copy()
    out["final_label"]  = final_labels
    out["final_reason"] = final_reasons
    return out


# ---------------------------------------------------------------------------
# Review-df initialisation
# ---------------------------------------------------------------------------

def init_review_df(out_df: pd.DataFrame, wide_df: pd.DataFrame) -> pd.DataFrame:
    df = out_df.copy()
    if "is_valid_pred"  not in df.columns: df["is_valid_pred"]  = df["pred_label"].map(label_is_valid)
    if "final_label"    not in df.columns: df["final_label"]    = df["pred_label"].astype(str)
    if "true_label"     not in df.columns: df["true_label"]     = df["final_label"].astype(str)
    if "is_valid_true"  not in df.columns: df["is_valid_true"]  = df["true_label"].map(label_is_valid).astype(bool)
    if "Reviewed"       not in df.columns: df["Reviewed"]       = False
    df["is_valid_final"] = df["is_valid_true"].astype(bool)

    if "Concentration" in df.columns:
        conc = df["Concentration"]
    elif "Concentration" in wide_df.columns:
        conc = df["Test Id"].map(wide_df.set_index("Test Id")["Concentration"])
    else:
        conc = df["Test Id"].astype(str).map(extract_conc_from_curve_id)
    df["Concentration"] = pd.to_numeric(conc, errors="coerce")
    df["CurveKey"]      = df.apply(
        lambda r: make_curve_key(str(r["Test Id"]), r["Concentration"]), axis=1
    )
    return df


# ---------------------------------------------------------------------------
# Classifier audit & Grofit input builders
# ---------------------------------------------------------------------------

def build_classifier_output(
    *, wide_df: pd.DataFrame, out_df: pd.DataFrame,
    review_df: pd.DataFrame | None, manual_review_mode: bool,
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
            return build_classifier_audit_df(**kwargs, processed_wide_df=processed_wide_df)
        except TypeError:
            pass   # older growthqa build
    return build_classifier_audit_df(**kwargs)


def build_grofit_input_df(
    *, wide_df: pd.DataFrame, out_df: pd.DataFrame,
    review_df: pd.DataFrame | None, manual_review_mode: bool,
    meta_df: pd.DataFrame | None = None,
    audit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if not isinstance(audit_df, pd.DataFrame):
        audit_df = build_classifier_output(
            wide_df=wide_df, out_df=out_df, review_df=review_df,
            manual_review_mode=manual_review_mode, meta_df=meta_df,
        )
    return _build_grofit_artifact(wide_original_df=wide_df, audit_df=audit_df)


# ---------------------------------------------------------------------------
# Export ZIP
# ---------------------------------------------------------------------------

def build_export_zip(
    *, wide_df: pd.DataFrame, out_df: pd.DataFrame,
    review_df: pd.DataFrame | None,
    gc_fit: pd.DataFrame, gc_boot: pd.DataFrame,
    dr_fit: pd.DataFrame, dr_boot: pd.DataFrame,
    proc_wide_df: pd.DataFrame | None,
    grofit_opts: GrofitOptions, settings: InferenceSettings,
    mode_label: str, file_stem: str, predicting_model: str,
    # optional selections
    auto_bootstrap_scope:    str | None = None,
    auto_preferred_model:    str | None = None,
    auto_response_metric:    str | None = None,
    auto_dr_bootstrap:       str | None = None,
    selected_gc_bootstrap:   str | None = None,
    selected_preferred_fit:  str | None = None,
    selected_response_metric: str | None = None,
    selected_dr_bootstrap:   str | None = None,
    export_label_filter:     str  = "Valid",
    export_dr_include_unsure:  bool = False,
    export_dr_include_invalid: bool = False,
    audit_df:        pd.DataFrame | None = None,
    grofit_df:       pd.DataFrame | None = None,
    grofit_tidy_all: pd.DataFrame | None = None,
    stage2_config:   dict | None = None,
) -> tuple[str, bytes]:
    """Build and return ``(zip_filename, zip_bytes)``."""
    # Late imports to avoid circular dependencies
    from growthqa.grofit.pipeline import run_grofit_pipeline
    from growthqa.viz.payloads import build_curve_payloads, build_dr_payload
    from plots import make_overlay_plot_payload, make_dr_plot

    zip_name = f"{mode_label}_{datetime.now().strftime('%m.%d.%y')}_{file_stem}.zip"

    classifier_df = audit_df if isinstance(audit_df, pd.DataFrame) else build_classifier_output(
        wide_df=wide_df, out_df=out_df, review_df=review_df,
        manual_review_mode=(mode_label == "MANUAL"), processed_wide_df=proc_wide_df,
    )
    grofit_input_df = grofit_df if isinstance(grofit_df, pd.DataFrame) else build_grofit_input_df(
        wide_df=wide_df, out_df=out_df, review_df=review_df,
        manual_review_mode=(mode_label == "MANUAL"), audit_df=classifier_df,
    )

    _feature_list = None
    try:
        fp = Path(MODEL_DIR) / "stage1_features.json"
        if fp.exists():
            _feature_list = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        pass

    opts_info = {**grofit_opts.__dict__,
                 "bootstrap_method": normalize_bootstrap_method(grofit_opts.bootstrap_method)}
    run_info = {
        "mode": mode_label,
        "timestamp": datetime.now().isoformat(),
        "file_stem": file_stem,
        "predicting_model": predicting_model,
        "grofit_options": opts_info,
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
        } if mode_label == "AUTO" else None,
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
            "python": sys.version, "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__,
        },
    }

    def _csv(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8")

    # --- determine allowed IDs by label filter ---
    label_candidates = (
        ["Pred Label", "pred_label", "True Label"] if mode_label == "AUTO"
        else ["True Label", "Pred Label", "pred_label"]
    )
    allowed_ids: list[str] = []
    if isinstance(classifier_df, pd.DataFrame) and not classifier_df.empty and "Test Id" in classifier_df.columns:
        lcol = next((c for c in label_candidates if c in classifier_df.columns), None)
        if lcol:
            labels = classifier_df[lcol].map(normalize_label)
            f = str(export_label_filter or "Valid").strip().lower()
            if f == "all":     mask = labels.isin(["Valid", "Invalid", "Unsure"])
            elif f == "invalid": mask = labels == "Invalid"
            elif f == "unsure":  mask = labels == "Unsure"
            else:                mask = labels == "Valid"
            allowed_ids = classifier_df.loc[mask, "Test Id"].astype(str).drop_duplicates().tolist()

    # --- filter GC tables ---
    gc_fit_out  = gc_fit.copy()
    gc_boot_out = gc_boot.copy()
    for tbl in (gc_fit_out, gc_boot_out):
        if "add.id" not in tbl.columns or tbl.empty:
            continue
        if allowed_ids:
            tbl.drop(tbl.index[~tbl["add.id"].astype(str).isin(allowed_ids)], inplace=True)
        else:
            tbl.drop(tbl.index, inplace=True)

    # --- DR re-computation for filtered label set ---
    dr_fit_out, dr_boot_out = dr_fit.copy(), dr_boot.copy()
    dr_allowed_ids: list[str] = []
    if isinstance(classifier_df, pd.DataFrame) and not classifier_df.empty and "Test Id" in classifier_df.columns:
        lcol = next((c for c in label_candidates if c in classifier_df.columns), None)
        if lcol:
            _lbl = classifier_df[lcol].map(normalize_label).astype(str)
            keep = _lbl.eq("Valid")
            if export_dr_include_unsure:  keep |= _lbl.eq("Unsure")
            if export_dr_include_invalid: keep |= _lbl.eq("Invalid")
            dr_allowed_ids = classifier_df.loc[keep, "Test Id"].astype(str).drop_duplicates().tolist()

    if isinstance(grofit_tidy_all, pd.DataFrame) and not grofit_tidy_all.empty:
        dr_curves = grofit_tidy_all[
            grofit_tidy_all["curve_id"].astype(str).isin(dr_allowed_ids)
        ].copy() if dr_allowed_ids else grofit_tidy_all.iloc[0:0].copy()
        if not dr_curves.empty:
            try:
                dr_res = run_grofit_pipeline(
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
                dr_fit_out  = dr_res.get("dr_fit",  pd.DataFrame())
                dr_boot_out = dr_res.get("dr_boot", pd.DataFrame())
            except Exception:
                dr_fit_out = dr_boot_out = pd.DataFrame()

    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("classifier_output.csv", _csv(classifier_df))
        zf.writestr("grofit_input.csv",      _csv(grofit_input_df))
        zf.writestr("run_info.json",          json.dumps(run_info, indent=2))
        zf.writestr("gcFit.csv",              _csv(gc_fit_out))
        if isinstance(gc_boot_out, pd.DataFrame) and not gc_boot_out.empty:
            zf.writestr("gcBoot.csv", _csv(gc_boot_out))
        if isinstance(dr_fit_out,  pd.DataFrame) and not dr_fit_out.empty:
            zf.writestr("drFit.csv",  _csv(dr_fit_out))
        if isinstance(dr_boot_out, pd.DataFrame) and not dr_boot_out.empty:
            zf.writestr("drBoot.csv", _csv(dr_boot_out))

        try:
            perf = Path(MODEL_DIR) / "classifier_performance_latest.csv"
            if perf.exists():
                zf.writestr("classifier_performance.csv", perf.read_bytes())
        except Exception:
            pass

        # --- per-curve HTML plots ---
        n_plots = 0
        if allowed_ids and isinstance(grofit_tidy_all, pd.DataFrame) and not grofit_tidy_all.empty:
            fit_wide = proc_wide_df if isinstance(proc_wide_df, pd.DataFrame) and not proc_wide_df.empty else wide_df
            lp = pd.DataFrame({"Test Id": allowed_ids})
            if isinstance(classifier_df, pd.DataFrame) and not classifier_df.empty and "Test Id" in classifier_df.columns:
                lm = classifier_df.drop_duplicates("Test Id").set_index("Test Id")
                if "Pred Label" in lm.columns:
                    lp["pred_label"]  = lp["Test Id"].map(lm["Pred Label"]).map(normalize_label)
                if "True Label" in lm.columns:
                    lp["final_label"] = lp["Test Id"].map(lm["True Label"]).map(normalize_label)
                else:
                    lp["final_label"] = lp.get("pred_label", "Valid")
                lp["Reviewed"] = False

            payloads = build_curve_payloads(
                curves_df=grofit_tidy_all, raw_wide=wide_df, proc_wide=fit_wide,
                labels_df=lp, gc_boot=gc_boot if isinstance(gc_boot, pd.DataFrame) else None,
                spline_s=grofit_opts.spline_s, smooth_gc=grofit_opts.smooth_gc,
                spline_auto_cv=grofit_opts.spline_auto_cv,
                include_bootstrap=bool(isinstance(gc_boot, pd.DataFrame) and not gc_boot.empty),
                test_id=file_stem, curve_ids=allowed_ids,
            )
            for tid in allowed_ids:
                pl = payloads.get(str(tid))
                if not pl:
                    continue
                pl = dict(pl)
                for k in ("t_raw", "y_raw", "t_proc", "y_proc"):
                    pl[k] = np.array([], dtype=float)
                fig = make_overlay_plot_payload(
                    pl, title=str(tid), show_spline=True, show_model=True,
                    show_bootstrap=bool(pl.get("bootstrap", {}).get("ran", False)),
                )
                safe_tid = re.sub(r"[^A-Za-z0-9._-]+", "_", str(tid)).strip("_") or "curve"
                zf.writestr(f"plots/{safe_tid}.html", fig.to_html(full_html=True, include_plotlyjs="cdn"))
                n_plots += 1

        # --- DR plot ---
        if (isinstance(dr_fit_out, pd.DataFrame) and not dr_fit_out.empty
                and isinstance(gc_fit_out, pd.DataFrame) and not gc_fit_out.empty
                and isinstance(classifier_df, pd.DataFrame) and not classifier_df.empty
                and "Test Id" in classifier_df.columns):
            ldr = pd.DataFrame()
            ldr["Test Id"] = classifier_df["Test Id"].astype(str)
            label_src = ("True Label" if "True Label" in classifier_df.columns
                         else "Pred Label" if "Pred Label" in classifier_df.columns else None)
            ldr["final_label"] = classifier_df[label_src].map(normalize_label) if label_src else ""
            ldr["pred_label"]  = ldr["final_label"]
            ldr["Reviewed"]    = False
            dr_test_id = None
            if "name" in dr_fit_out.columns:
                names = dr_fit_out["name"].dropna().astype(str).tolist()
                dr_test_id = names[0] if names else None
            if not ldr.empty:
                dr_payload = build_dr_payload(
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
                dr_fig = make_dr_plot(
                    dr_payload,
                    show_bootstrap=bool(isinstance(dr_boot_out, pd.DataFrame) and not dr_boot_out.empty),
                )
                zf.writestr("plots/Dose_Response.html", dr_fig.to_html(full_html=True, include_plotlyjs="cdn"))
                n_plots += 1

        if n_plots == 0:
            zf.writestr("plots/README.txt", "No plot assets could be generated for this run.")

    return zip_name, bio.getvalue()
