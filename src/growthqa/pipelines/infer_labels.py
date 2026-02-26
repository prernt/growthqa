from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import importlib
import joblib
import numpy as np
import pandas as pd
import platform
import sklearn

from growthqa.pipelines.build_meta_dataset import run_merge_preprocess_meta
from growthqa.preprocess.timegrid import parse_time_from_header

# NEW Stage-2 (evidence-based)
from growthqa.stage2.late_window import (
    Stage2ConfigEvidence,
    EvidenceScores,
    compute_evidence_scores,
    compute_stage2_checker_status,
)


# ============================================================
# Model loading utilities (unchanged from your zip)
# ============================================================
def assert_runtime_matches_model(model_path: str) -> None:
    mp = Path(model_path)
    manifest = mp.with_suffix(".manifest.json")
    if not manifest.exists():
        return

    m = __import__("json").loads(manifest.read_text(encoding="utf-8"))
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
        print(
            "Model/runtime version mismatch detected:\n"
            + "\n".join(["- " + p for p in problems])
            + "\nProceeding anyway; retrain or regenerate models to silence this warning.",
            file=sys.stderr,
        )

def _install_legacy_sklearn_pickle_aliases() -> None:
    """
    Backward-compatibility shim for old sklearn pickles.
    Some older HistGradientBoosting models reference a private module path:
      sklearn.ensemble._hist_gradient_boosting.loss
    Newer sklearn versions moved losses under sklearn._loss.loss.
    """
    legacy_mod = "sklearn.ensemble._hist_gradient_boosting.loss"
    if legacy_mod in sys.modules:
        return
    try:
        new_mod = importlib.import_module("sklearn._loss.loss")
    except Exception:
        return
    sys.modules[legacy_mod] = new_mod


def load_model_pipeline(model_path: str):
    _install_legacy_sklearn_pickle_aliases()
    assert_runtime_matches_model(model_path)
    return joblib.load(model_path)


def discover_models(model_dir: str | Path) -> dict[str, Path]:
    p = Path(model_dir)
    if not p.exists():
        return {}
    return {f.stem: f for f in sorted(p.glob("*.joblib"))}


def _label_from_stem(stem: str) -> str:
    s = stem.lower()
    if "hgb" in s or "hist" in s:
        return "HGB"
    if "rf" in s or "random" in s:
        return "RF"
    if "lr" in s or "logreg" in s or "logistic" in s:
        return "LR"
    return stem


def _label_is_valid(label: object) -> bool:
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return False
    return str(label).strip().lower() in {"valid", "true", "1"}


def _normalize_label_text(label: object, default: str = "Unsure") -> str:
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return default
    s = str(label).strip().lower()
    if s in {"valid", "true", "1"}:
        return "Valid"
    if s in {"invalid", "false", "0"}:
        return "Invalid"
    if s in {"unsure", "unknown"}:
        return "Unsure"
    return str(label).strip() if str(label).strip() else default


def _labels_to_prob_valid(labels: np.ndarray) -> np.ndarray:
    lbl = np.char.lower(labels.astype(str))
    prob = np.full(lbl.shape, np.nan, dtype=float)
    prob[np.isin(lbl, ["valid", "true", "1"])] = 1.0
    prob[np.isin(lbl, ["invalid", "false", "0"])] = 0.0
    return prob


def predict_hard_with_confidence(pipeline, meta_df: pd.DataFrame):
    non_features = {"FileName", "Test Id"}
    X = meta_df.drop(columns=[c for c in meta_df.columns if c in non_features], errors="ignore")

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

    pred_label_norm = []
    for v in pred_label:
        s = str(v).strip().lower()
        if s in {"1", "true", "valid"}:
            pred_label_norm.append("Valid")
        elif s in {"0", "false", "invalid"}:
            pred_label_norm.append("Invalid")
        else:
            pred_label_norm.append(str(v).strip())
    return np.array(pred_label_norm, dtype=object), conf, p_valid


def _safe_get_setting(settings: Any, key: str, default: Any) -> Any:
    if isinstance(settings, dict):
        return settings.get(key, default)
    return getattr(settings, key, default)


# ============================================================
# Curve key helpers (kept from your zip)
# ============================================================
def _extract_conc_from_curve_id(curve_id: str) -> float | None:
    if curve_id is None:
        return None
    s = str(curve_id)
    m = __import__("re").search(
        r"\[(?:\s*Conc\s*=\s*)?([0-9]+(?:\.[0-9]+)?)\s*\]",
        s,
        flags=__import__("re").IGNORECASE,
    )
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _find_concentration_col(df: pd.DataFrame) -> str | None:
    candidates = ["concentration", "Concentration", "conc", "Conc", "dose", "Dose", "drug_conc", "DrugConc"]
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _fmt_conc_for_key(v: object) -> str:
    n = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    if not np.isfinite(n):
        return ""
    return f"{float(n):g}"


def _test_id_encodes_conc(test_id: object) -> bool:
    if test_id is None:
        return False
    s = str(test_id)
    return __import__("re").search(
        r"\[(?:\s*Conc\s*=\s*)?([0-9]+(?:\.[0-9]+)?)\s*\]",
        s,
        flags=__import__("re").IGNORECASE,
    ) is not None


def _attach_curve_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Test Id" not in out.columns:
        return out
    out["Test Id"] = out["Test Id"].astype(str)

    enc = out["Test Id"].map(_test_id_encodes_conc)

    if "Concentration" in out.columns:
        out["Concentration"] = pd.to_numeric(out["Concentration"], errors="coerce")
        conc_txt = out["Concentration"].map(_fmt_conc_for_key)
        has_conc = out["Concentration"].notna()

        out["curve_key"] = out["Test Id"]
        use_append = has_conc & (~enc)
        out.loc[use_append, "curve_key"] = out.loc[use_append, "Test Id"] + "||" + conc_txt.loc[use_append]
    else:
        out["curve_key"] = out["Test Id"]

    return out


# ============================================================
# Wide -> tidy grofit (unchanged from your zip)
# ============================================================
def wide_original_to_grofit_tidy(
    wide_original: pd.DataFrame,
    *,
    file_tag: str,
    test_id_col: str = "Test Id",
) -> pd.DataFrame:
    if test_id_col not in wide_original.columns:
        raise ValueError(f"Expected '{test_id_col}' in canonical wide input.")

    time_cols = [c for c in wide_original.columns if parse_time_from_header(str(c)) is not None]
    if not time_cols:
        raise ValueError("No time columns detected in wide_original (expected T#.## (h) headers).")

    conc_col = _find_concentration_col(wide_original)
    id_vars = [test_id_col] + ([conc_col] if conc_col is not None else [])
    tidy = wide_original.melt(id_vars=id_vars, value_vars=time_cols, var_name="_time_label", value_name="y")
    tidy["time"] = tidy["_time_label"].map(lambda s: float(parse_time_from_header(str(s))))
    tidy.drop(columns=["_time_label"], inplace=True)
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


# ============================================================
# NEW Stage-2 evidence feature computation from wide
# ============================================================
def _get_time_cols(wide_df: pd.DataFrame) -> list[str]:
    cols = [c for c in wide_df.columns if parse_time_from_header(str(c)) is not None]
    return sorted(cols, key=lambda c: float(parse_time_from_header(str(c))))


def _compute_stage2_features_from_wide_evidence(
    wide_raw_df: pd.DataFrame,
    *,
    cfg: Stage2ConfigEvidence,
) -> pd.DataFrame:
    """
    Computes Stage-2 evidence scores PER CURVE using full-horizon raw wide table.

    Returns scalar-only columns (CSV/UI friendly).
    """
    time_cols = _get_time_cols(wide_raw_df)
    rows: list[dict[str, object]] = []

    for _, row_raw in wide_raw_df.iterrows():
        tid = str(row_raw.get("Test Id", ""))
        conc = pd.to_numeric(pd.Series([row_raw.get("Concentration", np.nan)]), errors="coerce").iloc[0]
        curve_key = row_raw.get("curve_key", tid)

        ev = compute_evidence_scores(row_raw, time_cols, cfg)

        has_late = bool(ev.n_late_points >= cfg.min_late_points)

        rows.append(
            {
                "Test Id": tid,
                "Concentration": conc,
                "curve_key": curve_key,
                "has_late_data": has_late,
                "late_window_start": float(cfg.stage2_start),
                "late_n_points": int(ev.n_late_points),
                "late_span_hours": float(ev.late_span_hours) if np.isfinite(ev.late_span_hours) else np.nan,
                # Core evidence
                "growth_z_like": float(ev.growth_z_like),
                "artifact_score": float(ev.artifact_score),
                "data_quality": float(ev.data_quality),
                "decision_confidence": float(ev.confidence),
                # Supporting
                "late_slope": float(ev.late_slope) if np.isfinite(ev.late_slope) else np.nan,
                "late_delta": float(ev.late_delta) if np.isfinite(ev.late_delta) else np.nan,
                "noise_level": float(ev.noise_level) if np.isfinite(ev.noise_level) else np.nan,
                # Flags (thresholded)
                "late_growth_detected": bool(ev.growth_z_like >= cfg.growth_z_threshold),
                "artifact_detected": bool(ev.artifact_score >= cfg.artifact_score_threshold),
            }
        )

    return pd.DataFrame(rows)


def _assign_stage2_checker_outputs(
    out_df: pd.DataFrame,
    *,
    cfg: Stage2ConfigEvidence,
) -> pd.DataFrame:
    """
    Uses the evidence columns already merged into out_df to compute:

      - Stage 2 Label  (checker status: Corroborated / Contradiction / Insufficient)
      - Label Reason
      - Pred Label (final conservative label: contradiction -> Unsure, else keep Stage-1)

    This gives you a thesis-defensible behavior immediately.
    """
    stage2_labels: list[str] = []
    reasons: list[str] = []
    final_labels: list[str] = []

    for _, row in out_df.iterrows():
        s1 = _normalize_label_text(row.get("pred_label", ""))
        s1_conf = pd.to_numeric(pd.Series([row.get("pred_confidence", np.nan)]), errors="coerce").iloc[0]

        ev = EvidenceScores(
            growth_z_like=float(pd.to_numeric(pd.Series([row.get("growth_z_like", 0.0)]), errors="coerce").iloc[0]),
            artifact_score=float(pd.to_numeric(pd.Series([row.get("artifact_score", 0.5)]), errors="coerce").iloc[0]),
            data_quality=float(pd.to_numeric(pd.Series([row.get("data_quality", 0.0)]), errors="coerce").iloc[0]),
            confidence=float(pd.to_numeric(pd.Series([row.get("decision_confidence", 0.0)]), errors="coerce").iloc[0]),
            late_slope=float(pd.to_numeric(pd.Series([row.get("late_slope", np.nan)]), errors="coerce").iloc[0]),
            late_delta=float(pd.to_numeric(pd.Series([row.get("late_delta", np.nan)]), errors="coerce").iloc[0]),
            noise_level=float(pd.to_numeric(pd.Series([row.get("noise_level", np.nan)]), errors="coerce").iloc[0]),
            n_late_points=int(pd.to_numeric(pd.Series([row.get("late_n_points", 0)]), errors="coerce").iloc[0]),
            late_span_hours=float(pd.to_numeric(pd.Series([row.get("late_span_hours", np.nan)]), errors="coerce").iloc[0]),
        )

        status, reason, _ = compute_stage2_checker_status(
            stage1_label=s1,
            stage1_confidence=float(s1_conf) if np.isfinite(s1_conf) else np.nan,
            evidence=ev,
            cfg=cfg,
        )

        stage2_labels.append(status)
        reasons.append(reason)

        # Conservative final decision (thesis-friendly):
        # any contradiction -> Unsure, else preserve Stage-1
        if status == "Contradiction":
            final_labels.append("Unsure")
        elif status == "Insufficient":
            final_labels.append(s1 if s1 else "Unsure")
        else:
            final_labels.append(s1 if s1 else "Unsure")

    out = out_df.copy()
    out["Stage 2 Label"] = stage2_labels
    out["Label Reason"] = reasons
    out["Pred Label"] = final_labels
    out["final_label"] = final_labels  # keep the alias many places expect
    return out


# ============================================================
# Main inference function (Stage-1 same, Stage-2 replaced)
# ============================================================
def run_label_inference_from_uploaded_wide(
    wide_df: pd.DataFrame,
    settings: Any,
    model_dir: str,
    model_name: str = "Average",
    stage2_start: float = 16.0,
    unsure_conf_threshold: float | None = None,
) -> dict[str, pd.DataFrame]:
    if "Test Id" not in wide_df.columns:
        raise ValueError("Uploaded canonical wide data must include 'Test Id'.")

    # full horizon (for Stage-2 evidence)
    wide_raw_df = _attach_curve_key(wide_df.copy())

    # ---- Stage-1 pre-processing still runs on early window (your existing design) ----
    # We keep your original approach: write early-window to temp CSV, run merge/preprocess/meta.
    # Stage-2 ignores truncation and reads directly from wide_raw_df.
    time_cols_all = [c for c in wide_raw_df.columns if parse_time_from_header(str(c)) is not None]
    non_time_cols = [c for c in wide_raw_df.columns if c not in time_cols_all]
    early_cols = [c for c in time_cols_all if float(parse_time_from_header(str(c))) <= float(stage2_start)]
    wide_early_raw_df = wide_raw_df[non_time_cols + early_cols].copy()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tmp_wide_csv = Path(td) / "wide_input.csv"
        wide_early_raw_df.drop(columns=["curve_key"], errors="ignore").to_csv(tmp_wide_csv, index=False)

        blank_default = "RAW" if bool(_safe_get_setting(settings, "input_is_raw", False)) else "ALREADY"
        raw_merged_df, final_merged_df, meta_df = run_merge_preprocess_meta(
            inputs=[str(tmp_wide_csv)],
            out_raw=None,
            out_final=None,
            out_meta=None,
            step=float(_safe_get_setting(settings, "step", 0.5)),
            min_points=int(_safe_get_setting(settings, "min_points", 3)),
            low_res_threshold=int(_safe_get_setting(settings, "low_res_threshold", 7)),
            tmax_hours=_safe_get_setting(settings, "tmax_hours", 16.0),
            auto_tmax=bool(_safe_get_setting(settings, "auto_tmax", False)),
            auto_tmax_coverage=float(_safe_get_setting(settings, "auto_tmax_coverage", 0.8)),
            blank_subtracted=True,
            clip_negatives=bool(_safe_get_setting(settings, "clip_negatives", False)),
            global_blank=_safe_get_setting(settings, "global_blank", None),
            blank_status_csv=None,
            blank_default=blank_default,
            smooth_method=str(_safe_get_setting(settings, "smooth_method", "SGF")),
            smooth_window=int(_safe_get_setting(settings, "smooth_window", 5)),
            normalize=str(_safe_get_setting(settings, "normalize", "MINMAX")),
            loglevel="ERROR",
            augment_trunc=False,
        )

    meta_df = _attach_curve_key(meta_df)

    # ---- Stage-1 model inference (unchanged) ----
    available_models = discover_models(model_dir)
    model_label_map: dict[str, Path] = {}
    for stem, p in available_models.items():
        label = _label_from_stem(stem)
        if label in model_label_map:
            label = f"{label}-{stem}"
        model_label_map[label] = p
    if not model_label_map:
        raise FileNotFoundError(f"No trained model found in {model_dir}.")

    if model_name == "Average":
        pipelines = {label: load_model_pipeline(str(path)) for label, path in model_label_map.items()}
        per_model_preds = []
        for lbl, pipe in pipelines.items():
            plabel, pconf, pvalid = predict_hard_with_confidence(pipe, meta_df)
            per_model_preds.append((lbl, plabel, pconf, pvalid))
        valid_probs_list = []
        for _, plabel, _, pvalid in per_model_preds:
            if np.any(np.isfinite(pvalid)):
                valid_probs_list.append(pvalid)
            else:
                valid_probs_list.append(_labels_to_prob_valid(plabel))
        valid_probs = np.vstack(valid_probs_list)

        eps = 1e-9
        p_clipped = np.clip(valid_probs, eps, 1 - eps)
        model_certainty = np.nanmean(np.abs(p_clipped - 0.5), axis=1)
        if model_certainty.sum() > 0:
            model_weights = model_certainty / model_certainty.sum()
        else:
            model_weights = np.ones(len(valid_probs_list)) / len(valid_probs_list)

        avg_valid = np.nansum(valid_probs * model_weights[:, np.newaxis], axis=0)
        final_prob = np.where(np.isnan(avg_valid), 0.5, avg_valid)
        pred_label = np.where(final_prob >= 0.5, "Valid", "Invalid")
        pred_conf = np.maximum(final_prob, 1 - final_prob)
    else:
        chosen = model_name if model_name in model_label_map else "Average"
        if chosen == "Average":
            return run_label_inference_from_uploaded_wide(
                wide_df=wide_df,
                settings=settings,
                model_dir=model_dir,
                model_name="Average",
                stage2_start=stage2_start,
                unsure_conf_threshold=unsure_conf_threshold,
            )
        pipe = load_model_pipeline(str(model_label_map[chosen]))
        pred_label, pred_conf, p_valid = predict_hard_with_confidence(pipe, meta_df)
        final_prob = p_valid if np.any(np.isfinite(p_valid)) else _labels_to_prob_valid(pred_label)

    out_df = _attach_curve_key(meta_df.copy())

    if "Concentration" not in out_df.columns and "Concentration" in wide_raw_df.columns:
        key_to_conc = dict(zip(wide_raw_df["curve_key"], pd.to_numeric(wide_raw_df["Concentration"], errors="coerce")))
        out_df["Concentration"] = out_df["curve_key"].map(key_to_conc)
        out_df = _attach_curve_key(out_df)

    out_df["pred_label"] = pred_label
    out_df["pred_confidence"] = np.round(pred_conf, 4)
    out_df["confidence_valid"] = np.round(final_prob, 4)
    out_df["confidence_invalid"] = np.round(1 - final_prob, 4)
    out_df["is_valid_pred"] = out_df["pred_label"].map(_label_is_valid).astype(bool)

    out_df["Predicted S1 Label"] = out_df["pred_label"].astype(str)
    out_df["S1 Confidence Valid"] = out_df["confidence_valid"]

    # ---- NEW Stage-2 evidence computation ----
    stage2_cfg = Stage2ConfigEvidence(stage2_start=float(stage2_start))

    stage2_df = _compute_stage2_features_from_wide_evidence(wide_raw_df, cfg=stage2_cfg)

    out_df = out_df.merge(
        stage2_df.drop(columns=["Test Id", "Concentration"], errors="ignore"),
        on=["curve_key"],
        how="left",
    )

    # Checker outputs + conservative final label
    out_df = _assign_stage2_checker_outputs(out_df, cfg=stage2_cfg)

    # Authoritative sparse override: sparse curves can never remain Valid.
    too_sparse_mask = pd.to_numeric(out_df.get("too_sparse", False), errors="coerce").fillna(0).astype(int).eq(1)
    if too_sparse_mask.any():
        out_df.loc[too_sparse_mask, "final_label"] = "Unsure"
        out_df.loc[too_sparse_mask, "Pred Label"] = "Unsure"
        out_df.loc[too_sparse_mask, "pred_label"] = "Unsure"
        out_df.loc[too_sparse_mask, "Label Reason"] = "TOO_SPARSE_OVERRIDE"

    # UI/manual review init
    out_df["Reviewed"] = False
    out_df["is_valid_final"] = out_df["final_label"].map(_label_is_valid).astype(bool)

    if "Is_Valid" in wide_df.columns:
        out_df["Is_Valid_input"] = out_df["Test Id"].map(wide_df.set_index("Test Id")["Is_Valid"])
    elif "is_valid" in wide_df.columns:
        out_df["Is_Valid_input"] = out_df["Test Id"].map(wide_df.set_index("Test Id")["is_valid"])

    file_tag = str(out_df["FileName"].iloc[0]) if "FileName" in out_df.columns and not out_df.empty else "uploaded"
    grofit_tidy_all = wide_original_to_grofit_tidy(wide_df, file_tag=file_tag)

    grofit_tidy_all["curve_key"] = grofit_tidy_all["curve_id"].astype(str)
    has_conc_g = pd.to_numeric(grofit_tidy_all["concentration"], errors="coerce").notna()
    grofit_tidy_all.loc[has_conc_g, "curve_key"] = (
        grofit_tidy_all.loc[has_conc_g, "curve_id"].astype(str)
        + "||"
        + pd.to_numeric(grofit_tidy_all.loc[has_conc_g, "concentration"], errors="coerce").map(_fmt_conc_for_key)
    )

    pred_map_df = out_df[["curve_key", "is_valid_final", "pred_label", "final_label", "pred_confidence"]].drop_duplicates(
        "curve_key"
    )
    grofit_tidy_all = grofit_tidy_all.merge(pred_map_df, on="curve_key", how="left")
    grofit_tidy_all["is_valid_final"] = grofit_tidy_all["is_valid_final"].fillna(False).astype(bool)

    return {
        "raw_merged_df": raw_merged_df,
        "final_merged_df": final_merged_df,
        "meta_df": meta_df,
        "out_df": out_df,
        "grofit_tidy_all": grofit_tidy_all,
        "stage2_config": stage2_cfg.to_dict(),
    }
