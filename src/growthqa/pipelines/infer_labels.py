from __future__ import annotations
import json
import re
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
from growthqa.pipelines.build_meta_dataset import (
    run_merge_preprocess_meta,
    TRAIN_STEP_HOURS,
    TRAIN_TMAX_HOURS,
    TRAIN_SMOOTH_METHOD,
    TRAIN_SMOOTH_WINDOW,
    TRAIN_NORMALIZE,
)
from growthqa.config import (
    MIN_POINTS, MAX_GAP_HOURS_OVERRIDE, MISSING_FRAC_OVERRIDE,
    LATE_WINDOW_REFERENCE_STEP_HOURS, LATE_WINDOW_MAX_MISSING_FRAC,
    MIN_LATE_POINTS_FLOOR, MIN_LATE_POINTS_CEILING,MISSING_FEATURE_FRAC_OVERRIDE,
    MIN_LATE_HOURS_ANCHOR, MIN_LATE_POINTS_FALLBACK_RATE_PER_HOUR, STAGE1_SELECTED_FEATURES
)
from growthqa.preprocess.timegrid import parse_time_from_header, get_sorted_time_columns
from growthqa.stage2.late_window import (
    Stage2ConfigEvidence,
    EvidenceScores,
    compute_evidence_scores,
    compute_stage2_checker_status,
)
from growthqa.io.tidy import (
    extract_conc_from_curve_id as _extract_conc_from_curve_id,
    find_concentration_col as _find_concentration_col,
    wide_to_grofit_tidy as _canonical_wide_to_grofit_tidy,
)


def _temp_dir_context():
    if sys.version_info >= (3, 10):
        return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    else:
        return tempfile.TemporaryDirectory()


def assert_runtime_matches_model(model_path: str) -> None:
    mp = Path(model_path)
    manifest = mp.with_suffix(".manifest.json")
    if not manifest.exists():
        return
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
        print(
            "Model/runtime version mismatch detected:\n"
            + "\n".join(["- " + p for p in problems])
            + "\nProceeding anyway; retrain or regenerate models to silence this warning.",
            file=sys.stderr,
        )

def _read_val_balanced_accuracy(model_path: str) -> float | None:
    mp = Path(model_path)
    manifest = mp.with_suffix(".manifest.json")
    if not manifest.exists():
        return None
    try:
        m = json.loads(manifest.read_text(encoding="utf-8"))
        v = m.get("val_balanced_accuracy", None)
        return float(v) if v is not None else None
    except Exception:
        return None

def _install_legacy_sklearn_pickle_aliases() -> None:
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


def label_from_stem(stem: str) -> str:
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


def _fmt_conc_for_key(v: object) -> str:
    n = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    if not np.isfinite(n):
        return ""
    return f"{float(n):g}"


def _test_id_encodes_conc(test_id: object) -> bool:
    if test_id is None:
        return False
    s = str(test_id)
    return re.search(
        r"\[(?:\s*Conc\s*=\s*)?([0-9]+(?:\.[0-9]+)?)\s*\]",
        s,
        flags=re.IGNORECASE,
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


def wide_original_to_grofit_tidy(
    wide_original: pd.DataFrame,
    *,
    file_tag: str,
    test_id_col: str = "Test Id",
) -> pd.DataFrame:
    if test_id_col not in wide_original.columns:
        raise ValueError(f"Expected '{test_id_col}' in canonical wide input.")
    return _canonical_wide_to_grofit_tidy(wide_original, file_tag=file_tag, test_id_col=test_id_col)


def _compute_stage2_features_from_wide_evidence(
    wide_raw_df: pd.DataFrame,
    *,
    cfg: Stage2ConfigEvidence,
) -> pd.DataFrame:
    time_cols = get_sorted_time_columns(wide_raw_df)
    rows: list[dict[str, object]] = []

    for _, row_raw in wide_raw_df.iterrows():
        tid = str(row_raw.get("Test Id", ""))
        conc = pd.to_numeric(pd.Series([row_raw.get("Concentration", np.nan)]), errors="coerce").iloc[0]
        curve_key = row_raw.get("curve_key", tid)

        ev = compute_evidence_scores(row_raw, time_cols, cfg)

        has_late = bool(ev.late_coverage_ok)

        rows.append(
            {
                "Test Id": tid,
                "Concentration": conc,
                "curve_key": curve_key,
                "has_late_data": has_late,
                "late_n_points": int(ev.n_late_points),
                "min_late_points_required": int(ev.min_late_points_required),
                "late_span_hours": float(ev.late_span_hours) if np.isfinite(ev.late_span_hours) else np.nan,
                "growth_z_like": float(ev.growth_z_like),
                "artifact_score": float(ev.artifact_score),
                "decline_score": float(ev.decline_score),
                "data_quality": float(ev.data_quality),
                "decision_confidence": float(ev.confidence),
                "late_slope": float(ev.late_slope) if np.isfinite(ev.late_slope) else np.nan,
                "late_delta": float(ev.late_delta) if np.isfinite(ev.late_delta) else np.nan,
                "noise_level": float(ev.noise_level) if np.isfinite(ev.noise_level) else np.nan,
                "late_growth_detected": bool(ev.growth_z_like >= cfg.growth_z_threshold),
                "artifact_detected": bool(ev.artifact_score >= cfg.artifact_score_threshold),
            }
        )

    return pd.DataFrame(rows)


def _assign_stage2_checker_outputs(
    out_df: pd.DataFrame,
    *,
    cfg: Stage2ConfigEvidence,
    unsure_conf_threshold: float | None = None,
) -> pd.DataFrame:
    stage2_labels: list[str] = []
    pred_labels: list[str] = []
    reasons: list[str] = []
    final_labels: list[str] = []

    for _, row in out_df.iterrows():
        s1 = _normalize_label_text(row.get("pred_label", ""))
        s1_conf = pd.to_numeric(pd.Series([row.get("pred_confidence", np.nan)]), errors="coerce").iloc[0]
        s1_conf_valid = pd.to_numeric(pd.Series([row.get("confidence_valid", np.nan)]), errors="coerce").iloc[0]
        _late_n_raw = pd.to_numeric(pd.Series([row.get("late_n_points", 0)]), errors="coerce").iloc[0]

        ev = EvidenceScores(
            growth_z_like=float(pd.to_numeric(pd.Series([row.get("growth_z_like", 0.0)]), errors="coerce").iloc[0]),
            artifact_score=float(pd.to_numeric(pd.Series([row.get("artifact_score", 0.5)]), errors="coerce").iloc[0]),
            data_quality=float(pd.to_numeric(pd.Series([row.get("data_quality", 0.0)]), errors="coerce").iloc[0]),
            confidence=float(pd.to_numeric(pd.Series([row.get("decision_confidence", 0.0)]), errors="coerce").iloc[0]),
            decline_score=float(pd.to_numeric(pd.Series([row.get("decline_score", 0.0)]), errors="coerce").iloc[0]),
            late_slope=float(pd.to_numeric(pd.Series([row.get("late_slope", np.nan)]), errors="coerce").iloc[0]),
            late_delta=float(pd.to_numeric(pd.Series([row.get("late_delta", np.nan)]), errors="coerce").iloc[0]),
            noise_level=float(pd.to_numeric(pd.Series([row.get("noise_level", np.nan)]), errors="coerce").iloc[0]),
            n_late_points=int(_late_n_raw) if pd.notna(_late_n_raw) else 0,
            late_span_hours=float(pd.to_numeric(pd.Series([row.get("late_span_hours", np.nan)]), errors="coerce").iloc[0]),
        )

        status, reason, _ = compute_stage2_checker_status(
            stage1_label=s1,
            stage1_confidence=float(s1_conf) if np.isfinite(s1_conf) else np.nan,
            evidence=ev,
            cfg=cfg,
        )
        stage2_labels.append(status)
        low_confidence = False
        if unsure_conf_threshold is not None and np.isfinite(s1_conf_valid):
            margin = min(float(s1_conf_valid), 1.0 - float(s1_conf_valid))
            low_confidence = margin > float(unsure_conf_threshold) + 1e-9

        if low_confidence:
            label_for_reason = s1 if s1 else "Unknown"
            pred_labels.append("Unsure")
            reasons.append(f"S1_LOW_CONFIDENCE: Confidence({label_for_reason}) = {s1_conf_valid:.2f}")
            final_labels.append("Unsure")
        elif status == "Contradiction":
            stage1_conf_for_reason = float(s1_conf) if np.isfinite(s1_conf) else float("nan")
            label_for_reason = s1 if s1 else "Unknown"
            pred_labels.append(s1 if s1 else "Unsure")
            reasons.append(f"{reason}: Confidence({label_for_reason}) = {stage1_conf_for_reason:.2f}")
            final_labels.append("Unsure")
        elif status == "Insufficient":
            pred_labels.append(s1 if s1 else "Unsure")
            reasons.append(reason)
            final_labels.append(s1 if s1 else "Unsure")
        else:
            pred_labels.append(s1 if s1 else "Unsure")
            reasons.append(reason)
            final_labels.append(s1 if s1 else "Unsure")

    out = out_df.copy()
    out["Stage 2 Label"] = stage2_labels
    out["Label Reason"] = reasons
    out["Pred Label"] = pred_labels
    out["Final Label (S1+S2)"] = final_labels
    out["final_label"] = final_labels 
    return out


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

    dup_ids = wide_df["Test Id"][wide_df["Test Id"].duplicated(keep=False)]
    if not dup_ids.empty:
        examples = sorted(set(dup_ids.astype(str)))[:5]
        raise ValueError(
            "Uploaded data contains duplicate 'Test Id' values, which is not "
            "supported -- each curve in a single upload must have a unique "
            "Test Id (rename the repeated wells/curves, e.g. 'A01' and "
            "'A01_2', before re-uploading). Duplicated value(s): "
            + ", ".join(examples)
            + (" ..." if len(set(dup_ids.astype(str))) > 5 else "")
        )
    wide_raw_df = _attach_curve_key(wide_df.copy())
    time_cols_all = [c for c in wide_raw_df.columns if parse_time_from_header(str(c)) is not None]
    non_time_cols = [c for c in wide_raw_df.columns if c not in time_cols_all]
    early_cols = [c for c in time_cols_all if float(parse_time_from_header(str(c))) <= float(stage2_start)]
    wide_early_raw_df = wide_raw_df[non_time_cols + early_cols].copy()

    with _temp_dir_context() as td:
        tmp_wide_csv = Path(td) / "wide_input.csv"
        wide_early_raw_df.drop(columns=["curve_key"], errors="ignore").to_csv(tmp_wide_csv, index=False)
        raw_merged_df, final_merged_df, meta_df = run_merge_preprocess_meta(
            inputs=[str(tmp_wide_csv)],
            out_raw=None,
            out_final=None,
            out_meta=None,
            step=TRAIN_STEP_HOURS,
            min_points=MIN_POINTS,
            tmax_hours=TRAIN_TMAX_HOURS,
            blank_subtracted=False,
            clip_negatives=False,
            global_blank=None,
            blank_default="ALREADY",
            smooth_method=TRAIN_SMOOTH_METHOD,
            smooth_window=TRAIN_SMOOTH_WINDOW,
            normalize=TRAIN_NORMALIZE,
            loglevel="ERROR",
            augment_trunc=False,

        )

    meta_df = _attach_curve_key(meta_df)
    available_models = discover_models(model_dir)
    model_label_map: dict[str, Path] = {}
    for stem, p in available_models.items():
        label = label_from_stem(stem)
        if label in model_label_map:
            label = f"{label}-{stem}"
        model_label_map[label] = p
    if not model_label_map:
        raise FileNotFoundError(f"No trained model found in {model_dir}.")

    if model_name == "Average":
        pipelines = {}
        for label, path in model_label_map.items():
            try:
                pipelines[label] = load_model_pipeline(str(path))
            except Exception as e:
                print(
                    f"Skipping model '{label}' at {path}: failed to load "
                    f"({type(e).__name__}: {e}). Continuing with remaining "
                    f"ensemble members.",
                    file=sys.stderr,
                )
        if not pipelines:
            raise RuntimeError(
                f"No model in {model_dir} could be loaded in this environment. "
                "Retrain the classifier here (Train / Refresh Classifier) to "
                "regenerate models compatible with the installed library versions."
            )
        per_model_preds = []
        for lbl, pipe in pipelines.items():
            try:
                plabel, pconf, pvalid = predict_hard_with_confidence(pipe, meta_df)
            except Exception as e:
                print(
                    f"Skipping model '{lbl}': loaded but failed to predict "
                    f"({type(e).__name__}: {e}). Continuing with remaining "
                    f"ensemble members.",
                    file=sys.stderr,
                )
                continue
            per_model_preds.append((lbl, plabel, pconf, pvalid))
        if not per_model_preds:
            raise RuntimeError(
                f"No model in {model_dir} could produce predictions in this "
                "environment. Retrain the classifier here (Train / Refresh "
                "Classifier) to regenerate compatible models."
            )
        valid_probs_list = []
        for _, plabel, _, pvalid in per_model_preds:
            if np.any(np.isfinite(pvalid)):
                valid_probs_list.append(pvalid)
            else:
                valid_probs_list.append(_labels_to_prob_valid(plabel))
        valid_probs = np.vstack(valid_probs_list)
        val_scores = np.array(
            [_read_val_balanced_accuracy(str(model_label_map[lbl])) for lbl, _, _, _ in per_model_preds],
            dtype=float,
        )
        if np.all(np.isfinite(val_scores)) and np.nansum(val_scores) > 0:
            model_weights = val_scores / np.nansum(val_scores)
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
    out_df["S1 Confidence Valid"] = out_df["confidence_valid"]
    stage2_cfg = Stage2ConfigEvidence(
        stage2_start=float(stage2_start),
        late_window_reference_step_hours=float(LATE_WINDOW_REFERENCE_STEP_HOURS),
        late_window_max_missing_frac=float(LATE_WINDOW_MAX_MISSING_FRAC),
        min_late_points_floor=int(MIN_LATE_POINTS_FLOOR),
        min_late_points_ceiling=int(MIN_LATE_POINTS_CEILING),
        min_late_hours_anchor=float(MIN_LATE_HOURS_ANCHOR),
        min_late_points_fallback_rate_per_hour=float(MIN_LATE_POINTS_FALLBACK_RATE_PER_HOUR),
    )
    stage2_df = _compute_stage2_features_from_wide_evidence(wide_raw_df, cfg=stage2_cfg)

    out_df = out_df.merge(
        stage2_df.drop(columns=["Test Id", "Concentration"], errors="ignore"),
        on=["curve_key"],
        how="left",
    )
    out_df = _assign_stage2_checker_outputs(out_df, cfg=stage2_cfg, unsure_conf_threshold=unsure_conf_threshold)
    ood_gap_mask = (
        pd.to_numeric(out_df.get("max_gap_hours", np.nan), errors="coerce") > MAX_GAP_HOURS_OVERRIDE
    ) | (
        pd.to_numeric(out_df.get("missing_frac_on_grid", np.nan), errors="coerce") > MISSING_FRAC_OVERRIDE
    )
    ood_gap_mask = ood_gap_mask.fillna(False)
    if ood_gap_mask.any():
        out_df.loc[ood_gap_mask, "final_label"] = "Unsure"
        out_df.loc[ood_gap_mask, "Final Label (S1+S2)"] = "Unsure"
        out_df.loc[ood_gap_mask, "Pred Label"] = "Unsure"
        out_df.loc[ood_gap_mask, "pred_label"] = "Unsure"
        out_df.loc[ood_gap_mask, "Label Reason"] = "OUT_OF_DISTRIBUTION_GAP_OVERRIDE"
    _feat_cols = [c for c in STAGE1_SELECTED_FEATURES if c in out_df.columns]
    n_missing = out_df[_feat_cols].isna().sum(axis=1)
    missing_frac = n_missing / max(len(_feat_cols), 1)
    out_df["n_features_missing"] = n_missing
    out_df["feature_missing_frac"] = missing_frac.round(3)
    feature_completeness_mask = (missing_frac > MISSING_FEATURE_FRAC_OVERRIDE).fillna(False)
    if feature_completeness_mask.any():
        out_df.loc[feature_completeness_mask, "final_label"] = "Unsure"
        out_df.loc[feature_completeness_mask, "Final Label (S1+S2)"] = "Unsure"
        out_df.loc[feature_completeness_mask, "Pred Label"] = "Unsure"
        out_df.loc[feature_completeness_mask, "pred_label"] = "Unsure"
        out_df.loc[feature_completeness_mask, "Label Reason"] = "FEATURE_COMPLETENESS_OVERRIDE"
    too_sparse_mask = pd.to_numeric(out_df.get("too_sparse", False), errors="coerce").fillna(0).astype(int).eq(1)
    if too_sparse_mask.any():
        out_df.loc[too_sparse_mask, "final_label"] = "Unsure"
        out_df.loc[too_sparse_mask, "Final Label (S1+S2)"] = "Unsure"
        out_df.loc[too_sparse_mask, "Pred Label"] = "Unsure"
        out_df.loc[too_sparse_mask, "pred_label"] = "Unsure"
        out_df.loc[too_sparse_mask, "Label Reason"] = "TOO_SPARSE_OVERRIDE"

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