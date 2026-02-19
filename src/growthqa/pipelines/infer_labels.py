from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import platform
import sklearn
import importlib
import re
from growthqa.pipelines.build_meta_dataset import run_merge_preprocess_meta
from growthqa.preprocess.timegrid import parse_time_from_header
from growthqa.stage2.late_window import (
    Stage2Config,
    compose_final_label,
    compute_has_late_data_from_raw,
    compute_late_features,
    get_time_cols,
    compute_stage2_label,
)


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


def load_model_pipeline(model_path: str):
    _install_legacy_sklearn_pickle_aliases()
    assert_runtime_matches_model(model_path)
    return joblib.load(model_path)


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


def _extract_conc_from_curve_id(curve_id: str) -> float | None:
    if curve_id is None:
        return None
    s = str(curve_id)
    m = __import__("re").search(r"\[(?:\s*Conc\s*=\s*)?([0-9]+(?:\.[0-9]+)?)\s*\]", s, flags=__import__("re").IGNORECASE)
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
    # Matches: [0.1] or [Conc=0.1] etc.
    return __import__("re").search(r"\[(?:\s*Conc\s*=\s*)?([0-9]+(?:\.[0-9]+)?)\s*\]", s, flags=__import__("re").IGNORECASE) is not None



def _attach_curve_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Test Id" not in out.columns:
        return out
    out["Test Id"] = out["Test Id"].astype(str)

    # If Test Id already encodes concentration (e.g. "...[0.1]"), NEVER append "||conc"
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


def _split_wide_raw_by_time(
    wide_raw_df: pd.DataFrame,
    *,
    stage2_start: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_cols = [c for c in wide_raw_df.columns if parse_time_from_header(str(c)) is not None]
    non_time_cols = [c for c in wide_raw_df.columns if c not in time_cols]
    early_cols = [c for c in time_cols if float(parse_time_from_header(str(c))) <= float(stage2_start)]
    late_cols = [c for c in time_cols if float(parse_time_from_header(str(c))) > float(stage2_start)]
    wide_early_raw_df = wide_raw_df[non_time_cols + early_cols].copy()
    wide_late_raw_df = wide_raw_df[non_time_cols + late_cols].copy()
    return wide_early_raw_df, wide_late_raw_df


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


def _compute_stage2_features_from_wide(
    wide_raw_df: pd.DataFrame,
    wide_late_df: pd.DataFrame,
    *,
    cfg: Stage2Config,
) -> pd.DataFrame:
    raw_time_cols = get_time_cols(wide_raw_df)
    late_time_cols = get_time_cols(wide_late_df)
    raw_times = np.array([parse_time_from_header(str(c)) for c in raw_time_cols], dtype=float)
    late_header_mask = np.isfinite(raw_times) & (raw_times > float(cfg.stage2_start))
    late_raw_cols = [c for c, keep in zip(raw_time_cols, late_header_mask) if keep]
    rows: list[dict[str, object]] = []
    for idx, row_raw in wide_raw_df.iterrows():
        row_late = wide_late_df.loc[idx]
        tid = str(row_raw.get("Test Id"))
        conc = pd.to_numeric(pd.Series([row_raw.get("Concentration", np.nan)]), errors="coerce").iloc[0]
        has_late_raw, raw_observed_tmax = compute_has_late_data_from_raw(
            row_raw,
            raw_time_cols,
            stage2_start=cfg.stage2_start,
        )
        if late_raw_cols:
            y_late_raw = pd.to_numeric(row_raw[late_raw_cols], errors="coerce").to_numpy(dtype=float)
            n_late_raw = int(np.sum(np.isfinite(y_late_raw)))
        else:
            n_late_raw = 0
        f = compute_late_features(row_late, late_time_cols, cfg) if has_late_raw else {}
        late_tmax = raw_observed_tmax if has_late_raw else np.nan
        late_n_points = n_late_raw if has_late_raw else 0
        late_too_sparse = bool(has_late_raw and late_n_points < int(cfg.late_min_points))
        rows.append(
            {
                "Test Id": tid,
                "Concentration": conc,
                "curve_key": row_raw.get("curve_key", tid),
                "has_late_data": bool(has_late_raw),
                "raw_observed_tmax": raw_observed_tmax,
                "late_window_start": float(cfg.stage2_start) if has_late_raw else np.nan,
                "late_tmax": late_tmax,
                "late_n_points": late_n_points,
                "late_slope": f.get("late_slope", np.nan),
                "late_delta": f.get("late_delta", np.nan),
                "late_max_increase": f.get("late_max_increase", np.nan),
                "late_growth_detected": f.get("late_growth_detected", False),
                "plateau_detected": f.get("plateau_detected", False),
                "decline_detected": f.get("decline_detected", False),
                "drift_detected": f.get("drift_detected", False),
                "noise_detected": f.get("noise_detected", False),
                "sigma_noise": f.get("sigma_noise", np.nan),
                "late_linearity_r2": f.get("late_linearity_r2", np.nan),
                "late_too_sparse": bool(f.get("late_too_sparse", False) or late_too_sparse),
                "_artifact_strong": f.get("_artifact_strong", False),
                "_early_weak": f.get("_early_weak", False),
            }
        )
    return pd.DataFrame(rows)


def _assign_final_reason_labels(
    out_df: pd.DataFrame,
    *,
    cfg: Stage2Config,
) -> pd.DataFrame:
    stage2_labels: list[str] = []
    final_labels: list[str] = []
    reasons: list[str] = []

    for _, row in out_df.iterrows():
        s1_label = _normalize_label_text(row.get("pred_label", ""))
        s1_conf = pd.to_numeric(pd.Series([row.get("pred_confidence", np.nan)]), errors="coerce").iloc[0]
        stage2_label, label_reason = compute_stage2_label(
            stage1_label=s1_label,
            stage1_conf=s1_conf,
            late_features=row.to_dict(),
            cfg=cfg,
        )
        stage2_labels.append(stage2_label if stage2_label else np.nan)
        reasons.append(label_reason if label_reason else np.nan)
        final_labels.append(
            compose_final_label(
                stage1_label=s1_label,
                stage1_conf=s1_conf,
                stage2_label=stage2_label,
                stage2_reason=label_reason,
                cfg=cfg,
            )
        )

    out = out_df.copy()
    out["Predicted S1 Label"] = out.get("Predicted S1 Label", out.get("pred_label", "")).astype(str)
    out["Stage 2 Label"] = stage2_labels
    out["Label Reason"] = reasons
    out["Pred Label"] = final_labels
    out["Pred Confidence"] = pd.to_numeric(out.get("pred_confidence"), errors="coerce")
    # Backward compatibility for older consumers.
    out["final_label"] = final_labels
    out["final_reason"] = reasons
    return out


_STAGE2_FEATURE_COLS: list[str] = [
    "has_late_data",
    "raw_observed_tmax",
    "late_window_start",
    "late_tmax",
    "late_n_points",
    "late_slope",
    "late_delta",
    "late_max_increase",
    "late_growth_detected",
    "plateau_detected",
    "decline_detected",
    "drift_detected",
    "noise_detected",
    "sigma_noise",
    "late_linearity_r2",
    "late_too_sparse",
    "_artifact_strong",
    "_early_weak",
]


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
    # Canonical full-horizon raw wide table. Stage-2 MUST read from this dataframe.
    wide_raw_df = _attach_curve_key(wide_df.copy())
    wide_early_raw_df, wide_late_raw_df = _split_wide_raw_by_time(wide_raw_df, stage2_start=float(stage2_start))

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tmp_wide_csv = Path(td) / "wide_input.csv"
        # Stage-1 preprocessing/modeling still runs with tmax settings (typically 16h).
        # This does NOT affect Stage-2, which reads directly from wide_raw_df.
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
        avg_valid = np.nanmean(valid_probs, axis=0)
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
    out_df["S1 Confidence Invalid"] = out_df["confidence_invalid"]
    # Clarify that this tmax comes from Stage-1 early-pass preprocessing.
    if "observed_tmax" in out_df.columns:
        out_df["early_observed_tmax"] = pd.to_numeric(out_df["observed_tmax"], errors="coerce")
    else:
        out_df["early_observed_tmax"] = np.nan

    # Stage-2 uses post-16h evidence from original wide data (never from truncated stage-1 tables).
    stage2_cfg = Stage2Config(stage2_start=float(stage2_start))
    stage2_df = _compute_stage2_features_from_wide(wide_raw_df, wide_late_raw_df, cfg=stage2_cfg)
    # Remove early-pass late-feature columns from meta_df to avoid _x/_y suffix collisions.
    out_df = out_df.drop(columns=[c for c in _STAGE2_FEATURE_COLS if c in out_df.columns], errors="ignore")
    out_df = out_df.merge(
        stage2_df.drop(columns=["Test Id", "Concentration"], errors="ignore"),
        on=["curve_key"],
        how="left",
    )
    out_df["late_min_points"] = int(stage2_cfg.late_min_points)

    out_df = _assign_final_reason_labels(
        out_df,
        cfg=stage2_cfg,
    )
    out_df["true_label"] = out_df["Pred Label"].astype(str)
    out_df["is_valid_true"] = out_df["true_label"].map(_label_is_valid).astype(bool)
    out_df["Reviewed"] = False
    # Backward compatibility with existing consumers.
    out_df["is_valid_final"] = out_df["is_valid_true"].astype(bool)

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
    pred_map_df = out_df[["curve_key", "is_valid_true", "pred_label", "final_label", "pred_confidence"]].drop_duplicates("curve_key")
    grofit_tidy_all = grofit_tidy_all.merge(pred_map_df, on="curve_key", how="left")
    grofit_tidy_all["is_valid_true"] = grofit_tidy_all["is_valid_true"].fillna(False).astype(bool)

    return {
        "raw_merged_df": raw_merged_df,
        "final_merged_df": final_merged_df,
        "meta_df": meta_df,
        "out_df": out_df,
        "grofit_tidy_all": grofit_tidy_all,
        "stage2_config": stage2_cfg.to_dict(),
    }
