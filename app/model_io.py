# app/pipeline/model_io.py
"""
Classifier model discovery, loading, prediction, and training.
No Streamlit dependency.
"""
from __future__ import annotations

import importlib
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import MODEL_DIR, ROOT, TRAIN_META
from utils import (
    assert_runtime_matches_model,
    install_legacy_sklearn_pickle_aliases,
)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def has_trained_models(model_dir: Path = MODEL_DIR) -> bool:
    p = Path(model_dir)
    return p.exists() and any(p.glob("*.joblib"))


def discover_models(model_dir: Path = MODEL_DIR) -> dict[str, Path]:
    p = Path(model_dir)
    return {} if not p.exists() else {f.stem: f for f in sorted(p.glob("*.joblib"))}


def label_from_stem(stem: str) -> str:
    s = stem.lower()
    if "hgb" in s or "hist" in s: return "HGB"
    if "rf"  in s or "random" in s: return "RF"
    if "lr"  in s or "logreg" in s or "logistic" in s: return "LR"
    return stem


def build_model_label_map(model_dir: Path = MODEL_DIR) -> dict[str, Path]:
    """Return ``{display_label: path}`` for every saved model file."""
    label_map: dict[str, Path] = {}
    for stem, p in discover_models(model_dir).items():
        lbl = label_from_stem(stem)
        if lbl in label_map:
            lbl = f"{lbl}-{stem}"
        label_map[lbl] = p
    return label_map


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model_pipeline(model_path: str):
    install_legacy_sklearn_pickle_aliases()
    assert_runtime_matches_model(model_path)
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_hard_with_confidence(
    pipeline, meta_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run *pipeline* on *meta_df* without thresholds or 'Unsure' labels.

    Returns
    -------
    pred_label_norm : np.ndarray[str]  – "Valid" / "Invalid" / raw string
    confidence      : np.ndarray[float] – max(predict_proba), or NaN
    p_valid         : np.ndarray[float] – P(valid class), or NaN
    """
    non_features = {"FileName", "Test Id"}
    X = meta_df.drop(columns=[c for c in meta_df.columns if c in non_features], errors="ignore")

    expected = getattr(pipeline, "feature_names_in_", None)
    if expected is not None:
        X = X.reindex(columns=[str(c) for c in expected], fill_value=np.nan)

    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="coerce")

    pred_label = pipeline.predict(X)
    conf    = np.full(len(X), np.nan, dtype=float)
    p_valid = np.full(len(X), np.nan, dtype=float)

    if hasattr(pipeline, "predict_proba"):
        proba   = pipeline.predict_proba(X)
        conf    = np.max(proba, axis=1).astype(float)
        classes = getattr(pipeline, "classes_", None)
        if classes is not None and len(classes) == proba.shape[1]:
            cls_list = [str(c).strip().lower() for c in classes]
            for key in ("valid", "true", "1"):
                if key in cls_list:
                    p_valid = proba[:, cls_list.index(key)].astype(float)
                    break

    norm = []
    for v in pred_label:
        s = str(v).strip().lower()
        if s in {"1", "true", "valid"}:     norm.append("Valid")
        elif s in {"0", "false", "invalid"}: norm.append("Invalid")
        else:                                norm.append(str(v).strip())
    return np.array(norm, dtype=object), conf, p_valid


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training() -> None:
    """Train the classifier in-process, clearing old artefacts first."""
    train_mod = importlib.import_module("growthqa.classifier.train_from_meta")
    art_dir   = ROOT / "classifier_output" / "saved_models_selected"
    lockfile  = ROOT / "classifier_output" / "requirements_lock.txt"

    clf_root = art_dir.parent
    if clf_root.exists():
        for child in clf_root.iterdir():
            child.unlink() if child.is_file() else shutil.rmtree(child)
    clf_root.mkdir(parents=True, exist_ok=True)

    train_mod.TRAIN_META_CSV = str(TRAIN_META)
    train_mod.ART_DIR        = str(art_dir)
    train_mod.LOCKFILE_OUT   = str(lockfile)
    train_mod.main()


def train_classifier_from_meta_file(
    *, meta_csv_path, models_out_dir, selected_features=None,
) -> dict:
    """Compatibility wrapper – prefers module-level function, falls back gracefully."""
    import growthqa.pipelines.auto_train_classifier as _mod
    if hasattr(_mod, "train_classifier_from_meta_file"):
        return _mod.train_classifier_from_meta_file(
            meta_csv_path=meta_csv_path,
            models_out_dir=models_out_dir,
            selected_features=selected_features,
        )
    from growthqa.classifier.train_from_meta import train_from_meta_csv
    out_dir = Path(models_out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return train_from_meta_csv(
        meta_csv=meta_csv_path, art_dir=out_dir,
        selected_features=selected_features,
    )
