# growthqa/src/growthqa/classifier/train_from_meta.py
from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
import sklearn


# -----------------------------
# Paths (relative to repo root)
# -----------------------------
ROOT = Path(__file__).resolve().parents[3]
TRAIN_META_CSV = ROOT / "data" / "train_data" / "meta.csv"
ART_DIR = ROOT / "classifier_output" / "saved_models_selected"
LOCKFILE_OUT = ROOT / "classifier_output" / "requirements_lock.txt"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# -----------------------------
# Label handling
# -----------------------------
def normalize_label(series: pd.Series) -> pd.Series:
    """
    Converts label column into 0/1.
    Accepts: True/False, 1/0, "Valid"/"Invalid", "valid"/"invalid"
    """
    s = series.copy()

    if s.dtype == bool:
        return s.astype(int)

    # numeric
    if pd.api.types.is_numeric_dtype(s):
        # assume already 0/1
        return s.fillna(0).astype(int)

    # string-like
    s2 = s.astype(str).str.strip().str.lower()
    mapping = {
        "true": 1,
        "1": 1,
        "valid": 1,
        "yes": 1,
        "y": 1,
        "false": 0,
        "0": 0,
        "invalid": 0,
        "no": 0,
        "n": 0,
    }
    out = s2.map(mapping)
    if out.isna().any():
        # fallback: try to coerce
        out = pd.to_numeric(s2, errors="coerce")
    return out.fillna(0).astype(int)


def detect_label_col(df: pd.DataFrame) -> str:
    candidates = ["Is_Valid", "is_valid", "label", "y", "_y"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find label column. Tried: {candidates}. Columns: {list(df.columns)[:30]}...")


# -----------------------------
# Feature cleaning / engineering
# (matches PDF logic)
# -----------------------------
DROP_COLS = [
    "FileName",
    "Test Id",
    "Model Name",
    "meta_label",          # leaks procedural labeling
    "too_sparse",
    "low_resolution",
    "had_outliers",
    "best_model_name",
    "max_OD",              # often constant
]

HEAVY_TAILED = [
    "initial_OD",
    "dip_fraction",
    "largest_drop_frac",
    "logistic_fit_mse",
    "noise_residual_std",
]

# Your final selected features from PDF
FINAL_FEATURES = [
    "auc",
    "max_slope",
    "growth_phase_duration",
    "dip_fraction",
    "symmetry_factor",
    "noise_residual_std",
    "best_model_AIC",
    "multi_phase_flag",
]


def build_model_matrix(meta: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    df_model = meta.copy()

    # Replace infinities
    df_model = df_model.replace([np.inf, -np.inf], np.nan)

    y = normalize_label(df_model[label_col])
    X = df_model.drop(columns=[label_col], errors="ignore")

    # Drop known non-features / leakage columns
    X = X.drop(columns=[c for c in DROP_COLS if c in X.columns], errors="ignore")

    # Ensure numeric coercion where possible
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Conservative log1p features
    for col in HEAVY_TAILED:
        if col in X.columns:
            X[f"log1p_{col}"] = np.log1p(X[col])

    # Simple derived feature
    if "final_OD" in X.columns and "plateau_OD" in X.columns:
        X["final_minus_plateau"] = X["final_OD"] - X["plateau_OD"]

    return X, y


# -----------------------------
# Models (matches PDF)
# -----------------------------
def build_models() -> Dict[str, Pipeline]:
    lr = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    rf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=600,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=-1,
            min_samples_leaf=2,
        )),
    ])

    hgb = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=500,
            random_state=RANDOM_STATE,
        )),
    ])

    return {"LR": lr, "RF": rf, "HGB": hgb}


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def pretty_confusion(model: Pipeline, X: pd.DataFrame, y_true: pd.Series, title: str):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, pred)
    print(title)
    print(cm)
    print("")


def fit_and_eval(models: Dict[str, Pipeline],
                 X_train: pd.DataFrame, y_train: pd.Series,
                 X_val: pd.DataFrame, y_val: pd.Series,
                 X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    rows = []
    fitted: Dict[str, Pipeline] = {}

    # sample weights for HGB only (as in PDF)
    sw = compute_sample_weight(class_weight="balanced", y=y_train)

    for name, model in models.items():
        if name == "HGB":
            model.fit(X_train, y_train, clf__sample_weight=sw)
        else:
            model.fit(X_train, y_train)

        fitted[name] = model

        def eval_split(split_name: str, Xs: pd.DataFrame, ys: pd.Series):
            proba = model.predict_proba(Xs)[:, 1]
            pred = (proba >= 0.5).astype(int)
            m = compute_metrics(ys, pred, proba)
            m.update({"model": name, "split": split_name})
            return m

        rows.append(eval_split("train", X_train, y_train))
        rows.append(eval_split("val", X_val, y_val))
        rows.append(eval_split("test", X_test, y_test))

    return pd.DataFrame(rows), fitted


def write_model_manifest(model_path: Path):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
        "model_file": model_path.name,
    }
    model_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_requirements_lock(out_path: str):
    """
    Writes a pip-freeze lockfile using the *current python interpreter*.
    This is what makes 'train once, load forever (in pinned env)' work.
    """
    import subprocess, sys
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        subprocess.check_call([sys.executable, "-m", "pip", "freeze"], stdout=f)


def main():
    print("Loading meta:", TRAIN_META_CSV)
    meta = pd.read_csv(TRAIN_META_CSV)

    label_col = detect_label_col(meta)
    print("Label column:", label_col)

    X, y = build_model_matrix(meta, label_col=label_col)

    # Feature set: Selected (as per PDF)
    X_sel = X[[c for c in FINAL_FEATURES if c in X.columns]].copy()
    missing = [c for c in FINAL_FEATURES if c not in X.columns]
    if missing:
        print("WARNING: missing selected features:", missing)

    print("Selected feature count:", X_sel.shape[1])
    print("All feature count:", X.shape[1])

    # Split: train/val/test = 60/20/20 (like PDF: test=20%, then val=25% of trainval)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_sel, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=RANDOM_STATE
    )

    print(f"Train size: {len(y_train)}  Val size: {len(y_val)}  Test size: {len(y_test)}")
    print("Train class balance:\n", y_train.value_counts(normalize=True))

    models = build_models()

    results, fitted = fit_and_eval(models, X_train, y_train, X_val, y_val, X_test, y_test)
    print("\n=== Results (Selected features) ===")
    print(results.sort_values(["split", "model"]).to_string(index=False))

    for name, model in fitted.items():
        pretty_confusion(model, X_test, y_test, title=f"[Selected] {name} Test Confusion Matrix")

    # Save artifacts
    art_dir = Path(ART_DIR)
    art_dir.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Saving run:", run_tag)

    # Save each trained pipeline
    for name, model in fitted.items():
        out_path = art_dir / f"{name}_selected_pipeline_{run_tag}.joblib"
        joblib.dump(model, out_path)
        print("Saved:", out_path)

    # Save selected features list (critical for inference)
    feat_path = art_dir / f"selected_features_{run_tag}.json"
    feat_path.write_text(json.dumps(FINAL_FEATURES, indent=2), encoding="utf-8")
    print("Saved:", feat_path)

    # Thresholds are in the PDF; keep them as metadata (even if Streamlit won’t use them)
    thresholds = {"valid_th": 0.70, "invalid_th": 0.30, "proba_positive_class": "valid(1)"}
    th_path = art_dir / f"thresholds_{run_tag}.json"
    th_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    print("Saved:", th_path)

    # Save a run summary CSV
    summary_path = art_dir / f"train_results_selected_{run_tag}.csv"
    results.to_csv(summary_path, index=False)
    print("Saved:", summary_path)

    print("\nDONE.")


if __name__ == "__main__":
    main()
