from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from growthqa.classifier.save_manifest import write_model_manifest
from growthqa.config import (
    ROOT,
    TRAIN_META_CSV,
    MODEL_DIR as ART_DIR,
    LOCKFILE_OUT,
    RANDOM_STATE,
    STAGE1_FEATURE_GROUPS,
    STAGE1_CANDIDATE_POOL,
    STAGE1_SELECTED_FEATURES,
    IDENTIFIER_COLS,
    LEAKAGE_COLS,
)
np.random.seed(RANDOM_STATE)


def normalize_label(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float).round().astype("Int64")
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
        out = pd.to_numeric(s2, errors="coerce")
    return out.round().astype("Int64")


def detect_label_col(df: pd.DataFrame) -> str:
    for c in ["Is_Valid", "is_valid", "label", "y", "_y"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find label column in training_meta.csv")


def build_model_matrix(meta: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], pd.DataFrame]:
    df = meta.copy().replace([np.inf, -np.inf], np.nan)
    y_all = normalize_label(df[label_col])
    keep = y_all.notna()
    df = df.loc[keep].copy()
    y = y_all.loc[keep].astype(int)

    groups = (
        df["base_curve_id"].astype(str)
        if "base_curve_id" in df.columns
        else df["Test Id"].astype(str)
    )

    drop_cols = set([label_col]) | IDENTIFIER_COLS | LEAKAGE_COLS
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
        elif X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.select_dtypes(include=[np.number]).copy()

    pool_cols = [c for c in STAGE1_CANDIDATE_POOL if c in X.columns]
    missing_pool_cols = [c for c in STAGE1_CANDIDATE_POOL if c not in X.columns]
    if missing_pool_cols:
        raise ValueError(
            "training_meta.csv is missing expected Stage 1 candidate features: "
            + ", ".join(missing_pool_cols)
        )
    X = X[pool_cols]

    if X.shape[1] == 0:
        raise ValueError("No numeric training features found after dropping identifier/leakage columns.")
    feature_cols = list(X.columns)
    eval_cols = [c for c in ["source_type", "train_horizon", "is_censored", "too_sparse"] if c in df.columns]
    eval_df = df[eval_cols].copy() if eval_cols else pd.DataFrame(index=df.index)
    return X, y, groups, feature_cols, eval_df

def _retire_previous_run_artifacts(art_dir: Path) -> List[str]:
    patterns = [
        "*_selected_pipeline_*.joblib",
        "*_selected_pipeline_*.manifest.json",
        "selected_features_*.json",
        "thresholds_*.json",
        "train_results_selected_*.csv",
    ]
    removed = []
    for pattern in patterns:
        for f in art_dir.glob(pattern):
            try:
                f.unlink()
                removed.append(str(f))
            except OSError:
                pass
    return removed


def build_models(*, random_state: int = RANDOM_STATE) -> Dict[str, Pipeline]:
    lr = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=random_state)),
        ]
    )
    rf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("clf", RandomForestClassifier(
                n_estimators=600,
                random_state=random_state,
                class_weight="balanced_subsample",
                n_jobs=-1,
            )),
        ]
    )
    hgb = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("clf", HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.08,
                random_state=random_state,
            )),
        ]
    )
    return {"LR": lr, "RF": rf, "HGB": hgb}


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
        )
        out = {
            "balanced_acc": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
        out["pr_auc"] = average_precision_score(y_true, y_proba)
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    return out


def _slice_metrics(df_eval: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray, *, model: str, split: str) -> List[dict]:
    rows = []
    base = compute_metrics(y_true, y_pred, y_proba)
    base.update({"model": model, "split": split, "slice_col": "overall", "slice_val": "all", "n": int(len(y_true))})
    rows.append(base)

    for col in ["source_type", "train_horizon", "is_censored", "too_sparse"]:
        if col not in df_eval.columns:
            continue
        vals = df_eval[col]
        if col == "train_horizon":
            vals = pd.to_numeric(vals, errors="coerce").round(3)
        for v in sorted(vals.dropna().unique().tolist()):
            m = vals == v
            if int(np.sum(m)) < 5:
                continue
            mt = compute_metrics(y_true[m], y_pred[m], y_proba[m])
            mt.update({
                "model": model,
                "split": split,
                "slice_col": col,
                "slice_val": str(v),
                "n": int(np.sum(m)),
            })
            rows.append(mt)
    return rows


def _group_split(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, *, random_state: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(X))
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    trainval_idx, test_idx = next(gss_outer.split(idx, y, groups))

    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state + 1)
    inner_groups = groups.iloc[trainval_idx]
    train_rel, val_rel = next(gss_inner.split(trainval_idx, y.iloc[trainval_idx], inner_groups))
    train_idx = trainval_idx[train_rel]
    val_idx = trainval_idx[val_rel]
    return train_idx, val_idx, test_idx


def fit_and_eval(
    models: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    eval_df: pd.DataFrame,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    rows = []
    fitted: Dict[str, Pipeline] = {}

    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_val, y_val = X.iloc[idx_val], y.iloc[idx_val]
    X_test, y_test = X.iloc[idx_test], y.iloc[idx_test]

    sw = compute_sample_weight(class_weight="balanced", y=y_train)
    for name, model in models.items():
        if name == "HGB":
            model.fit(X_train, y_train, clf__sample_weight=sw)
        else:
            model.fit(X_train, y_train)
        fitted[name] = model

        for split, Xi, yi, ei in [
            ("train", X_train, y_train, eval_df.iloc[idx_train]),
            ("val", X_val, y_val, eval_df.iloc[idx_val]),
            ("test", X_test, y_test, eval_df.iloc[idx_test]),
        ]:
            proba = model.predict_proba(Xi)[:, 1]
            pred = (proba >= 0.5).astype(int)
            rows.extend(_slice_metrics(ei, yi, pred, proba, model=name, split=split))
    return pd.DataFrame(rows), fitted


def write_requirements_lock(out_path: str):
    import subprocess
    import sys

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        subprocess.check_call([sys.executable, "-m", "pip", "freeze"], stdout=f)


def main():
    out = train_from_meta_csv(
        meta_csv=TRAIN_META_CSV, art_dir=ART_DIR, selected_features=STAGE1_SELECTED_FEATURES,
    )

    print("Training complete:", json.dumps(out, indent=2))


def train_from_meta_csv(
    *,
    meta_csv: str | Path = TRAIN_META_CSV,
    art_dir: str | Path = ART_DIR,
    run_tag: str | None = None,
    write_lockfile: bool = True,
    selected_features: List[str] | None = None,
    retire_previous_runs: bool = True,
    random_state: int = RANDOM_STATE,
) -> dict:
    meta_csv = Path(meta_csv)
    art_dir = Path(art_dir)
    art_dir.mkdir(parents=True, exist_ok=True)

    retired_files: List[str] = []
    if retire_previous_runs:
        retired_files = _retire_previous_run_artifacts(art_dir)

    meta = pd.read_csv(meta_csv)
    label_col = detect_label_col(meta)
    X, y, groups, feature_cols, eval_df = build_model_matrix(meta, label_col=label_col)

    if selected_features:
        missing = [c for c in selected_features if c not in X.columns]
        if missing:
            raise ValueError(
                "Selected training features are missing from training_meta.csv: "
                + ", ".join(missing)
            )
        X = X[selected_features].copy()
        feature_cols = list(selected_features)

    train_idx, val_idx, test_idx = _group_split(X, y, groups, random_state=random_state)

    overlap = set(groups.iloc[train_idx]).intersection(set(groups.iloc[test_idx]))
    if overlap:
        raise RuntimeError("Group split leakage detected: base_curve_id appears in both train and test.")

    models = build_models(random_state=random_state)
    results, fitted = fit_and_eval(models, X, y, eval_df, train_idx, val_idx, test_idx)

    if run_tag is None:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    val_overall = results[(results["split"] == "val") & (results["slice_col"] == "overall")]
    val_balanced_acc_by_model = {
        row["model"]: float(row["balanced_acc"]) for _, row in val_overall.iterrows()
    }

    model_paths = {}
    manifest_paths = {}
    for name, model in fitted.items():
        out_path = art_dir / f"{name}_selected_pipeline_{run_tag}.joblib"
        joblib.dump(model, out_path)
        model_paths[name] = str(out_path)
        manifest_paths[name] = str(
            write_model_manifest(
                out_path,
                extra={
                    "feature_columns": feature_cols,
                    "group_split_col": "base_curve_id" if "base_curve_id" in meta.columns else "Test Id",
                    "val_balanced_accuracy": val_balanced_acc_by_model.get(name, None),
                },
            )
        )

    feat_path = art_dir / f"selected_features_{run_tag}.json"
    feat_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    # Stable feature file for inference/troubleshooting and external consumers.
    (art_dir / "stage1_features.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    thresholds = {"valid_th": 0.70, "invalid_th": 0.30, "proba_positive_class": "valid(1)"}
    th_path = art_dir / f"thresholds_{run_tag}.json"
    th_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    summary_path = art_dir / f"train_results_selected_{run_tag}.csv"
    results.to_csv(summary_path, index=False)

    if write_lockfile:
        write_requirements_lock(str(LOCKFILE_OUT))

    return {
        "run_tag": run_tag,
        "meta_csv": str(meta_csv),
        "label_col": label_col,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "group_split_col": "base_curve_id" if "base_curve_id" in meta.columns else "Test Id",
        "model_paths": model_paths,
        "manifest_paths": manifest_paths,
        "selected_features_path": str(feat_path),
        "thresholds_path": str(th_path),
        "results_path": str(summary_path),
        "lockfile_path": str(LOCKFILE_OUT) if write_lockfile else None,
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "retired_previous_run_files": retired_files,

    }

def evaluate_split_stability(
    *,
    meta_csv: str | Path = TRAIN_META_CSV,
    seeds: List[int] = [42, 43, 44, 45, 46],
    selected_features: List[str] | None = None,
    tmp_root: str | Path = "/tmp/growthqa_split_stability",
) -> pd.DataFrame:
    tmp_root = Path(tmp_root)
    rows = []
    for i, seed in enumerate(seeds):
        art_dir = tmp_root / f"seed_{seed}_{i}"
        train_from_meta_csv(
            meta_csv=meta_csv,
            art_dir=art_dir,
            run_tag=f"stability_{seed}_{i}",
            write_lockfile=False,
            selected_features=selected_features,
            random_state=seed,
        )
        summary = pd.read_csv(art_dir / f"train_results_selected_stability_{seed}_{i}.csv")
        test_overall = summary[(summary["split"] == "test") & (summary["slice_col"] == "overall")].copy()
        test_overall["seed"] = seed
        rows.append(test_overall)

    all_runs = pd.concat(rows, ignore_index=True)

    summary_rows = []
    for model_name, grp in all_runs.groupby("model"):
        for metric in ["balanced_acc", "precision", "recall", "f1", "roc_auc"]:
            if metric not in grp.columns:
                continue
            summary_rows.append({
                "model": model_name,
                "metric": metric,
                "mean": float(grp[metric].mean()),
                "std": float(grp[metric].std(ddof=1)),
                "min": float(grp[metric].min()),
                "max": float(grp[metric].max()),
                "n_seeds": int(len(grp)),
            })
    stability_summary = pd.DataFrame(summary_rows)

    import shutil
    shutil.rmtree(tmp_root, ignore_errors=True)

    return stability_summary

if __name__ == "__main__":
    main()