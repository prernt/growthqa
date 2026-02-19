from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib


# --- Grofit pieces (from earlier) ---
from growthqa.grofit.adapters import build_tidy_for_grofit, predictions_to_curve_map
from growthqa.grofit.pipeline import run_grofit_pipeline


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True)
class FullGrofitPipelineConfig:
    # classifier / validity
    valid_threshold: float = 0.5           # if only p_valid exists
    prefer_label: bool = True              # if Predicted Label exists, use it
    pred_delim: str = "_"                  # "testSample1_BY4741" split

    # grofit params
    response_var: str = "mu"               # "A" | "mu" | "lag" | "integral"
    have_atleast: int = 6
    gc_boot_B: int = 200
    dr_boot_B: int = 300
    random_state: int = 42

    # spline behavior in grofit stage
    spline_auto_cv: bool = True
    spline_s: Optional[float] = None
    dr_x_transform: Optional[str] = "log1p"
    dr_fit_method: str = "auto"


# ============================================================
# I/O helpers
# ============================================================

def _read_any(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _write_any(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


# ============================================================
# Classifier stage
# ============================================================

def _try_build_meta_with_growthqa(input_path: Union[str, Path], tmp_dir: Path) -> Path:
    """
    Best-effort: tries to build a meta-features file using existing growthqa pipeline code.
    This keeps your training/inference harmonized (same preprocessing + feature extraction).

    It returns a path to a generated meta csv.

    If your internal APIs differ, this function will raise with a clear message
    telling you where to wire in your existing function.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Option A: If you have a callable pipeline function.
    # (Many growthqa repos have something like pipelines/build_meta_dataset.py)
    try:
        from growthqa.pipelines.build_meta_dataset import build_meta_dataset  # type: ignore
        # Expected: build_meta_dataset(input_path=..., out_dir=..., ...)
        meta_path = build_meta_dataset(input_path=str(input_path), out_dir=str(tmp_dir))  # type: ignore
        return Path(meta_path)
    except Exception:
        pass

    # Option B: If your CLI module exposes a python-callable function.
    try:
        from growthqa.cli.merge_meta_cli import run_merge_meta  # type: ignore
        meta_path = run_merge_meta(input_path=str(input_path), out_dir=str(tmp_dir))  # type: ignore
        return Path(meta_path)
    except Exception:
        pass

    # Option C: As a last resort, fail with actionable wiring note.
    raise RuntimeError(
        "Could not find a callable meta-building entrypoint in your current codebase.\n"
        "Wire one of these into _try_build_meta_with_growthqa():\n"
        "  - growthqa.pipelines.build_meta_dataset.build_meta_dataset(...)\n"
        "  - growthqa.cli.merge_meta_cli.run_merge_meta(...)\n"
        "Or expose a function that takes (input_path, out_dir) and returns meta.csv path."
    )


def _predict_from_meta(
    meta_df: pd.DataFrame,
    model_joblib_path: Union[str, Path],
) -> pd.DataFrame:
    """
    Loads a saved joblib pipeline and predicts.
    Expects the model supports predict_proba (preferred) or predict.
    Returns a predictions dataframe with at least:
      Test Id, Predicted Label, Confidence (Valid)
    """
    model = joblib.load(str(model_joblib_path))

    # Determine ID column
    id_col = None
    for c in ["Test Id", "test_id", "TestID", "testid"]:
        if c in meta_df.columns:
            id_col = c
            break
    if id_col is None:
        raise ValueError("meta_df must contain a curve identifier column like 'Test Id'.")

    # Features = all non-id non-label columns
    drop_cols = {id_col}
    for c in ["label", "Label", "y", "is_valid", "Is_Valid", "Predicted Label"]:
        if c in meta_df.columns:
            drop_cols.add(c)

    X = meta_df.drop(columns=[c for c in drop_cols if c in meta_df.columns], errors="ignore")

    # predict
    p_valid = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # assume binary, valid class could be 1 or "Valid". Try to infer.
        if proba.ndim == 2 and proba.shape[1] == 2:
            # heuristic: take column for class 1 if classes_ exists and has "Valid"/1
            if hasattr(model, "classes_"):
                classes = list(getattr(model, "classes_"))
                if "Valid" in classes:
                    idx = classes.index("Valid")
                elif 1 in classes:
                    idx = classes.index(1)
                else:
                    idx = 1
            else:
                idx = 1
            p_valid = proba[:, idx].astype(float)
    # fallback
    if p_valid is None:
        pred = model.predict(X)
        # map to 0/1-ish
        p_valid = np.array([1.0 if str(v).lower() == "valid" or v == 1 else 0.0 for v in pred], dtype=float)

    pred_label = np.where(p_valid >= 0.5, "Valid", "Invalid")

    out = pd.DataFrame(
        {
            "Test Id": meta_df[id_col].astype(str),
            "Predicted Label": pred_label,
            "Confidence (Valid)": p_valid,
            "Confidence (Invalid)": 1.0 - p_valid,
        }
    )
    return out


# ============================================================
# Full end-to-end runner
# ============================================================

def run_full_input_to_grofit(
    input_path: Union[str, Path],
    outdir: Union[str, Path],
    *,
    model_joblib_path: Union[str, Path],
    file_test_id: Optional[str] = None,
    cfg: FullGrofitPipelineConfig = FullGrofitPipelineConfig(),
    output_mode: str = "r_style",
) -> Dict[str, Path]:
    """
    Runs:
      input -> meta (growthqa) -> classifier -> predictions -> tidy -> grofit -> outputs

    Returns dict of output paths.
    """
    input_path = Path(input_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # infer file_test_id (needed to match predictions like "testSample1_*")
    if file_test_id is None:
        file_test_id = input_path.stem.replace("__", "")

    # --------------------
    # 1) Build meta features using existing growthqa pipeline (best-effort)
    # --------------------
    tmp_dir = outdir / "_tmp_meta"
    try:
        meta_path = _try_build_meta_with_growthqa(input_path, tmp_dir)
        meta_df = pd.read_csv(meta_path) if meta_path.suffix.lower() != ".xlsx" else pd.read_excel(meta_path)
    except Exception as e:
        # If you already have meta/debug outputs, you can pass them externally later.
        raise RuntimeError(
            f"Meta building failed.\nReason: {e}\n"
            "If you want a temporary workaround, export a meta_features_debug.csv via your existing pipeline and "
            "feed it into classification directly."
        )

    # --------------------
    # 2) Classifier predictions (joblib model)
    # --------------------
    pred_df = _predict_from_meta(meta_df, model_joblib_path=model_joblib_path)
    pred_path = None
    if output_mode != "r_style":
        pred_path = outdir / "predictions.csv"
        _write_any(pred_df, pred_path)

    # --------------------
    # 3) Build tidy curves from raw input + attach validity using predictions
    # --------------------
    raw_df = _read_any(input_path)

    tidy_all = build_tidy_for_grofit(
        raw_df,
        predictions_df=pred_df,
        file_test_id=file_test_id,
        delim=cfg.pred_delim,
    )

    tidy_all_path = None
    tidy_valid_path = None
    if output_mode != "r_style":
        tidy_all_path = outdir / "tidy_all.csv"
        _write_any(tidy_all, tidy_all_path)

        tidy_valid = tidy_all[tidy_all["is_valid"]].copy()
        tidy_valid_path = outdir / "tidy_valid.csv"
        _write_any(tidy_valid, tidy_valid_path)

    # --------------------
    # 4) Grofit pipeline
    # --------------------
    grofit_res = run_grofit_pipeline(
        curves_df=tidy_all,
        response_var=cfg.response_var,  # uses spline_mu etc
        have_atleast=cfg.have_atleast,
        gc_boot_B=cfg.gc_boot_B,
        dr_boot_B=cfg.dr_boot_B,
        spline_auto_cv=cfg.spline_auto_cv,
        spline_s=cfg.spline_s,
        dr_x_transform=cfg.dr_x_transform,
        dr_fit_method=cfg.dr_fit_method,
        random_state=cfg.random_state,
        export_dir=outdir if output_mode == "r_style" else None,
    )

    growth_path = None
    dr_path = None
    zip_path = grofit_res.get("zip_path")

    return {
        "predictions": pred_path,
        "tidy_all": tidy_all_path,
        "tidy_valid": tidy_valid_path,
        "growth_curve_results": growth_path,
        "dose_response_results": dr_path,
        "zip_path": zip_path,
    }
