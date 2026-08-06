# app/model_io.py
"""
Classifier model helpers for the Streamlit layer. No Streamlit dependency.

The discovery and label-mapping logic, plus model loading, prediction and the
ensemble, live once in the pipeline layer (growthqa.pipelines.infer_labels).
This module re-exports the two helpers the UI needs and adds only the small
``has_trained_models`` convenience check.
"""
from __future__ import annotations

from pathlib import Path

from config import MODEL_DIR

# Single source of truth for these helpers is the pipeline layer.
from growthqa.pipelines.infer_labels import (  # noqa: F401
    discover_models,
    label_from_stem,
)
from growthqa.pipelines.auto_train_classifier import (  # noqa: F401
    train_classifier_from_meta_file,
)

__all__ = [
    "has_trained_models",
    "discover_models",
    "label_from_stem",
    "train_classifier_from_meta_file",
]


def has_trained_models(model_dir: Path = MODEL_DIR) -> bool:
    p = Path(model_dir)
    return p.exists() and any(p.glob("*.joblib"))