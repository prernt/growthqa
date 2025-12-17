from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn


def write_model_manifest(model_path: str | Path) -> Path:
    """
    Write <model>.manifest.json next to <model>.joblib with runtime versions.
    This must be called with a MODEL FILE path, not a directory.
    """
    mp = Path(model_path)
    if mp.suffix.lower() != ".joblib":
        raise ValueError(f"write_model_manifest expects a .joblib file, got: {mp}")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
        "model_file": mp.name,
    }
    out = mp.with_suffix(".manifest.json")
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out
