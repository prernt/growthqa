# src/growthqa/grofit/export.py
from __future__ import annotations
import io
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

def export_results_zip(
    *,
    gc_fit: pd.DataFrame,
    dr_fit: pd.DataFrame,
    gc_boot: Optional[pd.DataFrame],
    dr_boot: Optional[pd.DataFrame],
    dr_audit: Optional[pd.DataFrame] = None,      # item 8: AIC audit
    out_dir: Path,
    zip_name: str = "grofit_outputs.zip",
    cleanup_csv: bool = True,
) -> Dict[str, Any]:
    """Write Grofit-compatible CSV outputs + optional audit tables."""
    import datetime
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = []

    def _write(df: pd.DataFrame, name: str) -> Path:
        p = out_dir / name
        df.to_csv(p, index=False)
        csv_paths.append(p)
        return p

    _write(gc_fit, "gcFit.csv")
    _write(dr_fit, "drFit.csv")
    if gc_boot is not None and not gc_boot.empty:
        _write(gc_boot, "gcBoot.csv")
    if dr_boot is not None and not dr_boot.empty:
        _write(dr_boot, "drBoot.csv")
    # item 8: AIC/model-selection audit table
    if dr_audit is not None and not dr_audit.empty:
        _write(dr_audit, "drAudit.csv")

    # item 12: manifest with version + timestamp
    from .pipeline import PIPELINE_VERSION, SCHEMA_VERSION
    manifest = pd.DataFrame([{
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.datetime.utcnow().isoformat(),
        "gc_curves": len(gc_fit),
        "dr_experiments": len(dr_fit),
    }])
    _write(manifest, "manifest.csv")

    # Build ZIP
    zip_path = out_dir / zip_name
    bio = io.BytesIO()
    for fobj in (bio, zip_path):
        with zipfile.ZipFile(fobj if fobj is bio else str(fobj), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in csv_paths:
                zf.write(p, arcname=p.name)

    if cleanup_csv:
        for p in csv_paths:
            try:
                p.unlink()
            except Exception:
                pass

    return {"zip_bytes": bio.getvalue(), "zip_path": zip_path}