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
    out_dir: Path,
    zip_name: str = "grofit_outputs.zip",
    cleanup_csv: bool = True,
) -> Dict[str, Any]:
    """
    Write only the required CSVs and return a ZIP containing them.
    Files written:
      - gcFit.csv
      - drFit.csv
      - gcBoot.csv (optional)
      - drBoot.csv (optional)
      - <zip_name>
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = []
    gc_fit_path = out_dir / "gcFit.csv"
    dr_fit_path = out_dir / "drFit.csv"

    gc_fit.to_csv(gc_fit_path, index=False)
    dr_fit.to_csv(dr_fit_path, index=False)
    csv_paths.extend([gc_fit_path, dr_fit_path])

    if gc_boot is not None and not gc_boot.empty:
        gc_boot_path = out_dir / "gcBoot.csv"
        gc_boot.to_csv(gc_boot_path, index=False)
        csv_paths.append(gc_boot_path)

    if dr_boot is not None and not dr_boot.empty:
        dr_boot_path = out_dir / "drBoot.csv"
        dr_boot.to_csv(dr_boot_path, index=False)
        csv_paths.append(dr_boot_path)

    zip_path = out_dir / zip_name
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in csv_paths:
            zf.write(p, arcname=p.name)
    zip_bytes = bio.getvalue()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in csv_paths:
            zf.write(p, arcname=p.name)

    if cleanup_csv:
        for p in csv_paths:
            try:
                p.unlink()
            except Exception:
                pass

    return {"zip_bytes": zip_bytes, "zip_path": zip_path}
