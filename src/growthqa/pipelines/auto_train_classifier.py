from __future__ import annotations
import shutil
from pathlib import Path
from growthqa.classifier.train_from_meta import STAGE1_SELECTED_FEATURES, train_from_meta_csv
from growthqa.config import TRAIN_META_CSV as DEFAULT_TRAIN_META, MODEL_DIR as DEFAULT_MODELS_DIR


def train_classifier_from_meta_file(
    *,
    meta_csv_path: str | Path = DEFAULT_TRAIN_META,
    models_out_dir: str | Path = DEFAULT_MODELS_DIR,
    selected_features: list[str] | None = None,
) -> dict:
    models_out_dir = Path(models_out_dir)
    if models_out_dir.exists():
        shutil.rmtree(models_out_dir)
    models_out_dir.mkdir(parents=True, exist_ok=True)

    return train_from_meta_csv(
        meta_csv=meta_csv_path,
        art_dir=models_out_dir,
        selected_features=selected_features if selected_features is not None else STAGE1_SELECTED_FEATURES,
    )