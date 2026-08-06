# Generating `training_meta.csv`

Quick-start guide to regenerate the GrowthQA classifier training dataset.

## Synthetic dataset contents

`timeseries_wide_SD1.csv` - 900 curves, grid 0–16 h @ 0.5 h, seed 123, 11 subtypes:

| Valid (503) | n   | Invalid (397) | n   |
| ---         | --- | ---           | --- |
| plain       | 203 | obvious       | 80  |
| fast        | 100 | diauxic       | 60  |
| late        | 100 | subtle        | 60  |
| decline     | 100 | nearreal      | 55  |
|             |     | decline_only  | 55  |
|             |     | noise         | 45  |
|             |     | nogrowth      | 42  |

Valid families: Logistic, Gompertz, BoundedModifiedGompertz, Richards, decline
(growth + death phase). Invalid families: diauxic (two growth phases),
obvious (mid-curve crash), subtle (localized artifact), nearreal (suppressed
plateau), decline_only (decay only), noise (no usable signal), nogrowth
(flat/negative control).

Paired with the lab file: `lab_14.75h_0.25.csv` (92 curves) → 992 base curves total.

## Commands

**1. (Optional) Regenerate the synthetic file:**

```bash
python -m growthqa.cli.main synth --seed 123 --max-time 16 --time-step 0.5 --output-dir data/pipeline_data --file-stem SD1
```

**2. Merge + build the training dataset:**

```bash
python -m growthqa.cli.main build-train-meta --synthetic data/pipeline_data/timeseries_wide_SD1.csv --lab data/pipeline_data/lab_14.75h_0.25.csv --out-dir data/train_data
```

Writes `raw_merged.csv`, `final_merged.csv`, `training_meta.csv` to `data/train_data/`. Drop `--lab` to build synthetic-only (skips `raw_merged.csv`).

**3. Train the classifier:**

Streamlit UI → **Train / Refresh Classifier**, or:

```python
from growthqa.pipelines.auto_train_classifier import train_classifier_from_meta_file
from growthqa.config import STAGE1_SELECTED_FEATURES

train_classifier_from_meta_file(
    meta_csv_path="data/train_data/training_meta.csv",
    models_out_dir="classifier_output/saved_models_selected",
    selected_features=STAGE1_SELECTED_FEATURES,
)
```

**4. Validate a freshly generated dataset:**

```bash
python -m growthqa.pipelines.validate_thesis_pipeline \
    data/pipeline_data/timeseries_wide_SD1.csv \
    data/pipeline_data/lab_14.75h_0.25.csv
```

## What happens in the background (step 2)

1. Merge synthetic + lab, tag `source_type`
2. Interpolate onto common grid → `raw_merged.csv`
3. Truncation augmentation (tail, horizons 8/10/12/14.75/16 h, 3/curve) + gap augmentation (internal gaps/missingness, 30% of curves, 2/curve) - both raw-first, independent
4. Re-interpolate combined augmented data
5. Preprocess: smoothing (SGF, window 5), MIN-MAX normalization
6. Meta-feature extraction → `training_meta.csv`

## Expected output

| Quantity              | Value                                                 |
| ---                   | ---                                                   |
| Base curves           | 992                                                   |
| `raw_merged.csv`      | 992 rows                                              |
| `training_meta.csv`   | 3,572 rows x 38 cols (2,153 Valid/1,419 Invalid)      |
| Source split          | 3,236 synthetic/336 lab(all 336 lab rows are Valid)   |
| Stage 1 features      | `STAGE1_TOP_10_FEATURES` (10 features, `config.py`)   |
