# Generating `training_meta.csv`

This document describes how the GrowthQA classifier training dataset
(`training_meta.csv`) is produced. It is written for users who want to 
regenerate the dataset from scratch.

The classifier is trained only on `training_meta.csv`. Raw growth curves are
never fed to the classifier directly. Each row of `training_meta.csv` is the
scalar feature vector of one *observation state* of one curve, together with its
ground-truth `Is_Valid` label.

The dataset is generated **once**, from the terminal. Retraining the classifier
reuses the committed `training_meta.csv` and does not rebuild it.

---

## Inputs

Two wide-format CSV files are required, both with the same layout (one row per
curve, time points as `T<hours> (h)` columns, plus `FileName`, `Test Id`,
`Model Name`, `Is_Valid`):

| Input | Role | File in this repo |
| --- | --- | --- |
| Synthetic curves | controlled diversity and failure modes | `timeseries_wide_SD1.csv` (900 curves to 16 h at 0.5 h) |
| Laboratory curves | real biological and measurement variability | a wide lab CSV, e.g. `lab_14.75h_0.25.csv` (92 curves) |

The laboratory file is withheld from the public repository for data protection.
Supply your own wide-format lab CSV in the same column layout. The reproducible
synthetic chain and the scalar-feature `training_meta.csv` are committed.

---

## Pipeline

```
synthetic curves  +  laboratory curves
            |
            v
   merge into one wide table          (explicit source_type tag per input)
            |
            v
   interpolate onto a common grid     0..16 h at 0.5 h, no extrapolation
            |
            v
   truncation augmentation            horizons 8/10/12/14.75/16 h, 3 per curve
            |
            v
   preprocess                         blank handling, Savitzky-Golay smoothing,
            |                          MIN-MAX normalisation
            v
   meta-feature extraction            one scalar feature row per state
            |
            v
   training_meta.csv  (+ raw_merged.csv, final_merged.csv)
```

The implementation is `run_merge_preprocess_meta` and `build_training_meta` in
`src/growthqa/pipelines/build_meta_dataset.py`. All preprocessing settings are
pinned to the `TRAIN_*` constants in that module, and the inference path
(`infer_labels.py`) imports the same constants, so an uploaded curve is
preprocessed identically to the training set.

Pinned configuration:

| Constant | Value |
| --- | --- |
| `TRAIN_STEP_HOURS` | 0.5 |
| `TRAIN_TMAX_HOURS` | 16.0 |
| `TRAIN_TRUNC_HORIZONS` | 8, 10, 12, 14.75, 16 |
| `TRAIN_TRUNC_PER_CURVE` | 3 |
| `TRAIN_TRUNC_SEED` | 123 |
| `TRAIN_SMOOTH_METHOD` | SGF (Savitzky-Golay) |
| `TRAIN_SMOOTH_WINDOW` | 5 |
| `TRAIN_NORMALIZE` | MINMAX |

Truncation runs **before** feature extraction, because a feature such as the
maximum observed OD changes when only part of a curve is seen. Each truncated
state therefore receives its own feature extraction. Grouping by
`base_curve_id` is preserved so that all truncated states of one curve stay in
the same train/validation/test split.

---

## Steps to generate

All commands run from the repository root with the package importable
(`PYTHONPATH=src`, or the installed `growthqa` console script).

### 1. (Optional) Regenerate the synthetic file

The committed `timeseries_wide_SD1.csv` is the canonical synthetic input. With
seed 123 it is exactly reproducible:

```bash
python -m growthqa.cli.main synth --seed 123 --max-time 16 --time-step 0.5 --output-dir data/gr_data --file-stem SD1
```

This writes `data/train_data/timeseries_wide_SD1.csv` (900 curves; 503 valid,
397 invalid; 11 subtypes) and a `run_info.xlsx` provenance file.

### 2. Build the training dataset

```bash
python -m growthqa.cli.main build-train-meta \
    --synthetic data/gr_data/timeseries_wide_SD1.csv \
    --lab       data/train_data/lab_14.75h_0.25.csv \
    --out-dir   data/gr_data
```
This writes three files into `data/gr_data/`:

- `raw_merged.csv` – original curves on the common grid (no augmentation)
- `final_merged.csv` – per-horizon truncated curves after preprocessing
- `training_meta.csv` – one scalar feature row per state, with `Is_Valid`

The `lab` file is optional and without it only 2 files are written into
`data/train_data/`:  `final_merged.csv` and `training_meta.csv`.

The command exposes only the two inputs and the output directory. Every
preprocessing setting comes from the `TRAIN_*` constants, so the dataset cannot
be built with settings that differ from inference.

### 3. Train the classifier

Training reads `training_meta.csv` only; it does not rebuild the dataset.

- From the Streamlit UI: click **Train / Refresh Classifier**.
- Programmatically:

```python
from growthqa.pipelines.auto_train_classifier import train_classifier_from_meta_file
from growthqa.classifier.train_from_meta import STAGE1_SELECTED_FEATURES

train_classifier_from_meta_file(
    meta_csv_path="data/train_data/training_meta.csv",
    models_out_dir="classifier_output/saved_models_selected",
    selected_features=STAGE1_SELECTED_FEATURES,
)
```

The three pipelines (LR, RF, HGB) plus their manifests, the feature list, the
decision thresholds and a per-slice metrics CSV are written to
`classifier_output/saved_models_selected/`.

---

## Expected output

With the committed `timeseries_wide_SD1.csv` (900 synthetic) and
`lab_14.75h_0.25.csv` (92 lab):

| Quantity | Value |
| --- | --- |
| Base curves | 992 |
| Rows in `training_meta.csv` | 2 976 (992 × 3 states) |
| Columns | 49 |
| Label balance | 1 842 Valid / 1 134 Invalid |
| Source split | 2 700 synthetic / 276 lab |

The eight features used by the Stage-1 classifier are `observed_tmax`,
`net_change_per_hour`, `max_slope`, `lag_time_est`, `monotonicity_fraction`,
`largest_drop_frac`, `roughness`, and `final_to_peak_ratio`.

---

## Validation

To confirm a freshly generated dataset is well formed (required columns,
truncation correctness, feature integration, and no `base_curve_id` leakage
across the train/test split):

```bash
python -m growthqa.pipelines.validate_thesis_pipeline \
    data/train_data/timeseries_wide_SD1.csv \
    data/train_data/lab_14.75h_0.25.csv
```

A successful run prints `Validation passed.`
