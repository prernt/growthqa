# GrowthQA

GrowthQA classifies the quality of bacterial growth curves and fits growth
and dose-response parameters on the curves it accepts. It is a Python
reimplementation and extension of the R Grofit package (Kahm et al., 2010),
built as a Streamlit application.

The package is `growthqa`. This repository's distribution name is `growthqa`.

## What it does

An uploaded batch of growth curves goes through two independent classifiers
before anything is fitted:

1. **Stage 1** - a machine-learning classifier (logistic regression, random
   forest and histogram gradient boosting, combined into a
   validation-accuracy-weighted ensemble) scores each curve's early window
   (up to 16 h) and predicts Valid, Invalid, or Unsure.
2. **Stage 2** - a late-window evidence checker. It reads only the raw data
   after 16 h, without normalizing, smoothing, or interpolating and never
   asserts Valid or Invalid on its own. It can only corroborate Stage 1's
   call or flag a contradiction, which routes the curve to Unsure.

A set of deterministic overrides (out-of-distribution gap size, missing-data
fraction, feature completeness) can also force a curve to Unsure, independent
of what either classifier concluded.

Curves that pass both stages go to the **Grofit pipeline**: parametric model
fitting, spline fitting with bootstrap confidence intervals and
dose-response fitting across concentration series.

## Input formats

The app accepts exactly two file layouts, both keyed on a `Test Id` column:

- **Wide**: one row per curve, with time points as `T<hours> (h)` columns.
- **Long**: one row per curve per timepoint, with a `Time (h)` column.

Both are standardized internally to a canonical wide table before
processing. Concentration is read from a dedicated column if present, or
parsed from the Test Id (a bracketed `[Conc=...]` pattern).

## Installation

```bash
git clone https://github.com/prernt/growthqa.git
cd growthqa
pip install -e .
```

Requires Python >=3.11,<3.12. Dependencies (numpy, pandas, scipy,
scikit-learn, statsmodels, streamlit and others) are pinned in
`pyproject.toml`.

## Running the app

```bash
streamlit run app/streamlit_app.py
```

The sidebar loads trained models from `classifier_output/saved_models_selected`.
If no trained models are present, use **Train / Refresh Classifier** in the
app, or train from the command line — see
[`src/growthqa/synthetic/README_training_meta_generation.md`](src/growthqa/synthetic/README_training_meta_generation.md)
for how the training dataset and models are built.

## Output

Two downloadable archives per run:

- **Results zip** - for the biologist: `gcFit.csv`, `gcBoot.csv`, `drFit.csv`,
  and per-curve and dose-response plots.
- **Auditing zip** - for verification: `run_info.json` (full run
  configuration, including all Stage 2 thresholds and the exact selected
  feature list), `classifier_audit.csv` (per-curve labels, confidences,
  Stage 2 evidence and the full processed time series),
  `grofit_input.csv`, `gc_audit.csv`, `dr_audit.csv` and
  `classifier_performance.csv`.

## Project structure

```
app/                    Streamlit UI layer
src/growthqa/
  config.py              Pinned constants shared by training and inference
  io/                    File parsing and format conversion
  preprocess/            Interpolation, smoothing, normalization, augmentation
  features/              Meta-feature extraction
  classifier/            Stage 1 training
  stage2/                Late-window evidence checker
  grofit/                Parametric and spline fitting, dose-response
  pipelines/             End-to-end training and inference entry points
  synthetic/             Synthetic curve generator
  cli/                   Command-line entry points
data/
  pipeline_data/         Synthetic and lab source CSVs
  train_data/            raw_merged.csv, final_merged.csv, training_meta.csv
  test_data/             Sample files for manual testing
classifier_output/
  saved_models_selected/ Trained model artifacts (.joblib) and manifests
```

## Background

This is the practical implementation behind a Master's thesis extending the
Grofit growth-curve fitting methodology (Kahm et al., 2010) with a
quality-classification stage.