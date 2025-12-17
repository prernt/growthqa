# GrowthQC Streamlit App

Preprocess bacterial growth curves, extract meta-features, and classify validity (single model or ensemble).

## Quick start
1. Create a virtual env and install deps:
   ```
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```
2. Run the app:
   ```
   streamlit run app/streamlit_app.py
   ```
   Open http://localhost:8501.

## Using the app
- Model selection: default "Average" (ensemble). Or pick HGB/RF/LR explicitly.
- Blank subtraction: check "Input is raw" if your file is not blank-subtracted; optionally set Global blank.
- Upload: one CSV/XLSX. Sample files available in the app under "Download sample input files":
  - sample_wide.csv: `Test Id` + time columns `T#.0 (h)`.
  - sample_long.csv: `Time (h)` + wells as columns.
- Run validation: click the button. Results are cached so you can interact without reruns.
- Predictions table: click a row to sync the interactive curve viewer.
- Curve viewer: dropdown of Test Ids (sorted), with Model/Label/Confidence/p_valid and QC flags.
- Downloads:
  - Main CSV/XLSX: original non-time columns (excluding FileName/Model Name/Is_Valid), Predicted Label, timepoints, PredictingModel.
  - Debug: meta_features_debug.csv with selected meta features + PredictingModel.
- Errors: friendly banner; if parsing fails, match the sample formats and email details to the listed contact.

## Defaults (preprocess/meta)
- Smoothing: SGF (window=5)
- Normalization: MAX
- clip_negatives: False
- step=0.25, min_points=3, low_res_threshold=7, auto_tmax=True

## Deployment
- Streamlit Cloud: push to GitHub, set entry `streamlit run app/streamlit_app.py`.
- Other hosts: `streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0` behind a reverse proxy; manage with systemd/pm2/supervisor.
- Keep models small or host externally; store secrets in `.streamlit/secrets.toml` (never in git).
