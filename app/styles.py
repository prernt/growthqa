# app/ui/styles.py
"""
Injects the application-wide CSS.  Call ``inject_css()`` once at page top.
"""
from __future__ import annotations
import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Source+Code+Pro:wght@400;600&display=swap');
:root {
    --ink: #2b1f1a; --muted: #7a6a5f; --panel: #fffaf4; --line: #d9c8b8;
    --blue: #c26d3a; --blue-dark: #9a4f27; --bg1: #f7efe5; --bg2: #eadfce;
}
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, var(--bg1), var(--bg2));
    font-family: 'Space Grotesk', system-ui, sans-serif;
    color: var(--ink);
}
@media (prefers-color-scheme: dark) {
    :root {
        --ink: #ffffff; --muted: #d2c4b6; --panel: #2a1f1a; --line: #4a3a32;
        --blue: #d07a45; --blue-dark: #b06133; --bg1: #1a1411; --bg2: #2a201a;
    }
}
.metrics-panel { height:520px; overflow-y:auto; margin-top:-542px; padding-top:0; }
.metrics-header { margin-bottom:-5px; margin-top:-15px; font-size:1.7rem; font-weight:700; }
.metrics-subheader { margin-top:0; margin-bottom:4px; opacity:.95; font-size:1.5rem; font-weight:500; }
.curve-title { font-weight:600; margin:0 0 6px 0; }
.ui-row-title { font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); margin-bottom:6px; }
.ui-inline-note { font-size:12px; color:var(--muted); }
.ui-pill { display:inline-block; padding:4px 10px; border-radius:999px; background:#f1e2d3; color:#6e3b1c; font-size:12px; font-weight:600; }
.stButton > button { background:var(--blue); color:#fff; border-radius:8px; border:2px solid var(--blue-dark); padding:.65rem 1.1rem; font-weight:600; box-shadow:0 2px 0 rgba(0,0,0,.15); }
.stButton > button:disabled { opacity:.55; }
.stDownloadButton > button { border-radius:8px; border:2px solid var(--blue-dark); }
.stDataFrame, .stDataEditor { background:var(--panel); }

/* Review actions */
.review-actions .stButton > button,
.review-actions .stDownloadButton > button { height:44px; min-height:44px; padding:0 12px; width:190px; }
.review-actions [data-testid="stCheckbox"] label {
    background:var(--blue) !important; color:#fff !important;
    border:2px solid var(--blue-dark) !important; border-radius:8px !important;
    padding:8px 12px !important; width:190px !important;
    box-shadow:0 2px 0 rgba(0,0,0,.15) !important;
    display:flex !important; align-items:center !important; gap:8px !important;
}
.review-actions [data-testid="stCheckbox"] input { accent-color:#fff !important; }

/* Action buttons */
.action-buttons .stDownloadButton > button {
    background:var(--blue) !important; color:#fff !important;
    border:2px solid var(--blue-dark) !important; box-shadow:0 2px 0 rgba(0,0,0,.15) !important;
    border-radius:8px !important; height:44px !important;
}
.action-buttons [data-testid="stFileUploader"] button {
    background:var(--blue) !important; color:#fff !important;
    border:2px solid var(--blue-dark) !important; border-radius:8px !important;
    height:44px !important; width:100% !important; font-weight:600 !important;
    box-shadow:0 2px 0 rgba(0,0,0,.15) !important;
}
.action-buttons [data-testid="stFileUploader"] section,
.action-buttons [data-testid="stFileUploaderDropzone"] {
    padding:0 !important; border:none !important; background:transparent !important;
}
.action-buttons [data-testid="stFileUploaderDropzoneInstructions"],
.action-buttons [data-testid="stFileUploader"] svg,
.action-buttons [data-testid="stFileUploader"] small,
.action-buttons [data-testid="stFileUploader"] ul { display:none !important; }
.action-buttons [data-testid="stFileUploader"] button span { display:none !important; }
.action-buttons [data-testid="stFileUploader"] button::before { content:"#7B Upload True Labels"; }

/* Target manual review upload by id */
div[data-testid="stFileUploader"]:has(input[id="manual_review_upload"]) section,
div[data-testid="stFileUploader"]:has(input[id="manual_review_upload"]) [data-testid="stFileUploaderDropzone"] {
    padding:0 !important; border:none !important; background:transparent !important;
}
div[data-testid="stFileUploader"]:has(input[id="manual_review_upload"]) [data-testid="stFileUploaderDropzoneInstructions"],
div[data-testid="stFileUploader"]:has(input[id="manual_review_upload"]) svg,
div[data-testid="stFileUploader"]:has(input[id="manual_review_upload"]) small,
div[data-testid="stFileUploader"]:has(input[id="manual_review_upload"]) ul { display:none !important; }
div[data-testid="stFileUploader"]:has(input[id="manual_review_upload"]) button {
    background:var(--blue) !important; color:#fff !important;
    border:2px solid var(--blue-dark) !important; border-radius:8px !important;
    height:44px !important; width:100% !important; font-weight:600 !important;
    box-shadow:0 2px 0 rgba(0,0,0,.15) !important;
}
div[data-testid="stFileUploader"]:has(input[id="manual_review_upload"]) button span { display:none !important; }
div[data-testid="stFileUploader"]:has(input[id="manual_review_upload"]) button::before { content:"#7B Upload True Labels"; }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
