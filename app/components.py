# app/ui/components.py
"""
Reusable Streamlit UI primitives: pills, error display, metric rows.
"""
from __future__ import annotations
import streamlit as st


def st_pills_multi(
    label: str,
    options: list[str],
    default: list[str] | None = None,
    key: str | None = None,
) -> list[str]:
    """Backward-compatible multi-select pills (st.pills or st.multiselect)."""
    default_vals = [str(v) for v in (default or []) if str(v) in {str(o) for o in options}]
    if hasattr(st, "pills"):
        sel = st.pills(label, options=options, default=default_vals,
                       selection_mode="multi", key=key)
        return [] if sel is None else [str(v) for v in sel]
    sel = st.multiselect(label, options=options, default=default_vals, key=key)
    return [str(v) for v in sel]


def show_friendly_error(exc: Exception) -> None:
    st.markdown(
        """<div style="padding:12px;border-radius:8px;border:2px solid #c00;
        background:#ffecec;color:#600;font-size:17px;font-weight:700;">
        Run failed. Please write what you were doing and send the issue details/logs
        to <a href="mailto:theprerna@uni-koblenz.de">theprerna@uni-koblenz.de</a>
        for improvement and feedback.</div>""",
        unsafe_allow_html=True,
    )
    st.caption(f"Error: {type(exc).__name__}: {exc}")


def line_with_tooltip(label: str, value: object, tooltip: str) -> None:
    safe_val = "" if value is None else value
    icon = (f'<sup style="margin-left:6px;color:#888;" title="{tooltip}">🛈</sup>'
            if tooltip else "")
    st.markdown(
        f'<div title="{tooltip}"><strong>{label}:</strong> {safe_val}{icon}</div>',
        unsafe_allow_html=True,
    )


def render_metric_row(label: str, value: str | None = None,
                      font: str = "1.1rem") -> None:
    """Two-column label / value row used in the right metrics panel."""
    r_label, r_value = st.columns([1.2, 1.8], gap="small")
    r_label.markdown(
        f"<div style='font-size:{font};font-weight:600;'>{label}</div>",
        unsafe_allow_html=True,
    )
    r_value.markdown(
        f"<div style='font-size:{font};'>{'NA' if value is None else value}</div>",
        unsafe_allow_html=True,
    )


def render_select_row(label: str, options_list: list[str],
                      index_val: int, key_val: str,
                      font: str = "1.1rem") -> str:
    """Two-column label / selectbox row."""
    r_label, r_value, _ = st.columns([1.2, 0.78, 1.02], gap="small")
    r_label.markdown(
        f"<div style='font-size:{font};font-weight:600;'>{label}</div>",
        unsafe_allow_html=True,
    )
    return r_value.selectbox(label, options=options_list, index=index_val,
                             key=key_val, label_visibility="collapsed")
