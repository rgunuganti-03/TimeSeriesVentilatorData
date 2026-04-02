"""
ui/dashboard.py
---------------
Streamlit dashboard for the Ventilator Waveform Simulator.

Aesthetic direction: Clinical dark — precision instrument.
Feels like an ICU monitor. Sharp signal colors, monospaced readouts,
tight layout. Not a generic data app.

Run from project root:
    streamlit run app.py
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Adjust path so this module works when called from app.py at project root
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.conditions import get_condition, get_condition_meta, list_conditions
from generator.waveforms import generate_breath_cycles as _generate_phase1
from generator.ode_solver import generate_breath_cycles as _generate_phase2

_GENERATORS = {
    "Phase 1 — Rule-Based":           _generate_phase1,
    "Phase 2 — ODE Single-Compartment": _generate_phase2,
}


# ---------------------------------------------------------------------------
# Theme constants — one place to change colors
# ---------------------------------------------------------------------------

COLOR_BG        = "#0a0e14"       # near-black background
COLOR_PANEL     = "#111720"       # slightly lighter panel
COLOR_BORDER    = "#1e2a38"       # subtle border
COLOR_TEXT      = "#c9d6e3"       # soft white text
COLOR_MUTED     = "#4a5a6a"       # muted labels

COLOR_PRESSURE  = "#38bdf8"       # sky blue  — Pressure
COLOR_FLOW      = "#fbbf24"       # amber     — Flow
COLOR_VOLUME    = "#34d399"       # emerald   — Volume
COLOR_ACCENT    = "#38bdf8"       # matches pressure

SIGNAL_COLORS = {
    "pressure": COLOR_PRESSURE,
    "flow":     COLOR_FLOW,
    "volume":   COLOR_VOLUME,
}

SIGNAL_UNITS = {
    "pressure": "cmH₂O",
    "flow":     "L/s",
    "volume":   "mL",
}

SIGNAL_LABELS = {
    "pressure": "Pressure",
    "flow":     "Flow",
    "volume":   "Volume",
}


# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------

def configure_page():
    st.set_page_config(
        page_title="Ventilator Waveform Simulator",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ---------------------------------------------------------------------------
# CSS injection — clinical dark theme
# ---------------------------------------------------------------------------

def inject_css():
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

      /* Global background */
      .stApp, [data-testid="stAppViewContainer"] {{
          background-color: {COLOR_BG};
          color: {COLOR_TEXT};
      }}

      /* Sidebar */
      [data-testid="stSidebar"] {{
          background-color: {COLOR_PANEL};
          border-right: 1px solid {COLOR_BORDER};
      }}
      [data-testid="stSidebar"] * {{
          color: {COLOR_TEXT} !important;
          font-family: 'JetBrains Mono', monospace !important;
      }}

      /* Sidebar sliders accent */
      .stSlider [data-baseweb="slider"] div[role="slider"] {{
          background-color: {COLOR_ACCENT} !important;
          border-color: {COLOR_ACCENT} !important;
      }}
      .stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"] {{
          color: {COLOR_ACCENT} !important;
      }}

      /* Select box */
      .stSelectbox label, .stSlider label {{
          font-family: 'JetBrains Mono', monospace !important;
          font-size: 0.75rem !important;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: {COLOR_MUTED} !important;
      }}

      /* Metric cards */
      [data-testid="metric-container"] {{
          background-color: {COLOR_PANEL};
          border: 1px solid {COLOR_BORDER};
          border-radius: 4px;
          padding: 12px 16px;
      }}
      [data-testid="metric-container"] label {{
          font-family: 'JetBrains Mono', monospace !important;
          font-size: 0.7rem !important;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          color: {COLOR_MUTED} !important;
      }}
      [data-testid="metric-container"] [data-testid="stMetricValue"] {{
          font-family: 'JetBrains Mono', monospace !important;
          font-size: 1.4rem !important;
          font-weight: 600;
          color: {COLOR_TEXT} !important;
      }}

      /* Header */
      .dash-header {{
          font-family: 'Syne', sans-serif;
          font-size: 1.6rem;
          font-weight: 800;
          color: {COLOR_TEXT};
          letter-spacing: -0.02em;
          line-height: 1.1;
      }}
      .dash-sub {{
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.7rem;
          color: {COLOR_MUTED};
          text-transform: uppercase;
          letter-spacing: 0.12em;
          margin-top: 2px;
      }}
      .condition-badge {{
          display: inline-block;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.65rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          padding: 3px 10px;
          border-radius: 2px;
          background-color: {COLOR_BORDER};
          color: {COLOR_ACCENT};
          border: 1px solid {COLOR_ACCENT}44;
          margin-top: 6px;
      }}
      .signal-tag {{
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.6rem;
          text-transform: uppercase;
          letter-spacing: 0.1em;
      }}

      /* Dividers */
      hr {{ border-color: {COLOR_BORDER}; margin: 8px 0; }}

      /* Download button */
      .stDownloadButton button {{
          font-family: 'JetBrains Mono', monospace !important;
          font-size: 0.72rem !important;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          background-color: transparent !important;
          border: 1px solid {COLOR_ACCENT}88 !important;
          color: {COLOR_ACCENT} !important;
          border-radius: 3px !important;
      }}
      .stDownloadButton button:hover {{
          background-color: {COLOR_ACCENT}15 !important;
          border-color: {COLOR_ACCENT} !important;
      }}

      /* Description box */
      .condition-desc {{
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.68rem;
          color: {COLOR_MUTED};
          line-height: 1.6;
          padding: 8px 10px;
          border-left: 2px solid {COLOR_ACCENT}66;
          background-color: {COLOR_ACCENT}08;
          border-radius: 0 3px 3px 0;
          margin-top: 6px;
      }}

      /* Hide Streamlit branding */
      #MainMenu, footer, header {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — condition selector + parameter sliders
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[dict, str, int, str]:
    """
    Render sidebar controls.
    Returns (params dict, condition_name, n_cycles, model_name).
    """
    with st.sidebar:
        st.markdown(
            '<div class="dash-sub" style="margin-bottom:12px;">— Signal Parameters —</div>',
            unsafe_allow_html=True,
        )

        # Model selector
        model_name = st.radio(
            "Generator Model",
            options=list(_GENERATORS.keys()),
            index=0,
            help=(
                "Phase 1 uses rule-based analytical waveforms. "
                "Phase 2 solves the single-compartment lung ODE with scipy."
            ),
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Condition selector
        conditions = list_conditions()
        condition_name = st.selectbox(
            "Respiratory Condition",
            options=conditions,
            index=0,
        )

        # Show condition description
        meta = get_condition_meta(condition_name)
        st.markdown(
            f'<div class="condition-desc">{meta["description"]}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Load preset as defaults for sliders
        preset = get_condition(condition_name)

        st.markdown(
            '<div class="dash-sub" style="margin-bottom:8px;">Adjust Parameters</div>',
            unsafe_allow_html=True,
        )

        rr = st.slider(
            "Respiratory Rate (bpm)",
            min_value=5, max_value=40,
            value=int(preset["respiratory_rate"]), step=1,
        )
        tv = st.slider(
            "Tidal Volume (mL)",
            min_value=100, max_value=900,
            value=int(preset["tidal_volume_mL"]), step=10,
        )
        compliance = st.slider(
            "Compliance (mL/cmH₂O)",
            min_value=5, max_value=150,
            value=int(preset["compliance_mL_per_cmH2O"]), step=1,
        )
        resistance = st.slider(
            "Resistance (cmH₂O/L/s)",
            min_value=1, max_value=50,
            value=int(preset["resistance_cmH2O_L_s"]), step=1,
        )
        ie = st.slider(
            "I:E Ratio (Insp fraction)",
            min_value=0.20, max_value=1.0,
            value=float(preset["ie_ratio"]), step=0.05,
            help="0.5 = 1:2 ratio (standard). Lower = more time for expiration.",
        )
        peep = st.slider(
            "PEEP (cmH₂O)",
            min_value=0, max_value=20,
            value=int(preset["peep_cmH2O"]), step=1,
        )
        n_cycles = st.slider(
            "Breath Cycles",
            min_value=1, max_value=20,
            value=5, step=1,
        )

        params = {
            "respiratory_rate":          rr,
            "tidal_volume_mL":           tv,
            "compliance_mL_per_cmH2O":   compliance,
            "resistance_cmH2O_L_s":      resistance,
            "ie_ratio":                  ie,
            "peep_cmH2O":                peep,
        }

        return params, condition_name, n_cycles, model_name


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def render_header(condition_name: str, model_name: str):
    phase_label = "Phase 2 — ODE" if "Phase 2" in model_name else "Phase 1 — Rule-Based"
    col_title, col_spacer = st.columns([3, 1])
    with col_title:
        st.markdown(
            f"""
            <div class="dash-header">Ventilator Waveform Simulator</div>
            <div class="dash-sub">Aiden Medical · Time Series Ventilator Data · {phase_label}</div>
            <div class="condition-badge">▶ {condition_name}</div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<hr style='margin-top:14px;'>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Metric strip — peak values
# ---------------------------------------------------------------------------

def render_metrics(result: dict, params: dict):
    peak_p  = result["pressure"].max()
    peak_f  = result["flow"].max()
    peak_v  = result["volume"].max()
    min_f   = result["flow"].min()
    t_total = result["time"].max()
    plateau = np.percentile(result["pressure"], 90)   # proxy for plateau pressure

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Peak Pressure",  f"{peak_p:.1f} cmH₂O")
    c2.metric("Plateau ~P",     f"{plateau:.1f} cmH₂O")
    c3.metric("Peak Flow ↑",    f"{peak_f:.2f} L/s")
    c4.metric("Peak Flow ↓",    f"{min_f:.2f} L/s")
    c5.metric("Tidal Volume",   f"{peak_v:.0f} mL")
    c6.metric("Duration",       f"{t_total:.1f} s")


# ---------------------------------------------------------------------------
# Waveform plot — 3 stacked subplots
# ---------------------------------------------------------------------------

def render_waveform_plot(result: dict, condition_name: str):
    time = result["time"]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
    )

    signals = ["pressure", "flow", "volume"]

    for i, sig in enumerate(signals, start=1):
        y    = result[sig]
        col  = SIGNAL_COLORS[sig]
        unit = SIGNAL_UNITS[sig]
        label= SIGNAL_LABELS[sig]

        # Fill under curve for visual depth
        fill_color = col.replace("#", "") 
        fig.add_trace(
            go.Scatter(
                x=time, y=y,
                mode="lines",
                name=f"{label} ({unit})",
                line=dict(color=col, width=1.8),
                fill="tozeroy",
                fillcolor=f"rgba({_hex_to_rgb(col)}, 0.07)",
                hovertemplate=f"<b>{label}</b><br>t=%{{x:.3f}}s<br>%{{y:.2f}} {unit}<extra></extra>",
            ),
            row=i, col=1,
        )

        # Y-axis zero line
        fig.add_hline(
            y=0,
            line=dict(color=COLOR_BORDER, width=1, dash="dot"),
            row=i, col=1,
        )

        # Y-axis label
        fig.update_yaxes(
            title_text=f"{label}<br><span style='font-size:9px'>{unit}</span>",
            title_font=dict(color=col, size=11, family="JetBrains Mono"),
            tickfont=dict(color=COLOR_MUTED, size=9, family="JetBrains Mono"),
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            showgrid=True,
            row=i, col=1,
        )

    fig.update_xaxes(
        title_text="Time (s)",
        title_font=dict(color=COLOR_MUTED, size=10, family="JetBrains Mono"),
        tickfont=dict(color=COLOR_MUTED, size=9, family="JetBrains Mono"),
        gridcolor=COLOR_BORDER,
        showgrid=True,
        row=3, col=1,
    )
    # Hide x-axis labels on rows 1 and 2
    for r in [1, 2]:
        fig.update_xaxes(showticklabels=False, row=r, col=1)

    fig.update_layout(
        height=620,
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_PANEL,
        font=dict(family="JetBrains Mono", color=COLOR_TEXT),
        margin=dict(l=10, r=20, t=20, b=10),
        showlegend=False,
        hovermode="x unified",
    )

    # Signal label annotations — one per subplot row
    # yref paper coords: row1 top ~0.99, row2 ~0.64, row3 ~0.30
    row_y_positions = [0.99, 0.64, 0.30]
    for idx, sig in enumerate(signals):
        sig_col = SIGNAL_COLORS[sig]
        fig.add_annotation(
            text=(
                f"<span style='color:{sig_col};font-family:JetBrains Mono;"
                f"font-size:11px;text-transform:uppercase;"
                f"letter-spacing:0.1em'>● {SIGNAL_LABELS[sig]}</span>"
            ),
            xref="paper", yref="paper",
            x=0.01, y=row_y_positions[idx],
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=11),
        )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Export panel
# ---------------------------------------------------------------------------

def render_export(result: dict, params: dict, condition_name: str):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="dash-sub" style="margin-bottom:10px;">— Export —</div>',
        unsafe_allow_html=True,
    )

    col_csv, col_json, col_spacer = st.columns([1, 1, 3])

    # --- CSV export ---
    df = pd.DataFrame({
        "time_s":           result["time"],
        "pressure_cmH2O":   result["pressure"],
        "flow_Ls":          result["flow"],
        "volume_mL":        result["volume"],
    })
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"ventilator_{condition_name.lower()}_{ts}.csv"

    with col_csv:
        st.download_button(
            label="↓ Download CSV",
            data=csv_bytes,
            file_name=csv_filename,
            mime="text/csv",
        )

    # --- JSON scenario config ---
    scenario = {
        "condition":             condition_name,
        "generated_at":          datetime.now().isoformat(),
        "n_cycles":              int(result["time"].max() * params["respiratory_rate"] / 60) + 1,
        "respiratory_rate":      params["respiratory_rate"],
        "tidal_volume_mL":       params["tidal_volume_mL"],
        "compliance_mL_per_cmH2O": params["compliance_mL_per_cmH2O"],
        "resistance_cmH2O_L_s":  params["resistance_cmH2O_L_s"],
        "ie_ratio":              params["ie_ratio"],
        "peep_cmH2O":            params["peep_cmH2O"],
    }
    json_bytes = json.dumps(scenario, indent=2).encode("utf-8")
    json_filename = f"scenario_{condition_name.lower()}_{ts}.json"

    with col_json:
        st.download_button(
            label="↓ Download JSON",
            data=json_bytes,
            file_name=json_filename,
            mime="application/json",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> str:
    """Convert #rrggbb to 'r, g, b' string for rgba() CSS."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r}, {g}, {b}"


# ---------------------------------------------------------------------------
# Main render function — called by app.py
# ---------------------------------------------------------------------------

def render():
    configure_page()
    inject_css()

    params, condition_name, n_cycles, model_name = render_sidebar()

    render_header(condition_name, model_name)

    # Generate waveforms using the selected model
    generate = _GENERATORS[model_name]
    with st.spinner("Generating waveforms..."):
        result = generate(params, n_cycles=n_cycles)

    render_metrics(result, params)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    render_waveform_plot(result, condition_name)
    render_export(result, params, condition_name)