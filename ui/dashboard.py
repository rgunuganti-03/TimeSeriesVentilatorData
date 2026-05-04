"""
ui/dashboard.py
---------------
Streamlit dashboard for the Ventilator Waveform Simulator.

Phase 3 changes:
    - Engines: VCV and PCV only (rule-based and ODE models removed)
    - Mode-specific sidebar parameters:
        VCV: tidal volume slider, flow pattern radio button
        PCV: inspiratory pressure slider, rise time slider
        Both: respiratory rate, compliance, resistance, PEEP, I:E selectbox
    - Tidal volume hidden in PCV (dependent variable)
    - Inspiratory pressure hidden in VCV (not a VCV setting)
    - I:E ratio as selectbox with labeled clinical options (1:1, 1:2, 1:3)
    - Updated metric strip:
        VCV: PPeak, Pplat, Driving P, Mean Paw, Peak Flow up,
             Peak Flow down, Minute Vent, Auto-PEEP
        PCV: PPeak, Delivered VT, Driving P, Mean Paw, Peak Flow up,
             Fill Fraction, Minute Vent, Auto-PEEP
    - Duration removed from metric strip (no clinical value)
    - Plateau ~P hidden in PCV (equals PPeak by definition)

Aesthetic direction: Clinical dark — precision instrument.

Run from project root:
    streamlit run app.py
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.conditions    import get_condition, get_condition_meta, list_conditions
from generator.vcv_generator import generate_breath_cycles as _gen_vcv
from generator.pcv_generator import generate_breath_cycles as _gen_pcv


# ---------------------------------------------------------------------------
# Engine registry — VCV and PCV only
# ---------------------------------------------------------------------------

ENGINES = {
    "VCV": {
        "key":   "vcv",
        "fn":    _gen_vcv,
        "label": "VCV · Volume-Controlled",
        "icon":  "▣",
    },
    "PCV": {
        "key":   "pcv",
        "fn":    _gen_pcv,
        "label": "PCV · Pressure-Controlled",
        "icon":  "◈",
    },
}

ENGINE_NAMES = list(ENGINES.keys())

# I:E ratio selectbox options — label → float value
IE_OPTIONS = {
    "1:1  (ie = 1.0)": 1.0,
    "1:2  (ie = 0.5)": 0.5,
    "1:3  (ie = 0.33)": 0.33,
}


# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------

COLOR_BG       = "#0a0e14"
COLOR_PANEL    = "#111720"
COLOR_BORDER   = "#1e2a38"
COLOR_TEXT     = "#c9d6e3"
COLOR_MUTED    = "#4a5a6a"

COLOR_PRESSURE = "#38bdf8"   # sky blue
COLOR_FLOW     = "#fbbf24"   # amber
COLOR_VOLUME   = "#34d399"   # emerald
COLOR_ACCENT   = "#a78bfa"   # violet — engine badge, PCV accents

SIGNAL_COLORS = {
    "pressure": COLOR_PRESSURE,
    "flow":     COLOR_FLOW,
    "volume":   COLOR_VOLUME,
}
SIGNAL_UNITS = {
    "pressure": "cmH\u2082O",
    "flow":     "L/s",
    "volume":   "mL",
}
SIGNAL_LABELS = {
    "pressure": "Pressure",
    "flow":     "Flow",
    "volume":   "Volume",
}


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

def configure_page():
    st.set_page_config(
        page_title="Ventilator Waveform Simulator",
        page_icon="\U0001fac1",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def inject_css():
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

      .stApp, [data-testid="stAppViewContainer"] {{
          background-color: {COLOR_BG};
          color: {COLOR_TEXT};
      }}
      [data-testid="stSidebar"] {{
          background-color: {COLOR_PANEL};
          border-right: 1px solid {COLOR_BORDER};
      }}
      [data-testid="stSidebar"] * {{
          color: {COLOR_TEXT} !important;
          font-family: 'JetBrains Mono', monospace !important;
      }}
      .stSlider [data-baseweb="slider"] div[role="slider"] {{
          background-color: {COLOR_PRESSURE} !important;
          border-color: {COLOR_PRESSURE} !important;
      }}
      .stSelectbox label, .stSlider label, .stRadio label {{
          font-family: 'JetBrains Mono', monospace !important;
          font-size: 0.75rem !important;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: {COLOR_MUTED} !important;
      }}
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
          font-size: 1.35rem !important;
          font-weight: 600;
          color: {COLOR_TEXT} !important;
      }}
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
      .badge {{
          display: inline-block;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.65rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          padding: 3px 10px;
          border-radius: 2px;
          background-color: {COLOR_BORDER};
          color: {COLOR_PRESSURE};
          border: 1px solid {COLOR_PRESSURE}44;
          margin-top: 6px;
          margin-right: 6px;
      }}
      .badge-engine {{
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
          margin-right: 6px;
      }}
      .condition-desc {{
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.68rem;
          color: {COLOR_MUTED};
          line-height: 1.6;
          padding: 8px 10px;
          border-left: 2px solid {COLOR_PRESSURE}66;
          background-color: {COLOR_PRESSURE}08;
          border-radius: 0 3px 3px 0;
          margin-top: 6px;
      }}
      .section-label {{
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.65rem;
          color: {COLOR_MUTED};
          text-transform: uppercase;
          letter-spacing: 0.12em;
          margin: 10px 0 4px 0;
      }}
      .stDownloadButton button {{
          font-family: 'JetBrains Mono', monospace !important;
          font-size: 0.72rem !important;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          background-color: transparent !important;
          border: 1px solid {COLOR_PRESSURE}88 !important;
          color: {COLOR_PRESSURE} !important;
          border-radius: 3px !important;
      }}
      .stDownloadButton button:hover {{
          background-color: {COLOR_PRESSURE}15 !important;
          border-color: {COLOR_PRESSURE} !important;
      }}
      hr {{ border-color: {COLOR_BORDER}; margin: 8px 0; }}
      #MainMenu, footer, header {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _ie_default_index(ie_value: float) -> int:
    """Return the IE_OPTIONS index whose value is closest to ie_value."""
    values = list(IE_OPTIONS.values())
    return min(range(len(values)), key=lambda i: abs(values[i] - ie_value))


def render_sidebar():
    """
    Render sidebar controls.
    Returns (params dict, condition_name, engine_name, n_cycles).
    """
    with st.sidebar:
        st.markdown(
            '<div class="dash-sub" style="margin-bottom:12px;">'
            '— Signal Parameters —</div>',
            unsafe_allow_html=True,
        )

        # --- Engine selector --------------------------------------------
        engine_name = st.selectbox(
            "Simulation Engine",
            options=ENGINE_NAMES,
            index=0,
            help=(
                "VCV: ventilator prescribes flow — pressure is derived. "
                "PCV: ventilator prescribes pressure — volume is derived."
            ),
        )
        engine_key = ENGINES[engine_name]["key"]

        st.markdown("<hr>", unsafe_allow_html=True)

        # --- Condition selector -----------------------------------------
        condition_name = st.selectbox(
            "Respiratory Condition",
            options=list_conditions(),
            index=0,
        )
        meta = get_condition_meta(condition_name)
        st.markdown(
            f'<div class="condition-desc">{meta["description"]}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-label">Parameters</div>',
            unsafe_allow_html=True,
        )

        # Load condition preset — drives all slider default values.
        # The key= argument on each widget includes condition_name and
        # engine_name so Streamlit reinitialises the slider from value=
        # whenever the condition or engine changes. Manual adjustments
        # within a session are preserved until a new condition is selected.
        preset = get_condition(condition_name)

        # --- Shared parameters ------------------------------------------
        rr = st.slider(
            "Respiratory Rate (bpm)", 5, 40,
            value=int(preset["respiratory_rate"]),
            step=1,
            key=f"rr_{condition_name}_{engine_name}",
        )
        compliance = st.slider(
            "Compliance (mL/cmH\u2082O)", 5, 150,
            value=int(preset["compliance_mL_per_cmH2O"]),
            step=1,
            key=f"compliance_{condition_name}_{engine_name}",
        )
        resistance = st.slider(
            "Resistance (cmH\u2082O/L/s)", 1, 50,
            value=int(preset["resistance_cmH2O_L_s"]),
            step=1,
            key=f"resistance_{condition_name}_{engine_name}",
        )

        # I:E ratio — selectbox with clinical labels, default from preset
        ie_label = st.selectbox(
            "I:E Ratio",
            options=list(IE_OPTIONS.keys()),
            index=_ie_default_index(preset["ie_ratio"]),
            help="Inspiratory to expiratory time ratio.",
            key=f"ie_{condition_name}_{engine_name}",
        )
        ie = IE_OPTIONS[ie_label]

        peep = st.slider(
            "PEEP (cmH\u2082O)", 0, 20,
            value=int(preset["peep_cmH2O"]),
            step=1,
            key=f"peep_{condition_name}_{engine_name}",
        )

        # --- VCV-specific parameters ------------------------------------
        if engine_key == "vcv":
            st.markdown(
                '<div class="section-label" style="margin-top:10px;">'
                'VCV Settings</div>',
                unsafe_allow_html=True,
            )
            tv = st.slider(
                "Tidal Volume (mL)", 100, 900,
                value=int(preset["tidal_volume_mL"]),
                step=10,
                help="Target volume delivered per breath.",
                key=f"tv_{condition_name}_{engine_name}",
            )
            flow_pattern = st.radio(
                "Flow Pattern",
                options=["decelerating", "square"],
                index=0,
                help=(
                    "Decelerating: higher initial flow, tapers through "
                    "inspiration — most common clinical default. "
                    "Square: constant flow — easier to read compliance "
                    "and resistance from the pressure waveform."
                ),
                key=f"flow_{condition_name}_{engine_name}",
            )

        # --- PCV-specific parameters ------------------------------------
        if engine_key == "pcv":
            st.markdown(
                '<div class="section-label" style="margin-top:10px;">'
                'PCV Settings</div>',
                unsafe_allow_html=True,
            )
            insp_pressure = st.slider(
                "Inspiratory Pressure (cmH\u2082O above PEEP)",
                min_value=1, max_value=35,
                value=15, step=1,
                help=(
                    "Driving pressure above PEEP applied during inspiration. "
                    "Delivered tidal volume depends on this setting plus "
                    "patient compliance and resistance."
                ),
                key=f"insp_p_{condition_name}_{engine_name}",
            )
            rise_time = st.slider(
                "Rise Time (s)",
                min_value=0.0, max_value=0.4,
                value=0.0, step=0.1,
                help=(
                    "Time for pressure to ramp from PEEP to PIP. "
                    "0.0 = square wave step (maximum initial flow). "
                    "Longer rise times reduce peak flow and improve "
                    "patient comfort in spontaneously breathing patients."
                ),
                key=f"rise_{condition_name}_{engine_name}",
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        n_cycles = st.slider("Breath Cycles", 1, 20, 5, step=1)

        # --- Assemble params dict ---------------------------------------
        params = {
            "respiratory_rate":        rr,
            "compliance_mL_per_cmH2O": compliance,
            "resistance_cmH2O_L_s":    resistance,
            "ie_ratio":                ie,
            "peep_cmH2O":              peep,
        }

        if engine_key == "vcv":
            params["tidal_volume_mL"] = tv
            params["flow_pattern"]    = flow_pattern
        else:
            # tidal_volume_mL required by validator — actual volume
            # is derived from insp_pressure inside the PCV generator
            params["tidal_volume_mL"]     = 500
            params["insp_pressure_cmH2O"] = insp_pressure
            params["rise_time_s"]         = rise_time

        return params, condition_name, engine_name, n_cycles


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def render_header(condition_name, engine_name):
    engine_label = ENGINES[engine_name]["label"]
    engine_icon  = ENGINES[engine_name]["icon"]
    st.markdown(
        f'<div class="dash-header">Ventilator Waveform Simulator</div>'
        f'<div class="dash-sub">Aiden Medical \u00b7 '
        f'Time Series Ventilator Data</div>'
        f'<span class="badge">\u25b6 {condition_name}</span>'
        f'<span class="badge-engine">{engine_icon} {engine_label}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='margin-top:14px;'>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Metric strip
# ---------------------------------------------------------------------------

def _metric_card(col, label, value, unit):
    """Render one metric as a custom HTML card — no truncation."""
    col.markdown(
        f"""
        <div style="
            background-color: {COLOR_PANEL};
            border: 1px solid {COLOR_BORDER};
            border-radius: 4px;
            padding: 12px 16px;
        ">
            <div style="
                font-family: \'JetBrains Mono\', monospace;
                font-size: 0.7rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: {COLOR_ACCENT};
                margin-bottom: 4px;
            ">{label}</div>
            <div style="
                font-family: \'JetBrains Mono\', monospace;
                font-size: 1.4rem;
                font-weight: 600;
                color: {COLOR_TEXT};
                line-height: 1;
            ">{value} <span style="
                font-size: 0.8rem;
                font-weight: 400;
                color: {COLOR_MUTED};
            ">{unit}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(result, params, engine_key):
    """
    VCV strip (8 metrics):
        PPeak | Pplat | Driving P | Mean Paw |
        Peak Flow up | Peak Flow down | Minute Vent | Auto-PEEP

    PCV strip (8 metrics):
        PPeak | Delivered VT | Driving P | Mean Paw |
        Peak Flow up | Fill Fraction | Minute Vent | Auto-PEEP

    Pplat hidden in PCV — equals PPeak by definition.
    Tidal volume shown as delivered VT in PCV (dependent variable).
    Duration removed — no clinical diagnostic value.
    Uses custom HTML cards to prevent value truncation.
    """
    peep      = params["peep_cmH2O"]
    peak_p    = float(result["pressure"].max())
    peak_f    = float(result["flow"].max())
    min_f     = float(result["flow"].min())
    peak_v    = float(result["volume"].max())
    mean_paw  = float(np.mean(result["pressure"]))
    auto_peep = max(0.0, float(result["pressure"][-1]) - peep)
    rr        = params["respiratory_rate"]
    minute_vent = rr * peak_v / 1000.0

    cols = st.columns(8)

    if engine_key == "vcv":
        pplat     = float(np.percentile(result["pressure"], 90))
        driving_p = pplat - peep

        metrics = [
            ("Peak Pressure", f"{peak_p:.1f}",    "cmH₂O"),
            ("Plateau ~P",    f"{pplat:.1f}",     "cmH₂O"),
            ("Driving P",     f"{driving_p:.1f}", "cmH₂O"),
            ("Mean Paw",      f"{mean_paw:.1f}",  "cmH₂O"),
            ("Peak Flow ↑", f"{peak_f:.2f}", "L/s"),
            ("Peak Flow ↓", f"{min_f:.2f}",  "L/s"),
            ("Minute Vent",   f"{minute_vent:.1f}", "L/min"),
            ("Auto-PEEP",     f"{auto_peep:.2f}", "cmH₂O"),
        ]

    else:
        insp_p    = params.get("insp_pressure_cmH2O", 0)
        driving_p = float(insp_p)
        pip       = peep + insp_p
        pip_mask  = result["pressure"] >= (pip - 0.5)
        fill_frac = float(np.clip(
            np.sum(pip_mask) / len(result["pressure"]), 0.0, 1.0
        )) if pip > peep else 0.0

        metrics = [
            ("Peak Pressure",  f"{peak_p:.1f}",    "cmH₂O"),
            ("Delivered VT",   f"{peak_v:.0f}",    "mL"),
            ("Driving P",      f"{driving_p:.1f}", "cmH₂O"),
            ("Mean Paw",       f"{mean_paw:.1f}",  "cmH₂O"),
            ("Peak Flow ↑", f"{peak_f:.2f}",  "L/s"),
            ("Fill Fraction",  f"{fill_frac:.2f}", ""),
            ("Minute Vent",    f"{minute_vent:.1f}", "L/min"),
            ("Auto-PEEP",      f"{auto_peep:.2f}", "cmH₂O"),
        ]

    for col, (label, value, unit) in zip(cols, metrics):
        _metric_card(col, label, value, unit)


# ---------------------------------------------------------------------------
# Waveform plot
# ---------------------------------------------------------------------------

def render_waveform_plot(result, condition_name):
    time = result["time"]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
    )

    for i, sig in enumerate(["pressure", "flow", "volume"], start=1):
        col   = SIGNAL_COLORS[sig]
        unit  = SIGNAL_UNITS[sig]
        label = SIGNAL_LABELS[sig]

        fig.add_trace(
            go.Scatter(
                x=time, y=result[sig],
                mode="lines",
                name=f"{label} ({unit})",
                line=dict(color=col, width=1.8),
                fill="tozeroy",
                fillcolor=f"rgba({_hex_to_rgb(col)}, 0.07)",
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"t=%{{x:.3f}}s<br>%{{y:.2f}} {unit}<extra></extra>"
                ),
            ),
            row=i, col=1,
        )

        fig.add_hline(
            y=0,
            line=dict(color=COLOR_BORDER, width=1, dash="dot"),
            row=i, col=1,
        )
        fig.update_yaxes(
            title_text=(
                f"{label}<br>"
                f"<span style='font-size:9px'>{unit}</span>"
            ),
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
    for r in [1, 2]:
        fig.update_xaxes(showticklabels=False, row=r, col=1)

    fig.update_layout(
        height=640,
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_PANEL,
        font=dict(family="JetBrains Mono", color=COLOR_TEXT),
        margin=dict(l=10, r=20, t=20, b=10),
        showlegend=False,
        hovermode="x unified",
    )

    row_y_positions = [0.99, 0.64, 0.30]
    for idx, sig in enumerate(["pressure", "flow", "volume"]):
        sig_col = SIGNAL_COLORS[sig]
        fig.add_annotation(
            text=(
                f"<span style='color:{sig_col};"
                f"font-family:JetBrains Mono;"
                f"font-size:11px;text-transform:uppercase;"
                f"letter-spacing:0.1em'>\u25cf {SIGNAL_LABELS[sig]}</span>"
            ),
            xref="paper", yref="paper",
            x=0.01, y=row_y_positions[idx],
            xanchor="left", yanchor="top",
            showarrow=False, font=dict(size=11),
        )

    st.plotly_chart(
        fig, use_container_width=True, config={"displayModeBar": False}
    )


# ---------------------------------------------------------------------------
# Export panel
# ---------------------------------------------------------------------------

def render_export(result, params, condition_name, engine_name):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="dash-sub" style="margin-bottom:10px;">— Export —</div>',
        unsafe_allow_html=True,
    )

    col_csv, col_json, _ = st.columns([1, 1, 3])
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    engine_key = ENGINES[engine_name]["key"]

    csv_bytes = pd.DataFrame({
        "time_s":         result["time"],
        "pressure_cmH2O": result["pressure"],
        "flow_Ls":        result["flow"],
        "volume_mL":      result["volume"],
    }).to_csv(index=False).encode("utf-8")

    with col_csv:
        st.download_button(
            label="\u2193 Download CSV",
            data=csv_bytes,
            file_name=(
                f"ventilator_{condition_name.lower()}_"
                f"{engine_key}_{ts}.csv"
            ),
            mime="text/csv",
        )

    # Build JSON scenario — include all params cleanly
    scenario = {
        "condition":    condition_name,
        "engine":       engine_key,
        "generated_at": datetime.now().isoformat(),
    }
    scenario.update({
        k: v for k, v in params.items()
        if k != "tidal_volume_mL" or engine_key == "vcv"
    })
    json_bytes = json.dumps(scenario, indent=2).encode("utf-8")

    with col_json:
        st.download_button(
            label="\u2193 Download JSON",
            data=json_bytes,
            file_name=(
                f"scenario_{condition_name.lower()}_"
                f"{engine_key}_{ts}.json"
            ),
            mime="application/json",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)}, {int(h[2:4],16)}, {int(h[4:6],16)}"


def _run_engine(engine_name, params, n_cycles):
    """Dispatch to the correct generator with mode-specific param defaults."""
    return ENGINES[engine_name]["fn"](params, n_cycles=n_cycles)


# ---------------------------------------------------------------------------
# Main render — called by app.py
# ---------------------------------------------------------------------------

def render():
    configure_page()
    inject_css()

    params, condition_name, engine_name, n_cycles = render_sidebar()
    engine_key = ENGINES[engine_name]["key"]

    render_header(condition_name, engine_name)

    with st.spinner("Generating waveforms..."):
        result = _run_engine(engine_name, params, n_cycles)

    render_metrics(result, params, engine_key)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    render_waveform_plot(result, condition_name)
    render_export(result, params, condition_name, engine_name)