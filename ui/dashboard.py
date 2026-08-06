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
from generator.psv_generator import generate_breath_cycles as _gen_psv
from generator.prvc_generator import generate_breath_cycles as _gen_prvc
from generator.simv_generator import generate_breath_cycles as _gen_simv  

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
    "PSV": {
        "key":   "psv",
        "fn":    _gen_psv,
        "label": "PSV · Pressure-Support",
        "icon":  "*",
    },
    "PRVC": {
        "key":   "prvc",
        "fn":    _gen_prvc,
        "label": "PRVC · Pressure-Regulated Volume Control",
        "icon":  "◆",   # pick anything distinct from ▣ ◈ *
    },
    "SIMV": {                                                        
        "key":   "simv",                                             
        "fn":    _gen_simv,                                          
        "label": "SIMV · Synchronized Intermittent Mandatory",        
        "icon":  "◐",   # distinct from ▣ ◈ * ◆                      
    },     
}

ENGINE_NAMES = list(ENGINES.keys())

# I:E ratio selectbox options — label → float value
IE_OPTIONS = {
    "1:1  (ie = 1.0)": 1.0,
    "1:2  (ie = 0.5)": 0.5,
    "1:3  (ie = 0.33)": 0.33,
    "1:4  (ie = 0.25)": 0.25,
    "1:5  (ie = 0.20)": 0.20,
}


# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------

COLOR_BG       = "#0a0e14"
COLOR_PANEL    = "#111720"
COLOR_BORDER   = "#1e2a38"
COLOR_TEXT     = "#c9d6e3"
COLOR_MUTED    = "#4a5a6a"

COLOR_PRESSURE = "#34d399"   # emerald
COLOR_FLOW     = "#fbbf24"   # amber
COLOR_VOLUME   = "#38bdf8"   # sky blue
COLOR_ACCENT   = "#a78bfa"   # violet — engine badge, PCV accents

SIGNAL_COLORS = {
    "pressure": COLOR_PRESSURE,
    "flow":     COLOR_FLOW,
    "volume":   COLOR_VOLUME,
}
SIGNAL_UNITS = {
    "pressure": "cmH\u2082O",
    "flow":     "l/s",
    "volume":   "ml",
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

def _pcv_default_driving_pressure(preset: dict) -> int:
    rr  = preset["respiratory_rate"]
    C   = preset["compliance_ml_per_cmH2O"]
    R   = preset["resistance_cmH2O_L_s"]
    ie  = preset["ie_ratio"]
    V_T = preset["tidal_volume_ml"]

    t_cycle = 60.0 / rr
    t_insp  = t_cycle * ie / (1.0 + ie)
    tau     = R * C / 1000.0
    ff      = 1.0 - np.exp(-t_insp / tau)
    delta_P = V_T / (C * ff)

    return int(round(min(delta_P, 35)))  # clamp to slider max


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
                "PSV: ventilator prescribes pressure support after a patient-initiated breath — volume is derived."
                "SIMV: alternates mandatory VC/PC breaths with spontaneous PSV-style breaths, synchronized to patient effort."
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
        is_neonatal = preset.get("population", "adult") == "neonate"

        # --- Shared parameters ------------------------------------------

        if is_neonatal:
            compliance = st.slider(
                "Compliance (ml/cmH\u2082O)", 0.1, 8.0,
                value=float(preset["compliance_ml_per_cmH2O"]),
                step=0.1,
                key=f"compliance_{condition_name}_{engine_name}",
            )
            resistance = st.slider(
                "Resistance (cmH\u2082O/L/s)", 40, 200,
                value=int(preset["resistance_cmH2O_L_s"]),
                step=5,
                key=f"resistance_{condition_name}_{engine_name}",
            )
        else:
            compliance = st.slider(
                "Compliance (ml/cmH\u2082O)", 5, 150,
                value=int(preset["compliance_ml_per_cmH2O"]),
                step=1,
                key=f"compliance_{condition_name}_{engine_name}",
            )
            resistance = st.slider(
                "Resistance (cmH\u2082O/L/s)", 1, 50,
                value=int(preset["resistance_cmH2O_L_s"]),
                step=1,
                key=f"resistance_{condition_name}_{engine_name}",
            )
        peep = st.slider(
            "PEEP (cmH\u2082O)", 0, 20,
            value=int(preset["peep_cmH2O"]),
            step=1,
            key=f"peep_{condition_name}_{engine_name}",
        )

        mode = None                                                              
        if engine_key == "simv":                                                 
            st.markdown(                                                        
                '<div class="section-label" style="margin-top:10px;">'          
                'SIMV Settings</div>',                                          
                unsafe_allow_html=True,                                          
            )                                                                    
            mode = st.radio(                                                     
                "Mandatory Breath Type",                                         
                options=["VC", "PC"],                                            
                index=0,                                                         
                help=(                                                           
                    "VC: mandatory breaths deliver a set tidal volume "          
                    "(like VCV). PC: mandatory breaths deliver a set "           
                    "inspiratory pressure (like PCV). Spontaneous breaths "      
                    "between mandatory breaths are always pressure-"             
                    "supported, regardless of this choice."                      
                ),                                                                
                key=f"mode_{condition_name}_{engine_name}",                      
            )           

# --- RR and I:E — VCV, PCV and PRVC only (PSV uses effort_rate below) -
        if engine_key in ("vcv", "pcv", "prvc", "simv"):
            _rr_default = int(preset["respiratory_rate"])                                  
            if engine_key == "simv":                                                       
                _rr_default = max(4, round(_rr_default * 0.5))    
            rr = st.slider(
                "Respiratory Rate (bpm)", 5, 40,
                value=_rr_default,
                step=1,
                key=f"rr_{condition_name}_{engine_name}",
            )
            ie_label = st.selectbox(
                "I:E Ratio",
                options=list(IE_OPTIONS.keys()),
                index=_ie_default_index(preset["ie_ratio"]),
                help="Inspiratory to expiratory time ratio.",
                key=f"ie_{condition_name}_{engine_name}",
            )
            ie = IE_OPTIONS[ie_label]

        # --- VCV-specific: flow pattern only ------------------------------
        if engine_key == "vcv" or (engine_key == "simv" and mode == "VC"):
            st.markdown(
                '<div class="section-label" style="margin-top:10px;">'
                'VCV Settings</div>',
                unsafe_allow_html=True,
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

        # --- Tidal volume — VCV (its own setting) and PRVC (its target) --
        if engine_key in ("vcv", "prvc") or (engine_key == "simv" and mode == "VC"):
            if is_neonatal:
                tv = st.slider(
                    "Tidal Volume (ml)", 3, 40,
                    value=int(preset["tidal_volume_ml"]),
                    step=1,
                    help="Target volume delivered per breath.",
                    key=f"tv_{condition_name}_{engine_name}",
                )
            else:
                tv = st.slider(
                    "Tidal Volume (ml)", 100, 900,
                    value=int(preset["tidal_volume_ml"]),
                    step=10,
                    help="Target volume delivered per breath.",
                    key=f"tv_{condition_name}_{engine_name}",
                )

        # --- PCV-specific: inspiratory pressure only -----------------------
        if engine_key == "pcv" or (engine_key == "simv" and mode == "PC"):
            st.markdown(
                '<div class="section-label" style="margin-top:10px;">'
                'PCV Settings</div>',
                unsafe_allow_html=True,
            )
            insp_pressure = st.slider(
                "Inspiratory Pressure (cmH\u2082O above PEEP)",
                min_value=1, max_value=35,
                value=_pcv_default_driving_pressure(preset), step=1,
                help=(
                    "Driving pressure above PEEP applied during inspiration. "
                    "Delivered tidal volume depends on this setting plus "
                    "patient compliance and resistance."
                ),
                key=f"insp_p_{condition_name}_{engine_name}",
            )

        # --- Rise time — PCV and PRVC ---------------------------------------
        # --- Rise time — PCV and PRVC ---------------------------------------
        if engine_key in ("pcv", "prvc", "simv"):
            if is_neonatal:
                rise_time = st.slider(
                    "Rise Time (s)",
                    min_value=0.0, max_value=0.4,
                    value=float(preset.get("rise_time_s", 0.05)) if engine_key == "pcv" else 0.10, step=0.01,
                    help=(
                        "Time for pressure to ramp from PEEP to PIP. "
                        "0.0 = square wave step (maximum initial flow). "
                        "Longer rise times reduce peak flow and improve "
                        "patient comfort in spontaneously breathing patients."
                    ),
                    key=f"rise_{condition_name}_{engine_name}",
                )
            else:
                rise_time = st.slider(
                    "Rise Time (s)",
                    min_value=0.0, max_value=0.4,
                    value=0.0 if engine_key == "pcv" else 0.10, step=0.1,
                    help=(
                        "Time for pressure to ramp from PEEP to PIP. "
                        "0.0 = square wave step (maximum initial flow). "
                        "Longer rise times reduce peak flow and improve "
                        "patient comfort in spontaneously breathing patients."
                    ),
                    key=f"rise_{condition_name}_{engine_name}",
                )

        

        # --- PSV-specific parameters ------------------------------------
        # (unchanged — this block is already correct, shown here only so
        # you can see exactly where it sits relative to everything else)
        if engine_key == "psv":
            st.markdown(
                '<div class="section-label" style="margin-top:10px;">'
                'PSV — Ventilator Settings</div>',
                unsafe_allow_html=True,
            )
            ps = st.slider(
                "Pressure Support (cmH\u2082O above PEEP)",
                min_value=1, max_value=30,
                value=int(preset["pressure_support_cmH2O"]),
                step=1,
                help=(
                    "Pressure level the ventilator adds above PEEP during "
                    "each patient-triggered inspiration."
                ),
                key=f"ps_{condition_name}_{engine_name}",
            )
            fct = st.select_slider(
                "Flow Cycle Threshold",
                options=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                         0.50, 0.55, 0.60, 0.65, 0.70],
                value=float(preset["flow_cycle_threshold"]),
                help=(
                    "Inspiration ends when flow decays to this fraction "
                    "of peak. Low (0.10) → delayed cycling risk. "
                    "High (0.40) → premature cycling risk."
                ),
                key=f"fct_{condition_name}_{engine_name}",
            )
            thr = st.slider(
                "Trigger Threshold (cmH\u2082O)",
                min_value=0.5, max_value=3.0,
                value=float(preset["trigger_threshold_cmH2O"]),
                step=0.5,
                help=(
                    "Minimum inspiratory effort required to trigger the "
                    "ventilator. Higher values require more patient effort."
                ),
                key=f"thr_{condition_name}_{engine_name}",
            )
            rise_time = st.slider(
                "Rise Time (s)",
                min_value=0.0, max_value=0.4,
                value=0.1, step=0.1,
                help=(
                    "Time for airway pressure to ramp from PEEP to PS level. "
                    "0.0 = instantaneous step."
                ),
                key=f"rise_psv_{condition_name}_{engine_name}",
            )

            st.markdown(
                '<div class="section-label" style="margin-top:10px;">'
                'PSV — Patient Effort Model</div>',
                unsafe_allow_html=True,
            )
            if is_neonatal:
                effort_rate = st.slider(
                "Effort Rate (breaths/min)", 8, 70,
                value=int(preset["effort_rate_per_min"]),
                step=1,
                help=(
                        "Patient's neural respiratory rate — the rate at which "
                        "the patient attempts to breathe regardless of whether "
                        "each effort successfully triggers the ventilator."
                    ),
                key=f"erate_{condition_name}_{engine_name}",
                
            )
            else:
                effort_rate = st.slider(
                    "Effort Rate (breaths/min)", 8, 40,
                    value=int(preset["effort_rate_per_min"]),
                    step=1,
                    help=(
                        "Patient's neural respiratory rate — the rate at which "
                        "the patient attempts to breathe regardless of whether "
                        "each effort successfully triggers the ventilator."
                    ),
                    key=f"erate_{condition_name}_{engine_name}",
                )
            pmus = st.slider(
                "Peak Effort (Pmus cmH\u2082O)", 2, 25,
                value=int(preset["pmus_peak_cmH2O"]),
                step=1,
                help=(
                    "Mean peak inspiratory muscle pressure. Higher values "
                    "reflect stronger patient drive and larger tidal volumes."
                ),
                key=f"pmus_{condition_name}_{engine_name}",
            )
            effort_dur = st.slider(
                "Effort Duration (s)", 0.3, 1.4,
                value=float(preset["effort_duration_s"]),
                step=0.1,
                help=(
                    "Duration of each inspiratory effort (neural Ti). "
                    "Mismatch with ventilator Ti produces dyssynchrony."
                ),
                key=f"edur_{condition_name}_{engine_name}",
            )
            pmus_cv = st.slider(
                "Effort Variability (CV)", 0.0, 0.4,
                value=float(preset["pmus_cv"]),
                step=0.1,
                help=(
                    "Coefficient of variation of breath-to-breath Pmus. "
                    "Drives the tidal volume variability that distinguishes "
                    "PSV from mandatory modes."
                ),
                key=f"pcv_{condition_name}_{engine_name}",
            )
            if st.button(
                "⟳ Regenerate",
                help="Draw a new set of stochastic Pmus samples.",
                key="psv_regen",
            ):
                st.session_state["psv_seed"] = (
                    st.session_state.get("psv_seed", 42) + 1
                )
                st.rerun()
        
        if engine_key == "simv":                                                          
            st.markdown(                                                                  
                '<div class="section-label" style="margin-top:10px;">'                    
                'SIMV — Synchronization Window</div>',                                     
                unsafe_allow_html=True,                                                    
            )                                                                              
            f_window = st.slider(                                                          
                "Sync Window (fraction of mandatory cycle)",                               
                min_value=0.05, max_value=0.60,                                            
                value=0.25, step=0.05,                                                     
                help=(                                                                     
                    "Fraction of the mandatory cycle time, immediately "                   
                    "before the scheduled mandatory breath, during which "                 
                    "patient effort synchronizes (rather than replaces) "                  
                    "the mandatory breath. No single ventilator vendor "                   
                    "uses the same value for this -- 0.15-0.30 is the "                    
                    "literature-grounded tunable range used for this "                     
                    "project (see the SIMV grounding doc)."                                
                ),                                                                          
                key=f"fwindow_{condition_name}_{engine_name}",                             
            )                                                                              

            st.markdown(                                                                  
                '<div class="section-label" style="margin-top:10px;">'                    
                'SIMV — Spontaneous Breath Settings</div>',                                
                unsafe_allow_html=True,                                                    
            )                                                                              
            ps = st.slider(                                                                
                "Pressure Support (cmH\u2082O above PEEP)",                                
                min_value=1, max_value=30,                                                 
                value=int(preset["pressure_support_cmH2O"]), step=1,                       
                help="Pressure applied above PEEP for spontaneous breaths.",               
                key=f"simv_ps_{condition_name}_{engine_name}",                             
            )                                                                              
            fct = st.select_slider(                                                        
                "Flow Cycle Threshold",                                                    
                options=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,                   
                         0.50, 0.55, 0.60, 0.65, 0.70],                                    
                value=float(preset["flow_cycle_threshold"]),                               
                help=(                                                                     
                    "Spontaneous breaths cycle off when flow decays to "                   
                    "this fraction of peak. Low -> delayed cycling risk. "                 
                    "High -> premature cycling risk."                                      
                ),                                                                          
                key=f"simv_fct_{condition_name}_{engine_name}",                            
            )                                                                              
            thr = st.slider(                                                               
                "Trigger Threshold (cmH\u2082O)",                                          
                min_value=0.5, max_value=3.0,                                              
                value=float(preset["trigger_threshold_cmH2O"]), step=0.5,                  
                help=(                                                                     
                    "Minimum patient effort required to trigger either a "                 
                    "synchronized mandatory breath (inside the window) or "                
                    "a spontaneous breath (outside it)."                                   
                ),                                                                          
                key=f"simv_thr_{condition_name}_{engine_name}",                            
            )                                                                              
            effort_rate = st.slider(                                                       
                "Effort Rate (breaths/min)", 8, 40,                                        
                value=int(preset["effort_rate_per_min"]), step=1,                          
                help="Patient's own neural respiratory rate.",                             
                key=f"simv_erate_{condition_name}_{engine_name}",                          
            )                                                                              
            pmus = st.slider(                                                               
                "Peak Effort (Pmus cmH\u2082O)", 2, 25,                                    
                value=int(preset["pmus_peak_cmH2O"]), step=1,                              
                key=f"simv_pmus_{condition_name}_{engine_name}",                           
            )                                                                              
            effort_dur = st.slider(                                                        
                "Effort Duration (s)", 0.3, 1.4,                                           
                value=float(preset["effort_duration_s"]), step=0.1,                        
                key=f"simv_edur_{condition_name}_{engine_name}",                           
            )                                                                              
            pmus_cv = st.slider(                                                            
                "Effort Variability (CV)", 0.0, 0.4,                                       
                value=float(preset["pmus_cv"]), step=0.1,                                  
                key=f"simv_pcv_{condition_name}_{engine_name}",                            
            )                                                                              
            if st.button(                                                                   
                "⟳ Regenerate",                                                             
                help="Draw a new set of stochastic patient-effort samples.",               
                key="simv_regen",                                                         
            ):                                                                              
                st.session_state["simv_seed"] = (                                          
                    st.session_state.get("simv_seed", 42) + 1                              
                )                                                                            
                st.rerun() 

        st.markdown("<hr>", unsafe_allow_html=True)

        # --- PRVC-specific: pressure ceiling --------------------------------
        if engine_key == "prvc":
            ceiling = st.slider(
                "Pressure Ceiling (above PEEP)", min_value=15, max_value=35,
                value=int(preset.get("pressure_ceiling_cmH2O", 30)), step=1,
                help=(
                    "Safety limit on the adaptive pressure staircase. If the "
                    "algorithm needs more than this to hit the volume "
                    "target, it stops climbing and the scenario becomes "
                    "ceiling-limited (unable to guarantee VT)."
                ),
                key=f"ceiling_{condition_name}_{engine_name}",
            )

        # --- Breath cycle count — unconditional, every engine needs this ---
        _ncycles_default = 12 if engine_key in ("psv", "prvc") else (8 if engine_key == "simv" else 5) 
        n_cycles = st.slider(
            "Breath Cycles", 1, 30, _ncycles_default, step=1,
            help=(
                "PSV: use \u2265 12 cycles for COPD/Bronchospasm so auto-PEEP "
                "reaches steady state. PRVC: use \u2265 12 cycles (\u2265 25 "
                "for COPD/Bronchospasm) so the pressure staircase has room "
                "to converge."
                "SIMV: this counts mandatory macro-cycles, not total "                                          
                "breaths -- spontaneous breaths interleave on top of these." 
            ) if engine_key in ("psv", "prvc","simv") else None,
        )

        # --- Assemble params dict ---------------------------------------
        if engine_key == "vcv":
            params = {
                "respiratory_rate":        rr,
                "tidal_volume_ml":         tv,
                "compliance_ml_per_cmH2O": compliance,
                "resistance_cmH2O_L_s":    resistance,
                "ie_ratio":                ie,
                "peep_cmH2O":              peep,
                "flow_pattern":            flow_pattern,
                "condition":               condition_name,
            }

        elif engine_key == "pcv":
            params = {
                "respiratory_rate":        rr,
                "insp_pressure_cmH2O":     insp_pressure,
                "compliance_ml_per_cmH2O": compliance,
                "resistance_cmH2O_L_s":    resistance,
                "ie_ratio":                ie,
                "peep_cmH2O":              peep,
                "rise_time_s":             rise_time,
                "tidal_volume_ml":         500,   # required by validator, not used
                "condition":               condition_name,
            }

        elif engine_key == "psv":
            params = {
                "pressure_support_cmH2O":  ps,
                "peep_cmH2O":              peep,
                "rise_time_s":             rise_time,
                "flow_cycle_threshold":    fct,
                "trigger_threshold_cmH2O": thr,
                "pmus_peak_cmH2O":         pmus,
                "effort_rate_per_min":     effort_rate,
                "effort_duration_s":       effort_dur,
                "pmus_cv":                 pmus_cv,
                "compliance_ml_per_cmH2O": compliance,
                "resistance_cmH2O_L_s":    resistance,
                "condition":               condition_name,
            }

        elif engine_key == "prvc":
            params = {
                "vt_target_ml":            tv,
                "respiratory_rate":        rr,
                "peep_cmH2O":              peep,
                "ie_ratio":                ie,
                "pressure_ceiling_cmH2O":  ceiling,
                "rise_time_s":             rise_time,
                "compliance_ml_per_cmH2O": compliance,
                "resistance_cmH2O_L_s":    resistance,
                "condition":               condition_name,
            }
        
        elif engine_key == "simv":                                                          
            params = {                                                                       
                "mandatory_mode":           mode,                                            
                "respiratory_rate":         rr,                                              
                "peep_cmH2O":               peep,                                            
                "ie_ratio":                 ie,                                              
                "rise_time_s":              rise_time,                                       
                "f_window":                 f_window,                                        
                "pressure_support_cmH2O":   ps,                                              
                "flow_cycle_threshold":     fct,                                             
                "trigger_threshold_cmH2O":  thr,                                             
                "pmus_peak_cmH2O":          pmus,                                            
                "effort_rate_per_min":      effort_rate,                                     
                "effort_duration_s":        effort_dur,                                      
                "pmus_cv":                  pmus_cv,                                         
                "compliance_ml_per_cmH2O":  compliance,                                      
                "resistance_cmH2O_L_s":     resistance,                                      
                "condition":                condition_name,                                  
            }                                                                                
            if mode == "VC":                                                                 
                params["tidal_volume_ml"] = tv                                               
                params["flow_pattern"]    = flow_pattern                                     
            else:                                                                             
                params["insp_pressure_cmH2O"] = insp_pressure 
        params["population"] = preset.get("population", "adult")
        params["weight_kg"]  = preset.get("weight_kg", 3.0 if is_neonatal else 70.0)

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
    auto_peep = result["auto_peep_cmH2O"]
    

    population = params.get("population", "adult")
    if population == "neonate":
        if "patient_vt_ml" in result and "delivered_vt_ml" in result:
            patient_vt   = result["patient_vt_ml"]
            delivered_vt = result["delivered_vt_ml"]
            if delivered_vt and delivered_vt > 0:
                leak_pct = 100.0 * (1.0 - patient_vt / delivered_vt)
                leak_col = st.columns(1)[0]
                _metric_card(leak_col, "Leak (measured)", f"{leak_pct:.0f}", "%")
        else:
            leak_frac = params.get("ett_cuff_leak_fraction",
                                    params.get("cuff_leak_fraction", 0.0))
            if leak_frac:
                leak_col = st.columns(1)[0]
                _metric_card(leak_col, "Leak (configured)", f"{leak_frac * 100:.0f}", "%")
    

    if engine_key == "vcv":
        cols = st.columns(9)
        
        pplat     = result["pplat_cmH2O"]
        driving_p = result["driving_p_cmH2O"]
        rr        = params["respiratory_rate"]
        minute_vent = result.get("minute_vent_l", rr * peak_v / 1000.0)

        metrics = [
            ("P Peak", f"{peak_p:.1f}",    "cmH₂O"),
            ("P Plat",    f"{pplat:.1f}",     "cmH₂O"),
            ("P Drive",     f"{driving_p:.1f}", "cmH₂O"),
            ("Tidal Vol",     f"{result['delivered_vt_ml']:.0f}",    "ml"),
            ("P Mean",      f"{mean_paw:.1f}",  "cmH₂O"),
            ("Peak Flow Insp", f"{peak_f:.2f}", "l/s"),
            ("Peak Flow Exp", f"{min_f:.2f}",  "l/s"),
            ("Minute Vol",   f"{minute_vent:.1f}", "l/min"),
            ("Auto-PEEP",     f"{auto_peep:.2f}", "cmH₂O"),
        ]

    elif engine_key == "pcv":
        cols = st.columns(9)
        insp_p    = params.get("insp_pressure_cmH2O", 0)
        driving_p = float(insp_p)
        fill_frac = result["fill_fraction"]
        rr        = params["respiratory_rate"]
        minute_vent = result.get("minute_vent_l", rr * peak_v / 1000.0)

        metrics = [
            ("P Peak",  f"{peak_p:.1f}",    "cmH₂O"),
            ("Delivered VT", f"{result['delivered_vt_ml']:.0f}", "ml"),
            ("P Drive",      f"{driving_p:.1f}", "cmH₂O"),
            ("P Mean",       f"{mean_paw:.1f}",  "cmH₂O"),
            ("Peak Flow Insp", f"{peak_f:.2f}",  "l/s"),
            ("Peak Flow Exp", f"{min_f:.2f}",  "l/s"),
            ("Fill Fraction",  f"{fill_frac:.2f}", ""),
            ("Minute Vol",    f"{minute_vent:.1f}", "l/min"),
            ("Auto-PEEP",      f"{auto_peep:.2f}", "cmH₂O"),
        ]

    elif engine_key == "psv":
        cols = st.columns(9)

        delivered_vt  = result.get("delivered_vt_ml",              0.0)
        patient_vt    = result.get("patient_vt_ml",    delivered_vt)
        auto_peep     = result.get("auto_peep_cmH2O",              0.0)
        fill_frac     = result.get("fill_fraction",                0.0)
        pres_pel      = result.get("pres_pel_ratio",               0.0)
        ineff_frac    = result.get("ineffective_trigger_fraction",  0.0)
        trig_rr       = result.get("triggered_breath_rate",         0.0)
        minute_vent   = result.get("minute_vent_l",                 0.0)

        metrics = [
            ("Peak Pressure",  f"{peak_p:.1f}",       "cmH₂O"),
            ("Delivered VT",   f"{delivered_vt:.0f}",  "ml"),
            ("Patient VT",     f"{patient_vt:.0f}",    "ml"),
            ("Auto-PEEP",      f"{auto_peep:.2f}",    "cmH₂O"),
            ("Fill Fraction",  f"{fill_frac:.3f}",     ""),
            ("Pres/Pel",       f"{pres_pel:.2f}",      ""),
            ("Ineff Frac",     f"{ineff_frac:.2f}",    ""),
            ("Trig RR",        f"{trig_rr:.1f}",       "bpm"),
            ("Minute Vent",    f"{minute_vent:.1f}",   "l/min"),
        ]
        

        # Dyssynchrony label summary bar
        labels = result.get("breath_dyssynchrony_labels", [])
        if labels:
            from collections import Counter
            counts = Counter(labels)
            total  = len(labels)
            parts  = [
                f"{lbl.replace('_', ' ')}: {n}/{total}"
                for lbl, n in sorted(counts.items())
                if n > 0
            ]
            st.markdown(
                f'<div style="font-family:JetBrains Mono,monospace;'
                f'font-size:0.65rem;color:{COLOR_MUTED};margin-top:6px;">'
                f'Dyssynchrony — {" · ".join(parts)}</div>',
                unsafe_allow_html=True,
            )


    elif engine_key == "prvc":
        cols = st.columns(7)

        delivered_vt = result["delivered_vt_ml"]
        driving_p    = result["driving_p_cmH2O"]
        minute_vent  = result["minute_vent_l"]
        breaths_conv = result["breaths_to_converge"]
        ppeak_final  = result["ppeak_final_breath_cmH2O"]

        metrics = [
            ("PPeak",              f"{ppeak_final:.1f}",       "cmH₂O"),
            ("Delivered VT",       f"{delivered_vt:.0f}", "ml"),
            ("Driving P (final)",  f"{driving_p:.1f}",    "cmH₂O"),
            ("P Mean",             f"{mean_paw:.1f}",     "cmH₂O"),
            ("Auto-PEEP",          f"{auto_peep:.2f}",    "cmH₂O"),
            ("Minute Vol",         f"{minute_vent:.1f}",  "l/min"),
            ("Breaths to Converge",
             f"{breaths_conv}" if breaths_conv else "—",  ""),
        ]
    
    elif engine_key == "simv":                                                              
        cols = st.columns(8)                                                                 

        mand_vt     = result["mandatory_delivered_vt_ml"]                                    
        spont_vt    = result["spontaneous_delivered_vt_ml"]                                  
        sync_frac   = result["mandatory_synchronized_fraction"]                              
        n_spont     = result["n_spontaneous_breaths"]                                        
        driving_p   = result["driving_p_cmH2O"]                                              
        minute_vent = result["minute_vent_l"]                                                

        metrics = [                                                                          
            ("P Peak",          f"{peak_p:.1f}",                          "cmH₂O"),          
            ("Mandatory VT",    f"{mand_vt:.0f}",                         "ml"),             
            ("Spontaneous VT",  f"{spont_vt:.0f}" if n_spont else "—",    "ml"),             
            ("P Drive (mand.)", f"{driving_p:.1f}",                       "cmH₂O"),          
            ("P Mean",          f"{mean_paw:.1f}",                        "cmH₂O"),          
            ("Synchronized",    f"{sync_frac*100:.0f}",                   "%"),              
            ("Spont. Breaths",  f"{n_spont}",                             ""),               
            ("Auto-PEEP",       f"{auto_peep:.2f}",                       "cmH₂O"),          
        ]     

    for col, (label, value, unit) in zip(cols, metrics):
        _metric_card(col, label, value, unit)
    
    if engine_key == "simv" and result.get("ineffective_trigger_fraction", 0) > 0:          
        st.markdown(                                                                         
            f'<div style="font-family:JetBrains Mono,monospace;'                            
            f'font-size:0.65rem;color:{COLOR_MUTED};margin-top:6px;">'                      
            f'Ineffective efforts: '                                                         
            f'{result["ineffective_trigger_fraction"]*100:.0f}%</div>',                     
            unsafe_allow_html=True,                                                          
        )       

    if engine_key == "prvc":
        if result["ceiling_limited"]:
            st.warning(
                "⚠ Ceiling-limited — the pressure ceiling was reached before "
                "the volume target could be met. This scenario did not "
                "converge; delivered Vt stayed below target."
            )
        elif result["converged"]:
            st.success("✓ Converged — working pressure stabilized within tolerance.")
        else:
            st.info("Not yet converged within the selected breath cycle count.")

    if engine_key != "psv" and not result.get("equilibrium_reached", True):
        st.warning(
            "⚠ Volume has not stabilized after the selected breath cycles. "
            "Increase Breath Cycles for accurate delivered VT and auto-PEEP metrics."
        )
        

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
        "flow_ls":        result["flow"],
        "volume_ml":      result["volume"],
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
        if k != "tidal_volume_ml" or engine_key in ("vcv", "simv")
    })

    if engine_key == "psv":
        scenario["breath_dyssynchrony_labels"] = result.get(
            "breath_dyssynchrony_labels", []
        )
        scenario["auto_peep_cmH2O"]              = result.get("auto_peep_cmH2O", 0.0)
        scenario["ineffective_trigger_fraction"]  = result.get(
            "ineffective_trigger_fraction", 0.0
        )
        scenario["pres_pel_ratio"]               = result.get("pres_pel_ratio", 0.0)
        scenario["is_valid"]                     = result.get("is_valid", True)
        scenario["invalid_reason"]               = result.get("invalid_reason", "")

    if engine_key == "simv":                                                                 
        scenario["breath_records"] = result.get("breath_records", [])                       
        scenario["n_mandatory_breaths"] = result.get("n_mandatory_breaths", 0)                
        scenario["n_spontaneous_breaths"] = result.get("n_spontaneous_breaths", 0)            
        scenario["mandatory_synchronized_fraction"] = result.get(                       
            "mandatory_synchronized_fraction", 0.0                                            
        )                                                                                      
        scenario["auto_peep_cmH2O"] = result.get("auto_peep_cmH2O", 0.0)                     
        scenario["ineffective_trigger_fraction"] = result.get(                                
            "ineffective_trigger_fraction", 0.0                                               
        )                                                                                      
        scenario["is_valid"] = result.get("is_valid", True)                                  
        scenario["invalid_reason"] = result.get("invalid_reason", "")     

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
    if ENGINES[engine_name]["key"] == "psv":
        # Fixed seed gives reproducible stochastic draws on the same params.
        # Stored in session state so a Regenerate button can increment it.
        seed = st.session_state.get("psv_seed", 42)
        return ENGINES[engine_name]["fn"](params, n_cycles=n_cycles, seed=seed)
    if ENGINES[engine_name]["key"] == "simv":                                                
        seed = st.session_state.get("simv_seed", 42)                                         
        return ENGINES[engine_name]["fn"](params, n_cycles=n_cycles, seed=seed)
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