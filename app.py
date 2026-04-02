"""
app.py
------
Entry point for the Ventilator Waveform Simulator.

Run with:
    streamlit run app.py
"""

from ui.dashboard import render

if __name__ == "__main__":
    render()
else:
    # Streamlit imports and runs the module directly — render on import
    render()