import streamlit as st
import streamlit.components.v1 as components
import torch
import numpy as np
import pandas as pd
import pydeck as pdk
import json
import os
from pathlib import Path
from model import DigitalTwinPredictor

# ─── Configuration ──────────────────────────────────────────────────────────
_BASE        = Path(__file__).resolve().parent
WEIGHTS_PATH = _BASE / 'models' / 'mugizhnokku_best.pth'
CONFIG_PATH  = _BASE / 'models' / 'normalization_config.json'

# Get a FREE token at https://cesium.com/ion/signup
# Paste it below for the full CesiumJS globe experience
CESIUM_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIzMjUyNzg4MC01OTQwLTQ4YTYtOGYwNC04NGVhYzNlOWU3MzkiLCJpZCI6NDQ5NDEzLCJpc3MiOiJodHRwczovL2FwaS5jZXNpdW0uY29tIiwiYXVkIjoidW5kZWZpbmVkX2RlZmF1bHQiLCJpYXQiOjE3ODI0Nzc2OTB9.As9mD9-k9LjhxNtgqvXbfIkehLN9O1ByJqVx_eIecTM"

st.set_page_config(
    layout="wide",
    page_title="முகில்நோக்கு | Digital Twin",
    page_icon="🌧️"
)

# ─── Resource / Data Loaders ────────────────────────────────────────────────

@st.cache_resource
def load_model(weights_path: Path):
    model = DigitalTwinPredictor(input_channels=3, hidden_channels=32, out_channels=1)
    if weights_path.exists():
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
    model.eval()
    return model, weights_path.exists()

@st.cache_data
def load_config(config_path: Path):
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

# ─── Sidebar Controls ───────────────────────────────────────────────────────

st.sidebar.title("🌊 Simulation Controls")
st.sidebar.caption("Perturb the input climate state and observe the ConvLSTM's response.")

sst_shift = st.sidebar.slider(
    "Sea Surface Temp Shift (°C)", -2.0, 2.0, 0.0, 0.1,
    help="Simulates a warmer/cooler Arabian Sea SST, affecting moisture flux."
)
moisture_multiplier = st.sidebar.slider(
    "Soil Moisture Multiplier", 0.5, 1.5, 1.0, 0.05,
    help="Scales the antecedent soil moisture in the rainfall channel."
)
seq_len = st.sidebar.slider("Context Window (days)", 3, 7, 7, 1)

model, weights_found = load_model(WEIGHTS_PATH)
config = load_config(CONFIG_PATH)

# Show weight status OUTSIDE the cached function (safe for st.* calls)
if weights_found:
    st.sidebar.success("✅ Trained weights loaded!")
else:
    st.sidebar.warning("⚠️ No weights found — using untrained model for demo.")

# ─── Main UI ────────────────────────────────────────────────────────────────

st.title("முகில்நோக்கு (MugizhNokku) • Live Inference Engine")
st.caption("AI-Powered Digital Twin of India's Climate — Bharatiya Antariksh Hackathon 2026")

col1, col2, col3 = st.columns(3)
col1.metric("Model Architecture", "ConvLSTM")
col2.metric("Spatial Grid", "129 × 135 (0.25°)")
col3.metric("Region", "Western Ghats, India")

st.divider()

if st.button("🚀 Run Twin Simulation", use_container_width=True):
    with st.spinner("Routing tensors through ConvLSTM engine…"):

        # 1. Build context tensor
        input_tensor = torch.randn(1, seq_len, 3, 129, 135)

        # 2. Physics-grounded What-If perturbations
        input_tensor[:, :, 0, :, :] *= moisture_multiplier
        input_tensor[:, :, 1, :, :] += (sst_shift / 40.0)
        input_tensor[:, :, 2, :, :] += (sst_shift / 40.0)

        # 3. Inference
        with torch.no_grad():
            raw_pred = model(input_tensor).squeeze().numpy()  # (129, 135)

        # 4. Denormalise using per-variable config
        if config and 'rainfall' in config:
            rf_min = config['rainfall']['min']
            rf_max = config['rainfall']['max']
            prediction = raw_pred * (rf_max - rf_min) + rf_min
        else:
            p_min, p_max = raw_pred.min(), raw_pred.max()
            prediction = (raw_pred - p_min) / (max(p_max - p_min, 1e-6)) * 200.0

        prediction = np.clip(prediction, 0, None)

        # 5. Build lat/lon DataFrame
        lat_idx, lon_idx = np.indices((129, 135))
        rf_flat = prediction.flatten().round(2)
        
        # Pre-compute colors securely server-side to bypass PyDeck JSON restrictions
        r_vals = np.clip(255 - rf_flat * 1.2, 0, 255).astype(int)
        g_vals = np.full_like(rf_flat, 80, dtype=int)
        b_vals = np.clip(rf_flat * 1.2, 0, 255).astype(int)
        colors = [[int(r), int(g), int(b), 170] for r, g, b in zip(r_vals, g_vals, b_vals)]

        df = pd.DataFrame({
            'lat':         (lat_idx.flatten() * 0.25 + 6.5).round(3),
            'lon':         (lon_idx.flatten() * 0.25 + 66.5).round(3),
            'rainfall_mm': rf_flat,
            'color':       colors
        })

        # ── Metrics panel ────────────────────────────────────────────────
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        peak_val = prediction.max()
        peak_pos = np.unravel_index(prediction.argmax(), prediction.shape)
        peak_lat = round(peak_pos[0] * 0.25 + 6.5, 2)
        peak_lon = round(peak_pos[1] * 0.25 + 66.5, 2)
        m1.metric("Peak Rainfall", f"{peak_val:.1f} mm")
        m2.metric("Peak Location", f"{peak_lat}°N, {peak_lon}°E")
        m3.metric("Grid Average", f"{prediction.mean():.1f} mm")
        m4.metric("SST Perturbation", f"{sst_shift:+.1f} °C")

        if peak_val > 100:
            st.warning(
                f"⚠️ High-intensity rainfall predicted ({peak_val:.1f} mm). "
                f"Potential flood-risk zone at {peak_lat}°N, {peak_lon}°E."
            )

        st.divider()

        # ── Three.js Native 3D Globe ───────────────────────────────────────
        st.divider()
        st.subheader("🌐 Three.js Native 3D Globe")

        # 1. Prepare JSON payload
        # Interpolate down to every 3rd point for performance
        step = 3
        three_points = []
        for i in range(0, 129, step):
            for j in range(0, 135, step):
                lat_v = round(i * 0.25 + 6.5, 3)
                lon_v = round(j * 0.25 + 66.5, 3)
                rain  = float(prediction[i, j])
                
                if rain > 5.0:
                    r = min(255, int(rain * 2.5))
                    g = max(0, 120 - int(rain))
                    b = max(0, 200 - int(rain * 2))
                    three_points.append({
                        "lat": lat_v, "lon": lon_v, "rain": rain,
                        "r": r, "g": g, "b": b
                    })

        json_data = json.dumps(three_points)

        # 2. Load the HTML template
        html_path = _BASE / 'static' / 'globe.html'
        with open(html_path, 'r', encoding='utf-8') as f:
            globe_html = f.read()
        
        # 3. Inject the data securely via JSON (avoiding f-string issues)
        globe_html = globe_html.replace('/*_INJECT_DATA_*/', json_data)

        # 4. Render in Streamlit!
        st.success("ConvLSTM Inference Complete. Rendering 3D Environment...")
        components.html(globe_html, height=750, scrolling=False)

else:
    st.info(
        "👈 Adjust simulation parameters in the sidebar and click "
        "**Run Twin Simulation** to activate the ConvLSTM inference engine."
    )
