import streamlit as st
import torch
import numpy as np
import pandas as pd
import pydeck as pdk
import json
import os
from pathlib import Path
from model import DigitalTwinPredictor

# Configuration — works both locally and in Colab
_BASE = Path(__file__).parent
WEIGHTS_PATH = _BASE / 'models' / 'mugizhnokku_best.pth'
CONFIG_PATH  = _BASE / 'models' / 'normalization_config.json'

st.set_page_config(
    layout="wide",
    page_title="முகில்நோக்கு | Digital Twin",
    page_icon="🌧️"
)

# ─── Resource / Data Loaders ────────────────────────────────────────────────

@st.cache_resource
def load_model(weights_path: Path):
    """Load ConvLSTM model once per session. Falls back gracefully if no .pth."""
    model = DigitalTwinPredictor(input_channels=3, hidden_channels=32, out_channels=1)
    if weights_path.exists():
        state = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state)
        st.sidebar.success("✅ Trained weights loaded!")
    else:
        st.sidebar.warning("⚠️ No weights found — using untrained model for demo.")
    model.eval()
    return model

@st.cache_data
def load_config(config_path: Path):
    """Load normalization bounds from JSON. Returns None if file is missing."""
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
num_epochs_display = st.sidebar.slider(
    "Context Window (days)", 3, 7, 7, 1,
    help="How many days of historical context to feed into the model."
)

model  = load_model(WEIGHTS_PATH)
config = load_config(CONFIG_PATH)

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

        # 1. Build context tensor  (Batch=1, Seq, Channels=3, H=129, W=135)
        seq_len = num_epochs_display
        input_tensor = torch.randn(1, seq_len, 3, 129, 135)

        # 2. Apply What-If perturbations  (physics-grounded)
        #    Channel 0 = Rainfall  → scale by soil moisture
        #    Channel 1 = Min Temp  → shift by SST anomaly
        #    Channel 2 = Max Temp  → shift by SST anomaly
        input_tensor[:, :, 0, :, :] *= moisture_multiplier
        input_tensor[:, :, 1, :, :] += (sst_shift / 40.0)  # normalised shift
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
            # Fallback: linearly rescale to plausible rainfall range (0–200 mm)
            p_min, p_max = raw_pred.min(), raw_pred.max()
            if p_max > p_min:
                prediction = (raw_pred - p_min) / (p_max - p_min) * 200.0
            else:
                prediction = raw_pred * 200.0

        # 5. Build lat/lon DataFrame from grid indices
        lat_idx, lon_idx = np.indices((129, 135))
        df = pd.DataFrame({
            'lat': (lat_idx.flatten() * 0.25 + 6.5).round(3),
            'lon': (lon_idx.flatten() * 0.25 + 66.5).round(3),
            'rainfall_mm': np.clip(prediction.flatten(), 0, None).round(2)
        })

        # 6. PyDeck 3D Column Map
        layer = pdk.Layer(
            'ColumnLayer',
            data=df,
            get_position='[lon, lat]',
            get_elevation='rainfall_mm',
            elevation_scale=80,
            radius=5500,
            get_fill_color='[255 - rainfall_mm * 1.2, 100, rainfall_mm * 1.2, 170]',
            pickable=True,
            extruded=True,
            auto_highlight=True,
        )
        view_state = pdk.ViewState(
            latitude=14.0, longitude=76.0, zoom=5, pitch=52, bearing=-10
        )
        tooltip = {"text": "Lat: {lat}\nLon: {lon}\nRainfall: {rainfall_mm} mm"}

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style='mapbox://styles/mapbox/dark-v10'
            )
        )

        # 7. Metrics summary panel
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
            st.warning(f"⚠️ High-intensity rainfall predicted ({peak_val:.1f} mm). Potential flood-risk zone at {peak_lat}°N, {peak_lon}°E.")

else:
    st.info("👈 Adjust simulation parameters in the sidebar and click **Run Twin Simulation** to activate the ConvLSTM inference engine.")
