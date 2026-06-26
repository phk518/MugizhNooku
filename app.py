import streamlit as st
import streamlit.components.v1 as components
import torch
import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
import json
import os
from pathlib import Path
from model import DigitalTwinPredictor

# ─── Configuration ──────────────────────────────────────────────────────────
_BASE        = Path(__file__).parent
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
        df = pd.DataFrame({
            'lat':         (lat_idx.flatten() * 0.25 + 6.5).round(3),
            'lon':         (lon_idx.flatten() * 0.25 + 66.5).round(3),
            'rainfall_mm': rf_flat,
            'color_r':     np.clip(255 - rf_flat * 1.2, 0, 255).astype(int),
            'color_g':     80,
            'color_b':     np.clip(rf_flat * 1.2, 0, 255).astype(int)
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

        # ── Dual-Tab Visualization ───────────────────────────────────────
        tab1, tab2 = st.tabs(["🌐 3D Globe View", "📊 PyDeck Column Map"])

        # ── TAB 1: Plotly 3D Globe ────────────────────────────────────────
        with tab1:
            step = 2
            lats, lons, rains = [], [], []
            for i in range(0, 129, step):
                for j in range(0, 135, step):
                    lats.append(round(i * 0.25 + 6.5,  3))
                    lons.append(round(j * 0.25 + 66.5, 3))
                    rains.append(float(prediction[i, j]))

            fig = go.Figure()

            fig.add_trace(go.Scattergeo(
                lat=lats, lon=lons,
                mode='markers',
                marker=dict(
                    size=[max(3, r * 0.15) for r in rains],
                    color=rains,
                    colorscale=[
                        [0.0,  '#0d47a1'],
                        [0.3,  '#1976d2'],
                        [0.5,  '#ffeb3b'],
                        [0.75, '#ff5722'],
                        [1.0,  '#b71c1c'],
                    ],
                    cmin=0,
                    cmax=float(np.percentile(rains, 99)),
                    colorbar=dict(
                        title="Rainfall (mm)",
                        thickness=15,
                        len=0.6,
                        bgcolor='rgba(0,0,0,0.5)',
                        tickfont=dict(color='white'),
                        titlefont=dict(color='white'),
                    ),
                    opacity=0.85,
                ),
                text=[f"{r:.1f} mm" for r in rains],
                hovertemplate="<b>Rainfall: %{text}</b><br>Lat: %{lat}°N<br>Lon: %{lon}°E<extra></extra>",
                name='Predicted Rainfall',
            ))

            fig.update_layout(
                paper_bgcolor='#0a0a1a',
                plot_bgcolor='#0a0a1a',
                margin=dict(l=0, r=0, t=30, b=0),
                title=dict(
                    text="🌧️ முகில்நோக்கு · ConvLSTM T+1 Rainfall Forecast",
                    font=dict(color='white', size=15),
                    x=0.5
                ),
                geo=dict(
                    projection_type='orthographic',
                    projection_rotation=dict(lon=80, lat=20, roll=0),
                    showland=True,   landcolor='#1b2a1b',
                    showocean=True,  oceancolor='#0d1b2a',
                    showlakes=True,  lakecolor='#0d1b2a',
                    showcountries=True, countrycolor='#444',
                    showcoastlines=True, coastlinecolor='#555',
                    showframe=False,
                    bgcolor='#0a0a1a',
                    lonaxis=dict(range=[60, 100]),
                    lataxis=dict(range=[5, 40]),
                ),
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # ── TAB 2: PyDeck Column Map ──────────────────────────────────────
        with tab2:
            layer = pdk.Layer(
                'ColumnLayer', data=df,
                get_position='[lon, lat]',
                get_elevation='rainfall_mm',
                elevation_scale=80, radius=5500,
                get_fill_color='[color_r, color_g, color_b, 170]',
                pickable=True, extruded=True, auto_highlight=True,
            )
            view_state = pdk.ViewState(
                latitude=14.0, longitude=76.0, zoom=5, pitch=52, bearing=-10
            )
            tooltip = {"text": "Lat: {lat}\nLon: {lon}\nRainfall: {rainfall_mm} mm"}
            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer], initial_view_state=view_state, tooltip=tooltip,
                    map_style='mapbox://styles/mapbox/dark-v10'
                )
            )

else:
    st.info(
        "👈 Adjust simulation parameters in the sidebar and click "
        "**Run Twin Simulation** to activate the ConvLSTM inference engine."
    )
