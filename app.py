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
        tab1, tab2 = st.tabs(["🌐 CesiumJS 3D Globe", "📊 PyDeck Column Map"])

        # ── TAB 1: CesiumJS 3D Globe ─────────────────────────────────────
        with tab1:
            # Build a CZML data packet from the prediction grid
            # Interpolate down to every 4th point for performance (32x34 points)
            step = 4
            cesium_points = []
            for i in range(0, 129, step):
                for j in range(0, 135, step):
                    lat_v = round(i * 0.25 + 6.5, 3)
                    lon_v = round(j * 0.25 + 66.5, 3)
                    rain  = float(prediction[i, j])
                    height = rain * 800  # exaggerate for visibility
                    # Colour: blue (low) → red (high) in 0-255 RGB
                    r = min(255, int(rain * 2.5))
                    g = max(0, 120 - int(rain))
                    b = max(0, 200 - int(rain * 2))
                    cesium_points.append(
                        f"{{lon:{lon_v}, lat:{lat_v}, h:{height:.1f}, "
                        f"r:{r}, g:{g}, b:{b}, rain:{rain:.1f}}}"
                    )

            cesium_js_data = ",\n".join(cesium_points)
            has_token = CESIUM_TOKEN != "YOUR_CESIUM_ION_TOKEN_HERE"

            cesium_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>MugizhNokku CesiumJS</title>
  <script>window.CESIUM_BASE_URL = "https://cesium.com/downloads/cesiumjs/releases/1.114/Build/Cesium/";</script>
  <script src="https://cesium.com/downloads/cesiumjs/releases/1.114/Build/Cesium/Cesium.js"></script>
  <link href="https://cesium.com/downloads/cesiumjs/releases/1.114/Build/Cesium/Widgets/widgets.css" rel="stylesheet"/>
  <style>
    html,body,#cesiumContainer{{width:100%;height:520px;margin:0;padding:0;overflow:hidden;background:#000;}}
    #info{{position:absolute;top:10px;left:50%;transform:translateX(-50%);
           background:rgba(0,0,0,0.7);color:#fff;padding:6px 14px;border-radius:20px;
           font-family:sans-serif;font-size:13px;z-index:99;}}
  </style>
</head>
<body>
<div id="cesiumContainer"></div>
<div id="info">🌧️ முகில்நோக்கு · ConvLSTM T+1 Rainfall Forecast (mm)</div>
<script>
    {'Cesium.Ion.defaultAccessToken = "' + CESIUM_TOKEN + '";' if has_token else '// No token — using offline terrain'}
  const viewer = new Cesium.Viewer("cesiumContainer", {{
    animation: false, timeline: false, fullscreenButton: false,
    baseLayerPicker: false, homeButton: false, sceneModePicker: false,
    navigationHelpButton: false, geocoder: false,
    {'terrain: Cesium.Terrain.fromWorldTerrain(),' if has_token else ''}
  }});

  const points = [{cesium_js_data}];

  points.forEach(p => {{
    viewer.entities.add({{
      position: Cesium.Cartesian3.fromDegrees(p.lon, p.lat, p.h),
      cylinder: {{
        length: p.h < 100 ? 100 : p.h,
        topRadius: 0,
        bottomRadius: 12000,
        material: new Cesium.ColorMaterialProperty(
          Cesium.Color.fromBytes(p.r, p.g, p.b, 200)
        ),
        outline: false,
      }},
      description: `Rainfall: ${{p.rain.toFixed(1)}} mm<br>Lat: ${{p.lat}}°N | Lon: ${{p.lon}}°E`,
    }});
  }});

  viewer.camera.flyTo({{
    destination: Cesium.Cartesian3.fromDegrees(76.0, 14.0, 1800000),
    orientation: {{ heading: Cesium.Math.toRadians(0),
                    pitch: Cesium.Math.toRadians(-45), roll: 0 }}
  }});
</script>
</body>
</html>"""

            components.html(cesium_html, height=520)
            if not has_token:
                st.info(
                    "💡 Get a **free** Cesium Ion token at [cesium.com/ion/signup](https://cesium.com/ion/signup) "
                    "and paste it into `CESIUM_TOKEN` in `app.py` to unlock full 3D world terrain."
                )

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
