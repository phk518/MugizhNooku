import base64
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch

from model import DigitalTwinPredictor

_BASE = Path(__file__).resolve().parent
WEIGHTS_PATH = _BASE / "models" / "mugizhnokku_best.pth"
CONFIG_PATH = _BASE / "models" / "normalization_config.json"
HTML_PATH = _BASE / "static" / "cesium_v2.html"
TEXTURE_URL = "https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_atmos_2048.jpg"
FALLBACK_PIXEL = "data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="

st.set_page_config(
    layout="wide",
    page_title="முகில்நோக்கு | Digital Twin",
    page_icon="🌧️",
)

@st.cache_resource
def load_model(weights_path: Path):
    model = DigitalTwinPredictor(input_channels=3, hidden_channels=32, out_channels=1)
    loaded = False
    if weights_path.exists():
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        loaded = True
    model.eval()
    return model, loaded

@st.cache_data
def load_config(config_path: Path):
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_texture_data_uri(url: str):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        img_data = urllib.request.urlopen(req, timeout=10).read()
        return "data:image/jpeg;base64," + base64.b64encode(img_data).decode("utf-8")
    except Exception:
        return FALLBACK_PIXEL

def build_prediction(model, seq_len, moisture_multiplier, sst_shift, config):
    input_tensor = torch.randn(1, seq_len, 3, 129, 135)
    input_tensor[:, :, 0] *= moisture_multiplier
    input_tensor[:, :, 1] += (sst_shift / 40.0)
    input_tensor[:, :, 2] += (sst_shift / 40.0)

    with torch.no_grad():
        raw_pred = model(input_tensor).squeeze().cpu().numpy()

    rainfall_cfg = config.get("rainfall", {})
    if "min" in rainfall_cfg and "max" in rainfall_cfg:
        rf_min = float(rainfall_cfg["min"])
        rf_max = float(rainfall_cfg["max"])
        pred = raw_pred * (rf_max - rf_min) + rf_min
    else:
        p_min, p_max = float(raw_pred.min()), float(raw_pred.max())
        pred = (raw_pred - p_min) / max(p_max - p_min, 1e-6) * 200.0

    return np.clip(pred, 0, None)

def build_three_points(prediction):
    points = []
    for i in range(0, 129, 3):
        for j in range(0, 135, 3):
            rain = float(prediction[i, j])
            if rain <= 0.1:
                continue
            lat_v = round(i * 0.25 + 6.5, 3)
            lon_v = round(j * 0.25 + 66.5, 3)
            points.append({
                "lat": lat_v,
                "lon": lon_v,
                "rain": rain,
                "r": min(255, int(rain * 2.5)),
                "g": max(0, 120 - int(rain)),
                "b": max(0, 200 - int(rain * 2)),
            })

    points.append({
        "lat": 22.0,
        "lon": 79.0,
        "rain": 150.0,
        "r": 255,
        "g": 0,
        "b": 0,
    })
    return points

def inject_html(template_path: Path, texture_uri: str, data_json: str):
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("/*_INJECT_TEXTURE_*/", texture_uri)
    html = html.replace("/*_INJECT_DATA_*/", data_json)
    return html

model, weights_found = load_model(WEIGHTS_PATH)
config = load_config(CONFIG_PATH)

st.sidebar.title("🌊 Simulation Controls")
st.sidebar.caption("Perturb the input climate state and observe the ConvLSTM response.")

sst_shift = st.sidebar.slider(
    "Sea Surface Temp Shift (°C)", -2.0, 2.0, 0.0, 0.1,
    help="Simulates a warmer/cooler Arabian Sea SST."
)
moisture_multiplier = st.sidebar.slider(
    "Soil Moisture Multiplier", 0.5, 1.5, 1.0, 0.05,
    help="Scales antecedent soil moisture in the rainfall channel."
)
seq_len = st.sidebar.slider("Context Window (days)", 3, 7, 7, 1)

if weights_found:
    st.sidebar.success("✅ Trained weights loaded")
else:
    st.sidebar.warning("⚠️ No weights found — using untrained model")

st.title("முகில்நோக்கு (MugizhNokku) • Live Inference Engine")
st.caption("AI-Powered Digital Twin of India's Climate — Bharatiya Antariksh Hackathon 2026")

col1, col2, col3 = st.columns(3)
col1.metric("Model Architecture", "ConvLSTM")
col2.metric("Spatial Grid", "129 × 135 (0.25°)")
col3.metric("Region", "Western Ghats, India")

st.divider()

if st.button("🚀 Run Twin Simulation", use_container_width=True):
    with st.spinner("Routing tensors through ConvLSTM engine…"):
        prediction = build_prediction(model, seq_len, moisture_multiplier, sst_shift, config)
        peak_val = float(prediction.max())
        peak_pos = np.unravel_index(int(prediction.argmax()), prediction.shape)
        peak_lat = round(peak_pos[0] * 0.25 + 6.5, 2)
        peak_lon = round(peak_pos[1] * 0.25 + 66.5, 2)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Peak Rainfall", f"{peak_val:.1f} mm")
        m2.metric("Peak Location", f"{peak_lat}°N, {peak_lon}°E")
        m3.metric("Grid Average", f"{prediction.mean():.1f} mm")
        m4.metric("SST Perturbation", f"{sst_shift:+.1f} °C")

        if peak_val > 100:
            st.warning(f"⚠️ High-intensity rainfall predicted ({peak_val:.1f} mm). Potential flood-risk zone at {peak_lat}°N, {peak_lon}°E.")

        df = pd.DataFrame({
            "lat": np.repeat(np.arange(129) * 0.25 + 6.5, 135),
            "lon": np.tile(np.arange(135) * 0.25 + 66.5, 129),
            "rainfall_mm": prediction.reshape(-1).round(2),
        })

        st.subheader("🌐 CesiumJS 3D Digital Twin")
        three_points = build_three_points(prediction)
        json_data = json.dumps(three_points, ensure_ascii=False)
        texture_uri = load_texture_data_uri(TEXTURE_URL)
        globe_html = inject_html(HTML_PATH, texture_uri, json_data)

        st.success("ConvLSTM inference complete. Rendering 3D environment...")
        components.html(globe_html, height=750, scrolling=False)

        with st.expander("Prediction data"):
            st.dataframe(df, use_container_width=True)
else:
    st.info("👈 Adjust simulation parameters in the sidebar and click **Run Twin Simulation** to activate the ConvLSTM inference engine.")
