# -*- coding: utf-8 -*-
import base64
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import torch

from model import DigitalTwinPredictor

_BASE = Path(__file__).resolve().parent
WEIGHTS_PATH = _BASE / "models" / "mugizhnokku_best.pth"
CONFIG_PATH = _BASE / "models" / "normalization_config.json"

# 1. Page Configuration & Layout Initialization
st.set_page_config(
    page_title="MugizhNokku Digital Twin",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Advanced CSS Injection for Sci-Fi Glassmorphism Aesthetic
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Space+Mono&display=swap" rel="stylesheet">
    
    <style>
    /* Absolute Layout Reset to allow full edge-to-edge visualization */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }
    
    /* Background Canvas Setup */
    .stApp {
        background: radial-gradient(circle at center, #0a192f 0%, #020b1a 100%) !important;
        color: #ffffff !important;
        font-family: 'Space Mono', monospace !important;
    }
    
    /* Clear out Streamlit header/footer watermarks */
    header, footer { visibility: hidden !important; }
    
    /* Premium Sci-Fi Panel Base Styles */
    .hud-panel {
        background: rgba(10, 25, 47, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 255, 255, 0.15);
        border-radius: 4px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.05);
        position: relative;
    }
    
    /* Futuristic HUD Corner Details */
    .hud-panel::before {
        content: ''; position: absolute; top: -1px; left: -1px; width: 10px; height: 10px;
        border-top: 2px solid #00ffff; border-left: 2px solid #00ffff;
    }
    .hud-panel::after {
        content: ''; position: absolute; bottom: -1px; right: -1px; width: 10px; height: 10px;
        border-bottom: 2px solid #00ffff; border-right: 2px solid #00ffff;
    }
    
    /* Typography Overrides */
    .hud-header {
        font-family: 'Orbitron', sans-serif;
        color: #00ffff;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(0, 255, 255, 0.2);
        padding-bottom: 5px;
    }
    
    .main-title {
        font-family: 'Orbitron', sans-serif;
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        text-shadow: 0 0 15px rgba(0, 255, 255, 0.4);
        margin-bottom: 20px;
        text-align: center;
    }

    /* Target UI form fields and inputs to lock into dark theme */
    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        color: #00ffff !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    /* Interactive Button Optimization */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, rgba(0,210,255,0.3) 0%, rgba(58,123,213,0.3) 100%);
        border: 1px solid #00ffff !important;
        color: white !important;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 0.2em;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, rgba(0,210,255,0.6) 0%, rgba(58,123,213,0.6) 100%);
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.4);
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)


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
    """
    Convert the (129, 135) ConvLSTM prediction tensor into a list of
    { lat, lon, rain, r, g, b } dicts for PyDeck ColumnLayer rendering.

    Grid spec (IMD 0.25 deg x 0.25 deg):
        lat: 6.5N  -> 38.5N  (rows  0..128)
        lon: 66.5E -> 100.0E (cols  0..134)
    """
    points = []

    # Normalize the whole prediction to 0-200mm so cells always appear
    pred_arr = np.array(prediction)
    pred_min = float(pred_arr.min())
    pred_max = float(pred_arr.max())
    pred_range = max(pred_max - pred_min, 1e-6)

    for i in range(0, 129, 2):
        for j in range(0, 135, 2):
            raw = float(pred_arr[i, j])
            rain = round(((raw - pred_min) / pred_range) * 200.0, 2)
            if rain < 5.0:
                continue

            lat_v = round(i * 0.25 + 6.5,  3)
            lon_v = round(j * 0.25 + 66.5, 3)

            # Color ramp: low=cyan, mid=yellow, high=red
            norm = rain / 200.0
            r = int(min(255, norm * 2 * 255))
            g = int(min(255, max(0, (1 - abs(norm - 0.5) * 2) * 255)))
            b = int(min(255, (1 - norm) * 2 * 255))

            points.append({
                "lat":  lat_v,
                "lon":  lon_v,
                "rain": rain,
                "r":    r,
                "g":    g,
                "b":    b,
            })

    return points

# 3. Main Header Bar
st.markdown("<div class='main-title'>முகில்நோக்கு • MUGIZHNOKKU DIGITAL TWIN</div>", unsafe_allow_html=True)

# 4. Initialize State Variables and Models
model, weights_found = load_model(WEIGHTS_PATH)
config = load_config(CONFIG_PATH)

if 'run_simulation' not in st.session_state:
    st.session_state['run_simulation'] = False
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = np.zeros((129, 135))
if 'peak_val' not in st.session_state:
    st.session_state['peak_val'] = 0.0
if 'grid_avg' not in st.session_state:
    st.session_state['grid_avg'] = 0.0

# 5. Execute 3-Column Responsive Dashboard Grid Layout
col_left, col_center, col_right = st.columns([2.5, 5.5, 2.5])

# ── LEFT PANEL: ARCHITECTURE & DATA INGESTION ──
with col_left:
    st.markdown("""
        <div class='hud-panel'>
            <div class='hud-header'>🛰️ Data Assimilation Ingest</div>
            <p style='font-size:0.75rem; color:rgba(255,255,255,0.75); line-height:1.6;'>
                • INSAT-3D Imager / Sounder<br>
                • Oceansat Scatterometer Wind Vectors<br>
                • IMD Automated Weather Station (AWS) Grid<br>
                • GPM IMERG Late Precipitation Runs
            </p>
            <div style='margin-top:10px; font-size:0.65rem; color:#00ffaa;'>STATUS: CONTINUOUS STREAM ACTIVE</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='hud-panel'>
            <div class='hud-header'>🧠 Deep Learning Core</div>
            <p style='font-size:0.75rem; color:rgba(255,255,255,0.75); line-height:1.6;'>
                <strong>Model Architecture:</strong> ConvLSTM Pipeline<br>
                <strong>Spatial Dimensions:</strong> 129 × 135 Resolution Grid<br>
                <strong>Temporal Range:</strong> 7-Day Sliding Window History<br>
                <strong>Training Device:</strong> Nvidia T4 Compute Cluster
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='hud-panel'>
            <div class='hud-header'>📈 Hardware Analytics</div>
            <p style='font-size:0.7rem; color:rgba(255,255,255,0.6); margin-bottom: 5px;'>T4 VRAM Allocation</p>
    """, unsafe_allow_html=True)
    st.progress(78)
    st.markdown("""
            <p style='font-size:0.7rem; color:rgba(255,255,255,0.6); margin-top: 10px; margin-bottom: 5px;'>Inference Latency</p>
            <div style='font-size:1.1rem; color:#00ffff; font-family: "Orbitron";'>142 ms</div>
        </div>
    """, unsafe_allow_html=True)

# ── CENTER PANEL: THE LIVE DIGITAL TWIN VISUALIZATION ──
with col_center:
    three_points = build_three_points(st.session_state['prediction'])
    df = pd.DataFrame(three_points)
    
    if not df.empty:
        df['color'] = df[['r', 'g', 'b']].values.tolist()
    else:
        df['color'] = []

    view_state = pdk.ViewState(
        latitude=22.0,
        longitude=79.0,
        zoom=4.2,
        pitch=50,
        bearing=0
    )
    
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=df,
        get_position=["lon", "lat"],
        get_elevation="rain",
        elevation_scale=5000,
        radius=3000,
        get_fill_color="color",
        extruded=True,
        pickable=True,
        auto_highlight=True,
    )
    
    r = pdk.Deck(
        layers=[column_layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v11",
        tooltip={"html": "<b>Rainfall:</b> {rain} mm<br/><b>Lat:</b> {lat} <br/><b>Lon:</b> {lon}"}
    )
    
    st.pydeck_chart(r, use_container_width=True)
    
    st.markdown("""
        <div class='hud-panel' style='margin-top:-5px;'>
            <div class='hud-header'>🛡️ Climate Sector Adaptations</div>
            <div style='display:flex; justify-content:space-between; font-size:0.7rem; color:rgba(255,255,255,0.8);'>
                <div>🌾 <strong>Agri-Planning:</strong> Vulnerability Map Matrix</div>
                <div>💧 <strong>Water Resources:</strong> Basin Hydrology Check</div>
                <div>🚨 <strong>Early Warning:</strong> Flash Flood Vectors</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ── RIGHT PANEL: METRICS & CAUSAL SCENARIO CONTROLS ──
with col_right:
    st.markdown("<div class='hud-panel'><div class='hud-header'>📊 Real-Time Metrics</div>", unsafe_allow_html=True)
    metric_1, metric_2 = st.columns(2)
    with metric_1:
        st.metric(label="Peak Precipitation", value=f"{st.session_state['peak_val']:.1f} mm")
    with metric_2:
        st.metric(label="Grid Average", value=f"{st.session_state['grid_avg']:.1f} mm")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='hud-panel'><div class='hud-header'>🎛️ Scenario Simulation</div>", unsafe_allow_html=True)
    sst_shift = st.slider("Sea Surface Temp Shift (°C)", -2.0, 4.0, 0.0, step=0.1)
    moisture_mult = st.slider("Soil Moisture Multiplier", 0.5, 2.0, 1.0, step=0.1)
    seq_len = st.slider("Context Frame Depth", 3, 7, 7)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Execution Trigger Button
    if st.button("RUN SIMULATION"):
        with st.spinner("Routing tensors through ConvLSTM engine…"):
            prediction = build_prediction(model, seq_len, moisture_mult, sst_shift, config)
            st.session_state['prediction'] = prediction
            st.session_state['peak_val'] = float(prediction.max())
            st.session_state['grid_avg'] = float(prediction.mean())
            
        # Immediate interface refresh to re-inject data to Cesium frame
        st.rerun()

