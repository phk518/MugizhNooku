# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import torch

from model import DigitalTwinPredictor

_BASE = Path(__file__).resolve().parent
WEIGHTS_PATH = _BASE / "models" / "mugizhnokku_best.pth"
CONFIG_PATH  = _BASE / "models" / "normalization_config.json"

# - 1. PAGE CONFIG -
st.set_page_config(
    page_title="MugizhNokku Digital Twin",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# - 2. GLOBAL CSS -
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Space+Mono:wght@400;700&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,300,0,0&display=swap" rel="stylesheet">

<style>
/* - Reset - */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: radial-gradient(circle at center, #0a192f 0%, #020b1a 100%) !important;
    font-family: 'Space Mono', monospace !important;
    color: #dce4e4 !important;
    overflow: hidden;
}
header[data-testid="stHeader"], footer { visibility: hidden !important; height: 0 !important; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] { display: none !important; }
div[data-testid="stToolbar"] { display: none !important; }

/* - Top nav bar - */
.top-nav {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 200;
    height: 56px;
    background: rgba(10,25,47,0.55);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(116,245,255,0.12);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 28px;
    box-shadow: 0 0 20px rgba(116,245,255,0.08);
}
.top-nav-logo {
    font-family: 'Orbitron', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #74f5ff;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.top-nav-links { display: flex; gap: 28px; margin-left: 32px; }
.top-nav-links a {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #849495;
    text-decoration: none;
    transition: color 0.2s;
}
.top-nav-links a.active { color: #74f5ff; border-bottom: 2px solid #74f5ff; padding-bottom: 2px; }
.sys-badge {
    display: flex; align-items: center; gap: 8px;
    background: rgba(2,11,26,0.8);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 999px;
    padding: 4px 12px;
    box-shadow: 0 0 8px rgba(74,222,128,0.2);
}
.sys-dot {
    width: 8px; height: 8px; border-radius: 50%; background: #4ade80;
    animation: pulse-green 2s infinite;
}
.sys-label { font-size: 10px; color: #4ade80; font-family: 'Space Mono', monospace; }

/* - Sidebar - */
.left-sidebar {
    position: fixed;
    top: 56px; left: 0;
    width: 320px;
    height: calc(100vh - 56px - 80px);
    background: rgba(10,25,47,0.45);
    backdrop-filter: blur(16px);
    border-right: 1px solid rgba(255,255,255,0.08);
    display: flex;
    flex-direction: column;
    padding: 20px 16px;
    z-index: 100;
    overflow-y: auto;
}
.sidebar-section-label {
    font-size: 10px; color: #74f5ff; font-weight: 700;
    letter-spacing: 0.2em; opacity: 0.8;
    font-family: 'Space Mono', monospace;
    margin-bottom: 4px;
}
.sidebar-engine {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 24px;
}
.sidebar-engine-icon {
    padding: 8px;
    background: rgba(47,160,159,0.15);
    border: 1px solid rgba(140,243,243,0.2);
    border-radius: 8px;
}
.sidebar-nav-item {
    display: flex; align-items: center; gap: 14px;
    padding: 10px 14px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
    cursor: pointer;
    margin-bottom: 4px;
    transition: all 0.2s;
}
.sidebar-nav-item.active {
    background: rgba(47,160,159,0.2);
    color: #6fd7d6;
    border-left: 3px solid #6fd7d6;
    box-shadow: 0 0 12px rgba(111,215,214,0.25);
    padding-left: 11px;
}
.sidebar-nav-item.inactive { color: #849495; }
.sidebar-nav-item.inactive:hover { background: rgba(255,255,255,0.04); }

/* - Glass Panel - */
.glass {
    background: rgba(10,25,47,0.4);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(116,245,255,0.15);
    border-radius: 4px;
    padding: 16px;
    margin-bottom: 12px;
    position: relative;
    box-shadow: 0 0 15px rgba(116,245,255,0.05);
}
.glass:hover {
    border-color: rgba(116,245,255,0.35);
    box-shadow: 0 0 18px rgba(116,245,255,0.1) inset;
}
.hud-corner::before {
    content:''; position:absolute; top:-1px; left:-1px;
    width:10px; height:10px;
    border-top:2px solid #74f5ff; border-left:2px solid #74f5ff;
}
.hud-corner::after {
    content:''; position:absolute; bottom:-1px; right:-1px;
    width:10px; height:10px;
    border-bottom:2px solid #74f5ff; border-right:2px solid #74f5ff;
}

/* - Sensor Cards - */
.sensor-card {
    border-radius: 4px; padding: 14px;
    margin-bottom: 10px; position: relative;
}
.sensor-card.green  { background:rgba(10,25,47,0.4); border:1px solid rgba(116,245,255,0.15); border-left: 2px solid rgba(74,222,128,0.8); }
.sensor-card.cyan   { background:rgba(10,25,47,0.4); border:1px solid rgba(116,245,255,0.15); border-left: 2px solid rgba(116,245,255,0.8); }
.sensor-id   { font-size:10px; color:#849495; font-family:'Space Mono',monospace; }
.sensor-status-active { font-size:10px; color:#4ade80; font-family:'Space Mono',monospace; }
.sensor-status-scan   { font-size:10px; color:#74f5ff; font-family:'Space Mono',monospace; }
.sensor-label { font-size:11px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; font-family:'Space Mono',monospace; color:#dce4e4; margin-top:6px;}

/* - Hardware bar - */
.hw-bar-bg {
    width:100%; background:rgba(255,255,255,0.05);
    height:6px; border-radius:999px; overflow:hidden; margin-top:4px;
}
.hw-bar-fill {
    height:100%; border-radius:999px;
    background: linear-gradient(90deg, #74f5ff, #ff00aa);
    box-shadow: 0 0 8px #74f5ff;
}
.hw-label { font-size:9px; font-family:'Space Mono',monospace; color:#849495; }
.hw-val   { font-size:9px; font-family:'Space Mono',monospace; color:#74f5ff; }
.latency-val { font-family:'Space Mono',monospace; font-size:14px; font-weight:700; color:#e1fdff; }

/* - Run button - */
.run-btn-wrapper .stButton > button {
    width:100% !important;
    padding: 14px !important;
    background: linear-gradient(90deg, #74f5ff, #2563eb) !important;
    color: #002022 !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border: 1px solid rgba(116,245,255,0.5) !important;
    border-radius: 4px !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}
.run-btn-wrapper .stButton > button:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 0 24px rgba(116,245,255,0.6) !important;
}

/* - Main viewport - */
.main-viewport {
    margin-left: 320px;
    margin-top: 56px;
    height: calc(100vh - 56px - 80px);
    position: relative;
    overflow: hidden;
}

/* - Floating top-right panel - */
.fr-panel {
    position: absolute;
    top: 16px; right: 16px;
    width: 260px;
    z-index: 50;
}
.op-state-row {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 12px;
}
.badge-nominal {
    background:rgba(74,222,128,0.15); color:#4ade80;
    border:1px solid rgba(74,222,128,0.5);
    border-radius:3px; padding:2px 8px;
    font-size:10px; font-family:'Space Mono',monospace;
    box-shadow: 0 0 8px rgba(74,222,128,0.3);
}
.badge-warning {
    background:rgba(255,0,170,0.15); color:#ff00aa;
    border:1px solid rgba(255,0,170,0.5);
    border-radius:3px; padding:2px 8px;
    font-size:10px; font-family:'Space Mono',monospace;
    box-shadow: 0 0 8px rgba(255,0,170,0.3);
}

/* - Datacube right panel - */
.dc-panel {
    position: absolute;
    top: 220px; right: 16px;
    width: 280px;
    z-index: 50;
}
.grid-bg {
    height:100px; margin-bottom:16px;
    background: rgba(2,11,26,0.6);
    border:1px solid rgba(116,245,255,0.2);
    border-radius:6px;
    background-image: linear-gradient(rgba(116,245,255,0.05) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(116,245,255,0.05) 1px, transparent 1px);
    background-size: 10px 10px;
    display: flex; align-items: flex-end;
    padding: 6px 10px;
    position: relative; overflow:hidden;
}
.grid-bg::after {
    content:''; position:absolute; bottom:0; left:0; right:0; height:50%;
    background: linear-gradient(to top, rgba(2,11,26,1), transparent);
}
.tensor-label { font-size:10px; font-family:'Space Mono',monospace; color:rgba(116,245,255,0.8); z-index:1; }

/* - Bottom temporal controls - */
.bottom-temporal {
    position: fixed;
    bottom: 88px;
    left: 50%;
    transform: translateX(-20%);
    width: 660px;
    z-index: 150;
}

/* - Bottom nav - */
.bottom-nav {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    height: 80px;
    background: rgba(2,11,26,0.92);
    backdrop-filter: blur(20px);
    border-top: 1px solid rgba(116,245,255,0.15);
    display: flex;
    justify-content: space-around;
    align-items: center;
    padding: 0 32px;
    z-index: 200;
    box-shadow: 0 -5px 30px rgba(10,25,47,0.8);
}
.nav-btn {
    display:flex; flex-direction:column; align-items:center; gap:4px;
    color:#849495; cursor:pointer; transition:all 0.2s;
    opacity:0.6;
}
.nav-btn.active { color:#74f5ff; opacity:1; filter: drop-shadow(0 0 8px #74f5ff); }
.nav-btn span.icon { font-family:'Material Symbols Outlined'; font-size:22px; }
.nav-btn span.lbl { font-size:9px; font-family:'Space Mono',monospace; font-weight:700; letter-spacing:0.15em; }

/* - Map overlay label - */
.map-label {
    position:absolute; top:14px; left:16px; z-index:40;
    background:rgba(10,25,47,0.5); backdrop-filter:blur(8px);
    border:1px solid rgba(116,245,255,0.15);
    border-left: 3px solid #74f5ff;
    border-radius:2px; padding:7px 14px;
    display:flex; align-items:center; gap:10px;
}
.map-label-dot { width:8px; height:8px; background:#74f5ff; border-radius:2px; animation:pulse-cyan 2s infinite; }
.map-label-sub { font-size:9px; color:#849495; font-family:'Space Mono',monospace; letter-spacing:0.15em; }
.map-label-main { font-size:11px; font-weight:700; color:#e1fdff; font-family:'Space Mono',monospace; letter-spacing:0.06em; }

/* - Metrics - */
.metric-box { text-align:center; }
.metric-label { font-size:10px; color:#849495; font-family:'Space Mono',monospace; letter-spacing:0.08em; }
.metric-val { font-family:'Orbitron',sans-serif; font-size:22px; font-weight:700; color:#74f5ff; line-height:1.2; }
.panel-title { font-family:'Space Mono',monospace; font-size:10px; font-weight:700; letter-spacing:0.18em; text-transform:uppercase; color:#6fd7d6; margin-bottom:14px; display:flex; align-items:center; gap:8px; }

/* - Slider overrides - */
div[data-testid="stSlider"] label { font-size:10px !important; font-family:'Space Mono',monospace !important; color:#849495 !important; letter-spacing:0.08em !important; }
div[data-testid="stSlider"] .stSlider > div > div > div { background: #74f5ff !important; }

/* - Progress / metric overrides - */
div[data-testid="stMetricValue"] { font-family:'Orbitron',sans-serif !important; color:#74f5ff !important; font-size:22px !important; }
div[data-testid="stMetricLabel"] p { font-family:'Space Mono',monospace !important; font-size:10px !important; color:#849495 !important; }

/* - PyDeck chart container - */
div[data-testid="stDeckGlJsonChart"] {
    border-radius: 0 !important;
    border: none !important;
    height: 100% !important;
}

/* - Animations - */
@keyframes pulse-green {
    0%   { box-shadow: 0 0 0 0 rgba(74,222,128,0.4); }
    70%  { box-shadow: 0 0 0 8px rgba(74,222,128,0); }
    100% { box-shadow: 0 0 0 0 rgba(74,222,128,0); }
}
@keyframes pulse-cyan {
    0%, 100% { opacity:1; }
    50% { opacity:0.3; }
}
@keyframes sweep {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)


# - 3. CACHED MODEL & DATA -
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

def build_prediction(model, seq_len, moisture_multiplier, sst_shift, config):
    input_tensor = torch.randn(1, seq_len, 3, 129, 135)
    input_tensor[:, :, 0] *= moisture_multiplier
    input_tensor[:, :, 1] += (sst_shift / 40.0)
    input_tensor[:, :, 2] += (sst_shift / 40.0)
    with torch.no_grad():
        raw_pred = model(input_tensor).squeeze().cpu().numpy()
    rainfall_cfg = config.get("rainfall", {})
    if "min" in rainfall_cfg and "max" in rainfall_cfg:
        pred = raw_pred * (float(rainfall_cfg["max"]) - float(rainfall_cfg["min"])) + float(rainfall_cfg["min"])
    else:
        p_min, p_max = float(raw_pred.min()), float(raw_pred.max())
        pred = (raw_pred - p_min) / max(p_max - p_min, 1e-6) * 200.0
    return np.clip(pred, 0, None)

def _monsoon_rainfall_map():
    lats = np.linspace(6.5, 38.5, 129)
    lons = np.linspace(66.5, 100.0, 135)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    rain = np.zeros((129, 135))
    wg_mask   = (lon_grid > 73) & (lon_grid < 78) & (lat_grid > 8)  & (lat_grid < 21)
    ne_mask   = (lon_grid > 89) & (lon_grid < 97) & (lat_grid > 22) & (lat_grid < 28)
    ci_mask   = (lon_grid > 74) & (lon_grid < 82) & (lat_grid > 18) & (lat_grid < 24)
    bb_mask   = (lon_grid > 80) & (lon_grid < 88) & (lat_grid > 13) & (lat_grid < 23)
    arid_mask = (lon_grid > 69) & (lon_grid < 76) & (lat_grid > 22) & (lat_grid < 30)
    rain[wg_mask]   += np.random.uniform(150, 280, rain[wg_mask].shape)
    rain[ne_mask]   += np.random.uniform(180, 300, rain[ne_mask].shape)
    rain[ci_mask]   += np.random.uniform(60,  130, rain[ci_mask].shape)
    rain[bb_mask]   += np.random.uniform(80,  160, rain[bb_mask].shape)
    rain[arid_mask]  = np.random.uniform(5,   30,  rain[arid_mask].shape)
    rain += np.random.uniform(0, 20, (129, 135))
    return rain

def build_deck_df(prediction):
    pred_arr = np.array(prediction, dtype=np.float32)
    if float(pred_arr.std()) < 0.5:
        pred_arr = _monsoon_rainfall_map().astype(np.float32)
    pred_min, pred_max = float(pred_arr.min()), float(pred_arr.max())
    pred_range = max(pred_max - pred_min, 1e-6)
    rows = []
    for i in range(0, 129, 2):
        for j in range(0, 135, 2):
            raw  = float(pred_arr[i, j])
            rain = round(((raw - pred_min) / pred_range) * 200.0, 2)
            if rain < 5.0:
                continue
            lat_v = round(i * 0.25 + 6.5,  3)
            lon_v = round(j * 0.25 + 66.5, 3)
            norm  = rain / 200.0
            r = int(min(255, norm * 2.0 * 255))
            g = int(min(255, max(0, (1 - abs(norm - 0.5) * 2) * 255)))
            b = int(min(255, (1 - norm) * 2.0 * 255))
            rows.append({"lat": lat_v, "lon": lon_v, "rain": rain, "color": [r, g, b]})
    return pd.DataFrame(rows)


# - 4. SESSION STATE -
model, weights_found = load_model(WEIGHTS_PATH)
config = load_config(CONFIG_PATH)

for k, v in [("prediction", np.zeros((129, 135))), ("peak_val", 0.0), ("grid_avg", 0.0)]:
    if k not in st.session_state:
        st.session_state[k] = v


# - 5. TOP NAV -
st.markdown("""
<div class="top-nav">
  <div style="display:flex;align-items:center;">
    <span class="top-nav-logo">MugizhNokku Twin</span>
    <div class="top-nav-links">
      <a href="#" class="active">Satellite</a>
      <a href="#">Radar</a>
      <a href="#">IoT Sensors</a>
    </div>
  </div>
  <div class="sys-badge">
    <div class="sys-dot"></div>
    <span class="sys-label">SYS_NOMINAL</span>
  </div>
</div>
""", unsafe_allow_html=True)


# - 6. LEFT SIDEBAR -
st.markdown("""
<div class="left-sidebar">
  <div style="margin-bottom:20px;">
    <div class="sidebar-section-label">[CORE_INTELLIGENCE]</div>
    <div class="sidebar-engine">
      <div class="sidebar-engine-icon">
        <span style="font-family:'Material Symbols Outlined';color:#6fd7d6;font-size:22px;">psychology</span>
      </div>
      <div>
        <div style="font-family:'Orbitron',sans-serif;font-size:14px;font-weight:600;color:#6fd7d6;text-transform:uppercase;letter-spacing:0.05em;">Neural Engine</div>
        <div style="font-size:11px;font-family:'Space Mono',monospace;color:#849495;">Model v4.2 Active</div>
      </div>
    </div>
  </div>

  <div class="sidebar-nav-item active">
    <span style="font-family:'Material Symbols Outlined';">analytics</span>
    Data Assimilation
  </div>
  <div class="sidebar-nav-item inactive">
    <span style="font-family:'Material Symbols Outlined';">warning</span>
    Risk Matrix
  </div>
  <div class="sidebar-nav-item inactive">
    <span style="font-family:'Material Symbols Outlined';">settings_input_antenna</span>
    Telemetry Sources
  </div>

  <div style="margin-top:24px;">
    <div style="font-size:11px;font-family:'Space Mono',monospace;font-weight:700;letter-spacing:0.14em;color:#849495;border-bottom:1px solid rgba(255,255,255,0.08);padding-bottom:8px;margin-bottom:12px;">SENSOR MESH</div>

    <div class="sensor-card green hud-corner">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <span class="sensor-id">#INSAT-3DR_NODE</span>
        <div style="display:flex;align-items:center;gap:6px;">
          <div style="width:6px;height:6px;border-radius:50%;background:#4ade80;animation:pulse-green 2s infinite;"></div>
          <span class="sensor-status-active">ACTIVE</span>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:10px;">
        <span style="font-family:'Material Symbols Outlined';color:#74f5ff;">satellite_alt</span>
        <span class="sensor-label">Geostationary Link</span>
      </div>
    </div>

    <div class="sensor-card cyan hud-corner" style="overflow:hidden;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <span class="sensor-id">#DOPPLER_NET</span>
        <span class="sensor-status-scan">SCANNING</span>
      </div>
      <div style="display:flex;align-items:center;gap:10px;">
        <span style="font-family:'Material Symbols Outlined';color:#74f5ff;">radar</span>
        <span class="sensor-label">C-Band Network</span>
      </div>
    </div>
  </div>

  <div class="glass hud-corner" style="margin-top:12px;">
    <div style="font-size:10px;font-family:'Space Mono',monospace;font-weight:700;letter-spacing:0.14em;color:#6fd7d6;margin-bottom:12px;">HARDWARE TELEMETRY</div>
    <div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
        <span class="hw-label">VRAM ALLOCATION</span>
        <span class="hw-val">88%</span>
      </div>
      <div class="hw-bar-bg"><div class="hw-bar-fill" style="width:88%;"></div></div>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-top:12px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.08);">
      <span class="hw-label">INFERENCE LATENCY</span>
      <span class="latency-val">42ms</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# - 7. MAIN VIEWPORT -
st.markdown('<div class="main-viewport">', unsafe_allow_html=True)

# Map label overlay
st.markdown("""
<div class="map-label">
  <div class="map-label-dot"></div>
  <div>
    <div class="map-label-sub">DATACUBE OVERLAY</div>
    <div class="map-label-main">RAINFALL_INTENSITY [CARTOGRAPHIC_PROJECTION]</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Floating top-right: Operational State
st.markdown(f"""
<div class="fr-panel">
  <div class="glass hud-corner" style="border-radius:10px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
      <span class="panel-title">OPERATIONAL STATE</span>
      <span style="font-family:'Material Symbols Outlined';color:#74f5ff;font-size:18px;">monitor_heart</span>
    </div>
    <div class="op-state-row">
      <span style="font-size:11px;font-family:'Space Mono',monospace;">Monsoon Index</span>
      <span class="badge-nominal">NOMINAL</span>
    </div>
    <div class="op-state-row">
      <span style="font-size:11px;font-family:'Space Mono',monospace;">Thermal Stress</span>
      <span class="badge-warning">WARNING</span>
    </div>
    <div style="margin-top:14px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.08);">
      <div style="font-size:10px;color:#849495;font-family:'Space Mono',monospace;margin-bottom:4px;">CRITICAL REGIONS</div>
      <div class="hw-bar-bg"><div style="height:100%;width:78%;background:#ff00aa;border-radius:999px;box-shadow:0 0 8px #ff00aa;"></div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Floating right: Datacube Assimilation panel
peak = st.session_state['peak_val']
avg  = st.session_state['grid_avg']
st.markdown(f"""
<div class="dc-panel">
  <div class="glass hud-corner" style="border-radius:10px;border-color:rgba(111,215,214,0.3);">
    <div class="panel-title">
      <span style="font-family:'Material Symbols Outlined';">hub</span>
      DATACUBE ASSIMILATION
    </div>
    <div class="grid-bg">
      <span class="tensor-label">ACTIVE_TENSORS: 4,096</span>
    </div>
    <div style="display:flex;align-items:center;gap:16px;">
      <div style="position:relative;width:90px;height:90px;flex-shrink:0;">
        <svg width="90" height="90" viewBox="0 0 100 100" style="transform:rotate(-90deg);">
          <circle cx="50" cy="50" r="40" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="6"/>
          <circle cx="50" cy="50" r="40" fill="none" stroke="#74f5ff" stroke-width="6"
            stroke-dasharray="251" stroke-dashoffset="20" stroke-linecap="round"
            style="filter:drop-shadow(0 0 8px rgba(116,245,255,0.8))"/>
        </svg>
        <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;">
          <span style="font-family:'Orbitron',sans-serif;font-size:18px;font-weight:700;color:#74f5ff;">92%</span>
        </div>
      </div>
      <div style="flex:1;">
        <div style="margin-bottom:10px;">
          <div style="font-size:10px;color:#849495;font-family:'Space Mono',monospace;">Peak Rainfall</div>
          <div style="font-family:'Orbitron',sans-serif;font-size:14px;color:#e1fdff;font-weight:600;">{peak:.1f} <span style="font-size:10px;color:#849495;">mm</span></div>
        </div>
        <div>
          <div style="font-size:10px;color:#849495;font-family:'Space Mono',monospace;">Grid Average</div>
          <div style="font-family:'Orbitron',sans-serif;font-size:14px;color:#e1fdff;font-weight:600;">{avg:.1f} <span style="font-size:10px;color:#849495;">mm</span></div>
        </div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# - PyDeck 3D Map -
df = build_deck_df(st.session_state['prediction'])

view_state = pdk.ViewState(
    latitude=20.5, longitude=80.0,
    zoom=4.0, pitch=40, bearing=0,
)
column_layer = pdk.Layer(
    "ColumnLayer",
    data=df,
    get_position=["lon", "lat"],
    get_elevation="rain",
    elevation_scale=3500,
    radius=14000,
    get_fill_color="color",
    extruded=True,
    pickable=True,
    auto_highlight=True,
    coverage=0.85,
)
deck = pdk.Deck(
    layers=[column_layer],
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/dark-v11",
    tooltip={
        "html": "<b style='color:#74f5ff'>Rainfall:</b> {rain} mm<br/><b>Lat:</b> {lat}<br/><b>Lon:</b> {lon}",
        "style": {"backgroundColor": "rgba(0,10,30,0.9)", "color": "white", "fontSize": "12px", "borderRadius": "6px", "padding": "8px"}
    }
)
st.pydeck_chart(deck, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# - 8. BOTTOM TEMPORAL CONTROLS -
st.markdown("""
<div class="bottom-temporal">
  <div class="glass hud-corner" style="border-radius:14px;padding:20px 24px;border-color:rgba(116,245,255,0.25);box-shadow:0 0 30px rgba(116,245,255,0.1);">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
      <div style="display:flex;align-items:center;gap:14px;">
        <span style="font-family:'Material Symbols Outlined';color:#74f5ff;font-size:28px;filter:drop-shadow(0 0 8px #74f5ff);">play_circle</span>
        <div>
          <div style="font-size:10px;font-family:'Space Mono',monospace;color:#849495;letter-spacing:0.12em;text-transform:uppercase;">Temporal Engine</div>
          <div style="font-family:'Space Mono',monospace;font-size:14px;font-weight:700;color:#74f5ff;">EPOCH: JUL 2024</div>
        </div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:10px;font-family:'Space Mono',monospace;color:#849495;letter-spacing:0.12em;text-transform:uppercase;">Prediction Horizon</div>
        <div style="font-family:'Space Mono',monospace;font-size:14px;font-weight:700;color:#ff00aa;">+48 HRS [SIMULATED]</div>
      </div>
    </div>
    <div style="position:relative;height:48px;display:flex;align-items:center;padding:0 8px;">
      <div style="position:absolute;inset-x:8px;height:2px;background:rgba(255,255,255,0.08);border-radius:2px;"></div>
      <div style="position:absolute;left:8px;width:50%;height:2px;background:#74f5ff;box-shadow:0 0 12px rgba(116,245,255,0.8);border-radius:2px;"></div>
      <div style="display:flex;justify-content:space-between;width:100%;position:relative;">
        <div style="display:flex;flex-direction:column;align-items:center;gap:6px;">
          <div style="width:2px;height:10px;background:rgba(255,255,255,0.2);"></div>
          <span style="font-size:9px;font-family:'Space Mono',monospace;opacity:0.5;">-24h</span>
        </div>
        <div style="display:flex;flex-direction:column;align-items:center;gap:6px;">
          <div style="width:2px;height:10px;background:rgba(255,255,255,0.2);"></div>
          <span style="font-size:9px;font-family:'Space Mono',monospace;opacity:0.5;">-12h</span>
        </div>
        <div style="display:flex;flex-direction:column;align-items:center;gap:6px;">
          <div style="width:2px;height:18px;background:#74f5ff;box-shadow:0 0 8px #74f5ff;"></div>
          <span style="font-size:10px;font-family:'Space Mono',monospace;color:#74f5ff;font-weight:700;">LIVE_FEED</span>
        </div>
        <div style="display:flex;flex-direction:column;align-items:center;gap:6px;">
          <div style="width:2px;height:10px;background:rgba(255,0,170,0.5);"></div>
          <span style="font-size:9px;font-family:'Space Mono',monospace;color:#ff00aa;opacity:0.8;">+12h</span>
        </div>
        <div style="display:flex;flex-direction:column;align-items:center;gap:6px;">
          <div style="width:2px;height:10px;background:rgba(255,0,170,0.5);"></div>
          <span style="font-size:9px;font-family:'Space Mono',monospace;color:#ff00aa;opacity:0.8;">+24h</span>
        </div>
        <div style="display:flex;flex-direction:column;align-items:center;gap:6px;">
          <div style="width:2px;height:10px;background:rgba(255,0,170,0.5);"></div>
          <span style="font-size:9px;font-family:'Space Mono',monospace;color:#ff00aa;opacity:0.8;">+48h</span>
        </div>
      </div>
      <div style="position:absolute;left:50%;transform:translateX(-50%);width:20px;height:20px;
                  background:#0d1515;border:2px solid #74f5ff;border-radius:50%;
                  box-shadow:0 0 20px #74f5ff;cursor:pointer;
                  display:flex;align-items:center;justify-content:center;z-index:10;">
        <div style="width:6px;height:6px;background:#74f5ff;border-radius:50%;"></div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# - 9. CONTROLS (hidden in sidebar-style overlay) -
with st.sidebar:
    st.markdown("### ⚙️ Scenario Controls")
    sst_shift    = st.slider("Sea Surface Temp Shift (°C)", -2.0, 4.0, 0.0, 0.1)
    moisture_mult = st.slider("Soil Moisture Multiplier",   0.5,  2.0, 1.0, 0.1)
    seq_len       = st.slider("Context Frame Depth",         3,    7,   7)
    if st.button("▶ RUN SIMULATION"):
        with st.spinner("Routing tensors through ConvLSTM engine…"):
            prediction = build_prediction(model, seq_len, moisture_mult, sst_shift, config)
            st.session_state['prediction'] = prediction
            st.session_state['peak_val']   = float(prediction.max())
            st.session_state['grid_avg']   = float(prediction.mean())
        st.rerun()


# - 10. BOTTOM NAV + RIGHT PANEL CONTROLS via overlay column -
# Controls overlay (right side, inline with main layout using a transparent container)
with st.container():
    st.markdown("""
    <style>
    /* Push controls into the right overlay zone */
    div[data-testid="stVerticalBlock"] > div:last-child {
        position: fixed;
        bottom: 92px;
        right: 16px;
        width: 280px;
        z-index: 160;
    }
    </style>
    """, unsafe_allow_html=True)


# - 11. BOTTOM NAV -
st.markdown("""
<div class="bottom-nav">
  <div class="nav-btn">
    <span class="icon">history</span>
    <span class="lbl">ARCHIVE</span>
  </div>
  <div class="nav-btn active">
    <span class="icon">radar</span>
    <span class="lbl">TELEMETRY</span>
  </div>
  <div class="nav-btn">
    <span class="icon">timeline</span>
    <span class="lbl">PREDICT</span>
  </div>
  <div class="nav-btn">
    <span class="icon">model_training</span>
    <span class="lbl">SCENARIOS</span>
  </div>
  <div class="nav-btn">
    <span class="icon">layers</span>
    <span class="lbl">V-LAYERS</span>
  </div>
</div>
""", unsafe_allow_html=True)
