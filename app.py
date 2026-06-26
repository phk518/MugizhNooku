import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk

st.set_page_config(layout="wide", page_title="முகில்நோக்கு Dashboard")
st.title("முகில்நோக்கு (MugizhNokku) • Climate Digital Twin")

st.sidebar.header("What-If Simulation Parameters")
temp_delta = st.sidebar.slider("Sea Surface Temp Shift (°C)", -3.0, 3.0, 0.0, 0.1)
moisture_scaling = st.sidebar.slider("Soil Moisture Multiplier", 0.5, 1.5, 1.0, 0.1)

if st.sidebar.button("Run Twin Simulation Model"):
    st.sidebar.success("Inference processing routed to backend...")
    
    # Generate mock coordinates over India's Western Ghats region for rendering verification
    lats = np.random.uniform(8.0, 20.0, 500)
    lons = np.random.uniform(72.5, 77.5, 500)
    # Simulate mathematical response to slider variables
    base_rain = np.random.uniform(10, 150, 500) * moisture_scaling + (temp_delta * 12)
    
    df = pd.DataFrame({'lat': lats, 'lon': lons, 'rainfall': np.clip(base_rain, 0, None)})
    
    # Generate 3D Spatial Grid Columns using PyDeck
    layer = pdk.Layer(
        'ColumnLayer',
        data=df,
        get_position='[lon, lat]',
        get_elevation='rainfall',
        elevation_scale=500,
        radius=6000,
        get_fill_color=['rainfall * 1.5', 50, 200, 160],
        pickable=True,
        extruded=True,
    )
    
    view_state = pdk.ViewState(latitude=14.0, longitude=75.0, zoom=5, pitch=50)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
