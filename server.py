import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import numpy as np
from pathlib import Path
from model import DigitalTwinPredictor

app = FastAPI(title="MugizhNokku Digital Twin API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_BASE = Path(__file__).resolve().parent
WEIGHTS_PATH = _BASE / 'models' / 'mugizhnokku_best.pth'

print("Loading PyTorch model into memory...")
model = DigitalTwinPredictor(input_channels=3, hidden_channels=32, out_channels=1)
try:
    if WEIGHTS_PATH.exists():
        state = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        print("✅ Weights loaded successfully!")
    else:
        print("⚠️ No weights found, using untrained model.")
except Exception as e:
    print(f"⚠️ Error loading weights: {e}")
model.eval()

@app.get("/api/predict")
def predict():
    """Generates T+1 inference and strictly returns non-zero rainfall coordinates."""
    # Simulate an input tensor (Batch=1, Seq=7, Channels=3, H=129, W=135)
    input_tensor = torch.randn(1, 7, 3, 129, 135)
    
    with torch.no_grad():
        prediction = model(input_tensor)
        
    lat_idx, lon_idx = np.indices((129, 135))
    rf_flat = prediction.flatten().round(2).numpy()
    
    data = []
    # Optimization: Filter empty areas to heavily reduce JSON payload size and boost browser 3D FPS
    for lat, lon, rf in zip((lat_idx.flatten() * 0.25 + 6.5).round(3),
                            (lon_idx.flatten() * 0.25 + 66.5).round(3),
                            rf_flat):
        if rf > 5.0:  # Threshold filter
            r = int(np.clip(255 - rf * 1.2, 0, 255))
            b = int(np.clip(rf * 1.2, 0, 255))
            data.append({
                "lat": float(lat),
                "lon": float(lon),
                "rain": float(rf),
                "h": float(rf * 3000),  # Scale for visual cylinder height
                "r": r,
                "g": 80,
                "b": b
            })
            
    return {"data": data}

# Mount the static web UI natively
web_dir = _BASE / "web_ui" / "public"
web_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
