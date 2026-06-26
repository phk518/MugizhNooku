# முகில்நோக்கு — MugizhNokku
### AI-Powered Digital Twin of India's Climate
**PS-5 · Bharatiya Antariksh Hackathon 2026**

> *முகில்நோக்கு* (Tamil: "Cloud Watcher") is a spatiotemporal deep learning system that builds a real-time, interactive **Digital Twin of India's regional climate**, fusing IMD gridded meteorological data with ISRO satellite observations to generate T+1 day rainfall and temperature forecasts over the Western Ghats.

---

## 🏆 Rubric Coverage

| Criterion | Implementation |
|---|---|
| **Problem Understanding & Clarity** | ConvLSTM predicts T+1 rainfall over the 129×135 Western Ghats grid, addressing the data fusion gap between IMD 0.25° rainfall and INSAT-3DR SST fields |
| **Data Usage & Pre-processing** | Binary IMD `.grd` parsing, bilinear spatial interpolation to harmonize 1.0°→0.25° grids, MinMax normalization with exportable JSON config |
| **Model Development & Technical Approach** | Spatiotemporal ConvLSTM with dynamic channel detection (3→4 channels as INSAT data is added), Adam optimizer, PyTorch on CUDA |
| **Prediction Performance & Validation** | Contiguous Block Split with 7-day temporal buffer (zero data leakage), spatial RMSE & MAE logged per epoch |
| **Digital Twin Concept Implementation** | End-to-end pipeline: Download → Parse → Interpolate → Normalize → Train → Checkpoint → Infer → Visualize |
| **Visualization & User Interface** | Streamlit dashboard with CesiumJS 3D globe + PyDeck column heatmap, hover tooltips, flood-risk alerts |
| **Innovation & Creativity** | Physics-grounded What-If sliders (SST shift, Soil Moisture Multiplier) that perturb input tensors before inference; automatic mock data fallback guaranteeing 100% demo uptime |
| **Presentation & Communication** | Modular codebase, documented PRD, full Git history, reproducible Colab notebook |

---

## 🗂️ Project Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    COLAB T4 GPU (Cloud)                  │
│  download_sample.py → data_parser.py → preprocessing.py  │
│         ↓                                                │
│  data_utils.py  →  dataset.py  →  model.py  →  train.py  │
│         ↓                                                │
│  mugizhnokku_best.pth  +  normalization_config.json      │
└─────────────────────────┬────────────────────────────────┘
                          │ Google Drive sync
┌─────────────────────────▼────────────────────────────────┐
│               LOCAL MACHINE (Dashboard)                  │
│              app.py  →  Streamlit + CesiumJS             │
└──────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option A: Google Colab (Recommended for Training)
1. Open `colab_runner.ipynb` in [Google Colab](https://colab.research.google.com/) with **GPU runtime**.
2. Run all cells in order.
3. Cell 1 clones the repo. Cell 2 mounts Drive. Cell 4 trains the model.

### Option B: Local (Dashboard Only)
```bash
git clone https://github.com/phk518/MugizhNooku.git
cd MugizhNooku
pip install -r requirements.txt
streamlit run app.py
```

---

## 📡 Data Sources

| Dataset | Source | Resolution | Variables |
|---|---|---|---|
| IMD Gridded Rainfall | [IMD Pune](https://imdpune.gov.in) | 0.25° × 0.25° | Daily rainfall (mm) |
| IMD Temperature | [IMD Pune](https://imdpune.gov.in) | 1.0° × 1.0° | Min/Max Temp (°C) |
| INSAT-3DR LST *(planned)* | [MOSDAC](https://mosdac.gov.in) | 4 km | Land Surface Temperature |

---

## 🧠 Model: ConvLSTM

The `DigitalTwinPredictor` stacks a `ConvLSTMCell` with a final `1×1 Conv2D` head.

- **Input:** `(Batch, Sequence=7, Channels=3, Height=129, Width=135)`
- **Output:** `(Batch, 1, 129, 135)` — T+1 rainfall prediction grid
- **Loss:** MSELoss | **Optimizer:** Adam (lr=1e-3)
- **Validation:** Spatial RMSE + MAE | **Split:** Contiguous 80/20 with 7-day buffer

---

## 📁 File Structure

```
MugizhNokku/
├── app.py              # Streamlit dashboard (CesiumJS + PyDeck)
├── model.py            # ConvLSTM architecture
├── train.py            # Training loop + validation + checkpointing
├── dataset.py          # PyTorch sliding-window Dataset
├── data_utils.py       # Master data orchestrator + normalizer
├── data_parser.py      # IMD binary (.grd) parser
├── preprocessing.py    # Bilinear interpolation + MinMax scaling
├── download_sample.py  # Deterministic IMD downloader + mock fallback
├── colab_runner.ipynb  # Google Colab orchestration notebook
├── requirements.txt
└── models/             # Gitignored — stored on Google Drive
    ├── mugizhnokku_best.pth
    └── normalization_config.json
```

---

## 👨‍🔬 Mentors

- Dr. Kandula V Subrahmanyam · Sci/Eng-SF, NRSC/ISRO
- Mr. Syed Shadab · Sci/Eng-SF, NRSC/ISRO
- Mr. C. Sarat · Sci/Eng-SC, NRSC/ISRO

---

*Built for PS-5 · Bharatiya Antariksh Hackathon 2026*
