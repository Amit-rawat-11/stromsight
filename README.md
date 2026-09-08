# StormSight: Physics-Aware PyTorch AI for Cyclone Analysis & Early Warning

StormSight is a **PyTorch deep-learning research prototype** for multi-satellite cyclone analysis, intensity classification, track forecasting, and rapid intensification prediction.

---

## 1. System Architecture

```
                       MULTI-SATELLITE OBSERVATIONS (ASYNCHRONOUS DATA)
        INSAT-3DR (10:00 UTC)   Meteosat-11 (10:20 UTC)   NOAA-20 (10:10 UTC)
          Δt = 20 min            Δt = 0 min (Ref)          Δt = 10 min
               │                        │                         │
               └────────────────────────┼─────────────────────────┘
                                        │
                         METEOROLOGICAL PHYSICAL DATA
             (Sea Surface Temp, Central Pressure, Wind Shear, Moisture)
                                        │
                                        ▼
                            PREPROCESSING & ENCODERS
               ┌────────────────────────┼────────────────────────┐
               ▼                        ▼                        ▼
         ImageEncoder          SpatioTemporalEncoder      PhysicsEncoder
     (PyTorch ResNet18 CNN)   (Learned Positional MLP   (PyTorch Physics MLP
        128-d Embedding      & Recency Decay w(Δt))    & Missing Variable Mask)
               │                        │                        │
               └────────────────────────┼────────────────────────┘
                                        │
                                        ▼
                                 FEATURE FUSION
                    (PyTorch Multimodal Gated Cross-Attention)
                                        │
                                        ▼
                              STORMSIGHT MODEL CORE
                       (Transformer Encoder Backbone)
                                        │
                                        ▼
                                 PREDICTION HEADS
       ┌───────────────────┬───────────────────┬───────────────────┐
       ▼                   ▼                   ▼                   ▼
Intensity Class      Confidence Score     Future Track       Rapid Intensification
  (6 Classes)      (Temperature Scaled)   (+6h to +48h)       (Binary Risk Head)
       │                   │                   │                   │
       └───────────────────┴─────────┬─────────┴───────────────────┘
                                     │
                                     ▼
                       CONFIDENCE & DECISION ROUTING
                       /                           \
           High Confidence (≥ 85%)          Low Confidence / Anomaly (< 85%)
                       │                           │
          Auto-Add to Training Pool          Human Meteorologist Review Queue
                       │                           │
                       └──────────────┬────────────┘
                                      │
                                      ▼
                             ACTIVE LEARNING LOOP
                  (Candidate Model Training & Promotion Checks)
```

---

## 2. Dependencies & Installation

### Requirements
- Python 3.9+
- PyTorch 2.0+
- FastAPI & Uvicorn
- NumPy & Scikit-Learn
- Pillow / Requests

### Installation Commands

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn scikit-learn numpy pillow requests
```

---

## 3. How to Run & Train

### A. Train PyTorch Model (`train.py`)
Executes PyTorch backpropagation training with AdamW, CosineAnnealingLR, early stopping, and storm-level split validation:

```bash
source .venv/bin/activate
python3 train.py --epochs 10 --batch_size 4
```

*Output Checkpoints*: Saves `checkpoints/checkpoint.pt` and `checkpoints/scaler.json`.

### B. Evaluate Held-Out Test Cyclones (`evaluate.py`)
Computes accuracy, macro F1, 24h track MAE (km), rapid intensification F1, and skill improvement vs persistence baseline:

```bash
python3 evaluate.py
```

### C. Launch FastAPI REST Server & Command Center UI (`server.py`)

```bash
python3 server.py
```

- REST API Docs: `http://localhost:8000/docs`
- Command Center UI: `http://localhost:8000`

---

## 4. Operational Modes: DEMO MODE vs REAL MODE

| Dimension | DEMO MODE | REAL MODE |
| :--- | :--- | :--- |
| **Cyclone Track Source** | Verified IBTrACS historical tracks (Amphan, Katrina, Fani, Tip, Tauktae, Haiyan) | Live WMO / IBTrACS real-time feeds |
| **Satellite Imagery** | Synthetic IR cloud top thermal intensity matrix tiles | Real GeoTIFF / NetCDF / NASA GIBS satellite tiles loaded via `SatelliteDatasetInterface.load_satellite_image()` |
| **Meteorological Inputs** | Real historical & live Open-Meteo REST API parameters | Real atmospheric reanalysis / Buoy sensors |
| **Model Weights** | PyTorch model trained on training split | Fine-tuned candidate model checkpoint (`checkpoint.pt`) |

---

## 5. System Limitations & Scientific Scope

1. **Remote Sensing Imagery Provider**: In `REAL_MODE`, real satellite imagery files (NetCDF4 / GeoTIFF) must be provided in `data/satellite_tiles/` or passed via path parameters to `SatelliteDatasetInterface.load_satellite_image()`.
2. **Track Forecast Coordinate Frames**: Track predictions are output as relative latitude/longitude displacement vectors $(\Delta\text{lat}, \Delta\text{lon})$ from the current storm center.
3. **Calibrated Confidence**: Confidence scores are calculated using Temperature Scaling ($T=1.25$) on validation split logits. Epistemic uncertainty is represented via normalized entropy over intensity class probability distributions.
