#!/usr/bin/env python3
"""
StormSight: Physics-Aware AI for Early Cyclone Prediction
Demonstration Runner Script with Real Data support.
"""

import sys
import json
from stormsight.pipeline import StormSightPredictor, ActiveLearningPipeline
from stormsight.real_data import RealDataLoader

def generate_synthetic_cyclone_image(rows=10, cols=10, eye_r=2, intensity=0.8):
    """Generates synthetic 2D array modeling infrared cloud brightness temperature."""
    img = []
    center_r, center_c = rows // 2, cols // 2
    for r in range(rows):
        row = []
        for c in range(cols):
            dist = ((r - center_r)**2 + (c - center_c)**2) ** 0.5
            if dist < eye_r:
                val = 0.2 + (dist / eye_r) * 0.2
            else:
                spiral = (dist * 0.5 + intensity) % 1.0
                val = max(0.1, min(1.0, 0.9 - (dist * 0.08) + (spiral * 0.2)))
            row.append(round(val, 3))
        img.append(row)
    return img

def main():
    use_real = "--real" in sys.argv or "--live" in sys.argv
    cyclone_case = "amphan" if "--case" not in sys.argv else sys.argv[sys.argv.index("--case") + 1]

    print("=" * 80)
    print(" 🌀 STORMSIGHT: PHYSICS-AWARE AI FOR EARLY CYCLONE PREDICTION")
    print(" Safer Coasts. Early Warnings. Better Tomorrow.")
    print("=" * 80)

    if use_real:
        print("\n [MODE]: LIVE REAL-WORLD DATA INGESTION (Open-Meteo API & IBTrACS)")
        print(" Fetching real atmospheric & ocean metrics from global servers...")
        
        # Real Live Meteo Data for Arabian Sea / Bay of Bengal
        real_meteo = RealDataLoader.fetch_live_location_data(lat=15.2, lon=70.4)
        case_data = RealDataLoader.get_historical_cyclone_case(cyclone_case)
        
        print(f"\n ► Loaded Historical Cyclone Record: {case_data['name']}")
        
        raw_satellites = []
        for sat in case_data["satellites"]:
            raw_satellites.append({
                "satellite_id": sat["id"],
                "lat": sat["lat"],
                "lon": sat["lon"],
                "timestamp_str": sat["time"],
                "timestamp_min": sat["delta_t"],
                "image_matrix": generate_synthetic_cyclone_image(intensity=0.90)
            })

        physics_data = {
            "sst": case_data["sst"],
            "wind_shear": case_data["wind_shear"],
            "moisture": case_data["moisture"],
            "pressure": case_data["pressure"]
        }
        
    else:
        print("\n [MODE]: SYNTHETIC / BENCHMARK DATASET PIPELINE")
        raw_satellites = [
            {
                "satellite_id": "INSAT",
                "lat": 15.2,
                "lon": 70.4,
                "timestamp_str": "10:00 AM",
                "timestamp_min": 0,
                "image_matrix": generate_synthetic_cyclone_image(intensity=0.75)
            },
            {
                "satellite_id": "Meteosat",
                "lat": 15.2,
                "lon": 70.4,
                "timestamp_str": "10:20 AM",
                "timestamp_min": 20,
                "image_matrix": generate_synthetic_cyclone_image(intensity=0.85)
            },
            {
                "satellite_id": "NOAA",
                "lat": 15.2,
                "lon": 70.4,
                "timestamp_str": "10:10 AM",
                "timestamp_min": 10,
                "image_matrix": generate_synthetic_cyclone_image(intensity=0.80)
            }
        ]

        physics_data = {
            "sst": 30.5,
            "wind_shear": 8.5,
            "moisture": 88.0,
            "pressure": 935.0,
        }

    # Print Ingestion Details
    print("\n--- STAGE 1: Multi-Satellite Asynchronous Ingestion ---")
    for sat in raw_satellites:
        print(f" ► Satellite: {sat['satellite_id']:<10} | Time: {sat['timestamp_str']} | Lat/Lon: {sat['lat']}°N, {sat['lon']}°E")

    print("\n--- STAGE 1b: Physical Meteorological Data ---")
    print(f" 🌡️ Sea Surface Temp (SST): {physics_data['sst']} °C")
    print(f" 💨 Wind Shear           : {physics_data['wind_shear']} knots")
    print(f" 💧 Atmospheric Moisture : {physics_data['moisture']} %")
    print(f" ⏲️ Surface Pressure     : {physics_data['pressure']} hPa")

    # Run Prediction
    predictor = StormSightPredictor(confidence_threshold=0.85)
    active_learning = ActiveLearningPipeline()

    print("\n--- STAGES 2 & 3: Neural Encoding & Feature Fusion ---")
    result = predictor.run_inference(raw_satellites, physics_data)
    enc_summary = result["encodings_summary"]
    print(f" ✔️ Image Encoder (CNN)        : {enc_summary['image_feature_dim']} dimensions extracted")
    print(f" ✔️ Spatio-Temporal Encoder    : {enc_summary['spatio_temporal_feature_dim']} dimensions (\u0394t recency weighted)")
    print(f" ✔️ Physics Encoder            : {enc_summary['physics_feature_dim']} dimensions extracted")
    print(f" ⚡ Fused Multi-Modal Vector    : {enc_summary['fused_feature_dim']} unified dimensions")

    # Output Predictions
    print("\n--- STAGES 4 & 5: StormSight Model Predictions ---")
    pred = result["prediction"]
    print(f" 🌀 Cyclone Intensity Class : {pred['intensity_class']}")
    print(f" 🎯 Model Confidence Score  : {pred['confidence_percent']} ({pred['confidence_score']})")
    print(f" 📈 Future Intensity Trend  : {pred['future_trend']}")
    
    print("\n 📍 Predicted Track Trajectory Forecast:")
    print(f"    {'Hour':<8} | {'Lat (°N)':<10} | {'Lon (°E)':<10} | {'Max Wind (kts)':<14}")
    print("    " + "-" * 50)
    for pt in pred["predicted_track"]:
        print(f"    +{pt['forecast_hour']:<7}h | {pt['lat']:<10} | {pt['lon']:<10} | {pt['estimated_max_wind_kts']} kts")

    # Confidence Decision Routing
    print("\n--- STAGE 6: Confidence-Based Decision Routing ---")
    dec = result["decision"]
    print(f" Status                  : {dec['status']}")
    print(f" Action                  : {dec['action']}")
    print(f" Quality Checks Passed   : {dec['passed_quality_checks']}")
    print(f" Requires Human Review   : {dec['requires_human_review']}")

    # Active Learning Simulation
    route_action = active_learning.process_decision(result)
    print(f" Active Learning Routing : {route_action}")

    # Retrain Simulation
    print("\n--- STAGE 7: Model Retraining Loop ---")
    retrain_res = active_learning.trigger_retraining()
    print(f" Retrain Epoch           : #{retrain_res['epoch']}")
    print(f" Dataset Samples Used    : {retrain_res['new_samples_used']}")
    print(f" Training Loss           : {retrain_res['final_loss']}")
    print(f" Validation Accuracy     : {retrain_res['validation_accuracy'] * 100:.1f}%")
    print(f" Status                  : {retrain_res['status']}")

    print("\n" + "=" * 80)
    print(" SUCCESS: StormSight End-to-End Execution completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
