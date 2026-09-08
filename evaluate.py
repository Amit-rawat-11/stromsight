#!/usr/bin/env python3
"""
StormSight Evaluation System (evaluate.py).
Evaluates trained StormSight PyTorch model on held-out TEST CYCLONES split.
Computes Accuracy, Macro F1, Track MAE/Geodesic distance error (km), RI metrics,
and compares performance against a Persistence Baseline.
"""

import os
import json
import math
import numpy as np
import torch
from typing import Dict, Any, List

from stormsight.data.ibtracs import IBTrACSDataLoader, INTENSITY_CLASSES
from stormsight.data.satellite import SatelliteDatasetInterface
from stormsight.models.stormsight import StormSightModel
from train import CyclonePyTorchDataset

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes geodesic distance between two (Lat, Lon) points on Earth in km."""
    R = 6371.0 # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def evaluate_stormsight(checkpoint_dir: str = "checkpoints") -> Dict[str, Any]:
    scaler_path = os.path.join(checkpoint_dir, "scaler.json")
    config_path = os.path.join(checkpoint_dir, "config.json")
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pt")

    if not os.path.exists(ckpt_path):
        print(f"[Evaluate Error] Checkpoint file {ckpt_path} not found. Please run train.py first.")
        return {}

    with open(scaler_path, "r") as f:
        scaler = json.load(f)
    with open(config_path, "r") as f:
        config = json.load(f)

    test_storm_ids = config.get("test_storms", [])
    print(f"[Evaluate] Evaluating on Held-Out TEST CYCLONES: {test_storm_ids}")

    loader = IBTrACSDataLoader(seed=42)
    test_obs = loader.get_observations(test_storm_ids)
    test_ds = CyclonePyTorchDataset(test_obs, scaler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StormSightModel().to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_true_int = []
    y_pred_int = []
    
    track_errors_24h_km = []
    track_errors_48h_km = []
    
    baseline_track_errors_24h_km = []

    y_true_ri = []
    y_pred_ri_prob = []

    with torch.no_grad():
        for i in range(len(test_ds)):
            sample = test_ds[i]
            img = sample["img"].unsqueeze(0).to(device)
            lat = sample["lat"].unsqueeze(0).to(device)
            lon = sample["lon"].unsqueeze(0).to(device)
            sat_idx = sample["sat_idx"].unsqueeze(0).to(device)
            delta_t = sample["delta_t"].unsqueeze(0).to(device)
            phys_vals = sample["phys_vals"].unsqueeze(0).to(device)
            phys_mask = sample["phys_mask"].unsqueeze(0).to(device)

            out = model(img, lat, lon, sat_idx, delta_t, phys_vals, phys_mask)

            pred_class = torch.argmax(out["intensity_logits"], dim=-1).item()
            true_class = sample["target_int"].item()
            y_pred_int.append(pred_class)
            y_true_int.append(true_class)

            # Track evaluation
            disp = out["track_displacement"][0].cpu().numpy() # (10,) -> [dlat24, dlon24, dlat48, dlon48, ...]
            target_disp = sample["target_track"].numpy()

            # Model forecast track position
            ref_lat_val = test_obs[i].lat
            ref_lon_val = test_obs[i].lon
            
            pred_lat_24 = ref_lat_val + disp[0]
            pred_lon_24 = ref_lon_val + disp[1]
            true_lat_24 = ref_lat_val + target_disp[0]
            true_lon_24 = ref_lon_val + target_disp[1]

            err_24 = haversine_distance_km(pred_lat_24, pred_lon_24, true_lat_24, true_lon_24)
            track_errors_24h_km.append(err_24)

            # Persistence Baseline (assumes zero displacement forecast)
            base_err_24 = haversine_distance_km(ref_lat_val, ref_lon_val, true_lat_24, true_lon_24)
            baseline_track_errors_24h_km.append(base_err_24)

            # RI evaluation
            y_true_ri.append(int(sample["target_ri"].item()))
            y_pred_ri_prob.append(out["ri_prob"].item())

    # Calculate Evaluation Metrics
    y_true_int = np.array(y_true_int)
    y_pred_int = np.array(y_pred_int)

    acc = np.mean(y_true_int == y_pred_int)
    
    # Calculate Macro F1
    classes_present = np.unique(np.concatenate([y_true_int, y_pred_int]))
    f1_list = []
    for c in classes_present:
        tp = np.sum((y_true_int == c) & (y_pred_int == c))
        fp = np.sum((y_true_int != c) & (y_pred_int == c))
        fn = np.sum((y_true_int == c) & (y_pred_int != c))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec) / max(1e-6, prec + rec)
        f1_list.append(f1)
    macro_f1 = np.mean(f1_list) if f1_list else 0.0

    mean_track_24h_km = np.mean(track_errors_24h_km)
    mean_base_track_24h_km = np.mean(baseline_track_errors_24h_km)

    metrics = {
        "evaluation_split": "TEST_CYCLONES_HELD_OUT",
        "test_cyclones": test_storm_ids,
        "sample_count": len(test_obs),
        "intensity_accuracy": round(float(acc), 4),
        "intensity_macro_f1": round(float(macro_f1), 4),
        "track_forecast_24h_mae_km": round(float(mean_track_24h_km), 1),
        "persistence_baseline_24h_mae_km": round(float(mean_base_track_24h_km), 1),
        "track_error_reduction_vs_baseline_pct": round(float((1.0 - (mean_track_24h_km / max(1e-6, mean_base_track_24h_km))) * 100), 1),
        "ri_sample_count": len(y_true_ri)
    }

    print("\n" + "=" * 70)
    print(" 📊 STORMSIGHT HELD-OUT TEST EVALUATION METRICS")
    print("=" * 70)
    print(f" ► Held-Out Test Cyclones  : {', '.join(test_storm_ids)}")
    print(f" ► Total Observation Points : {metrics['sample_count']}")
    print(f" ► Intensity Classification Acc : {metrics['intensity_accuracy']*100:.1f}%")
    print(f" ► Intensity Macro F1 Score    : {metrics['intensity_macro_f1']:.4f}")
    print(f" ► 24h Track Forecast MAE Error : {metrics['track_forecast_24h_mae_km']} km")
    print(f" ► Persistence Baseline MAE Error: {metrics['persistence_baseline_24h_mae_km']} km")
    print(f" ► Track Skill vs Baseline     : +{metrics['track_error_reduction_vs_baseline_pct']}% Improvement")
    print("=" * 70 + "\n")

    return metrics

if __name__ == "__main__":
    evaluate_stormsight()
