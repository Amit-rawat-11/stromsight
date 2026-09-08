#!/usr/bin/env python3
"""
StormSight PyTorch Training System (train.py).
Trains the StormSight multi-task deep learning model on IBTrACS cyclone dataset.
Applies training-set-only scalers, early stopping, and model checkpointing.
"""

import os
import sys
import json
import random
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from stormsight.data.ibtracs import IBTrACSDataLoader, CycloneObservation, map_wind_to_class_idx
from stormsight.data.satellite import SatelliteDatasetInterface
from stormsight.models.stormsight import StormSightModel

# Set reproducible seeds
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CyclonePyTorchDataset(Dataset):
    """PyTorch Dataset for Cyclone Observations."""
    def __init__(self, obs_list: List[CycloneObservation], scaler: Dict[str, Dict[str, float]], mode: str = "DEMO_MODE"):
        self.obs_list = obs_list
        self.scaler = scaler
        self.sat_interface = SatelliteDatasetInterface(mode=mode)

    def __len__(self):
        return len(self.obs_list)

    def __getitem__(self, idx: int):
        obs = self.obs_list[idx]
        
        # Standardize physical numerical variables using training set scaler
        def norm(val, key):
            if val is None:
                return 0.0, 0.0
            mean = self.scaler[key]["mean"]
            std = max(1e-6, self.scaler[key]["std"])
            return (val - mean) / std, 1.0

        sst_n, sst_mask = norm(obs.sst_celsius, "sst_celsius")
        press_n, press_mask = norm(obs.pressure_hpa, "pressure_hpa")
        shear_n, shear_mask = norm(obs.wind_shear_kts, "wind_shear_kts")
        moist_n, moist_mask = norm(obs.moisture_pct, "moisture_pct")

        lat_n, _ = norm(obs.lat, "lat")
        lon_n, _ = norm(obs.lon, "lon")

        # Generate synthetic/real image matrix tensor (1, 64, 64)
        intensity_proxy = max(0.1, min(1.0, obs.max_wind_kts / 150.0))
        sat_sample = self.sat_interface.create_observation_sample(
            platform_id="INSAT-3DR",
            timestamp=obs.timestamp,
            time_delta_min=0.0,
            lat=obs.lat,
            lon=obs.lon,
            intensity=intensity_proxy
        )
        img_tensor = torch.tensor(sat_sample.image_matrix, dtype=torch.float32).unsqueeze(0) # (1, 64, 64)

        phys_vals = torch.tensor([sst_n, press_n, shear_n, moist_n], dtype=torch.float32)
        phys_mask = torch.tensor([sst_mask, press_mask, shear_mask, moist_mask], dtype=torch.float32)

        lat_tensor = torch.tensor([lat_n], dtype=torch.float32)
        lon_tensor = torch.tensor([lon_n], dtype=torch.float32)
        sat_idx_tensor = torch.tensor([0], dtype=torch.long)
        delta_t_tensor = torch.tensor([0.0], dtype=torch.float32)

        # Targets
        target_int = torch.tensor(obs.intensity_class_idx, dtype=torch.long)
        target_track = torch.tensor([
            obs.future_track_24h[0], obs.future_track_24h[1],
            obs.future_track_48h[0], obs.future_track_48h[1],
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=torch.float32)
        
        target_fut_int = torch.tensor([
            float(obs.intensity_class_idx), float(obs.intensity_class_idx), float(obs.intensity_class_idx), 0.0, 0.0
        ], dtype=torch.float32)
        
        target_ri = torch.tensor(1.0 if obs.rapid_intensification_24h else 0.0, dtype=torch.float32)

        return {
            "img": img_tensor,
            "lat": lat_tensor,
            "lon": lon_tensor,
            "sat_idx": sat_idx_tensor,
            "delta_t": delta_t_tensor,
            "phys_vals": phys_vals,
            "phys_mask": phys_mask,
            "target_int": target_int,
            "target_track": target_track,
            "target_fut_int": target_fut_int,
            "target_ri": target_ri,
        }

def train_stormsight(
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 1e-3,
    output_dir: str = "checkpoints"
) -> Dict[str, Any]:
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Executing on PyTorch device: {device}")

    # 1. Load Data with Storm-level Splits
    loader = IBTrACSDataLoader(seed=42)
    train_ids, val_ids, test_ids = loader.get_storm_split(train_ratio=0.6, val_ratio=0.2)
    
    train_obs = loader.get_observations(train_ids)
    val_obs = loader.get_observations(val_ids)
    
    print(f"[Train] Cyclones Split -> Train: {len(train_ids)} storms ({len(train_obs)} pts) | Val: {len(val_ids)} storms ({len(val_obs)} pts)")

    # 2. Fit Scaler ONLY on Training Set
    scaler = loader.fit_training_scaler(train_obs)
    with open(os.path.join(output_dir, "scaler.json"), "w") as f:
        json.dump(scaler, f, indent=2)

    # 3. Create PyTorch DataLoaders
    train_ds = CyclonePyTorchDataset(train_obs, scaler)
    val_ds = CyclonePyTorchDataset(val_obs, scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 4. Instantiate Model & Optimizer
    model = StormSightModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    history = []

    print("[Train] Starting PyTorch Backpropagation Training Loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            
            img = batch["img"].to(device)
            lat = batch["lat"].to(device)
            lon = batch["lon"].to(device)
            sat_idx = batch["sat_idx"].to(device)
            delta_t = batch["delta_t"].to(device)
            phys_vals = batch["phys_vals"].to(device)
            phys_mask = batch["phys_mask"].to(device)

            target_int = batch["target_int"].to(device)
            target_track = batch["target_track"].to(device)
            target_fut_int = batch["target_fut_int"].to(device)
            target_ri = batch["target_ri"].to(device)

            outputs = model(img, lat, lon, sat_idx, delta_t, phys_vals, phys_mask)
            loss, loss_dict = model.compute_loss(outputs, target_int, target_track, target_fut_int, target_ri)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs["intensity_logits"], dim=-1)
            train_correct += (preds == target_int).sum().item()
            train_total += target_int.size(0)

        scheduler.step()

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                img = batch["img"].to(device)
                lat = batch["lat"].to(device)
                lon = batch["lon"].to(device)
                sat_idx = batch["sat_idx"].to(device)
                delta_t = batch["delta_t"].to(device)
                phys_vals = batch["phys_vals"].to(device)
                phys_mask = batch["phys_mask"].to(device)

                target_int = batch["target_int"].to(device)
                target_track = batch["target_track"].to(device)
                target_fut_int = batch["target_fut_int"].to(device)
                target_ri = batch["target_ri"].to(device)

                outputs = model(img, lat, lon, sat_idx, delta_t, phys_vals, phys_mask)
                loss, _ = model.compute_loss(outputs, target_int, target_track, target_fut_int, target_ri)

                val_loss += loss.item()
                preds = torch.argmax(outputs["intensity_logits"], dim=-1)
                val_correct += (preds == target_int).sum().item()
                val_total += target_int.size(0)

        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))
        val_acc = val_correct / max(1, val_total)

        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")

        record = {"epoch": epoch, "train_loss": round(avg_train_loss, 4), "val_loss": round(avg_val_loss, 4), "val_acc": round(val_acc, 4)}
        history.append(record)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_acc": val_acc
            }, checkpoint_path)

    # Save Model Config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "model_version": "v3.0-PyTorch-Transformer",
            "epochs": epochs,
            "best_val_loss": round(best_val_loss, 4),
            "train_storms": train_ids,
            "val_storms": val_ids,
            "test_storms": test_ids
        }, f, indent=2)

    print(f" ✔️ PyTorch Training Completed Successfully! Best Checkpoint Saved to {output_dir}/checkpoint.pt")
    return {"history": history, "best_val_loss": best_val_loss}

if __name__ == "__main__":
    train_stormsight(epochs=10)
