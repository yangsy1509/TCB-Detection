# -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# =========================================================
# 0. Paths
# =========================================================
SCENE_ROOT = Path(os.environ['TCB_SCENE_ROOT'])
MODEL_DIR = Path(os.environ['TCB_MODEL_DIR'])
CKPT_PATH = Path(os.environ['TCB_CHECKPOINT'])
MEAN_PATH = MODEL_DIR / "train_channel_mean.npy"
STD_PATH = MODEL_DIR / "train_channel_std.npy"

OUT_DIR = Path(os.environ['TCB_OUTPUT_DIR'])
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "hourly_full_scene_patch_predictions_2023_2024.csv"

# =========================================================
# 1. Settings
# =========================================================
TILE = 512
BATCH_SIZE = 16
THRESHOLD = 0.5
USE_HOURS = range(1, 7)  # 01–06 UTC hourly
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAMBAVISION_ROOT = os.environ['TCB_MAMBAVISION_ROOT']
sys.path.append(MAMBAVISION_ROOT)

from mambavision.models import create_model

# =========================================================
# 2. Load normalization stats
# =========================================================
mean = np.load(MEAN_PATH).astype(np.float32).reshape(4, 1, 1)
std = np.load(STD_PATH).astype(np.float32).reshape(4, 1, 1)

print("=== Normalization stats ===")
print("mean:", mean.reshape(-1))
print("std :", std.reshape(-1))

# =========================================================
# 3. Model
# =========================================================
def build_model():
    return create_model("mamba_vision_T", pretrained=False, num_classes=1, in_chans=4, resolution=512)

def load_model():
    model = build_model()
    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()
print(f"Loaded model: {CKPT_PATH}")
print(f"Device: {DEVICE}")

# =========================================================
# 4. Scene / patch utilities
# =========================================================
def is_hourly_scene(scene_dir):
    dt = scene_dir.name
    if re.fullmatch(r"20\d{10}", dt) is None:
        return False
    hour = int(dt[8:10])
    minute = int(dt[10:12])
    return (minute == 0) and (hour in USE_HOURS)

def crop_to_tiles(rgb, cth, tile=512):
    h = min(rgb.shape[1], cth.shape[0])
    w = min(rgb.shape[2], cth.shape[1])
    h2 = (h // tile) * tile
    w2 = (w // tile) * tile
    top = (h - h2) // 2
    left = (w - w2) // 2
    rgb = rgb[:, top:top+h2, left:left+w2]
    cth = cth[top:top+h2, left:left+w2]
    return rgb, cth, top, left

def make_scene_patches(rgb, cth, timestamp):
    rgb, cth, top, left = crop_to_tiles(rgb, cth, TILE)
    nrow = rgb.shape[1] // TILE
    ncol = rgb.shape[2] // TILE
    patches, metas = [], []

    for r in range(nrow):
        for c in range(ncol):
            y0, y1 = r * TILE, (r + 1) * TILE
            x0, x1 = c * TILE, (c + 1) * TILE

            rgb_p = rgb[:, y0:y1, x0:x1].astype(np.float32)
            cth_p = cth[y0:y1, x0:x1].astype(np.float32)

            cth_valid_ratio = float(np.isfinite(cth_p).mean())

            # 학습 코드와 동일하게: 먼저 정규화하고, 그 다음 NaN을 0으로 대체
            x = np.concatenate([rgb_p, cth_p[None, :, :]], axis=0)
            x = (x - mean) / std
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            patches.append(x)
            metas.append({
                "timestamp": timestamp,
                "patch_year": int(timestamp[:4]),
                "patch_name": f"{timestamp}_{r:02d}_{c:02d}.npy",
                "patch_row": r,
                "patch_col": c,
                "scene_top_crop": top,
                "scene_left_crop": left,
                "cth_valid_ratio": cth_valid_ratio,
            })

    return patches, metas

# =========================================================
# 5. Resume support
# =========================================================
done_timestamps = set()
if OUT_CSV.exists():
    old = pd.read_csv(OUT_CSV, usecols=["timestamp"])
    done_timestamps = set(old["timestamp"].astype(str).unique())
    print(f"[Resume] Existing CSV found: {OUT_CSV}")
    print(f"[Resume] Done timestamps: {len(done_timestamps):,}")

scene_dirs = sorted([p for p in SCENE_ROOT.iterdir() if p.is_dir() and is_hourly_scene(p)])
scene_dirs = [p for p in scene_dirs if p.name not in done_timestamps]

print(f"Hourly scenes remaining: {len(scene_dirs):,}")

# =========================================================
# 6. Inference
# =========================================================
write_header = not OUT_CSV.exists()

with torch.no_grad():
    for scene_dir in tqdm(scene_dirs, desc="Hourly full-scene inference", unit="scene"):
        timestamp = scene_dir.name
        rgb_path = scene_dir / "RGB.npy"
        cth_path = scene_dir / "CTH.npy"

        if not rgb_path.exists() or not cth_path.exists():
            print(f"[SKIP missing] {timestamp}")
            continue

        try:
            rgb = np.load(rgb_path).astype(np.float32)
            cth = np.load(cth_path).astype(np.float32)

            if rgb.ndim != 3 or rgb.shape[0] != 3:
                raise ValueError(f"Invalid RGB shape: {rgb.shape}")
            if cth.ndim != 2:
                raise ValueError(f"Invalid CTH shape: {cth.shape}")

            patches, metas = make_scene_patches(rgb, cth, timestamp)
            records = []

            for i in range(0, len(patches), BATCH_SIZE):
                batch_np = np.stack(patches[i:i+BATCH_SIZE])
                batch = torch.from_numpy(batch_np).float().to(DEVICE, non_blocking=True)

                logits = model(batch)
                probs = torch.sigmoid(logits.view(-1)).detach().cpu().numpy()

                for meta, prob in zip(metas[i:i+BATCH_SIZE], probs):
                    meta["tcb_prob"] = float(prob)
                    meta["pred"] = int(prob >= THRESHOLD)
                    records.append(meta)

            out = pd.DataFrame(records)
            out.to_csv(OUT_CSV, mode="a", header=write_header, index=False)
            write_header = False

        except Exception as e:
            print(f"[FAILED] {timestamp}: {type(e).__name__}: {e}")
            continue

print("\nDone.")
print(f"Saved: {OUT_CSV}")

df = pd.read_csv(OUT_CSV)
print("\n=== Prediction summary ===")
print(f"Total patches: {len(df):,}")
print(df["pred"].value_counts().rename(index={0: "non-TCB", 1: "TCB"}))
print("\nBy year:")
print(df.groupby(["patch_year", "pred"]).size().unstack(fill_value=0))


