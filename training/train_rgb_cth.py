# -*- coding: utf-8 -*-
# Public paper-training entry point. Original numerical/training operations retained.
import os
import random
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import seaborn as sns

# =========================
# Paths & Splits
# =========================
MATCHED_CSV = Path(os.environ['TCB_CATALOG'])

YEARS_TRAIN = {'2019', '2020', '2021'}
YEARS_VAL   = {'2022'}
YEARS_TEST  = {'2023', '2024'}

MODEL_SAVE_DIR = os.environ['TCB_OUTPUT_DIR']
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# =========================
# Reproducibility & Device
# =========================
random_seed = int(os.environ.get('TCB_TRAINING_SEED', '1004'))
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 1) Build manifest
# =========================
def build_manifest_from_matched_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    
    df["year"] = df["year"].astype(str)

    df["path"] = df["rgb_path"]
    df["label"] = df["target"].astype(int)
    df["name"] = df["filename"]

    df = df[["path", "label", "year", "name", "ctt_path", "cth_path"]].copy()

    df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)

    print("=== Loaded matched CSV ===")
    print(f"CSV path: {csv_path}")
    print(f"Total samples: {len(df)}")
    print(df.groupby(["year", "label"]).size().rename("count").unstack(fill_value=0))
    print("-" * 60)

    return df


df_all = build_manifest_from_matched_csv(MATCHED_CSV)

def balance_set_ratio(df, false_ratio=3, seed=1004, name="Set"):
    df_true = df[df["label"] == 1]
    df_false = df[df["label"] == 0]

    n_true = len(df_true)
    n_false_orig = len(df_false)
    target_false = n_true * false_ratio

    if len(df_false) > target_false:
        df_false = df_false.sample(target_false, random_state=seed)

    df_balanced = (
        pd.concat([df_true, df_false])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    print(f"\n=== {name} balancing applied ===")
    print(f"True           : {len(df_true)}")
    print(f"False original : {n_false_orig}")
    print(f"False used     : {len(df_false)} (ratio={false_ratio}:1)")
    print(f"Total used     : {len(df_balanced)}")

    return df_balanced


def split_by_year(df: pd.DataFrame):
    tr = df[df["year"].isin(YEARS_TRAIN)].reset_index(drop=True)
    va = df[df["year"].isin(YEARS_VAL)].reset_index(drop=True)
    te = df[df["year"].isin(YEARS_TEST)].reset_index(drop=True)
    return tr, va, te

df_train, df_val, df_test = split_by_year(df_all)

FALSE_RATIO = 3

df_train = balance_set_ratio(df_train, false_ratio=FALSE_RATIO, seed=int(os.environ.get('TCB_SAMPLING_SEED', '1004')), name="Train")
df_val   = balance_set_ratio(df_val,   false_ratio=FALSE_RATIO, seed=int(os.environ.get('TCB_SAMPLING_SEED', '1004')), name="Val")
df_test  = balance_set_ratio(df_test,  false_ratio=FALSE_RATIO, seed=int(os.environ.get('TCB_SAMPLING_SEED', '1004')), name="Test")

def describe_split(name, df):
    if len(df) == 0:
        print(f"[{name}] empty")
        print("-" * 60)
        return
    pos = (df["label"] == 1).sum()
    neg = (df["label"] == 0).sum()
    print(f"[{name}] total={len(df):6d} | pos={pos:5d} ({pos/len(df):.2%}) | neg={neg:5d} ({neg/len(df):.2%})")
    print(df.groupby(["year", "label"]).size().rename("count").unstack(fill_value=0))
    print("-" * 60)

print("=== Split Summary ===")
describe_split("Train", df_train)
describe_split("Val  ", df_val)
describe_split("Test ", df_test)

# =========================
# 2) Array utilities
# =========================
def load_npy_rgb_cth(rgb_path: str, cth_path: str) -> np.ndarray:
    """
    RGB 3채널 + CTH 1채널을 합쳐서 (4, H, W) 반환
    """
    rgb = np.load(rgb_path)
    cth = np.load(cth_path)

    rgb = rgb.astype(np.float32, copy=False)
    cth = cth.astype(np.float32, copy=False)

    # RGB: HWC -> CHW
    if rgb.ndim != 3:
        raise ValueError(f"Expected RGB 3D array, got shape={rgb.shape} at {rgb_path}")

    if rgb.shape[-1] == 3 and rgb.shape[0] != 3:
        rgb = np.transpose(rgb, (2, 0, 1))

    if rgb.shape[0] != 3:
        raise ValueError(f"Expected RGB 3 channels, got shape={rgb.shape} at {rgb_path}")

    # CTH: (H,W) 또는 (1,H,W) 또는 (H,W,1) 처리
    if cth.ndim == 2:
        cth = cth[None, :, :]  # (1,H,W)
    elif cth.ndim == 3:
        if cth.shape[0] == 1:
            pass
        elif cth.shape[-1] == 1:
            cth = np.transpose(cth, (2, 0, 1))
        else:
            raise ValueError(f"Expected CTH 1 channel, got shape={cth.shape} at {cth_path}")
    else:
        raise ValueError(f"Expected CTH 2D or 3D array, got shape={cth.shape} at {cth_path}")

    # 크기 체크
    if rgb.shape[1:] != cth.shape[1:]:
        raise ValueError(
            f"Shape mismatch: RGB {rgb.shape} vs CTH {cth.shape}\n"
            f"RGB path: {rgb_path}\n"
            f"CTH path: {cth_path}"
        )

    arr = np.concatenate([rgb, cth], axis=0)  # (4,H,W)

    return arr

# =========================
# 3) Compute train mean/std
# =========================
def compute_channel_mean_std(df: pd.DataFrame, num_channels: int = 4):
    """
    Train set 전체에서 채널별 mean/std 계산
    RGB 3채널 + CTH 1채널 = 총 4채널
    NaN은 제외
    """
    sum_c = np.zeros(num_channels, dtype=np.float64)
    sumsq_c = np.zeros(num_channels, dtype=np.float64)
    count_c = np.zeros(num_channels, dtype=np.int64)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing train mean/std"):
        arr = load_npy_rgb_cth(row["path"], row["cth_path"])  # (4,H,W)

        for c in range(num_channels):
            ch = arr[c]
            finite = np.isfinite(ch)
            if not finite.any():
                continue

            vals = ch[finite].astype(np.float64)
            sum_c[c] += vals.sum()
            sumsq_c[c] += np.square(vals).sum()
            count_c[c] += vals.size

    mean_c = np.zeros(num_channels, dtype=np.float32)
    std_c  = np.ones(num_channels, dtype=np.float32)

    for c in range(num_channels):
        if count_c[c] == 0:
            mean_c[c] = 0.0
            std_c[c] = 1.0
            continue

        mean = sum_c[c] / count_c[c]
        var = (sumsq_c[c] / count_c[c]) - (mean ** 2)
        var = max(var, 1e-12)
        std = np.sqrt(var)

        mean_c[c] = np.float32(mean)
        std_c[c] = np.float32(std)

    return mean_c, std_c

mean_path = os.path.join(MODEL_SAVE_DIR, "train_channel_mean.npy")
std_path  = os.path.join(MODEL_SAVE_DIR, "train_channel_std.npy")

if os.path.exists(mean_path) and os.path.exists(std_path):
    print("\nLoading saved normalization stats...")
    train_mean = np.load(mean_path)
    train_std  = np.load(std_path)

else:
    print("\nComputing train mean/std...")
    train_mean, train_std = compute_channel_mean_std(df_train, num_channels=4)

    np.save(mean_path, train_mean)
    np.save(std_path, train_std)

print("\n=== Train normalization stats ===")
channel_names = ["R", "G", "B", "CTH"]
for i, ch_name in enumerate(channel_names):
    print(f"{ch_name:>5s} | mean={train_mean[i]:.6f} | std={train_std[i]:.6f}")

# =========================
# 4) Dataset
# =========================
class RGBCTHNPYDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mean: np.ndarray, std: np.ndarray, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _augment(self, x):
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])
        return x

    def _normalize(self, arr_chw: np.ndarray) -> np.ndarray:
        out = arr_chw.copy()
        for c in range(out.shape[0]):
            out[c] = (out[c] - self.mean[c]) / self.std[c]
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return out

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        rgb_path = row["path"]
        cth_path = row["cth_path"]
        label = float(row["label"])

        arr = load_npy_rgb_cth(rgb_path, cth_path)  # (4,H,W)
        arr = self._normalize(arr)

        x = torch.from_numpy(arr).contiguous()

        if self.augment:
            x = self._augment(x)

        y = torch.tensor([label], dtype=torch.float32)
        name = row["name"]

        return x, y, name
    
# =========================
# 5) DataLoaders
# =========================
batch_size_train = 16
batch_size_eval  = 16

train_ds = RGBCTHNPYDataset(df_train, mean=train_mean, std=train_std, augment=True)
val_ds   = RGBCTHNPYDataset(df_val,   mean=train_mean, std=train_std, augment=False)
test_ds  = RGBCTHNPYDataset(df_test,  mean=train_mean, std=train_std, augment=False)

train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size_eval,  shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=1,                shuffle=False, num_workers=2, pin_memory=True)

# =========================
# 6) Model
# =========================
import sys
sys.path.insert(0, os.environ['TCB_MAMBAVISION_ROOT'])

from mambavision.models import create_model

PRETRAIN_CKPT = os.environ['TCB_PRETRAINED_CHECKPOINT']
os.makedirs(os.path.dirname(PRETRAIN_CKPT), exist_ok=True)

# ---------------------------------------------------------
# 1) RGB 3채널 pretrained 모델 생성
# ---------------------------------------------------------
from argparse import Namespace
from contextlib import nullcontext
_safe_context = (torch.serialization.safe_globals([Namespace])
                 if hasattr(torch.serialization, 'safe_globals') else nullcontext())
with _safe_context:
    pretrained_rgb_model = create_model(
        "mamba_vision_T",
        pretrained=True,
        model_path=PRETRAIN_CKPT,
        num_classes=1000,
        in_chans=3,
        resolution=512,
    )

pretrained_state = pretrained_rgb_model.state_dict()
del pretrained_rgb_model

# ---------------------------------------------------------
# 2) RGB+CTH 4채널 binary 모델 생성
# ---------------------------------------------------------
model = create_model(
    "mamba_vision_T",
    pretrained=False,
    num_classes=1,
    in_chans=4,
    resolution=512,
)

model_state = model.state_dict()

# ---------------------------------------------------------
# 3) 첫 번째 convolution weight 확장
# ---------------------------------------------------------
first_conv_key = "patch_embed.conv_down.0.weight"

old_w = pretrained_state[first_conv_key]      # [out, 3, k, k]
new_w = model_state[first_conv_key].clone()   # [out, 4, k, k]

print("old first conv:", old_w.shape)
print("new first conv:", new_w.shape)

# RGB 3채널은 pretrained 그대로 사용
new_w[:, :3, :, :] = old_w

# CTH 1채널은 RGB weight 평균으로 초기화
new_w[:, 3:4, :, :] = old_w.mean(dim=1, keepdim=True)

pretrained_state[first_conv_key] = new_w

# ---------------------------------------------------------
# 4) ImageNet classification head 제거
# ---------------------------------------------------------
pretrained_state.pop("head.weight", None)
pretrained_state.pop("head.bias", None)

# ---------------------------------------------------------
# 5) shape 맞는 weight만 로드
# ---------------------------------------------------------
filtered_state = {}

for k, v in pretrained_state.items():
    if k in model_state and model_state[k].shape == v.shape:
        filtered_state[k] = v
    else:
        if k in model_state:
            print(f"Skip: {k} | pretrained {v.shape} vs model {model_state[k].shape}")

missing, unexpected = model.load_state_dict(filtered_state, strict=False)

print("\n=== Pretrained loading summary ===")
print(f"Loaded keys     : {len(filtered_state)}")
print(f"Missing keys    : {missing}")
print(f"Unexpected keys : {unexpected}")

model = model.to(device)

print(model.patch_embed.conv_down[0])
print(model.head)

# =========================
# 6-1) Model parameter count
# =========================
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print("\n=== Model Parameters ===")
    print(f"Total params        : {total_params:,}")
    print(f"Trainable params    : {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")
    print(f"Total params (M)    : {total_params / 1e6:.2f} M")
    print(f"Trainable params (M): {trainable_params / 1e6:.2f} M")

count_parameters(model)

# =========================
# 7) Loss, Optim
# =========================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# =========================
# 8) Training Loop
# =========================
from sklearn.metrics import (
    f1_score, classification_report, precision_recall_fscore_support, confusion_matrix
)

epochs = int(os.environ.get('TCB_EPOCHS', '500'))
best_val_loss = float('inf')
best_val_f1 = 0.0

best_ckpt_loss = None
best_ckpt_f1 = None

loss_arr, loss_arr_val = [], []

for epoch in range(epochs):
    # --- Train ---
    model.train()
    running_loss = 0.0
    n_batches = 0

    for x, y, _ in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}] Train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    avg_train_loss = running_loss / max(n_batches, 1)
    loss_arr.append(avg_train_loss)
    print(f"[Epoch {epoch+1:03d}/{epochs}] Train: loss={avg_train_loss:.6f}")

    # --- Val ---
    model.eval()
    running_val = 0.0
    n_batches_v = 0
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for x, y, _ in tqdm(val_loader, desc=f"[Epoch {epoch+1}/{epochs}] Val  ", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            running_val += loss.item()
            n_batches_v += 1

            probs = torch.sigmoid(logits).view(-1)
            preds = (probs > 0.5).long()

            all_probs.append(probs.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(y.view(-1).detach().cpu().numpy())

    avg_val_loss = running_val / max(n_batches_v, 1)
    loss_arr_val.append(avg_val_loss)

    all_probs  = np.concatenate(all_probs)  if len(all_probs)  else np.array([])
    all_preds  = np.concatenate(all_preds)  if len(all_preds)  else np.array([])
    all_labels = np.concatenate(all_labels) if len(all_labels) else np.array([])

    val_f1  = f1_score(all_labels, all_preds) if len(all_labels) else 0.0
    val_acc = (all_preds == all_labels).mean() if len(all_labels) else 0.0

    print(f"[Epoch {epoch+1:03d}/{epochs}] Val  : loss={avg_val_loss:.6f} | acc={val_acc:.4f} | f1={val_f1:.4f}")

    # -------------------------
    # Save best by loss
    # -------------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss

        best_ckpt_loss = os.path.join(
            MODEL_SAVE_DIR,
            f"best_loss_epoch{epoch+1:03d}.pth"
        )

        torch.save(model.state_dict(), best_ckpt_loss)

        print(f"◎ New BEST LOSS model saved")
        print(f"   path : {best_ckpt_loss}")
        print(f"   loss : {best_val_loss:.6f}")

    # -------------------------
    # Save best by F1
    # -------------------------
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1

        best_ckpt_f1 = os.path.join(
            MODEL_SAVE_DIR,
            f"best_f1_epoch{epoch+1:03d}.pth"
        )

        torch.save(model.state_dict(), best_ckpt_f1)

        print(f"◎ New BEST F1 model saved")
        print(f"   path : {best_ckpt_f1}")
        print(f"   f1   : {best_val_f1:.4f}")

print("\n==============================")
print("Training finished.")
print("==============================")

print(f"Best val loss : {best_val_loss:.6f}")
print(f"Loss model    : {best_ckpt_loss}")

print(f"Best val F1   : {best_val_f1:.4f}")
print(f"F1 model      : {best_ckpt_f1}")
