"""Portable catalog paths and unchanged paper patch operations."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
SCHEMES = {'RGB': (), 'RGB_CTH': ('cth',), 'RGB_CTT': ('ctt',), 'RGB_CTT_CTH': ('ctt', 'cth')}

def local_path(value):
    return Path(value)

def read_manifest(path, data_root, check_files=True):
    frame = pd.read_csv(path)
    required = {'filename', 'year', 'target', 'rgb_path', 'cth_path', 'ctt_path'}
    if not required <= set(frame):
        raise ValueError(f'Missing columns: {sorted(required-set(frame))}')
    if frame.filename.isna().any() or frame.filename.duplicated().any() or not frame.target.isin([0,1]).all():
        raise ValueError('Sample names must be unique and targets must be0/1.')
    root = Path(data_root).resolve()
    parents = {}
    for column in ['rgb_path','cth_path','ctt_path']:
        resolved=[]
        for value in frame[column]:
            relative=Path(str(value))
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Expected a dataset-relative path: {value}')
            if relative.parent not in parents:
                folder=(root/relative.parent).resolve()
                if not folder.is_relative_to(root):
                    raise ValueError(f'Data path escapes dataset root: {value}')
                if check_files:
                    with os.scandir(folder) as entries:
                        names={e.name for e in entries if e.is_file(follow_symlinks=False)}
                else:
                    names=None
                parents[relative.parent]=(folder,names)
            folder,names=parents[relative.parent]
            full=folder/relative.name
            if check_files and relative.name not in names:
                raise FileNotFoundError(full)
            resolved.append(str(full))
        frame[column]=resolved
    frame['path']=frame['rgb_path']
    frame['label']=frame['target'].astype(int)
    frame['name']=frame['filename']
    frame['year']=frame['year'].astype(str)
    return frame

def load_patch(row, extras):
    rgb_path = local_path(row["path"])
    rgb = np.load(rgb_path).astype(np.float32, copy=False)
    if rgb.ndim != 3:
        raise ValueError(f"Expected RGB 3D array: {rgb_path}, shape={rgb.shape}")
    if rgb.shape[-1] == 3 and rgb.shape[0] != 3:
        rgb = np.transpose(rgb, (2, 0, 1))
    if rgb.shape[0] != 3:
        raise ValueError(f"Expected RGB 3 channels: {rgb_path}, shape={rgb.shape}")
    channels = [rgb]
    for kind in extras:
        path = local_path(row[kind + "_path"])
        arr = np.load(path).astype(np.float32, copy=False)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            pass
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(f"Expected one {kind} channel: {path}, shape={arr.shape}")
        if arr.shape[1:] != rgb.shape[1:]:
            raise ValueError(f"RGB/{kind} shape mismatch: {rgb.shape}, {arr.shape}")
        channels.append(arr)
    result = channels[0] if not extras else np.concatenate(channels, axis=0)
    if result.shape[1:] != (512, 512):
        raise ValueError(f"Expected 512x512 without resizing: {row['name']}, {result.shape}")
    return result

def normalize_patch(arr, mean, std):
    # Identical order to the original datasets: z-score, then invalid -> zero.
    out = arr.copy()
    for channel in range(out.shape[0]):
        out[channel] = (out[channel] - mean[channel]) / std[channel]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
