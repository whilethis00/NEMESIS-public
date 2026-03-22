"""
KiTS23 Kidney Tumor Segmentation Dataset.

Download:
    git clone https://github.com/neheller/kits23
    cd kits23 && python -m starter_code.get_imaging

Directory structure after download:
    <data_root>/
        dataset.json
        cases/
            case_00000/
                imaging.nii.gz
                segmentation.nii.gz
            case_00001/ ...

Labels:
    0 = background
    1 = kidney (includes kidney + tumor + cyst)
    2 = tumor
    3 = cyst  (KiTS23 new addition)

For the benchmark we use:
    - Binary kidney vs tumor: classes 0, 1, 2  (3-class)
    - Or merged: kidney(1+3) vs tumor(2)        (3-class)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .synapse import normalize_ct, resize_volume, resize_label


HU_MIN_KIDNEY, HU_MAX_KIDNEY = -79.0, 304.0   # kidney-optimal window


def normalize_kidney_ct(volume: np.ndarray) -> np.ndarray:
    volume = np.clip(volume, HU_MIN_KIDNEY, HU_MAX_KIDNEY)
    return (volume - HU_MIN_KIDNEY) / (HU_MAX_KIDNEY - HU_MIN_KIDNEY)


class KiTS23Dataset(Dataset):
    """
    KiTS23 3D segmentation dataset.

    Args:
        data_root     : path to cloned kits23 directory (contains dataset.json)
        split         : 'train' | 'val' | 'test'
        roi_size      : random crop size during training; resize during val/test
        val_fraction  : fraction of training cases used as validation
        num_classes   : 3 (BG / kidney / tumor) or 4 (BG / kidney / tumor / cyst)
        label_fraction: fraction of labeled training cases (semi-supervised expt)
    """

    NUM_CLASSES = 3   # 0=BG, 1=kidney, 2=tumor (cyst merged with kidney)

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        roi_size: Tuple[int, int, int] = (128, 128, 128),
        val_fraction: float = 0.1,
        num_classes: int = 3,
        label_fraction: float = 1.0,
        seed: int = 42,
    ):
        super().__init__()
        self.split      = split
        self.roi_size   = roi_size
        self.num_classes = num_classes

        data_root = Path(data_root)
        meta_path = data_root / "dataset.json"

        if meta_path.exists():
            meta   = json.loads(meta_path.read_text())
            cases  = [c["case_id"] for c in meta.get("training", [])]
        else:
            # Fallback: glob for case_XXXXX directories
            cases = sorted(d.name for d in (data_root / "cases").iterdir() if d.is_dir())

        # Reproducible train/val split
        rng = random.Random(seed)
        cases_shuffled = list(cases)
        rng.shuffle(cases_shuffled)
        n_val  = max(1, int(len(cases_shuffled) * val_fraction))
        val_cases   = sorted(cases_shuffled[:n_val])
        train_cases = sorted(cases_shuffled[n_val:])

        if split == "train":
            chosen = train_cases
        elif split == "val":
            chosen = val_cases
        else:
            chosen = cases   # use all for final test

        # Label efficiency
        if split == "train" and label_fraction < 1.0:
            n = max(1, int(len(chosen) * label_fraction))
            chosen = sorted(rng.sample(chosen, n))
            print(f"[KiTS23] label_efficiency={label_fraction*100:.0f}% → {n} cases")

        cases_dir = data_root / "cases"
        self.samples: List[dict] = []
        for case_id in chosen:
            img_path = cases_dir / case_id / "imaging.nii.gz"
            seg_path = cases_dir / case_id / "segmentation.nii.gz"
            if not img_path.exists() or not seg_path.exists():
                print(f"[KiTS23] WARNING: {case_id} missing files, skipping.")
                continue
            self.samples.append({"image": str(img_path), "label": str(seg_path), "case": case_id})

        print(f"[KiTS23] split={split}, cases={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        import nibabel as nib

        s   = self.samples[idx]
        img = nib.load(s["image"]).get_fdata(dtype=np.float32)
        lbl = nib.load(s["label"]).get_fdata(dtype=np.float32)

        # Ensure (D, H, W)
        if img.ndim == 3 and img.shape[2] < img.shape[0]:
            img = img.transpose(2, 0, 1)
            lbl = lbl.transpose(2, 0, 1)

        img = normalize_kidney_ct(img)
        lbl = np.round(lbl).astype(np.int64)

        # Merge cyst (3) with kidney (1) for 3-class setup
        if self.num_classes == 3:
            lbl = np.where(lbl == 3, 1, lbl)
        lbl = np.clip(lbl, 0, self.num_classes - 1)

        if self.split == "train":
            img, lbl = self._random_crop(img, lbl)
        else:
            img = resize_volume(img, self.roi_size)
            lbl = resize_label(lbl, self.roi_size)

        img_t = torch.from_numpy(img[None].astype(np.float32))
        lbl_t = torch.from_numpy(lbl.astype(np.int64))
        return {"image": img_t, "label": lbl_t, "case": s["case"]}

    def _random_crop(self, img, lbl):
        D, H, W = img.shape
        rd, rh, rw = self.roi_size

        # Try to include foreground
        fg = np.argwhere(lbl > 0)
        if len(fg) > 0:
            center = fg[np.random.randint(len(fg))]
            d0 = np.clip(center[0] - rd // 2, 0, max(0, D - rd))
            h0 = np.clip(center[1] - rh // 2, 0, max(0, H - rh))
            w0 = np.clip(center[2] - rw // 2, 0, max(0, W - rw))
        else:
            d0 = random.randint(0, max(0, D - rd))
            h0 = random.randint(0, max(0, H - rh))
            w0 = random.randint(0, max(0, W - rw))

        # Pad if necessary
        img_ = np.zeros(self.roi_size, dtype=img.dtype)
        lbl_ = np.zeros(self.roi_size, dtype=lbl.dtype)
        dd = min(rd, D - d0); dh = min(rh, H - h0); dw = min(rw, W - w0)
        img_[:dd, :dh, :dw] = img[d0:d0+dd, h0:h0+dh, w0:w0+dw]
        lbl_[:dd, :dh, :dw] = lbl[d0:d0+dd, h0:h0+dh, w0:w0+dw]
        return img_, lbl_
