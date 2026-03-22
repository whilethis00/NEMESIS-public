"""
MSD Task07 - Pancreas Segmentation Dataset.

Download from http://medicaldecathlon.com (Task07_Pancreas.tar)

Directory structure after extraction:
    <data_root>/
        dataset.json
        imagesTr/  pancreas_XXX.nii.gz  (281 training cases)
        labelsTr/  pancreas_XXX.nii.gz
        imagesTs/  (test, no labels)

Labels:
    0 = background
    1 = pancreas
    2 = tumor (cancer)

For the benchmark we evaluate:
    - Pancreas Dice (class 1+2 merged, or just class 1)
    - Tumor Dice (class 2)
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


HU_MIN_PANC, HU_MAX_PANC = -96.0, 215.0   # pancreas-optimal window


def normalize_panc_ct(volume: np.ndarray) -> np.ndarray:
    volume = np.clip(volume, HU_MIN_PANC, HU_MAX_PANC)
    return (volume - HU_MIN_PANC) / (HU_MAX_PANC - HU_MIN_PANC)


class MSDPancreasDataset(Dataset):
    """
    MSD Task07 Pancreas 3D dataset.

    Args:
        data_root     : path to extracted Task07_Pancreas directory
        split         : 'train' | 'val' | 'test'
        roi_size      : crop size during training / resize during evaluation
        val_fraction  : fraction used for validation
        num_classes   : 2 (BG / pancreas, merge tumor) or 3 (BG / pancreas / tumor)
        label_fraction: fraction of labeled cases (semi-supervised)
    """

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
        self.split       = split
        self.roi_size    = roi_size
        self.num_classes = num_classes

        data_root = Path(data_root)
        meta_path = data_root / "dataset.json"

        if not meta_path.exists():
            raise FileNotFoundError(f"dataset.json not found in {data_root}")

        meta = json.loads(meta_path.read_text())
        # MSD format: {"training": [{"image": "imagesTr/xxx.nii.gz", "label": "..."}, ...]}
        all_pairs = meta.get("training", [])

        rng = random.Random(seed)
        shuffled = list(all_pairs)
        rng.shuffle(shuffled)
        n_val   = max(1, int(len(shuffled) * val_fraction))
        val_p   = shuffled[:n_val]
        train_p = shuffled[n_val:]

        if split == "train":
            chosen = train_p
        elif split == "val":
            chosen = val_p
        else:
            test_dir = data_root / "imagesTs"
            chosen   = [{"image": str(p), "label": None}
                        for p in sorted(test_dir.glob("*.nii.gz"))]

        if split == "train" and label_fraction < 1.0:
            n      = max(1, int(len(chosen) * label_fraction))
            chosen = rng.sample(chosen, n)
            print(f"[MSD Pancreas] label_efficiency={label_fraction*100:.0f}% → {n} cases")

        self.samples: List[dict] = []
        for pair in chosen:
            # Paths may be relative ("imagesTr/xxx.nii.gz") or absolute
            img_path = Path(pair["image"])
            if not img_path.is_absolute():
                img_path = data_root / img_path
            lbl_path = Path(pair["label"]) if pair.get("label") else None
            if lbl_path is not None and not lbl_path.is_absolute():
                lbl_path = data_root / lbl_path

            if not img_path.exists():
                print(f"[MSD] WARNING: {img_path} not found, skipping.")
                continue
            self.samples.append({
                "image": str(img_path),
                "label": str(lbl_path) if lbl_path else None,
                "case": img_path.stem,
            })

        print(f"[MSD Pancreas] split={split}, cases={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        import nibabel as nib

        s   = self.samples[idx]
        img = nib.load(s["image"]).get_fdata(dtype=np.float32)

        # Ensure (D, H, W) ordering
        if img.ndim == 3 and img.shape[2] < img.shape[0]:
            img = img.transpose(2, 0, 1)

        img = normalize_panc_ct(img)

        if s["label"] is not None:
            lbl = nib.load(s["label"]).get_fdata(dtype=np.float32)
            if lbl.ndim == 3 and lbl.shape[2] < lbl.shape[0]:
                lbl = lbl.transpose(2, 0, 1)
            lbl = np.round(lbl).astype(np.int64)
            # Merge: if num_classes==2, treat tumor(2) as pancreas(1)
            if self.num_classes == 2:
                lbl = np.where(lbl == 2, 1, lbl)
            lbl = np.clip(lbl, 0, self.num_classes - 1)
        else:
            lbl = np.zeros(img.shape, dtype=np.int64)

        if self.split == "train":
            img, lbl = self._fg_crop(img, lbl)
        else:
            img = resize_volume(img, self.roi_size)
            lbl = resize_label(lbl, self.roi_size)

        img_t = torch.from_numpy(img[None].astype(np.float32))
        lbl_t = torch.from_numpy(lbl.astype(np.int64))
        return {"image": img_t, "label": lbl_t, "case": s["case"]}

    def _fg_crop(self, img: np.ndarray, lbl: np.ndarray):
        """Foreground-centered random crop."""
        D, H, W = img.shape
        rd, rh, rw = self.roi_size

        fg = np.argwhere(lbl > 0)
        if len(fg) > 0:
            center = fg[np.random.randint(len(fg))]
            d0 = int(np.clip(center[0] - rd // 2, 0, max(0, D - rd)))
            h0 = int(np.clip(center[1] - rh // 2, 0, max(0, H - rh)))
            w0 = int(np.clip(center[2] - rw // 2, 0, max(0, W - rw)))
        else:
            d0 = random.randint(0, max(0, D - rd))
            h0 = random.randint(0, max(0, H - rh))
            w0 = random.randint(0, max(0, W - rw))

        img_c = np.zeros(self.roi_size, dtype=img.dtype)
        lbl_c = np.zeros(self.roi_size, dtype=lbl.dtype)
        dd = min(rd, D - d0); dh = min(rh, H - h0); dw = min(rw, W - w0)
        img_c[:dd, :dh, :dw] = img[d0:d0+dd, h0:h0+dh, w0:w0+dw]
        lbl_c[:dd, :dh, :dw] = lbl[d0:d0+dd, h0:h0+dh, w0:w0+dw]
        return img_c, lbl_c
