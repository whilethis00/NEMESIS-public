"""
Synapse Multi-organ Segmentation Dataset (BTCV-based).

Supports two common formats:
  1. NIfTI  : imagesTr/*.nii.gz + labelsTr/*.nii.gz  (BTCV / MSD style)
  2. NPZ/H5 : train_npz/*.npz  + test_vol_h5/*.npy.h5 (TransUNet style)

Label mapping (9 classes including background):
  0=BG, 1=aorta, 2=gallbladder, 3=spleen, 4=left kidney,
  5=right kidney, 6=liver, 7=stomach, 8=pancreas

Expected directory structure (NIfTI format):
  <data_root>/
    imagesTr/  case0001.nii.gz ...
    labelsTr/  case0001.nii.gz ...
    splits.json   (optional, else auto-split 18/12)

Expected directory structure (TransUNet/NPZ format):
  <data_root>/
    train_npz/  case0005_slice000.npz ...
    test_vol_h5/ case0001.npy.h5 ...
    lists/list_Synapse/train.txt   (optional)

Usage:
    train_ds = SynapseDataset(data_root, split="train", roi_size=(128,128,128))
    val_ds   = SynapseDataset(data_root, split="val",   roi_size=(128,128,128))
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

HU_MIN, HU_MAX = -125.0, 275.0   # abdomen window
FOREGROUND_OVERSAMPLE_RATIO = 0.9  # foreground-centered crop 비율


def normalize_ct(volume: np.ndarray) -> np.ndarray:
    """Clip to abdomen HU window and scale to [0, 1]."""
    volume = np.clip(volume, HU_MIN, HU_MAX)
    return (volume - HU_MIN) / (HU_MAX - HU_MIN)


def random_crop_with_fg(
    img: np.ndarray,
    lbl: np.ndarray,
    roi_size: Tuple[int, int, int],
    fg_ratio: float = FOREGROUND_OVERSAMPLE_RATIO,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    fg_ratio 확률로 foreground voxel을 중심으로 크롭,
    나머지는 완전 랜덤 크롭.
    """
    D, H, W = img.shape
    rd, rh, rw = roi_size

    # Pad if needed
    pad = [(0, max(0, rd - D)), (0, max(0, rh - H)), (0, max(0, rw - W))]
    if any(p[1] > 0 for p in pad):
        img = np.pad(img, pad, mode="constant", constant_values=0)
        lbl = np.pad(lbl, pad, mode="constant", constant_values=0)
        D, H, W = img.shape

    fg_voxels = np.argwhere(lbl > 0)
    use_fg = (len(fg_voxels) > 0) and (random.random() < fg_ratio)

    if use_fg:
        # foreground voxel 하나를 랜덤으로 골라 크롭 중심으로
        center = fg_voxels[random.randint(0, len(fg_voxels) - 1)]
        d0 = int(np.clip(center[0] - rd // 2, 0, D - rd))
        h0 = int(np.clip(center[1] - rh // 2, 0, H - rh))
        w0 = int(np.clip(center[2] - rw // 2, 0, W - rw))
    else:
        d0 = random.randint(0, D - rd)
        h0 = random.randint(0, H - rh)
        w0 = random.randint(0, W - rw)

    return img[d0:d0+rd, h0:h0+rh, w0:w0+rw], lbl[d0:d0+rd, h0:h0+rh, w0:w0+rw]


def resize_volume(vol: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    """Trilinear resize using torch.nn.functional."""
    t = torch.from_numpy(vol[None, None].astype(np.float32))  # (1,1,D,H,W)
    t = torch.nn.functional.interpolate(t, size=target, mode="trilinear", align_corners=False)
    return t[0, 0].numpy()


def resize_label(lbl: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    """Nearest-neighbour resize for labels."""
    t = torch.from_numpy(lbl[None, None].astype(np.float32))
    t = torch.nn.functional.interpolate(t, size=target, mode="nearest")
    return t[0, 0].numpy().astype(np.int64)


# ---------------------------------------------------------------------------
# NIfTI-based dataset
# ---------------------------------------------------------------------------

SYNAPSE_ORGAN_NAMES = [
    "background", "aorta", "gallbladder", "spleen",
    "left_kidney", "right_kidney", "liver", "stomach", "pancreas",
]

# Default 18-train / 12-test split (from TransUNet / SwinUNETR papers)
DEFAULT_TRAIN_CASES = [
    "0002", "0003", "0006", "0007", "0008", "0009", "0010", "0021",
    "0022", "0025", "0027", "0028", "0030", "0031", "0032", "0033",
    "0034", "0038",
]
DEFAULT_TEST_CASES = [
    "0001", "0004", "0005", "0011", "0013", "0018",
    "0019", "0020", "0026", "0029", "0035", "0036",
]


class SynapseNIfTIDataset(Dataset):
    """
    3D NIfTI dataset.  Each __getitem__ returns a randomly cropped ROI during
    training and the full volume during validation.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        roi_size: Tuple[int, int, int] = (128, 128, 128),
        num_classes: int = 9,
        train_cases: Optional[List[str]] = None,
        test_cases: Optional[List[str]] = None,
        label_fraction: float = 1.0,   # for label-efficiency experiments
        seed: int = 42,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.split      = split
        self.roi_size   = roi_size
        self.num_classes = num_classes
        self.transform  = transform

        data_root = Path(data_root)
        img_dir   = data_root / "imagesTr"
        lbl_dir   = data_root / "labelsTr"

        # Resolve case IDs
        if train_cases is None:
            train_cases = DEFAULT_TRAIN_CASES
        if test_cases is None:
            test_cases  = DEFAULT_TEST_CASES

        cases = train_cases if split in ("train", "val") else test_cases

        # Label-efficiency subset (reproducible)
        if split == "train" and label_fraction < 1.0:
            rng = random.Random(seed)
            n   = max(1, int(len(cases) * label_fraction))
            cases = sorted(rng.sample(cases, n))
            print(f"[Synapse] Label efficiency {label_fraction*100:.0f}%: "
                  f"using {n}/{len(train_cases)} cases → {cases}")

        self.samples: List[dict] = []
        for case_id in cases:
            img_path = img_dir / f"img{case_id}.nii.gz"
            lbl_path = lbl_dir / f"label{case_id}.nii.gz"
            # Fallback naming conventions
            if not img_path.exists():
                img_path = img_dir / f"case{case_id}_0000.nii.gz"
            if not img_path.exists():
                img_path = img_dir / f"{case_id}.nii.gz"
            if not lbl_path.exists():
                lbl_path = lbl_dir / f"case{case_id}.nii.gz"
            if not lbl_path.exists():
                lbl_path = lbl_dir / f"{case_id}.nii.gz"

            if not img_path.exists() or not lbl_path.exists():
                print(f"[Synapse] WARNING: case {case_id} not found, skipping.")
                continue
            self.samples.append({"image": str(img_path), "label": str(lbl_path), "case": case_id})

        print(f"[Synapse NIfTI] split={split}, cases={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        import nibabel as nib

        sample = self.samples[idx]
        img = nib.load(sample["image"]).get_fdata(dtype=np.float32)   # (H, W, D) or (D, H, W)
        lbl = nib.load(sample["label"]).get_fdata(dtype=np.float32)

        # Ensure (D, H, W)
        if img.ndim == 3 and img.shape[2] < img.shape[0]:
            img = img.transpose(2, 0, 1)
            lbl = lbl.transpose(2, 0, 1)

        img = normalize_ct(img)
        lbl = np.round(lbl).astype(np.int64)
        lbl = np.clip(lbl, 0, self.num_classes - 1)

        if self.split == "train":
            img, lbl = self._random_crop(img, lbl)
        # else: keep full-resolution for sliding window inference

        img_t = torch.from_numpy(img[None].astype(np.float32))   # (1, D, H, W)
        lbl_t = torch.from_numpy(lbl.astype(np.int64))           # (D, H, W)

        if self.transform is not None:
            img_t, lbl_t = self.transform(img_t, lbl_t)

        return {"image": img_t, "label": lbl_t, "case": sample["case"]}

    def _random_crop(
        self, img: np.ndarray, lbl: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return random_crop_with_fg(img, lbl, self.roi_size)


# ---------------------------------------------------------------------------
# TransUNet NPZ/H5 dataset (2D-slice training, 3D-volume validation)
# ---------------------------------------------------------------------------

class SynapseNPZDataset(Dataset):
    """
    TransUNet-style 2D-slice NPZ dataset for training.
    Each npz file has keys: 'image' (H,W) and 'label' (H,W).
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        list_dir: Optional[str] = None,
        label_fraction: float = 1.0,
        seed: int = 42,
    ):
        from pathlib import Path as _P
        self.split = split
        data_root  = _P(data_root)

        if split == "train":
            npz_dir = data_root / "train_npz"
            if list_dir:
                txt = _P(list_dir) / "train.txt"
                names = [l.strip() for l in txt.read_text().splitlines() if l.strip()]
            else:
                names = [p.stem for p in sorted(npz_dir.glob("*.npz"))]

            if label_fraction < 1.0:
                rng  = random.Random(seed)
                names = sorted(rng.sample(names, max(1, int(len(names) * label_fraction))))
            self.files = [str(npz_dir / f"{n}.npz") for n in names]
        else:
            h5_dir = data_root / "test_vol_h5"
            self.files = sorted(str(p) for p in h5_dir.glob("*.npy.h5"))

        print(f"[Synapse NPZ] split={split}, files={len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        if path.endswith(".npz"):
            data = np.load(path)
            img  = data["image"].astype(np.float32)   # (H, W)
            lbl  = data["label"].astype(np.int64)
            # Stack 3 copies to make 3D mini-volume (or use as 2D slice)
            img_t = torch.from_numpy(img[None])   # (1, H, W)
            lbl_t = torch.from_numpy(lbl)
        else:
            import h5py
            with h5py.File(path, "r") as f:
                img = f["image"][:]   # (D, H, W)
                lbl = f["label"][:]
            img  = normalize_ct(img.astype(np.float32))
            img_t = torch.from_numpy(img[None].astype(np.float32))   # (1,D,H,W)
            lbl_t = torch.from_numpy(lbl.astype(np.int64))
        return {"image": img_t, "label": lbl_t, "case": Path(path).stem}


# ---------------------------------------------------------------------------
# BTCV Dataset (already-downloaded, 14-class → 9-class Synapse remapping)
# ---------------------------------------------------------------------------

# BTCV raw label index → Synapse 9-class index
# BTCV: 0=BG,1=spleen,2=rkid,3=lkid,4=gall,5=eso,6=liver,7=sto,
#        8=aorta,9=IVC,10=veins,11=pancreas,12=rad,13=lad
# Synapse: 0=BG,1=aorta,2=gallbladder,3=spleen,4=lkid,5=rkid,6=liver,7=stomach,8=pancreas
BTCV_TO_SYNAPSE = np.array([
    0,   # 0 BG        → BG
    3,   # 1 spleen    → spleen
    5,   # 2 rkid      → right kidney
    4,   # 3 lkid      → left kidney
    2,   # 4 gall      → gallbladder
    0,   # 5 eso       → BG (excluded)
    6,   # 6 liver     → liver
    7,   # 7 stomach   → stomach
    1,   # 8 aorta     → aorta
    0,   # 9 IVC       → BG (excluded)
    0,   # 10 veins    → BG (excluded)
    8,   # 11 pancreas → pancreas
    0,   # 12 rad      → BG (excluded)
    0,   # 13 lad      → BG (excluded)
], dtype=np.int64)

SYNAPSE_ORGAN_NAMES = [
    "background", "aorta", "gallbladder", "spleen",
    "left_kidney", "right_kidney", "liver", "stomach", "pancreas",
]


class BTCVDataset(Dataset):
    """
    BTCV dataset for segmentation benchmarks.

    Reads from btcv.json or dataset_0.json for train/val/test splits.
    Applies BTCV 14-class → Synapse 9-class label remapping.

    Args:
        data_root         : path to BTCV root (contains btcv.json + data/)
        split             : 'train' | 'val' | 'test'
        roi_size          : random crop during training; resize during val/test
        label_fraction    : fraction of labeled train cases (semi-supervised)
        seed              : random seed for label fraction sampling
        samples_per_volume: virtual repetition per epoch (train only), enables
                            more gradient updates without reloading files
    """

    DEFAULT_DATA_ROOT = "data/BTCV"

    def __init__(
        self,
        data_root: str = DEFAULT_DATA_ROOT,
        split: str = "train",
        roi_size: Tuple[int, int, int] = (128, 128, 128),
        label_fraction: float = 1.0,
        seed: int = 42,
        samples_per_volume: int = 1,
    ):
        super().__init__()
        self.split    = split
        self.roi_size = roi_size
        self.samples_per_volume = samples_per_volume if split == "train" else 1
        data_root     = Path(data_root)

        # Load JSON split
        json_path = data_root / "data" / "dataset_0.json"
        if not json_path.exists():
            json_path = data_root / "btcv.json"
        if not json_path.exists():
            raise FileNotFoundError(f"No JSON found in {data_root}")

        import json as _json
        meta = _json.loads(json_path.read_text())

        # Determine path prefix (paths in JSON may be relative to data_root or data/)
        # btcv.json uses "data/imagesTr/..." relative to data_root
        # dataset_0.json uses "imagesTr/..." relative to data_root/data/
        def resolve(rel: str) -> Path:
            for base in [data_root, data_root / "data"]:
                p = base / rel
                if p.exists():
                    return p
            # last resort: strip leading component
            return data_root / rel

        train_pairs = meta.get("training", [])
        val_pairs   = meta.get("validation", [])

        if split == "train":
            pairs = train_pairs
        elif split in ("val", "test"):
            pairs = val_pairs if val_pairs else train_pairs[-6:]
        else:
            pairs = train_pairs + val_pairs

        # Label efficiency subset
        if split == "train" and label_fraction < 1.0:
            rng = random.Random(seed)
            n   = max(1, int(len(pairs) * label_fraction))
            pairs = rng.sample(pairs, n)
            print(f"[BTCV] label_efficiency={label_fraction*100:.0f}% → {n}/{len(train_pairs)} cases")

        self.samples: List[dict] = []
        for p in pairs:
            img_path = resolve(p["image"])
            lbl_path = resolve(p["label"])
            if not img_path.exists() or not lbl_path.exists():
                print(f"[BTCV] WARNING: missing files for {p['image']}, skipping.")
                continue
            case_id = Path(p["image"]).stem.replace("img", "")
            self.samples.append({"image": str(img_path), "label": str(lbl_path), "case": case_id})

        print(f"[BTCV] split={split}, cases={len(self.samples)}, "
              f"samples_per_volume={self.samples_per_volume}")

    def __len__(self):
        return len(self.samples) * self.samples_per_volume

    def __getitem__(self, idx: int):
        import nibabel as nib
        case_idx = idx % len(self.samples)
        s        = self.samples[case_idx]
        img = nib.load(s["image"]).get_fdata(dtype=np.float32)
        lbl = nib.load(s["label"]).get_fdata(dtype=np.float32)
        if img.ndim == 3:
            img = img.transpose(2, 0, 1)
            lbl = lbl.transpose(2, 0, 1)
        img = normalize_ct(img)
        lbl_int = np.round(lbl).astype(np.int64)
        lbl_int = np.clip(lbl_int, 0, len(BTCV_TO_SYNAPSE) - 1)
        lbl_syn = BTCV_TO_SYNAPSE[lbl_int]

        if self.split == "train":
            img, lbl_syn = self._random_crop(img, lbl_syn)
        # else: keep full-resolution for sliding window inference

        img_t = torch.from_numpy(img[None].astype(np.float32))
        lbl_t = torch.from_numpy(lbl_syn.astype(np.int64))
        return {"image": img_t, "label": lbl_t, "case": s["case"]}

    def _random_crop(self, img: np.ndarray, lbl: np.ndarray):
        return random_crop_with_fg(img, lbl, self.roi_size)


# ---------------------------------------------------------------------------
# Auto-detect format
# ---------------------------------------------------------------------------

def build_synapse_dataset(
    data_root: str,
    split: str = "train",
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    label_fraction: float = 1.0,
    seed: int = 42,
    samples_per_volume: int = 1,
    **kwargs,
) -> Dataset:
    """
    Auto-detect Synapse data format and return appropriate Dataset.
    Priority: BTCV (local) > NIfTI > NPZ/H5
    """
    data_root = Path(data_root)

    # BTCV: has btcv.json or data/dataset_0.json
    has_btcv_json = (data_root / "btcv.json").exists() or \
                    (data_root / "data" / "dataset_0.json").exists()
    if has_btcv_json:
        return BTCVDataset(
            data_root=str(data_root),
            split=split,
            roi_size=roi_size,
            label_fraction=label_fraction,
            seed=seed,
            samples_per_volume=samples_per_volume,
        )
    elif (data_root / "imagesTr").is_dir():
        return SynapseNIfTIDataset(
            data_root=str(data_root),
            split=split,
            roi_size=roi_size,
            label_fraction=label_fraction,
            seed=seed,
            **kwargs,
        )
    elif (data_root / "train_npz").is_dir() or (data_root / "test_vol_h5").is_dir():
        return SynapseNPZDataset(
            data_root=str(data_root),
            split=split,
            label_fraction=label_fraction,
            seed=seed,
        )
    else:
        raise FileNotFoundError(
            f"Cannot find Synapse/BTCV data at '{data_root}'.\n"
            "Expected: btcv.json / data/dataset_0.json (BTCV), "
            "imagesTr/ (NIfTI), or train_npz/ (TransUNet NPZ)."
        )
