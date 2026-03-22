"""
BTCVClassificationDataset - Multi-label organ presence classification per superpatch.

Each 128³ superpatch is labeled with a binary vector of length 8 indicating
presence of the 8 Synapse organs [aorta, gallbladder, spleen, left_kidney,
right_kidney, liver, stomach, pancreas] (BTCV label indices 8, 4, 1, 3, 2, 6, 7, 11).

Preprocessing:
  - HU clip [-175, 250], normalize to [0, 1]  (matches task specification)
  - Non-overlapping 128³ superpatches, padded as needed

Split:
  - 24 train / 6 val (dataset_0.json)

Usage:
    train_ds = BTCVClassificationDataset(data_root, split="train")
    val_ds   = BTCVClassificationDataset(data_root, split="val")
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HU window (task specification)
HU_MIN, HU_MAX = -175.0, 250.0

# Classification organs (8 classes), ordered as specified
CLS_ORGAN_NAMES = [
    "aorta",        # BTCV label 8
    "gallbladder",  # BTCV label 4
    "spleen",       # BTCV label 1
    "left_kidney",  # BTCV label 3
    "right_kidney", # BTCV label 2
    "liver",        # BTCV label 6
    "stomach",      # BTCV label 7
    "pancreas",     # BTCV label 11
]
NUM_CLS_ORGANS = 8  # length of binary label vector

# BTCV raw label indices for each classification organ (parallel to CLS_ORGAN_NAMES)
BTCV_ORGAN_INDICES = [8, 4, 1, 3, 2, 6, 7, 11]

# Minimum voxel count to declare an organ "present" in a superpatch
PRESENCE_THRESHOLD = 100


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def normalize_ct(volume: np.ndarray) -> np.ndarray:
    """Clip to HU window [-175, 250] and scale to [0, 1]."""
    volume = np.clip(volume, HU_MIN, HU_MAX)
    return (volume - HU_MIN) / (HU_MAX - HU_MIN)


def compute_organ_labels(
    label_patch: np.ndarray,
    threshold: int = PRESENCE_THRESHOLD,
) -> np.ndarray:
    """
    Compute binary label vector for one superpatch.

    Args:
        label_patch : integer label array, shape (D, H, W)
        threshold   : minimum voxel count to declare organ present

    Returns:
        labels : float32 array of shape (NUM_CLS_ORGANS,), values in {0, 1}
    """
    labels = np.zeros(NUM_CLS_ORGANS, dtype=np.float32)
    for i, btcv_idx in enumerate(BTCV_ORGAN_INDICES):
        if np.count_nonzero(label_patch == btcv_idx) > threshold:
            labels[i] = 1.0
    return labels


def extract_superpatches(
    img: np.ndarray,
    lbl: np.ndarray,
    superpatch_size: Tuple[int, int, int] = (128, 128, 128),
    threshold: int = PRESENCE_THRESHOLD,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Divide a volume into non-overlapping superpatch_size³ patches.

    Padding is applied to the right/bottom/back if necessary. Returns a list of
    (image_tensor, label_vector) pairs per superpatch.

    Args:
        img            : float32 CT array (D, H, W), already normalized
        lbl            : int64 label array (D, H, W), raw BTCV indices
        superpatch_size: target patch size (must divide evenly after padding)
        threshold      : minimum voxels for organ presence

    Returns:
        List of (image_tensor (1,D,H,W), label_tensor (NUM_CLS_ORGANS,))
    """
    pd, ph, pw = superpatch_size
    D, H, W = img.shape

    # Compute padded size (ceiling to next multiple of superpatch_size)
    D_pad = int(np.ceil(D / pd)) * pd
    H_pad = int(np.ceil(H / ph)) * ph
    W_pad = int(np.ceil(W / pw)) * pw

    # Pad if necessary
    if (D_pad, H_pad, W_pad) != (D, H, W):
        pad_d = D_pad - D
        pad_h = H_pad - H
        pad_w = W_pad - W
        img = np.pad(img, ((0, pad_d), (0, pad_h), (0, pad_w)),
                     mode="constant", constant_values=0.0)
        lbl = np.pad(lbl, ((0, pad_d), (0, pad_h), (0, pad_w)),
                     mode="constant", constant_values=0)

    samples = []
    for d in range(0, D_pad, pd):
        for h in range(0, H_pad, ph):
            for w in range(0, W_pad, pw):
                img_patch = img[d:d+pd, h:h+ph, w:w+pw]
                lbl_patch = lbl[d:d+pd, h:h+ph, w:w+pw]

                img_t = torch.from_numpy(img_patch[None].astype(np.float32))  # (1, D, H, W)
                lbl_t = torch.from_numpy(compute_organ_labels(lbl_patch, threshold))  # (8,)

                samples.append((img_t, lbl_t))

    return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BTCVClassificationDataset(Dataset):
    """
    Multi-label organ presence classification dataset built from BTCV volumes.

    Training mode:
        Returns individual superpatches as flat samples.
        __getitem__ returns a single (image_tensor, label_tensor) pair.

    Validation mode:
        Returns all superpatches of one volume grouped together.
        __getitem__ returns a dict with keys 'patches' (N, 1, D, H, W),
        'labels' (N, 8), and 'case' (str).
        This enables volume-level metric aggregation.

    Args:
        data_root      : path to BTCV root containing data/dataset_0.json
        split          : 'train' or 'val'
        superpatch_size: size of each superpatch (must be (128,128,128) to match encoder)
        num_classes    : number of organ classes (8)
        label_fraction : fraction of labeled training cases (semi-supervised)
        seed           : random seed for label fraction sampling
        threshold      : minimum voxel count to declare organ present
    """

    DEFAULT_DATA_ROOT = "data/BTCV"
    )

    def __init__(
        self,
        data_root: str = DEFAULT_DATA_ROOT,
        split: str = "train",
        superpatch_size: Tuple[int, int, int] = (128, 128, 128),
        num_classes: int = NUM_CLS_ORGANS,
        label_fraction: float = 1.0,
        seed: int = 42,
        threshold: int = PRESENCE_THRESHOLD,
    ):
        super().__init__()
        assert split in ("train", "val"), f"Invalid split: '{split}'. Choose 'train' or 'val'."

        self.split = split
        self.superpatch_size = tuple(superpatch_size)
        self.num_classes = num_classes
        self.threshold = threshold
        self.data_root = Path(data_root)

        # Load JSON split (dataset_0.json: 24 train, 6 val)
        json_path = self.data_root / "data" / "dataset_0.json"
        if not json_path.exists():
            json_path = self.data_root / "btcv.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"No dataset JSON found in {self.data_root}. "
                "Expected data/dataset_0.json or btcv.json."
            )

        with open(json_path) as f:
            meta = json.load(f)

        pairs = meta["training"] if split == "train" else meta.get("validation", [])

        # Semi-supervised: subset of labeled cases
        if split == "train" and label_fraction < 1.0:
            rng = random.Random(seed)
            n = max(1, int(len(pairs) * label_fraction))
            pairs = rng.sample(pairs, n)
            print(f"[BTCVCls] label_fraction={label_fraction*100:.0f}% "
                  f"→ {n}/{len(meta['training'])} training cases")

        # Resolve file paths
        self.case_files: List[dict] = []
        for p in pairs:
            img_path = self._resolve(p["image"])
            lbl_path = self._resolve(p["label"])
            if not img_path.exists() or not lbl_path.exists():
                print(f"[BTCVCls] WARNING: missing files for {p['image']}, skipping.")
                continue
            case_id = Path(p["image"]).stem  # e.g. 'img0001'
            self.case_files.append({
                "image": str(img_path),
                "label": str(lbl_path),
                "case":  case_id,
            })

        print(f"[BTCVCls] split={split}, volumes={len(self.case_files)}, "
              f"superpatch_size={self.superpatch_size}, threshold={threshold}")

        # For training: pre-load all superpatches as a flat list
        # For val: keep volume-level grouping (lazy loading at __getitem__)
        if split == "train":
            self._flat_samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
            self._load_all_train_patches()
        else:
            # Val: index by volume; patches loaded lazily
            self._flat_samples = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, rel_path: str) -> Path:
        """Resolve relative JSON path against possible base directories."""
        for base in [self.data_root, self.data_root / "data"]:
            p = base / rel_path
            if p.exists():
                return p
        return self.data_root / rel_path  # will fail later with a clear error

    def _load_volume(self, img_path: str, lbl_path: str):
        """Load and preprocess a volume + label pair. Returns (img, lbl) numpy arrays."""
        import nibabel as nib

        img = nib.load(img_path).get_fdata(dtype=np.float32)
        lbl = nib.load(lbl_path).get_fdata(dtype=np.float32)

        # Ensure (D, H, W) orientation (NIfTI is often stored as H, W, D)
        if img.ndim == 3 and img.shape[2] < img.shape[0]:
            img = img.transpose(2, 0, 1)
            lbl = lbl.transpose(2, 0, 1)

        img = normalize_ct(img)
        lbl = np.round(lbl).astype(np.int64)

        return img, lbl

    def _load_all_train_patches(self):
        """Pre-load all volumes and extract superpatches for training."""
        print(f"[BTCVCls] Loading {len(self.case_files)} training volumes and "
              f"extracting superpatches...")
        for cf in self.case_files:
            img, lbl = self._load_volume(cf["image"], cf["label"])
            patches = extract_superpatches(img, lbl, self.superpatch_size, self.threshold)
            self._flat_samples.extend(patches)
        print(f"[BTCVCls] Total training superpatches: {len(self._flat_samples)}")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self.split == "train":
            return len(self._flat_samples)
        else:
            return len(self.case_files)

    def __getitem__(self, idx: int):
        if self.split == "train":
            img_t, lbl_t = self._flat_samples[idx]
            return {"image": img_t, "label": lbl_t}

        else:
            # Validation: return all superpatches of one volume
            cf = self.case_files[idx]
            img, lbl = self._load_volume(cf["image"], cf["label"])
            patches = extract_superpatches(img, lbl, self.superpatch_size, self.threshold)

            images = torch.stack([p[0] for p in patches], dim=0)   # (N, 1, D, H, W)
            labels = torch.stack([p[1] for p in patches], dim=0)   # (N, 8)

            return {
                "patches": images,
                "labels":  labels,
                "case":    cf["case"],
            }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_cls_dataset(
    data_root: str,
    split: str = "train",
    superpatch_size: Tuple[int, int, int] = (128, 128, 128),
    num_classes: int = NUM_CLS_ORGANS,
    label_fraction: float = 1.0,
    seed: int = 42,
    threshold: int = PRESENCE_THRESHOLD,
) -> BTCVClassificationDataset:
    """Convenience factory for classification dataset."""
    return BTCVClassificationDataset(
        data_root=data_root,
        split=split,
        superpatch_size=superpatch_size,
        num_classes=num_classes,
        label_fraction=label_fraction,
        seed=seed,
        threshold=threshold,
    )


# Alias for backward compatibility
build_btcv_cls_dataset = build_cls_dataset
