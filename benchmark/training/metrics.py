"""
Evaluation metrics for 3D medical image segmentation.

  - dice_per_class    : per-class Dice coefficient
  - hd95_per_class    : per-class 95th-percentile Hausdorff distance
  - compute_metrics   : combine both, return summary dict
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import label as nd_label
from scipy.spatial.distance import directed_hausdorff


# ---------------------------------------------------------------------------
# Dice
# ---------------------------------------------------------------------------

def dice_per_class(
    pred: np.ndarray,   # (D, H, W)  int labels
    target: np.ndarray, # (D, H, W)  int labels
    num_classes: int,
    ignore_bg: bool = True,
) -> np.ndarray:
    """
    Returns array of Dice scores shape (num_classes,).
    Background (class 0) is computed but can be excluded from the mean.
    """
    scores = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        p = (pred   == c).astype(np.float64)
        t = (target == c).astype(np.float64)
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        if denom < 1e-5:
            scores[c] = np.nan  # both pred & GT empty → exclude from mean
        else:
            scores[c] = 2.0 * inter / denom
    return scores


# ---------------------------------------------------------------------------
# HD95
# ---------------------------------------------------------------------------

def _surface_points(mask: np.ndarray) -> np.ndarray:
    """Return (N, 3) array of voxel coordinates on the binary mask surface."""
    from scipy.ndimage import binary_erosion
    surface = mask & ~binary_erosion(mask)
    return np.column_stack(np.where(surface))


def hd95_binary(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    """
    Symmetric HD95 between two binary masks.
    Returns np.inf if either mask is empty.
    """
    if pred_mask.sum() == 0 or target_mask.sum() == 0:
        return np.inf

    p_pts = _surface_points(pred_mask.astype(bool))
    t_pts = _surface_points(target_mask.astype(bool))

    if len(p_pts) == 0 or len(t_pts) == 0:
        return np.inf

    d_pt   = directed_hausdorff(p_pts, t_pts)[0]
    d_tp   = directed_hausdorff(t_pts, p_pts)[0]

    # Proper HD95 via all pairwise distances (brute-force, slow for large masks)
    # Fast approximation: use directed distances & take 95th pct
    from scipy.spatial import cKDTree
    tree_t = cKDTree(t_pts)
    tree_p = cKDTree(p_pts)

    dists_pt, _ = tree_t.query(p_pts, k=1)
    dists_tp, _ = tree_p.query(t_pts, k=1)

    all_dists = np.concatenate([dists_pt, dists_tp])
    return float(np.percentile(all_dists, 95))


def hd95_per_class(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Returns HD95 (mm) per class, shape (num_classes,).
    Class 0 (background) always returns np.inf (excluded from mean later).
    """
    scores = np.full(num_classes, np.inf, dtype=np.float64)
    scale  = np.array(voxel_spacing)   # apply to convert voxel → mm

    for c in range(1, num_classes):
        p = (pred   == c)
        t = (target == c)
        if p.sum() == 0 and t.sum() == 0:
            scores[c] = np.nan  # both empty → exclude from mean
            continue
        # Scale coordinates
        p_pts = _surface_points(p) * scale
        t_pts = _surface_points(t) * scale
        if len(p_pts) == 0 or len(t_pts) == 0:
            scores[c] = np.inf
            continue

        from scipy.spatial import cKDTree
        tree_t = cKDTree(t_pts)
        tree_p = cKDTree(p_pts)
        d1, _ = tree_t.query(p_pts, k=1)
        d2, _ = tree_p.query(t_pts, k=1)
        scores[c] = float(np.percentile(np.concatenate([d1, d2]), 95))

    return scores


# ---------------------------------------------------------------------------
# Combined metric computation
# ---------------------------------------------------------------------------

SYNAPSE_ORGAN_NAMES = [
    "background", "aorta", "gallbladder", "spleen",
    "left_kidney", "right_kidney", "liver", "stomach", "pancreas",
]


def compute_metrics(
    pred_logits: torch.Tensor,   # (B, C, D, H, W)  raw logits
    target: torch.Tensor,        # (B, D, H, W)      int labels
    num_classes: int,
    class_names: list | None = None,
    compute_hd95: bool = True,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
) -> dict:
    """
    Compute Dice (all cases) and HD95 (first case only for speed) batch-wise.

    Returns dict with keys:
      'dice'          : (C,) mean Dice over batch
      'mean_dice'     : scalar, mean over foreground classes
      'hd95'          : (C,) HD95 (or None if skipped)
      'mean_hd95'     : scalar
      'per_class'     : dict of {class_name: {dice: float, hd95: float}}
    """
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    pred_hard = pred_logits.argmax(dim=1).cpu().numpy()   # (B, D, H, W)
    tgt_np    = target.cpu().numpy()

    all_dice = []
    all_hd95 = []

    for b in range(pred_hard.shape[0]):
        d = dice_per_class(pred_hard[b], tgt_np[b], num_classes)
        all_dice.append(d)
        if compute_hd95 and b == 0:   # HD95 is slow; compute only for first sample
            h = hd95_per_class(pred_hard[b], tgt_np[b], num_classes, voxel_spacing)
            all_hd95.append(h)

    dice_mean = np.mean(all_dice, axis=0)  # (C,)
    hd95_mean = np.mean(all_hd95, axis=0) if all_hd95 else np.full(num_classes, np.nan)

    fg = list(range(1, num_classes))   # exclude background
    mean_dice = float(np.mean(dice_mean[fg]))
    valid_hd  = hd95_mean[fg]
    valid_hd  = valid_hd[np.isfinite(valid_hd)]
    mean_hd95 = float(np.mean(valid_hd)) if len(valid_hd) > 0 else np.nan

    per_class = {}
    for c, name in enumerate(class_names):
        if c == 0:
            continue
        per_class[name] = {
            "dice": float(dice_mean[c]),
            "hd95": float(hd95_mean[c]),
        }

    return {
        "dice":      dice_mean,
        "mean_dice": mean_dice,
        "hd95":      hd95_mean,
        "mean_hd95": mean_hd95,
        "per_class": per_class,
    }


def format_metrics(metrics: dict, class_names: list | None = None) -> str:
    """Pretty-print metrics dict."""
    lines = [
        f"Mean Dice : {metrics['mean_dice']*100:.2f}%",
        f"Mean HD95 : {metrics['mean_hd95']:.2f} mm",
        "Per-class Dice:",
    ]
    for name, vals in metrics["per_class"].items():
        lines.append(f"  {name:<20} Dice={vals['dice']*100:.2f}%  HD95={vals['hd95']:.2f}mm")
    return "\n".join(lines)
