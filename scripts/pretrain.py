#!/usr/bin/env python3
"""
NEMESIS Pretraining Script

Self-supervised pretraining of the NEMESIS encoder on unlabeled CT volumes
using Masked Autoencoders (MAE) with Noise-Enhanced Dual-Masking.

Usage:
    python scripts/pretrain.py \\
        --config configs/pretrain.yaml \\
        --exp_name NEMESIS_768_m0.5 \\
        --data_json data/combined_dataset.json \\
        --epochs 50 \\
        --batch_size 4 \\
        --mask_ratio 0.5 \\
        --device_ids 0

Resume from checkpoint:
    python scripts/pretrain.py \\
        --config configs/pretrain.yaml \\
        --exp_name NEMESIS_768_m0.5 \\
        --data_json data/combined_dataset.json \\
        --resume experiments/NEMESIS_768_m0.5/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from nemesis.models.mae import MAEgic3DMAE


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CTVolumeDataset(Dataset):
    """
    Dataset that reads NIfTI CT volumes listed in a JSON index file
    and yields randomly cropped 128^3 superpatches with HU normalisation.

    JSON format (created by scripts/create_dataset_json.py):
      {
        "training": [{"image": "rel/path/to/vol.nii.gz"}, ...],
        "validation": [...]
      }
    """

    HU_MIN = -175.0
    HU_MAX  =  250.0

    def __init__(
        self,
        json_path: str,
        base_dir: str,
        split: str = "train",
        superpatch_size: tuple = (128, 128, 128),
        min_foreground_ratio: float = 0.05,
        max_retries: int = 10,
    ):
        self.base_dir           = Path(base_dir)
        self.superpatch_size    = superpatch_size
        self.min_foreground_ratio = min_foreground_ratio
        self.max_retries        = max_retries

        with open(json_path) as f:
            meta = json.load(f)

        key = "training" if split == "train" else "validation"
        entries = meta.get(key, meta.get("data", []))
        self.paths = [self.base_dir / e["image"] for e in entries]
        self.paths = [p for p in self.paths if p.exists()]

        print(f"[CTVolumeDataset] split={split}, volumes={len(self.paths)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        import nibabel as nib

        for attempt in range(self.max_retries):
            try:
                path = self.paths[(idx + attempt) % len(self.paths)]
                vol  = nib.load(str(path)).get_fdata(dtype=np.float32)

                # Ensure (D, H, W) orientation
                if vol.ndim == 3 and vol.shape[2] < vol.shape[0]:
                    vol = vol.transpose(2, 0, 1)

                # HU normalisation → [0, 1]
                vol = np.clip(vol, self.HU_MIN, self.HU_MAX)
                vol = (vol - self.HU_MIN) / (self.HU_MAX - self.HU_MIN)

                D, H, W   = vol.shape
                pd, ph, pw = self.superpatch_size

                if D < pd or H < ph or W < pw:
                    # Pad if volume is smaller than superpatch size
                    pad_d = max(0, pd - D)
                    pad_h = max(0, ph - H)
                    pad_w = max(0, pw - W)
                    vol = np.pad(vol, ((0, pad_d), (0, pad_h), (0, pad_w)))
                    D, H, W = vol.shape

                # Random crop
                d0 = random.randint(0, D - pd)
                h0 = random.randint(0, H - ph)
                w0 = random.randint(0, W - pw)
                patch = vol[d0:d0+pd, h0:h0+ph, w0:w0+pw]

                # Skip empty patches (mostly air)
                if patch.mean() < self.min_foreground_ratio:
                    continue

                tensor = torch.from_numpy(patch[None].astype(np.float32))  # (1, D, H, W)
                return {"image": tensor, "path": str(path)}

            except Exception as e:
                logging.warning(f"[CTVolumeDataset] Error loading {path}: {e}")
                continue

        # Fallback: return zero patch
        pd, ph, pw = self.superpatch_size
        return {"image": torch.zeros(1, pd, ph, pw), "path": ""}


# ---------------------------------------------------------------------------
# Learning rate schedule (cosine with warmup)
# ---------------------------------------------------------------------------

def cosine_schedule_with_warmup(optimizer, warmup_epochs: int, total_epochs: int, min_lr: float):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(min_lr / optimizer.param_groups[0]["initial_lr"],
                   0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir: Path, exp_name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh  = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch  = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def save_checkpoint(path: Path, epoch: int, model: nn.Module,
                    optimizer, scheduler, val_loss: float, exp_name: str,
                    is_best: bool = False):
    state = {
        "epoch":                epoch,
        "model_state_dict":     (model.module.state_dict()
                                 if isinstance(model, nn.DataParallel)
                                 else model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss":             val_loss,
        "exp_name":             exp_name,
    }
    torch.save(state, path)
    if is_best:
        best_path = path.parent / "best_model.pt"
        shutil.copy2(path, best_path)


# ---------------------------------------------------------------------------
# Train / validate one epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model, loader, optimizer, scheduler, device,
    noise_levels, scaler, logger, epoch
) -> dict:
    model.train()
    total_loss = 0.0
    total_psnr = 0.0

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}")
    for batch in pbar:
        images = batch["image"].to(device)         # (B, 1, D, H, W)
        noise_std = random.choice(noise_levels)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            loss_dict, _, _, _ = model(images, noise_std=noise_std)

        loss = loss_dict["total_loss"].mean()
        psnr = loss_dict["psnr"].mean()

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        total_psnr += psnr.item()

        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         psnr=f"{psnr.item():.2f}",
                         lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    n = max(len(loader), 1)
    return {"loss": total_loss / n, "psnr": total_psnr / n}


@torch.no_grad()
def validate_epoch(model, loader, device, logger, epoch) -> dict:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0

    for batch in tqdm(loader, desc=f"[ Val ] Epoch {epoch}"):
        images = batch["image"].to(device)
        loss_dict, _, _, _ = model(images, noise_std=0.0)
        total_loss += loss_dict["total_loss"].mean().item()
        total_psnr += loss_dict["psnr"].mean().item()

    n = max(len(loader), 1)
    return {"loss": total_loss / n, "psnr": total_psnr / n}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="NEMESIS Pretraining")
    parser.add_argument("--config",     type=str,   required=True)
    parser.add_argument("--exp_name",   type=str,   required=True)
    parser.add_argument("--data_json",  type=str,   required=True,
                        help="Path to dataset JSON index (e.g. data/combined_dataset.json)")
    parser.add_argument("--data_base",  type=str,   default="data",
                        help="Base directory for resolving relative image paths in JSON")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--mask_ratio", type=float, default=None,
                        help="Masking ratio for both plane and axis directions")
    parser.add_argument("--embed_dim",  type=int,   default=None,
                        help="Encoder embedding dimension (384 / 576 / 768)")
    parser.add_argument("--device_ids", nargs="+",  type=int, default=[0])
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--amp",        action="store_true",
                        help="Enable FP16 mixed-precision training")
    parser.add_argument("--base_path",  type=str,   default="experiments")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    train_cfg = cfg["training"]
    if args.epochs     is not None: train_cfg["epochs"]     = args.epochs
    if args.batch_size is not None: train_cfg["batch_size"] = args.batch_size
    if args.embed_dim  is not None: cfg["model"]["encoder"]["embed_dim"] = args.embed_dim
    if args.mask_ratio is not None:
        cfg["model"]["masking"]["mask_ratio"] = args.mask_ratio

    set_seed(cfg.get("seed", 42))

    # Setup experiment directory
    exp_dir      = Path(args.base_path) / args.exp_name
    ckpt_dir     = exp_dir / "checkpoints"
    log_dir      = exp_dir / "logs"
    for d in [exp_dir, ckpt_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.config, exp_dir / "config.yaml")

    logger = setup_logging(log_dir, args.exp_name)
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data JSON: {args.data_json}")

    # Device
    if torch.cuda.is_available() and args.device_ids:
        device = torch.device(f"cuda:{args.device_ids[0]}")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}  GPUs: {args.device_ids}")

    # Dataset
    sup_size = tuple(cfg["data"]["superpatch"]["size"])
    train_ds = CTVolumeDataset(
        args.data_json, args.data_base, split="train",
        superpatch_size=sup_size,
        min_foreground_ratio=cfg["data"]["superpatch"].get("min_foreground_ratio", 0.05),
    )
    val_ds = CTVolumeDataset(
        args.data_json, args.data_base, split="val",
        superpatch_size=sup_size,
        min_foreground_ratio=cfg["data"]["superpatch"].get("min_foreground_ratio", 0.05),
    )

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # Model
    enc_cfg  = cfg["model"]["encoder"]
    dec_cfg  = cfg["model"]["decoder"]
    mask_cfg = cfg["model"]["masking"]
    mask_r   = mask_cfg["mask_ratio"]

    model = MAEgic3DMAE(
        superpatch_size=sup_size,
        patch_size=tuple(cfg["data"]["patch"]["size"]),
        in_channels=cfg["data"]["patch"]["channels"],
        embed_dim=enc_cfg["embed_dim"],
        depth=enc_cfg["depth"],
        num_heads=enc_cfg["num_heads"],
        decoder_depth=dec_cfg["depth"],
        decoder_num_heads=dec_cfg["num_heads"],
        dropout=enc_cfg.get("dropout", 0.1),
        drop_path_rate=enc_cfg.get("drop_path_rate", 0.1),
        pos_encoding_type=enc_cfg.get("pos_encoding_type", "sinusoidal"),
        num_maegic_tokens=enc_cfg.get("num_maegic_tokens", 8),
        spatial_mask_ratio=mask_r,    # plane masking
        depth_mask_ratio=mask_r,      # axis masking
        use_maegic_tokens=enc_cfg.get("use_maegic_tokens", True),
    )

    if len(args.device_ids) > 1:
        model = nn.DataParallel(model, device_ids=args.device_ids)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model params: {n_params:.1f}M")
    logger.info(f"Masking: plane={mask_r}, axis={mask_r}  "
                f"(effective combined ≈ {1-(1-mask_r)**2:.3f})")

    # Optimizer + scheduler
    opt_cfg = train_cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]),
    )
    # Set initial_lr for LR lambda
    for pg in optimizer.param_groups:
        pg["initial_lr"] = opt_cfg["lr"]

    sch_cfg   = train_cfg["scheduler"]
    total_steps = train_cfg["epochs"] * len(train_loader)
    scheduler = cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=sch_cfg["warmup_epochs"],
        total_epochs=train_cfg["epochs"],
        min_lr=sch_cfg["min_lr"],
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Resume
    start_epoch  = 1
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(
            ckpt["model_state_dict"]
        )
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        logger.info(f"Resumed from {args.resume} (epoch {ckpt['epoch']})")

    # Noise levels
    noise_levels = cfg.get("noise", {}).get("levels", [0.0, 0.01, 0.03, 0.05])

    # Training loop
    logger.info(f"Starting training: epochs={train_cfg['epochs']}, "
                f"batch={train_cfg['batch_size']}, noise_levels={noise_levels}")

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        t0 = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, noise_levels, scaler, logger, epoch
        )
        val_metrics = validate_epoch(model, val_loader, device, logger, epoch)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        logger.info(
            f"[{epoch:04d}/{train_cfg['epochs']}] "
            f"train_loss={train_metrics['loss']:.4f}  psnr={train_metrics['psnr']:.2f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_psnr={val_metrics['psnr']:.2f}  "
            f"lr={lr_now:.2e}  time={elapsed:.1f}s"
        )

        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            logger.info(f"  *** New best val_loss={best_val_loss:.4f} ***")

        # Save checkpoints
        ckpt_path = ckpt_dir / "latest.pt"
        save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler,
                        val_metrics["loss"], args.exp_name, is_best=is_best)

        if is_best:
            save_checkpoint(
                ckpt_dir / f"best_model_epoch_{epoch:03d}_val.pt",
                epoch, model, optimizer, scheduler,
                val_metrics["loss"], args.exp_name, is_best=False,
            )

        if epoch % args.save_interval == 0:
            save_checkpoint(
                ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                epoch, model, optimizer, scheduler,
                val_metrics["loss"], args.exp_name,
            )

    logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    logger.info(f"Best checkpoint: {ckpt_dir}/best_model.pt")


if __name__ == "__main__":
    main()
