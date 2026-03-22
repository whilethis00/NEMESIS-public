"""
Classification trainer for NEMESIS multi-label organ presence benchmark.

Features:
  - BCEWithLogitsLoss (multi-label)
  - Per-class AUROC + mean AUROC (sklearn)
  - F1 score at threshold=0.5 (per-class + mean)
  - Best model saved by mean AUROC
  - CSV logging: epoch, train_loss, val_mean_auroc, val_mean_f1, per-class AUROC
  - TensorBoard logging
  - AMP support (optional)
  - Early stopping
  - Resume / finetune support
"""

from __future__ import annotations

import csv
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, ModuleNotFoundError):
    class SummaryWriter:  # no-op stub when tensorboard is not installed
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass

try:
    from sklearn.metrics import roc_auc_score, f1_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    print("[ClsTrainer] WARNING: sklearn not available; AUROC and F1 will not be computed.")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ClsTrainer:
    """
    Multi-label classification trainer for NEMESIS superpatch benchmark.

    Args:
        model         : nn.Module accepting (B, 1, D, H, W) → (B, num_classes) logits
        train_loader  : DataLoader yielding batches with keys 'image', 'label'
        val_loader    : DataLoader yielding volumes with keys 'patches', 'labels', 'case'
        num_classes   : number of classification organs (8)
        output_dir    : directory for checkpoints, logs, CSV
        cfg           : training hyperparameters dict (see DEFAULT_CFG)
        class_names   : list of class name strings for logging
        resume_checkpoint  : path to a full trainer checkpoint (continues training)
        finetune_checkpoint: path to model-weights-only checkpoint (restarts from epoch 1)
    """

    DEFAULT_CFG: dict = dict(
        lr                  = 1e-3,
        weight_decay        = 1e-5,
        max_epochs          = 100,
        warmup_epochs       = 5,
        val_every           = 5,
        amp                 = False,
        grad_clip           = 1.0,
        early_stop_patience = 20,   # 0 = disabled; counts val rounds without improvement
        cls_threshold       = 0.5,  # sigmoid threshold for F1 / binary decision
    )

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        output_dir: str,
        cfg: Optional[dict] = None,
        class_names: Optional[List[str]] = None,
        resume_checkpoint: Optional[str] = None,
        finetune_checkpoint: Optional[str] = None,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.num_classes  = num_classes
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cfg = {**self.DEFAULT_CFG, **(cfg or {})}
        self.class_names = class_names or [f"organ_{i}" for i in range(num_classes)]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = self.model.to(self.device)

        # DataParallel: use all available GPUs automatically
        if torch.cuda.device_count() > 1:
            print(f"[ClsTrainer] Using DataParallel across {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer (only trains parameters that require gradients)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )

        # Cosine LR schedule
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg["max_epochs"],
            eta_min=self.cfg["lr"] * 1e-3,
        )

        # AMP scaler
        self.scaler = GradScaler(enabled=self.cfg["amp"])

        # Logging
        self.writer   = SummaryWriter(log_dir=str(self.output_dir / "tb_logs"))
        self.csv_path = self.output_dir / "metrics.csv"
        self.logger   = self._init_logger()

        self.best_mean_auroc = 0.0
        self.best_epoch      = 0
        self._no_improve     = 0

        # Handle resume / finetune
        self.start_epoch = 1
        if resume_checkpoint is not None:
            self._resume(resume_checkpoint)
        elif finetune_checkpoint is not None:
            self._finetune(finetune_checkpoint)

        # Init CSV (after resume so we know whether to append)
        self._init_csv(append=(resume_checkpoint is not None))

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _init_logger(self) -> logging.Logger:
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path  = log_dir / f"train_{timestamp}.log"

        logger = logging.getLogger(str(self.output_dir))
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh  = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        ch  = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        logger.info(f"Log file: {log_path}")
        return logger

    def _init_csv(self, append: bool = False):
        if append and self.csv_path.exists():
            return
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            per_class_headers = [f"auroc_{n}" for n in self.class_names]
            f1_headers        = [f"f1_{n}" for n in self.class_names]
            header = (["epoch", "train_loss", "val_mean_auroc", "val_mean_f1", "lr"]
                      + per_class_headers + f1_headers)
            w.writerow(header)

    def _log_csv(self, row: dict):
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            per_class_auroc = [row.get(f"auroc_{n}", "") for n in self.class_names]
            per_class_f1    = [row.get(f"f1_{n}", "")    for n in self.class_names]
            w.writerow([
                row.get("epoch", ""),
                row.get("train_loss", ""),
                row.get("val_mean_auroc", ""),
                row.get("val_mean_f1", ""),
                row.get("lr", ""),
            ] + per_class_auroc + per_class_f1)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> float:
        """Run full training loop. Returns best mean AUROC."""
        for epoch in range(self.start_epoch, self.cfg["max_epochs"] + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("train/lr",   lr,         epoch)

            row = {
                "epoch":      epoch,
                "train_loss": f"{train_loss:.6f}",
                "lr":         f"{lr:.2e}",
            }

            # Validation
            do_val = (epoch % self.cfg["val_every"] == 0
                      or epoch == self.cfg["max_epochs"])
            if do_val:
                val_metrics = self._val_epoch(epoch)
                mean_auroc  = val_metrics["mean_auroc"]
                mean_f1     = val_metrics["mean_f1"]

                self.writer.add_scalar("val/mean_auroc", mean_auroc, epoch)
                self.writer.add_scalar("val/mean_f1",    mean_f1,    epoch)
                for i, name in enumerate(self.class_names):
                    self.writer.add_scalar(f"val/auroc_{name}",
                                           val_metrics["per_class_auroc"][i], epoch)

                self.logger.info(
                    f"[Epoch {epoch:04d}/{self.cfg['max_epochs']}] "
                    f"loss={train_loss:.4f}  "
                    f"mean_auroc={mean_auroc:.4f}  "
                    f"mean_f1={mean_f1:.4f}  "
                    f"lr={lr:.2e}  "
                    f"time={time.time()-t0:.1f}s"
                )

                # Per-class
                auroc_strs = [f"{n}={val_metrics['per_class_auroc'][i]:.3f}"
                              for i, n in enumerate(self.class_names)]
                self.logger.info("  AUROC per class: " + "  ".join(auroc_strs))

                row.update({
                    "val_mean_auroc": f"{mean_auroc:.4f}",
                    "val_mean_f1":    f"{mean_f1:.4f}",
                    **{f"auroc_{n}": f"{val_metrics['per_class_auroc'][i]:.4f}"
                       for i, n in enumerate(self.class_names)},
                    **{f"f1_{n}": f"{val_metrics['per_class_f1'][i]:.4f}"
                       for i, n in enumerate(self.class_names)},
                })

                # Save best model (by mean AUROC)
                if mean_auroc > self.best_mean_auroc:
                    self.best_mean_auroc = mean_auroc
                    self.best_epoch      = epoch
                    self._no_improve     = 0
                    self._save_checkpoint(epoch, mean_auroc, is_best=True)
                    self.logger.info(
                        f"  *** New best mean AUROC: {mean_auroc:.4f} at epoch {epoch} ***"
                    )
                else:
                    self._no_improve += 1
                    patience = self.cfg["early_stop_patience"]
                    if patience > 0:
                        self.logger.info(
                            f"  [EarlyStopping] no improve {self._no_improve}/{patience}"
                        )
                        if self._no_improve >= patience:
                            self.logger.info(
                                f"  [EarlyStopping] triggered at epoch {epoch}, stopping."
                            )
                            self._log_csv(row)
                            break

                # Periodic checkpoint every 5 val cycles
                if epoch % (self.cfg["val_every"] * 5) == 0:
                    self._save_checkpoint(epoch, mean_auroc, is_best=False)

            else:
                self.logger.info(
                    f"[Epoch {epoch:04d}/{self.cfg['max_epochs']}] "
                    f"loss={train_loss:.4f}  lr={lr:.2e}  time={time.time()-t0:.1f}s"
                )

            self._log_csv(row)

        self.writer.close()
        self.logger.info(
            f"\n[Done] Best mean AUROC: {self.best_mean_auroc:.4f} @ epoch {self.best_epoch}"
        )
        return self.best_mean_auroc

    # ------------------------------------------------------------------
    # Epoch implementations
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(self.train_loader, 1):
            images = batch["image"].to(self.device)   # (B, 1, D, H, W)
            labels = batch["label"].to(self.device)   # (B, num_classes) float

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.cfg["amp"]):
                logits = self.model(images)            # (B, num_classes)
                loss   = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            if self.cfg["grad_clip"] > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg["grad_clip"]
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / max(len(self.train_loader), 1)

    def _val_epoch(self, epoch: int) -> dict:
        """
        Validate by running all superpatches of each volume and aggregating
        predictions at the superpatch level.

        Returns:
            dict with keys: mean_auroc, mean_f1, per_class_auroc, per_class_f1
        """
        self.model.eval()

        # Collect all predictions and targets
        all_probs:   List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        threshold = self.cfg["cls_threshold"]

        with torch.no_grad():
            for batch in self.val_loader:
                # Val loader returns dict with 'patches' (N_sp, 1, D, H, W) per volume
                patches = batch["patches"]   # (1, N_sp, 1, D, H, W) with batch_size=1
                targets = batch["labels"]    # (1, N_sp, num_classes)

                # Remove batch dim (batch_size=1 for val)
                patches = patches.squeeze(0)   # (N_sp, 1, D, H, W)
                targets = targets.squeeze(0)   # (N_sp, num_classes)

                # Process superpatches one at a time to avoid OOM
                batch_probs = []
                for sp_idx in range(patches.shape[0]):
                    sp = patches[sp_idx:sp_idx+1].to(self.device)  # (1, 1, D, H, W)
                    logits = self.model(sp)                          # (1, num_classes)
                    probs  = torch.sigmoid(logits).cpu()             # (1, num_classes)
                    batch_probs.append(probs)

                batch_probs = torch.cat(batch_probs, dim=0)  # (N_sp, num_classes)

                all_probs.append(batch_probs.numpy())
                all_targets.append(targets.numpy())

        # Concatenate across all volumes
        all_probs   = np.concatenate(all_probs,   axis=0)   # (total_sp, num_classes)
        all_targets = np.concatenate(all_targets, axis=0)   # (total_sp, num_classes)

        # Compute per-class AUROC
        per_class_auroc = np.zeros(self.num_classes, dtype=np.float32)
        per_class_f1    = np.zeros(self.num_classes, dtype=np.float32)

        if _SKLEARN_AVAILABLE:
            for c in range(self.num_classes):
                y_true = all_targets[:, c]
                y_score = all_probs[:, c]
                y_pred  = (y_score >= threshold).astype(int)

                # AUROC requires both classes present
                if y_true.sum() > 0 and (1 - y_true).sum() > 0:
                    per_class_auroc[c] = roc_auc_score(y_true, y_score)
                else:
                    # Degenerate case: all positive or all negative
                    per_class_auroc[c] = float("nan")

                per_class_f1[c] = f1_score(y_true, y_pred, zero_division=0)
        else:
            per_class_auroc[:] = float("nan")
            per_class_f1[:]    = 0.0

        # Mean AUROC (exclude NaN classes)
        valid_auroc = per_class_auroc[~np.isnan(per_class_auroc)]
        mean_auroc  = float(np.mean(valid_auroc)) if len(valid_auroc) > 0 else 0.0
        mean_f1     = float(np.mean(per_class_f1))

        return {
            "mean_auroc":      mean_auroc,
            "mean_f1":         mean_f1,
            "per_class_auroc": per_class_auroc.tolist(),
            "per_class_f1":    per_class_f1.tolist(),
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, metric: float, is_best: bool):
        name = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch:04d}.pt"
        path = self.output_dir / name
        # Unwrap DataParallel if needed
        model_state = (self.model.module.state_dict()
                       if isinstance(self.model, nn.DataParallel)
                       else self.model.state_dict())
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_mean_auroc":      metric,
        }, path)

    def load_best_checkpoint(self):
        path = self.output_dir / "best_model.pt"
        if not path.exists():
            raise FileNotFoundError(f"No best_model.pt in {self.output_dir}")
        ckpt = torch.load(path, map_location=self.device)
        self._unwrapped_model().load_state_dict(ckpt["model_state_dict"])
        self.logger.info(
            f"[ClsTrainer] Loaded best model "
            f"(epoch {ckpt['epoch']}, AUROC {ckpt['best_mean_auroc']:.4f})"
        )

    def _unwrapped_model(self):
        """Return the raw model, unwrapping DataParallel if needed."""
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _finetune(self, checkpoint_path: str):
        """Load model weights only; restart optimizer and training from epoch 1."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Finetune checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self._unwrapped_model().load_state_dict(ckpt["model_state_dict"])
        self.logger.info(
            f"[Finetune] Loaded weights from {path} (epoch {ckpt.get('epoch', 0)}), "
            "restarting training from epoch 1."
        )

    def _resume(self, checkpoint_path: str):
        """Resume training from a full checkpoint (model + optimizer + epoch)."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self._unwrapped_model().load_state_dict(ckpt["model_state_dict"])
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError:
            self.logger.info("[Resume] optimizer groups mismatch, resetting optimizer state.")
        epoch = ckpt.get("epoch", 0)
        self.start_epoch      = epoch + 1
        self.best_mean_auroc  = ckpt.get("best_mean_auroc", 0.0)
        self.best_epoch       = epoch
        # Fast-forward scheduler
        for _ in range(epoch):
            self.scheduler.step()
        self.logger.info(
            f"[Resume] Loaded checkpoint epoch={epoch}, "
            f"best_auroc={self.best_mean_auroc:.4f}, "
            f"resuming from epoch {self.start_epoch}."
        )


# ---------------------------------------------------------------------------
# Standalone evaluation helper
# ---------------------------------------------------------------------------

def evaluate_cls(
    model: nn.Module,
    val_loader: DataLoader,
    num_classes: int,
    class_names: List[str],
    threshold: float = 0.5,
    device: str = "cuda",
) -> dict:
    """
    Run evaluation on val_loader and return metrics dict.

    Returns:
        {
          'mean_auroc': float,
          'mean_f1':    float,
          'per_class_auroc': list of float,
          'per_class_f1':    list of float,
        }
    """
    device_ = torch.device(device if torch.cuda.is_available() else "cpu")
    model   = model.to(device_).eval()

    all_probs   = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            patches = batch["patches"].squeeze(0).to(device_)   # (N_sp, 1, D, H, W)
            targets = batch["labels"].squeeze(0)                 # (N_sp, num_classes)

            batch_probs = []
            for sp_idx in range(patches.shape[0]):
                sp     = patches[sp_idx:sp_idx+1]
                logits = model(sp)
                probs  = torch.sigmoid(logits).cpu()
                batch_probs.append(probs)

            batch_probs = torch.cat(batch_probs, dim=0)
            all_probs.append(batch_probs.numpy())
            all_targets.append(targets.numpy())

    all_probs   = np.concatenate(all_probs,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    per_class_auroc = np.zeros(num_classes, dtype=np.float32)
    per_class_f1    = np.zeros(num_classes, dtype=np.float32)

    if _SKLEARN_AVAILABLE:
        for c in range(num_classes):
            y_true  = all_targets[:, c]
            y_score = all_probs[:, c]
            y_pred  = (y_score >= threshold).astype(int)
            if y_true.sum() > 0 and (1 - y_true).sum() > 0:
                per_class_auroc[c] = roc_auc_score(y_true, y_score)
            else:
                per_class_auroc[c] = float("nan")
            per_class_f1[c] = f1_score(y_true, y_pred, zero_division=0)

    valid = per_class_auroc[~np.isnan(per_class_auroc)]
    mean_auroc = float(np.mean(valid)) if len(valid) > 0 else 0.0
    mean_f1    = float(np.mean(per_class_f1))

    print(f"\n{'Organ':<20} {'AUROC':>8} {'F1':>8}")
    print("-" * 40)
    for c, name in enumerate(class_names):
        auroc = per_class_auroc[c]
        f1_v  = per_class_f1[c]
        auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "  NaN "
        print(f"  {name:<18} {auroc_str:>8} {f1_v:>8.4f}")
    print("-" * 40)
    print(f"  {'Mean':<18} {mean_auroc:>8.4f} {mean_f1:>8.4f}")

    return {
        "mean_auroc":      mean_auroc,
        "mean_f1":         mean_f1,
        "per_class_auroc": per_class_auroc.tolist(),
        "per_class_f1":    per_class_f1.tolist(),
    }
