"""
Training and evaluation loop for NEMESIS segmentation benchmark.

Features:
  - Mixed-precision (AMP) training
  - DiceCELoss (MONAI)
  - Sliding-window inference (MONAI)
  - Per-epoch Dice + HD95 logging
  - Best-checkpoint saving
  - TensorBoard + CSV logging
"""

from __future__ import annotations

import csv
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from .metrics import compute_metrics, format_metrics, SYNAPSE_ORGAN_NAMES


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SegTrainer:
    """
    Generic 3D segmentation trainer.

    Args:
        model         : any nn.Module that accepts (B,1,D,H,W) → (B,C,D,H,W)
        train_loader  : training DataLoader  (batches have 'image','label' keys)
        val_loader    : validation DataLoader
        num_classes   : number of segmentation classes (including BG)
        output_dir    : directory to save checkpoints and logs
        cfg           : dict with training hyperparameters (see defaults below)
        class_names   : list of class name strings for logging
    """

    DEFAULT_CFG = dict(
        lr                = 1e-4,
        weight_decay      = 1e-5,
        max_epochs        = 500,
        warmup_epochs     = 10,
        val_every         = 10,
        sw_roi_size       = (128, 128, 128),
        sw_overlap        = 0.5,
        amp               = True,
        grad_clip         = 1.0,
        dice_ce_lambda    = 1.0,   # weight balance (DiceLoss weight = 1, CELoss weight = 1)
        early_stop_patience = 0,   # 0 = disabled
        compute_hd95        = False,  # HD95 is slow; enable only for final eval
    )

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        output_dir: str,
        cfg: Optional[dict] = None,
        class_names: Optional[list] = None,
        resume_checkpoint: Optional[str] = None,
    finetune_checkpoint: Optional[str] = None,  # weights only, restart training from epoch 1
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.num_classes  = num_classes
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cfg = {**self.DEFAULT_CFG, **(cfg or {})}
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)

        # Loss — class-weighted CE to prevent all-background collapse
        # background gets weight 0.1, foreground classes get 1.0
        ce_weight = torch.ones(num_classes)
        ce_weight[0] = 0.1
        self.criterion = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
            lambda_dice=1.0,
            lambda_ce=1.0,
            ce_weight=ce_weight.to(self.device),
        )

        # Optimizer
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg["max_epochs"],
            eta_min=self.cfg["lr"] * 1e-3,
        )

        # AMP scaler
        self.scaler = GradScaler(enabled=self.cfg["amp"])

        # MONAI metrics
        self.post_pred   = AsDiscrete(argmax=True, to_onehot=num_classes)
        self.post_label  = AsDiscrete(to_onehot=num_classes)
        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean_batch",
            get_not_nans=True,
        )

        # Logging
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb_logs"))
        self.csv_path = self.output_dir / "metrics.csv"
        self._init_csv(append=(resume_checkpoint is not None))
        self.logger = self._init_logger()

        self.best_mean_dice = 0.0
        self.best_epoch     = 0
        self._no_improve    = 0

        # Resume from checkpoint if provided
        self.start_epoch = 1
        if resume_checkpoint is not None:
            self._resume(resume_checkpoint)
        elif finetune_checkpoint is not None:
            self._finetune(finetune_checkpoint)

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

        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        logger.info(f"Log file: {log_path}")
        return logger

    # ------------------------------------------------------------------
    def _init_csv(self, append: bool = False):
        if append and self.csv_path.exists():
            return  # keep existing CSV
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            header = ["epoch", "train_loss", "val_mean_dice", "val_mean_hd95",
                      "lr"] + [f"dice_{n}" for n in self.class_names[1:]]
            w.writerow(header)

    def _log_csv(self, row: dict):
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([row.get(k, "") for k in
                        ["epoch", "train_loss", "val_mean_dice", "val_mean_hd95", "lr"]
                        + [f"dice_{n}" for n in self.class_names[1:]]])

    # ------------------------------------------------------------------
    def train(self):
        for epoch in range(self.start_epoch, self.cfg["max_epochs"] + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("train/lr",   lr, epoch)

            row = {"epoch": epoch, "train_loss": f"{train_loss:.6f}", "lr": f"{lr:.2e}"}

            if epoch % self.cfg["val_every"] == 0 or epoch == self.cfg["max_epochs"]:
                val_metrics = self._val_epoch(epoch)
                mean_dice   = val_metrics["mean_dice"]
                mean_hd95   = val_metrics["mean_hd95"]

                self.writer.add_scalar("val/mean_dice", mean_dice, epoch)
                self.writer.add_scalar("val/mean_hd95", mean_hd95 if np.isfinite(mean_hd95) else 0, epoch)
                for c_name, vals in val_metrics["per_class"].items():
                    self.writer.add_scalar(f"val/dice_{c_name}", vals["dice"], epoch)

                self.logger.info(
                    f"[Epoch {epoch:04d}/{self.cfg['max_epochs']}] "
                    f"loss={train_loss:.4f}  "
                    f"mean_dice={mean_dice*100:.2f}%  "
                    f"mean_hd95={mean_hd95:.2f}mm  "
                    f"lr={lr:.2e}  "
                    f"time={time.time()-t0:.1f}s"
                )

                row.update({
                    "val_mean_dice": f"{mean_dice:.4f}",
                    "val_mean_hd95": f"{mean_hd95:.2f}",
                    **{f"dice_{n}": f"{v['dice']:.4f}"
                       for n, v in val_metrics["per_class"].items()},
                })

                if mean_dice > self.best_mean_dice:
                    self.best_mean_dice = mean_dice
                    self.best_epoch     = epoch
                    self._no_improve    = 0
                    self._save_checkpoint(epoch, mean_dice, is_best=True)
                    self.logger.info(f"  *** New best: {mean_dice*100:.2f}% at epoch {epoch} ***")
                else:
                    self._no_improve += 1
                    patience = self.cfg["early_stop_patience"]
                    if patience > 0:
                        self.logger.info(f"  [EarlyStopping] no improve {self._no_improve}/{patience}")
                        if self._no_improve >= patience:
                            self.logger.info(f"  [EarlyStopping] triggered at epoch {epoch}, stopping.")
                            self._log_csv(row)
                            break

                if epoch % (self.cfg["val_every"] * 5) == 0:
                    self._save_checkpoint(epoch, mean_dice, is_best=False)

            else:
                self.logger.info(
                    f"[Epoch {epoch:04d}/{self.cfg['max_epochs']}] "
                    f"loss={train_loss:.4f}  lr={lr:.2e}  time={time.time()-t0:.1f}s"
                )

            self._log_csv(row)

        self.writer.close()
        self.logger.info(f"\n[Done] Best mean Dice: {self.best_mean_dice*100:.2f}% @ epoch {self.best_epoch}")
        return self.best_mean_dice

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(self.train_loader, 1):
            images = batch["image"].to(self.device)   # (B, 1, D, H, W)
            labels = batch["label"].to(self.device)   # (B, D, H, W)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.cfg["amp"]):
                logits = self.model(images)            # (B, C, D, H, W)
                # DiceCELoss expects labels as (B, 1, D, H, W)
                loss   = self.criterion(logits, labels.unsqueeze(1))

            self.scaler.scale(loss).backward()
            if self.cfg["grad_clip"] > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    # ------------------------------------------------------------------
    def _val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        all_preds  = []
        all_labels = []

        roi_size = tuple(self.cfg["sw_roi_size"])

        with torch.no_grad():
            for batch in self.val_loader:
                # collate_fn=lambda batch: batch[0] → batch is a raw dict
                images = batch["image"].unsqueeze(0).to(self.device)  # (1,1,D,H,W)
                labels = batch["label"].cpu()                          # (D,H,W)

                # Sliding-window inference on full-resolution volume
                logits = sliding_window_inference(
                    inputs   = images,
                    roi_size = roi_size,
                    sw_batch_size = 1,
                    predictor = self.model,
                    overlap   = self.cfg["sw_overlap"],
                    mode      = "gaussian",
                )

                all_preds.append(logits.cpu())
                all_labels.append(labels.unsqueeze(0))  # (1,D,H,W)

        # volumes have different sizes → compute metrics per case then average
        all_dice = []
        all_hd95 = []
        from .metrics import dice_per_class, hd95_per_class
        import numpy as np
        for logits, labels in zip(all_preds, all_labels):
            pred_hard = logits.argmax(dim=1).squeeze(0).numpy()  # (D,H,W)
            tgt_np    = labels.squeeze(0).numpy()                 # (D,H,W)
            all_dice.append(dice_per_class(pred_hard, tgt_np, self.num_classes))
            if self.cfg["compute_hd95"]:
                all_hd95.append(hd95_per_class(pred_hard, tgt_np, self.num_classes))

        dice_mean = np.nanmean(all_dice, axis=0)   # NaN = both empty, excluded
        hd95_mean = np.nanmean(all_hd95, axis=0) if all_hd95 else np.full(self.num_classes, np.nan)

        fg = list(range(1, self.num_classes))
        mean_dice = float(np.nanmean(dice_mean[fg]))
        valid_hd  = hd95_mean[fg]
        valid_hd  = valid_hd[np.isfinite(valid_hd)]
        mean_hd95 = float(np.mean(valid_hd)) if len(valid_hd) > 0 else float("nan")

        per_class = {
            name: {"dice": float(dice_mean[c]), "hd95": float(hd95_mean[c])}
            for c, name in enumerate(self.class_names) if c > 0
        }
        metrics = {
            "dice": dice_mean, "mean_dice": mean_dice,
            "hd95": hd95_mean, "mean_hd95": mean_hd95,
            "per_class": per_class,
        }
        return metrics

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, dice: float, is_best: bool):
        name = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch:04d}.pt"
        path = self.output_dir / name
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_mean_dice": dice,
        }, path)

    def load_best_checkpoint(self):
        path = self.output_dir / "best_model.pt"
        if not path.exists():
            raise FileNotFoundError(f"No best_model.pt in {self.output_dir}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.logger.info(f"[Trainer] Loaded best model (epoch {ckpt['epoch']}, "
                         f"dice {ckpt['best_mean_dice']*100:.2f}%)")

    def _finetune(self, checkpoint_path: str):
        """Load model weights only - restart optimizer/scheduler from epoch 1."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Finetune checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.logger.info(f"[Finetune] Loaded weights from {path} (epoch {ckpt.get('epoch',0)}), "
                         f"restarting training from epoch 1")

    def _resume(self, checkpoint_path: str):
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError:
            # optimizer groups changed (e.g. freeze→unfreeze), skip optimizer state
            self.logger.info("[Resume] optimizer groups mismatch, resetting optimizer")
        epoch = ckpt.get("epoch", 0)
        self.start_epoch = epoch + 1
        self.best_mean_dice = ckpt.get("best_mean_dice", 0.0)
        self.best_epoch = epoch
        # Fast-forward scheduler to the right epoch
        for _ in range(epoch):
            self.scheduler.step()
        self.logger.info(f"[Resume] Loaded checkpoint epoch={epoch}, "
                         f"best_dice={self.best_mean_dice*100:.2f}%, "
                         f"resuming from epoch {self.start_epoch}")


# ---------------------------------------------------------------------------
# Standalone evaluation (no training)
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    num_classes: int,
    class_names: list,
    sw_roi_size: tuple = (128, 128, 128),
    sw_overlap: float = 0.5,
    device: str = "cuda",
    compute_hd95: bool = True,
) -> dict:
    """Run evaluation on val_loader and return metrics dict."""
    device_ = torch.device(device if torch.cuda.is_available() else "cpu")
    model    = model.to(device_).eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device_)
            logits = sliding_window_inference(
                inputs=images,
                roi_size=sw_roi_size,
                sw_batch_size=1,
                predictor=model,
                overlap=sw_overlap,
                mode="gaussian",
            )
            all_preds.append(logits.cpu())
            all_labels.append(batch["label"].cpu())

    all_preds  = torch.cat(all_preds,  dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_metrics(
        pred_logits = all_preds,
        target      = all_labels,
        num_classes = num_classes,
        class_names = class_names,
        compute_hd95= compute_hd95,
    )
    print(format_metrics(metrics, class_names))
    return metrics
