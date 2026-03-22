#!/usr/bin/env python3
"""
NEMESIS Classification Benchmark - Training Script

Multi-label organ presence classification: for each 128³ CT superpatch,
predict which of 8 organs are present.

Usage examples:
  # Full BTCV benchmark with NEMESIS (frozen encoder)
  python train_classification.py --config configs/btcv_cls_nemesis.yaml

  # Fine-tune encoder (not frozen)
  python train_classification.py --config configs/btcv_cls_nemesis.yaml \\
      --model nemesis --freeze_encoder false

  # Random-init ViT ablation
  python train_classification.py --config configs/btcv_cls_nemesis.yaml \\
      --model random_vit --output_dir results/btcv_cls_random_vit

  # Semi-supervised: 50% labels
  python train_classification.py --config configs/btcv_cls_nemesis.yaml \\
      --label_fraction 0.5 --output_dir results/btcv_cls_nemesis_lf050

  # Evaluation only (load best checkpoint)
  python train_classification.py --config configs/btcv_cls_nemesis.yaml \\
      --eval_only --checkpoint results/btcv_cls_nemesis/best_model.pt

  # Resume interrupted training
  python train_classification.py --config configs/btcv_cls_nemesis.yaml \\
      --resume results/btcv_cls_nemesis/checkpoint_epoch_0050.pt

  # Fine-tune from existing model (restart epoch counter)
  python train_classification.py --config configs/btcv_cls_nemesis.yaml \\
      --finetune results/btcv_cls_nemesis/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add benchmark root to Python path
BENCH_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH_ROOT))

from datasets.btcv_cls  import (
    BTCVClassificationDataset,
    build_cls_dataset,
    CLS_ORGAN_NAMES,
    NUM_CLS_ORGANS,
)
from models.cls_models   import build_model
from training.cls_trainer import ClsTrainer, evaluate_cls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(cfg: dict, split: str) -> BTCVClassificationDataset:
    d = cfg["data"]
    return build_cls_dataset(
        data_root       = d["data_root"],
        split           = split,
        superpatch_size = tuple(d["superpatch_size"]),
        num_classes     = d.get("num_classes", NUM_CLS_ORGANS),
        label_fraction  = d.get("label_fraction", 1.0) if split == "train" else 1.0,
        seed            = cfg["experiment"].get("seed", 42),
        threshold       = d.get("presence_threshold", 100),
    )


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_cls_model(cfg: dict) -> torch.nn.Module:
    m    = cfg["model"]
    name = m["name"].lower()

    if name == "nemesis":
        return build_model(
            "nemesis",
            checkpoint_path   = m["checkpoint_path"],
            num_classes       = m.get("num_classes", NUM_CLS_ORGANS),
            freeze_encoder    = m.get("freeze_encoder", True),
            embed_dim         = m.get("embed_dim", 768),
            depth             = m.get("depth", 6),
            num_heads         = m.get("num_heads", 8),
            num_maegic_tokens = m.get("num_maegic_tokens", 8),
            use_maegic_tokens = m.get("use_maegic_tokens", True),
        )
    elif name == "random_vit":
        return build_model(
            "random_vit",
            num_classes      = m.get("num_classes", NUM_CLS_ORGANS),
            embed_dim        = m.get("embed_dim", 768),
            depth            = m.get("depth", 6),
            num_heads        = m.get("num_heads", 8),
            num_maegic_tokens= m.get("num_maegic_tokens", 8),
        )
    elif name == "resnet3d":
        return build_model(
            "resnet3d",
            num_classes  = m.get("num_classes", NUM_CLS_ORGANS),
            model_depth  = m.get("model_depth", 50),
        )
    elif name == "voco":
        return build_model(
            "voco",
            num_classes      = m.get("num_classes", NUM_CLS_ORGANS),
            img_size         = tuple(m.get("img_size", [128, 128, 128])),
            feature_size     = m.get("feature_size", 48),
            pretrained_path  = m.get("pretrained_path", None),
        )
    elif name == "suprem":
        return build_model(
            "suprem",
            num_classes      = m.get("num_classes", NUM_CLS_ORGANS),
            img_size         = tuple(m.get("img_size", [128, 128, 128])),
            feature_size     = m.get("feature_size", 48),
            pretrained_path  = m.get("pretrained_path", None),
        )
    else:
        raise ValueError(f"Unknown classification model: '{name}'. "
                         f"Choose from {list(['nemesis','random_vit','resnet3d','voco','suprem'])}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="NEMESIS Multi-label Organ Classification Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")

    # Model overrides
    parser.add_argument("--model", type=str, default=None,
                        choices=["nemesis", "random_vit", "resnet3d", "voco", "suprem"],
                        help="Model name (overrides config)")
    parser.add_argument("--freeze_encoder", type=str, default=None,
                        choices=["true", "false"],
                        help="Freeze encoder: 'true' for linear probe, 'false' for fine-tune")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to MAE checkpoint (overrides config)")
    parser.add_argument("--embed_dim", type=int, default=None,
                        help="Encoder embed_dim (overrides config, e.g. 384/576/768)")
    parser.add_argument("--depth", type=int, default=None,
                        help="Encoder depth (overrides config, e.g. 6/12)")

    # Data overrides
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to BTCV dataset root (overrides config)")
    parser.add_argument("--label_fraction", type=float, default=None,
                        help="Fraction of labeled training data, e.g. 0.5 for 50%%")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Training batch size (overrides config)")

    # Training overrides
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for checkpoints and logs (overrides config)")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Maximum training epochs (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides config)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config)")
    parser.add_argument("--early_stop_patience", type=int, default=None,
                        help="Early stopping patience in # of val rounds (0=disabled)")

    # Workflow flags
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, run evaluation only")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for eval-only mode")
    parser.add_argument("--resume", type=str, default=None,
                        help="Full trainer checkpoint to resume training from")
    parser.add_argument("--finetune", type=str, default=None,
                        help="Load model weights only; restart training from epoch 1")

    return parser.parse_args()


def apply_overrides(cfg: dict, args) -> dict:
    """Apply command-line overrides to config dict in-place."""
    if args.model:
        cfg["model"]["name"] = args.model
    if args.freeze_encoder is not None:
        cfg["model"]["freeze_encoder"] = (args.freeze_encoder.lower() == "true")
    if args.checkpoint_path:
        cfg["model"]["checkpoint_path"] = args.checkpoint_path
    if args.embed_dim is not None:
        cfg["model"]["embed_dim"] = args.embed_dim
    if args.depth is not None:
        cfg["model"]["depth"] = args.depth
    if args.data_root:
        cfg["data"]["data_root"] = args.data_root
    if args.label_fraction is not None:
        cfg["data"]["label_fraction"] = args.label_fraction
        # Auto-suffix output dir to track experiment
        if args.output_dir is None:
            pct = int(args.label_fraction * 100)
            cfg["output"]["dir"] = cfg["output"]["dir"] + f"_lf{pct:03d}"
    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir
    if args.batch_size:
        cfg["data"]["batch_size"] = args.batch_size
    if args.max_epochs:
        cfg["training"]["max_epochs"] = args.max_epochs
    if args.lr:
        cfg["training"]["lr"] = args.lr
    if args.seed:
        cfg["experiment"]["seed"] = args.seed
    if args.early_stop_patience is not None:
        cfg["training"]["early_stop_patience"] = args.early_stop_patience
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = apply_overrides(cfg, args)

    seed = cfg["experiment"].get("seed", 42)
    set_seed(seed)

    print("=" * 65)
    print(f"  Experiment : {cfg['experiment']['name']}")
    print(f"  Model      : {cfg['model']['name']}")
    print(f"  Dataset    : {cfg['data']['dataset']}  ({cfg['data']['data_root']})")
    print(f"  LabelFrac  : {cfg['data'].get('label_fraction', 1.0)*100:.0f}%")
    print(f"  FreezeEnc  : {cfg['model'].get('freeze_encoder', True)}")
    print(f"  Output     : {cfg['output']['dir']}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Build datasets
    # ------------------------------------------------------------------
    train_ds = build_dataset(cfg, split="train")
    val_ds   = build_dataset(cfg, split="val")

    d = cfg["data"]

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size  = d.get("batch_size", 8),
        shuffle     = True,
        num_workers = d.get("num_workers", 4),
        pin_memory  = True,
        drop_last   = True,
    )

    # Val loader: batch_size=1 because each item is one full volume's patches.
    # The model processes each superpatch individually inside the trainer.
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size  = 1,
        shuffle     = False,
        num_workers = d.get("num_workers", 4),
        pin_memory  = False,
    )

    print(f"Train superpatches: {len(train_ds)}, Val volumes: {len(val_ds)}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model       = build_cls_model(cfg)
    num_classes = d.get("num_classes", NUM_CLS_ORGANS)
    class_names = CLS_ORGAN_NAMES[:num_classes]

    n_total   = sum(p.numel() for p in model.parameters()) / 1e6
    n_train   = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model params: total={n_total:.1f}M  trainable={n_train:.1f}M")

    # ------------------------------------------------------------------
    # Evaluation-only mode
    # ------------------------------------------------------------------
    if args.eval_only:
        ckpt_path = Path(args.checkpoint) if args.checkpoint else \
                    Path(cfg["output"]["dir"]) / "best_model.pt"

        if not ckpt_path.exists():
            print(f"ERROR: checkpoint not found at {ckpt_path}")
            sys.exit(1)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

        metrics = evaluate_cls(
            model       = model,
            val_loader  = val_loader,
            num_classes = num_classes,
            class_names = class_names,
            threshold   = cfg["training"].get("cls_threshold", 0.5),
        )

        out_dir = Path(cfg["output"]["dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {results_path}")
        return

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    trainer = ClsTrainer(
        model               = model,
        train_loader        = train_loader,
        val_loader          = val_loader,
        num_classes         = num_classes,
        output_dir          = str(out_dir),
        cfg                 = cfg["training"],
        class_names         = class_names,
        resume_checkpoint   = args.resume,
        finetune_checkpoint = args.finetune,
    )

    best_auroc = trainer.train()
    print(f"\nTraining complete. Best mean AUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    main()
