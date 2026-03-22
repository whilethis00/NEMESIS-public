#!/usr/bin/env python3
"""
NEMESIS Segmentation Benchmark - Training Script

Usage examples:
  # Full Synapse benchmark with NEMESIS
  python train_segmentation.py --config configs/synapse_nemesis.yaml

  # SwinUNETR baseline
  python train_segmentation.py --config configs/synapse_swinunetr.yaml

  # Label efficiency: 10% labels, NEMESIS
  python train_segmentation.py --config configs/synapse_label_eff.yaml \\
      --label_fraction 0.10 --model nemesis --output_dir results/label_eff_10pct

  # KiTS23 OOD generalization
  python train_segmentation.py --config configs/kits23.yaml --data_root /data/kits23

  # Evaluation only (no training)
  python train_segmentation.py --config configs/synapse_nemesis.yaml \\
      --eval_only --checkpoint results/synapse_nemesis_full/best_model.pt
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add benchmark root to path
BENCH_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH_ROOT))

from datasets.synapse     import build_synapse_dataset, SYNAPSE_ORGAN_NAMES
from datasets.kits23      import KiTS23Dataset
from datasets.msd_pancreas import MSDPancreasDataset
from models.seg_models     import build_model
from training.trainer      import SegTrainer, evaluate


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


def load_config(path: str, overrides: dict | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

SYNAPSE_ORGANS = SYNAPSE_ORGAN_NAMES   # 9 classes


def build_dataset(cfg: dict, split: str):
    d = cfg["data"]
    dataset_name  = d["dataset"].lower()
    data_root     = d["data_root"]
    roi_size      = tuple(d["roi_size"])
    num_classes   = d["num_classes"]
    lf            = d.get("label_fraction", 1.0)
    seed          = cfg["experiment"].get("seed", 42)

    if dataset_name == "synapse":
        return build_synapse_dataset(
            data_root=data_root,
            split=split,
            roi_size=roi_size,
            label_fraction=lf if split == "train" else 1.0,
            seed=seed,
            samples_per_volume=d.get("samples_per_volume", 1) if split == "train" else 1,
        )
    elif dataset_name == "kits23":
        return KiTS23Dataset(
            data_root=data_root,
            split=split,
            roi_size=roi_size,
            num_classes=num_classes,
            label_fraction=lf if split == "train" else 1.0,
            val_fraction=d.get("val_fraction", 0.1),
            seed=seed,
        )
    elif dataset_name == "msd_pancreas":
        return MSDPancreasDataset(
            data_root=data_root,
            split=split,
            roi_size=roi_size,
            num_classes=num_classes,
            label_fraction=lf if split == "train" else 1.0,
            val_fraction=d.get("val_fraction", 0.1),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_class_names(cfg: dict) -> list:
    dataset = cfg["data"]["dataset"].lower()
    nc      = cfg["data"]["num_classes"]
    if dataset == "synapse":
        return SYNAPSE_ORGANS
    elif dataset == "kits23":
        names = ["background", "kidney", "tumor"]
        return names[:nc]
    elif dataset == "msd_pancreas":
        names = ["background", "pancreas", "tumor"]
        return names[:nc]
    return [f"class_{i}" for i in range(nc)]


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_seg_model(cfg: dict) -> torch.nn.Module:
    m = cfg["model"]
    name = m["name"].lower()

    if name == "nemesis":
        return build_model(
            "nemesis",
            checkpoint_path  = m["checkpoint_path"],
            num_classes      = m["num_classes"],
            freeze_encoder   = m.get("freeze_encoder", False),
            superpatch_size  = tuple(m.get("superpatch_size", [128,128,128])),
            patch_size       = tuple(m.get("patch_size",      [8,8,8])),
            embed_dim        = m.get("embed_dim", 384),
            decoder_channels = list(m.get("decoder_channels", [192,96,48])),
        )
    elif name == "random_vit":
        return build_model(
            "random_vit",
            num_classes      = m["num_classes"],
            superpatch_size  = tuple(m.get("superpatch_size", [128,128,128])),
            patch_size       = tuple(m.get("patch_size",      [8,8,8])),
            embed_dim        = m.get("embed_dim", 384),
            depth            = m.get("depth", 6),
            num_heads        = m.get("num_heads", 6),
            num_maegic_tokens= m.get("num_maegic_tokens", 8),
            decoder_channels = list(m.get("decoder_channels", [192,96,48])),
        )
    elif name == "swinunetr":
        return build_model(
            "swinunetr",
            num_classes      = m["num_classes"],
            img_size         = tuple(m.get("img_size", [96,96,96])),
            feature_size     = m.get("feature_size", 48),
            pretrained_path  = m.get("pretrained_path", None),
        )
    elif name == "voco":
        return build_model(
            "voco",
            num_classes      = m["num_classes"],
            img_size         = tuple(m.get("img_size", [96,96,96])),
            feature_size     = m.get("feature_size", 48),
            pretrained_path  = m.get("pretrained_path", None),
        )
    elif name == "suprem":
        return build_model(
            "suprem",
            num_classes      = m["num_classes"],
            img_size         = tuple(m.get("img_size", [96,96,96])),
            feature_size     = m.get("feature_size", 48),
            pretrained_path  = m.get("pretrained_path", None),
        )
    else:
        raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="NEMESIS Segmentation Benchmark")

    parser.add_argument("--config",       type=str,   required=True,
                        help="Path to YAML config file")

    # Overrides (take precedence over config file)
    parser.add_argument("--model",        type=str,   default=None,
                        choices=["nemesis", "random_vit", "swinunetr"],
                        help="Model name (overrides config)")
    parser.add_argument("--data_root",    type=str,   default=None,
                        help="Path to dataset root (overrides config)")
    parser.add_argument("--label_fraction", type=float, default=None,
                        help="Fraction of labeled training data (overrides config)")
    parser.add_argument("--output_dir",   type=str,   default=None,
                        help="Output directory (overrides config)")
    parser.add_argument("--max_epochs",   type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--batch_size",   type=int,   default=None)
    parser.add_argument("--seed",         type=int,   default=None)

    parser.add_argument("--early_stop_patience", type=int, default=None,
                        help="Early stopping patience in # of val rounds (0=disabled). "
                             "e.g. --early_stop_patience 10 with val_every=20 → stop after 200 epochs no improve")
    parser.add_argument("--eval_only",    action="store_true",
                        help="Skip training, only run evaluation")
    parser.add_argument("--checkpoint",   type=str,   default=None,
                        help="Path to model checkpoint for eval-only mode")
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--finetune",     type=str,   default=None,
                        help="Load weights only (no optimizer/scheduler state), restart from epoch 1")
    parser.add_argument("--no_hd95",      action="store_true",
                        help="Skip HD95 computation (faster)")

    return parser.parse_args()


def apply_overrides(cfg: dict, args) -> dict:
    if args.model:
        cfg["model"]["name"] = args.model
    if args.data_root:
        cfg["data"]["data_root"] = args.data_root
    if args.label_fraction is not None:
        cfg["data"]["label_fraction"] = args.label_fraction
        # Update output dir to include label fraction
        if args.output_dir is None:
            pct = int(args.label_fraction * 100)
            cfg["output"]["dir"] = cfg["output"]["dir"] + f"_lf{pct:03d}"
    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir
    if args.max_epochs:
        cfg["training"]["max_epochs"] = args.max_epochs
    if args.lr:
        cfg["training"]["lr"] = args.lr
    if args.batch_size:
        cfg["data"]["batch_size"] = args.batch_size
    if args.seed:
        cfg["experiment"]["seed"] = args.seed
    if args.early_stop_patience is not None:
        cfg["training"]["early_stop_patience"] = args.early_stop_patience
    return cfg


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = apply_overrides(cfg, args)

    seed = cfg["experiment"].get("seed", 42)
    set_seed(seed)

    print("=" * 60)
    print(f"  Experiment : {cfg['experiment']['name']}")
    print(f"  Model      : {cfg['model']['name']}")
    print(f"  Dataset    : {cfg['data']['dataset']}  ({cfg['data']['data_root']})")
    print(f"  LabelFrac  : {cfg['data'].get('label_fraction', 1.0)*100:.0f}%")
    print(f"  Output     : {cfg['output']['dir']}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Build datasets
    # ------------------------------------------------------------------
    train_ds = build_dataset(cfg, split="train")
    val_ds   = build_dataset(cfg, split="val")

    d = cfg["data"]
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size  = d.get("batch_size", 2),
        shuffle     = True,
        num_workers = d.get("num_workers", 4),
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size  = 1,    # sliding window inference needs batch=1
        shuffle     = False,
        num_workers = d.get("num_workers", 4),
        pin_memory  = True,
        collate_fn  = lambda batch: batch[0],  # full-res volumes vary in size
    )

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model       = build_seg_model(cfg)
    class_names = get_class_names(cfg)
    num_classes = cfg["data"]["num_classes"]

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model params: total={n_params:.1f}M  trainable={n_train:.1f}M")

    # ------------------------------------------------------------------
    # Eval-only mode
    # ------------------------------------------------------------------
    if args.eval_only:
        if args.checkpoint is None:
            # Try default best_model.pt
            ckpt_path = Path(cfg["output"]["dir"]) / "best_model.pt"
        else:
            ckpt_path = Path(args.checkpoint)

        if not ckpt_path.exists():
            print(f"ERROR: checkpoint not found at {ckpt_path}")
            sys.exit(1)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {ckpt_path}")

        # Use full test set
        test_ds = build_dataset(cfg, split="test")
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=1, shuffle=False,
            num_workers=d.get("num_workers", 4), pin_memory=True
        )

        metrics = evaluate(
            model=model,
            val_loader=test_loader,
            num_classes=num_classes,
            class_names=class_names,
            sw_roi_size=tuple(cfg["training"]["sw_roi_size"]),
            sw_overlap=cfg["training"]["sw_overlap"],
            compute_hd95=not args.no_hd95,
        )

        # Save results
        import json
        out_dir = Path(cfg["output"]["dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump({
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in metrics.items()
                if k != "per_class"
            } | {"per_class": metrics["per_class"]}, f, indent=2)
        print(f"\nResults saved to {results_path}")
        return

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    trainer = SegTrainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        num_classes  = num_classes,
        output_dir   = cfg["output"]["dir"],
        cfg          = cfg["training"],
        class_names  = class_names,
        resume_checkpoint    = args.resume,
        finetune_checkpoint  = args.finetune,
    )

    # Save config to output dir
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    trainer.train()


if __name__ == "__main__":
    main()
