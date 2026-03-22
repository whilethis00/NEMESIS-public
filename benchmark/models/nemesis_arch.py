"""
NEMESIS Architecture loader.

Imports MAEgic3DMAE from the nemesis package (included in this repository).
Set NEMESIS_ROOT env variable if running from outside the repo root.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Resolve nemesis package: repo_root/nemesis/
_REPO_ROOT = Path(__file__).resolve().parents[2]   # benchmark/models/ -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nemesis.models.mae import (
    MAEgic3DMAE,
    MAEgicEncoder,
    MAEgicDecoder,
    TransformerBlock,
    AdaptivePatchEmbedding,
    SinusoidalPE,
)

__all__ = [
    "MAEgic3DMAE",
    "MAEgicEncoder",
    "MAEgicDecoder",
    "TransformerBlock",
    "AdaptivePatchEmbedding",
    "SinusoidalPE",
    "build_nemesis_encoder",
]

# ---------------------------------------------------------------------------
# Default hyperparameters (confirmed from MAE_768_0.5.pt checkpoint)
# ---------------------------------------------------------------------------
NEMESIS_DEFAULT_CFG = dict(
    superpatch_size=(128, 128, 128),
    patch_size=(8, 8, 8),
    in_channels=1,
    embed_dim=768,
    depth=6,
    num_heads=8,
    decoder_depth=3,
    decoder_num_heads=8,
    dropout=0.1,
    drop_path_rate=0.1,
    pos_encoding_type="sinusoidal",
    num_maegic_tokens=8,
    spatial_mask_ratio=0.0,   # masking disabled at inference
    depth_mask_ratio=0.0,
)


def build_nemesis_encoder(checkpoint_path: str | None = None, strict: bool = True):
    """
    Build NEMESIS encoder and optionally load pretrained weights.

    Args:
        checkpoint_path : path to MAE_768_0.5.pt (or None for random init)
        strict          : whether to enforce strict weight loading

    Returns:
        MAEgic3DMAE (full model). For downstream use, access model.encoder
        after calling model.encoder.eval() to disable masking.

    Example::

        model = build_nemesis_encoder("pretrained/MAE_768_0.5.pt")
        encoder = model.encoder.eval()
        with torch.no_grad():
            tokens, _, _ = encoder(superpatch)   # (B, N, 768)
    """
    import torch

    model = MAEgic3DMAE(**NEMESIS_DEFAULT_CFG)

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=strict)
        if missing:
            print(f"[NEMESIS] Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[NEMESIS] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        epoch    = ckpt.get("epoch", "?")
        val_loss = ckpt.get("val_loss", "?")
        print(f"[NEMESIS] Loaded checkpoint: epoch={epoch}, val_loss={val_loss}")

    return model
