"""
Classification models for NEMESIS benchmark.

Multi-label organ presence classification: given a 128³ CT superpatch,
predict which of 8 organs are present (binary vector, BCEWithLogitsLoss).

Models:
  - NEMESISClassifier  : MAEgicEncoder (pre-trained, embed_dim=768) + linear head
  - RandomViTClassifier: Same MAEgic architecture, random init (ablation)

Encoder architecture (from MAE_768_0.5.pt inspection):
  embed_dim=768, depth=6, num_heads=8, patch_size=(8,8,8),
  num_maegic_tokens=8

Forward path:
  (B, 1, 128, 128, 128)
    → MAEgicEncoder → tokens (B, N, 768)
    → mean pool      → (B, 768)
    → linear head   → (B, 8) logits
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Architecture constants for MAE_768_0.5.pt
# ---------------------------------------------------------------------------

NEMESIS_768_CFG = dict(
    superpatch_size=(128, 128, 128),
    patch_size=(8, 8, 8),
    in_channels=1,
    embed_dim=768,
    depth=6,
    num_heads=8,
    dropout=0.1,
    drop_path_rate=0.1,
    pos_encoding_type="sinusoidal",
    num_maegic_tokens=8,
    spatial_mask_ratio=0.0,
    depth_mask_ratio=0.0,
)


def _build_mae_encoder(embed_dim: int = 768, depth: int = 6, num_heads: int = 8,
                        num_maegic_tokens: int = 8,
                        use_maegic_tokens: bool = True) -> nn.Module:
    """Instantiate MAEgicEncoder with given hyperparameters."""
    from .nemesis_arch import MAEgicEncoder

    return MAEgicEncoder(
        volume_size=(512, 512, 512),
        superpatch_size=NEMESIS_768_CFG["superpatch_size"],
        patch_size=NEMESIS_768_CFG["patch_size"],
        in_channels=NEMESIS_768_CFG["in_channels"],
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        dropout=NEMESIS_768_CFG["dropout"],
        drop_path_rate=NEMESIS_768_CFG["drop_path_rate"],
        pos_encoding_type=NEMESIS_768_CFG["pos_encoding_type"],
        num_maegic_tokens=num_maegic_tokens,
        spatial_mask_ratio=0.0,
        depth_mask_ratio=0.0,
        use_maegic_tokens=use_maegic_tokens,
    )


# ---------------------------------------------------------------------------
# NEMESIS Classification model
# ---------------------------------------------------------------------------

class NEMESISClassifier(nn.Module):
    """
    NEMESIS pre-trained encoder + linear classification head.

    The encoder is loaded from a MAE checkpoint (MAE_768_0.5.pt) that was
    pre-trained with embed_dim=768, depth=6, num_heads=8, patch_size=8³.

    Forward:
        (B, 1, 128, 128, 128)
        → MAEgicEncoder (eval, no masking) → (B, N, 768)
        → mean pool over N tokens          → (B, 768)
        → LayerNorm + Linear               → (B, num_classes) logits

    Args:
        checkpoint_path : path to MAE_768_0.5.pt  (required unless random init)
        num_classes     : number of output classes  (default 8 for Synapse organs)
        freeze_encoder  : freeze all encoder parameters (linear-probe mode)
        embed_dim       : encoder embedding dimension (768 for MAE_768_0.5.pt)
        depth           : number of transformer blocks (6)
        num_heads       : attention heads per block (8)
        num_maegic_tokens: number of MAEgic tokens for AdaptivePatchEmbedding (8)
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 8,
        freeze_encoder: bool = True,
        embed_dim: int = 768,
        depth: int = 6,
        num_heads: int = 8,
        num_maegic_tokens: int = 8,
        use_maegic_tokens: bool = True,
    ):
        super().__init__()

        # Build encoder
        self.encoder = _build_mae_encoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_maegic_tokens=num_maegic_tokens,
            use_maegic_tokens=use_maegic_tokens,
        )

        # Load pre-trained weights
        self._load_checkpoint(checkpoint_path, embed_dim=embed_dim, depth=depth)

        # Classification head: LayerNorm + Linear
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        # Optionally freeze encoder
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _load_checkpoint(self, checkpoint_path: str, embed_dim: int = 768, depth: int = 6):
        """Load MAE checkpoint, extracting only the encoder sub-module weights."""
        import os
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"MAE checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)

        # Extract encoder weights (keys starting with "encoder.")
        enc_state = {}
        for k, v in state.items():
            if k.startswith("encoder."):
                enc_state[k[len("encoder."):]] = v  # strip "encoder." prefix

        missing, unexpected = self.encoder.load_state_dict(enc_state, strict=True)
        if missing:
            print(f"[NEMESISClassifier] Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[NEMESISClassifier] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

        epoch    = ckpt.get("epoch", "?")
        val_loss = ckpt.get("val_loss", "?")
        exp_name = ckpt.get("exp_name", "?")
        print(f"[NEMESISClassifier] Loaded encoder from '{checkpoint_path}' "
              f"(epoch={epoch}, val_loss={val_loss}, exp={exp_name})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 1, 128, 128, 128) CT superpatch, normalized to [0, 1]

        Returns:
            logits : (B, num_classes)  raw (pre-sigmoid) logits
        """
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                tokens, _mask, _ids = self.encoder(x)   # (B, N, embed_dim)
        else:
            tokens, _mask, _ids = self.encoder(x)       # (B, N, embed_dim)

        # Mean pool over sequence tokens → (B, embed_dim)
        pooled = tokens.mean(dim=1)

        # Classification head
        out = self.norm(pooled)
        logits = self.head(out)   # (B, num_classes)
        return logits


# ---------------------------------------------------------------------------
# Random-init ViT baseline (same arch, no pre-training)
# ---------------------------------------------------------------------------

class RandomViTClassifier(nn.Module):
    """
    NEMESIS architecture (MAEgicEncoder) with random weight initialisation.
    Used as the 'no pre-training' ablation baseline.

    Args:
        num_classes     : number of output classes  (default 8)
        embed_dim       : encoder embedding dimension (768)
        depth           : number of transformer blocks (6)
        num_heads       : attention heads per block (8)
        num_maegic_tokens: MAEgic tokens for AdaptivePatchEmbedding (8)
    """

    def __init__(
        self,
        num_classes: int = 8,
        embed_dim: int = 768,
        depth: int = 6,
        num_heads: int = 8,
        num_maegic_tokens: int = 8,
    ):
        super().__init__()

        self.encoder = _build_mae_encoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_maegic_tokens=num_maegic_tokens,
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        print(f"[RandomViTClassifier] Randomly initialised encoder "
              f"(embed_dim={embed_dim}, depth={depth}, num_heads={num_heads})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 1, 128, 128, 128) CT superpatch, normalized to [0, 1]

        Returns:
            logits : (B, num_classes)
        """
        tokens, _mask, _ids = self.encoder(x)   # (B, N, embed_dim)
        pooled = tokens.mean(dim=1)              # (B, embed_dim)
        out = self.norm(pooled)
        logits = self.head(out)                  # (B, num_classes)
        return logits


# ---------------------------------------------------------------------------
# 3D ResNet baseline (MONAI)
# ---------------------------------------------------------------------------

class ResNet3DClassifier(nn.Module):
    """
    3D ResNet50 (MONAI) + linear classification head.
    Standard baseline for 3D medical image classification.
    """

    def __init__(
        self,
        num_classes: int = 8,
        model_depth: int = 50,
    ):
        super().__init__()
        from monai.networks.nets import resnet50, resnet18
        if model_depth == 50:
            backbone = resnet50(spatial_dims=3, n_input_channels=1, num_classes=num_classes)
        else:
            backbone = resnet18(spatial_dims=3, n_input_channels=1, num_classes=num_classes)
        # Replace final fc with our multi-label head
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.model = backbone
        print(f"[ResNet3D-{model_depth}] Random init, in_features={in_features}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# VoCo classification (SwinUNETR encoder + linear head)
# ---------------------------------------------------------------------------

class VoCoClassifier(nn.Module):
    """
    VoCo pre-trained SwinUNETR encoder + linear classification head.
    Extracts encoder features via global average pooling over the encoder output.
    """

    def __init__(
        self,
        num_classes: int = 8,
        img_size: tuple = (128, 128, 128),
        feature_size: int = 48,
        pretrained_path: str | None = None,
    ):
        super().__init__()
        from monai.networks.nets import SwinUNETR
        self.swin = SwinUNETR(
            img_size=img_size,
            in_channels=1,
            out_channels=num_classes,  # dummy, we extract encoder features
            feature_size=feature_size,
            use_checkpoint=True,
        )
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        # Classification head on top of SwinViT hidden features (feature_size*16)
        hidden_dim = feature_size * 16  # 768 for feature_size=48
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def _load_pretrained(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        state = {k.replace("swin_vit", "swinViT"): v for k, v in state.items()}
        missing, unexpected = self.swin.load_state_dict(state, strict=False)
        print(f"[VoCoClassifier] missing={len(missing)}, unexpected={len(unexpected)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use SwinUNETR's swinViT to get encoder hidden states
        hidden_states = self.swin.swinViT(x, self.swin.normalize)
        # hidden_states[-1]: (B, C, D/32, H/32, W/32)
        feat = hidden_states[-1]
        feat = feat.mean(dim=[2, 3, 4])  # global avg pool → (B, C)
        return self.fc(feat)


# ---------------------------------------------------------------------------
# SuPreM classification (SwinUNETR encoder + linear head)
# ---------------------------------------------------------------------------

class SuPremClassifier(nn.Module):
    """
    SuPreM supervised pre-trained SwinUNETR encoder + linear classification head.
    """

    def __init__(
        self,
        num_classes: int = 8,
        img_size: tuple = (128, 128, 128),
        feature_size: int = 48,
        pretrained_path: str | None = None,
    ):
        super().__init__()
        from monai.networks.nets import SwinUNETR
        self.swin = SwinUNETR(
            img_size=img_size,
            in_channels=1,
            out_channels=num_classes,
            feature_size=feature_size,
            use_checkpoint=True,
        )
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        hidden_dim = feature_size * 16  # 768
        self.fc = nn.Linear(hidden_dim, num_classes)

    def _load_pretrained(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("net", ckpt.get("state_dict", ckpt.get("model", ckpt)))
        # SuPreM: "module.backbone.swinViT.*" → "swinViT.*"
        new_state = {}
        for k, v in state.items():
            if k.startswith("module.backbone."):
                new_state[k[len("module.backbone."):]] = v
        missing, unexpected = self.swin.load_state_dict(new_state, strict=False)
        print(f"[SuPremClassifier] loaded swinViT: missing={len(missing)}, unexpected={len(unexpected)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.swin.swinViT(x, self.swin.normalize)
        feat = hidden_states[-1].mean(dim=[2, 3, 4])
        return self.fc(feat)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "nemesis":    NEMESISClassifier,
    "random_vit": RandomViTClassifier,
    "resnet3d":   ResNet3DClassifier,
    "voco":       VoCoClassifier,
    "suprem":     SuPremClassifier,
}


def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
