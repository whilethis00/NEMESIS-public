"""
Segmentation models for NEMESIS benchmark.

Models:
  - NEMESISSeg     : NEMESIS pre-trained encoder + UNet decoder
  - RandomViTSeg   : Same NEMESIS architecture, random init (ablation baseline)
  - SwinUNETRSeg   : MONAI SwinUNETR (Tang et al. CVPR'22)

All models accept (B, 1, D, H, W) CT volumes and output (B, num_classes, D, H, W) logits.
Sliding-window inference is handled externally via monai.inferers.sliding_window_inference.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# UNet-style segmentation decoder (shared by NEMESIS and RandomViT)
# ---------------------------------------------------------------------------

class ConvBnReLU(nn.Sequential):
    def __init__(self, in_c, out_c, kernel=3, pad=1):
        super().__init__(
            nn.Conv3d(in_c, out_c, kernel, padding=pad, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    """2× trilinear upsample + two ConvBnReLU."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv = nn.Sequential(
            ConvBnReLU(in_c, out_c),
            ConvBnReLU(out_c, out_c),
        )

    def forward(self, x):
        return self.conv(self.up(x))


class ViTSegDecoder(nn.Module):
    """
    Reshape (B, N, E) → (B, E, D_p, H_p, W_p) then 3× upsample to full superpatch size.

    Default path for 128³ input, patch_size=8:
      (B, 4096, 384) → (B, 384, 16, 16, 16)
        → UpBlock(384→192) → (B, 192, 32, 32, 32)
        → UpBlock(192→96)  → (B,  96, 64, 64, 64)
        → UpBlock( 96→48)  → (B,  48,128,128,128)
        → Conv1x1(48→num_classes)
    """
    def __init__(
        self,
        embed_dim: int = 384,
        patch_grid: tuple = (16, 16, 16),   # superpatch_size / patch_size
        channels: list  = (192, 96, 48),
        num_classes: int = 9,
    ):
        super().__init__()
        self.patch_grid = patch_grid

        blocks = []
        in_c = embed_dim
        for out_c in channels:
            blocks.append(UpBlock(in_c, out_c))
            in_c = out_c
        self.decoder = nn.Sequential(*blocks)
        self.head    = nn.Conv3d(in_c, num_classes, kernel_size=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, N, E)"""
        B, N, E = tokens.shape
        D, H, W = self.patch_grid
        assert N == D * H * W, f"Expected {D*H*W} tokens, got {N}"
        x = rearrange(tokens, "b (d h w) e -> b e d h w", d=D, h=H, w=W)
        x = self.decoder(x)
        return self.head(x)


class UNETRDecoder(nn.Module):
    """
    UNETR-style decoder: uses intermediate ViT block outputs as skip connections.

    Skip connections at decoder stages 0 and 1 (from encoder blocks at indices
    given by skip_block_indices). Each skip is projected to match the decoder
    channel count and added (not concatenated) after upsampling.

    Default path for 128³ input, patch_size=8, encoder depth=6:
      final tokens (B,4096,384) → reshape (B,384,16,16,16)
        → UpBlock(384→192) + skip[block3] → (B,192,32,32,32)
        → UpBlock(192→96)  + skip[block1] → (B, 96,64,64,64)
        → UpBlock( 96→48)              → (B, 48,128,128,128)
        → Conv1x1(48→num_classes)
    """
    def __init__(
        self,
        embed_dim: int = 384,
        patch_grid: tuple = (16, 16, 16),
        channels: list  = (192, 96, 48),
        num_classes: int = 9,
        num_skips: int = 2,   # how many skip connections to use
    ):
        super().__init__()
        self.patch_grid = patch_grid
        self.num_skips  = num_skips

        # Up-blocks
        blocks = []
        in_c = embed_dim
        for out_c in channels:
            blocks.append(UpBlock(in_c, out_c))
            in_c = out_c
        self.up_blocks = nn.ModuleList(blocks)

        # Skip projectors (one per skip connection, project embed_dim → matching channel)
        # Zero-init: skip contributes 0 at start, grows gradually during training
        self.skip_projs = nn.ModuleList([
            nn.Conv3d(embed_dim, channels[i], kernel_size=1)
            for i in range(num_skips)
        ])
        # kaiming normal init (default) - zero-init was too slow to converge

        self.head = nn.Conv3d(in_c, num_classes, kernel_size=1)

    def forward(self, tokens: torch.Tensor, skips: list) -> torch.Tensor:
        """
        tokens : (B, N, E)   final encoder output
        skips  : list of (B, N, E), deeper skip first
                 e.g. [block3_out, block1_out] for depth=6
        """
        B, N, E = tokens.shape
        D, H, W = self.patch_grid
        assert N == D * H * W

        x = rearrange(tokens, "b (d h w) e -> b e d h w", d=D, h=H, w=W)

        for i, up in enumerate(self.up_blocks):
            x = up(x)
            if i < self.num_skips and i < len(skips):
                skip = rearrange(skips[i], "b (d h w) e -> b e d h w", d=D, h=H, w=W)
                skip = self.skip_projs[i](skip)
                if skip.shape[2:] != x.shape[2:]:
                    skip = F.interpolate(skip, size=x.shape[2:],
                                         mode="trilinear", align_corners=False)
                x = x + skip

        return self.head(x)


# ---------------------------------------------------------------------------
# NEMESIS segmentation model
# ---------------------------------------------------------------------------

class NEMESISSeg(nn.Module):
    """
    NEMESIS encoder (pre-trained) + ViTSegDecoder.

    Args:
        checkpoint_path : path to best_model_epoch_149_val.pt
        num_classes     : including background  (default 9 for Synapse)
        freeze_encoder  : freeze all encoder parameters (linear-probe mode)
        superpatch_size : must match pre-training (128,128,128)
        patch_size      : must match pre-training (8,8,8)
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 9,
        freeze_encoder: bool = False,
        superpatch_size: tuple = (128, 128, 128),
        patch_size: tuple = (8, 8, 8),
        embed_dim: int = 384,
        decoder_channels: list = (192, 96, 48),
    ):
        super().__init__()
        from .nemesis_arch import build_nemesis_encoder

        full_model = build_nemesis_encoder(checkpoint_path, strict=True)
        self.encoder = full_model.encoder
        del full_model  # discard decoder to save memory

        patch_grid = tuple(s // p for s, p in zip(superpatch_size, patch_size))
        self.decoder = UNETRDecoder(
            embed_dim=embed_dim,
            patch_grid=patch_grid,
            channels=list(decoder_channels),
            num_classes=num_classes,
            num_skips=2,
        )

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Register hooks on encoder blocks 1 and 3 (0-indexed) for skip connections
        # Skip order: deeper block first (block 3 → stage 0, block 1 → stage 1)
        self._skip_features: list = []
        encoder_depth = len(self.encoder.blocks)
        skip_indices = [encoder_depth // 2, encoder_depth // 6]  # e.g. [3, 1] for depth=6
        skip_indices = [max(0, min(i, encoder_depth - 1)) for i in skip_indices]
        self._skip_indices = skip_indices
        self._hooks = []
        for idx in skip_indices:
            h = self.encoder.blocks[idx].register_forward_hook(
                lambda m, inp, out: self._skip_features.append(out)
            )
            self._hooks.append(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, D, H, W)  →  logits: (B, num_classes, D, H, W)"""
        self._skip_features = []
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                tokens, _mask, _ids = self.encoder(x)   # (B, N, E)
        else:
            tokens, _mask, _ids = self.encoder(x)   # (B, N, E)
        logits = self.decoder(tokens, self._skip_features)
        # resize to input spatial size (handles slight mismatch after upsampling)
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode="trilinear", align_corners=False)
        return logits


# ---------------------------------------------------------------------------
# Random-init ViT baseline (same arch, no pre-training)
# ---------------------------------------------------------------------------

class RandomViTSeg(nn.Module):
    """
    NEMESIS architecture with random weight initialisation.
    Used as the 'no pre-training' ablation baseline.
    """

    def __init__(
        self,
        num_classes: int = 9,
        superpatch_size: tuple = (128, 128, 128),
        patch_size: tuple = (8, 8, 8),
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        num_maegic_tokens: int = 8,
        decoder_channels: list = (192, 96, 48),
    ):
        super().__init__()
        from .nemesis_arch import MAEgicEncoder

        self.encoder = MAEgicEncoder(
            superpatch_size=superpatch_size,
            patch_size=patch_size,
            in_channels=1,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_maegic_tokens=num_maegic_tokens,
            spatial_mask_ratio=0.0,
            depth_mask_ratio=0.0,
        )
        patch_grid = tuple(s // p for s, p in zip(superpatch_size, patch_size))
        self.decoder = ViTSegDecoder(
            embed_dim=embed_dim,
            patch_grid=patch_grid,
            channels=list(decoder_channels),
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, _mask, _ids = self.encoder(x)
        logits = self.decoder(tokens)
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode="trilinear", align_corners=False)
        return logits


# ---------------------------------------------------------------------------
# SwinUNETR baseline (MONAI)
# ---------------------------------------------------------------------------

class SwinUNETRSeg(nn.Module):
    """
    Thin wrapper around MONAI SwinUNETR.

    Pre-trained weights from MONAI model zoo can be downloaded with:
        monai.bundle.load("swin_unetr_btcv_segmentation", ...)
    Or pass pretrained_path to a local .pt file.
    """

    def __init__(
        self,
        num_classes: int = 9,
        img_size: tuple = (96, 96, 96),
        feature_size: int = 48,
        pretrained_path: str | None = None,
    ):
        super().__init__()
        from monai.networks.nets import SwinUNETR

        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=1,
            out_channels=num_classes,
            feature_size=feature_size,
            use_checkpoint=True,
        )

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path: str):
        import torch
        ckpt = torch.load(path, map_location="cpu")
        # MONAI SwinUNETR pretrained key format
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        # Strip "swinViT." prefix if present (self-supervised pretrain style)
        new_state = {}
        for k, v in state.items():
            if k.startswith("swinViT."):
                new_state["swinViT." + k[len("swinViT."):]] = v
            else:
                new_state[k] = v
        missing, unexpected = self.model.load_state_dict(new_state, strict=False)
        print(f"[SwinUNETR] missing={len(missing)}, unexpected={len(unexpected)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# VoCo baseline (Wu et al. CVPR'24)
# ---------------------------------------------------------------------------

class VoCoSeg(nn.Module):
    """
    SwinUNETR with VoCo self-supervised pre-trained weights.

    Download weights from HuggingFace:
        huggingface-cli download Luffy503/VoCo --repo-type model --local-dir ./pretrained

    Recommended: VoCo_B_SSL_head.pt  (Base, feature_size=48, SSL pre-train)
                 VoComni_B.pt        (Base, feature_size=48, omni-supervised)
    """

    def __init__(
        self,
        num_classes: int = 9,
        img_size: tuple = (96, 96, 96),
        feature_size: int = 48,
        pretrained_path: str | None = None,
    ):
        super().__init__()
        from monai.networks.nets import SwinUNETR

        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=1,
            out_channels=num_classes,
            feature_size=feature_size,
            use_checkpoint=True,
        )

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        # Strip "module." prefix (DataParallel)
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        # VoCo uses "swin_vit" instead of "swinViT"
        state = {k.replace("swin_vit", "swinViT"): v for k, v in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        print(f"[VoCo] missing={len(missing)}, unexpected={len(unexpected)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# SuPreM baseline (Lee et al. / MrGiovanni)
# ---------------------------------------------------------------------------

class SuPremSeg(nn.Module):
    """
    SwinUNETR with SuPreM supervised pre-trained weights.

    Download weights from HuggingFace:
        huggingface-cli download MrGiovanni/SuPreM --repo-type model --local-dir ./pretrained

    Recommended: supervised_suprem_swinunetr_2100.pth  (SwinUNETR, feature_size=48)
    """

    def __init__(
        self,
        num_classes: int = 9,
        img_size: tuple = (96, 96, 96),
        feature_size: int = 48,
        pretrained_path: str | None = None,
    ):
        super().__init__()
        from monai.networks.nets import SwinUNETR

        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=1,
            out_channels=num_classes,
            feature_size=feature_size,
            use_checkpoint=True,
        )

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        # SuPreM uses custom arch: "module.backbone.swinViT.*" → "swinViT.*"
        state = ckpt.get("net", ckpt.get("state_dict", ckpt.get("model", ckpt)))
        new_state = {}
        for k, v in state.items():
            # strip "module.backbone." prefix to get swinViT weights
            if k.startswith("module.backbone."):
                new_state[k[len("module.backbone."):]] = v
        missing, unexpected = self.model.load_state_dict(new_state, strict=False)
        print(f"[SuPreM] loaded swinViT backbone: missing={len(missing)}, unexpected={len(unexpected)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "nemesis":    NEMESISSeg,
    "random_vit": RandomViTSeg,
    "swinunetr":  SwinUNETRSeg,
    "voco":       VoCoSeg,
    "suprem":     SuPremSeg,
}


def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
