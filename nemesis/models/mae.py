#!/usr/bin/env python3
"""
MAEgic3DMAE - 3D Masked Autoencoder for Medical Volumes

This module implements the core MAEgic 3D MAE model specifically designed
for medical CT volumes with superpatch-based training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, Optional, Tuple, Union
import logging
import math


logger = logging.getLogger(__name__)
def _log_stats(tag: str, t: torch.Tensor):
    with torch.no_grad():
        logger.info(f"{tag}: shape={tuple(t.shape)}, "
                    f"mean={float(t.mean()):.6f}, std={float(t.std()):.6f}, "
                    f"min={float(t.min()):.6f}, max={float(t.max()):.6f}")


class SinusoidalPE(nn.Module):
    """
    Fixed sinusoidal positional encoding on patch grid (D_p, H_p, W_p).
    Returns (1, N=D_p*H_p*W_p, E).
    """
    def __init__(self, embed_dim: int, num_patches, max_period: int = 10_000):
        super().__init__()
        self.embed_dim = embed_dim
        self.D, self.H, self.W = num_patches
        self.max_period = max_period
        self.axis_dim = (embed_dim // 3, embed_dim // 3, embed_dim - embed_dim // 3 * 2)

        table = self._build_table()
        self.register_buffer("table", table, persistent=False)  # (1, N, E)

    def _axis_embed(self, L: int, dim: int, device, dtype):
        half = dim // 2
        pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)              # (L, 1)
        inv = torch.exp(-math.log(self.max_period) *
                        torch.arange(half, device=device, dtype=dtype) / half)      # (half,)
        ang = pos * inv.unsqueeze(0)                                                # (L, half)
        out = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        if out.shape[1] < dim:  # 홀수 보정
            out = F.pad(out, (0, dim - out.shape[1]))
        return out

    def _build_table(self):
        device, dtype = torch.device("cpu"), torch.float32
        dx, dy, dz = self.axis_dim
        ez = self._axis_embed(self.D, dz, device, dtype)  # (D_p, dz)
        ey = self._axis_embed(self.H, dy, device, dtype)  # (H_p, dy)
        ex = self._axis_embed(self.W, dx, device, dtype)  # (W_p, dx)

        zz, yy, xx = torch.meshgrid(
            torch.arange(self.D), torch.arange(self.H), torch.arange(self.W), indexing="ij"
        )
        N = self.D * self.H * self.W
        zvec = ez[zz.reshape(-1)]
        yvec = ey[yy.reshape(-1)]
        xvec = ex[xx.reshape(-1)]
        pe = torch.cat([zvec, yvec, xvec], dim=1).unsqueeze(0)  # (1, N, E)
        logger.info(f"[PE/BUILD] grid=({self.D},{self.H},{self.W}), E={self.embed_dim}, axis_dim={self.axis_dim}")
        return pe

    def forward(self, *, device=None, dtype=None):
        pe = self.table
        if device is not None or dtype is not None:
            pe = pe.to(device=device if device is not None else pe.device,
                    dtype=dtype if dtype is not None else pe.dtype)
        if not hasattr(self, "_pe_logged"):
            logger.info(f"[PE/FWD] pe_table={tuple(pe.shape)} device={pe.device} dtype={pe.dtype}")
            self._pe_logged = True
        return pe  # (1, N, E)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_patches,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        merge_mode: str = "avg",      # 'avg' | 'concat'
        cross: bool=True
    ):
        super().__init__()
        assert merge_mode in ("avg", "concat"), "merge_mode must be 'avg' or 'concat'."
        self.T = num_patches[0]
        self.merge_mode = merge_mode

        # 2D cross attentions
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.depth_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Merging
        if self.merge_mode == "concat":
            self.merge_linear = nn.Linear(embed_dim * 2, embed_dim)
        else:
            self.merge_linear = None

        # Norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # FFN (GeLU)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(dropout)
        )

    @staticmethod
    def _to_grid(x: torch.Tensor, T: int) -> torch.Tensor:
        """
        (B, N, E) -> (B, T, L, E) with L = N // T
        """
        B, N, E = x.shape
        assert N % T == 0, f"N({N}) must be divisible by T_fixed({T})."
        L = N // T

        return x.view(B, T, L, E), L

    @staticmethod
    def _to_seq(x_grid: torch.Tensor) -> torch.Tensor:
        """
        (B, T, L, E) -> (B, N, E)
        """
        B, T, L, E = x_grid.shape
        return x_grid.reshape(B, T * L, E)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, E)
        """
        B, N, E = x.shape

        # Reshape to (B, T, L, E)
        x_grid, L = self._to_grid(x, self.T)                            # (B, T, L, E)

        # Pre-norm for stability
        h = self.norm1(x_grid)                                          # (B, T, L, E)

        # Parallel self-attention branches
        in_spat = h.reshape(-1, L, E).contiguous()                      # (B*T, L, E)
        out_spat, _ = self.spatial_attn(in_spat, in_spat, in_spat, need_weights=False)
        out_spat = out_spat.reshape(-1, self.T, L, E)

        in_dept = h.permute(0, 2, 1, 3).reshape(-1, self.T, E).contiguous()         # (B*L, T, E)
        out_dept, _ = self.depth_attn(in_dept, in_dept, in_dept, need_weights=False)    # (B*L, T, E)
        out_dept = out_dept.reshape(-1, L, self.T, E).permute(0, 2, 1, 3).contiguous()  # (B, T, L, E)

        # Merge
        if self.merge_mode == "concat":
            merged = torch.cat([out_spat, out_dept], dim=-1)            # (B, T, L, 2*E)
            merged = self.merge_linear(merged)                          # (B, T, L, E)
        else:
            merged = 0.5 * (out_spat + out_dept)                        # (B, T, L, E)

        # Residual + FFN
        x_grid = x_grid + merged                                        # (B, T, L, E)
        x_grid = self.norm2(x_grid)
        x_grid = x_grid + self.ffn(x_grid)                              # (B, T, L, E)

        # Back to (B,N,D)
        return self._to_seq(x_grid)                                     # (B, N, E)


class AdaptivePatchEmbedding(nn.Module):
    """
    3D Adaptive Patch Embedding:
      - Linear path:    (B, N, p_d, p_h*p_w) --Linear(p_h*p_w->E)--> (B, N, p_d, E)
      - MT-Attn path:   concat([x_vis, MT]) -> self-attn -> MT rows만 취득 (B*N, N_mt, E)
      - Gate alpha:     sigmoid(alpha)로 두 경로 가중합
    """
    _log_once: bool = False  # class-level: survives DataParallel replica creation

    def __init__(self, embed_dim: int, num_maegic_tokens: int, patch_size, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_maegic_tokens == 0
        p_d, p_h, p_w = patch_size

        self.embed_dim = embed_dim
        self.sub_embed_dim = embed_dim // num_maegic_tokens
        self.num_maegic_tokens = num_maegic_tokens

        self.linear = nn.Linear(p_d * p_h * p_w, self.embed_dim, bias=True)

        # MT-Attn branch
        self.embed = nn.Linear(p_h * p_w, self.sub_embed_dim, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, p_d, self.sub_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.01)
        self.mt = nn.Parameter(torch.zeros(1, num_maegic_tokens, self.sub_embed_dim))
        nn.init.xavier_uniform_(self.mt)
        self.norm_qkv = nn.LayerNorm(self.sub_embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=self.sub_embed_dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.proj = nn.Linear(self.sub_embed_dim, self.sub_embed_dim)

        # Gating
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 초기 0.5 → sigmoid로 (0,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, p_d, p_hw = x.shape

        # [ONE-TIME LOG] class-level flag survives DataParallel replica creation
        first = not AdaptivePatchEmbedding._log_once
        if first:
            AdaptivePatchEmbedding._log_once = True
            # ▶ 최초 1회 찍고 싶은 상단 요약 로그(입력/세팅)
            logger.info(f"[APE] in x={tuple(x.shape)} (p_d={p_d}, p_hw={p_hw}), "
                        f"E={self.embed_dim}, E_sub={self.sub_embed_dim}, N_mt={self.num_maegic_tokens}")

        # Linear branch
        lin = self.linear(x.reshape(B, N, -1))  # (B, N, E)
        if first: _log_stats("[APE] lin", lin)

        # MT-Attn branch
        x_emb = self.embed(x).reshape(B * N, p_d, -1)
        x_emb = x_emb + self.pos_embed
        if first: _log_stats("[APE] x_emb(+pos)", x_emb)

        mt = self.mt.expand(B * N, -1, -1)
        seq = torch.cat([x_emb, mt], dim=1)
        seq = self.norm_qkv(seq)
        if first: _log_stats("[APE] seq(norm_qkv)", seq)

        attn_out, _ = self.attn(seq, seq, seq, need_weights=False)
        attn_out = self.proj(attn_out)
        if first: _log_stats("[APE] attn_out(proj)", attn_out)

        pat = attn_out[:, -self.num_maegic_tokens:, :].reshape(B, N, -1)
        if first: _log_stats("[APE] mt_path", pat)

        a = torch.sigmoid(self.alpha)
        h = (1 - a) * lin + a * pat
        if first:
            logger.info(f"[APE] gate alpha={float(a):.6f} out={tuple(h.shape)}")
            _log_stats("[APE] out", h)
        return h


class MAEgicEncoder(nn.Module):
    """
    Apply dual masking strategy:
    1. Mask spatial patches randomly
    2. Mask depth-wise additional patches randomly
    """
    _log_once: bool = False  # class-level: survives DataParallel replica creation


    def __init__(
        self,
        volume_size=(512, 512, 512),
        superpatch_size=(128, 128, 128),
        patch_size=(8, 8, 8),
        in_channels=1,
        embed_dim=768,
        depth=12,
        num_heads=8,
        dropout=0.1,
        drop_path_rate=0.1,
        pos_encoding_type='sinusoidal',
        num_maegic_tokens=8,
        spatial_mask_ratio=0,
        depth_mask_ratio=0,
        use_maegic_tokens=True,
    ):
        super().__init__()
        self.volume_size = volume_size
        self.superpatch_size = superpatch_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_maegic_tokens = num_maegic_tokens
        self.num_patches = [s // p for s, p in zip(superpatch_size, patch_size)]
        self.spatial_mask_ratio = spatial_mask_ratio
        self.depth_mask_ratio = depth_mask_ratio
        self.use_maegic_tokens = use_maegic_tokens

        if use_maegic_tokens:
            # Adaptive aggregator (3D Adaptive Patch Embedding)
            self.adaptive_agg = AdaptivePatchEmbedding(
                embed_dim=embed_dim, num_maegic_tokens=num_maegic_tokens, patch_size=patch_size,
                num_heads=num_heads, dropout=dropout
            )
        else:
            # Standard linear patch embedding (ablation: no MAEgic tokens)
            p_d, p_h, p_w = patch_size
            self.patch_embed = nn.Linear(p_d * p_h * p_w, embed_dim, bias=True)
            logger.info("[ABLATION] use_maegic_tokens=False: standard linear patch embedding")

        # Positional embedding
        assert pos_encoding_type == 'sinusoidal'
        if pos_encoding_type == 'sinusoidal':
            self.pos_embed = SinusoidalPE(embed_dim, self.num_patches)
            self.pe_scale = nn.Parameter(torch.tensor(0.1))
        # else:
        #     self.pos_embed = nn.Parameter(torch.zeros(1, np.prod(self.num_patches) * in_channels))
        #     nn.init.xavier_uniform_(self.pos_embed)
        #     # self.pe_scale = torch.tensor(1)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_patches=self.num_patches
            )
            for i in range(depth)
        ])

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
    
    def _patchify(self, x):
        """Convert volume to patches"""
        B, C, D, H, W = x.shape
        p_d, p_h, p_w = self.patch_size
        D_p, H_p, W_p = self.num_patches
        
        x = x.reshape(B, C, D_p, p_d, H_p, p_h, W_p, p_w)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)
        x = x.reshape(B, D_p * H_p * W_p, p_d, p_h * p_w * C)
        
        return x
    
    def forward(self, x, positions=None):
        x = self._patchify(x)  # (B, N, p_d, p_h*p_w)

        # [ONE-TIME LOG] class-level flag survives DataParallel replica creation
        enc_first = not MAEgicEncoder._log_once
        if enc_first:
            MAEgicEncoder._log_once = True

        # masking
        x_vis, mask, ids_keep = self._apply_dual_masking(x)
        B, N, _, _ = x.shape
        masked = mask.float().mean().item()
        if enc_first:
            logger.info(f"[MASK] actual_mask_ratio={masked:.4f} "
                        f"(expected≈ {1.0 - (1.0 - self.spatial_mask_ratio)*(1.0 - self.depth_mask_ratio):.4f})")

        # Patch embedding
        if self.use_maegic_tokens:
            h = self.adaptive_agg(x_vis)
        else:
            B_vis, N_vis, p_d, p_hw = x_vis.shape
            h = self.patch_embed(x_vis.reshape(B_vis, N_vis, -1))  # (B, N_vis, embed_dim)

        # pos embed (원래 있던 코드)
        pe = self.pos_embed(device=h.device, dtype=h.dtype).expand(h.shape[0], -1, -1)
        pe_vis = pe.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        if enc_first:
            logger.info(f"[ENC] N_total={x.shape[1]}, N_vis={x_vis.shape[1]}, ids_keep={tuple(ids_keep.shape)}")
            logger.info(f"[ENC] h(before PE)={tuple(h.shape)}, pe={tuple(pe.shape)}, "
                        f"pe_vis={tuple(pe_vis.shape)}, pe_scale={float(self.pe_scale.data)}")
            _log_stats("[ENC] h(before PE) stats", h)
            _log_stats("[ENC] pe_vis stats", pe_vis)

        h = h + self.pe_scale * pe_vis

        if enc_first:
            logger.info(f"[ENC] h(after  PE)={tuple(h.shape)}")
            _log_stats("[ENC] h(after  PE) stats", h)

        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return h, mask, ids_keep
    
    def _generate_patch_positions(self, batch_size, superpatch_positions):
        """Generate 3D positions for each patch within superpatches"""
        D_p, H_p, W_p = self.num_patches
        
        # Create relative positions within superpatch
        positions = []
        for d in range(D_p):
            for h in range(H_p):
                for w in range(W_p):
                    positions.append([d, h, w])
        
        positions = torch.tensor(positions, device=next(self.parameters()).device)
        positions = positions.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, 3)
        
        # If superpatch positions are provided, add them as offset
        if superpatch_positions is not None:
            # superpatch_positions: (B, 3) -> (B, 1, 3)
            offset = superpatch_positions.unsqueeze(1)
            positions = positions + offset
        
        return positions
    
    def _apply_dual_masking(self, x):
        """
        Apply dual masking strategy:
        1. Mask spatial patches randomly
        2. Mask depth-wise additional patches randomly
        """
        B, N, p_d, p_hw = x.shape
        D_p, H_p, W_p = self.num_patches
        spatial_patches = H_p * W_p

        # 학습시에만 마스킹, 평가/추론에서는 마스킹 0
        spatial_mask_ratio = self.spatial_mask_ratio if self.training else 0.0
        depth_mask_ratio   = self.depth_mask_ratio   if self.training else 0.0

        # 가시 패치 수 계산(최소 1 보장)
        num_spatial_vis = int(spatial_patches * (1.0 - spatial_mask_ratio))
        num_spatial_vis = max(1, min(spatial_patches, num_spatial_vis))

        num_depth_vis = int(num_spatial_vis * (1.0 - depth_mask_ratio))
        num_depth_vis = max(1, min(num_spatial_vis, num_depth_vis))

        # True=masked, False=visible
        mask = torch.ones(B, N, device=x.device, dtype=torch.bool)

        for b in range(B):
            # Spatial masking (공통 가시 패치 선택)
            spatial_vis = torch.randperm(spatial_patches, device=x.device)[:num_spatial_vis]
 
            for d in range(D_p):
                # Depth masking (depth별 추가 랜덤 가시 선택)
                depth_vis = torch.randperm(num_spatial_vis, device=x.device)[:num_depth_vis]
                mask[b, d * spatial_patches + spatial_vis[depth_vis]] = False

        # 가시 패치만 모아 순서를 복구할 인덱스 생성
        ids_restore = torch.argsort(mask.float(), dim=1)
        ids_keep = ids_restore[:, :num_depth_vis * D_p]

        # 가시 패치만 선택
        x_masked = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, p_d, p_hw)
        )

        return x_masked, mask, ids_keep


class MAEgicDecoder(nn.Module):
    """MAEgic decoder for reconstruction"""
    _log_once: bool = False  # class-level: survives DataParallel replica creation

    def __init__(
        self,
        volume_size=(512, 512, 512),
        superpatch_size=(128, 128, 128),
        patch_size=(8, 8, 8),
        in_channels=1,
        embed_dim=768,
        depth=8,
        num_heads=8,
        dropout=0.1,
        num_maegic_tokens=8,
        pos_encoding_type='sinusoidal',
    ):
        super().__init__()
        self.superpatch_size = superpatch_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_maegic_tokens = num_maegic_tokens
        self.num_patches = [s // p for s, p in zip(superpatch_size, patch_size)]
        # self.total_patches = np.prod(self.num_patches)  # 미사용이면 제거
        
        # Mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.mask_token)

        # Positional embedding
        assert pos_encoding_type == 'sinusoidal'
        if pos_encoding_type == 'sinusoidal':
            self.pos_embed = SinusoidalPE(embed_dim, self.num_patches)
            self.pe_scale = nn.Parameter(torch.tensor(0.1))
        # else:
        #     self.pos_embed = nn.Parameter(torch.zeros(1, np.prod(self.num_patches) * in_channels))
        #     nn.init.xavier_uniform_(self.pos_embed)
        #     # self.pe_scale = torch.tensor(1)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_patches=self.num_patches
            )
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Reconstruction head
        patch_dim = np.prod(patch_size) * in_channels
        self.head = nn.Linear(embed_dim, patch_dim)
    
    def forward(self, x_vis, ids_keep):
        B, N_vis = x_vis.shape[:2]
        N = int(np.prod(self.num_patches))

        x = self.mask_token.expand(B, N, -1).clone()
        x.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim), x_vis)

        # [ONE-TIME LOG] class-level flag survives DataParallel replica creation
        dec_first = not MAEgicDecoder._log_once
        if dec_first:
            MAEgicDecoder._log_once = True
            logger.info(f"[DEC] x_init(mask tokens)={tuple(x.shape)}, N={N}, N_vis={x_vis.shape[1]}")
            mi, ma = int(ids_keep.min()), int(ids_keep.max())
            logger.info(f"[DEC] scatter: ids_keep={tuple(ids_keep.shape)}, range=[{mi},{ma}]")
            _log_stats("[DEC] x(after scatter) stats", x)

        pos = self.pos_embed(device=x.device, dtype=x.dtype)
        if dec_first:
            logger.info(f"[DEC] pos_embed={tuple(pos.shape)}, pe_scale={float(self.pe_scale.data)}")
            _log_stats("[DEC] pos stats", pos)

        x = x + self.pe_scale * pos

        if dec_first:
            logger.info(f"[DEC] x(after  PE)={tuple(x.shape)}")
            _log_stats("[DEC] x(after  PE) stats", x)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        reconstructed = self.head(x)
        return reconstructed


class MAEgic3DMAE(nn.Module):
    """
    Complete MAEgic 3D MAE model for medical volumes.

    This is the main model class that combines encoder and decoder
    with medical-specific features and denoising capabilities.
    """
    _sync_once: bool = False  # class-level: survives DataParallel replica creation

    def __init__(
        self,
        volume_size=(512, 512, 512),
        superpatch_size=(128, 128, 128),
        patch_size=(8, 8, 8),
        in_channels=1,
        embed_dim=768,
        depth=12,
        num_heads=8,
        decoder_depth=8,
        decoder_num_heads=8,
        dropout=0.1,
        drop_path_rate=0.1,
        pos_encoding_type='sinusoidal',
        num_maegic_tokens=8,
        spatial_mask_ratio=0,
        depth_mask_ratio=0,
        use_maegic_tokens=True,
    ):
        super().__init__()

        self.volume_size = volume_size
        self.superpatch_size = superpatch_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.spatial_mask_ratio = spatial_mask_ratio
        self.depth_mask_ratio = depth_mask_ratio
        self.num_patches = [s // p for s, p in zip(superpatch_size, patch_size)]

        # Encoder
        self.encoder = MAEgicEncoder(
            volume_size=volume_size,
            superpatch_size=superpatch_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            pos_encoding_type=pos_encoding_type,
            num_maegic_tokens=num_maegic_tokens,
            spatial_mask_ratio=spatial_mask_ratio,
            depth_mask_ratio=depth_mask_ratio,
            use_maegic_tokens=use_maegic_tokens,
        )
        
        # Decoder  
        self.decoder = MAEgicDecoder(
            volume_size=volume_size,
            superpatch_size=superpatch_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            dropout=dropout,
            num_maegic_tokens=num_maegic_tokens,
            pos_encoding_type=pos_encoding_type,
        )
    
    def forward(self, x, positions=None, noise_std=0.0):
        """
        Args:
            x: (B, C, D, H, W) input superpatches
            positions: (B, 3) positions in original volume
            noise_std: noise level for denoised training
        Returns:
            loss_dict: dictionary of losses
            reconstructed: (B, C, D, H, W) reconstructed superpatches
            mask_info: dictionary with mask information
        """
        # keep encoder/decoder pe_scale identical (log once per phase)
        with torch.no_grad():
            dec_before = float(self.decoder.pe_scale.data)
            self.decoder.pe_scale.data.copy_(self.encoder.pe_scale.data)
            sync_first = not MAEgic3DMAE._sync_once
            if sync_first:
                MAEgic3DMAE._sync_once = True
                logger.info(
                    f"[SYNC] pe_scale enc→dec: {float(self.encoder.pe_scale.data):.6f} "
                    f"(dec_before={dec_before:.6f})"
                )     
        
        # Add noise for denoised training
        if noise_std > 0:
            noise = torch.randn_like(x) * noise_std
            x_noisy = x + noise
        else:
            x_noisy = x
        
        # Encode
        latent, mask, ids_keep = self.encoder(x_noisy, positions)
        
        # Decode
        pred_patches = self.decoder(latent, ids_keep)
        
        # Reshape to volume
        reconstructed = self._unpatchify(pred_patches)
        
        # Compute losses
        loss_dict = self._compute_losses(x, reconstructed, mask)
        
        # Mask info for visualization (use tensors for DataParallel gather compatibility)
        mask_info = {
            'mask': mask,
            'ids_keep': ids_keep,
            'spatial_mask_ratio': torch.tensor(self.spatial_mask_ratio, device=x.device),
            'depth_mask_ratio': torch.tensor(self.depth_mask_ratio, device=x.device),
        }
        
        return loss_dict, reconstructed, mask_info, x_noisy
    
    def _unpatchify(self, x):
        """Convert patch predictions back to volume"""
        B = x.shape[0]
        D_p, H_p, W_p = [s // p for s, p in zip(self.superpatch_size, self.patch_size)]
        p_d, p_h, p_w = self.patch_size
        C = self.in_channels
        
        # Reshape
        x = x.reshape(B, D_p, H_p, W_p, p_d, p_h, p_w, C)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        x = x.reshape(B, C, D_p * p_d, H_p * p_h, W_p * p_w)
        
        return x
    
    def _patchify(self, x):
        """Convert volume to patches"""
        B, C, D, H, W = x.shape
        p_d, p_h, p_w = self.patch_size
        D_p, H_p, W_p = self.num_patches
        
        x = x.reshape(B, C, D_p, p_d, H_p, p_h, W_p, p_w)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)
        x = x.reshape(B, D_p * H_p * W_p, p_d * p_h * p_w * C)
        
        return x

    def _compute_losses(self, target, pred, mask):
        """Compute reconstruction losses"""
        # Convert to patches
        target_patches = self._patchify(target)
        pred_patches = self._patchify(pred)
        
        # MSE loss on masked patches
        mse_loss = (pred_patches - target_patches) ** 2
        mse_loss = mse_loss.mean(dim=-1)  # (B, N)
        
        # Masked loss (only on masked patches)
        mask_sum = mask.sum()
        if mask_sum > 0:
            masked_mse = (mse_loss * mask).sum() / mask_sum
        else:
            # If no masking (mask_ratio=0), use full volume loss
            masked_mse = mse_loss.mean()
        
        # Full volume MSE (for monitoring)
        full_mse = F.mse_loss(pred, target)
        
        # Total loss
        total_loss = masked_mse

        # PSNR: -10 * log_10 (MSE)
        psnr = -10.0 * torch.log10(full_mse + 1e-8)
        
        return {
            'total_loss': total_loss,
            'recon_loss': masked_mse,
            'full_mse': full_mse,
            'masked_ratio': mask.float().mean(),
            'psnr': psnr
        }

    def reset_one_time_logs(self):
        """APE/ENC/DEC의 1회 로그 토글을 리셋 (다음 배치에서 다시 1회 찍힘)"""
        # 모델 레벨 SYNC 원타임 로그 리셋 (class-level)
        MAEgic3DMAE._sync_once = False
        MAEgicEncoder._log_once = False
        MAEgicDecoder._log_once = False
        AdaptivePatchEmbedding._log_once = False