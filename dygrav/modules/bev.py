"""Lightweight BEV/polar histogram builder (fixed-iteration, pre-allocated)."""

from __future__ import annotations
from typing import Optional
import torch


class BEVBuilder:
    def __init__(self, depth_h: int = 480, depth_w: int = 640, r_bins: int = 32, theta_bins: int = 36):
        self.depth_h = depth_h
        self.depth_w = depth_w
        self.r_bins = r_bins
        self.theta_bins = theta_bins
        # Precompute angle lookup per pixel column
        cols = torch.arange(depth_w).float()
        self.theta_lookup = (cols / max(1, depth_w - 1)) * (2 * torch.pi)

    @torch.no_grad()
    def build(self, depth_map: torch.Tensor) -> torch.Tensor:
        """Convert a depth map [H,W] into a fixed-size polar histogram [theta_bins, r_bins]."""
        H, W = depth_map.shape[-2], depth_map.shape[-1]
        # Clamp and scale radius to bins
        r = torch.clamp(depth_map, min=0.0)
        r_norm = torch.clamp(r / (r.max() + 1e-6), 0, 0.999)
        r_idx = (r_norm * self.r_bins).long()
        # Tile theta per row
        theta = self.theta_lookup[:W].to(depth_map.device).unsqueeze(0).expand(H, W)
        theta_norm = torch.clamp(theta / (2 * torch.pi), 0, 0.999)
        t_idx = (theta_norm * self.theta_bins).long()
        bev = torch.zeros(self.theta_bins, self.r_bins, device=depth_map.device)
        # Accumulate counts (fixed iteration over pixels)
        for y in range(H):
            ti = t_idx[y]
            ri = r_idx[y]
            bev.index_put_((ti, ri), torch.ones_like(ri, dtype=bev.dtype), accumulate=True)
        # Normalize
        bev = bev / bev.sum().clamp_min(1e-6)
        # Flatten to vector
        return bev.flatten()


