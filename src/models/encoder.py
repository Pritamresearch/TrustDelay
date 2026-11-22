"""
Module 1: Ego Perception Encoder & Anchor Query Generator.
Extracts compact BEV features and candidate 3D queries from ego sensor observations.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BEVEncoder(nn.Module):
    """
    Compact BEV CNN Backbone for Real-Time Inference on Edge / RTX 3050 GPUs.
    Takes (B, in_channels, H, W) -> Returns (B, feat_dim, H, W)
    """
    def __init__(self, in_channels: int = 4, feat_dim: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.block1 = ConvBlock(32, 48, stride=1)
        self.block2 = ConvBlock(48, feat_dim, stride=1)
        self.out_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        return self.out_conv(h)


class QueryGenerator(nn.Module):
    """
    Extracts sparse 3D candidate queries from BEV feature maps using differentiable grid anchor sampling.
    Generates (B, num_queries, feat_dim) and initial bounding box candidates (B, num_queries, 7).
    """
    def __init__(self, feat_dim: int = 64, num_queries: int = 121, grid_w: int = 100, grid_h: int = 100):
        super().__init__()
        import math
        self.feat_dim = feat_dim
        self.num_queries = num_queries
        self.grid_w = grid_w
        self.grid_h = grid_h

        # Compute fixed 2D grid anchors in metric coordinates [-42, 42] x [-42, 42]
        side = int(round(math.sqrt(num_queries)))
        xs = np.linspace(-42.0, 42.0, side, dtype=np.float32)
        ys = np.linspace(-42.0, 42.0, side, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        grid_pts = np.stack([gx.flatten(), gy.flatten()], axis=-1)  # (N, 2)
        self.register_buffer("grid_anchors", torch.from_numpy(grid_pts))

        # Query feature projection and refinement
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.score_head = nn.Linear(feat_dim, 1)
        self.box_head = nn.Linear(feat_dim, 7)

        # Initialize box head to small residuals
        nn.init.zeros_(self.box_head.weight)
        nn.init.zeros_(self.box_head.bias)

    def forward(self, bev_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            queries: (B, num_queries, feat_dim)
            query_boxes: (B, num_queries, 7) [cx, cy, cz, dx, dy, dz, yaw]
            query_scores: (B, num_queries)
        """
        B, C, H, W = bev_feat.shape
        device = bev_feat.device
        anchors = self.grid_anchors.to(device)  # (N, 2) in meters [-42, 42]
        N = anchors.shape[0]

        # Normalized coordinates for F.grid_sample: [-1, 1]
        # x corresponds to row/H, y corresponds to col/W
        norm_x = (anchors[:, 0] / 50.0).clamp(-1.0, 1.0)
        norm_y = (anchors[:, 1] / 50.0).clamp(-1.0, 1.0)
        # grid_sample expects (x, y) where x is horizontal/col/W and y is vertical/row/H
        grid = torch.stack([norm_y, norm_x], dim=-1).unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1)  # (B, N, 1, 2)

        # Differentiable feature sampling at anchor locations
        sampled = F.grid_sample(bev_feat, grid, align_corners=True).squeeze(-1).permute(0, 2, 1)  # (B, N, C)
        query_feats = self.feat_mlp(sampled)

        scores = torch.sigmoid(self.score_head(query_feats)).squeeze(-1)  # (B, N)
        res = self.box_head(query_feats)  # (B, N, 7)

        # Reconstruct metric 3D boxes from grid anchors and learned residuals
        cx = anchors[:, 0:1].unsqueeze(0) + res[:, :, 0:1]
        cy = anchors[:, 1:2].unsqueeze(0) + res[:, :, 1:2]
        cz = -1.2 + res[:, :, 2:3]
        dx = 4.6 * torch.exp(torch.clamp(res[:, :, 3:4], -1.0, 1.0))
        dy = 2.1 * torch.exp(torch.clamp(res[:, :, 4:5], -1.0, 1.0))
        dz = 1.6 * torch.exp(torch.clamp(res[:, :, 5:6], -1.0, 1.0))
        yaw = res[:, :, 6:7]

        query_boxes = torch.cat([cx, cy, cz, dx, dy, dz, yaw], dim=-1)
        return query_feats, query_boxes, scores
