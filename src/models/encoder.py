"""
Module 1: Ego Perception Encoder & Anchor Query Generator.
Extracts compact BEV features and candidate 3D queries from ego sensor observations.
"""
from __future__ import annotations
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
    Extracts sparse 3D candidate queries from BEV feature maps.
    Generates (B, num_queries, feat_dim) and initial bounding box candidates (B, num_queries, 7).
    """
    def __init__(self, feat_dim: int = 64, num_queries: int = 64, grid_w: int = 100, grid_h: int = 100):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_queries = num_queries
        self.grid_w = grid_w
        self.grid_h = grid_h

        # Heatmap head for spatial object priors
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(feat_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
        # Bounding box attribute head: [cx, cy, cz, dx, dy, dz, yaw]
        self.box_head = nn.Sequential(
            nn.Conv2d(feat_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 7, kernel_size=1)
        )

    def forward(self, bev_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            queries: (B, num_queries, feat_dim)
            query_boxes: (B, num_queries, 7) [cx, cy, cz, dx, dy, dz, yaw]
            query_scores: (B, num_queries)
        """
        B, C, H, W = bev_feat.shape
        heatmap = torch.sigmoid(self.heatmap_head(bev_feat))  # (B, 1, H, W)
        box_maps = self.box_head(bev_feat)                   # (B, 7, H, W)

        # Flatten spatial dimensions
        flat_heat = heatmap.view(B, -1)                      # (B, H*W)
        flat_feat = bev_feat.flatten(2).permute(0, 2, 1)     # (B, H*W, C)
        flat_boxes = box_maps.flatten(2).permute(0, 2, 1)    # (B, H*W, 7)

        # Top-K query sampling based on heatmap response
        k = min(self.num_queries, flat_heat.shape[1])
        top_scores, top_indices = torch.topk(flat_heat, k=k, dim=1)

        idx_expanded_feat = top_indices.unsqueeze(-1).expand(-1, -1, C)
        idx_expanded_box = top_indices.unsqueeze(-1).expand(-1, -1, 7)

        queries = torch.gather(flat_feat, 1, idx_expanded_feat)
        query_boxes = torch.gather(flat_boxes, 1, idx_expanded_box)

        # Re-center coordinates in metric ego frame
        xs = (top_indices % W).float() / W * 100.0 - 50.0
        ys = (top_indices // W).float() / H * 100.0 - 50.0
        query_boxes[:, :, 0] += xs
        query_boxes[:, :, 1] += ys

        return queries, query_boxes, top_scores
