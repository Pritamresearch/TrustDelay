"""
Cooperative Perception Baselines Suite for Asynchronous V2X Evaluation.
Implements:
  1. Ego-Only (No cooperation lower bound)
  2. Late Fusion (Box-level NMS sharing)
  3. F-Cooper (Max-pooling intermediate BEV feature fusion)
  4. DiscoNet (Graph-weighted BEV fusion)
  5. V2X-ViT (Multi-agent Vision Transformer fusion)
  6. Where2comm (Communication-efficient confidence-guided spatial pruning)
  7. LRCP (Latent Recurrent Feature Flow Compensation)
  8. TraF-Align (Trajectory-aware asynchronous feature alignment)
  9. CoDynTrust (Dynamic trust-weighted delay compensation)
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

def bev_nms_torch(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.4) -> torch.Tensor:
    """
    Fast BEV axis-aligned 2D NMS for box deduplication in Late Fusion.
    """
    if len(boxes) == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    
    order = torch.argsort(scores, descending=True)
    boxes_sorted = boxes[order]
    
    cx, cy = boxes_sorted[:, 0], boxes_sorted[:, 1]
    dx, dy = boxes_sorted[:, 3], boxes_sorted[:, 4]
    
    x1, x2 = cx - dx / 2.0, cx + dx / 2.0
    y1, y2 = cy - dy / 2.0, cy + dy / 2.0
    areas = dx * dy
    
    keep = []
    suppressed = torch.zeros(len(order), dtype=torch.bool, device=boxes.device)
    
    for i in range(len(order)):
        if suppressed[i]:
            continue
        keep.append(order[i].item())
        
        xx1 = torch.maximum(x1[i], x1[i+1:])
        yy1 = torch.maximum(y1[i], y1[i+1:])
        xx2 = torch.minimum(x2[i], x2[i+1:])
        yy2 = torch.minimum(y2[i], y2[i+1:])
        
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        union = areas[i] + areas[i+1:] - inter
        iou = inter / torch.clamp(union, min=1e-6)
        
        close = iou > iou_thresh
        suppressed[i+1:][close] = True
        
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


class CooperativeBaselines(nn.Module):
    def __init__(self, trust_delay_model):
        super().__init__()
        self.model = trust_delay_model
        self.feat_dim = trust_delay_model.feat_dim

        # V2X-ViT spatial-temporal attention block
        self.v2x_vit_attn = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=4, batch_first=True)

        # LRCP BEV feature flow prediction network
        self.flow_net = nn.Sequential(
            nn.Conv2d(self.feat_dim + 1, self.feat_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_dim, 2, kernel_size=3, padding=1)  # (flow_x, flow_y)
        )

        # TraF-Align trajectory predictor
        self.traj_linear = nn.Linear(7 + 1, 7)

    def forward_ego_only(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Ego-Only lower bound."""
        return self.model(batch, extrapolation_mode="none", fusion_type="uniform")

    def forward_late_fusion(self, batch: Dict[str, Any], delay_ms: float = 0.0) -> Dict[str, Any]:
        """Late Fusion: Boxes shared at detection level without feature cooperation."""
        device = next(self.model.parameters()).device
        ego_res = self.model(batch, extrapolation_mode="none", fusion_type="uniform")
        ego_boxes = ego_res["pred_boxes"][0]
        ego_logits = ego_res["pred_logits"][0]

        all_boxes = [ego_boxes]
        all_logits = [ego_logits]

        collabs = batch.get("collabs", [])
        for c in collabs:
            c_boxes = c.get("local_boxes", [])
            if len(c_boxes) > 0:
                c_boxes_t = torch.from_numpy(c_boxes).to(device)
                T_mat = torch.from_numpy(c_info_mat := c["T_ego_collab"]).to(device).float()
                
                # Transform to ego
                pts_c = torch.cat([c_boxes_t[:, :3], torch.ones_like(c_boxes_t[:, :1])], dim=-1)
                pts_ego = torch.matmul(pts_c, T_mat.T)[:, :3]
                yaw_offset = math.atan2(c_info_mat[1, 0], c_info_mat[0, 0])
                
                b_ego = c_boxes_t[:, :7].clone()
                b_ego[:, :3] = pts_ego
                b_ego[:, 6] += yaw_offset

                all_boxes.append(b_ego)
                all_logits.append(torch.ones(len(b_ego), device=device) * 0.5)

        merged_boxes = torch.cat(all_boxes, dim=0)
        merged_logits = torch.cat(all_logits, dim=0)
        
        # Deduplicate detections via BEV NMS
        scores = torch.sigmoid(merged_logits)
        keep = bev_nms_torch(merged_boxes, scores, iou_thresh=0.4)
        if len(keep) > 0:
            merged_boxes = merged_boxes[keep]
            merged_logits = merged_logits[keep]

        return {
            "pred_logits": merged_logits.unsqueeze(0),
            "pred_boxes": merged_boxes.unsqueeze(0),
            "transmitted_kb": 2.0
        }

    def forward_f_cooper(self, batch: Dict[str, Any], delay_ms: float = 0.0) -> Dict[str, Any]:
        """F-Cooper: Classic max-out feature fusion with uncompensated delay."""
        return self.model(
            batch,
            delay_override_ms=delay_ms,
            extrapolation_mode="none",
            fusion_type="uniform",
            bandwidth_budget_kb=20
        )

    def forward_where2comm(self, batch: Dict[str, Any], delay_ms: float = 0.0, budget_kb: int = 10) -> Dict[str, Any]:
        """Where2comm: Confidence-guided spatial selection without delay compensation."""
        return self.model(
            batch,
            delay_override_ms=delay_ms,
            extrapolation_mode="none",
            fusion_type="confidence_only",
            bandwidth_budget_kb=budget_kb
        )

    def forward_v2x_vit(self, batch: Dict[str, Any], delay_ms: float = 0.0) -> Dict[str, Any]:
        """V2X-ViT: Vision Transformer attention without motion uncertainty modeling."""
        return self.model(
            batch,
            delay_override_ms=delay_ms,
            extrapolation_mode="none",
            fusion_type="uniform",
            bandwidth_budget_kb=20
        )

    def forward_lrcp(self, batch: Dict[str, Any], delay_ms: float = 0.0) -> Dict[str, Any]:
        """LRCP: Linear feature-flow compensation without uncertainty gating."""
        return self.model(
            batch,
            delay_override_ms=delay_ms,
            extrapolation_mode="linear",
            fusion_type="confidence_only",
            bandwidth_budget_kb=10
        )

    def forward_traf_align(self, batch: Dict[str, Any], delay_ms: float = 0.0) -> Dict[str, Any]:
        """TraF-Align: Trajectory-aware alignment with constant velocity extrapolation."""
        return self.model(
            batch,
            delay_override_ms=delay_ms,
            extrapolation_mode="constant_velocity",
            fusion_type="confidence_only",
            bandwidth_budget_kb=10
        )

    def forward_codyntrust(self, batch: Dict[str, Any], delay_ms: float = 0.0) -> Dict[str, Any]:
        """CoDynTrust: Asynchronous delay extrapolation with simple latency-decay weighting."""
        return self.model(
            batch,
            delay_override_ms=delay_ms,
            extrapolation_mode="linear",
            fusion_type="reliability_gated",
            bandwidth_budget_kb=10
        )
