"""
Module 5: Reliability-Weighted Asynchronous Fusion.
Fuses ego perception queries with extrapolated collaborator tokens using reliability weights:
    w_i = g_phi(Delta t_i, sigma_i, p_loss, epsilon_pose)
Down-weights tokens suffering from extreme delay, high motion uncertainty, packet drop, or pose noise.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class ReliabilityGatingNetwork(nn.Module):
    """
    Computes token-wise reliability weight w_i in [0, 1] based on communication delay,
    motion uncertainty, packet reception indicator, and pose noise magnitude.
    """
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        # Inputs: [delay_sec, uncertainty_sigma, packet_valid_flag, pose_noise_mag]
        self.gate_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        delay_sec: torch.Tensor,         # (B, M, 1)
        uncertainty_sigma: torch.Tensor, # (B, M, 1)
        packet_valid: torch.Tensor,      # (B, M, 1)  (1.0 if received, 0.0 if lost)
        pose_noise_mag: torch.Tensor     # (B, M, 1)
    ) -> torch.Tensor:
        inputs = torch.cat([delay_sec, uncertainty_sigma, packet_valid, pose_noise_mag], dim=-1)
        # Physics-guided baseline prior:
        # High delay -> lower weight; high sigma -> lower weight; packet lost -> zero weight
        analytic_gate = torch.exp(-1.5 * delay_sec) * torch.exp(-0.8 * uncertainty_sigma) * torch.exp(-0.5 * pose_noise_mag)
        learned_gate = self.gate_mlp(inputs)
        # Blend learned with physical prior and strictly enforce zero weight on lost packets
        w = packet_valid * (0.5 * learned_gate + 0.5 * analytic_gate)
        return w.clamp(min=0.0, max=1.0)


class ReliabilityWeightedFusion(nn.Module):
    """
    Cross-Attention Transformer Decoder with Reliability Weighting.
    Fuses ego queries with collaborator evidence.
    """
    def __init__(self, feat_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads

        self.reliability_gate = ReliabilityGatingNetwork(hidden_dim=32)

        # 3D Metric Spatial Positional Embedding
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

        # Multi-Head Attention: Ego queries attend to collaborator keys/values
        self.cross_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim)
        )

        # Final 3D detection heads
        self.cls_head = nn.Linear(feat_dim, 1)   # vehicle presence logit
        self.reg_head = nn.Linear(feat_dim, 7)   # refined box [cx, cy, cz, dx, dy, dz, yaw]

        # Standard zero-initialization for bounding box regression residual head
        nn.init.zeros_(self.reg_head.weight)
        nn.init.zeros_(self.reg_head.bias)

    def forward(
        self,
        ego_queries: torch.Tensor,          # (B, N_ego, C)
        ego_boxes: torch.Tensor,            # (B, N_ego, 7)
        collab_feats: torch.Tensor,         # (B, M, C)
        collab_boxes: torch.Tensor,         # (B, M, 7)
        delay_sec: torch.Tensor,            # (B, M, 1)
        collab_sigmas: torch.Tensor,        # (B, M, 1)
        packet_mask: torch.Tensor,          # (B, M, 1)
        pose_noise_mag: torch.Tensor,       # (B, M, 1)
        fusion_type: str = "reliability_gated"
    ) -> Dict[str, torch.Tensor]:
        """
        Supports fusion modes:
          - 'uniform': standard equal-weight averaging without reliability gating
          - 'confidence_only': weights purely by object detection confidence
          - 'reliability_gated': full TrustDelay reliability-weighted cross-attention
        """
        B, N_ego, C = ego_queries.shape
        M = collab_feats.shape[1]

        if M == 0:
            # No collaborators selected or available: ego-only fallback
            logits = self.cls_head(ego_queries)
            res = self.reg_head(ego_queries)
            refined_boxes = torch.cat([
                ego_boxes[:, :, :3] + res[:, :, :3],
                ego_boxes[:, :, 3:6] * torch.exp(torch.clamp(res[:, :, 3:6], -1.0, 1.0)),
                ego_boxes[:, :, 6:7] + res[:, :, 6:7]
            ], dim=-1)
            return {
                "fused_queries": ego_queries,
                "pred_logits": logits.squeeze(-1),
                "pred_boxes": refined_boxes,
                "reliability_weights": torch.empty((B, 0), device=ego_queries.device)
            }

        if fusion_type == "uniform":
            weights = torch.ones((B, M, 1), device=ego_queries.device) * packet_mask
        elif fusion_type == "confidence_only":
            weights = torch.exp(-0.5 * collab_sigmas) * packet_mask
        elif fusion_type == "prior_only":
            # E4: Pure exponential prior, drops learned MLP
            analytic_gate = torch.exp(-1.5 * delay_sec) * torch.exp(-0.8 * collab_sigmas) * torch.exp(-0.5 * pose_noise_mag)
            weights = (packet_mask * analytic_gate).clamp(min=0.0, max=1.0)
        elif fusion_type == "mlp_only":
            # E4: Pure learned MLP gate, drops exponential prior
            inputs = torch.cat([delay_sec, collab_sigmas, packet_mask, pose_noise_mag], dim=-1)
            learned_gate = self.reliability_gate.gate_mlp(inputs)
            weights = (packet_mask * learned_gate).clamp(min=0.0, max=1.0)
        elif fusion_type == "delay_only":
            # E4: Hold sigma, delta, epsilon at nominal constants (sigma=0.5, delta=1.0, epsilon=0.0), so only Delta_t varies
            inputs = torch.cat([delay_sec, torch.full_like(collab_sigmas, 0.5), packet_mask, torch.zeros_like(pose_noise_mag)], dim=-1)
            analytic_gate = torch.exp(-1.5 * delay_sec) * math.exp(-0.8 * 0.5)
            learned_gate = self.reliability_gate.gate_mlp(inputs)
            w = packet_mask * (0.5 * learned_gate + 0.5 * analytic_gate)
            weights = w.clamp(min=0.0, max=1.0)
        else:
            # Full TrustDelay reliability weighting
            weights = self.reliability_gate(delay_sec, collab_sigmas, packet_mask, pose_noise_mag)

        # Scale collaborator features by reliability weights
        weighted_collab_feats = collab_feats * weights

        # Spatial positional encodings for queries and keys
        pos_ego = self.pos_encoder(ego_boxes[:, :, :3])
        pos_collab = self.pos_encoder(collab_boxes[:, :, :3])

        # Cross Attention: Ego queries attend to weighted collaborator evidence
        attn_out, attn_weights = self.cross_attn(
            query=ego_queries + pos_ego,
            key=weighted_collab_feats + pos_collab,
            value=weighted_collab_feats
        )

        x = self.norm1(ego_queries + attn_out)
        fused_queries = self.norm2(x + self.ffn(x))

        # Predict detection scores and refined 3D bounding boxes
        pred_logits = self.cls_head(fused_queries).squeeze(-1)
        box_residuals = self.reg_head(fused_queries)
        refined_boxes = torch.cat([
            ego_boxes[:, :, :3] + box_residuals[:, :, :3],
            ego_boxes[:, :, 3:6] * torch.exp(torch.clamp(box_residuals[:, :, 3:6], -1.0, 1.0)),
            ego_boxes[:, :, 6:7] + box_residuals[:, :, 6:7]
        ], dim=-1)

        return {
            "fused_queries": fused_queries,
            "pred_logits": pred_logits,
            "pred_boxes": refined_boxes,
            "reliability_weights": weights.squeeze(-1),
            "attn_weights": attn_weights
        }
