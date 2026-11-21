"""
Module 3: Uncertainty-Aware Motion Extrapolator.
Extrapolates delayed object states and query tokens to the ego vehicle's current timestamp:
    x_hat_{i,t} = f_theta(x_{i, t - Delta t_i}, v_i, a_i, Delta t_i, sigma_i)
Predicts per-object motion uncertainty and assigns confidence scores.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class MotionExtrapolator(nn.Module):
    """
    Learned Kinematic & Uncertainty-Aware Motion Extrapolation Head.
    Operates on delayed 3D bounding boxes and query features.
    """
    def __init__(self, feat_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.feat_dim = feat_dim

        # Input: [feat_dim + 7 (box) + 1 (delay_sec) + 2 (velocity) + 2 (accel)]
        in_dim = feat_dim + 7 + 1 + 2 + 2

        self.motion_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Output: [dx, dy, dz, dyaw] residual corrections
        self.delta_pos_head = nn.Linear(hidden_dim, 4)

        # Output: [vx, vy, ax, ay] estimated kinematic parameters
        self.kinematics_head = nn.Linear(hidden_dim, 4)

        # Uncertainty head: outputs log(sigma^2) for [position, heading]
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2)  # [log_sigma_pos^2, log_sigma_yaw^2]
        )

        # Feature updater: updates query feature to account for motion and temporal transition
        self.feature_updater = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True)
        )

        # Standard zero-initialization for residual heads (ensures clean initialization at baseline)
        nn.init.zeros_(self.delta_pos_head.weight)
        nn.init.zeros_(self.delta_pos_head.bias)
        nn.init.zeros_(self.uncertainty_head[2].weight)
        nn.init.zeros_(self.uncertainty_head[2].bias)
        nn.init.zeros_(self.kinematics_head.weight)
        nn.init.zeros_(self.kinematics_head.bias)

    def forward(
        self,
        query_feats: torch.Tensor,       # (B, N, C)
        delayed_boxes: torch.Tensor,     # (B, N, 7) [cx, cy, cz, dx, dy, dz, yaw]
        delay_sec: torch.Tensor,         # (B, 1) or (B, N, 1)
        mode: str = "learned_uncertainty",
        initial_vel: Optional[torch.Tensor] = None # (B, N, 2) [vx, vy] optional prior
    ) -> Dict[str, torch.Tensor]:
        """
        Extrapolates state to current ego timestamp according to specified mode:
          - 'none': identity (no correction)
          - 'linear': linear extrapolation with constant velocity assumption
          - 'constant_velocity': 1st-order constant velocity kinematic model (a = 0)
          - 'learned_uncertainty': full TrustDelay uncertainty-aware motion extrapolator (v*dt + 0.5*a*dt^2 + delta)
        """
        B, N, C = query_feats.shape
        if delay_sec.dim() == 2:
            delay_sec = delay_sec.unsqueeze(1)
        if delay_sec.shape[1] != N:
            delay_sec = delay_sec.expand(B, N, 1)

        if mode == "none":
            # Baseline: Zero extrapolation under delay
            dummy_log_var = torch.zeros((B, N, 2), device=query_feats.device)
            return {
                "extrapolated_boxes": delayed_boxes.clone(),
                "extrapolated_feats": query_feats.clone(),
                "uncertainty_sigma": torch.ones((B, N), device=query_feats.device) * 0.5,
                "motion_confidence": torch.ones((B, N), device=query_feats.device),
                "log_var": dummy_log_var
            }

        # Initialize default velocities: use initial_vel if provided; else initialize to 0 so static cars aren't pushed
        if initial_vel is not None:
            init_vx = initial_vel[:, :, 0:1]
            init_vy = initial_vel[:, :, 1:2]
        else:
            init_vx = torch.zeros((B, N, 1), device=query_feats.device)
            init_vy = torch.zeros((B, N, 1), device=query_feats.device)
        init_ax = torch.zeros_like(init_vx)
        init_ay = torch.zeros_like(init_vy)

        # Assemble motion feature
        kin_input = torch.cat([query_feats, delayed_boxes, delay_sec, init_vx, init_vy, init_ax, init_ay], dim=-1)
        h = self.motion_net(kin_input)

        # Predict learned kinematics & residuals
        kin_preds = self.kinematics_head(h)  # [vx, vy, ax, ay]
        pred_vx = kin_preds[:, :, 0:1] + init_vx
        pred_vy = kin_preds[:, :, 1:2] + init_vy
        pred_ax = kin_preds[:, :, 2:3]
        pred_ay = kin_preds[:, :, 3:4]

        delta = self.delta_pos_head(h)       # [dcx, dcy, dcz, dyaw]
        log_var = self.uncertainty_head(h)   # [log_sigma_pos^2, log_sigma_yaw^2]
        log_var = torch.clamp(log_var, min=-4.0, max=4.0)

        dt = delay_sec
        if mode in ("linear", "constant_velocity"):
            # Constant velocity kinematics: x + v * dt (acceleration a = 0, no learned residuals)
            extrap_cx = delayed_boxes[:, :, 0:1] + pred_vx * dt
            extrap_cy = delayed_boxes[:, :, 1:2] + pred_vy * dt
            extrap_cz = delayed_boxes[:, :, 2:3]
            extrap_yaw = delayed_boxes[:, :, 6:7]
        else:
            # Full TrustDelay learned uncertainty-aware extrapolation: 2nd-order kinematics + learned residual
            extrap_cx = delayed_boxes[:, :, 0:1] + pred_vx * dt + 0.5 * pred_ax * (dt ** 2) + delta[:, :, 0:1]
            extrap_cy = delayed_boxes[:, :, 1:2] + pred_vy * dt + 0.5 * pred_ay * (dt ** 2) + delta[:, :, 1:2]
            extrap_cz = delayed_boxes[:, :, 2:3] + delta[:, :, 2:3]
            extrap_yaw = delayed_boxes[:, :, 6:7] + delta[:, :, 3:4]

        extrapolated_boxes = torch.cat([
            extrap_cx, extrap_cy, extrap_cz,
            delayed_boxes[:, :, 3:6],  # retain dimensions dx, dy, dz
            extrap_yaw
        ], dim=-1)

        # Uncertainty metrics
        sigma_pos = torch.exp(0.5 * log_var[:, :, 0])  # std dev in meters
        # As delay increases, aleatoric/motion variance grows proportionally
        scaled_sigma = sigma_pos * (1.0 + 2.0 * dt.squeeze(-1))
        # Motion confidence in [0, 1]: decays with uncertainty
        motion_conf = torch.exp(-0.5 * scaled_sigma)

        # Updated feature representation
        updated_feats = self.feature_updater(torch.cat([query_feats, h], dim=-1))

        return {
            "extrapolated_boxes": extrapolated_boxes,
            "extrapolated_feats": updated_feats,
            "uncertainty_sigma": scaled_sigma,
            "motion_confidence": motion_conf,
            "log_var": log_var
        }
