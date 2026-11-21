"""
TrustDelay: Complete Unified Multi-Agent Architecture.
Integrates all 5 modules:
  1. Ego Perception Encoder & Query Generator
  2. Collaborator Selector
  3. Uncertainty-Aware Motion Extrapolator
  4. Bandwidth-Aware Message Selector
  5. Reliability-Weighted Asynchronous Fusion

Strictly leak-free: All proposals and queries are produced by the neural network from LiDAR point clouds.
Ground truth is never read, passed, or injected into any network tensor.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional

from src.models.encoder import BEVEncoder, QueryGenerator
from src.models.collaborator_selector import CollaboratorSelector
from src.models.motion_extrapolator import MotionExtrapolator
from src.models.bandwidth_selector import BandwidthMessageSelector
from src.models.reliability_fusion import ReliabilityWeightedFusion

class TrustDelayModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.feat_dim = cfg.feat_dim

        # Module 1: Ego Perception Encoder & Query Generator
        self.encoder = BEVEncoder(in_channels=4, feat_dim=self.feat_dim)
        self.query_gen = QueryGenerator(
            feat_dim=self.feat_dim,
            num_queries=cfg.num_queries,
            grid_w=cfg.bev_w,
            grid_h=cfg.bev_h
        )

        # Module 2: Collaborator Selector
        self.collab_selector = CollaboratorSelector(
            top_k=cfg.top_k_collaborators,
            w_perception=cfg.selector_weight_perception,
            w_distance=cfg.selector_weight_distance,
            w_rel_velocity=cfg.selector_weight_rel_velocity,
            w_comm_cost=cfg.selector_weight_comm_cost
        )

        # Module 3: Uncertainty-Aware Motion Extrapolator
        self.motion_extrapolator = MotionExtrapolator(feat_dim=self.feat_dim, hidden_dim=self.feat_dim)

        # Module 4: Bandwidth-Aware Message Selector
        self.bandwidth_selector = BandwidthMessageSelector(bytes_per_token=cfg.bytes_per_query)

        # Module 5: Reliability-Weighted Asynchronous Fusion
        self.fusion = ReliabilityWeightedFusion(feat_dim=self.feat_dim, num_heads=4)

    def forward(
        self,
        batch: Dict[str, Any],
        delay_override_ms: Optional[float] = None,
        packet_loss_override: Optional[float] = None,
        pose_noise_trans: float = 0.0,
        pose_noise_rot: float = 0.0,
        bandwidth_budget_kb: int = 10,
        extrapolation_mode: Optional[str] = None,
        fusion_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Pure neural inference pipeline.
        Inputs: Sensor point-cloud BEV tensors and SE(3) poses only.
        """
        device = next(self.encoder.parameters()).device
        extrap_mode = extrapolation_mode or self.cfg.extrapolation_mode
        fuse_mode = fusion_type or "attention"

        # -------------------------------------------------------------
        # Module 1: Ego Perception Encoding & Proposal Generation
        # -------------------------------------------------------------
        ego_bev_raw = batch["ego_bev"].to(device)
        if ego_bev_raw.dim() == 3:
            ego_bev_raw = ego_bev_raw.unsqueeze(0)  # (1, C, H, W)

        ego_bev_feat = self.encoder(ego_bev_raw)
        ego_queries, ego_boxes, ego_scores = self.query_gen(ego_bev_feat)

        # -------------------------------------------------------------
        # Module 2: Collaborator Selection
        # -------------------------------------------------------------
        collabs_raw = batch.get("collabs", [])
        if not collabs_raw or fuse_mode == "ego_only":
            fuse_res = self.fusion(
                ego_queries=ego_queries,
                ego_boxes=ego_boxes,
                collab_feats=torch.empty((1, 0, self.feat_dim), device=device),
                collab_boxes=torch.empty((1, 0, 7), device=device),
                delay_sec=torch.empty((1, 0, 1), device=device),
                collab_sigmas=torch.empty((1, 0, 1), device=device),
                packet_mask=torch.empty((1, 0, 1), device=device),
                pose_noise_mag=torch.empty((1, 0, 1), device=device),
                fusion_type=fuse_mode
            )
            return {
                "pred_logits": fuse_res["pred_logits"],
                "pred_boxes": fuse_res["pred_boxes"],
                "transmitted_kb": 0.0,
                "selected_collabs": [],
                "extrapolated_boxes": torch.empty((1, 0, 7), device=device),
                "log_var": None
            }

        selected_indices, collab_scores = self.collab_selector(collabs_raw)
        selected_collabs = [collabs_raw[i] for i in selected_indices]

        # -------------------------------------------------------------
        # Modules 3 & 4: Collaborator Processing, Extrapolation & Budget
        # -------------------------------------------------------------
        gathered_collab_feats = []
        gathered_collab_boxes = []
        gathered_collab_log_vars = []
        gathered_delays = []
        gathered_sigmas = []
        gathered_packet_masks = []
        gathered_pose_mags = []
        total_comm_kb = 0.0

        p_loss = packet_loss_override if packet_loss_override is not None else self.cfg.default_packet_loss

        for c_info in selected_collabs:
            if delay_override_ms is not None:
                delay_ms = delay_override_ms
            else:
                delay_ms = float(c_info.get("delay_ms", 0.0))
            delay_sec_val = delay_ms / 1000.0

            # Encode collaborator local BEV using shared encoder
            c_bev_np = c_info["bev_features"]
            c_bev_tensor = torch.from_numpy(c_bev_np).unsqueeze(0).to(device)
            c_bev_feat = self.encoder(c_bev_tensor)
            # Generate proposals in collaborator frame purely from sensor features
            c_queries, c_local_boxes, c_scores = self.query_gen(c_bev_feat)

            # Spatial Alignment: Transform collaborator queries into ego frame
            T_ego_collab = c_info["T_ego_collab"].copy()

            # Apply GPS/IMU Pose Noise perturbation if specified
            pose_noise_mag_val = 0.0
            if pose_noise_trans > 0.0 or pose_noise_rot > 0.0:
                t_noise = np.random.normal(0.0, pose_noise_trans, size=(3,))
                r_noise = np.radians(np.random.normal(0.0, pose_noise_rot))
                pose_noise_mag_val = float(np.linalg.norm(t_noise) + abs(r_noise))
                T_ego_collab[0, 3] += t_noise[0]
                T_ego_collab[1, 3] += t_noise[1]
                c_yaw, s_yaw = math.cos(r_noise), math.sin(r_noise)
                R_noise_2d = np.array([[c_yaw, -s_yaw], [s_yaw, c_yaw]])
                T_ego_collab[:2, :2] = R_noise_2d @ T_ego_collab[:2, :2]

            # Transform 3D candidate centers into ego frame
            pts_c = torch.cat([c_local_boxes[:, :, :3], torch.ones_like(c_local_boxes[:, :, :1])], dim=-1)
            T_tensor = torch.from_numpy(T_ego_collab).float().to(device)
            pts_ego = torch.matmul(pts_c, T_tensor.T)[:, :, :3]

            collab_yaw_offset = math.atan2(T_ego_collab[1, 0], T_ego_collab[0, 0])
            c_yaw_in_ego = c_local_boxes[:, :, 6:7] + collab_yaw_offset
            c_boxes_in_ego = torch.cat([
                pts_ego,
                c_local_boxes[:, :, 3:6],
                c_yaw_in_ego
            ], dim=-1)

            # Module 3: Uncertainty-Aware Motion Extrapolation (Learned from features)
            delay_sec_tensor = torch.tensor([[[delay_sec_val]]], device=device)
            extrap_dict = self.motion_extrapolator(
                query_feats=c_queries,
                delayed_boxes=c_boxes_in_ego,
                delay_sec=delay_sec_tensor,
                mode=extrap_mode,
                initial_vel=None  # Learned velocity from features, zero GT leakage
            )
            extrap_boxes = extrap_dict["extrapolated_boxes"]
            extrap_feats = extrap_dict["extrapolated_feats"]
            uncertainty_sigma = extrap_dict["uncertainty_sigma"]

            # Module 4: Bandwidth Selection & Packet Loss
            bw_res = self.bandwidth_selector(
                query_feats=extrap_feats,
                query_boxes=extrap_boxes,
                scores=c_scores,
                uncertainty_sigma=uncertainty_sigma,
                budget_kb=bandwidth_budget_kb,
                packet_loss_rate=p_loss,
                log_vars=extrap_dict.get("log_var")
            )

            gathered_collab_feats.append(bw_res["selected_feats"])
            gathered_collab_boxes.append(bw_res["selected_boxes"])
            if bw_res.get("selected_log_vars") is not None:
                gathered_collab_log_vars.append(bw_res["selected_log_vars"])
            num_sel = bw_res["num_tokens"]

            gathered_delays.append(torch.full((1, num_sel, 1), delay_sec_val, device=device))
            gathered_sigmas.append(bw_res["selected_sigmas"].unsqueeze(-1))
            gathered_packet_masks.append(bw_res["loss_mask"].unsqueeze(-1))
            gathered_pose_mags.append(torch.full((1, num_sel, 1), pose_noise_mag_val, device=device))
            total_comm_kb += bw_res["transmitted_kb"]

        if gathered_collab_feats:
            collab_feats_cat = torch.cat(gathered_collab_feats, dim=1)
            collab_boxes_cat = torch.cat(gathered_collab_boxes, dim=1)
            delays_cat = torch.cat(gathered_delays, dim=1)
            sigmas_cat = torch.cat(gathered_sigmas, dim=1)
            masks_cat = torch.cat(gathered_packet_masks, dim=1)
            poses_cat = torch.cat(gathered_pose_mags, dim=1)
            log_vars_cat = torch.cat(gathered_collab_log_vars, dim=1) if gathered_collab_log_vars else None
        else:
            collab_feats_cat = torch.empty((1, 0, self.feat_dim), device=device)
            collab_boxes_cat = torch.empty((1, 0, 7), device=device)
            delays_cat = torch.empty((1, 0, 1), device=device)
            sigmas_cat = torch.empty((1, 0, 1), device=device)
            masks_cat = torch.empty((1, 0, 1), device=device)
            poses_cat = torch.empty((1, 0, 1), device=device)
            log_vars_cat = None

        # -------------------------------------------------------------
        # Module 5: Reliability-Weighted Asynchronous Fusion
        # -------------------------------------------------------------
        fuse_res = self.fusion(
            ego_queries=ego_queries,
            ego_boxes=ego_boxes,
            collab_feats=collab_feats_cat,
            collab_boxes=collab_boxes_cat,
            delay_sec=delays_cat,
            collab_sigmas=sigmas_cat,
            packet_mask=masks_cat,
            pose_noise_mag=poses_cat,
            fusion_type=fuse_mode
        )

        return {
            "pred_logits": fuse_res["pred_logits"],
            "pred_boxes": fuse_res["pred_boxes"],
            "reliability_weights": fuse_res.get("reliability_weights", None),
            "transmitted_kb": total_comm_kb,
            "selected_collabs": selected_indices,
            "extrapolated_boxes": collab_boxes_cat,
            "log_var": log_vars_cat
        }
