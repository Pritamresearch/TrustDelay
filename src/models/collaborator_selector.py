"""
Module 2: Collaborator Selector.
Jointly determines whether cooperation is needed and selects top-K optimal collaborators
based on expected perception gain, spatial overlap, distance, relative velocity, and communication cost.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple

class CollaboratorSelector(nn.Module):
    """
    Evaluates surround CAV candidates and selects the most beneficial subset under communication constraints.
    """
    def __init__(
        self,
        top_k: int = 2,
        w_perception: float = 1.0,
        w_distance: float = 0.5,
        w_rel_velocity: float = 0.3,
        w_overlap: float = 0.8,
        w_comm_cost: float = 0.4,
        max_dist: float = 100.0,
        max_rel_vel: float = 25.0
    ):
        super().__init__()
        self.top_k = top_k
        self.w_perception = w_perception
        self.w_distance = w_distance
        self.w_rel_velocity = w_rel_velocity
        self.w_overlap = w_overlap
        self.w_comm_cost = w_comm_cost
        self.max_dist = max_dist
        self.max_rel_vel = max_rel_vel

        # Small scoring MLP to blend geometric and learned utility
        self.score_mlp = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def compute_overlap_ratio(self, T_ego_collab: np.ndarray, ego_fov: float = 50.0) -> float:
        """
        Estimate 2D BEV field-of-view overlap fraction between ego and collaborator.
        """
        c_x, c_y = float(T_ego_collab[0, 3]), float(T_ego_collab[1, 3])
        dist = math.hypot(c_x, c_y)
        if dist >= 2.0 * ego_fov:
            return 0.0
        # Circular overlap area approximation
        r = ego_fov
        d = dist
        overlap_area = 2 * (r**2 * math.acos(d / (2 * r)) - (d / 4) * math.sqrt(max(0.0, 4 * r**2 - d**2)))
        total_area = math.pi * r**2
        return float(np.clip(overlap_area / total_area, 0.0, 1.0))

    def forward(
        self,
        collab_meta_list: List[Dict[str, Any]],
        ego_uncertainty_score: float = 0.5
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Input:
            collab_meta_list: list of dicts for each candidate collaborator containing:
                - distance (float)
                - rel_speed (float)
                - T_ego_collab (4x4 np.ndarray)
                - num_boxes (int or float expected perception gain)
                - comm_cost (float in KB)
        Returns:
            selected_indices: list of chosen top-K collaborator indices
            collaborator_scores: torch.Tensor of shape (N_candidates,)
        """
        if not collab_meta_list:
            return [], torch.empty((0,))

        features = []
        for c in collab_meta_list:
            dist = float(c.get("distance", 30.0))
            rel_v = float(c.get("rel_speed", 0.0))
            T_mat = c.get("T_ego_collab", np.eye(4))
            overlap = self.compute_overlap_ratio(T_mat)
            n_boxes = float(len(c.get("local_boxes", [])))
            # Expected perception gain: more high-value objects in blind spots / complementary FoV
            perc_gain = min(1.0, (n_boxes / 10.0) * (1.0 - overlap * 0.5))
            comm_cost = float(c.get("comm_cost", 10.0)) / 20.0  # normalized

            norm_dist = min(1.0, dist / self.max_dist)
            norm_vel = min(1.0, rel_v / self.max_rel_vel)

            features.append([perc_gain, 1.0 - norm_dist, norm_vel, overlap, comm_cost])

        feat_tensor = torch.tensor(features, dtype=torch.float32)
        device = next(self.score_mlp.parameters()).device
        feat_tensor = feat_tensor.to(device)

        # Compute utility score via combination of physics priors and MLP
        mlp_scores = self.score_mlp(feat_tensor).squeeze(-1)
        
        # Analytic prior score
        prior_scores = (
            self.w_perception * feat_tensor[:, 0]
            + self.w_distance * feat_tensor[:, 1]
            - self.w_rel_velocity * feat_tensor[:, 2]
            + self.w_overlap * feat_tensor[:, 3]
            - self.w_comm_cost * feat_tensor[:, 4]
        )
        total_scores = 0.5 * mlp_scores + 0.5 * torch.sigmoid(prior_scores)

        # Select Top-K
        k = min(self.top_k, len(collab_meta_list))
        top_k_indices = torch.topk(total_scores, k=k).indices.tolist()

        return top_k_indices, total_scores
