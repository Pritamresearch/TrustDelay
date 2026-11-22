"""
Module 4: Bandwidth-Aware Message Selector.
Selects and compresses high-value queries/tokens under strict communication bandwidth budgets
(1 KB, 5 KB, 10 KB, 20 KB), with support for lossy channel simulation (packet loss).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

class BandwidthMessageSelector(nn.Module):
    """
    Enforces communication byte budgets by pruning low-utility tokens and applying quantization.
    """
    def __init__(self, bytes_per_token: int = 64):
        super().__init__()
        self.bytes_per_token = bytes_per_token

    def compute_max_tokens_for_budget(self, budget_kb: int) -> int:
        """Calculate token allocation given budget in kilobytes."""
        total_bytes = budget_kb * 1024
        return max(4, total_bytes // self.bytes_per_token)

    def forward(
        self,
        query_feats: torch.Tensor,       # (B, N, C)
        query_boxes: torch.Tensor,       # (B, N, 7)
        scores: torch.Tensor,            # (B, N)
        uncertainty_sigma: torch.Tensor, # (B, N)
        budget_kb: int = 10,
        packet_loss_rate: float = 0.0,
        log_vars: Optional[torch.Tensor] = None, # (B, N, 2)
        rank_by_score_only: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Selects top tokens within budget and applies packet loss channel masking.
        Returns selected tokens, boxes, transmission volume, and packet mask.
        If rank_by_score_only is True (e.g. E8 sigma-free arm), ranks purely by score c_j without sigma_j.
        """
        B, N, C = query_feats.shape
        max_tokens = max(1, min(N, int(self.compute_max_tokens_for_budget(budget_kb))))

        # Information utility score: high object confidence, low motion uncertainty
        # Tokens with high confidence and low uncertainty are prioritized
        if rank_by_score_only:
            utility_score = scores.clone()
        else:
            utility_score = scores * torch.exp(-0.5 * uncertainty_sigma)  # (B, N)

        # Select Top-K tokens according to budget
        top_utility, top_indices = torch.topk(utility_score, k=max_tokens, dim=1)

        idx_feat = top_indices.unsqueeze(-1).expand(-1, -1, C)
        idx_box = top_indices.unsqueeze(-1).expand(-1, -1, 7)

        selected_feats = torch.gather(query_feats, 1, idx_feat)
        selected_boxes = torch.gather(query_boxes, 1, idx_box)
        selected_scores = torch.gather(scores, 1, top_indices)
        selected_sigmas = torch.gather(uncertainty_sigma, 1, top_indices)

        selected_log_vars = None
        if log_vars is not None:
            idx_lv = top_indices.unsqueeze(-1).expand(-1, -1, log_vars.shape[-1])
            selected_log_vars = torch.gather(log_vars, 1, idx_lv)

        # Channel Simulation: Packet Loss
        # If packets are lost over the wireless channel, dropped tokens become zero/masked
        loss_mask = torch.ones((B, max_tokens), device=query_feats.device)
        if packet_loss_rate > 0.0:
            rand_vals = torch.rand((B, max_tokens), device=query_feats.device)
            loss_mask = (rand_vals >= packet_loss_rate).float()

            selected_feats = selected_feats * loss_mask.unsqueeze(-1)
            selected_scores = selected_scores * loss_mask

        # Transmission metrics
        transmitted_bytes = max_tokens * self.bytes_per_token
        transmitted_kb = transmitted_bytes / 1024.0

        return {
            "selected_feats": selected_feats,
            "selected_boxes": selected_boxes,
            "selected_scores": selected_scores,
            "selected_sigmas": selected_sigmas,
            "selected_log_vars": selected_log_vars,
            "loss_mask": loss_mask,
            "transmitted_kb": transmitted_kb,
            "num_tokens": max_tokens
        }
