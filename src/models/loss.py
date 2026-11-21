"""
Loss Functions for TrustDelay.
Includes:
  1. 3D Detection Focal Classification Loss
  2. Bounding Box Smooth L1 Regression Loss
  3. Heteroscedastic Negative Log-Likelihood (NLL) Motion Uncertainty Loss
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class TrustDelayLoss(nn.Module):
    def __init__(self, cls_weight: float = 2.0, reg_weight: float = 1.5, unc_weight: float = 0.5):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.unc_weight = unc_weight

    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        gt_boxes: torch.Tensor,
        extrap_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates composite loss for joint end-to-end training.
        """
        device = pred_dict["pred_logits"].device
        logits = pred_dict["pred_logits"]    # (B, N)
        boxes = pred_dict["pred_boxes"]      # (B, N, 7)
        B, N = logits.shape

        if gt_boxes.shape[0] == 0:
            # Negative frame
            cls_loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
            zero = torch.tensor(0.0, device=device)
            return {
                "loss_total": self.cls_weight * cls_loss,
                "loss_cls": cls_loss,
                "loss_reg": zero,
                "loss_unc": zero
            }

        # Bipartite matching / Nearest-Neighbor target assignment in metric BEV
        # gt_boxes: (M, 7)
        pred_centers = boxes[0, :, :2]  # (N, 2)
        gt_centers = gt_boxes[:, :2]    # (M, 2)

        # Distances: (N, M)
        dists = torch.cdist(pred_centers, gt_centers)
        min_dists, matched_gt_idx = torch.min(dists, dim=1)

        # Match threshold: 3.0 meters
        pos_mask = min_dists < 3.0
        targets_cls = pos_mask.float()

        # Classification Focal Loss
        p = torch.sigmoid(logits[0]).detach()
        alpha, gamma = 0.25, 2.0
        focal_weight = alpha * (1 - p)**gamma * targets_cls + (1 - alpha) * p**gamma * (1 - targets_cls)
        cls_loss = F.binary_cross_entropy_with_logits(logits[0], targets_cls, weight=focal_weight, reduction="mean")

        # Regression Loss on Positives
        if pos_mask.sum() > 0:
            pos_boxes = boxes[0, pos_mask]
            matched_gts = gt_boxes[matched_gt_idx[pos_mask]]
            reg_loss = F.smooth_l1_loss(pos_boxes, matched_gts, beta=1.0)
        else:
            reg_loss = torch.tensor(0.0, device=device)

        # Heteroscedastic Motion Extrapolation Uncertainty Loss
        unc_loss = torch.tensor(0.0, device=device)
        if extrap_dict is not None and "log_var" in extrap_dict and pos_mask.sum() > 0:
            log_var = extrap_dict["log_var"][0, pos_mask]  # (K, 2)
            # NLL: 0.5 * exp(-log_var) * diff^2 + 0.5 * log_var
            pos_diff = pos_boxes[:, :2] - matched_gts[:, :2]
            sq_err = torch.sum(pos_diff ** 2, dim=-1, keepdim=True)
            var = torch.exp(log_var[:, :1])
            unc_loss = torch.mean(0.5 * sq_err / torch.clamp(var, min=1e-3) + 0.5 * log_var[:, :1])

        total_loss = (
            self.cls_weight * cls_loss +
            self.reg_weight * reg_loss +
            self.unc_weight * unc_loss
        )

        return {
            "loss_total": total_loss,
            "loss_cls": cls_loss,
            "loss_reg": reg_loss,
            "loss_unc": unc_loss
        }
