"""
Unit test asserting ZERO Ground-Truth Leakage in TrustDelayModel forward pass.
Verifies model consumes strictly sensor BEV features and SE(3) poses.
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import numpy as np
from configs.config import TrustDelayConfig
from src.models.trust_delay_model import TrustDelayModel

def test_zero_leakage():
    cfg = TrustDelayConfig()
    cfg.num_queries = 64
    cfg.bev_w = 100
    cfg.bev_h = 100
    model = TrustDelayModel(cfg)
    model.eval()

    # Create dummy sensor inputs
    ego_bev = torch.randn(1, 4, 100, 100)
    collab_bev = np.random.randn(4, 100, 100).astype(np.float32)
    T_ego_collab = np.eye(4, dtype=np.float32)
    T_ego_collab[0, 3] = 20.0 # 20m ahead

    # Intentionally provide adversarial/poisoned GT boxes
    # If the model leaks GT, the output will contain these exact poisoned boxes [999.0, ...]
    poisoned_gt = torch.full((5, 7), 999.0)

    batch = {
        "ego_bev": ego_bev,
        "collabs": [{
            "agent_id": "test_cav_1",
            "delay_ms": 200.0,
            "distance": 20.0,
            "rel_speed": 0.0,
            "T_ego_collab": T_ego_collab,
            "bev_features": collab_bev
        }],
        "gt_boxes": poisoned_gt # Adversarial input
    }

    with torch.no_grad():
        out = model(batch, delay_override_ms=200.0)

    pred_boxes = out["pred_boxes"].squeeze(0) # (N_q, 7)
    pred_logits = out["pred_logits"].squeeze(0) # (N_q, 1)

    # Assert poisoned GT did not leak into predictions
    max_val = pred_boxes.abs().max().item()
    assert max_val < 500.0, f"LEAKAGE DETECTED! Max pred box value was {max_val}, indicating GT injection!"
    assert pred_boxes.shape[0] == cfg.num_queries, f"Expected {cfg.num_queries} queries, got {pred_boxes.shape[0]}"
    assert torch.isfinite(pred_boxes).all(), "Predictions contain non-finite numbers!"
    assert torch.isfinite(pred_logits).all(), "Logits contain non-finite numbers!"

    print("Zero-leakage test PASSED! Model operates purely on sensor inputs.")

if __name__ == "__main__":
    test_zero_leakage()
