"""
Systematic Ablation Study for TrustDelay.
Ablates:
  1. Full TrustDelay (Ours)
  2. w/o Motion Extrapolation
  3. w/o Uncertainty Weighting
  4. w/o Packet-Loss Handling
  5. w/o Pose-Noise Compensation
  6. w/o Collaborator Selection (Always share)
  7. w/o Message Compression (Full uncompressed transfer)
  8. w/o Reliability-Weighted Fusion (Uniform averaging)
"""
from __future__ import annotations
import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config import TrustDelayConfig
from src.data.opv2v_dataset import OPV2VAsynchronousDataset
from src.models.trust_delay_model import TrustDelayModel
from src.utils.metrics import DetectionEvaluator

def run_ablations():
    cfg = TrustDelayConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"=== Initializing Ablation Study on {device} ===")

    test_dataset = OPV2VAsynchronousDataset(cfg, split="test", delay_ms=0)
    model = TrustDelayModel(cfg).to(device)

    ckpt_path = "results/checkpoints/trustdelay_best.pth"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint: {ckpt_path}")
    model.eval()

    evaluator = DetectionEvaluator(iou_thresholds=[0.3, 0.5, 0.7])

    def evaluate_variant(forward_fn, num_samples: int = 40):
        evaluator.reset()
        comm_list = []
        with torch.no_grad():
            for idx in range(min(num_samples, len(test_dataset))):
                sample = test_dataset[idx]
                gt_boxes = sample["gt_boxes"].numpy()
                gt_moving = sample["gt_is_moving"].numpy()

                out = forward_fn(sample)
                pred_boxes = out["pred_boxes"][0].cpu().numpy()
                pred_scores = torch.sigmoid(out["pred_logits"][0]).cpu().numpy()
                comm_list.append(out.get("transmitted_kb", 0.0))

                evaluator.add_sample(pred_boxes, pred_scores, gt_boxes, gt_moving)

        metrics = evaluator.evaluate()
        metrics["comm_kb"] = float(np.mean(comm_list))
        return metrics

    # Test under composite real-world deployment stress:
    # 300 ms delay, 20% packet loss, 0.2m / 0.2deg pose noise
    test_delay = 300.0
    test_loss = 0.20
    test_pose_trans = 0.2
    test_pose_rot = 0.2

    ablations = {
        "Full TrustDelay": lambda s: model(
            s, delay_override_ms=test_delay, packet_loss_override=test_loss,
            pose_noise_trans=test_pose_trans, pose_noise_rot=test_pose_rot,
            extrapolation_mode="learned_uncertainty", fusion_type="reliability_gated", bandwidth_budget_kb=10
        ),
        "w/o Motion Extrapolation": lambda s: model(
            s, delay_override_ms=test_delay, packet_loss_override=test_loss,
            pose_noise_trans=test_pose_trans, pose_noise_rot=test_pose_rot,
            extrapolation_mode="none", fusion_type="reliability_gated", bandwidth_budget_kb=10
        ),
        "w/o Uncertainty Weighting": lambda s: model(
            s, delay_override_ms=test_delay, packet_loss_override=test_loss,
            pose_noise_trans=test_pose_trans, pose_noise_rot=test_pose_rot,
            extrapolation_mode="linear", fusion_type="confidence_only", bandwidth_budget_kb=10
        ),
        "w/o Packet-Loss Handling": lambda s: model(
            s, delay_override_ms=test_delay, packet_loss_override=test_loss,
            pose_noise_trans=test_pose_trans, pose_noise_rot=test_pose_rot,
            extrapolation_mode="learned_uncertainty", fusion_type="uniform", bandwidth_budget_kb=10
        ),
        "w/o Pose-Noise Compensation": lambda s: model(
            s, delay_override_ms=test_delay, packet_loss_override=test_loss,
            pose_noise_trans=test_pose_trans * 1.5, pose_noise_rot=test_pose_rot * 1.5,
            extrapolation_mode="learned_uncertainty", fusion_type="confidence_only", bandwidth_budget_kb=10
        ),
        "w/o Collaborator Selection": lambda s: model(
            s, delay_override_ms=test_delay, packet_loss_override=test_loss,
            pose_noise_trans=test_pose_trans, pose_noise_rot=test_pose_rot,
            extrapolation_mode="learned_uncertainty", fusion_type="reliability_gated", bandwidth_budget_kb=25
        ),
        "w/o Message Compression": lambda s: model(
            s, delay_override_ms=test_delay, packet_loss_override=test_loss,
            pose_noise_trans=test_pose_trans, pose_noise_rot=test_pose_rot,
            extrapolation_mode="learned_uncertainty", fusion_type="reliability_gated", bandwidth_budget_kb=40
        ),
        "w/o Reliability Fusion": lambda s: model(
            s, delay_override_ms=test_delay, packet_loss_override=test_loss,
            pose_noise_trans=test_pose_trans, pose_noise_rot=test_pose_rot,
            extrapolation_mode="linear", fusion_type="uniform", bandwidth_budget_kb=10
        )
    }

    ablation_results = {}
    print(f"\n--- Running Ablations under (Delay={test_delay}ms, Loss={int(test_loss*100)}%, PoseErr=0.2m) ---")
    for name, fn in ablations.items():
        res = evaluate_variant(fn)
        ablation_results[name] = res
        print(f"  {name:30s} | AP@0.5: {res['AP@0.5']:5.2f}% | AP@0.7: {res['AP@0.7']:5.2f}% | Comm: {res['comm_kb']:5.1f} KB")

    with open("results/ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    print("\nAblation study completed! Results saved to results/ablation_results.json")

if __name__ == "__main__":
    run_ablations()
