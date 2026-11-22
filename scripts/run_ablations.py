"""
Systematic Ablation Study for TrustDelay.
Evaluates 8 architectural variants on the OPV2V test split under identical perturbations (at 300 ms delay):
  1. Full TrustDelay (Ours)
  2. w/o Motion Extrapolation (extrapolation_mode='none')
  3. w/o Uncertainty Weighting (ignore motion uncertainty in gating)
  4. w/o Packet-Loss Handling (ignore loss masks in gating)
  5. w/o Pose-Noise Compensation (ignore pose noise magnitude in gating)
  6. w/o Collaborator Selection (all collabs transmitted, disable selector)
  7. w/o Message Compression (uncompressed token transmission, budget 100 KB)
  8. w/o Reliability Fusion (equal uniform feature weighting)

Computes mean ± std over 3 fixed seeds and outputs to results/ablation_results.json.
"""
from __future__ import annotations
import os
import sys
import time
import json
import argparse
import subprocess
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config import TrustDelayConfig
from src.data.opv2v_dataset import OPV2VAsynchronousDataset
from src.models.trust_delay_model import TrustDelayModel
from src.utils.metrics import DetectionEvaluator

def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"

def run_ablations():
    parser = argparse.ArgumentParser(description="Run TrustDelay systematic ablation study")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/trustdelay_best.pth")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", type=str, default="results/ablation_results.json")
    args = parser.parse_args()

    cfg = TrustDelayConfig()
    if args.data_root:
        cfg.data_root = args.data_root
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    git_commit = get_git_commit()

    print(f"=== Initializing Honest Ablation Study (Commit: {git_commit[:8]}) ===")
    print(f"Loading checkpoint: {args.checkpoint}")
    model = TrustDelayModel(cfg).to(device)
    if os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print("  --> Model weights successfully loaded.")
    else:
        print(f"  [WARNING] Checkpoint {args.checkpoint} not found. Running with initial weights.")

    model.eval()
    test_dataset = OPV2VAsynchronousDataset(cfg, split="test")
    print(f"Preloading test dataset into RAM cache ({len(test_dataset)} frames)...")
    for i in range(len(test_dataset)):
        _ = test_dataset[i]
    print(f"Preload complete. Cached {len(test_dataset.bev_cache)} BEV feature maps in memory.")
    evaluator = DetectionEvaluator(iou_thresholds=[0.3, 0.5, 0.7])

    variants = [
        "Full TrustDelay (Ours)",
        "w/o Motion Extrapolation",
        "w/o Uncertainty Weighting",
        "w/o Packet-Loss Handling",
        "w/o Pose-Noise Compensation",
        "w/o Collaborator Selection",
        "w/o Message Compression",
        "w/o Reliability Fusion"
    ]

    ablation_records = {
        "git_commit": git_commit,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_path": args.checkpoint,
        "dataset_split": f"test ({len(test_dataset)} frames, Culver City)",
        "delay_ms": 300.0,
        "stress_conditions": "packet_loss=0.10, pose_noise=0.1m/0.1deg",
        "seeds": args.seeds,
        "variants": {}
    }

    delay_ms = 300.0
    p_loss = 0.10
    pose_trans = 0.1
    pose_rot = 0.1

    # Ensure dataset is loaded with true re-timed 300 ms frame lag (m = 3 frames)
    test_dataset.set_delay_ms(int(delay_ms))

    print("\n--- Running 8-Variant Ablation Suite at 300 ms Delay ---")
    for var in variants:
        seed_03 = []
        seed_05 = []
        seed_07 = []
        comm_kbs = []

        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            evaluator.reset()

            for idx in range(len(test_dataset)):
                sample = test_dataset[idx]
                gt_boxes = sample["gt_boxes"].numpy()
                gt_moving = sample["gt_is_moving"].numpy()

                with torch.inference_mode():
                    if var == "Full TrustDelay (Ours)":
                        out = model(sample, delay_override_ms=delay_ms, packet_loss_override=p_loss,
                                    pose_noise_trans=pose_trans, pose_noise_rot=pose_rot, bandwidth_budget_kb=8.4)
                    elif var == "w/o Motion Extrapolation":
                        out = model(sample, delay_override_ms=delay_ms, packet_loss_override=p_loss,
                                    pose_noise_trans=pose_trans, pose_noise_rot=pose_rot, bandwidth_budget_kb=8.4,
                                    extrapolation_mode="none")
                    elif var == "w/o Uncertainty Weighting":
                        out = model(sample, delay_override_ms=delay_ms, packet_loss_override=p_loss,
                                    pose_noise_trans=pose_trans, pose_noise_rot=pose_rot, bandwidth_budget_kb=8.4,
                                    ignore_uncertainty=True)
                    elif var == "w/o Packet-Loss Handling":
                        out = model(sample, delay_override_ms=delay_ms, packet_loss_override=p_loss,
                                    pose_noise_trans=pose_trans, pose_noise_rot=pose_rot, bandwidth_budget_kb=8.4,
                                    ignore_packet_mask=True)
                    elif var == "w/o Pose-Noise Compensation":
                        out = model(sample, delay_override_ms=delay_ms, packet_loss_override=p_loss,
                                    pose_noise_trans=pose_trans, pose_noise_rot=pose_rot, bandwidth_budget_kb=8.4,
                                    ignore_pose_noise=True)
                    elif var == "w/o Collaborator Selection":
                        out = model(sample, delay_override_ms=delay_ms, packet_loss_override=p_loss,
                                    pose_noise_trans=pose_trans, pose_noise_rot=pose_rot, bandwidth_budget_kb=8.4,
                                    disable_collab_selection=True)
                    elif var == "w/o Message Compression":
                        out = model(sample, delay_override_ms=delay_ms, packet_loss_override=p_loss,
                                    pose_noise_trans=pose_trans, pose_noise_rot=pose_rot, bandwidth_budget_kb=100.0)
                    elif var == "w/o Reliability Fusion":
                        out = model(sample, delay_override_ms=delay_ms, packet_loss_override=p_loss,
                                    pose_noise_trans=pose_trans, pose_noise_rot=pose_rot, bandwidth_budget_kb=8.4,
                                    fusion_type="uniform")

                p_boxes = out["pred_boxes"][0].cpu().numpy()
                p_logits = out["pred_logits"][0].cpu().numpy()
                p_scores = 1.0 / (1.0 + np.exp(-np.clip(p_logits, -10.0, 10.0)))
                comm_kbs.append(float(out.get("transmitted_kb", 8.4)))

                evaluator.add_sample(p_boxes, p_scores, gt_boxes, gt_moving)

            res = evaluator.evaluate()
            seed_03.append(res["AP@0.3"])
            seed_05.append(res["AP@0.5"])
            seed_07.append(res["AP@0.7"])

        m_03, s_03 = float(np.mean(seed_03)), float(np.std(seed_03))
        m_05, s_05 = float(np.mean(seed_05)), float(np.std(seed_05))
        m_07, s_07 = float(np.mean(seed_07)), float(np.std(seed_07))
        avg_kb = float(np.mean(comm_kbs))

        ablation_records["variants"][var] = {
            "AP@0.3": round(m_03, 2),
            "AP@0.3_std": round(s_03, 2),
            "AP@0.5": round(m_05, 2),
            "AP@0.5_std": round(s_05, 2),
            "AP@0.7": round(m_07, 2),
            "AP@0.7_std": round(s_07, 2),
            "comm_kb": round(avg_kb, 1)
        }
        print(f"  {var:30s} | AP@0.5: {m_05:5.1f} ± {s_05:3.1f}% | AP@0.7: {m_07:5.1f} ± {s_07:3.1f}% | Comm: {avg_kb:4.1f} KB")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(ablation_records, f, indent=2)
    print(f"\nAblation study results successfully saved to {args.output}")

if __name__ == "__main__":
    run_ablations()
