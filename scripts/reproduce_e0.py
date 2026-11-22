"""
Experiment 0: Reproduction of Baseline Metrics at Delta_t = 0 ms.
Verifies released checkpoint against target table before modifying code.
"""
from __future__ import annotations
import os
import sys
import json
import time
import subprocess
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config import TrustDelayConfig
from src.data.opv2v_dataset import OPV2VAsynchronousDataset
from src.models.trust_delay_model import TrustDelayModel
from src.utils.metrics import DetectionEvaluator
from scripts.evaluate import evaluate_single_frame

def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"

def run_e0():
    cfg = TrustDelayConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt_path = "results/checkpoints/trustdelay_best.pth"
    commit = get_git_commit()

    print(f"=== Experiment 0: Baseline Reproduction (Commit: {commit[:8]}) ===")
    print(f"Loading checkpoint: {ckpt_path}")
    model = TrustDelayModel(cfg).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    test_dataset = OPV2VAsynchronousDataset(cfg, split="test")
    evaluator = DetectionEvaluator(iou_thresholds=[0.3, 0.5, 0.7])

    methods = [
        "Ego-Only",
        "Late Fusion",
        "No-Compensation Fusion",
        "Constant-Velocity Alignment",
        "TrustDelay (Ours)"
    ]

    target = {
        "Ego-Only": {"payload": 0.0, "AP@0.3": 41.75, "AP@0.5": 1.06, "AP@0.7": 0.01},
        "Late Fusion": {"payload": 3.8, "AP@0.3": 40.65, "AP@0.5": 1.03, "AP@0.7": 0.00},
        "No-Compensation Fusion": {"payload": 8.4, "AP@0.3": 39.00, "AP@0.5": 1.64, "AP@0.7": 0.01},
        "Constant-Velocity Alignment": {"payload": 8.4, "AP@0.3": 46.25, "AP@0.5": 4.80, "AP@0.7": 0.11},
        "TrustDelay (Ours)": {"payload": 8.4, "AP@0.3": 51.85, "AP@0.5": 11.13, "AP@0.7": 0.68}
    }

    results = {}
    mismatches = []

    for method in methods:
        evaluator.reset()
        comm_kbs = []
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            gt_boxes = sample["gt_boxes"].numpy()
            gt_moving = sample["gt_is_moving"].numpy()

            p_boxes, p_scores, kb = evaluate_single_frame(
                model=model,
                sample=sample,
                method=method,
                device=device,
                delay_ms=0.0,
                packet_loss=0.0,
                pose_trans=0.0,
                pose_rot=0.0,
                budget_kb=8.4
            )
            evaluator.add_sample(p_boxes, p_scores, gt_boxes, gt_moving)
            comm_kbs.append(kb)

        res = evaluator.evaluate()
        avg_comm = float(np.mean(comm_kbs))
        results[method] = {
            "payload_kb": round(avg_comm, 2),
            "AP@0.3": round(res["AP@0.3"], 2),
            "AP@0.5": round(res["AP@0.5"], 2),
            "AP@0.7": round(res["AP@0.7"], 2)
        }

        # Check against target
        t = target[method]
        diff_03 = abs(results[method]["AP@0.3"] - t["AP@0.3"])
        diff_05 = abs(results[method]["AP@0.5"] - t["AP@0.5"])
        diff_07 = abs(results[method]["AP@0.7"] - t["AP@0.7"])

        print(f"{method:28s} | AP@0.3: {results[method]['AP@0.3']:5.2f} (diff {diff_03:4.2f}) | "
              f"AP@0.5: {results[method]['AP@0.5']:5.2f} (diff {diff_05:4.2f}) | "
              f"AP@0.7: {results[method]['AP@0.7']:5.2f} (diff {diff_07:4.2f}) | "
              f"Payload: {avg_comm:4.1f} KB")

        if max(diff_03, diff_05, diff_07) > 0.1:
            mismatches.append((method, diff_03, diff_05, diff_07))

    artifact = {
        "experiment": "E0",
        "description": "Reproduction of baseline metrics at Delta_t = 0 ms",
        "git_commit": commit,
        "checkpoint_path": ckpt_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split": "test (90 frames, Culver City)",
        "results": results,
        "reproduced": len(mismatches) == 0
    }

    os.makedirs("results/experiments", exist_ok=True)
    out_path = "results/experiments/e0_reproduction.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nArtifact saved to {out_path}")
    if mismatches:
        print(f"[FAILED] E0 Mismatches detected (> 0.1 points): {mismatches}")
        sys.exit(1)
    else:
        print("[SUCCESS] Experiment 0 reproduction PASSED! All numbers match within 0.1 points.")

if __name__ == "__main__":
    run_e0()
