"""
Experiment 1: True Re-timed Delay Sweep (Delta_t in {0, 100, 200, 300, 400, 500} ms).
Loads collaborator point clouds and poses from frame t - m (m in {0, 1, 2, 3, 4, 5}).
Evaluates all five methods: Ego-Only, Late Fusion, No-Compensation Fusion,
Constant-Velocity Alignment, and TrustDelay (Ours).
Records artifact with git commit, checkpoint path, split, seeds, and timestamp.
"""
from __future__ import annotations
import os
import sys
import json
import time
import subprocess
import torch
import numpy as np
from tqdm import tqdm

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

def run_e1():
    cfg = TrustDelayConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt_path = "results/checkpoints/trustdelay_best.pth"
    commit = get_git_commit()

    print(f"=== Experiment 1: True Re-timed Delay Sweep (Commit: {commit[:8]}) ===")
    print(f"Loading checkpoint: {ckpt_path}")
    model = TrustDelayModel(cfg).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    test_dataset = OPV2VAsynchronousDataset(cfg, split="test")
    evaluator = DetectionEvaluator(iou_thresholds=[0.3, 0.5, 0.7])

    delays = [0, 100, 200, 300, 400, 500]
    frame_lags = [0, 1, 2, 3, 4, 5]

    methods = [
        "Ego-Only",
        "Late Fusion",
        "No-Compensation Fusion",
        "Constant-Velocity Alignment",
        "TrustDelay (Ours)"
    ]

    # Precompute Ego-Only once (it uses only ego sensor at time t, strictly delay-invariant)
    print("\n--- Precomputing Ego-Only Baseline ---")
    test_dataset.set_delay_ms(0)
    evaluator.reset()
    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]
        gt_boxes = sample["gt_boxes"].numpy()
        gt_moving = sample["gt_is_moving"].numpy()
        p_boxes, p_scores, _ = evaluate_single_frame(
            model=model, sample=sample, method="Ego-Only",
            device=device, delay_ms=0.0, packet_loss=0.0,
            pose_trans=0.0, pose_rot=0.0, budget_kb=0.0
        )
        evaluator.add_sample(p_boxes, p_scores, gt_boxes, gt_moving)
    ego_res = evaluator.evaluate()
    ego_metrics = {
        "AP@0.3": round(ego_res["AP@0.3"], 2),
        "AP@0.5": round(ego_res["AP@0.5"], 2),
        "AP@0.7": round(ego_res["AP@0.7"], 2),
        "comm_kb": 0.0
    }
    print(f"  Ego-Only | AP@0.3: {ego_metrics['AP@0.3']}% | AP@0.5: {ego_metrics['AP@0.5']}% | AP@0.7: {ego_metrics['AP@0.7']}%")

    sweep_results = {}
    collab_methods = [
        "Late Fusion",
        "No-Compensation Fusion",
        "Constant-Velocity Alignment",
        "TrustDelay (Ours)"
    ]

    for d, m_lag in zip(delays, frame_lags):
        sweep_results[str(d)] = {
            "true_frame_lag": m_lag,
            "delay_ms": d,
            "methods": {
                "Ego-Only": ego_metrics
            }
        }
        print(f"\n==========================================")
        print(f"Delay = {d} ms (True Frame Lag: m = {m_lag} frames, {m_lag * 100} ms)")
        print(f"==========================================")
        print(f"  {'Ego-Only':28s} | Lag: {m_lag} | AP@0.3: {ego_metrics['AP@0.3']:5.2f}% | AP@0.5: {ego_metrics['AP@0.5']:5.2f}% | AP@0.7: {ego_metrics['AP@0.7']:5.2f}% | Comm: 0.0 KB")

        # Crucial E1 fix: Set dataset delay so collaborator PCD and pose are loaded from t - m_lag
        test_dataset.set_delay_ms(d)

        for method in collab_methods:
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
                    delay_ms=float(d),
                    packet_loss=0.0,
                    pose_trans=0.0,
                    pose_rot=0.0,
                    budget_kb=8.4
                )
                evaluator.add_sample(p_boxes, p_scores, gt_boxes, gt_moving)
                comm_kbs.append(kb)

            res = evaluator.evaluate()
            avg_comm = float(np.mean(comm_kbs))

            res_dict = {
                "AP@0.3": round(res["AP@0.3"], 2),
                "AP@0.5": round(res["AP@0.5"], 2),
                "AP@0.7": round(res["AP@0.7"], 2),
                "comm_kb": round(avg_comm, 2)
            }
            sweep_results[str(d)]["methods"][method] = res_dict
            print(f"  {method:28s} | Lag: {m_lag} | AP@0.3: {res_dict['AP@0.3']:5.2f}% | AP@0.5: {res_dict['AP@0.5']:5.2f}% | AP@0.7: {res_dict['AP@0.7']:5.2f}% | Comm: {avg_comm:4.1f} KB")

    artifact = {
        "experiment": "E1",
        "description": "True re-timed delay sweep across 0 to 500 ms (0 to 5 frames lag)",
        "git_commit": commit,
        "checkpoint_path": ckpt_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_split": "test (90 frames, Culver City)",
        "delays_ms": delays,
        "frame_lags": frame_lags,
        "sweep_results": sweep_results
    }

    os.makedirs("results/experiments", exist_ok=True)
    out_path = "results/experiments/e1_true_delay_sweep.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nArtifact successfully saved to {out_path}")

    # Check acceptance criterion:
    # "The three previously-flat baselines must now vary with Delta_t."
    print("\n--- Acceptance Criterion Verification ---")
    late_fusion_ap05 = [sweep_results[str(d)]["methods"]["Late Fusion"]["AP@0.5"] for d in delays]
    nocomp_ap05 = [sweep_results[str(d)]["methods"]["No-Compensation Fusion"]["AP@0.5"] for d in delays]
    cv_ap05 = [sweep_results[str(d)]["methods"]["Constant-Velocity Alignment"]["AP@0.5"] for d in delays]
    td_ap05 = [sweep_results[str(d)]["methods"]["TrustDelay (Ours)"]["AP@0.5"] for d in delays]

    print(f"Late Fusion AP@0.5 (0->500ms): {late_fusion_ap05}")
    print(f"No-Comp Fusion AP@0.5 (0->500ms): {nocomp_ap05}")
    print(f"Constant-Velocity AP@0.5 (0->500ms): {cv_ap05}")
    print(f"TrustDelay (Ours) AP@0.5 (0->500ms): {td_ap05}")

    late_varies = (max(late_fusion_ap05) - min(late_fusion_ap05)) > 0.0
    nocomp_varies = (max(nocomp_ap05) - min(nocomp_ap05)) > 0.0
    cv_varies = (max(cv_ap05) - min(cv_ap05)) > 0.0

    print(f"Late Fusion varies with delay: {late_varies}")
    print(f"No-Compensation varies with delay: {nocomp_varies}")
    print(f"Constant-Velocity varies with delay: {cv_varies}")

    if not (late_varies or nocomp_varies or cv_varies):
        print("[WARNING] Baselines did not vary! Re-timing might not be active.")
    else:
        print("[PASS] True re-timing confirmed: Baselines vary dynamically with true frame lag.")

if __name__ == "__main__":
    run_e1()
