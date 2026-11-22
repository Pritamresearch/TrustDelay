"""
Experiment 4: Uniform Distance-Based Non-Maximum Suppression (NMS) Evaluation.
Evaluates Table 1 (true re-timed delay sweep 0 to 500 ms) across all 5 methods:
  1. Without NMS (Raw query predictions)
  2. With Uniform NMS (2.5 m center distance threshold applied uniformly to all arms)

Baselines evaluated:
  - Ego-Only
  - Late Fusion
  - No-Compensation Fusion
  - Constant-Velocity Alignment
  - TrustDelay (Ours)

Saves artifact to results/experiments/e4_nms_evaluation.json.
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

def run_e4():
    cfg = TrustDelayConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt_path = "results/checkpoints/trustdelay_best.pth"
    commit = get_git_commit()

    print(f"=== Experiment 4: Uniform Non-Maximum Suppression Evaluation (Commit: {commit[:8]}) ===")
    print(f"Loading checkpoint: {ckpt_path}")
    model = TrustDelayModel(cfg).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    test_dataset = OPV2VAsynchronousDataset(cfg, split="test")
    evaluator = DetectionEvaluator(iou_thresholds=[0.3, 0.5, 0.7])

    delays = [0, 100, 200, 300, 400, 500]
    frame_lags = [0, 1, 2, 3, 4, 5]

    all_methods = [
        "Ego-Only",
        "Late Fusion",
        "No-Compensation Fusion",
        "Constant-Velocity Alignment",
        "TrustDelay (Ours)"
    ]

    results = {
        "without_nms": {},
        "with_nms": {},
        "delta_nms_minus_raw": {}
    }

    # Evaluate for both modes
    for apply_nms, mode_key in [(False, "without_nms"), (True, "with_nms")]:
        mode_label = "WITH 2.5m UNIFORM NMS" if apply_nms else "WITHOUT NMS (RAW QUERIES)"
        print(f"\n{'='*30} Evaluating Table 1: {mode_label} {'='*30}")

        # 1. Precompute Ego-Only
        test_dataset.set_delay_ms(0)
        evaluator.reset()
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            gt_boxes = sample["gt_boxes"].numpy()
            gt_moving = sample["gt_is_moving"].numpy()
            p_boxes, p_scores, _ = evaluate_single_frame(
                model=model, sample=sample, method="Ego-Only",
                device=device, delay_ms=0.0, apply_nms=apply_nms
            )
            evaluator.add_sample(p_boxes, p_scores, gt_boxes, gt_moving)
        ego_res = evaluator.evaluate()
        ego_metrics = {
            "AP@0.3": round(ego_res["AP@0.3"], 2),
            "AP@0.5": round(ego_res["AP@0.5"], 2),
            "AP@0.7": round(ego_res["AP@0.7"], 2),
            "comm_kb": 0.0
        }
        print(f"Ego-Only | AP@0.3: {ego_metrics['AP@0.3']}% | AP@0.5: {ego_metrics['AP@0.5']}% | AP@0.7: {ego_metrics['AP@0.7']}%")

        table_data = {}
        for d, m_lag in zip(delays, frame_lags):
            print(f"\nDelay = {d} ms (True lag: {m_lag} frames):")
            table_data[str(d)] = {
                "delay_ms": d,
                "true_frame_lag": m_lag,
                "methods": {
                    "Ego-Only": ego_metrics
                }
            }
            test_dataset.set_delay_ms(d)

            for method in all_methods:
                if method == "Ego-Only":
                    continue
                evaluator.reset()
                comm_kbs = []
                for idx in range(len(test_dataset)):
                    sample = test_dataset[idx]
                    gt_boxes = sample["gt_boxes"].numpy()
                    gt_moving = sample["gt_is_moving"].numpy()
                    p_boxes, p_scores, c_kb = evaluate_single_frame(
                        model=model, sample=sample, method=method,
                        device=device, delay_ms=float(d), apply_nms=apply_nms
                    )
                    comm_kbs.append(c_kb)
                    evaluator.add_sample(p_boxes, p_scores, gt_boxes, gt_moving)

                res = evaluator.evaluate()
                avg_kb = float(np.mean(comm_kbs))
                m_dict = {
                    "AP@0.3": round(res["AP@0.3"], 2),
                    "AP@0.5": round(res["AP@0.5"], 2),
                    "AP@0.7": round(res["AP@0.7"], 2),
                    "comm_kb": round(avg_kb, 1)
                }
                table_data[str(d)]["methods"][method] = m_dict
                print(f"  {method:28s} | AP@0.3: {m_dict['AP@0.3']:5.2f}% | AP@0.5: {m_dict['AP@0.5']:5.2f}% | AP@0.7: {m_dict['AP@0.7']:4.2f}% | Comm: {avg_kb:4.1f} KB")

        results[mode_key] = table_data

    # Compute deltas (with_nms - without_nms)
    print(f"\n{'='*30} Comparison: Impact of Uniform NMS (Delta: With NMS - Without NMS) {'='*30}")
    delta_data = {}
    for d in [str(x) for x in delays]:
        delta_data[d] = {}
        print(f"\nDelay = {d} ms:")
        for m in all_methods:
            raw_03 = results["without_nms"][d]["methods"][m]["AP@0.3"]
            raw_05 = results["without_nms"][d]["methods"][m]["AP@0.5"]
            raw_07 = results["without_nms"][d]["methods"][m]["AP@0.7"]

            nms_03 = results["with_nms"][d]["methods"][m]["AP@0.3"]
            nms_05 = results["with_nms"][d]["methods"][m]["AP@0.5"]
            nms_07 = results["with_nms"][d]["methods"][m]["AP@0.7"]

            d_03 = round(nms_03 - raw_03, 2)
            d_05 = round(nms_05 - raw_05, 2)
            d_07 = round(nms_07 - raw_07, 2)

            delta_data[d][m] = {
                "delta_AP@0.3": d_03,
                "delta_AP@0.5": d_05,
                "delta_AP@0.7": d_07
            }
            print(f"  {m:28s} | dAP@0.3: {d_03:+5.2f}% | dAP@0.5: {d_05:+5.2f}% | dAP@0.7: {d_07:+5.2f}%")

    results["delta_nms_minus_raw"] = delta_data

    artifact = {
        "experiment": "E4",
        "description": "Uniform distance-based NMS (2.5m) evaluation across all arms on Table 1 delay sweep",
        "git_commit": commit,
        "checkpoint_path": ckpt_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_split": "test (90 frames, Culver City)",
        "nms_threshold_m": 2.5,
        "table1_without_nms": results["without_nms"],
        "table1_with_uniform_nms": results["with_nms"],
        "table1_delta": delta_data
    }

    out_path = "results/experiments/e4_nms_evaluation.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nArtifact saved successfully to: {out_path}")

if __name__ == "__main__":
    run_e4()
