"""
Experiment 4: Decompose the Gate (Prior-Only vs MLP-Only vs Delay-Only vs Full Gate).
Evaluates the gate decomposition across all 6 true delay points (m = 0 to 5 frames):
  1. Full Gate (0.5 * learned MLP + 0.5 * exponential prior)
  2. Prior-Only (exponential prior exp(-1.5*dt)*exp(-0.8*sigma)*exp(-0.5*eps), learned MLP dropped)
  3. MLP-Only (learned MLP gate only, exponential prior dropped)
  4. Delay-Only (sigma, delta, epsilon held constant at dataset means, only Delta_t varies)
Saves artifact to results/experiments/e4_gate_decomposition.json.
"""
from __future__ import annotations
import os
import sys
import json
import time
import subprocess
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.metrics import DetectionEvaluator
from scripts.dump_predictions import run_nms_np

CACHE_DIR = "results/predictions_cache"

def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"

def run_e4():
    commit = get_git_commit()
    print(f"=== E4: Decompose the Gate (Commit: {commit[:8]}) ===")

    gt_file = os.path.join(CACHE_DIR, "ground_truth.npz")
    gt_data = np.load(gt_file, allow_pickle=True)
    gt_boxes_all = gt_data["gt_boxes"]
    gt_moving_all = gt_data["gt_is_moving"]
    N_frames = len(gt_boxes_all)

    delays = [0, 100, 200, 300, 400, 500]
    frame_lags = [0, 1, 2, 3, 4, 5]

    gate_variants = [
        ("Full Gate (Ours)", "trustdelay_full", "0.5*MLP + 0.5*exp(-1.5dt)*exp(-0.8sigma)*exp(-0.5eps)"),
        ("Prior-Only Gate", "trustdelay_prior_only", "exp(-1.5dt)*exp(-0.8sigma)*exp(-0.5eps) (drops learned MLP)"),
        ("MLP-Only Gate", "trustdelay_mlp_only", "Learned MLP gate only (drops physics prior)"),
        ("Delay-Only Gate", "trustdelay_delay_only", "Only Delta_t varies (sigma, delta, eps held at nominal constants)"),
        ("Ungated (Uniform)", "trustdelay_no_gate", "Uniform weights w_j = 1.0 (no gating)")
    ]

    evaluator = DetectionEvaluator(iou_thresholds=[0.3, 0.5, 0.7])
    results = {}

    for label, prefix, desc in gate_variants:
        results[label] = {
            "description": desc,
            "delays": {}
        }
        for d in delays:
            fpath = os.path.join(CACHE_DIR, f"{prefix}_lag{d}.npz")
            data = np.load(fpath, allow_pickle=True)
            p_boxes_all = data["pred_boxes"]
            p_scores_all = data["pred_scores"]
            gw_all = data.get("gate_weights", [])

            # Compute mean gate weight w_j across all frames and tokens
            if len(gw_all) > 0 and len(gw_all[0]) > 0:
                mean_w = float(np.mean(np.concatenate(gw_all)))
            elif prefix == "trustdelay_no_gate":
                mean_w = 1.0
            else:
                mean_w = 0.0

            evaluator.reset()
            for i in range(N_frames):
                b = np.asarray(p_boxes_all[i], dtype=np.float32)
                sc = np.asarray(p_scores_all[i], dtype=np.float32)
                v_mask = sc >= 0.05
                if np.any(v_mask):
                    b = b[v_mask]
                    sc = sc[v_mask]
                else:
                    top_k = min(5, len(sc))
                    top_idx = np.argsort(-sc)[:top_k]
                    b = b[top_idx]
                    sc = sc[top_idx]
                b, sc = run_nms_np(b, sc, dist_thresh=2.5)
                evaluator.add_sample(b, sc, gt_boxes_all[i], gt_moving_all[i])
            r = evaluator.evaluate()
            results[label]["delays"][str(d)] = {
                "AP@0.3": round(r["AP@0.3"], 2),
                "AP@0.5": round(r["AP@0.5"], 2),
                "AP@0.7": round(r["AP@0.7"], 2),
                "mean_w_j": round(mean_w, 4)
            }

    # Print Formatted Table
    print("\n" + "=" * 110)
    print("TABLE: GATE DECOMPOSITION ACROSS COMMUNICATION LAG (AP@0.5, Post-NMS)")
    print("=" * 110)
    header = f"{'Gate Variant':25s} | " + " | ".join([f"m={m} ({d}ms)" for m, d in zip(frame_lags, delays)])
    print(header)
    print("-" * 110)

    for label, _, _ in gate_variants:
        row_str = f"{label:25s} | "
        vals = []
        for d in delays:
            ap05 = results[label]["delays"][str(d)]["AP@0.5"]
            vals.append(f"{ap05:5.2f}%")
        row_str += " | ".join(vals)
        print(row_str)
    print("-" * 110)

    # Scientific comparison: Prior vs Full Gate
    prior_vs_full = [
        results["Prior-Only Gate"]["delays"][str(d)]["AP@0.5"] - results["Full Gate (Ours)"]["delays"][str(d)]["AP@0.5"]
        for d in delays
    ]
    mlp_vs_full = [
        results["MLP-Only Gate"]["delays"][str(d)]["AP@0.5"] - results["Full Gate (Ours)"]["delays"][str(d)]["AP@0.5"]
        for d in delays
    ]
    delay_only_vs_full = [
        results["Delay-Only Gate"]["delays"][str(d)]["AP@0.5"] - results["Full Gate (Ours)"]["delays"][str(d)]["AP@0.5"]
        for d in delays
    ]

    print("\n--- Key Findings ---")
    print(f"Prior-Only vs Full Gate diff:      mean {np.mean(prior_vs_full):+.2f}% (range: {np.min(prior_vs_full):+.2f}% to {np.max(prior_vs_full):+.2f}%)")
    print(f"MLP-Only vs Full Gate diff:        mean {np.mean(mlp_vs_full):+.2f}% (range: {np.min(mlp_vs_full):+.2f}% to {np.max(mlp_vs_full):+.2f}%)")
    print(f"Delay-Only vs Full Gate diff:      mean {np.mean(delay_only_vs_full):+.2f}% (range: {np.min(delay_only_vs_full):+.2f}% to {np.max(delay_only_vs_full):+.2f}%)")

    analysis = {
        "prior_vs_full_mean_diff": round(float(np.mean(prior_vs_full)), 2),
        "mlp_vs_full_mean_diff": round(float(np.mean(mlp_vs_full)), 2),
        "delay_only_vs_full_mean_diff": round(float(np.mean(delay_only_vs_full)), 2),
        "scale_shift_caveat": (
            "Note that Full Gate combines 0.5*MLP + 0.5*Prior, whereas Prior-Only and MLP-Only evaluate each "
            "component unweighted (1.0*Prior and 1.0*MLP respectively), resulting in an intrinsic scale shift "
            "in raw gate weight magnitudes."
        ),
        "verdict": (
            "If Prior-Only matches or outperforms Full Gate, the physical delay decay accounts for the entire "
            "reliable admission benefit, indicating that the learned MLP gate does not provide additional utility."
        )
    }

    artifact = {
        "experiment": "E4",
        "description": "Gate Decomposition into Prior-Only, MLP-Only, Delay-Only, and Ungated across True Lag",
        "checkpoint_path": "results/checkpoints/trustdelay_best.pth",
        "checkpoint_sha256": "37c6f2e59b75436917355e3c97c3e8c970b64d1ed6ff78f6f89006b57d0e6d58",
        "git_commit": commit,
        "split": "test",
        "seeds": [42],
        "frame_policy": "clamp_to_first",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "analysis": analysis
    }

    out_file = "results/experiments/e4_gate_decomposition.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nArtifact saved to: {out_file}")

if __name__ == "__main__":
    run_e4()
