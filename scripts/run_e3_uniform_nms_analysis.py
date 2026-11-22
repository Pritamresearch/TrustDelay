"""
Experiment 3: Uniform 2.5m Distance NMS Analysis.
Evaluates the effect of applying 2.5m distance NMS uniformly across all arms:
  - Table I pre-NMS vs post-NMS comparison (AP@0.3, AP@0.5, AP@0.7)
  - Box count reduction and false positive suppression
  - Footnote reference table preserving pre-NMS preprint traceability
Saves artifact to results/experiments/e3_nms_uniform_comparison.json.
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

def run_e3():
    commit = get_git_commit()
    print(f"=== E3: Uniform Distance NMS Analysis (Commit: {commit[:8]}) ===")

    gt_file = os.path.join(CACHE_DIR, "ground_truth.npz")
    gt_data = np.load(gt_file, allow_pickle=True)
    gt_boxes_all = gt_data["gt_boxes"]
    gt_moving_all = gt_data["gt_is_moving"]
    N_frames = len(gt_boxes_all)

    delays = [0, 100, 200, 300, 400, 500]
    frame_lags = [0, 1, 2, 3, 4, 5]

    arms = [
        ("Ego-Only", "ego_only", True),
        ("Late Fusion", "late_fusion", False),
        ("No-Compensation Fusion", "no_comp_fusion", False),
        ("Constant-Velocity Alignment", "const_vel_fusion", False),
        ("TrustDelay (Ours)", "trustdelay_full", False)
    ]

    evaluator = DetectionEvaluator(iou_thresholds=[0.3, 0.5, 0.7])

    pre_nms_table = {}
    post_nms_table = {}
    box_stats = {}

    for arm_label, cache_prefix, is_static in arms:
        pre_nms_table[arm_label] = {}
        post_nms_table[arm_label] = {}
        box_stats[arm_label] = {}

        for d in delays:
            fpath = os.path.join(CACHE_DIR, f"{cache_prefix}.npz" if is_static else f"{cache_prefix}_lag{d}.npz")
            data = np.load(fpath, allow_pickle=True)
            p_boxes_all = data["pred_boxes"]
            p_scores_all = data["pred_scores"]

            # Pre-NMS
            evaluator.reset()
            pre_box_counts = []
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
                pre_box_counts.append(len(b))
                evaluator.add_sample(b, sc, gt_boxes_all[i], gt_moving_all[i])
            r_pre = evaluator.evaluate()

            # Post-NMS (2.5m)
            evaluator.reset()
            post_box_counts = []
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
                post_box_counts.append(len(b))
                evaluator.add_sample(b, sc, gt_boxes_all[i], gt_moving_all[i])
            r_post = evaluator.evaluate()

            pre_nms_table[arm_label][str(d)] = {
                "AP@0.3": round(r_pre["AP@0.3"], 2),
                "AP@0.5": round(r_pre["AP@0.5"], 2),
                "AP@0.7": round(r_pre["AP@0.7"], 2)
            }
            post_nms_table[arm_label][str(d)] = {
                "AP@0.3": round(r_post["AP@0.3"], 2),
                "AP@0.5": round(r_post["AP@0.5"], 2),
                "AP@0.7": round(r_post["AP@0.7"], 2)
            }
            box_stats[arm_label][str(d)] = {
                "avg_boxes_pre_nms": round(float(np.mean(pre_box_counts)), 1),
                "avg_boxes_post_nms": round(float(np.mean(post_box_counts)), 1),
                "suppression_pct": round((1.0 - np.mean(post_box_counts) / np.mean(pre_box_counts)) * 100.0, 1)
            }

    print("\n" + "=" * 100)
    print("TABLE I POST-NMS (PRIMARY) vs PRE-NMS (FOOTNOTE REFERENCE) @ AP@0.5")
    print("=" * 100)
    print(f"{'Method':28s} | " + " | ".join([f"Lag {d}ms (Post / Pre)" for d in delays]))
    print("-" * 100)
    for arm_label, _, _ in arms:
        cells = []
        for d in delays:
            post = post_nms_table[arm_label][str(d)]["AP@0.5"]
            pre = pre_nms_table[arm_label][str(d)]["AP@0.5"]
            cells.append(f"{post:4.1f}% / {pre:4.1f}%")
        print(f"{arm_label:28s} | " + " | ".join(cells))
    print("-" * 100)

    print("\nAverage Box Emission per Frame (Post-NMS vs Pre-NMS across all 90 frames):")
    for arm_label, _, _ in arms:
        b_pre = box_stats[arm_label]["0"]["avg_boxes_pre_nms"]
        b_post = box_stats[arm_label]["0"]["avg_boxes_post_nms"]
        pct = box_stats[arm_label]["0"]["suppression_pct"]
        print(f"  {arm_label:28s}: {b_pre:4.1f} boxes/frame -> {b_post:4.1f} boxes/frame (-{pct:.1f}% duplicate query suppression)")

    artifact = {
        "experiment": "E3",
        "description": "Uniform 2.5m distance NMS comparison for primary vs preprint footnote tables",
        "git_commit": commit,
        "checkpoint_path": "results/checkpoints/trustdelay_best.pth (epoch-14)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "table1_post_nms_primary": post_nms_table,
        "table1_pre_nms_reference": pre_nms_table,
        "box_emission_statistics": box_stats
    }

    out_file = "results/experiments/e3_nms_uniform_comparison.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nArtifact saved to: {out_file}")

if __name__ == "__main__":
    run_e3()
