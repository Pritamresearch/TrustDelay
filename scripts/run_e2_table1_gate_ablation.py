"""
Experiment 2: Gate Ablation under True Lag across Table I (0 to 500 ms).
Evaluates Full TrustDelay (gated) vs TrustDelay w/o Gate (uniform attention, kinematics retained)
across all 6 true-delay columns (m = 0 to 5 frames).
Evaluates both with 2.5m NMS (primary) and pre-NMS (reference).
Saves artifact to results/experiments/e2_gate_ablation_table1.json.
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

def run_e2():
    commit = get_git_commit()
    print(f"=== E2: Gate Ablation in Table I (Commit: {commit[:8]}) ===")

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
        ("TrustDelay w/o Gate", "trustdelay_no_gate", False),
        ("TrustDelay (Ours)", "trustdelay_full", False)
    ]

    evaluator = DetectionEvaluator(iou_thresholds=[0.3, 0.5, 0.7])

    table1_post_nms = {str(d): {} for d in delays}
    table1_pre_nms = {str(d): {} for d in delays}

    for d in delays:
        for arm_label, cache_prefix, is_static in arms:
            if is_static:
                fpath = os.path.join(CACHE_DIR, f"{cache_prefix}.npz")
            else:
                fpath = os.path.join(CACHE_DIR, f"{cache_prefix}_lag{d}.npz")

            data = np.load(fpath, allow_pickle=True)
            p_boxes_all = data["pred_boxes"]
            p_scores_all = data["pred_scores"]
            comm_kb = float(np.mean(data["comm_kb"]))

            # Pre-NMS
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
                evaluator.add_sample(b, sc, gt_boxes_all[i], gt_moving_all[i])
            r_pre = evaluator.evaluate()

            table1_pre_nms[str(d)][arm_label] = {
                "AP@0.3": round(r_pre["AP@0.3"], 2),
                "AP@0.5": round(r_pre["AP@0.5"], 2),
                "AP@0.7": round(r_pre["AP@0.7"], 2),
                "comm_kb": round(comm_kb, 1)
            }

            # Post-NMS (2.5m)
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
            r_post = evaluator.evaluate()

            table1_post_nms[str(d)][arm_label] = {
                "AP@0.3": round(r_post["AP@0.3"], 2),
                "AP@0.5": round(r_post["AP@0.5"], 2),
                "AP@0.7": round(r_post["AP@0.7"], 2),
                "comm_kb": round(comm_kb, 1)
            }

    # Print Formatted Table I with Gate Ablation row (Post-NMS)
    print("\n" + "=" * 115)
    print("TABLE I (POST-NMS): DELAY SWEEP WITH ISOLATED GATE ABLATION ('TrustDelay w/o Gate')")
    print("=" * 115)
    header = f"{'Method':28s} | {'Payload':7s} | " + " | ".join([f"m={m} ({d}ms)" for m, d in zip(frame_lags, delays)])
    print(header)
    print("-" * 115)

    gate_deltas = {}
    for arm_label, _, _ in arms:
        row_str = f"{arm_label:28s} | "
        comm_val = table1_post_nms["0"][arm_label]["comm_kb"]
        row_str += f"{comm_val:4.1f} KB | "
        vals = []
        for d in delays:
            ap05 = table1_post_nms[str(d)][arm_label]["AP@0.5"]
            vals.append(f"{ap05:5.2f}%")
        row_str += " | ".join(vals)
        print(row_str)

    print("-" * 115)

    # Acceptance check: does gated beat ungated at every lag?
    print("\n--- Gate Advantage (AP@0.5 Delta: Gated TrustDelay - TrustDelay w/o Gate) ---")
    gated_beats_ungated_all = True
    for d, m in zip(delays, frame_lags):
        gated_ap = table1_post_nms[str(d)]["TrustDelay (Ours)"]["AP@0.5"]
        ungated_ap = table1_post_nms[str(d)]["TrustDelay w/o Gate"]["AP@0.5"]
        diff = round(gated_ap - ungated_ap, 2)
        gate_deltas[str(d)] = diff
        status = "PASS (Gated > Ungated)" if diff > 0 else ("MATCH" if diff == 0 else "FAIL (Ungated >= Gated)")
        if diff < 0:
            gated_beats_ungated_all = False
        print(f"  Delay {d:3d} ms (m={m}): Gated {gated_ap:5.2f}% vs Ungated {ungated_ap:5.2f}% -> Delta: {diff:+5.2f}% [{status}]")

    print(f"\nAcceptance criterion verdict: Gated beats ungated across all lags? {gated_beats_ungated_all}")

    artifact = {
        "experiment": "E2",
        "description": "Table I delay sweep including isolated TrustDelay w/o Gate arm",
        "git_commit": commit,
        "checkpoint_path": "results/checkpoints/trustdelay_best.pth (epoch-14)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "table1_post_nms_primary": table1_post_nms,
        "table1_pre_nms_reference": table1_pre_nms,
        "gate_advantage_deltas": gate_deltas,
        "gated_beats_ungated_all_lags": gated_beats_ungated_all
    }

    out_file = "results/experiments/e2_gate_ablation_table1.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"Artifact saved to: {out_file}")

if __name__ == "__main__":
    run_e2()
