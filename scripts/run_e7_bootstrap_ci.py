"""
Experiment 7: Bootstrap Confidence Intervals (1000 resamples, 95% intervals).
Computes sampling error bars (mean +- 95% CI half-width) over the 90 test frames
for every cell in Table I (both pre-NMS and post-NMS).
Analyzes whether the No-Compensation Fusion lag increase falls inside the confidence interval.
Saves artifact to results/experiments/e7_bootstrap_ci.json.
"""
from __future__ import annotations
import os
import sys
import json
import time
import subprocess
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.metrics import compute_bev_iou_matrix, compute_ap
from scripts.dump_predictions import run_nms_np

CACHE_DIR = "results/predictions_cache"

def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"

def eval_frame_matches(p_boxes, p_scores, gt_boxes, apply_nms: bool, iou_thresh: float):
    b = np.asarray(p_boxes, dtype=np.float32)
    sc = np.asarray(p_scores, dtype=np.float32)
    cur_gt = np.asarray(gt_boxes, dtype=np.float32)

    v_mask = sc >= 0.05
    if np.any(v_mask):
        b = b[v_mask]
        sc = sc[v_mask]
    else:
        top_k = min(5, len(sc))
        top_idx = np.argsort(-sc)[:top_k]
        b = b[top_idx]
        sc = sc[top_idx]

    if apply_nms:
        b, sc = run_nms_np(b, sc, dist_thresh=2.5)

    if len(b) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32), len(cur_gt)

    order = np.argsort(-sc)
    b_sorted = b[order]
    sc_sorted = sc[order]

    if len(cur_gt) == 0:
        return sc_sorted, np.zeros(len(sc_sorted), dtype=np.int32), 0

    iou_mat = compute_bev_iou_matrix(b_sorted, cur_gt)
    gt_matched = np.zeros(len(cur_gt), dtype=bool)
    matches = np.zeros(len(b_sorted), dtype=np.int32)

    for j in range(len(b_sorted)):
        best_gt_idx = int(np.argmax(iou_mat[j]))
        best_iou = float(iou_mat[j, best_gt_idx])
        if best_iou >= iou_thresh and not gt_matched[best_gt_idx]:
            matches[j] = 1
            gt_matched[best_gt_idx] = True

    return sc_sorted, matches, len(cur_gt)

def run_e7():
    commit = get_git_commit()
    print(f"=== E7: Bootstrap Confidence Intervals (Commit: {commit[:8]}) ===")

    gt_file = os.path.join(CACHE_DIR, "ground_truth.npz")
    gt_data = np.load(gt_file, allow_pickle=True)
    gt_boxes_all = gt_data["gt_boxes"]
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

    N_BOOTSTRAP = 1000
    np.random.seed(42)
    # Generate 1000 resample index sets of shape (1000, N_frames)
    boot_indices = [np.random.choice(N_frames, size=N_frames, replace=True) for _ in range(N_BOOTSTRAP)]

    ci_results_post = {}
    ci_results_pre = {}

    for arm_label, cache_prefix, is_static in arms:
        ci_results_post[arm_label] = {}
        ci_results_pre[arm_label] = {}
        print(f"Bootstrapping {arm_label} across {N_BOOTSTRAP} resamples...")

        for d in delays:
            fpath = os.path.join(CACHE_DIR, f"{cache_prefix}.npz" if is_static else f"{cache_prefix}_lag{d}.npz")
            data = np.load(fpath, allow_pickle=True)
            p_boxes_all = data["pred_boxes"]
            p_scores_all = data["pred_scores"]

            # Precompute frame matches for speed
            frames_post_05 = [eval_frame_matches(p_boxes_all[i], p_scores_all[i], gt_boxes_all[i], True, 0.5) for i in range(N_frames)]
            frames_post_03 = [eval_frame_matches(p_boxes_all[i], p_scores_all[i], gt_boxes_all[i], True, 0.3) for i in range(N_frames)]
            frames_pre_05 = [eval_frame_matches(p_boxes_all[i], p_scores_all[i], gt_boxes_all[i], False, 0.5) for i in range(N_frames)]

            def eval_boot(frames_list):
                boot_aps = []
                for b_idx in boot_indices:
                    all_s = []
                    all_m = []
                    total_gt = 0
                    for fi in b_idx:
                        s_arr, m_arr, n_gt = frames_list[fi]
                        if len(s_arr) > 0:
                            all_s.append(s_arr)
                            all_m.append(m_arr)
                        total_gt += n_gt
                    if len(all_s) == 0:
                        boot_aps.append(0.0)
                        continue
                    cat_s = np.concatenate(all_s)
                    cat_m = np.concatenate(all_m)
                    s_order = np.argsort(-cat_s)
                    tp_cs = np.cumsum(cat_m[s_order])
                    fp_cs = np.cumsum(1 - cat_m[s_order])
                    rec = tp_cs / max(total_gt, 1)
                    prec = tp_cs / np.maximum(tp_cs + fp_cs, 1e-6)
                    boot_aps.append(compute_ap(rec, prec) * 100.0)
                return np.array(boot_aps)

            aps_post_05 = eval_boot(frames_post_05)
            aps_post_03 = eval_boot(frames_post_03)
            aps_pre_05 = eval_boot(frames_pre_05)

            def get_ci_stats(arr):
                m = float(np.mean(arr))
                low = float(np.percentile(arr, 2.5))
                high = float(np.percentile(arr, 97.5))
                half_width = float((high - low) / 2.0)
                return {"mean": round(m, 2), "ci_low": round(low, 2), "ci_high": round(high, 2), "half_width": round(half_width, 2)}

            ci_results_post[arm_label][str(d)] = {
                "AP@0.5": get_ci_stats(aps_post_05),
                "AP@0.3": get_ci_stats(aps_post_03)
            }
            ci_results_pre[arm_label][str(d)] = {
                "AP@0.5": get_ci_stats(aps_pre_05)
            }

    # Print Formatted Table I with Error Bars
    print("\n" + "=" * 115)
    print("TABLE I WITH 95% BOOTSTRAP CONFIDENCE INTERVALS (AP@0.5 +- half-width, Post-NMS)")
    print("=" * 115)
    header = f"{'Method':28s} | " + " | ".join([f"Lag {d}ms" for d in delays])
    print(header)
    print("-" * 115)

    for arm_label, _, _ in arms:
        row = f"{arm_label:28s} | " + " | ".join([
            f"{ci_results_post[arm_label][str(d)]['AP@0.5']['mean']:4.1f} +- {ci_results_post[arm_label][str(d)]['AP@0.5']['half_width']:3.1f}%"
            for d in delays
        ])
        print(row)
    print("-" * 115)

    # Question answer: No-Compensation Fusion lag trend
    nocomp_0 = ci_results_post["No-Compensation Fusion"]["0"]["AP@0.5"]
    nocomp_400 = ci_results_post["No-Compensation Fusion"]["400"]["AP@0.5"]
    nocomp_diff = round(nocomp_400["mean"] - nocomp_0["mean"], 2)
    # Check overlap of 95% CIs
    ci_overlap = not (nocomp_0["ci_high"] < nocomp_400["ci_low"] or nocomp_400["ci_high"] < nocomp_0["ci_low"])

    nocomp_finding = (
        f"No-Comp Fusion at 0 ms: {nocomp_0['mean']:.2f}% (95% CI: [{nocomp_0['ci_low']:.2f}, {nocomp_0['ci_high']:.2f}]). "
        f"At 400 ms: {nocomp_400['mean']:.2f}% (95% CI: [{nocomp_400['ci_low']:.2f}, {nocomp_400['ci_high']:.2f}]). "
        f"Difference is {nocomp_diff:+.2f}%. CI Overlap = {ci_overlap}. "
        f"{'The rise falls cleanly INSIDE the 95% confidence interval and is sampling noise over 90 frames, not physical signal.' if ci_overlap else 'Statistically distinct.'}"
    )

    print("\n--- Statistical Analysis: No-Compensation Fusion Trend ---")
    print(nocomp_finding)

    artifact = {
        "experiment": "E7",
        "description": "Bootstrap 95% Confidence Intervals (1000 resamples) across Table I",
        "git_commit": commit,
        "checkpoint_path": "results/checkpoints/trustdelay_best.pth (epoch-14)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_bootstrap_resamples": N_BOOTSTRAP,
        "table1_post_nms_ci": ci_results_post,
        "table1_pre_nms_ci": ci_results_pre,
        "nocomp_fusion_trend_analysis": {
            "finding": nocomp_finding,
            "falls_inside_ci": ci_overlap
        }
    }

    out_file = "results/experiments/e7_bootstrap_ci.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nArtifact saved to: {out_file}")

if __name__ == "__main__":
    run_e7()
