"""
Experiment 5: Gate Ablation at Tight Budgets (1.0 KB, 5.0 KB, 8.4 KB at 200 ms true lag).
Evaluates Gated TrustDelay vs Ungated (Uniform) across bandwidth budgets:
  - 1.0 KB (8 tokens)
  - 5.0 KB (40 tokens)
  - 8.4 KB (67 tokens)
Computes absolute and relative gate advantage as a function of bandwidth.
Saves artifact to results/experiments/e5_tight_budgets.json.
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

def run_e5():
    commit = get_git_commit()
    print(f"=== E5: Gate Ablation at Tight Budgets (Commit: {commit[:8]}) ===")

    gt_file = os.path.join(CACHE_DIR, "ground_truth.npz")
    gt_data = np.load(gt_file, allow_pickle=True)
    gt_boxes_all = gt_data["gt_boxes"]
    gt_moving_all = gt_data["gt_is_moving"]
    N_frames = len(gt_boxes_all)

    budgets = [1.0, 5.0, 8.4]
    tokens_map = {1.0: 8, 5.0: 40, 8.4: 67}
    evaluator = DetectionEvaluator(iou_thresholds=[0.3, 0.5, 0.7])

    results = {}

    for b in budgets:
        results[str(b)] = {"tokens": tokens_map[b]}
        for mode in ["gated", "ungated"]:
            fpath = os.path.join(CACHE_DIR, f"budget_{b}kb_{mode}.npz")
            data = np.load(fpath, allow_pickle=True)
            p_boxes_all = data["pred_boxes"]
            p_scores_all = data["pred_scores"]

            evaluator.reset()
            for i in range(N_frames):
                b_arr = np.asarray(p_boxes_all[i], dtype=np.float32)
                sc_arr = np.asarray(p_scores_all[i], dtype=np.float32)
                v_mask = sc_arr >= 0.05
                if np.any(v_mask):
                    b_arr = b_arr[v_mask]
                    sc_arr = sc_arr[v_mask]
                else:
                    top_k = min(5, len(sc_arr))
                    top_idx = np.argsort(-sc_arr)[:top_k]
                    b_arr = b_arr[top_idx]
                    sc_arr = sc_arr[top_idx]
                b_arr, sc_arr = run_nms_np(b_arr, sc_arr, dist_thresh=2.5)
                evaluator.add_sample(b_arr, sc_arr, gt_boxes_all[i], gt_moving_all[i])
            r = evaluator.evaluate()
            results[str(b)][mode] = {
                "AP@0.3": round(r["AP@0.3"], 2),
                "AP@0.5": round(r["AP@0.5"], 2),
                "AP@0.7": round(r["AP@0.7"], 2)
            }

        gated_05 = results[str(b)]["gated"]["AP@0.5"]
        ungated_05 = results[str(b)]["ungated"]["AP@0.5"]
        diff_abs = round(gated_05 - ungated_05, 2)
        diff_rel = round((diff_abs / max(ungated_05, 0.01)) * 100.0, 1)

        results[str(b)]["advantage"] = {
            "abs_diff_pts": diff_abs,
            "rel_advantage_pct": diff_rel
        }

    # Print Formatted Table
    print("\n" + "=" * 90)
    print("TABLE: GATE ADVANTAGE AS A FUNCTION OF BANDWIDTH BUDGET (True Lag = 200 ms)")
    print("=" * 90)
    print(f"{'Budget':10s} | {'Tokens':8s} | {'Gated AP@0.5':14s} | {'Ungated AP@0.5':16s} | {'Abs Diff':10s} | {'Rel Gain'}")
    print("-" * 90)
    for b in budgets:
        tok = results[str(b)]["tokens"]
        g_05 = results[str(b)]["gated"]["AP@0.5"]
        u_05 = results[str(b)]["ungated"]["AP@0.5"]
        diff_a = results[str(b)]["advantage"]["abs_diff_pts"]
        diff_r = results[str(b)]["advantage"]["rel_advantage_pct"]
        print(f"{b:4.1f} KB    | {tok:2d} tokens | {g_05:6.2f}%       | {u_05:6.2f}%         | {diff_a:+6.2f} pts | {diff_r:+6.1f}%")
    print("-" * 90)

    # Trend Analysis
    diffs = [results[str(b)]["advantage"]["rel_advantage_pct"] for b in budgets]
    rel_grows_as_budget_tightens = diffs[0] > diffs[-1]

    analysis = {
        "trend_summary": (
            f"At 1.0 KB (8 tokens), gate relative advantage is {results['1.0']['advantage']['rel_advantage_pct']:+.1f}%. "
            f"At 8.4 KB (67 tokens), gate relative advantage is {results['8.4']['advantage']['rel_advantage_pct']:+.1f}%. "
            f"Relative advantage {'GROWS' if rel_grows_as_budget_tightens else 'SHRINKS'} as budget tightens."
        ),
        "supports_budget_framing": rel_grows_as_budget_tightens
    }
    print(f"\n[Analysis] {analysis['trend_summary']}")

    artifact = {
        "experiment": "E5",
        "description": "Gate Advantage across Bandwidth Budgets at 200 ms True Lag",
        "git_commit": commit,
        "checkpoint_path": "results/checkpoints/trustdelay_best.pth (epoch-14)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "operating_point": {
            "delay_ms": 200.0,
            "true_frame_lag": 2,
            "budgets_kb": budgets
        },
        "results": results,
        "analysis": analysis
    }

    out_file = "results/experiments/e5_tight_budgets.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"Artifact saved to: {out_file}")

if __name__ == "__main__":
    run_e5()
