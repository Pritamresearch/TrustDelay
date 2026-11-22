"""
Experiment 3: Decompose the Extrapolation Ablation.
Evaluates the 4-way factorial decomposition of the extrapolation module:
  1. Full TrustDelay: Kinematic Projection ON, Feature Update ON
  2. Projection-Only-Off: Kinematic Projection OFF (p_hat = p), Feature Update ON
  3. Feature-Update-Off: Kinematic Projection ON, Feature Update OFF (q_tilde = q)
  4. Both-Off: Kinematic Projection OFF, Feature Update OFF

Operating point:
  True delay: 300 ms (m = 3 frames lag)
  Packet loss: 10% (0.10)
  Pose noise: 0.1 m / 0.1 deg
  Seeds: [42, 43, 44]
  Dataset split: test (90 frames)

Calculates the attributable share of the degradation for kinematic projection vs. feature update.
Saves artifact to results/experiments/e3_extrapolation_decomposition.json.
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

def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"

def run_e3():
    cfg = TrustDelayConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt_path = "results/checkpoints/trustdelay_best.pth"
    commit = get_git_commit()

    print(f"=== Experiment 3: Extrapolation Ablation Decomposition (Commit: {commit[:8]}) ===")
    print(f"Loading checkpoint: {ckpt_path}")
    model = TrustDelayModel(cfg).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    test_dataset = OPV2VAsynchronousDataset(cfg, split="test")
    evaluator = DetectionEvaluator(iou_thresholds=[0.3, 0.5, 0.7])

    delay_ms = 300.0
    true_frame_lag = 3
    test_dataset.set_delay_ms(int(delay_ms))

    p_loss = 0.10
    pose_trans = 0.10
    pose_rot = 0.10
    seeds = [42, 43, 44]
    budget_kb = 8.4

    modes = [
        ("Full TrustDelay", "learned", "Kinematics ON, Feature Update ON"),
        ("Projection-Only-Off", "projection_only_off", "Kinematics OFF (p_hat = p), Feature Update ON"),
        ("Feature-Update-Off", "feature_update_off", "Kinematics ON, Feature Update OFF (q_tilde = q)"),
        ("Both-Off", "none", "Kinematics OFF, Feature Update OFF")
    ]

    results = {}
    print(f"\nOperating Point: True Delay = {int(delay_ms)} ms ({true_frame_lag} frames), Packet Loss = {p_loss*100:.0f}%, Pose Noise = {pose_trans}m/{pose_rot}deg")
    print("-" * 85)

    for label, mode_key, desc in modes:
        seed_03, seed_05, seed_07, comm_kbs = [], [], [], []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            evaluator.reset()

            for idx in range(len(test_dataset)):
                sample = test_dataset[idx]
                gt_boxes = sample["gt_boxes"].numpy()
                gt_moving = sample["gt_is_moving"].numpy()

                with torch.inference_mode():
                    out = model(
                        sample,
                        delay_override_ms=delay_ms,
                        packet_loss_override=p_loss,
                        pose_noise_trans=pose_trans,
                        pose_noise_rot=pose_rot,
                        bandwidth_budget_kb=budget_kb,
                        extrapolation_mode=mode_key
                    )

                p_boxes = out["pred_boxes"][0].cpu().numpy()
                p_logits = out["pred_logits"][0].cpu().numpy()
                p_scores = 1.0 / (1.0 + np.exp(-np.clip(p_logits, -10.0, 10.0)))
                comm_kbs.append(float(out.get("transmitted_kb", budget_kb)))
                evaluator.add_sample(p_boxes, p_scores, gt_boxes, gt_moving)

            res = evaluator.evaluate()
            seed_03.append(res["AP@0.3"])
            seed_05.append(res["AP@0.5"])
            seed_07.append(res["AP@0.7"])

        m_03, s_03 = float(np.mean(seed_03)), float(np.std(seed_03))
        m_05, s_05 = float(np.mean(seed_05)), float(np.std(seed_05))
        m_07, s_07 = float(np.mean(seed_07)), float(np.std(seed_07))
        avg_kb = float(np.mean(comm_kbs))

        results[label] = {
            "mode_key": mode_key,
            "description": desc,
            "AP@0.3_mean": round(m_03, 2),
            "AP@0.3_std": round(s_03, 2),
            "AP@0.5_mean": round(m_05, 2),
            "AP@0.5_std": round(s_05, 2),
            "AP@0.7_mean": round(m_07, 2),
            "AP@0.7_std": round(s_07, 2),
            "comm_kb": round(avg_kb, 1),
            "seed_runs": {
                "seeds": seeds,
                "AP@0.3": [round(x, 2) for x in seed_03],
                "AP@0.5": [round(x, 2) for x in seed_05],
                "AP@0.7": [round(x, 2) for x in seed_07]
            }
        }
        print(f"  {label:22s} | AP@0.3: {m_03:5.2f} ± {s_03:4.2f}% | AP@0.5: {m_05:5.2f} ± {s_05:4.2f}% | AP@0.7: {m_07:5.2f} ± {s_07:4.2f}% | {desc}")

    # Calculate attributable share of degradation
    full_ap05 = results["Full TrustDelay"]["AP@0.5_mean"]
    proj_off_ap05 = results["Projection-Only-Off"]["AP@0.5_mean"]
    feat_off_ap05 = results["Feature-Update-Off"]["AP@0.5_mean"]
    both_off_ap05 = results["Both-Off"]["AP@0.5_mean"]

    total_drop = full_ap05 - both_off_ap05
    proj_drop = full_ap05 - proj_off_ap05
    feat_drop = full_ap05 - feat_off_ap05

    proj_share_pct = (proj_drop / total_drop * 100.0) if abs(total_drop) > 1e-5 else 0.0
    feat_share_pct = (feat_drop / total_drop * 100.0) if abs(total_drop) > 1e-5 else 0.0

    print("\n--- Decomposition Summary (AP@0.5) ---")
    print(f"Full TrustDelay:         {full_ap05:5.2f}%")
    print(f"Projection-Only-Off:     {proj_off_ap05:5.2f}% (Drop: {proj_drop:+5.2f} pts)")
    print(f"Feature-Update-Off:      {feat_off_ap05:5.2f}% (Drop: {feat_drop:+5.2f} pts)")
    print(f"Both-Off (Raw Features): {both_off_ap05:5.2f}% (Total Drop: {total_drop:+5.2f} pts)")
    print(f"Kinematic Projection Share: {proj_share_pct:5.1f}% of total degradation")
    print(f"Feature Update Share:       {feat_share_pct:5.1f}% of total degradation")

    analysis = {
        "full_ap05": full_ap05,
        "proj_only_off_ap05": proj_off_ap05,
        "feat_update_off_ap05": feat_off_ap05,
        "both_off_ap05": both_off_ap05,
        "total_drop_pts": round(total_drop, 2),
        "kinematic_drop_pts": round(proj_drop, 2),
        "feature_update_drop_pts": round(feat_drop, 2),
        "kinematic_share_pct": round(proj_share_pct, 1),
        "feature_update_share_pct": round(feat_share_pct, 1)
    }

    artifact = {
        "experiment": "E3",
        "description": "Decomposed extrapolation ablation into kinematic projection vs learned feature update",
        "git_commit": commit,
        "checkpoint_path": ckpt_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "operating_point": {
            "true_delay_ms": delay_ms,
            "true_frame_lag": true_frame_lag,
            "packet_loss_rate": p_loss,
            "pose_noise_translation_m": pose_trans,
            "pose_noise_rotation_deg": pose_rot,
            "bandwidth_budget_kb": budget_kb,
            "seeds": seeds
        },
        "results": results,
        "decomposition_analysis": analysis
    }

    out_path = "results/experiments/e3_extrapolation_decomposition.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nArtifact saved successfully to: {out_path}")

if __name__ == "__main__":
    run_e3()
