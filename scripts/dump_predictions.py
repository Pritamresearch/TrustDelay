"""
E0: Unified Prediction-Dumping Pass for Laptop-Scale TrustDelay Experiments.
Saves raw per-frame predictions (boxes, scores, gate weights, comm volume) to disk
for all experiment arms and operating conditions before any thresholding, NMS or AP computation.

Total GPU budget: < 3 minutes on RTX 3050 Laptop GPU.
All downstream experiments (E1 to E9) compute offline over this cache.
"""
from __future__ import annotations
import os
import sys
import time
import math
import subprocess
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config import TrustDelayConfig
from src.data.opv2v_dataset import OPV2VAsynchronousDataset
from src.models.trust_delay_model import TrustDelayModel
from tests.test_model_leakage import test_zero_leakage

CACHE_DIR = "results/predictions_cache"

def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"

def run_nms_np(boxes: np.ndarray, scores: np.ndarray, dist_thresh: float = 2.5):
    if len(boxes) <= 1:
        return boxes, scores
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        idx = order[0]
        keep.append(idx)
        if len(order) == 1:
            break
        curr_box = boxes[idx]
        rem_boxes = boxes[order[1:]]
        dist = np.linalg.norm(curr_box[:2] - rem_boxes[:, :2], axis=1)
        suppress = dist < dist_thresh
        order = order[1:][~suppress]
    return boxes[keep], scores[keep]

def dump_predictions():
    # 0. Zero-leakage sentinel check
    print("Verifying zero ground-truth leakage...")
    test_zero_leakage()
    print("Zero-leakage test PASSED! Proceeding to prediction dumping.")

    os.makedirs(CACHE_DIR, exist_ok=True)
    cfg = TrustDelayConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt_path = "results/checkpoints/trustdelay_best.pth"
    commit = get_git_commit()

    print(f"=== E0: Prediction Dump (Commit: {commit[:8]}) ===")
    print(f"Loading checkpoint: {ckpt_path}")
    model = TrustDelayModel(cfg).to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print("Preloading 90 test frames into memory cache...")
    dataset = OPV2VAsynchronousDataset(cfg, split="test")
    # Pre-cache raw samples in memory to eliminate disk IO during sweep
    samples = [dataset[i] for i in range(len(dataset))]
    N_frames = len(samples)
    print(f"Loaded {N_frames} frames into RAM cache.")

    # 1. Save ground truth cache once
    gt_file = os.path.join(CACHE_DIR, "ground_truth.npz")
    gt_boxes_list = [s["gt_boxes"].numpy() for s in samples]
    gt_moving_list = [s["gt_is_moving"].numpy() for s in samples]
    np.savez_compressed(
        gt_file,
        gt_boxes=np.array(gt_boxes_list, dtype=object),
        gt_is_moving=np.array(gt_moving_list, dtype=object),
        frame_ids=np.arange(N_frames)
    )
    print(f"Ground truth cached to: {gt_file}")

    t0_all = time.time()

    # Helper function to run one pass and save cache
    def run_and_cache(cache_key: str, run_fn):
        out_path = os.path.join(CACHE_DIR, f"{cache_key}.npz")
        if os.path.exists(out_path):
            return
        
        all_pred_boxes = []
        all_pred_scores = []
        all_gate_weights = []
        all_comm_kb = []

        for idx in range(N_frames):
            sample = samples[idx]
            boxes, scores, gate_w, comm_kb = run_fn(sample)
            all_pred_boxes.append(boxes)
            all_pred_scores.append(scores)
            all_gate_weights.append(gate_w)
            all_comm_kb.append(comm_kb)

        np.savez_compressed(
            out_path,
            pred_boxes=np.array(all_pred_boxes, dtype=object),
            pred_scores=np.array(all_pred_scores, dtype=object),
            gate_weights=np.array(all_gate_weights, dtype=object),
            comm_kb=np.array(all_comm_kb, dtype=np.float32),
            frame_ids=np.arange(N_frames)
        )

    # -------------------------------------------------------------
    # 1. Ego-Only Baseline
    # -------------------------------------------------------------
    print("\n--- Dumping Ego-Only Baseline ---")
    def run_ego_only(sample):
        with torch.inference_mode():
            out = model(sample, fusion_type="ego_only")
            p_boxes = out["pred_boxes"][0].cpu().numpy()
            p_logits = out["pred_logits"][0].cpu().numpy()
            p_scores = 1.0 / (1.0 + np.exp(-np.clip(p_logits, -10.0, 10.0)))
        return p_boxes, p_scores, np.empty((0,), dtype=np.float32), 0.0

    run_and_cache("ego_only", run_ego_only)

    # -------------------------------------------------------------
    # 2. Table I Sweep across Delays (0 to 500 ms)
    # -------------------------------------------------------------
    delays = [0, 100, 200, 300, 400, 500]

    for d in delays:
        print(f"\n--- Dumping Table I Arms @ Delay {d} ms ---")
        dataset.set_delay_ms(d)
        # Refresh samples with collaborator lag
        d_samples = [dataset[i] for i in range(N_frames)]

        # 2a. Late Fusion
        def make_late_fusion_fn(motion_comp: bool = False):
            def run_lf(sample):
                with torch.inference_mode():
                    out_ego = model(sample, fusion_type="ego_only")
                    ego_b = out_ego["pred_boxes"][0].cpu().numpy()
                    ego_logits = out_ego["pred_logits"][0].cpu().numpy()
                    ego_s = 1.0 / (1.0 + np.exp(-np.clip(ego_logits, -10.0, 10.0)))

                    collabs = sample.get("collabs", [])
                    if len(collabs) > 0 and len(collabs[0]["bev_features"]) > 0:
                        c_info = collabs[0]
                        c_bev = torch.from_numpy(c_info["bev_features"]).unsqueeze(0).to(device)
                        c_feat = model.encoder(c_bev)
                        c_queries, c_local_boxes, _ = model.query_gen(c_feat)
                        c_logits = model.fusion.cls_head(c_queries).squeeze(-1)[0].cpu().numpy()
                        c_res = model.fusion.reg_head(c_queries)[0].cpu().numpy()
                        c_s = 1.0 / (1.0 + np.exp(-np.clip(c_logits, -10.0, 10.0)))

                        c_b_collab = c_local_boxes[0].cpu().numpy()
                        c_b_ref = np.hstack([
                            c_b_collab[:, :3] + c_res[:, :3],
                            c_b_collab[:, 3:6] * np.exp(np.clip(c_res[:, 3:6], -1.0, 1.0)),
                            c_b_collab[:, 6:7] + c_res[:, 6:7]
                        ])

                        # Confidence filtering at collaborator
                        c_mask = c_s >= 0.05
                        if np.any(c_mask):
                            c_b_ref = c_b_ref[c_mask]
                            c_s = c_s[c_mask]
                            c_q_filt = c_queries[:, c_mask]
                            c_b_collab_filt = c_local_boxes[:, c_mask]
                        else:
                            top_k = min(5, len(c_s))
                            top_idx = np.argsort(-c_s)[:top_k]
                            c_b_ref = c_b_ref[top_idx]
                            c_s = c_s[top_idx]
                            c_q_filt = c_queries[:, top_idx]
                            c_b_collab_filt = c_local_boxes[:, top_idx]

                        T_ego_collab = c_info["T_ego_collab"].copy()
                        pts_c = np.hstack([c_b_ref[:, :3], np.ones((len(c_b_ref), 1))])
                        pts_e = (T_ego_collab @ pts_c.T).T[:, :3]
                        yaw_offset = math.atan2(T_ego_collab[1, 0], T_ego_collab[0, 0])

                        if motion_comp and d > 0:
                            # Project transmitted box centres forward by p + v_hat * dt
                            dt_val = d / 1000.0
                            dt_t = torch.tensor([[[dt_val]]], device=device).expand(-1, c_q_filt.shape[1], -1)
                            init_v = torch.zeros((1, c_q_filt.shape[1], 1), device=device)
                            kin_inp = torch.cat([c_q_filt, c_b_collab_filt, dt_t, init_v, init_v, init_v, init_v], dim=-1)
                            h = model.motion_extrapolator.motion_net(kin_inp)
                            kin_preds = model.motion_extrapolator.kinematics_head(h)[0].cpu().numpy()
                            c_v_collab = np.vstack([kin_preds[:, 0], kin_preds[:, 1]])
                            R_ego_c = T_ego_collab[:2, :2]
                            v_ego = R_ego_c @ c_v_collab
                            pts_e[:, 0] += v_ego[0] * dt_val
                            pts_e[:, 1] += v_ego[1] * dt_val

                        c_b_in_ego = np.hstack([pts_e, c_b_ref[:, 3:6], (c_b_ref[:, 6] + yaw_offset)[:, None]])
                        comm_kb = float(len(c_b_ref) * 8 * 4 / 1024.0)

                        merged_b = np.vstack([ego_b, c_b_in_ego])
                        merged_s = np.concatenate([ego_s, c_s])
                    else:
                        merged_b = ego_b
                        merged_s = ego_s
                        comm_kb = 0.0

                return merged_b, merged_s, np.empty((0,), dtype=np.float32), comm_kb
            return run_lf

        run_and_cache(f"late_fusion_lag{d}", make_late_fusion_fn(motion_comp=False))
        run_and_cache(f"mc_late_fusion_lag{d}", make_late_fusion_fn(motion_comp=True))

        # Query-based arms helper
        def make_query_arm_fn(extrap_mode: str, fuse_type: str, rank_score_only: bool = False, ignore_unc: bool = False):
            def run_query_arm(sample):
                with torch.inference_mode():
                    out = model(
                        sample,
                        extrapolation_mode=extrap_mode,
                        fusion_type=fuse_type,
                        delay_override_ms=float(d),
                        packet_loss_override=0.0,
                        pose_noise_trans=0.0,
                        pose_noise_rot=0.0,
                        bandwidth_budget_kb=8.4,
                        rank_by_score_only=rank_score_only,
                        ignore_uncertainty=ignore_unc
                    )
                    p_boxes = out["pred_boxes"][0].cpu().numpy()
                    p_logits = out["pred_logits"][0].cpu().numpy()
                    p_scores = 1.0 / (1.0 + np.exp(-np.clip(p_logits, -10.0, 10.0)))
                    comm_kb = float(out.get("transmitted_kb", 8.4))
                    gate_w = out.get("reliability_weights", torch.empty(0)).cpu().numpy().flatten()
                return p_boxes, p_scores, gate_w, comm_kb
            return run_query_arm

        run_and_cache(f"no_comp_fusion_lag{d}", make_query_arm_fn("none", "uniform"))
        run_and_cache(f"const_vel_fusion_lag{d}", make_query_arm_fn("constant_velocity", "uniform"))
        run_and_cache(f"trustdelay_no_gate_lag{d}", make_query_arm_fn("learned", "uniform"))
        run_and_cache(f"trustdelay_prior_only_lag{d}", make_query_arm_fn("learned", "prior_only"))
        run_and_cache(f"trustdelay_mlp_only_lag{d}", make_query_arm_fn("learned", "mlp_only"))
        run_and_cache(f"trustdelay_delay_only_lag{d}", make_query_arm_fn("learned", "delay_only"))
        run_and_cache(f"trustdelay_sigma_free_lag{d}", make_query_arm_fn("learned", "reliability_gated", rank_score_only=True, ignore_unc=True))
        run_and_cache(f"trustdelay_full_lag{d}", make_query_arm_fn("learned", "reliability_gated"))

    # -------------------------------------------------------------
    # 3. Table II Ablations @ True Lag 300 ms (m=3), 10% loss, 0.1m/0.1deg
    # -------------------------------------------------------------
    print("\n--- Dumping Table II Ablations (m=3, loss=10%, noise=0.1m/0.1deg) ---")
    dataset.set_delay_ms(300)
    samples_300 = [dataset[i] for i in range(N_frames)]
    seeds = [42, 43, 44]

    ablation_arms = [
        ("full_trustdelay", {}),
        ("wo_extrapolation", {"extrapolation_mode": "none"}),
        ("wo_reliability_fusion", {"fusion_type": "uniform"}),
        ("wo_uncertainty_weighting", {"ignore_uncertainty": True}),
        ("wo_pose_compensation", {"ignore_pose_noise": True}),
        ("wo_packet_loss_handling", {"ignore_packet_mask": True}),
        ("wo_collab_selection", {"disable_collab_selection": True}),
        ("wo_message_compression", {"bandwidth_budget_kb": 100.0})
    ]

    for arm_name, arm_kwargs in ablation_arms:
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            def make_ablation_fn(kwargs):
                def run_abl(sample):
                    with torch.inference_mode():
                        out = model(
                            sample,
                            delay_override_ms=300.0,
                            packet_loss_override=0.10,
                            pose_noise_trans=0.10,
                            pose_noise_rot=0.10,
                            bandwidth_budget_kb=kwargs.get("bandwidth_budget_kb", 8.4),
                            extrapolation_mode=kwargs.get("extrapolation_mode", "learned"),
                            fusion_type=kwargs.get("fusion_type", "reliability_gated"),
                            ignore_uncertainty=kwargs.get("ignore_uncertainty", False),
                            ignore_packet_mask=kwargs.get("ignore_packet_mask", False),
                            ignore_pose_noise=kwargs.get("ignore_pose_noise", False),
                            disable_collab_selection=kwargs.get("disable_collab_selection", False)
                        )
                        p_boxes = out["pred_boxes"][0].cpu().numpy()
                        p_logits = out["pred_logits"][0].cpu().numpy()
                        p_scores = 1.0 / (1.0 + np.exp(-np.clip(p_logits, -10.0, 10.0)))
                        comm_kb = float(out.get("transmitted_kb", 8.4))
                        gate_w = out.get("reliability_weights", torch.empty(0)).cpu().numpy().flatten()
                    return p_boxes, p_scores, gate_w, comm_kb
                return run_abl

            run_and_cache(f"table2_{arm_name}_seed{seed}", make_ablation_fn(arm_kwargs))

    # -------------------------------------------------------------
    # 4. Tight Budgets (E5): 1.0, 5.0, 8.4 KB @ 200 ms true lag (m=2)
    # -------------------------------------------------------------
    print("\n--- Dumping Tight Budget Arms @ Delay 200 ms (1.0, 5.0, 8.4 KB) ---")
    dataset.set_delay_ms(200)
    samples_200 = [dataset[i] for i in range(N_frames)]
    budgets = [1.0, 5.0, 8.4]

    for b_kb in budgets:
        for fuse_type, fuse_name in [("reliability_gated", "gated"), ("uniform", "ungated")]:
            def make_budget_fn(b_val, f_type):
                def run_b(sample):
                    with torch.inference_mode():
                        out = model(
                            sample,
                            delay_override_ms=200.0,
                            packet_loss_override=0.0,
                            pose_noise_trans=0.0,
                            pose_noise_rot=0.0,
                            bandwidth_budget_kb=b_val,
                            extrapolation_mode="learned",
                            fusion_type=f_type
                        )
                        p_boxes = out["pred_boxes"][0].cpu().numpy()
                        p_logits = out["pred_logits"][0].cpu().numpy()
                        p_scores = 1.0 / (1.0 + np.exp(-np.clip(p_logits, -10.0, 10.0)))
                        comm_kb = float(out.get("transmitted_kb", b_val))
                        gate_w = out.get("reliability_weights", torch.empty(0)).cpu().numpy().flatten()
                    return p_boxes, p_scores, gate_w, comm_kb
                return run_b

            run_and_cache(f"budget_{b_kb}kb_{fuse_name}", make_budget_fn(b_kb, fuse_type))

    # -------------------------------------------------------------
    # 5. Erasure Rate Sweep (E6): 0% to 50% loss @ 300 ms true lag
    # -------------------------------------------------------------
    print("\n--- Dumping Erasure Sweep Arms (0% to 50% loss) ---")
    loss_rates = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    dataset.set_delay_ms(300)

    for loss_r in loss_rates:
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            # TrustDelay
            def make_erasure_fn(l_rate, extrap, fuse):
                def run_e(sample):
                    with torch.inference_mode():
                        out = model(
                            sample,
                            delay_override_ms=300.0,
                            packet_loss_override=l_rate,
                            pose_noise_trans=0.0,
                            pose_noise_rot=0.0,
                            bandwidth_budget_kb=8.4,
                            extrapolation_mode=extrap,
                            fusion_type=fuse
                        )
                        p_boxes = out["pred_boxes"][0].cpu().numpy()
                        p_logits = out["pred_logits"][0].cpu().numpy()
                        p_scores = 1.0 / (1.0 + np.exp(-np.clip(p_logits, -10.0, 10.0)))
                        comm_kb = float(out.get("transmitted_kb", 8.4))
                        gate_w = out.get("reliability_weights", torch.empty(0)).cpu().numpy().flatten()
                    return p_boxes, p_scores, gate_w, comm_kb
                return run_e

            run_and_cache(f"erasure_{int(loss_r*100)}pct_trustdelay_seed{seed}", make_erasure_fn(loss_r, "learned", "reliability_gated"))
            run_and_cache(f"erasure_{int(loss_r*100)}pct_const_vel_seed{seed}", make_erasure_fn(loss_r, "constant_velocity", "uniform"))

    total_time = time.time() - t0_all
    print(f"\n=======================================================")
    print(f"Prediction Dumping Complete in {total_time:.1f} seconds ({total_time/60.0:.2f} minutes)!")
    print(f"All raw predictions cached in: {CACHE_DIR}")
    print(f"=======================================================")

if __name__ == "__main__":
    dump_predictions()
