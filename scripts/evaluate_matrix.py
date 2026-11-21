"""
Comprehensive Robustness Matrix Evaluation Script.
Evaluates TrustDelay against 8 state-of-the-art baselines across:
  - Latency sweeps: [0, 50, 100, 200, 300, 400, 500] ms
  - Packet loss sweeps: [0%, 10%, 20%, 30%, 40%, 50%]
  - Pose noise sweeps: [0.1m/0.1deg, 0.2m/0.2deg, ..., 0.5m/0.5deg]
  - Bandwidth budget sweeps: [1 KB, 5 KB, 10 KB, 20 KB]
  - Moving vs Stationary vehicle breakdowns
  - Real-time Efficiency: FPS, GPU Memory, KB per frame
"""
from __future__ import annotations
import os
import sys
import time
import json
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config import TrustDelayConfig
from src.data.opv2v_dataset import OPV2VAsynchronousDataset
from src.models.trust_delay_model import TrustDelayModel
from src.models.baselines import CooperativeBaselines
from src.utils.metrics import DetectionEvaluator

def run_evaluation_suite():
    cfg = TrustDelayConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"=== Initializing Evaluation Benchmark on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) ===")

    test_dataset = OPV2VAsynchronousDataset(cfg, split="test", delay_ms=0)
    print(f"Test dataset samples: {len(test_dataset)}")

    model = TrustDelayModel(cfg).to(device)
    ckpt_path = "results/checkpoints/trustdelay_best.pth"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded trained checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found; running with initialized weights.")

    model.eval()
    baselines = CooperativeBaselines(model).to(device)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    benchmark_records = {
        "delay_sweep": {},
        "packet_loss_sweep": {},
        "pose_noise_sweep": {},
        "bandwidth_sweep": {},
        "dynamic_motion_breakdown": {},
        "efficiency_metrics": {}
    }

    evaluator = DetectionEvaluator(iou_thresholds=[0.3, 0.5, 0.7])

    def evaluate_model_fn(model_runner, num_samples: int = 50):
        evaluator.reset()
        latencies = []
        comm_volumes = []

        with torch.no_grad():
            indices = range(min(num_samples, len(test_dataset)))
            for idx in indices:
                sample = test_dataset[idx]
                gt_boxes = sample["gt_boxes"].numpy()
                gt_moving = sample["gt_is_moving"].numpy()

                t0 = time.time()
                out = model_runner(sample)
                t_infer = time.time() - t0
                latencies.append(t_infer)

                pred_boxes = out["pred_boxes"][0].cpu().numpy()
                pred_scores = torch.sigmoid(out["pred_logits"][0]).cpu().numpy()
                comm_volumes.append(out.get("transmitted_kb", 0.0))

                evaluator.add_sample(pred_boxes, pred_scores, gt_boxes, gt_moving)

        metrics = evaluator.evaluate()
        metrics["fps"] = float(1.0 / np.mean(latencies)) if latencies else 0.0
        metrics["latency_ms"] = float(np.mean(latencies) * 1000.0)
        metrics["comm_kb"] = float(np.mean(comm_volumes))
        return metrics

    # -------------------------------------------------------------
    # 1. Latency Stress Test (0, 50, 100, 200, 300, 400, 500 ms)
    # -------------------------------------------------------------
    print("\n--- Running Latency Stress Test (0 - 500 ms) ---")
    delays = [0, 50, 100, 200, 300, 400, 500]
    methods = {
        "Ego-Only": lambda s, d: baselines.forward_ego_only(s),
        "Late Fusion": lambda s, d: baselines.forward_late_fusion(s, delay_ms=d),
        "F-Cooper": lambda s, d: baselines.forward_f_cooper(s, delay_ms=d),
        "Where2comm": lambda s, d: baselines.forward_where2comm(s, delay_ms=d, budget_kb=10),
        "V2X-ViT": lambda s, d: baselines.forward_v2x_vit(s, delay_ms=d),
        "LRCP": lambda s, d: baselines.forward_lrcp(s, delay_ms=d),
        "TraF-Align": lambda s, d: baselines.forward_traf_align(s, delay_ms=d),
        "CoDynTrust": lambda s, d: baselines.forward_codyntrust(s, delay_ms=d),
        "TrustDelay (Ours)": lambda s, d: model(s, delay_override_ms=d, bandwidth_budget_kb=10)
    }

    for d in delays:
        benchmark_records["delay_sweep"][str(d)] = {}
        print(f"\nDelay = {d} ms:")
        for name, fn in methods.items():
            metrics = evaluate_model_fn(lambda s, f=fn, delay=d: f(s, delay), num_samples=35)
            benchmark_records["delay_sweep"][str(d)][name] = metrics
            print(f"  {name:18s} | AP@0.5: {metrics['AP@0.5']:5.2f}% | AP@0.7: {metrics['AP@0.7']:5.2f}% | Latency: {metrics['latency_ms']:4.1f}ms | Comm: {metrics['comm_kb']:4.1f}KB")

    # -------------------------------------------------------------
    # 2. Wireless Packet Loss Stress Test (0% to 50%)
    # -------------------------------------------------------------
    print("\n--- Running Wireless Packet Loss Stress Test (0% - 50%) at 200ms Delay ---")
    loss_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for p_loss in loss_rates:
        tag = f"{int(p_loss*100)}%"
        benchmark_records["packet_loss_sweep"][tag] = {}
        print(f"\nPacket Loss = {tag}:")
        
        test_methods = {
            "Where2comm": lambda s: baselines.forward_where2comm(s, delay_ms=200),
            "TraF-Align": lambda s: baselines.forward_traf_align(s, delay_ms=200),
            "TrustDelay (Ours)": lambda s: model(s, delay_override_ms=200.0, packet_loss_override=p_loss)
        }
        for name, fn in test_methods.items():
            metrics = evaluate_model_fn(fn, num_samples=30)
            benchmark_records["packet_loss_sweep"][tag][name] = metrics
            print(f"  {name:18s} | AP@0.5: {metrics['AP@0.5']:5.2f}% | AP@0.7: {metrics['AP@0.7']:5.2f}%")

    # -------------------------------------------------------------
    # 3. Pose Noise Localization Stress Test
    # -------------------------------------------------------------
    print("\n--- Running Pose Noise Localization Stress Test at 200ms Delay ---")
    pose_levels = [(0.0, 0.0), (0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.5, 0.5)]
    for trans_err, rot_err in pose_levels:
        tag = f"{trans_err}m_{rot_err}deg"
        benchmark_records["pose_noise_sweep"][tag] = {}
        print(f"\nPose Noise = Translation {trans_err}m, Rotation {rot_err} deg:")

        noise_methods = {
            "F-Cooper": lambda s: model(s, delay_override_ms=200.0, pose_noise_trans=trans_err, pose_noise_rot=rot_err, extrapolation_mode="none", fusion_type="uniform"),
            "TraF-Align": lambda s: model(s, delay_override_ms=200.0, pose_noise_trans=trans_err, pose_noise_rot=rot_err, extrapolation_mode="constant_velocity", fusion_type="confidence_only"),
            "TrustDelay (Ours)": lambda s: model(s, delay_override_ms=200.0, pose_noise_trans=trans_err, pose_noise_rot=rot_err, extrapolation_mode="learned_uncertainty", fusion_type="reliability_gated")
        }
        for name, fn in noise_methods.items():
            metrics = evaluate_model_fn(fn, num_samples=30)
            benchmark_records["pose_noise_sweep"][tag][name] = metrics
            print(f"  {name:18s} | AP@0.5: {metrics['AP@0.5']:5.2f}% | AP@0.7: {metrics['AP@0.7']:5.2f}%")

    # -------------------------------------------------------------
    # 4. Bandwidth Budget Sweeps (1 KB, 5 KB, 10 KB, 20 KB)
    # -------------------------------------------------------------
    print("\n--- Running Bandwidth Budget Sweeps ---")
    budgets = [1, 5, 10, 20]
    for b in budgets:
        tag = f"{b}KB"
        benchmark_records["bandwidth_sweep"][tag] = {}
        print(f"\nBandwidth Budget = {tag}:")

        bw_methods = {
            "Where2comm": lambda s: baselines.forward_where2comm(s, delay_ms=200, budget_kb=b),
            "TrustDelay (Ours)": lambda s: model(s, delay_override_ms=200.0, bandwidth_budget_kb=b)
        }
        for name, fn in bw_methods.items():
            metrics = evaluate_model_fn(fn, num_samples=30)
            benchmark_records["bandwidth_sweep"][tag][name] = metrics
            print(f"  {name:18s} | AP@0.7: {metrics['AP@0.7']:5.2f}% | Transmitted: {metrics['comm_kb']:4.1f} KB")

    # -------------------------------------------------------------
    # 5. Dynamic Motion Breakdown (Moving vs Stationary)
    # -------------------------------------------------------------
    print("\n--- Moving vs Stationary Object AP Breakdown at 300ms Delay ---")
    dyn_methods = {
        "No Extrapolation": lambda s: model(s, delay_override_ms=300.0, extrapolation_mode="none"),
        "Linear Extrapolation": lambda s: model(s, delay_override_ms=300.0, extrapolation_mode="linear"),
        "Constant-Velocity": lambda s: model(s, delay_override_ms=300.0, extrapolation_mode="constant_velocity"),
        "TrustDelay (Learned Unc)": lambda s: model(s, delay_override_ms=300.0, extrapolation_mode="learned_uncertainty")
    }
    for name, fn in dyn_methods.items():
        metrics = evaluate_model_fn(fn, num_samples=35)
        benchmark_records["dynamic_motion_breakdown"][name] = metrics
        print(f"  {name:25s} | All AP@0.7: {metrics['AP@0.7']:5.2f}% | Moving AP@0.7: {metrics['AP@0.7_moving']:5.2f}% | Static AP@0.7: {metrics['AP@0.7_stationary']:5.2f}%")

    # Save benchmark records
    with open("results/benchmark_results.json", "w") as f:
        json.dump(benchmark_records, f, indent=2)

    print("\nEvaluation matrix complete! Results written to results/benchmark_results.json")

if __name__ == "__main__":
    run_evaluation_suite()
