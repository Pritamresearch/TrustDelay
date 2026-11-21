"""
TrustDelay: Central Configuration.
Uncertainty-Aware Asynchronous Cooperative Perception under Communication Constraints.
"""
from __future__ import annotations
import os
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class TrustDelayConfig:
    # --- Experiment Identity ---
    exp_name: str = "trustdelay_real_eval"
    seed: int = 42
    device: str = "cuda"
    
    # --- Dataset Path (Configurable via environment variable or CLI) ---
    data_root: str = os.environ.get(
        "OPV2V_DATA_ROOT",
        r"C:\PhD\Collaborative autonomy\coop_v2v\coop_v2v\test_culver_city"
    )
    
    # --- Metric BEV Bounds (Standard OPV2V coordinate frame in meters) ---
    x_min: float = -50.0
    x_max: float = 50.0
    y_min: float = -50.0
    y_max: float = 50.0
    z_min: float = -3.0
    z_max: float = 3.0
    
    # --- BEV Resolution & Spatial Feature Map ---
    # 100x100 grid covers 100m x 100m at 1.0m/cell (fits RTX 3050 6GB memory & high FPS)
    bev_w: int = 100
    bev_h: int = 100
    feat_dim: int = 64
    num_queries: int = 120
    max_collaborators: int = 4
    
    # --- Latency & Temporal Asynchrony Settings (10 Hz = 100 ms per discrete frame lag) ---
    frame_interval_ms: float = 100.0
    delays_ms: List[int] = field(default_factory=lambda: [0, 100, 200, 300, 400, 500])
    default_delay_ms: int = 200
    
    # --- Communication Constraints & Channel Imperfections ---
    packet_loss_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    default_packet_loss: float = 0.0
    
    # Pose noise: translation in meters, rotation in degrees
    pose_noise_trans_levels: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5])
    pose_noise_rot_levels: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5])
    default_pose_trans_std: float = 0.0
    default_pose_rot_std: float = 0.0
    
    # --- Bandwidth Constraints ---
    bandwidth_budgets_kb: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    default_bandwidth_budget_kb: int = 10
    bytes_per_query: int = 128  # 64 FP16 features = 128 bytes
    
    # --- Collaborator Selection Parameters ---
    top_k_collaborators: int = 2
    selector_weight_perception: float = 0.35
    selector_weight_distance: float = 0.25
    selector_weight_rel_velocity: float = 0.20
    selector_weight_comm_cost: float = 0.20
    
    # --- Training Hyperparameters ---
    lr: float = 2e-4
    weight_decay: float = 1e-4
    batch_size: int = 2
    epochs: int = 30
    grad_accum_steps: int = 2
    cls_weight: float = 2.0
    reg_weight: float = 1.0
    uncertainty_loss_weight: float = 0.5
    
    # --- Kinematic Extrapolation & Dynamic Thresholds ---
    extrapolation_mode: str = "learned_uncertainty"
    moving_velocity_thresh: float = 0.5  # m/s
    
    # --- Evaluation Parameters ---
    eval_iou_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    perturbation_seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
