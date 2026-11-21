"""
TrustDelay: Central Configuration
Uncertainty-Aware Asynchronous Cooperative Perception under Communication Constraints.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class TrustDelayConfig:
    # --- Experiment Identity ---
    exp_name: str = "trustdelay_v2v_opv2v"
    seed: int = 42
    device: str = "cuda"  # fallback to 'cpu' if cuda unavailable
    
    # --- Dataset Path (Using existing Culver City OPV2V subset without re-downloading) ---
    data_root: str = r"C:\PhD\Collaborative autonomy\coop_v2v\coop_v2v\test_culver_city"
    
    # --- Metric BEV Bounds (Ego frame, in meters) ---
    x_min: float = -50.0
    x_max: float = 50.0
    y_min: float = -50.0
    y_max: float = 50.0
    z_min: float = -3.0
    z_max: float = 3.0
    
    # --- BEV Resolution & Spatial Feature Map ---
    # 100x100 grid covers 100m x 100m at 1.0m/pixel (optimal for RTX 3050 6GB memory & high FPS)
    bev_w: int = 100
    bev_h: int = 100
    feat_dim: int = 64
    num_queries: int = 160
    max_collaborators: int = 4
    
    # --- Latency & Temporal Asynchrony Settings ---
    # Realistic 5G-V2X Delays: 0 to 500 ms (OPV2V frame step is 100 ms)
    delays_ms: List[int] = field(default_factory=lambda: [0, 50, 100, 200, 300, 400, 500])
    default_delay_ms: int = 200
    
    # --- Communication Constraints & Channel Imperfections ---
    packet_loss_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    default_packet_loss: float = 0.0
    
    # Pose noise: translation in meters, rotation in degrees
    pose_noise_trans_levels: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    pose_noise_rot_levels: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    default_pose_trans_std: float = 0.0
    default_pose_rot_std: float = 0.0
    
    # Bandwidth Budgets (in KB per collaborator per frame)
    # 1 KB (8 tokens), 5 KB (40 tokens), 10 KB (80 tokens), 20 KB (160 tokens)
    bandwidth_budgets_kb: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    bytes_per_query: int = 128  # [feat_dim (64xFP16=128B) + box & motion params (quantized/compressed)]
    
    # --- Module 1: Ego Perception Encoder ---
    backbone_type: str = "compact_bev"  # "compact_bev" or "pointpillars"
    use_lidar: bool = True
    use_camera: bool = True
    
    # --- Module 2: Collaborator Selector ---
    top_k_collaborators: int = 2
    selector_weight_perception: float = 1.0
    selector_weight_distance: float = 0.5
    selector_weight_rel_velocity: float = 0.3
    selector_weight_comm_cost: float = 0.4
    
    # --- Module 3: Uncertainty-Aware Motion Extrapolator ---
    extrapolation_mode: str = "learned_uncertainty"  # none | constant_velocity | linear | learned_uncertainty
    uncertainty_loss_weight: float = 0.5
    
    # --- Module 4: Bandwidth-Aware Message Selector ---
    message_compression_mode: str = "query_budget"  # query_budget | topk_spatial | full
    max_transmitted_tokens: int = 32
    
    # --- Module 5: Reliability-Weighted Asynchronous Fusion ---
    fusion_type: str = "reliability_gated"  # uniform | confidence_only | traf_align | lrcp | reliability_gated
    temperature: float = 1.0
    
    # --- Training & Optimization ---
    batch_size: int = 2
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 8
    cls_weight: float = 2.0
    reg_weight: float = 1.5
    extrap_weight: float = 1.0
    
    # --- Detection Evaluation Protocol ---
    iou_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    moving_velocity_thresh: float = 0.5  # m/s to distinguish moving vs stationary vehicles
    
    def validate(self):
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"Config data_root not found: {self.data_root}")
