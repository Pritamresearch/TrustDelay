"""
Communication & Physical Channel Perturbations Suite.
Simulates:
  - Heterogeneous Latency (0 to 500 ms)
  - Wireless Packet Loss (0% to 50%)
  - GPS/IMU Pose Misalignment (translation 0.1-0.5 m, rotation 0.1-0.5 deg)
  - Bandwidth Capacity Constraints (1 KB, 5 KB, 10 KB, 20 KB)
"""
from __future__ import annotations
import math
import numpy as np
from typing import Dict, Any, List

class CommunicationChannelSimulator:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def apply_pose_noise(
        self,
        T_ego_collab: np.ndarray,
        trans_std: float = 0.0,
        rot_std_deg: float = 0.0
    ) -> Tuple[np.ndarray, float]:
        """
        Adds Gaussian noise to relative pose calibration matrix.
        Returns perturbed matrix and total perturbation magnitude.
        """
        if trans_std <= 0.0 and rot_std_deg <= 0.0:
            return T_ego_collab.copy(), 0.0

        noisy_T = T_ego_collab.copy()
        t_noise = np.random.normal(0.0, trans_std, size=(3,))
        r_noise_deg = np.random.normal(0.0, rot_std_deg)
        r_noise_rad = np.radians(r_noise_deg)

        noisy_T[0, 3] += t_noise[0]
        noisy_T[1, 3] += t_noise[1]
        noisy_T[2, 3] += t_noise[2]

        c_yaw = math.cos(r_noise_rad)
        s_yaw = math.sin(r_noise_rad)
        R_noise = np.array([
            [c_yaw, -s_yaw, 0],
            [s_yaw,  c_yaw, 0],
            [0,      0,     1]
        ], dtype=np.float32)

        noisy_T[:3, :3] = R_noise @ noisy_T[:3, :3]
        total_mag = float(np.linalg.norm(t_noise) + abs(r_noise_rad))
        return noisy_T, total_mag

    def sample_delay_ms(self, base_delay_ms: float, jitter_ms: float = 20.0) -> float:
        """
        Models communication delay with stochastic wireless jitter.
        """
        if base_delay_ms <= 0:
            return 0.0
        jitter = np.random.normal(0.0, jitter_ms)
        return max(0.0, base_delay_ms + jitter)
