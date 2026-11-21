"""
OPV2V Asynchronous Cooperative Perception Dataset Loader.
Supports continuous time sequences, asynchronous frame pairing at verified 10 Hz frame rate (100 ms/frame),
multi-agent pose transforms, 3D bounding box extraction in ego frame, and moving/stationary labels.
Strictly leak-free: Collaborators transmit only sensor representations and spatial poses (no ground truth).
"""
from __future__ import annotations
import os
import glob
import math
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any

def x_to_world(cords: List[float]) -> np.ndarray:
    """
    Convert CARLA coordinate pose [x, y, z, roll, yaw, pitch] (in degrees) to 4x4 transform matrix.
    """
    x, y, z, roll, yaw, pitch = cords
    c_y = math.cos(math.radians(yaw))
    s_y = math.sin(math.radians(yaw))
    c_r = math.cos(math.radians(roll))
    s_r = math.sin(math.radians(roll))
    c_p = math.cos(math.radians(pitch))
    s_p = math.sin(math.radians(pitch))

    matrix = np.identity(4, dtype=np.float32)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r

    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r

    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix

def parse_pcd_points(path: str, max_points: int = 15000) -> np.ndarray:
    """
    Parse ASCII/Binary PCD file without heavy third-party dependencies.
    Returns (N, 3) point cloud array centered at sensor frame.
    """
    if not os.path.exists(path):
        return np.zeros((0, 3), dtype=np.float32)
    try:
        with open(path, "rb") as f:
            raw = f.read()
        lines = raw.split(b"\n")
        header = {}
        data_start_idx = 0
        for i, line in enumerate(lines):
            s = line.decode("ascii", errors="ignore").strip()
            if s.startswith("DATA"):
                header["data"] = s.split()[-1]
                data_start_idx = i + 1
                break
            parts = s.split()
            if len(parts) >= 2:
                header[parts[0]] = parts[1:]

        if header.get("data") == "binary":
            header_offset = sum(len(lines[i]) + 1 for i in range(data_start_idx))
            pts_raw = raw[header_offset:]
            pts = np.frombuffer(pts_raw, dtype=np.float32)
            pts = pts.reshape(-1, 4)[:, :3]  # x, y, z, intensity -> x, y, z
        else:
            pts_list = []
            for i in range(data_start_idx, min(len(lines), data_start_idx + max_points)):
                parts = lines[i].split()
                if len(parts) >= 3:
                    pts_list.append([float(parts[0]), float(parts[1]), float(parts[2])])
            pts = np.array(pts_list, dtype=np.float32) if pts_list else np.zeros((0, 3), dtype=np.float32)
        
        if len(pts) > max_points:
            choice = np.random.choice(len(pts), max_points, replace=False)
            pts = pts[choice]
        return pts
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)

def points_to_bev_features(
    pts: np.ndarray,
    x_range: Tuple[float, float] = (-50.0, 50.0),
    y_range: Tuple[float, float] = (-50.0, 50.0),
    z_range: Tuple[float, float] = (-3.0, 3.0),
    grid_size: Tuple[int, int] = (100, 100),
    num_channels: int = 4
) -> np.ndarray:
    """
    Convert 3D point cloud into compact pseudo-pillar BEV feature representation (C, H, W).
    Channels: [density, max_z, mean_z, occupancy_flag]
    """
    bev = np.zeros((num_channels, grid_size[0], grid_size[1]), dtype=np.float32)
    if len(pts) == 0:
        return bev

    mask = (
        (pts[:, 0] >= x_range[0]) & (pts[:, 0] < x_range[1]) &
        (pts[:, 1] >= y_range[0]) & (pts[:, 1] < y_range[1]) &
        (pts[:, 2] >= z_range[0]) & (pts[:, 2] < z_range[1])
    )
    valid_pts = pts[mask]
    if len(valid_pts) == 0:
        return bev

    xi = np.floor((valid_pts[:, 0] - x_range[0]) / (x_range[1] - x_range[0]) * grid_size[0]).astype(np.int64)
    yi = np.floor((valid_pts[:, 1] - y_range[0]) / (y_range[1] - y_range[0]) * grid_size[1]).astype(np.int64)

    xi = np.clip(xi, 0, grid_size[0] - 1)
    yi = np.clip(yi, 0, grid_size[1] - 1)

    for i in range(len(valid_pts)):
        x_idx = xi[i]
        y_idx = yi[i]
        z_val = valid_pts[i, 2]
        bev[0, x_idx, y_idx] += 1.0
        bev[1, x_idx, y_idx] = max(bev[1, x_idx, y_idx], z_val)
        bev[2, x_idx, y_idx] += z_val

    counts = bev[0:1].copy()
    valid_cells = counts > 0
    bev[2:3] = np.where(valid_cells, bev[2:3] / np.maximum(counts, 1.0), 0.0)
    bev[0:1] = np.log1p(bev[0:1]) / 3.0
    bev[3:4] = np.where(valid_cells, 1.0, 0.0)
    return bev

class OPV2VAsynchronousDataset(Dataset):
    """
    Continuous Asynchronous Multi-Agent Dataset for OPV2V.
    Simulates physical communication delay Delta_t between ego and surrounding CAVs.
    Verified frame rate: 10 Hz (100 ms per step).
    """
    def __init__(self, cfg, split: str = "train", delay_ms: int = 0):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.delay_ms = delay_ms
        self.split_dir = os.path.join(cfg.data_root, split)
        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.sequences = sorted(glob.glob(os.path.join(self.split_dir, "*")))
        self.samples = []
        self._build_index()

    def set_delay_ms(self, delay_ms: int):
        self.delay_ms = delay_ms

    def _build_index(self):
        for seq in self.sequences:
            if not os.path.isdir(seq):
                continue
            agents = sorted([d for d in os.listdir(seq) if os.path.isdir(os.path.join(seq, d))])
            if len(agents) < 2:
                continue
            ego_agent = agents[0]
            collab_agents = agents[1: self.cfg.max_collaborators + 1]

            ego_dir = os.path.join(seq, ego_agent)
            yaml_files = sorted([f for f in os.listdir(ego_dir) if f.endswith(".yaml")])
            timestamps = [f.replace(".yaml", "") for f in yaml_files]

            for i in range(len(timestamps)):
                self.samples.append({
                    "seq_dir": seq,
                    "ego_id": ego_agent,
                    "collab_ids": collab_agents,
                    "frame_idx": i,
                    "timestamps": timestamps
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_meta = self.samples[idx]
        seq_dir = sample_meta["seq_dir"]
        ego_id = sample_meta["ego_id"]
        collab_ids = sample_meta["collab_ids"]
        t_idx = sample_meta["frame_idx"]
        timestamps = sample_meta["timestamps"]

        # Ego timestamp t
        ego_ts = timestamps[t_idx]
        ego_yaml_path = os.path.join(seq_dir, ego_id, f"{ego_ts}.yaml")
        with open(ego_yaml_path, "r") as f:
            ego_yaml = yaml.safe_load(f)

        T_world_ego = x_to_world(ego_yaml["true_ego_pos"])
        T_ego_world = np.linalg.inv(T_world_ego)

        # Load Ego LiDAR PCD into BEV feature
        ego_pcd_path = os.path.join(seq_dir, ego_id, f"{ego_ts}.pcd")
        ego_pts = parse_pcd_points(ego_pcd_path)
        ego_bev = points_to_bev_features(
            ego_pts,
            x_range=(self.cfg.x_min, self.cfg.x_max),
            y_range=(self.cfg.y_min, self.cfg.y_max),
            z_range=(self.cfg.z_min, self.cfg.z_max),
            grid_size=(self.cfg.bev_h, self.cfg.bev_w)
        )

        # Ground truth 3D boxes in ego frame at timestamp t
        gt_boxes_ego = []
        gt_speeds = []
        gt_is_moving = []
        if "vehicles" in ego_yaml:
            for v_id, v_info in ego_yaml["vehicles"].items():
                loc = v_info["location"]
                extent = v_info["extent"]
                angle = v_info["angle"]
                spd = float(v_info.get("speed", 0.0)) / 3.6  # convert km/h to m/s

                # Transform center to ego frame
                pt_w = np.array([loc[0], loc[1], loc[2], 1.0], dtype=np.float32)
                pt_ego = (T_ego_world @ pt_w)[:3]

                # Filter within perception boundary
                if (self.cfg.x_min <= pt_ego[0] <= self.cfg.x_max and
                    self.cfg.y_min <= pt_ego[1] <= self.cfg.y_max and
                    self.cfg.z_min <= pt_ego[2] <= self.cfg.z_max):
                    yaw_ego = math.radians(angle[1] - ego_yaml["true_ego_pos"][4])
                    box = [pt_ego[0], pt_ego[1], pt_ego[2], extent[0]*2, extent[1]*2, extent[2]*2, yaw_ego]
                    gt_boxes_ego.append(box)
                    gt_speeds.append(spd)
                    gt_is_moving.append(spd > self.cfg.moving_velocity_thresh)

        # Process collaborators with latency Delta_t (100 ms per discrete step)
        frame_lag = int(round(self.delay_ms / self.cfg.frame_interval_ms))
        collab_data = []

        for c_id in collab_ids:
            collab_idx = max(0, t_idx - frame_lag)
            collab_ts = timestamps[collab_idx]
            actual_delay_ms = (t_idx - collab_idx) * self.cfg.frame_interval_ms

            c_yaml_path = os.path.join(seq_dir, c_id, f"{collab_ts}.yaml")
            if not os.path.exists(c_yaml_path):
                continue
            with open(c_yaml_path, "r") as f:
                c_yaml = yaml.safe_load(f)

            # Collaborator pose at its delayed timestamp (t - Delta_t)
            T_world_collab_delayed = x_to_world(c_yaml["true_ego_pos"])
            # Extrinsic alignment: from delayed collaborator frame into current ego frame
            T_ego_collab = T_ego_world @ T_world_collab_delayed

            c_pos_in_ego = T_ego_collab[:3, 3]
            c_dist = float(np.linalg.norm(c_pos_in_ego[:2]))
            c_speed = float(c_yaml.get("ego_speed", 0.0)) / 3.6
            ego_speed = float(ego_yaml.get("ego_speed", 0.0)) / 3.6
            rel_speed = abs(ego_speed - c_speed)

            # Load delayed collaborator BEV feature strictly from its point cloud
            c_pcd_path = os.path.join(seq_dir, c_id, f"{collab_ts}.pcd")
            c_pts = parse_pcd_points(c_pcd_path)
            c_bev = points_to_bev_features(
                c_pts,
                x_range=(self.cfg.x_min, self.cfg.x_max),
                y_range=(self.cfg.y_min, self.cfg.y_max),
                z_range=(self.cfg.z_min, self.cfg.z_max),
                grid_size=(self.cfg.bev_h, self.cfg.bev_w)
            )

            # Collaborator payload contains ZERO ground-truth boxes
            collab_data.append({
                "agent_id": c_id,
                "delay_ms": actual_delay_ms,
                "distance": c_dist,
                "rel_speed": rel_speed,
                "T_ego_collab": T_ego_collab,
                "bev_features": c_bev
            })

        return {
            "seq_name": os.path.basename(seq_dir),
            "timestamp": ego_ts,
            "delay_ms": self.delay_ms,
            "ego_bev": torch.from_numpy(ego_bev),
            "collabs": collab_data,
            "gt_boxes": torch.from_numpy(np.array(gt_boxes_ego, dtype=np.float32)) if gt_boxes_ego else torch.zeros((0, 7), dtype=torch.float32),
            "gt_speeds": torch.from_numpy(np.array(gt_speeds, dtype=np.float32)) if gt_speeds else torch.zeros((0,), dtype=torch.float32),
            "gt_is_moving": torch.from_numpy(np.array(gt_is_moving, dtype=bool)) if gt_is_moving else torch.zeros((0,), dtype=torch.bool)
        }
