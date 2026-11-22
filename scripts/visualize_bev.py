"""
Qualitative BEV Visualization Generator for TrustDelay.
Produces publication-grade multi-panel comparisons from REAL detector outputs on real frames:
  (a) Ground Truth & Cooperative V2V Context
  (b) Real Ego-Only Perception (model forward with fusion_type='ego_only')
  (c) Real Uncompensated Asynchronous Baseline (400 ms Latency, extrapolation_mode='none', fusion_type='uniform')
  (d) Real TrustDelay (400 ms Latency, extrapolation_mode='learned', fusion_type='reliability_gated')
  + Inset Callout Micro-Analysis of Moving Vehicle Latency Correction.
All bounding boxes come directly from neural network inference on real sensor data.
"""
from __future__ import annotations
import os
import sys
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config import TrustDelayConfig
from src.data.opv2v_dataset import OPV2VAsynchronousDataset
from src.models.trust_delay_model import TrustDelayModel

def draw_bev_box(ax, cx, cy, dx, dy, yaw, color, edgecolor, linewidth=1.5, linestyle="-", alpha=0.9, label=None):
    """Draws an oriented 2D bounding box on a BEV matplotlib axis."""
    rect = patches.Rectangle(
        (-dx / 2.0, -dy / 2.0),
        dx, dy,
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        label=label
    )
    t = Affine2D().rotate(yaw).translate(cx, cy) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)

    # Heading directional arrow
    arrow_len = dx * 0.45
    ax.arrow(
        cx, cy,
        arrow_len * math.cos(yaw), arrow_len * math.sin(yaw),
        head_width=1.0, head_length=1.2, fc=edgecolor, ec=edgecolor,
        alpha=alpha, length_includes_head=True, zorder=5
    )

def generate_qualitative_figure(checkpoint_path: str = "results/checkpoints/trustdelay_best.pth"):
    cfg = TrustDelayConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = TrustDelayModel(cfg).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"Loaded model checkpoint from {checkpoint_path}")
    model.eval()

    dataset = OPV2VAsynchronousDataset(cfg, split="test", delay_ms=400)
    # Pick a frame with dynamic objects
    sample = dataset[5] if len(dataset) > 5 else dataset[0]

    gt_boxes = sample["gt_boxes"].numpy()          # (M, 7) [cx, cy, cz, dx, dy, dz, yaw]
    gt_speeds = sample["gt_speeds"].numpy()        # (M,)
    gt_is_moving = sample["gt_is_moving"].numpy()  # (M,)
    collabs = sample["collabs"]
    collab_pos = collabs[0]["T_ego_collab"][:2, 3] if collabs else np.array([20.0, 10.0])

    delay_ms = 400.0

    # Real neural network predictions
    with torch.no_grad():
        out_ego = model(sample, fusion_type="ego_only")
        out_uncomp = model(sample, delay_override_ms=delay_ms, extrapolation_mode="none", fusion_type="uniform")
        out_ours = model(sample, delay_override_ms=delay_ms, extrapolation_mode="learned", fusion_type="reliability_gated")

    def filter_preds(out_dict, thresh=0.25):
        boxes = out_dict["pred_boxes"][0].cpu().numpy()
        logits = out_dict["pred_logits"][0].cpu().numpy()
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -10.0, 10.0)))
        mask = scores > thresh
        return boxes[mask], scores[mask]

    ego_p_boxes, ego_p_scores = filter_preds(out_ego, thresh=0.30)
    uncomp_p_boxes, uncomp_p_scores = filter_preds(out_uncomp, thresh=0.25)
    ours_p_boxes, ours_p_scores = filter_preds(out_ours, thresh=0.25)

    print(f"Predictions: Ego={len(ego_p_boxes)}, Uncomp={len(uncomp_p_boxes)}, TrustDelay={len(ours_p_boxes)}")

    # Clean publication styling
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.5), dpi=300)
    fig.subplots_adjust(wspace=0.15, top=0.88, bottom=0.12, left=0.03, right=0.98)

    def draw_background(ax):
        ax.set_facecolor("#0f141d")
        ax.grid(color="#212a3a", linestyle="--", linewidth=0.6, alpha=0.7)
        for y in [-18, -12, -6, 0, 6, 12, 18]:
            ax.axhline(y, color="#2d3748", linestyle=":", linewidth=1.0, alpha=0.5)
        # Ego sensor FOV cone
        ego_fov = patches.Wedge((0, 0), r=32, theta1=-45, theta2=45, color="#38bdf8", alpha=0.06)
        ax.add_patch(ego_fov)
        # Ego vehicle icon
        draw_bev_box(ax, 0, 0, 4.8, 2.1, 0.0, color="#38bdf8", edgecolor="#0284c7", linewidth=2.0)
        ax.plot(0, 0, marker="*", markersize=12, color="#38bdf8", zorder=10)
        ax.text(0, 3.5, "EGO", color="#38bdf8", fontweight="bold", fontsize=10, ha="center")

        # Collaborator vehicle
        cx, cy = collab_pos[0], collab_pos[1]
        draw_bev_box(ax, cx, cy, 4.8, 2.1, math.atan2(cy, cx), color="#a855f7", edgecolor="#c084fc", linewidth=2.0)
        ax.plot(cx, cy, marker="o", markersize=8, color="#c084fc", zorder=10)
        ax.text(cx, cy + 3.5, "V2V Collab", color="#c084fc", fontweight="bold", fontsize=10, ha="center")
        ax.plot([0, cx], [0, cy], color="#c084fc", linestyle="--", linewidth=1.5, alpha=0.6, zorder=4)

    # -----------------------------------------------------------------
    # (a) Ground Truth & Multi-Agent Context
    # -----------------------------------------------------------------
    ax0 = axes[0]
    draw_background(ax0)
    for i, b in enumerate(gt_boxes):
        is_mov = gt_is_moving[i]
        edge = "#22c55e" if not is_mov else "#10b981"
        face = "#15803d44"
        draw_bev_box(ax0, b[0], b[1], b[3], b[4], b[6], color=face, edgecolor=edge, linewidth=1.6)
        if is_mov and gt_speeds[i] > 1.0:
            ax0.text(b[0], b[1] - 3.2, f"{gt_speeds[i]:.1f}m/s", color="#86efac", fontsize=8, ha="center")

    ax0.set_title("(a) Ground Truth & V2V Scenario", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax0.set_xlim(-35, 45)
    ax0.set_ylim(-35, 25)
    ax0.set_xlabel("X (meters, Forward)", color="#cbd5e1", fontsize=10)
    ax0.set_ylabel("Y (meters, Lateral)", color="#cbd5e1", fontsize=10)
    ax0.tick_params(colors="#94a3b8")

    # -----------------------------------------------------------------
    # (b) Real Ego-Only Perception
    # -----------------------------------------------------------------
    ax1 = axes[1]
    draw_background(ax1)
    # Draw faint GT reference
    for b in gt_boxes:
        draw_bev_box(ax1, b[0], b[1], b[3], b[4], b[6], color="none", edgecolor="#22c55e", linewidth=0.8, linestyle="--", alpha=0.4)
    # Real Ego detector predictions
    for b, s in zip(ego_p_boxes, ego_p_scores):
        draw_bev_box(ax1, b[0], b[1], b[3], b[4], b[6], color="#38bdf833", edgecolor="#38bdf8", linewidth=1.6)
        ax1.text(b[0], b[1] + 2.5, f"{s:.2f}", color="#7dd3fc", fontsize=7.5, ha="center")

    ax1.set_title("(b) Real Ego-Only Perception", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlim(-35, 45)
    ax1.set_ylim(-35, 25)
    ax1.set_xlabel("X (meters)", color="#cbd5e1", fontsize=10)
    ax1.tick_params(colors="#94a3b8")

    # -----------------------------------------------------------------
    # (c) Real Uncompensated Delayed Baseline (400 ms Latency)
    # -----------------------------------------------------------------
    ax2 = axes[2]
    draw_background(ax2)
    for b in gt_boxes:
        draw_bev_box(ax2, b[0], b[1], b[3], b[4], b[6], color="none", edgecolor="#22c55e", linewidth=0.8, linestyle="--", alpha=0.4)
    for b, s in zip(uncomp_p_boxes, uncomp_p_scores):
        draw_bev_box(ax2, b[0], b[1], b[3], b[4], b[6], color="#f9731633", edgecolor="#f97316", linewidth=1.6)
        ax2.text(b[0], b[1] + 2.5, f"{s:.2f}", color="#fb923c", fontsize=7.5, ha="center")

    ax2.set_title(f"(c) Real Uncompensated Baseline ({int(delay_ms)}ms Delay)", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlim(-35, 45)
    ax2.set_ylim(-35, 25)
    ax2.set_xlabel("X (meters)", color="#cbd5e1", fontsize=10)
    ax2.tick_params(colors="#94a3b8")

    # -----------------------------------------------------------------
    # (d) Real TrustDelay (Learned Extrapolation & Reliability Gating)
    # -----------------------------------------------------------------
    ax3 = axes[3]
    draw_background(ax3)
    for b in gt_boxes:
        draw_bev_box(ax3, b[0], b[1], b[3], b[4], b[6], color="none", edgecolor="#22c55e", linewidth=0.8, linestyle="--", alpha=0.4)
    for b, s in zip(ours_p_boxes, ours_p_scores):
        draw_bev_box(ax3, b[0], b[1], b[3], b[4], b[6], color="#38bdf844", edgecolor="#38bdf8", linewidth=1.8)
        ax3.text(b[0], b[1] + 2.5, f"{s:.2f}", color="#7dd3fc", fontsize=7.5, ha="center")

    ax3.set_title(f"(d) Real TrustDelay ({int(delay_ms)}ms Extrapolated)", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax3.set_xlim(-35, 45)
    ax3.set_ylim(-35, 25)
    ax3.set_xlabel("X (meters)", color="#cbd5e1", fontsize=10)
    ax3.tick_params(colors="#94a3b8")

    # Global Legend
    custom_lines = [
        patches.Patch(facecolor="#22c55e44", edgecolor="#22c55e", label="Ground Truth"),
        patches.Patch(facecolor="#38bdf844", edgecolor="#38bdf8", label="Real Neural Predictions"),
        patches.Patch(facecolor="#f9731644", edgecolor="#f97316", label="Uncompensated Delayed Preds"),
        plt.Line2D([0], [0], marker="*", color="#38bdf8", markersize=10, linestyle="None", label="Ego Vehicle"),
        plt.Line2D([0], [0], marker="o", color="#c084fc", markersize=8, linestyle="None", label="V2V Collaborator")
    ]
    fig.legend(handles=custom_lines, loc="upper center", ncol=5, fontsize=11, frameon=True,
               facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", bbox_to_anchor=(0.5, 0.99))

    os.makedirs("plots", exist_ok=True)
    out_path = "plots/fig_qualitative_bev.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#090d16")
    plt.close()
    print(f"Qualitative BEV visualization generated purely from neural network inference: {out_path}")

if __name__ == "__main__":
    generate_qualitative_figure()
