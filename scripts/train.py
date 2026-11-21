"""
TrustDelay: Honest, Leak-Free Training Pipeline.
Trains Ego Encoder, Query Generator, Motion Extrapolator, and Reliability Fusion on OPV2V.
Inputs are purely LiDAR point-cloud BEV representations and SE(3) poses. Zero ground-truth leakage.
"""
from __future__ import annotations
import os
import sys
import time
import json
import argparse
import subprocess
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config import TrustDelayConfig
from src.data.opv2v_dataset import OPV2VAsynchronousDataset
from src.models.trust_delay_model import TrustDelayModel
from src.models.loss import TrustDelayLoss

def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"

def parse_args():
    parser = argparse.ArgumentParser(description="Train TrustDelay cleanly on OPV2V")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_root", type=str, default=None, help="Path to OPV2V dataset root")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    return parser.parse_args()

def train():
    args = parse_args()
    cfg = TrustDelayConfig()
    if args.data_root:
        cfg.data_root = args.data_root
    cfg.seed = args.seed
    cfg.lr = args.lr
    cfg.epochs = args.epochs

    # Set seeds for deterministic training
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    git_commit = get_git_commit()
    print(f"=== Initializing TrustDelay Training (Commit: {git_commit[:8]}) on {device} ===")
    print(f"Dataset root: {cfg.data_root}")

    # Initialize datasets
    train_dataset = OPV2VAsynchronousDataset(cfg, split="train")
    val_dataset = OPV2VAsynchronousDataset(cfg, split="validate")
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Initialize model, loss, optimizer, and scaler
    model = TrustDelayModel(cfg).to(device)
    criterion = TrustDelayLoss(cls_weight=cfg.cls_weight, reg_weight=cfg.reg_weight, unc_weight=cfg.uncertainty_loss_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    os.makedirs("results/checkpoints", exist_ok=True)
    best_val_loss = float("inf")
    metrics_log = {
        "git_commit": git_commit,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "seed": cfg.seed,
            "lr": cfg.lr,
            "epochs": cfg.epochs,
            "num_queries": cfg.num_queries,
            "feat_dim": cfg.feat_dim,
            "data_root": cfg.data_root
        },
        "epochs": []
    }

    print("\nStarting Clean Training Loop...")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_reg = 0.0
        total_unc = 0.0
        start_time = time.time()

        # Shuffle indices for this epoch
        perm = torch.randperm(len(train_dataset))
        optimizer.zero_grad()

        for step, idx in enumerate(perm):
            sample = train_dataset[idx.item()]
            gt_boxes = sample["gt_boxes"].to(device)

            # Sample realistic channel perturbations per training step
            delay_ms = float(np.random.choice([0, 100, 200, 300, 400, 500]))
            p_loss = float(np.random.uniform(0.0, 0.3))
            pose_trans = float(np.random.uniform(0.0, 0.2))
            pose_rot = float(np.random.uniform(0.0, 0.2))

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    pred_dict = model(
                        sample,
                        delay_override_ms=delay_ms,
                        packet_loss_override=p_loss,
                        pose_noise_trans=pose_trans,
                        pose_noise_rot=pose_rot
                    )
                    loss_dict = criterion(pred_dict, gt_boxes)
                    loss = loss_dict["loss_total"] / args.grad_accum

                scaler.scale(loss).backward()
                if (step + 1) % args.grad_accum == 0 or (step + 1) == len(perm):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                pred_dict = model(
                    sample,
                    delay_override_ms=delay_ms,
                    packet_loss_override=p_loss,
                    pose_noise_trans=pose_trans,
                    pose_noise_rot=pose_rot
                )
                loss_dict = criterion(pred_dict, gt_boxes)
                loss = loss_dict["loss_total"] / args.grad_accum
                loss.backward()
                if (step + 1) % args.grad_accum == 0 or (step + 1) == len(perm):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss_dict["loss_total"].item()
            total_cls += loss_dict["loss_cls"].item()
            total_reg += loss_dict["loss_reg"].item()
            total_unc += loss_dict["loss_unc"].item()

        scheduler.step()
        n_samples = len(train_dataset)
        epoch_dur = time.time() - start_time
        avg_train_loss = total_loss / n_samples

        # Full validation across all validation samples
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for v_idx in range(len(val_dataset)):
                v_sample = val_dataset[v_idx]
                v_gt = v_sample["gt_boxes"].to(device)
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        v_pred = model(v_sample, delay_override_ms=200.0)
                        v_loss = criterion(v_pred, v_gt)
                else:
                    v_pred = model(v_sample, delay_override_ms=200.0)
                    v_loss = criterion(v_pred, v_gt)
                val_loss += v_loss["loss_total"].item()

        avg_val_loss = val_loss / len(val_dataset)
        print(f"Epoch [{epoch:02d}/{cfg.epochs:02d}] ({epoch_dur:.1f}s) | Train Loss: {avg_train_loss:.4f} (Cls: {total_cls/n_samples:.3f}, Reg: {total_reg/n_samples:.3f}, Unc: {total_unc/n_samples:.3f}) | Val Loss: {avg_val_loss:.4f}")

        metrics_log["epochs"].append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "cls_loss": total_cls / n_samples,
            "reg_loss": total_reg / n_samples,
            "unc_loss": total_unc / n_samples
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "results/checkpoints/trustdelay_best.pth")
            print(f"  --> Saved new best checkpoint (Val Loss: {best_val_loss:.4f})")

    # Save final model checkpoint and training log
    torch.save(model.state_dict(), "results/checkpoints/trustdelay_final.pth")
    with open("results/training_metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"\nTraining successfully finished! Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()
