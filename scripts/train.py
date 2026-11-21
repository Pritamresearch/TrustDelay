"""
TrustDelay Training Pipeline.
Trains Ego Encoder, Motion Extrapolator, and Reliability Fusion on OPV2V under simulated delays.
"""
from __future__ import annotations
import os
import sys
import time
import json
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config import TrustDelayConfig
from src.data.opv2v_dataset import OPV2VAsynchronousDataset
from src.models.trust_delay_model import TrustDelayModel
from src.models.loss import TrustDelayLoss

def train_trustdelay():
    cfg = TrustDelayConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"=== Initializing TrustDelay Training on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) ===")

    # Dataset & Loader
    train_dataset = OPV2VAsynchronousDataset(cfg, split="train", delay_ms=100)
    val_dataset = OPV2VAsynchronousDataset(cfg, split="validate", delay_ms=100)

    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Model & Loss
    model = TrustDelayModel(cfg).to(device)
    criterion = TrustDelayLoss(cls_weight=cfg.cls_weight, reg_weight=cfg.reg_weight, unc_weight=cfg.uncertainty_loss_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    os.makedirs("results/checkpoints", exist_ok=True)
    best_val_loss = float("inf")
    metrics_log = []

    print("\nStarting Training Loop...")
    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_reg = 0.0
        total_unc = 0.0
        start_time = time.time()

        # Randomize delay per epoch to train across heterogeneous delays (0-500 ms)
        epoch_delay = int(torch.randint(0, 6, (1,)).item() * 100)
        train_dataset.set_delay_ms(epoch_delay)

        # Train on 50 samples per epoch for steady gradient updates
        sample_indices = torch.randperm(len(train_dataset))[:min(50, len(train_dataset))]

        for idx in sample_indices:
            sample = train_dataset[idx.item()]
            optimizer.zero_grad()

            gt_boxes = sample["gt_boxes"].to(device)

            pred_dict = model(sample, delay_override_ms=float(epoch_delay))
            loss_dict = criterion(pred_dict, gt_boxes)

            loss = loss_dict["loss_total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            total_cls += loss_dict["loss_cls"].item()
            total_reg += loss_dict["loss_reg"].item()
            total_unc += loss_dict["loss_unc"].item()

        scheduler.step()
        n_samples = len(sample_indices)
        epoch_dur = time.time() - start_time
        avg_loss = total_loss / n_samples

        # Quick validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_indices = torch.arange(min(20, len(val_dataset)))
            for v_idx in val_indices:
                v_sample = val_dataset[v_idx.item()]
                v_gt = v_sample["gt_boxes"].to(device)
                v_pred = model(v_sample, delay_override_ms=200.0)
                v_loss = criterion(v_pred, v_gt)
                val_loss += v_loss["loss_total"].item()
            avg_val_loss = val_loss / len(val_indices)

        print(f"Epoch [{epoch:02d}/{cfg.epochs:02d}] ({epoch_dur:.1f}s) | Train Loss: {avg_loss:.4f} (Cls: {total_cls/n_samples:.3f}, Reg: {total_reg/n_samples:.3f}, Unc: {total_unc/n_samples:.3f}) | Val Loss: {avg_val_loss:.4f}")

        metrics_log.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_loss": avg_val_loss,
            "epoch_delay_ms": epoch_delay
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "results/checkpoints/trustdelay_best.pth")

    # Save training history
    with open("results/training_metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

    # Save final model
    torch.save(model.state_dict(), "results/checkpoints/trustdelay_final.pth")
    print(f"\nTraining complete! Checkpoints saved in results/checkpoints/")

if __name__ == "__main__":
    train_trustdelay()
