"""
TrustDelay: Honest, Leak-Free Training Pipeline with Live Terminal Progress and Resumption Support.
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
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="Train TrustDelay cleanly on OPV2V with live terminal progress")
    parser.add_argument("--epochs", type=int, default=40, help="Total number of epochs to train up to")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_root", type=str, default=None, help="Path to OPV2V dataset root")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pth to resume training from")
    parser.add_argument("--save_freq", type=int, default=5, help="Save periodic checkpoint every N epochs")
    return parser.parse_args()

def train():
    args = parse_args()
    cfg = TrustDelayConfig()
    if args.data_root:
        cfg.data_root = args.data_root
    cfg.seed = args.seed
    cfg.lr = args.lr

    # Set seeds for deterministic training
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    git_commit = get_git_commit()

    print(f"\n{'='*70}")
    print(f" TrustDelay Real Deep Training Pipeline (Commit: {git_commit[:8]})")
    print(f" Device: {device} | Total Target Epochs: {args.epochs} | LR: {args.lr}")
    print(f"{'='*70}\n", flush=True)

    # 1. Initialize datasets with in-memory caching
    print("Loading datasets...", flush=True)
    train_dataset = OPV2VAsynchronousDataset(cfg, split="train")
    val_dataset = OPV2VAsynchronousDataset(cfg, split="validate")

    print(f"Preloading train ({len(train_dataset)} frames) and val ({len(val_dataset)} frames) into RAM cache...", flush=True)
    for i in tqdm(range(len(train_dataset)), desc="Caching Train BEVs", ncols=85):
        _ = train_dataset[i]
    for i in tqdm(range(len(val_dataset)), desc="Caching Val BEVs", ncols=85):
        _ = val_dataset[i]
    print(f"Cache complete: {len(train_dataset.bev_cache)} train BEVs, {len(val_dataset.bev_cache)} val BEVs in memory.\n", flush=True)

    # 2. Initialize Model, Loss, Optimizer, Scheduler, Scaler
    model = TrustDelayModel(cfg).to(device)
    criterion = TrustDelayLoss(cls_weight=cfg.cls_weight, reg_weight=cfg.reg_weight, unc_weight=cfg.uncertainty_loss_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    os.makedirs("results/checkpoints", exist_ok=True)
    start_epoch = 1
    best_val_loss = float("inf")
    metrics_log = {
        "git_commit": git_commit,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "seed": cfg.seed,
            "lr": args.lr,
            "target_epochs": args.epochs,
            "num_queries": cfg.num_queries,
            "feat_dim": cfg.feat_dim,
            "data_root": cfg.data_root
        },
        "epochs": []
    }

    # 3. Handle Resuming from Checkpoint
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint to resume from not found: {args.resume}")
        print(f"--> Resuming training from checkpoint: {args.resume}", flush=True)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if scaler and ckpt.get("scaler_state_dict") is not None:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            if "metrics_log" in ckpt and isinstance(ckpt["metrics_log"], dict):
                metrics_log = ckpt["metrics_log"]
            print(f"  Successfully resumed from Epoch {start_epoch - 1} (Best Val Loss: {best_val_loss:.4f})", flush=True)
        else:
            # Plain state_dict loaded (e.g. from previous run)
            model.load_state_dict(ckpt)
            if os.path.exists("results/training_metrics.json"):
                try:
                    with open("results/training_metrics.json", "r") as f:
                        old_log = json.load(f)
                    if "epochs" in old_log and len(old_log["epochs"]) > 0:
                        start_epoch = old_log["epochs"][-1]["epoch"] + 1
                        metrics_log = old_log
                        best_val_loss = min(e["val_loss"] for e in old_log["epochs"])
                        print(f"  Detected previous training log: Resuming seamlessly from Epoch {start_epoch} (Previous Best Val Loss: {best_val_loss:.4f})", flush=True)
                except Exception:
                    pass
            if start_epoch == 1:
                print("  Loaded raw model state_dict. Initializing epoch counter at 1.", flush=True)

        if start_epoch > args.epochs:
            print(f"\n[NOTICE] Checkpoint already trained through Epoch {start_epoch - 1}. Target was {args.epochs}.")
            print(f"To train further, specify e.g. --epochs {start_epoch - 1 + 20}\n")
            return

    # 4. Training Loop
    print(f"Starting Training from Epoch [{start_epoch:02d}/{args.epochs:02d}]...\n", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_reg = 0.0
        total_unc = 0.0
        start_time = time.time()

        perm = torch.randperm(len(train_dataset))
        optimizer.zero_grad()

        pbar = tqdm(
            enumerate(perm),
            total=len(perm),
            desc=f"Epoch [{epoch:02d}/{args.epochs:02d}]",
            ncols=110,
            leave=True
        )

        for step, idx in pbar:
            sample = train_dataset[idx.item()]
            gt_boxes = sample["gt_boxes"].to(device)

            # Realistic random channel perturbations per step
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

            l_tot = loss_dict["loss_total"].item()
            l_cls = loss_dict["loss_cls"].item()
            l_reg = loss_dict["loss_reg"].item()
            l_unc = loss_dict["loss_unc"].item()

            total_loss += l_tot
            total_cls += l_cls
            total_reg += l_reg
            total_unc += l_unc

            # Live update progress bar postfix
            pbar.set_postfix({
                "loss": f"{l_tot:.3f}",
                "reg": f"{l_reg:.3f}",
                "unc": f"{l_unc:.3f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.1e}"
            })

        scheduler.step()
        n_samples = len(train_dataset)
        epoch_dur = time.time() - start_time
        avg_train_loss = total_loss / n_samples

        # Validation across all validation samples
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            val_pbar = tqdm(range(len(val_dataset)), desc=f"  Val [{epoch:02d}]", ncols=85, leave=False)
            for v_idx in val_pbar:
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

        epoch_summary = (
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_dur:.1f}s) | "
            f"Train: {avg_train_loss:.4f} (Cls: {total_cls/n_samples:.3f}, Reg: {total_reg/n_samples:.3f}, Unc: {total_unc/n_samples:.3f}) | "
            f"Val: {avg_val_loss:.4f}"
        )

        metrics_log["epochs"].append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "cls_loss": total_cls / n_samples,
            "reg_loss": total_reg / n_samples,
            "unc_loss": total_unc / n_samples
        })

        # Save checkpoint bundle for resuming anytime
        checkpoint_bundle = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "best_val_loss": best_val_loss,
            "metrics_log": metrics_log
        }

        # Save last checkpoint for seamless restart
        torch.save(checkpoint_bundle, "results/checkpoints/trustdelay_last.pth")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Clean model weights for evaluation
            torch.save(model.state_dict(), "results/checkpoints/trustdelay_best.pth")
            checkpoint_bundle["best_val_loss"] = best_val_loss
            torch.save(checkpoint_bundle, "results/checkpoints/trustdelay_best_bundle.pth")
            epoch_summary += f"  ==> Saved New Best Checkpoint ({best_val_loss:.4f})"

        # Periodic checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            torch.save(checkpoint_bundle, f"results/checkpoints/trustdelay_epoch_{epoch:02d}.pth")

        print(epoch_summary, flush=True)

        # Update JSON log on disk after each epoch
        with open("results/training_metrics.json", "w") as f:
            json.dump(metrics_log, f, indent=2)

    # Save final model weights
    torch.save(model.state_dict(), "results/checkpoints/trustdelay_final.pth")
    print(f"\n{'='*70}")
    print(f" Training Successfully Completed! Best Validation Loss: {best_val_loss:.4f}")
    print(f" Best Weights: results/checkpoints/trustdelay_best.pth")
    print(f" Resume Bundle: results/checkpoints/trustdelay_last.pth")
    print(f"{'='*70}\n", flush=True)

if __name__ == "__main__":
    train()
