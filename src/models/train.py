#!/usr/bin/env python3
"""
src/models/train.py
--------------------
Training script for LocustNet.

Optimiser  : Adam (weight_decay=1e-4)
Scheduler  : CosineAnnealingLR
Loss       : Focal loss (binary) + CrossEntropy (phase, weight=0.3)
Stopping   : Early stopping on val AUC-ROC (patience=10)
Checkpoint : outputs/checkpoints/best_model.pt

Usage
-----
  python src/models/train.py                      # full run
  python src/models/train.py --sample 400         # 400 cells
  python src/models/train.py --fast               # 200 cells, 5 epochs (sanity check)
  python src/models/train.py --sample 400 --epochs 30
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import roc_auc_score, average_precision_score

from src.models.dataset import make_datasets, make_loaders
from src.models.architecture import build_model

CHECKPOINT_DIR = ROOT / "outputs/checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
TRAIN_LOG_PATH  = ROOT / "outputs/train_log.json"


# ── Loss functions ────────────────────────────────────────────────────────────

def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    """Binary focal loss (Lin et al., 2017)."""
    targets = targets.float()
    bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t  = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
    a_t  = alpha * targets + (1 - alpha) * (1 - targets)
    loss = a_t * ((1 - p_t) ** gamma) * bce
    return loss.mean()


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device,
             gamma: float, phase_weight: float) -> dict:
    model.eval()
    all_logits, all_labels_b, all_labels_p = [], [], []
    total_loss = 0.0
    n_batches  = 0

    for seq, lb, lp in loader:
        seq = seq.to(device)
        lb  = lb.to(device)
        lp  = lp.to(device)

        bl, pl, _ = model(seq)

        loss = (focal_loss(bl, lb.float(), gamma=gamma)
                + phase_weight * F.cross_entropy(pl, lp))
        total_loss += loss.item()
        n_batches  += 1

        all_logits.append(torch.sigmoid(bl).cpu().numpy())
        all_labels_b.append(lb.cpu().numpy())
        all_labels_p.append(lp.cpu().numpy())

    probs   = np.concatenate(all_logits)
    labels_b = np.concatenate(all_labels_b)
    labels_p = np.concatenate(all_labels_p)

    metrics = {"loss": total_loss / max(n_batches, 1)}

    if labels_b.sum() >= 2:
        metrics["auc_roc"] = float(roc_auc_score(labels_b, probs))
        metrics["pr_auc"]  = float(average_precision_score(labels_b, probs))
    else:
        metrics["auc_roc"] = float("nan")
        metrics["pr_auc"]  = float("nan")

    return metrics


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, optimiser: torch.optim.Optimizer,
                    epoch: int, val_auc: float, meta: dict):
    torch.save({
        "epoch":      epoch,
        "val_auc":    val_auc,
        "model_state":     model.state_dict(),
        "optimiser_state": optimiser.state_dict(),
        "meta":       meta,
    }, BEST_MODEL_PATH)
    print(f"    Checkpoint saved (epoch {epoch}  val AUC={val_auc:.4f})")


def load_checkpoint(model: nn.Module, path: Path = BEST_MODEL_PATH) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    return ckpt


# ── Main training loop ────────────────────────────────────────────────────────

def train(args):
    # Load config
    cfg_path = ROOT / "configs/config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Hyperparameters
    batch_size   = args.batch_size or cfg["training"]["batch_size"]
    max_epochs   = args.epochs    or cfg["training"]["max_epochs"]
    lr           = cfg["training"]["learning_rate"]
    wd           = cfg["training"]["weight_decay"]
    patience     = cfg["training"]["patience"]
    focal_gamma  = cfg["training"]["focal_loss_gamma"]
    phase_weight = 0.3   # relative weight of phase loss vs binary loss
    seq_len      = cfg["features"]["sequence_length"]

    # Datasets
    datasets = make_datasets(
        seq_len=seq_len,
        sample_cells=args.sample,
        seed=seed,
    )
    feat_cols = datasets["feat_cols"]
    loaders   = make_loaders(datasets, batch_size=batch_size)

    train_ds = datasets["train"]
    n_self = train_ds.n_self_features
    n_nbr  = train_ds.n_nbr_features
    print(f"  n_self={n_self}  n_nbr={n_nbr}")

    # Model
    model = build_model(n_self, n_nbr, cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # Optimiser + scheduler
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max_epochs, eta_min=lr * 0.01
    )

    # Training loop
    best_val_auc = -1.0
    patience_ctr = 0
    log = []

    print(f"\nTraining for up to {max_epochs} epochs (patience={patience}) ...")
    print("-" * 64)

    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n_batches  = 0

        for seq, lb, lp in loaders["train"]:
            seq = seq.to(device)
            lb  = lb.to(device)
            lp  = lp.to(device)

            optimiser.zero_grad()
            bl, pl, _ = model(seq)
            loss = (focal_loss(bl, lb.float(), gamma=focal_gamma)
                    + phase_weight * F.cross_entropy(pl, lp))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            train_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # Validation
        val_metrics = evaluate(model, loaders["val"], device, focal_gamma, phase_weight)
        val_auc = val_metrics.get("auc_roc", float("nan"))
        elapsed = time.time() - t0

        entry = {
            "epoch":      epoch,
            "train_loss": round(train_loss / max(n_batches, 1), 5),
            "val_loss":   round(val_metrics["loss"], 5),
            "val_auc":    round(val_auc, 5),
            "val_pr_auc": round(val_metrics.get("pr_auc", float("nan")), 5),
            "lr":         round(scheduler.get_last_lr()[0], 6),
        }
        log.append(entry)

        print(f"  Epoch {epoch:3d}/{max_epochs}  "
              f"train_loss={entry['train_loss']:.4f}  "
              f"val_loss={entry['val_loss']:.4f}  "
              f"val_AUC={entry['val_auc']:.4f}  "
              f"val_PR-AUC={entry['val_pr_auc']:.4f}  "
              f"({elapsed:.1f}s)")

        # Save best
        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_ctr = 0
            meta = {
                "feat_cols":    feat_cols,
                "n_self":       n_self,
                "n_nbr":        n_nbr,
                "seq_len":      seq_len,
                "feat_mean":    train_ds.feat_stats["mean"].tolist(),
                "feat_std":     train_ds.feat_stats["std"].tolist(),
            }
            save_checkpoint(model, optimiser, epoch, val_auc, meta)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\n  Early stopping (patience={patience} exhausted).")
                break

    # Save training log
    with open(TRAIN_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining log -> {TRAIN_LOG_PATH}")

    # Final test evaluation using best checkpoint
    print("\n--- Test evaluation (best checkpoint) ---")
    if BEST_MODEL_PATH.exists():
        ckpt = load_checkpoint(model)
        test_metrics = evaluate(model, loaders["test"], device, focal_gamma, phase_weight)
        print(f"  Test AUC-ROC : {test_metrics.get('auc_roc', float('nan')):.4f}")
        print(f"  Test PR-AUC  : {test_metrics.get('pr_auc',  float('nan')):.4f}")
        print(f"  Test loss    : {test_metrics['loss']:.4f}")

    print("\n" + "=" * 60)
    print("  Phase 3 training complete.")
    print(f"  Best val AUC-ROC: {best_val_auc:.4f}")
    print(f"  Checkpoint: {BEST_MODEL_PATH}")
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train LocustNet")
    parser.add_argument("--sample", type=int, default=None,
                        help="Subsample N cells (fast dev)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max_epochs from config")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch_size from config")
    parser.add_argument("--fast", action="store_true",
                        help="Quick sanity check: 200 cells, 5 epochs")
    args = parser.parse_args()

    if args.fast:
        args.sample = args.sample or 200
        args.epochs = args.epochs or 5
        args.batch_size = args.batch_size or 128

    print("=" * 60)
    print("  LocustWatch AI -- Phase 3 Neural Model Training")
    print("=" * 60)

    train(args)


if __name__ == "__main__":
    main()
