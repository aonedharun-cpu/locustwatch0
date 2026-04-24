#!/usr/bin/env python3
"""
src/evaluation/calibration.py
-------------------------------
Post-hoc probability calibration for LocustNet via temperature scaling.

Temperature scaling learns a single scalar T on the validation set such that
p_calibrated = sigmoid(logit / T). T > 1 softens predictions (reduces overconfidence);
T < 1 sharpens them.

Outputs
-------
  outputs/calibration_temperature.json   -- {"temperature": T}
  outputs/figures/fig_04d_calibration.png -- reliability diagrams before/after

Usage
-----
  python src/evaluation/calibration.py
  python src/evaluation/calibration.py --sample 500
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.dataset import make_datasets, make_loaders
from src.models.architecture import build_model

CHECKPOINT   = ROOT / "outputs/checkpoints/best_model.pt"
TEMP_FILE    = ROOT / "outputs/calibration_temperature.json"
FIGURES_DIR  = ROOT / "outputs/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Expected calibration error ────────────────────────────────────────────────

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() / len(probs) * abs(acc - conf)
    return float(ece)


def reliability_data(probs: np.ndarray, labels: np.ndarray,
                     n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centres, mean_accuracy, bin_counts)."""
    bins    = np.linspace(0, 1, n_bins + 1)
    centres = (bins[:-1] + bins[1:]) / 2
    accs    = []
    counts  = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        accs.append(labels[mask].mean() if mask.sum() > 0 else np.nan)
        counts.append(mask.sum())
    return centres, np.array(accs), np.array(counts)


# ── Temperature scaling ───────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.05)


def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    lr: float = 0.01, n_steps: int = 200) -> float:
    """Optimise temperature on NLL loss. Returns scalar T."""
    scaler  = TemperatureScaler()
    optim   = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=n_steps)
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    def closure():
        optim.zero_grad()
        scaled = scaler(logits_t)
        loss   = F.binary_cross_entropy_with_logits(scaled, labels_t)
        loss.backward()
        return loss

    optim.step(closure)
    return float(scaler.temperature.item())


# ── Collect logits from model ─────────────────────────────────────────────────

@torch.no_grad()
def get_logits(model: nn.Module, loader, device: torch.device):
    model.eval()
    all_logits, all_labels = [], []
    for seq, lb, _ in loader:
        seq = seq.to(device)
        bl, _, _ = model(seq)
        all_logits.append(bl.cpu().numpy())
        all_labels.append(lb.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


# ── Figure ────────────────────────────────────────────────────────────────────

def fig_reliability(probs_raw: np.ndarray, probs_cal: np.ndarray,
                    labels: np.ndarray, ece_raw: float, ece_cal: float,
                    temperature: float):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, probs, ece, title in [
        (axes[0], probs_raw, ece_raw, "Before calibration"),
        (axes[1], probs_cal, ece_cal, "After temperature scaling"),
    ]:
        centres, accs, counts = reliability_data(probs, labels)

        # Reliability curve
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.9, label="Perfect")
        valid = ~np.isnan(accs)
        ax.plot(centres[valid], accs[valid], "o-",
                color="#d7191c", linewidth=1.8, markersize=5, label="Model")

        # Histogram of confidence
        ax2 = ax.twinx()
        ax2.bar(centres, counts / counts.sum(), width=0.09,
                color="#2c7bb6", alpha=0.3)
        ax2.set_ylabel("Fraction of samples", fontsize=8, color="#2c7bb6")
        ax2.tick_params(axis="y", labelcolor="#2c7bb6", labelsize=7)

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Reliability Diagram  (T = {temperature:.3f})", fontsize=12)
    fig.tight_layout()
    out = FIGURES_DIR / "fig_04d_calibration.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  LocustWatch AI -- Phase 4 Calibration")
    print("=" * 60)

    if not CHECKPOINT.exists():
        print("[ERROR] Checkpoint not found. Run: python src/models/train.py --fast")
        sys.exit(1)

    device = torch.device("cpu")

    # Load datasets + model
    datasets = make_datasets(sample_cells=args.sample)
    loaders  = make_loaders(datasets, batch_size=512)

    ckpt = torch.load(CHECKPOINT, map_location=device)
    meta = ckpt["meta"]
    model = build_model(meta["n_self"], meta["n_nbr"])
    model.load_state_dict(ckpt["model_state"])

    # Collect logits on val set (calibration set)
    print("\nCollecting validation logits ...")
    val_logits, val_labels = get_logits(model, loaders["val"], device)
    print(f"  {len(val_logits):,} samples  pos={val_labels.sum():,}")

    # Fit temperature
    print("Fitting temperature scaling ...")
    T = fit_temperature(val_logits, val_labels)
    print(f"  Learned temperature T = {T:.4f}")

    # Evaluate on test set
    print("Evaluating on test set ...")
    test_logits, test_labels = get_logits(model, loaders["test"], device)
    probs_raw = torch.sigmoid(torch.tensor(test_logits)).numpy()
    probs_cal = torch.sigmoid(torch.tensor(test_logits / T)).numpy()

    ece_raw = expected_calibration_error(probs_raw, test_labels)
    ece_cal = expected_calibration_error(probs_cal, test_labels)
    print(f"  ECE before: {ece_raw:.4f}")
    print(f"  ECE after:  {ece_cal:.4f}  (improvement: {(ece_raw-ece_cal)/ece_raw*100:.1f}%)")

    # Save temperature
    TEMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TEMP_FILE, "w") as f:
        json.dump({"temperature": T, "ece_before": ece_raw, "ece_after": ece_cal}, f, indent=2)
    print(f"\nSaved -> {TEMP_FILE.name}")

    fig_reliability(probs_raw, probs_cal, test_labels, ece_raw, ece_cal, T)
    print("=" * 60)


if __name__ == "__main__":
    main()
