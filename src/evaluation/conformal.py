#!/usr/bin/env python3
"""
src/evaluation/conformal.py
-----------------------------
Split conformal prediction for locust outbreak risk.

Method: inductive (split) conformal classification.
  Calibration set  = validation split
  Nonconformity score s_i = 1 - p_hat(y_i)  (1 - predicted prob of true class)
  Threshold q_hat  = ceil((n+1)(1-alpha)) / n quantile of calibration scores
  Prediction set   = {y : s(x, y) <= q_hat}  (contains true label with >= 1-alpha coverage)

For binary classification this reduces to:
  Predict outbreak  if p_hat >= 1 - q_hat
  Coverage is guaranteed: P(y in C(x)) >= 1 - alpha on exchangeable data.

Outputs
-------
  outputs/conformal_threshold.json          -- {"q_hat": ..., "coverage": ..., ...}
  outputs/figures/fig_04e_conformal.png     -- coverage vs alpha + set-size histogram

Usage
-----
  python src/evaluation/conformal.py
  python src/evaluation/conformal.py --sample 500 --alpha 0.20
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.dataset import make_datasets, make_loaders
from src.models.architecture import build_model
from src.evaluation.calibration import get_logits

CHECKPOINT    = ROOT / "outputs/checkpoints/best_model.pt"
TEMP_FILE     = ROOT / "outputs/calibration_temperature.json"
CONFORMAL_OUT = ROOT / "outputs/conformal_threshold.json"
FIGURES_DIR   = ROOT / "outputs/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Conformal calibration ─────────────────────────────────────────────────────

def calibrate_conformal(probs_cal: np.ndarray, labels_cal: np.ndarray,
                        alpha: float) -> float:
    """
    Compute q_hat from calibration set.

    probs_cal : predicted probabilities for class 1
    labels_cal: true binary labels
    alpha     : miscoverage rate (e.g. 0.20 for 80% coverage)
    """
    # Nonconformity scores: s_i = 1 - p_hat(true class)
    scores = np.where(labels_cal == 1,
                      1 - probs_cal,       # for positives: 1 - p(1)
                      probs_cal)           # for negatives: 1 - p(0) = p(1)

    n = len(scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    q_hat = float(np.quantile(scores, level))
    return q_hat


def predict_sets(probs_test: np.ndarray, q_hat: float) -> np.ndarray:
    """
    Return prediction sets as (N, 2) boolean array.
    Column 0: is class-0 (no outbreak) in the set?
    Column 1: is class-1 (outbreak) in the set?
    """
    sets = np.zeros((len(probs_test), 2), dtype=bool)
    # Class 0 included if s(x, 0) = p(1) <= q_hat
    sets[:, 0] = probs_test <= q_hat
    # Class 1 included if s(x, 1) = 1-p(1) <= q_hat
    sets[:, 1] = (1 - probs_test) <= q_hat
    return sets


def evaluate_coverage(pred_sets: np.ndarray, labels: np.ndarray) -> dict:
    """Compute empirical coverage and average set size."""
    covered = pred_sets[np.arange(len(labels)), labels].mean()
    set_size = pred_sets.sum(axis=1).mean()
    empty    = (pred_sets.sum(axis=1) == 0).mean()
    both     = (pred_sets.sum(axis=1) == 2).mean()
    return {
        "coverage":    float(covered),
        "avg_set_size": float(set_size),
        "empty_rate":  float(empty),
        "both_rate":   float(both),
    }


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_conformal(probs_cal: np.ndarray, labels_cal: np.ndarray,
                  probs_test: np.ndarray, labels_test: np.ndarray,
                  target_alpha: float):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: Coverage vs alpha ───────────────────────────────────────────────
    alphas = np.linspace(0.05, 0.5, 30)
    coverages = []
    for a in alphas:
        q = calibrate_conformal(probs_cal, labels_cal, a)
        ps = predict_sets(probs_test, q)
        cov = evaluate_coverage(ps, labels_test)["coverage"]
        coverages.append(cov)

    ax = axes[0]
    ax.plot(1 - alphas, 1 - alphas, "k--", linewidth=0.9, label="Nominal (1-alpha)")
    ax.plot(1 - alphas, coverages, "o-",
            color="#d7191c", linewidth=1.8, markersize=4, label="Empirical coverage")
    ax.axvline(1 - target_alpha, color="#2c7bb6", linestyle=":", linewidth=1.2,
               label=f"Target (1-alpha={1-target_alpha:.0%})")
    ax.set_xlabel("Target coverage (1 - alpha)")
    ax.set_ylabel("Empirical coverage on test set")
    ax.set_title("Conformal Coverage Guarantee", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)

    # ── Right: prediction set composition at target alpha ────────────────────
    q_hat = calibrate_conformal(probs_cal, labels_cal, target_alpha)
    pred_sets = predict_sets(probs_test, q_hat)
    sizes = pred_sets.sum(axis=1)

    ax = axes[1]
    unique, counts = np.unique(sizes, return_counts=True)
    colours = {0: "#d7191c", 1: "#1a9850", 2: "#f1a340"}
    labels_map = {0: "Empty (abstain)", 1: "Singleton", 2: "Both classes (uncertain)"}
    bars = ax.bar([labels_map.get(u, str(u)) for u in unique],
                  counts / len(sizes) * 100,
                  color=[colours.get(u, "#888") for u in unique],
                  alpha=0.85, edgecolor="white")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{c/len(sizes)*100:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("% of test samples")
    ax.set_title(f"Prediction Set Sizes  (alpha={target_alpha:.2f},  "
                 f"q_hat={q_hat:.3f})", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Split Conformal Prediction", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "fig_04e_conformal.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--alpha",  type=float, default=0.20,
                        help="Miscoverage rate (default 0.20 = 80%% coverage)")
    args = parser.parse_args()

    print("=" * 60)
    print("  LocustWatch AI -- Phase 4 Conformal Prediction")
    print("=" * 60)

    if not CHECKPOINT.exists():
        print("[ERROR] Checkpoint not found. Run: python src/models/train.py --fast")
        sys.exit(1)

    device = torch.device("cpu")

    datasets = make_datasets(sample_cells=args.sample)
    loaders  = make_loaders(datasets, batch_size=512)

    ckpt  = torch.load(CHECKPOINT, map_location=device)
    meta  = ckpt["meta"]
    model = build_model(meta["n_self"], meta["n_nbr"])
    model.load_state_dict(ckpt["model_state"])

    # Load calibration temperature if available
    T = 1.0
    if TEMP_FILE.exists():
        with open(TEMP_FILE) as f:
            T = json.load(f).get("temperature", 1.0)
        print(f"  Using calibration temperature T={T:.4f}")

    print("\nCollecting probabilities ...")
    val_logits,  val_labels  = get_logits(model, loaders["val"],  device)
    test_logits, test_labels = get_logits(model, loaders["test"], device)

    probs_cal  = torch.sigmoid(torch.tensor(val_logits  / T)).numpy()
    probs_test = torch.sigmoid(torch.tensor(test_logits / T)).numpy()

    # Conformal calibration
    q_hat = calibrate_conformal(probs_cal, val_labels, args.alpha)
    print(f"\n  Calibration: n={len(probs_cal):,}  alpha={args.alpha}  q_hat={q_hat:.4f}")

    pred_sets = predict_sets(probs_test, q_hat)
    stats = evaluate_coverage(pred_sets, test_labels)

    print(f"  Test coverage : {stats['coverage']:.4f}  (target >= {1-args.alpha:.2f})")
    print(f"  Avg set size  : {stats['avg_set_size']:.4f}")
    print(f"  Empty sets    : {stats['empty_rate']*100:.2f}%")
    print(f"  Both-class    : {stats['both_rate']*100:.2f}%  (uncertain predictions)")

    result = {
        "alpha":       args.alpha,
        "q_hat":       q_hat,
        "temperature": T,
        **stats,
    }
    CONFORMAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFORMAL_OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved -> {CONFORMAL_OUT.name}")

    fig_conformal(probs_cal, val_labels, probs_test, test_labels, args.alpha)
    print("=" * 60)


if __name__ == "__main__":
    main()
