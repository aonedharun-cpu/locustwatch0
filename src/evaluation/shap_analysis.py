#!/usr/bin/env python3
"""
src/evaluation/shap_analysis.py
---------------------------------
Feature importance analysis for LocustWatch AI.

Two complementary methods:
  1. XGBoost TreeExplainer (SHAP) -- exact, fast, interpretable
  2. LocustNet gradient saliency   -- which features/time-steps drive predictions

Outputs (outputs/figures/)
  fig_04a_shap_summary.png       -- XGBoost SHAP beeswarm / bar summary
  fig_04b_shap_importance.png    -- top-20 mean |SHAP| bar chart
  fig_04c_neural_saliency.png    -- mean gradient saliency: feature x time-step heatmap

Usage
-----
  python src/evaluation/shap_analysis.py
  python src/evaluation/shap_analysis.py --sample 500
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from src.models.dataset import make_datasets, LABEL_BINARY, SKIP_COLS
from src.models.architecture import build_model

FIGURES_DIR    = ROOT / "outputs/figures"
CHECKPOINT     = ROOT / "outputs/checkpoints/best_model.pt"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[WARN] shap not installed -- XGBoost SHAP skipped. Run: pip install shap")


# ── XGBoost SHAP ──────────────────────────────────────────────────────────────

def xgb_shap_analysis(datasets: dict):
    if not HAS_SHAP:
        return

    print("\n[XGBoost SHAP]")
    feat_cols = datasets["feat_cols"]
    train_ds  = datasets["train"]
    test_ds   = datasets["test"]

    # Rebuild flat arrays from the dataset (last time-step features for tabular model)
    def get_last_step(ds):
        xs, ys = [], []
        for i in range(len(ds)):
            seq, lb, _ = ds[i]
            xs.append(seq[-1].numpy())  # last time step
            ys.append(lb.item())
        return np.array(xs, dtype=np.float32), np.array(ys)

    print("  Building tabular arrays from dataset ...")
    X_train, y_train = get_last_step(train_ds)
    X_test,  y_test  = get_last_step(test_ds)

    # Subsample for SHAP speed
    n_explain = min(2000, len(X_test))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), n_explain, replace=False)
    X_explain = X_test[idx]

    spw = max(1.0, (y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"  Fitting XGBoost (scale_pos_weight={spw:.1f}) ...")
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                           scale_pos_weight=spw, n_jobs=-1,
                           random_state=42, verbosity=0, eval_metric="aucpr")
    model.fit(X_train, y_train)

    print(f"  Computing SHAP values for {n_explain} test samples ...")
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_explain)   # (n_explain, n_features)

    # Feature labels (shorten long names)
    feat_labels = [f[:30] for f in feat_cols]

    # ── Fig 04a: bar summary ───────────────────────────────────────────────────
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top20_idx = np.argsort(mean_abs)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(20)
    ax.barh(y_pos, mean_abs[top20_idx][::-1],
            color="#2c7bb6", alpha=0.85, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feat_labels[i] for i in top20_idx[::-1]], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top-20 Feature Importance (XGBoost SHAP)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    out = FIGURES_DIR / "fig_04a_shap_importance.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")

    # ── Fig 04b: SHAP beeswarm (top 15 features) ─────────────────────────────
    top15_idx = top20_idx[:15]
    fig, ax = plt.subplots(figsize=(10, 7))
    # Manual beeswarm-style scatter
    for rank, feat_i in enumerate(top15_idx[::-1]):
        sv   = shap_vals[:, feat_i]
        fv   = X_explain[:, feat_i]
        norm = (fv - fv.min()) / ((fv.max() - fv.min()) + 1e-9)
        colours = plt.cm.coolwarm(norm)
        ax.scatter(sv, np.full(len(sv), rank) + np.random.normal(0, 0.08, len(sv)),
                   c=colours, s=6, alpha=0.6, linewidths=0)

    ax.set_yticks(range(15))
    ax.set_yticklabels([feat_labels[i] for i in top15_idx[::-1]], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on outbreak probability)")
    ax.set_title("SHAP Beeswarm: Top-15 Features (red = high feature value)",
                 fontsize=11)
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    out = FIGURES_DIR / "fig_04b_shap_beeswarm.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")

    # Save importance CSV
    importance_df = pd.DataFrame({
        "feature": feat_cols,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)
    out_csv = ROOT / "outputs/shap_importance.csv"
    importance_df.to_csv(out_csv, index=False)
    print(f"  Saved -> {out_csv.name}")


# ── Neural gradient saliency ──────────────────────────────────────────────────

def neural_saliency(datasets: dict):
    if not CHECKPOINT.exists():
        print("\n[Neural saliency] Checkpoint not found -- skipping.")
        print("  Run: python src/models/train.py --fast")
        return

    print("\n[LocustNet gradient saliency]")
    feat_cols = datasets["feat_cols"]
    test_ds   = datasets["test"]

    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    meta = ckpt["meta"]
    model = build_model(meta["n_self"], meta["n_nbr"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Collect positive-class samples for saliency (up to 200)
    pos_seqs = []
    for i in range(len(test_ds)):
        seq, lb, _ = test_ds[i]
        if lb.item() == 1:
            pos_seqs.append(seq)
        if len(pos_seqs) >= 200:
            break

    if not pos_seqs:
        print("  No positive samples in test set -- using random subset.")
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(test_ds), 200, replace=False)
        for i in idxs:
            seq, _, _ = test_ds[int(i)]
            pos_seqs.append(seq)

    X = torch.stack(pos_seqs)             # (N, seq_len, n_features)
    X.requires_grad_(True)

    bl, _, _ = model(X)
    prob = torch.sigmoid(bl)
    prob.sum().backward()

    saliency = X.grad.abs().detach().numpy()   # (N, seq_len, n_features)
    mean_sal = saliency.mean(axis=0)           # (seq_len, n_features)

    # ── Fig 04c: heatmap ──────────────────────────────────────────────────────
    seq_len    = mean_sal.shape[0]
    feat_short = [f[:22] for f in feat_cols]

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(mean_sal.T, aspect="auto", cmap="YlOrRd", origin="lower")
    fig.colorbar(im, ax=ax, label="Mean |gradient|")

    ax.set_xlabel(f"Time step (0 = oldest,  {seq_len-1} = most recent)")
    ax.set_ylabel("Feature")
    ax.set_yticks(range(len(feat_short)))
    ax.set_yticklabels(feat_short, fontsize=6)
    ax.set_title("LocustNet Gradient Saliency: Feature x Time-Step",
                 fontsize=12)
    fig.tight_layout()
    out = FIGURES_DIR / "fig_04c_neural_saliency.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Subsample N cells")
    args = parser.parse_args()

    print("=" * 60)
    print("  LocustWatch AI -- Phase 4 SHAP / Saliency Analysis")
    print("=" * 60)

    datasets = make_datasets(sample_cells=args.sample)
    xgb_shap_analysis(datasets)
    neural_saliency(datasets)

    print(f"\nFigures saved to outputs/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
