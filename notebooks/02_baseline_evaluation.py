#!/usr/bin/env python3
"""
notebooks/02_baseline_evaluation.py
-------------------------------------
Visualise Phase 2 baseline classifier results.
Run after src/models/baselines.py completes.

Generates:
  fig_02a_metric_comparison.png   -- grouped bar chart: AUC/F1/PR-AUC/Brier per model
  fig_02b_roc_curves.png          -- ROC curves (test split, all models)
  fig_02c_pr_curves.png           -- Precision-Recall curves (test split)
  fig_02d_spatial_cv.png          -- spatial CV AUC-ROC per region per model

Usage:
    python notebooks/02_baseline_evaluation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS_FILE = ROOT / "outputs/baseline_results.csv"
CURVES_FILE  = ROOT / "outputs/baseline_curves.parquet"
FIGURES_DIR  = ROOT / "outputs/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLOURS = {
    "logistic_regression": "#2c7bb6",
    "random_forest":       "#1a9850",
    "xgboost":             "#d7191c",
}
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
}


def load_results() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not RESULTS_FILE.exists():
        print("[ERROR] baseline_results.csv not found.")
        print("  Run: python src/models/baselines.py")
        sys.exit(1)
    df = pd.read_csv(RESULTS_FILE)
    print(f"Loaded results: {len(df)} rows")

    curves = None
    if CURVES_FILE.exists():
        curves = pd.read_parquet(CURVES_FILE)
        print(f"Loaded curves:  {len(curves)} rows")
    else:
        print("[WARN] baseline_curves.parquet not found -- ROC/PR plots skipped")

    return df, curves


def print_summary(df: pd.DataFrame):
    print("\n--- Temporal evaluation (test split) ---")
    test = df[(df["eval_type"] == "temporal") & (df["split"] == "test")]
    if len(test):
        print(test[["model", "auc_roc", "f1", "pr_auc", "brier"]].to_string(index=False,
              float_format=lambda x: f"{x:.4f}"))

    print("\n--- Spatial CV (mean across regions) ---")
    cv = df[df["eval_type"] == "spatial_cv"]
    if len(cv):
        cv_mean = cv.groupby("model")[["auc_roc", "f1", "pr_auc", "brier"]].mean()
        print(cv_mean.to_string(float_format=lambda x: f"{x:.4f}"))


def fig_metric_comparison(df: pd.DataFrame):
    """Grouped bar chart: AUC-ROC, F1, PR-AUC, Brier across models for each split."""
    metrics = ["auc_roc", "f1", "pr_auc", "brier"]
    metric_labels = ["AUC-ROC", "F1", "PR-AUC", "Brier"]
    splits = ["train", "val", "test"]
    temporal = df[df["eval_type"] == "temporal"]

    models = list(MODEL_COLOURS.keys())
    n_models = len(models)
    x = np.arange(len(metrics))
    width = 0.22

    fig, axes = plt.subplots(1, len(splits), figsize=(15, 5), sharey=False)

    for ax, split in zip(axes, splits):
        sub = temporal[temporal["split"] == split]
        for i, model in enumerate(models):
            row = sub[sub["model"] == model]
            if row.empty:
                continue
            vals = [row[m].values[0] for m in metrics]
            offset = (i - (n_models - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width,
                          label=MODEL_LABELS[model],
                          color=MODEL_COLOURS[model], alpha=0.85)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.005,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)

        ax.set_title(f"{split.capitalize()} split", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score" if split == "train" else "")
        ax.grid(True, alpha=0.3, axis="y")
        if split == "test":
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Baseline Model Performance by Split", fontsize=13, y=1.01)
    fig.tight_layout()

    out = FIGURES_DIR / "fig_02a_metric_comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")


def fig_roc_curves(curves: pd.DataFrame):
    """ROC curves for all models on the test split."""
    roc = curves[curves["curve"] == "roc"]
    if roc.empty:
        print("  [skip] No ROC curve data")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")

    for model in MODEL_COLOURS:
        sub = roc[roc["model"] == model].sort_values("x")
        if sub.empty:
            continue
        ax.plot(sub["x"], sub["y"],
                color=MODEL_COLOURS[model], linewidth=1.8,
                label=MODEL_LABELS[model])

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (test split)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_02b_roc_curves.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")


def fig_pr_curves(curves: pd.DataFrame):
    """Precision-Recall curves for all models on the test split."""
    pr = curves[curves["curve"] == "pr"]
    if pr.empty:
        print("  [skip] No PR curve data")
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    for model in MODEL_COLOURS:
        sub = pr[pr["model"] == model].sort_values("x")
        if sub.empty:
            continue
        ax.plot(sub["x"], sub["y"],
                color=MODEL_COLOURS[model], linewidth=1.8,
                label=MODEL_LABELS[model])

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (test split)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_02c_pr_curves.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")


def fig_spatial_cv(df: pd.DataFrame):
    """AUC-ROC per region per model from spatial leave-one-out CV."""
    cv = df[df["eval_type"] == "spatial_cv"]
    if cv.empty:
        print("  [skip] No spatial CV data")
        return

    regions = sorted(cv["split"].unique())
    models  = list(MODEL_COLOURS.keys())
    x = np.arange(len(regions))
    width = 0.25
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, model in enumerate(models):
        sub = cv[cv["model"] == model].set_index("split")
        vals = [sub.loc[r, "auc_roc"] if r in sub.index else np.nan for r in regions]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width,
                      label=MODEL_LABELS[model],
                      color=MODEL_COLOURS[model], alpha=0.85)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    region_labels = [r.replace("_", "\n") for r in regions]
    ax.set_xticklabels(region_labels, fontsize=8)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Spatial CV: AUC-ROC by Held-Out Region")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random")

    fig.tight_layout()
    out = FIGURES_DIR / "fig_02d_spatial_cv.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")


def main():
    print("=" * 60)
    print("  LocustWatch AI -- Phase 2 Baseline Evaluation")
    print("=" * 60)

    df, curves = load_results()
    print_summary(df)

    print("\nGenerating figures ...")
    fig_metric_comparison(df)

    if curves is not None:
        fig_roc_curves(curves)
        fig_pr_curves(curves)

    fig_spatial_cv(df)

    print(f"\nAll figures saved to outputs/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
