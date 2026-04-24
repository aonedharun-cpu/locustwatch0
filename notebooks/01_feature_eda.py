#!/usr/bin/env python3
"""
notebooks/01_feature_eda.py
----------------------------
EDA on the Phase 1 feature matrix (data/processed/features.parquet).
Run after build_features.py completes.

Generates:
  fig_01a_feature_correlations.png    -- heatmap of feature correlations
  fig_01b_class_imbalance.png         -- label distribution bars
  fig_01c_outbreak_feature_dists.png  -- KDE plots: feature distributions
                                         split by outbreak_30d (0 vs 1)
  fig_01d_outbreak_map.png            -- geographic density of positive labels

Usage:
    python notebooks/01_feature_eda.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FEATURES_FILE = ROOT / "data/processed/features.parquet"
FIGURES_DIR   = ROOT / "outputs/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_features() -> pd.DataFrame:
    if not FEATURES_FILE.exists():
        print("[ERROR] features.parquet not found.")
        print("  Run: python src/features/build_features.py")
        sys.exit(1)
    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded features.parquet: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def print_summary(df: pd.DataFrame):
    print("\n--- Feature Matrix Summary ---")
    print(f"  Cells:   {df['cell_id'].nunique():,}")
    print(f"  Weeks:   {df['week'].nunique()} "
          f"({df['week'].min().date()} -> {df['week'].max().date()})")
    print(f"  Cols:    {len(df.columns)}")
    print(f"  Memory:  {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"\n  Outbreak prevalence: "
          f"{df['outbreak_30d'].mean()*100:.2f}%  ({df['outbreak_30d'].sum():,} positive)")
    print(f"  Phase distribution:")
    print(f"    {df['phase_class'].value_counts().sort_index().to_dict()}")

    nan_rate = df.isnull().mean() * 100
    high_nan = nan_rate[nan_rate > 5].sort_values(ascending=False)
    if len(high_nan):
        print(f"\n  Columns with >5% NaN (expected for lag/nbr near boundaries):")
        for col, pct in high_nan.items():
            print(f"    {col:<45} {pct:.1f}%")


def fig_feature_correlations(df: pd.DataFrame):
    """Heatmap of Pearson correlations among core features."""
    # Use only core (non-lag, non-nbr) feature columns
    feat_cols = [
        c for c in df.columns
        if c not in ("cell_id", "lat", "lon", "week", "outbreak_30d", "phase_class")
        and "lag" not in c and "nbr" not in c
    ][:20]  # limit to first 20 for readability

    corr = df[feat_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(len(feat_cols)))
    ax.set_yticks(range(len(feat_cols)))
    ax.set_xticklabels(feat_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(feat_cols, fontsize=8)
    ax.set_title("Feature Correlations (core features)", fontsize=12)

    # Annotate cells with high correlation
    for i in range(len(feat_cols)):
        for j in range(len(feat_cols)):
            val = corr.iloc[i, j]
            if i != j and abs(val) > 0.5:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(val) > 0.7 else "black")

    fig.tight_layout()
    out = FIGURES_DIR / "fig_01a_feature_correlations.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")


def fig_class_imbalance(df: pd.DataFrame):
    """Bar charts for outbreak_30d and phase_class distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # outbreak_30d
    ax = axes[0]
    counts = df["outbreak_30d"].value_counts().sort_index()
    bars = ax.bar(["No outbreak (0)", "Outbreak (1)"], counts.values,
                  color=["#2c7bb6", "#d7191c"], alpha=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{val:,}\n({val/len(df)*100:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Binary label: outbreak_30d")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    # phase_class
    ax = axes[1]
    counts = df["phase_class"].value_counts().sort_index()
    phase_labels = {0: "None (0)", 1: "Solitarious (1)",
                    2: "Gregarious (2)", 3: "Swarming (3)"}
    colours = ["#aaaaaa", "#4dac26", "#f1a340", "#d7191c"]
    bars = ax.bar(
        [phase_labels.get(i, str(i)) for i in counts.index],
        counts.values,
        color=[colours[i] for i in counts.index],
        alpha=0.8,
    )
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{val:,}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Multi-class label: phase_class")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Class Imbalance", fontsize=13, y=1.01)
    fig.tight_layout()
    out = FIGURES_DIR / "fig_01b_class_imbalance.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")


def fig_feature_distributions(df: pd.DataFrame):
    """
    KDE/histogram overlays showing feature distributions split by outbreak_30d.
    Useful for spotting which features most separate the classes.
    """
    plot_features = [
        "rainfall_weekly_mm", "rainfall_roll_4w", "rainfall_anomaly",
        "temp_mean_c", "temp_anomaly",
        "ndvi", "ndvi_anomaly",
        "soil_moisture_surface", "wind_speed_ms", "humidity_pct",
    ]
    plot_features = [f for f in plot_features if f in df.columns]

    n_cols = 5
    n_rows = int(np.ceil(len(plot_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 2.8))
    axes = axes.ravel()

    df_neg = df[df["outbreak_30d"] == 0].sample(min(50_000, (df["outbreak_30d"]==0).sum()),
                                                  random_state=42)
    df_pos = df[df["outbreak_30d"] == 1]

    for i, feat in enumerate(plot_features):
        ax = axes[i]
        neg_vals = df_neg[feat].dropna()
        pos_vals = df_pos[feat].dropna()

        ax.hist(neg_vals, bins=40, density=True, alpha=0.5,
                color="#2c7bb6", label="No outbreak")
        ax.hist(pos_vals, bins=40, density=True, alpha=0.6,
                color="#d7191c", label="Outbreak")
        ax.set_title(feat, fontsize=8)
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    for j in range(len(plot_features), len(axes)):
        axes[j].set_visible(False)

    # Single shared legend
    axes[0].legend(fontsize=7)
    fig.suptitle("Feature Distributions: Outbreak vs. No Outbreak", fontsize=12)
    fig.tight_layout()

    out = FIGURES_DIR / "fig_01c_outbreak_feature_dists.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")


def fig_outbreak_map(df: pd.DataFrame):
    """
    Geographic scatter: fraction of weeks with outbreak per cell.
    Reveals spatial patterns in the label distribution.
    """
    cell_prevalence = (
        df.groupby(["lat", "lon"])["outbreak_30d"]
        .mean()
        .reset_index()
        .rename(columns={"outbreak_30d": "outbreak_rate"})
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    # Background: all cells grey
    ax.scatter(cell_prevalence["lon"], cell_prevalence["lat"],
               c="#dddddd", s=2, zorder=1)

    # Outbreak cells coloured by rate
    pos = cell_prevalence[cell_prevalence["outbreak_rate"] > 0]
    sc = ax.scatter(
        pos["lon"], pos["lat"],
        c=pos["outbreak_rate"], cmap="OrRd",
        s=10, vmin=0, vmax=pos["outbreak_rate"].quantile(0.95),
        zorder=2
    )
    plt.colorbar(sc, ax=ax, label="Fraction of weeks with outbreak")

    ax.set_title("Geographic Distribution of Outbreak Labels "
                 "(fraction of weeks positive)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-22, 77)
    ax.set_ylim(-7, 37)
    ax.grid(True, alpha=0.3)

    out = FIGURES_DIR / "fig_01d_outbreak_map.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")


def main():
    print("=" * 60)
    print("  LocustWatch AI -- Phase 1 Feature EDA")
    print("=" * 60)

    df = load_features()
    print_summary(df)

    print("\nGenerating figures ...")
    fig_feature_correlations(df)
    fig_class_imbalance(df)
    fig_feature_distributions(df)
    fig_outbreak_map(df)

    print(f"\nAll figures saved to outputs/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
