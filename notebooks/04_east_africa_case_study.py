#!/usr/bin/env python3
"""
notebooks/04_east_africa_case_study.py
----------------------------------------
2019-2020 East Africa locust outbreak retrospective.

Uses the trained LocustNet + conformal threshold to reconstruct
predicted risk across the Horn of Africa during the historically
severe 2019-2020 outbreak season.

Generates
---------
  fig_04f_risk_maps_2019_2020.png  -- 6-panel weekly risk maps (peak weeks)
  fig_04g_risk_timeseries.png      -- mean predicted risk over time per region
  fig_04h_fao_vs_predicted.png     -- scatter: predicted risk vs FAO record density

Usage
-----
  python notebooks/04_east_africa_case_study.py
  python notebooks/04_east_africa_case_study.py --sample 800
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.dataset import make_datasets, SKIP_COLS, ordered_feat_cols
from src.models.architecture import build_model

FIGURES_DIR   = ROOT / "outputs/figures"
CHECKPOINT    = ROOT / "outputs/checkpoints/best_model.pt"
CONFORMAL_OUT = ROOT / "outputs/conformal_threshold.json"
FAO_FILE      = ROOT / "data/processed/fao_clean.parquet"
FEATURES_FILE = ROOT / "data/processed/features.parquet"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Horn of Africa bounding box
HOA_LAT = (-2, 15)
HOA_LON = (35, 52)

RISK_TIERS = {"watch": 0.30, "warning": 0.60, "emergency": 0.85}
TIER_COLOURS = {
    "watch":     "#f1a340",
    "warning":   "#d7191c",
    "emergency": "#7b0000",
    "none":      "#dddddd",
}


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model: nn.Module, feat_stats: dict, feat_cols: list[str],
                  df: pd.DataFrame, seq_len: int = 12, batch_size: int = 512,
                  T: float = 1.0) -> pd.DataFrame:
    """
    Run model on all (cell, week) pairs in df for which a full seq_len window exists.
    Returns DataFrame with columns: cell_id, lat, lon, week, risk_prob.
    """
    import torch.nn as nn  # local import to avoid top-level dependency issue

    # Sort and normalise
    df = df.sort_values(["cell_id", "week"]).reset_index(drop=True)
    raw     = df[feat_cols].values.astype(np.float32)
    normed  = (raw - feat_stats["mean"]) / feat_stats["std"]
    normed  = np.nan_to_num(normed, nan=0.0)
    cell_ids = df["cell_id"].values

    # Build index of valid window end positions
    valid_idx = [
        i for i in range(seq_len - 1, len(df))
        if cell_ids[i - seq_len + 1] == cell_ids[i]
    ]

    if not valid_idx:
        return pd.DataFrame()

    model.eval()
    all_probs = []
    valid_arr = np.array(valid_idx)

    for start in range(0, len(valid_arr), batch_size):
        batch_ends = valid_arr[start:start + batch_size]
        seqs = np.stack([normed[e - seq_len + 1:e + 1] for e in batch_ends])
        seq_t = torch.tensor(seqs, dtype=torch.float32)
        bl, _, _ = model(seq_t)
        probs = torch.sigmoid(bl / T).cpu().numpy()
        all_probs.append(probs)

    all_probs = np.concatenate(all_probs)

    result = df.iloc[valid_arr][["cell_id", "lat", "lon", "week"]].copy()
    result["risk_prob"] = all_probs
    return result.reset_index(drop=True)


def tier(prob: float) -> str:
    if prob >= RISK_TIERS["emergency"]:
        return "emergency"
    if prob >= RISK_TIERS["warning"]:
        return "warning"
    if prob >= RISK_TIERS["watch"]:
        return "watch"
    return "none"


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_risk_maps(preds: pd.DataFrame, fao: pd.DataFrame):
    """6-panel map of peak risk weeks in 2019-2020."""
    # Pick 6 weeks with highest mean predicted risk in HOA
    hoa = preds[
        preds["lat"].between(*HOA_LAT) & preds["lon"].between(*HOA_LON)
    ].copy()
    if hoa.empty:
        print("  [skip] No predictions in HoA bounding box.")
        return

    weekly_mean = hoa.groupby("week")["risk_prob"].mean().sort_values(ascending=False)
    top_weeks   = weekly_mean.head(6).index.tolist()
    top_weeks   = sorted(top_weeks)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0, vmax=RISK_TIERS["emergency"])

    for ax, week in zip(axes, top_weeks):
        week_preds = hoa[hoa["week"] == week]

        # Background cells (grey)
        ax.scatter(week_preds["lon"], week_preds["lat"],
                   c="#eeeeee", s=15, zorder=1)

        # Coloured by risk
        sc = ax.scatter(week_preds["lon"], week_preds["lat"],
                        c=week_preds["risk_prob"], cmap=cmap, norm=norm,
                        s=15, zorder=2)

        # FAO records in this week +/- 2 weeks
        if not fao.empty:
            fao_week = fao[
                (fao["week"] >= pd.Timestamp(week) - pd.Timedelta(weeks=2)) &
                (fao["week"] <= pd.Timestamp(week) + pd.Timedelta(weeks=2)) &
                fao["cell_lat"].between(*HOA_LAT) &
                fao["cell_lon"].between(*HOA_LON)
            ]
            if len(fao_week):
                ax.scatter(fao_week["cell_lon"], fao_week["cell_lat"],
                           marker="*", c="blue", s=60, zorder=3,
                           label=f"FAO ({len(fao_week)})")
                ax.legend(fontsize=7, loc="upper right")

        ax.set_title(f"Week: {pd.Timestamp(week).date()}", fontsize=9)
        ax.set_xlim(HOA_LON)
        ax.set_ylim(HOA_LAT)
        ax.set_xlabel("Lon", fontsize=7)
        ax.set_ylabel("Lat", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    plt.colorbar(sc, ax=axes[-1], label="Outbreak risk", shrink=0.8)
    fig.suptitle("LocustNet Risk Maps -- Horn of Africa 2019-2020\n"
                 "(blue stars = FAO locust records within 2 weeks)",
                 fontsize=12)
    fig.tight_layout()
    out = FIGURES_DIR / "fig_04f_risk_maps_2019_2020.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")


def fig_risk_timeseries(preds: pd.DataFrame):
    """Mean weekly risk over time for HoA sub-regions."""
    sub_regions = {
        "Ethiopia/Somalia": lambda df: df[df["lat"].between(5, 12) & df["lon"].between(38, 48)],
        "Yemen":            lambda df: df[df["lat"].between(12, 20) & df["lon"].between(42, 52)],
        "Kenya":            lambda df: df[df["lat"].between(-2, 5)  & df["lon"].between(35, 42)],
    }

    fig, ax = plt.subplots(figsize=(13, 5))
    colours = {"Ethiopia/Somalia": "#d7191c", "Yemen": "#f1a340", "Kenya": "#1a9850"}

    for name, fn in sub_regions.items():
        sub = fn(preds)
        if sub.empty:
            continue
        ts = sub.groupby("week")["risk_prob"].mean()
        ts.index = pd.to_datetime(ts.index)
        ax.plot(ts.index, ts.values, linewidth=1.6,
                color=colours[name], label=name, alpha=0.9)

    # Risk tier bands
    ax.axhspan(RISK_TIERS["watch"], RISK_TIERS["warning"],
               color="#f1a340", alpha=0.10, label="Watch zone")
    ax.axhspan(RISK_TIERS["warning"], RISK_TIERS["emergency"],
               color="#d7191c", alpha=0.10, label="Warning zone")
    ax.axhspan(RISK_TIERS["emergency"], 1.0,
               color="#7b0000", alpha=0.10, label="Emergency zone")

    ax.set_xlabel("Week")
    ax.set_ylabel("Mean predicted outbreak probability")
    ax.set_title("Weekly Risk Timeseries by Sub-Region (2019-2020 period)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "fig_04g_risk_timeseries.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")


def fig_fao_vs_predicted(preds: pd.DataFrame, fao: pd.DataFrame):
    """Scatter: per-cell mean predicted risk vs FAO record count."""
    if fao.empty:
        return

    # HoA only
    hoa_preds = preds[preds["lat"].between(*HOA_LAT) & preds["lon"].between(*HOA_LON)]
    cell_risk = hoa_preds.groupby("cell_id")["risk_prob"].mean()

    fao_hoa = fao[
        fao["cell_lat"].between(*HOA_LAT) & fao["cell_lon"].between(*HOA_LON)
    ]
    cell_fao = fao_hoa.groupby("cell_id").size().rename("fao_count")

    merged = pd.concat([cell_risk, cell_fao], axis=1).fillna(0)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(merged["risk_prob"], merged["fao_count"],
                    alpha=0.5, s=15, c=merged["fao_count"],
                    cmap="YlOrRd", edgecolors="none")
    plt.colorbar(sc, ax=ax, label="FAO record count")
    ax.set_xlabel("Mean predicted outbreak probability")
    ax.set_ylabel("FAO locust record count (cell total)")
    ax.set_title("Predicted Risk vs. FAO Record Density\n(Horn of Africa cells)")
    ax.grid(True, alpha=0.3)

    corr = merged.corr().loc["risk_prob", "fao_count"]
    ax.text(0.05, 0.92, f"Pearson r = {corr:.3f}",
            transform=ax.transAxes, fontsize=10)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_04h_fao_vs_predicted.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import torch.nn as nn  # ensure nn available for run_inference

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  LocustWatch AI -- Phase 4 East Africa Case Study")
    print("=" * 60)

    if not CHECKPOINT.exists():
        print("[ERROR] Checkpoint not found. Run: python src/models/train.py --fast")
        sys.exit(1)

    # Load temperature and conformal threshold
    T     = 1.0
    q_hat = None
    if (ROOT / "outputs/calibration_temperature.json").exists():
        with open(ROOT / "outputs/calibration_temperature.json") as f:
            T = json.load(f).get("temperature", 1.0)
    if CONFORMAL_OUT.exists():
        with open(CONFORMAL_OUT) as f:
            q_hat = json.load(f).get("q_hat")

    # Load model
    device = torch.device("cpu")
    ckpt   = torch.load(CHECKPOINT, map_location=device)
    meta   = ckpt["meta"]
    model  = build_model(meta["n_self"], meta["n_nbr"])
    model.load_state_dict(ckpt["model_state"])
    feat_stats = {
        "mean": np.array(meta["feat_mean"], dtype=np.float32),
        "std":  np.array(meta["feat_std"],  dtype=np.float32),
    }
    feat_cols = meta["feat_cols"]
    seq_len   = meta["seq_len"]

    # Load features: 2019-2020 period only (plus seq_len look-back)
    print("\nLoading feature matrix ...")
    df = pd.read_parquet(FEATURES_FILE)
    if args.sample:
        rng = np.random.default_rng(42)
        cells = df["cell_id"].unique()
        chosen = rng.choice(cells, min(args.sample, len(cells)), replace=False)
        df = df[df["cell_id"].isin(chosen)].copy()

    week = pd.to_datetime(df["week"])
    cutoff = pd.Timestamp("2018-10-01")   # seq_len look-back before 2019
    end    = pd.Timestamp("2020-12-31")
    df_cs  = df[week >= cutoff].copy()
    df_cs  = df_cs[pd.to_datetime(df_cs["week"]) <= end].copy()
    print(f"  Case study rows: {len(df_cs):,}")

    # Run inference
    print("Running inference ...")
    preds = run_inference(model, feat_stats, feat_cols, df_cs,
                          seq_len=seq_len, T=T)
    # Filter to 2019+ predictions (discard look-back rows)
    preds = preds[pd.to_datetime(preds["week"]) >= "2019-01-01"].copy()
    print(f"  Predictions: {len(preds):,}  "
          f"(risk>0.3: {(preds['risk_prob']>0.3).sum():,})")

    # Load FAO records for overlay
    fao = pd.DataFrame()
    if FAO_FILE.exists():
        fao = pd.read_parquet(FAO_FILE)
        fao["week"] = pd.to_datetime(fao["week"])
        fao = fao[
            (fao["week"] >= "2019-01-01") & (fao["week"] <= "2020-12-31")
        ].copy()
        print(f"  FAO records (2019-2020): {len(fao):,}")

    print("\nGenerating figures ...")
    fig_risk_maps(preds, fao)
    fig_risk_timeseries(preds)
    fig_fao_vs_predicted(preds, fao)

    print(f"\nFigures saved to outputs/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
