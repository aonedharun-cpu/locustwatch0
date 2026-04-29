#!/usr/bin/env python3
"""
scripts/export_demo_predictions.py
------------------------------------
Pre-compute risk predictions for a small cell sample and save to
outputs/demo_predictions.parquet so the Streamlit Cloud demo can
render the interactive Folium map without the 2GB features file.

Usage
-----
  python scripts/export_demo_predictions.py
  python scripts/export_demo_predictions.py --cells 1000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.architecture import build_model

CHECKPOINT     = ROOT / "outputs/checkpoints/best_model.pt"
TEMP_FILE      = ROOT / "outputs/calibration_temperature.json"
FEATURES_FILE  = ROOT / "data/processed/features.parquet"
OUT_FILE       = ROOT / "outputs/demo_predictions.parquet"


@torch.no_grad()
def run_inference(model, feat_stats, feat_cols, df, seq_len, T=1.0, batch_size=512):
    df = df.sort_values(["cell_id", "week"]).reset_index(drop=True)
    raw    = df[feat_cols].values.astype(np.float32)
    normed = (raw - feat_stats["mean"]) / feat_stats["std"]
    normed = np.nan_to_num(normed, nan=0.0)
    cell_ids = df["cell_id"].values

    valid_idx = [
        i for i in range(seq_len - 1, len(df))
        if cell_ids[i - seq_len + 1] == cell_ids[i]
    ]
    if not valid_idx:
        return pd.DataFrame()

    model.eval()
    valid_arr = np.array(valid_idx)
    all_probs, all_stds = [], []

    for start in range(0, len(valid_arr), batch_size):
        ends = valid_arr[start:start + batch_size]
        seqs = np.stack([normed[e - seq_len + 1:e + 1] for e in ends])
        seq_t = torch.tensor(seqs, dtype=torch.float32)

        bl, _, _ = model(seq_t)
        probs = torch.sigmoid(bl / T).cpu().numpy()
        all_probs.append(probs)

        mean_mc, std_mc = model.mc_predict(seq_t, n_samples=5)
        all_stds.append(std_mc.cpu().numpy())

    result = df.iloc[valid_arr][["cell_id", "lat", "lon", "week"]].copy()
    result["risk_prob"] = np.concatenate(all_probs)
    result["risk_std"]  = np.concatenate(all_stds)
    return result.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=800,
                        help="Number of cells to include (default 800)")
    args = parser.parse_args()

    print("Loading model ...")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    meta = ckpt["meta"]
    model = build_model(meta["n_self"], meta["n_nbr"])
    model.load_state_dict(ckpt["model_state"])
    feat_stats = {
        "mean": np.array(meta["feat_mean"], dtype=np.float32),
        "std":  np.array(meta["feat_std"],  dtype=np.float32),
    }
    feat_cols = meta["feat_cols"]
    seq_len   = meta["seq_len"]

    T = 1.0
    if TEMP_FILE.exists():
        with open(TEMP_FILE) as f:
            T = json.load(f).get("temperature", 1.0)
    print(f"  Calibration T = {T:.4f}")

    print("Loading features ...")
    df = pd.read_parquet(FEATURES_FILE)
    df["week"] = pd.to_datetime(df["week"])

    # Sample cells -- bias toward Horn of Africa for a better demo
    hoa = df[df["lat"].between(-2, 20) & df["lon"].between(35, 55)]
    hoa_cells = hoa["cell_id"].unique()
    other_cells = np.setdiff1d(df["cell_id"].unique(), hoa_cells)

    rng = np.random.default_rng(42)
    n_hoa   = min(int(args.cells * 0.6), len(hoa_cells))
    n_other = min(args.cells - n_hoa, len(other_cells))
    chosen = np.concatenate([
        rng.choice(hoa_cells,   n_hoa,   replace=False),
        rng.choice(other_cells, n_other, replace=False),
    ])
    print(f"  Cells: {len(chosen):,} ({n_hoa} HoA + {n_other} other)")

    # Use all available weeks
    df_sample = df[df["cell_id"].isin(chosen)].copy()
    print(f"  Rows: {len(df_sample):,}")

    print("Running inference ...")
    preds = run_inference(model, feat_stats, feat_cols, df_sample, seq_len, T)
    print(f"  Predictions: {len(preds):,}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(OUT_FILE, index=False)
    size_mb = OUT_FILE.stat().st_size / 1e6
    print(f"Saved -> {OUT_FILE}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
