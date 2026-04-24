#!/usr/bin/env python3
"""
src/models/dataset.py
----------------------
PyTorch Dataset for spatiotemporal locust outbreak prediction.

Each sample is a (seq_len x n_features) window ending at a target week,
with binary outbreak_30d and multi-class phase_class labels.

Features are standardised using training-split statistics (mean/std).
NaN values are filled with 0 after standardisation (i.e., imputed to mean).

Neighbour (nbr) feature columns are placed LAST so architecture.py can
split them off at a fixed index for the spatial aggregation branch.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FEATURES_FILE = ROOT / "data/processed/features.parquet"
LABEL_BINARY  = "outbreak_30d"
LABEL_PHASE   = "phase_class"
SKIP_COLS     = {"cell_id", "lat", "lon", "week", LABEL_BINARY, LABEL_PHASE}


# ── Column ordering ───────────────────────────────────────────────────────────

def split_feature_cols(all_cols: list[str]) -> tuple[list[str], list[str]]:
    """Return (self_cols, nbr_cols). nbr_cols end with '_nbr'."""
    self_cols = [c for c in all_cols if not c.endswith("_nbr")]
    nbr_cols  = [c for c in all_cols if c.endswith("_nbr")]
    return self_cols, nbr_cols


def ordered_feat_cols(all_cols: list[str]) -> list[str]:
    """Return feature columns with self-features first, nbr-features last."""
    self_cols, nbr_cols = split_feature_cols(all_cols)
    return self_cols + nbr_cols


# ── Dataset ───────────────────────────────────────────────────────────────────

class LocustDataset(Dataset):
    """
    Sliding-window dataset over (cell_id, week) pairs.

    Parameters
    ----------
    df          : DataFrame for ONE split (already filtered by week range)
    feat_cols   : ordered feature columns (self-feats then nbr-feats)
    seq_len     : number of historical weeks per sample (default 12)
    feat_stats  : dict with 'mean' and 'std' arrays (shape: n_features).
                  If None, computed from this split (use training split stats
                  and pass them in for val/test).
    oversample  : if True, return sample weights for WeightedRandomSampler
    """

    def __init__(self, df: pd.DataFrame, feat_cols: list[str],
                 seq_len: int = 12, feat_stats: dict | None = None,
                 oversample: bool = False):
        self.seq_len   = seq_len
        self.feat_cols = feat_cols
        self.n_features = len(feat_cols)

        # Sort by cell_id then week (critical for window integrity)
        df = df.sort_values(["cell_id", "week"]).reset_index(drop=True)

        # Standardise features
        raw = df[feat_cols].values.astype(np.float32)
        if feat_stats is None:
            mean = np.nanmean(raw, axis=0)
            std  = np.nanstd(raw, axis=0)
            std[std < 1e-8] = 1.0
            feat_stats = {"mean": mean, "std": std}
        self.feat_stats = feat_stats

        normed = (raw - feat_stats["mean"]) / feat_stats["std"]
        normed = np.nan_to_num(normed, nan=0.0)   # fill NaN -> 0 (mean)

        # Labels
        labels_b = df[LABEL_BINARY].values.astype(np.int64)
        labels_p = df[LABEL_PHASE].values.astype(np.int64)

        # Track cell_id per row to enforce window boundaries
        cell_ids = df["cell_id"].values

        # Build index: list of end-row positions where a full window fits
        # A window [i-seq_len+1 .. i] is valid iff all rows belong to the same cell
        self.index  = []        # end row index in normed[]
        self.labels_b = labels_b
        self.labels_p = labels_p
        self.normed   = normed
        self.cell_ids = cell_ids

        for i in range(seq_len - 1, len(df)):
            if cell_ids[i - seq_len + 1] == cell_ids[i]:
                self.index.append(i)

        self.index = np.array(self.index, dtype=np.int64)

        if oversample:
            pos_idx = labels_b[self.index] == 1
            n_pos = pos_idx.sum()
            n_neg = len(self.index) - n_pos
            w = np.where(pos_idx,
                         n_neg / max(n_pos, 1),
                         1.0)
            self.sample_weights = w.astype(np.float64)
        else:
            self.sample_weights = None

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        end   = int(self.index[idx])
        start = end - self.seq_len + 1
        seq   = self.normed[start:end + 1]              # (seq_len, n_features)
        lb    = int(self.labels_b[end])
        lp    = int(self.labels_p[end])
        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(lb,  dtype=torch.long),
            torch.tensor(lp,  dtype=torch.long),
        )

    @property
    def n_nbr_features(self) -> int:
        return sum(1 for c in self.feat_cols if c.endswith("_nbr"))

    @property
    def n_self_features(self) -> int:
        return self.n_features - self.n_nbr_features

    @property
    def positive_count(self) -> int:
        return int((self.labels_b[self.index] == 1).sum())

    @property
    def negative_count(self) -> int:
        return int((self.labels_b[self.index] == 0).sum())


# ── Factory ───────────────────────────────────────────────────────────────────

def make_datasets(features_file: Path = FEATURES_FILE,
                  seq_len: int = 12,
                  sample_cells: int | None = None,
                  seed: int = 42) -> dict[str, LocustDataset]:
    """
    Load features.parquet and return train/val/test LocustDataset objects.
    Pass sample_cells to subsample a fixed number of cells (reproducible).
    """
    print(f"Loading {features_file.name} ...")
    df = pd.read_parquet(features_file)
    print(f"  {df.shape[0]:,} rows x {df.shape[1]} cols")

    if sample_cells is not None:
        rng = np.random.default_rng(seed)
        cells = df["cell_id"].unique()
        chosen = rng.choice(cells, size=min(sample_cells, len(cells)), replace=False)
        df = df[df["cell_id"].isin(chosen)].copy()
        print(f"  Sampled {sample_cells} cells -> {len(df):,} rows")

    # Feature column ordering (self before nbr)
    all_feat = [c for c in df.columns if c not in SKIP_COLS]
    feat_cols = ordered_feat_cols(all_feat)

    # Temporal splits
    week = pd.to_datetime(df["week"])
    train_df = df[week <= "2017-12-31"].copy()
    val_df   = df[(week >= "2018-01-01") & (week <= "2020-12-31")].copy()
    test_df  = df[week >= "2021-01-01"].copy()

    print(f"  Train rows: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    # Build training dataset first to get normalisation stats
    train_ds = LocustDataset(train_df, feat_cols, seq_len=seq_len, oversample=True)
    val_ds   = LocustDataset(val_df,   feat_cols, seq_len=seq_len,
                              feat_stats=train_ds.feat_stats)
    test_ds  = LocustDataset(test_df,  feat_cols, seq_len=seq_len,
                              feat_stats=train_ds.feat_stats)

    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        print(f"  {name:<6}: {len(ds):,} windows  "
              f"(pos {ds.positive_count:,} / neg {ds.negative_count:,})")

    return {"train": train_ds, "val": val_ds, "test": test_ds,
            "feat_cols": feat_cols}


def make_loaders(datasets: dict, batch_size: int = 256,
                 num_workers: int = 0) -> dict[str, DataLoader]:
    """Create DataLoaders from datasets dict returned by make_datasets()."""
    loaders = {}

    train_ds = datasets["train"]
    if train_ds.sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=train_ds.sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        loaders["train"] = DataLoader(train_ds, batch_size=batch_size,
                                       sampler=sampler, num_workers=num_workers,
                                       pin_memory=True)
    else:
        loaders["train"] = DataLoader(train_ds, batch_size=batch_size,
                                       shuffle=True, num_workers=num_workers,
                                       pin_memory=True)

    for split in ("val", "test"):
        loaders[split] = DataLoader(datasets[split], batch_size=batch_size * 2,
                                     shuffle=False, num_workers=num_workers,
                                     pin_memory=True)
    return loaders
