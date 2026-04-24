#!/usr/bin/env python3
"""
src/features/build_features.py
-------------------------------
Phase 1: Build the feature matrix from all five processed data sources.

Steps
-----
1. Load all five processed parquets (FAO, CHIRPS, ERA5, MODIS, SMAP)
2. Merge climate sources onto a common (lat, lon, week) backbone
3. Engineer features:
     - Rolling rainfall sums (4, 8, 12 weeks)
     - Lag features for all climate variables (t-4w, t-8w)
     - Spatial context: mean of 8 Moore-neighbourhood cells for each feature
4. Build labels from FAO records:
     - outbreak_30d: 1 if any FAO occurrence in this cell within +30 days
     - phase_class: max phase observed in cell within +30 days (0 if none)
5. Write data/processed/features.parquet

Output schema (features.parquet)
---------------------------------
    cell_id               str
    lat, lon              float64
    week                  datetime64[ns]
    --- climate ---
    rainfall_weekly_mm    float64
    rainfall_anomaly      float64
    rainfall_roll_4w      float64   rolling 4-week sum
    rainfall_roll_8w      float64
    rainfall_roll_12w     float64
    temp_mean_c           float64
    temp_anomaly          float64
    wind_speed_ms         float64
    wind_dir_sin          float64
    wind_dir_cos          float64
    humidity_pct          float64
    ndvi                  float64
    ndvi_anomaly          float64
    soil_moisture_surface   float64
    soil_moisture_rootzone  float64
    --- lag features (suffix _lag4w, _lag8w) ---
    rainfall_weekly_mm_lag4w ... soil_moisture_rootzone_lag8w
    --- spatial context (suffix _nbr) ---
    rainfall_weekly_mm_nbr ... soil_moisture_rootzone_nbr
    --- labels ---
    outbreak_30d          int8    (0 or 1)
    phase_class           int8    (0-3)

Usage
-----
    python src/features/build_features.py [--sample N]

    --sample N  Only process N randomly chosen cells (for fast dev iteration).
                Default: all cells.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED = ROOT / "data/processed"
OUT_FILE  = PROCESSED / "features.parquet"

FAO_FILE    = PROCESSED / "fao_clean.parquet"
CHIRPS_FILE = PROCESSED / "chirps_weekly_0p1deg.parquet"
ERA5_FILE   = PROCESSED / "era5_weekly_0p1deg.parquet"
MODIS_FILE  = PROCESSED / "modis_ndvi_weekly_0p1deg.parquet"
SMAP_FILE   = PROCESSED / "smap_weekly_0p1deg.parquet"

# ── Config ─────────────────────────────────────────────────────────────────────
RESOLUTION     = 0.1
ROLLING_WINS   = [4, 8, 12]          # weeks
LAG_WEEKS      = [4, 8]
LABEL_DAYS     = 30                  # FAO records within this window -> positive
LABEL_WEEKS    = int(np.ceil(LABEL_DAYS / 7))

# Features to create lags and spatial context for
LAG_FEATURES = [
    "rainfall_weekly_mm", "rainfall_anomaly",
    "temp_mean_c", "temp_anomaly",
    "wind_speed_ms", "wind_dir_sin", "wind_dir_cos", "humidity_pct",
    "ndvi", "ndvi_anomaly",
    "soil_moisture_surface", "soil_moisture_rootzone",
]

NBR_FEATURES = LAG_FEATURES  # same set for spatial context


def _snap(values: np.ndarray) -> np.ndarray:
    return np.round(values / RESOLUTION) * RESOLUTION


def _cell_id(lat_arr, lon_arr) -> pd.Series:
    return (
        pd.Series(np.round(lat_arr, 1).astype(str))
        + "_"
        + pd.Series(np.round(lon_arr, 1).astype(str))
    )


# ── 1. Load ────────────────────────────────────────────────────────────────────

def load_sources() -> dict:
    """Load all five processed parquets. Raise early with a clear message if missing."""
    sources = {
        "fao":    FAO_FILE,
        "chirps": CHIRPS_FILE,
        "era5":   ERA5_FILE,
        "modis":  MODIS_FILE,
        "smap":   SMAP_FILE,
    }
    missing = [name for name, path in sources.items() if not path.exists()]
    if missing:
        print(f"[ERROR] Missing processed files: {missing}")
        print("  Run the corresponding download_*.py scripts first.")
        sys.exit(1)

    print("Loading processed parquets ...")
    loaded = {}
    for name, path in sources.items():
        df = pd.read_parquet(path)
        loaded[name] = df
        print(f"  {name:<8} {len(df):>10,} rows")
    return loaded


# ── 2. Merge climate sources ───────────────────────────────────────────────────

def build_base_grid(sources: dict, sample_cells: int | None = None) -> pd.DataFrame:
    """
    Join ERA5, MODIS, SMAP on (lat, lon, week).
    Then left-join CHIRPS (may have fewer cells — fills NaN with 0 for rain).

    ERA5 is the backbone because it has the fullest spatial and temporal coverage.
    """
    print("\nBuilding base grid ...")

    era5  = sources["era5"].copy()
    modis = sources["modis"][["lat", "lon", "week", "ndvi", "ndvi_anomaly"]].copy()
    smap  = sources["smap"][["lat", "lon", "week",
                              "soil_moisture_surface", "soil_moisture_rootzone"]].copy()
    chirps = sources["chirps"][["lat", "lon", "week",
                                 "rainfall_weekly_mm", "rainfall_anomaly",
                                 "rainfall_clim_mm"]].copy()

    # Snap all coordinates to the shared grid
    for df in [era5, modis, smap, chirps]:
        df["lat"] = _snap(df["lat"].values)
        df["lon"] = _snap(df["lon"].values)

    # Optional: subsample cells for faster development
    if sample_cells is not None:
        all_cells = era5[["lat", "lon"]].drop_duplicates()
        sampled   = all_cells.sample(n=min(sample_cells, len(all_cells)), random_state=42)
        era5  = era5.merge(sampled,  on=["lat", "lon"])
        modis = modis.merge(sampled, on=["lat", "lon"])
        smap  = smap.merge(sampled,  on=["lat", "lon"])
        chirps = chirps.merge(sampled, on=["lat", "lon"])
        print(f"  Sampled {len(sampled):,} cells for development run.")

    # Merge ERA5 + MODIS + SMAP (all share the same grid)
    df = era5.merge(modis, on=["lat", "lon", "week"], how="left")
    df = df.merge(smap,  on=["lat", "lon", "week"], how="left")

    # CHIRPS may cover fewer cells — left join, then fill rainfall NaN with 0
    df = df.merge(chirps, on=["lat", "lon", "week"], how="left")
    df["rainfall_weekly_mm"] = df["rainfall_weekly_mm"].fillna(0.0)
    df["rainfall_anomaly"]   = df["rainfall_anomaly"].fillna(0.0)
    df["rainfall_clim_mm"]   = df["rainfall_clim_mm"].fillna(0.0)

    # Add cell_id
    df["cell_id"] = _cell_id(df["lat"].values, df["lon"].values)

    # Sort for rolling/lag operations
    df = df.sort_values(["cell_id", "week"]).reset_index(drop=True)

    print(f"  Base grid: {len(df):,} rows, {df['cell_id'].nunique():,} cells, "
          f"{df['week'].nunique()} weeks")
    return df


# ── 3. Rolling rainfall ────────────────────────────────────────────────────────

def add_rolling_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling sum columns for rainfall over 4, 8, 12 weeks."""
    print("Adding rolling rainfall features ...")

    rain = df.groupby("cell_id")["rainfall_weekly_mm"]
    for win in ROLLING_WINS:
        col = f"rainfall_roll_{win}w"
        df[col] = rain.transform(
            lambda s: s.rolling(win, min_periods=1).sum()
        )

    return df


# ── 4. Lag features ───────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shift each feature by LAG_WEEKS within each cell group.
    Naming: <feature>_lag4w, <feature>_lag8w
    """
    print("Adding lag features ...")

    existing = [f for f in LAG_FEATURES if f in df.columns]
    grouped  = df.groupby("cell_id")

    lag_frames = []
    for lag in LAG_WEEKS:
        lag_df = grouped[existing].shift(lag)
        lag_df.columns = [f"{c}_lag{lag}w" for c in existing]
        lag_frames.append(lag_df)

    df = pd.concat([df] + lag_frames, axis=1)
    return df


# ── 5. Spatial context ─────────────────────────────────────────────────────────

def add_spatial_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each cell, compute the mean of its 8 Moore-neighbourhood cells
    for each feature in NBR_FEATURES. Columns are named <feature>_nbr.

    Strategy: for each of the 8 direction offsets, shift lat/lon, merge back to
    get the neighbour's feature values, then average across all 8 directions.
    """
    print("Adding spatial context (Moore neighbourhood) ...")

    existing = [f for f in NBR_FEATURES if f in df.columns]

    # Detect the actual grid step from the data (works for both 0.1 and 0.5 deg)
    lat_vals = np.sort(df["lat"].unique())
    lon_vals = np.sort(df["lon"].unique())
    lat_step = float(np.diff(lat_vals).min().round(4)) if len(lat_vals) > 1 else RESOLUTION
    lon_step = float(np.diff(lon_vals).min().round(4)) if len(lon_vals) > 1 else RESOLUTION
    print(f"  Detected grid step: lat={lat_step} deg, lon={lon_step} deg")

    # Index the base data by (lat, lon, week) for fast lookup
    base = df[["lat", "lon", "week"] + existing].copy()

    OFFSETS = [
        (-lat_step,  0),           # N
        ( lat_step,  0),           # S
        ( 0,        -lon_step),    # W
        ( 0,         lon_step),    # E
        (-lat_step, -lon_step),    # NW
        (-lat_step,  lon_step),    # NE
        ( lat_step, -lon_step),    # SW
        ( lat_step,  lon_step),    # SE
    ]

    # Accumulate neighbour sums
    nbr_sum   = pd.DataFrame(0.0, index=df.index, columns=existing)
    nbr_count = pd.DataFrame(0,   index=df.index, columns=existing)

    for dlat, dlon in OFFSETS:
        ndigits = max(1, -int(np.floor(np.log10(min(lat_step, lon_step)))))
        shifted = base.copy()
        shifted["lat"] = (shifted["lat"] + dlat).round(ndigits)
        shifted["lon"] = (shifted["lon"] + dlon).round(ndigits)

        merged = df[["lat", "lon", "week"]].merge(
            shifted, on=["lat", "lon", "week"], how="left"
        )

        for feat in existing:
            valid_mask = merged[feat].notna()
            nbr_sum.loc[valid_mask, feat] += merged.loc[valid_mask, feat].values
            nbr_count.loc[valid_mask, feat] += 1

    # Compute mean; cells with no neighbours (border effects) get NaN
    nbr_mean = nbr_sum.div(nbr_count.replace(0, np.nan))
    nbr_mean.columns = [f"{c}_nbr" for c in existing]

    df = pd.concat([df, nbr_mean], axis=1)
    return df


# ── 6. Labels ─────────────────────────────────────────────────────────────────

def add_labels(df: pd.DataFrame, fao: pd.DataFrame) -> pd.DataFrame:
    """
    For each (cell_id, week), check whether any FAO outbreak is recorded
    in that cell within the next LABEL_DAYS days.

    outbreak_30d  : 1 if any record exists, else 0
    phase_class   : max phase_class in the window (0 if no record)
    """
    print("Building labels ...")

    # Snap FAO records to the nearest cell that actually exists in the feature
    # matrix. We cannot assume a fixed grid resolution here because the
    # synthetic data may use coarser steps (0.5 deg) than the real 0.1 deg grid.
    from scipy.spatial import cKDTree

    grid_cells = df[["lat", "lon", "cell_id"]].drop_duplicates(subset=["cell_id"])
    tree = cKDTree(grid_cells[["lat", "lon"]].values)

    fao = fao.copy()
    fao["date"] = pd.to_datetime(fao["date"])

    lat_col = "cell_lat" if "cell_lat" in fao.columns else "latitude"
    lon_col = "cell_lon" if "cell_lon" in fao.columns else "longitude"
    fao_coords = fao[[lat_col, lon_col]].values

    # Find nearest grid cell for each FAO record
    dist, idx = tree.query(fao_coords, k=1)
    fao["cell_id"] = grid_cells["cell_id"].iloc[idx].values

    # Warn if any FAO record is more than 1 deg from the nearest cell
    far = (dist > 1.0).sum()
    if far:
        print(f"  WARNING: {far} FAO records are >1 deg from nearest grid cell "
              f"(likely outside study region).")
    print(f"  FAO records matched to {fao['cell_id'].nunique():,} unique grid cells"
          f" (time range: {fao['date'].min().date()} - {fao['date'].max().date()})")

    # Build a (cell_id, week) -> (any_outbreak, max_phase) lookup
    # For each FAO record, find which weekly window it falls in
    # A record on date d falls in window [week_start, week_start + LABEL_DAYS)
    # for any week_start <= d < week_start + LABEL_DAYS
    # Equivalently: week_start in (d - LABEL_DAYS, d]
    # We implement this by creating label rows for each look-back week

    def _to_week_key(dates: pd.Series) -> pd.Series:
        """
        Convert dates to the Monday that ENDS the W-MON period they fall in.
        pd.date_range(..., freq="W-MON") produces these Mondays, so we must
        match that convention here.

        A W-MON period ends on Monday.  For a given date:
          - If Monday  -> same day
          - If Tue-Sun -> next Monday
        """
        dow = dates.dt.dayofweek          # 0=Monday … 6=Sunday
        days_to_next_mon = (7 - dow) % 7  # 0 if already Monday
        return (dates + pd.to_timedelta(days_to_next_mon, unit="D")).dt.normalize()

    label_records = []
    for lag_weeks in range(LABEL_WEEKS + 1):
        shifted = fao.copy()
        shifted["week"] = _to_week_key(
            shifted["date"] - pd.Timedelta(weeks=lag_weeks)
        )
        shifted["outbreak_30d"] = 1
        label_records.append(
            shifted[["cell_id", "week", "outbreak_30d", "phase_class"]]
        )

    labels = pd.concat(label_records, ignore_index=True)

    # Aggregate: if any record maps to (cell_id, week), outbreak=1
    labels = labels.groupby(["cell_id", "week"], as_index=False).agg(
        outbreak_30d=("outbreak_30d", "max"),
        phase_class=("phase_class", "max"),
    )

    # Merge into feature matrix
    df = df.merge(labels, on=["cell_id", "week"], how="left")
    df["outbreak_30d"] = df["outbreak_30d"].fillna(0).astype("int8")
    df["phase_class"]  = df["phase_class"].fillna(0).astype("int8")

    pos = df["outbreak_30d"].sum()
    total = len(df)
    print(f"  Positive labels: {pos:,} / {total:,} "
          f"({100*pos/total:.2f}%)")
    print(f"  Phase distribution: {df['phase_class'].value_counts().sort_index().to_dict()}")

    return df


# ── 7. Finalise ────────────────────────────────────────────────────────────────

def finalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop helper columns, downcast numeric types, and enforce column order.
    """
    drop_cols = [c for c in ["rainfall_clim_mm"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Downcast float64 -> float32 to halve memory footprint
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype("float32")

    # Canonical column order
    id_cols     = ["cell_id", "lat", "lon", "week"]
    label_cols  = ["outbreak_30d", "phase_class"]
    feat_cols   = [c for c in df.columns
                   if c not in id_cols + label_cols]
    df = df[id_cols + feat_cols + label_cols]

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build LocustWatch feature matrix")
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Process only N randomly chosen cells (fast dev mode)"
    )
    args = parser.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  LocustWatch AI -- Phase 1: Feature Engineering")
    print("=" * 60)

    sources = load_sources()

    df  = build_base_grid(sources, sample_cells=args.sample)
    df  = add_rolling_rainfall(df)
    df  = add_lag_features(df)
    df  = add_spatial_context(df)
    df  = add_labels(df, sources["fao"])
    df  = finalise(df)

    print(f"\nFinal feature matrix:")
    print(f"  Shape:   {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory:  {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    col_groups = {
        "ID":      [c for c in df.columns if c in ("cell_id","lat","lon","week")],
        "Climate": [c for c in df.columns if any(c.startswith(p)
                    for p in ("rain","temp","wind","humidity","ndvi","soil"))
                    and "lag" not in c and "nbr" not in c],
        "Lag":     [c for c in df.columns if "lag" in c],
        "Spatial": [c for c in df.columns if "nbr" in c],
        "Labels":  ["outbreak_30d", "phase_class"],
    }
    for grp, cols in col_groups.items():
        print(f"  {grp:<10} {len(cols)} cols")

    df.to_parquet(OUT_FILE, index=False)
    print(f"\n  Saved -> {OUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
