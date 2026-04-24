#!/usr/bin/env python3
"""
src/data/download_fao.py
------------------------
Download and clean FAO Desert Locust occurrence records.

Real data:  https://locust-hub-hqfao.hub.arcgis.com/
            Export as CSV from the ArcGIS hub (manual step -- no public API).
            Save the CSV to data/raw/fao/fao_locust_occurrences.csv

Synthetic fallback: if the raw file is absent, generates realistic synthetic
records so the rest of the pipeline can run end-to-end without real data.

Output: data/processed/fao_clean.parquet
Schema:
    latitude    float64   -- record centroid
    longitude   float64
    date        datetime64[ns]
    country     str
    phase_raw   str       -- original FAO phase string
    phase_class int       -- 0=unknown 1=solitarious 2=gregarious 3=swarming
    species     str
    week        datetime64[ns]  -- Monday-anchored week
    cell_lat    float64   -- 0.1° grid cell centroid
    cell_lon    float64
    cell_id     str       -- "{cell_lat}_{cell_lon}"
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RAW_FILE = ROOT / "data/raw/fao/fao_locust_occurrences.csv"
OUT_FILE = ROOT / "data/processed/fao_clean.parquet"

# ── Phase mapping ─────────────────────────────────────────────────────────────
PHASE_MAP = {
    "solitarious": 1, "solitary": 1,
    "transiens": 2, "transient": 2, "gregarious": 2,
    "swarming": 3, "swarm": 3, "band": 3, "hopper band": 3,
}

# ── Countries in study region ─────────────────────────────────────────────────
STUDY_COUNTRIES = [
    "Ethiopia", "Somalia", "Kenya", "Sudan", "South Sudan",
    "Eritrea", "Djibouti", "Yemen", "Saudi Arabia", "Oman",
    "Pakistan", "India", "Iran", "Afghanistan",
    "Chad", "Niger", "Mali", "Mauritania",
]


def _snap_to_grid(values: np.ndarray, resolution: float = 0.1) -> np.ndarray:
    """Round coordinates to nearest grid cell centre."""
    return np.round(values / resolution) * resolution


def _load_real(path: Path) -> pd.DataFrame:
    """Load and normalise a FAO CSV export."""
    df = pd.read_csv(path, low_memory=False)

    # FAO exports use varied column names -- normalise to lowercase
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    # Required column aliases
    col_aliases = {
        "lat": "latitude", "lon": "longitude", "lng": "longitude",
        "startdate": "date", "start_date": "date", "observation_date": "date",
        "locust_phase": "phase_raw", "phase": "phase_raw",
        "country_name": "country",
    }
    df = df.rename(columns=col_aliases)

    required = {"latitude", "longitude", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"FAO CSV missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "latitude", "longitude"])
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)

    if "phase_raw" not in df.columns:
        df["phase_raw"] = "unknown"
    if "species" not in df.columns:
        df["species"] = "Schistocerca gregaria"
    if "country" not in df.columns:
        df["country"] = "unknown"

    return df


def _make_synthetic() -> pd.DataFrame:
    """
    Generate synthetic FAO records that mimic real outbreak patterns.
    Concentrates records in the Horn of Africa and Arabian Peninsula.
    """
    print("  [synthetic] Real FAO file not found -- generating synthetic data.")
    rng = np.random.default_rng(42)

    n = 3000
    # Cluster centres: Horn of Africa, Yemen, Pakistan
    cluster_centres = [
        (10.0, 42.0), (8.0, 45.0), (12.0, 44.0),   # Ethiopia/Somalia
        (15.0, 44.0), (17.0, 47.0),                  # Yemen
        (28.0, 67.0), (26.0, 65.0),                  # Pakistan
        (13.0, 25.0), (16.0, 30.0),                  # Sudan
    ]
    weights = [0.18, 0.15, 0.12, 0.12, 0.08, 0.10, 0.08, 0.09, 0.08]

    chosen = rng.choice(len(cluster_centres), size=n, p=weights)
    lats = np.array([cluster_centres[i][0] for i in chosen]) + rng.normal(0, 1.5, n)
    lons = np.array([cluster_centres[i][1] for i in chosen]) + rng.normal(0, 2.0, n)

    # Clip to study region
    lats = np.clip(lats, -5.0, 35.0)
    lons = np.clip(lons, -20.0, 75.0)

    dates = pd.to_datetime("1985-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 38, n), unit="D"
    )

    phase_classes = rng.choice([1, 2, 3], size=n, p=[0.5, 0.3, 0.2])
    phase_labels = {1: "solitarious", 2: "gregarious", 3: "swarming"}
    phase_raw = [phase_labels[p] for p in phase_classes]

    country_by_lat = []
    for lat, lon in zip(lats, lons):
        if lat < 12 and lon > 38:
            country_by_lat.append("Somalia")
        elif lat < 15 and lon > 35:
            country_by_lat.append("Ethiopia")
        elif lat > 14 and lat < 20 and lon > 42:
            country_by_lat.append("Yemen")
        elif lat > 24 and lon > 60:
            country_by_lat.append("Pakistan")
        else:
            country_by_lat.append(rng.choice(STUDY_COUNTRIES))

    return pd.DataFrame({
        "latitude": lats,
        "longitude": lons,
        "date": dates,
        "country": country_by_lat,
        "phase_raw": phase_raw,
        "phase_class": phase_classes,
        "species": "Schistocerca gregaria",
    })


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns and snap to grid."""
    if "phase_class" not in df.columns:
        df["phase_class"] = (
            df["phase_raw"]
            .str.lower()
            .str.strip()
            .map(PHASE_MAP)
            .fillna(0)
            .astype(int)
        )

    # Monday-anchored week
    df["week"] = df["date"].dt.to_period("W-MON").apply(
        lambda p: p.start_time
    )

    # Snap to 0.1° grid
    df["cell_lat"] = _snap_to_grid(df["latitude"].values)
    df["cell_lon"] = _snap_to_grid(df["longitude"].values)
    df["cell_id"] = (
        df["cell_lat"].round(1).astype(str)
        + "_"
        + df["cell_lon"].round(1).astype(str)
    )

    # Filter to study region
    df = df[
        (df["latitude"].between(-5.0, 35.0)) &
        (df["longitude"].between(-20.0, 75.0))
    ].copy()

    return df.reset_index(drop=True)


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if RAW_FILE.exists():
        print(f"Loading real FAO data from {RAW_FILE}")
        df = _load_real(RAW_FILE)
    else:
        df = _make_synthetic()

    df = _clean(df)

    print(f"  Records:   {len(df):,}")
    print(f"  Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  Countries:  {df['country'].nunique()}")
    print(f"  Phases:     {df['phase_class'].value_counts().to_dict()}")

    df.to_parquet(OUT_FILE, index=False)
    print(f"  Saved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
