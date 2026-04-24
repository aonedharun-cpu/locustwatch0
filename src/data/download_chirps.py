#!/usr/bin/env python3
"""
src/data/download_chirps.py
---------------------------
Download CHIRPS v2.0 daily rainfall NetCDF files, regrid to 0.1°, and
aggregate to weekly sums.

Real data: https://www.chc.ucsb.edu/data/chirps
           Files: chirps-v2.0.YYYY.days_p05.nc  (one per year, ~180 MB each)

The script downloads via HTTP -- no login required for CHIRPS.
Run with internet access. Downloads go to data/raw/chirps/.

Synthetic fallback: if raw files are absent, generates plausible synthetic
rainfall so the pipeline runs without internet.

Output: data/processed/chirps_weekly_0p1deg.parquet
Schema:
    lat              float64
    lon              float64
    week             datetime64[ns]  -- Monday-anchored
    rainfall_weekly_mm  float64      -- weekly total
    rainfall_clim_mm    float64      -- climatological mean for that week-of-year
    rainfall_anomaly    float64      -- rainfall_weekly_mm - rainfall_clim_mm
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RAW_DIR = ROOT / "data/raw/chirps"
OUT_FILE = ROOT / "data/processed/chirps_weekly_0p1deg.parquet"

# Study region from config
LAT_MIN, LAT_MAX = -5.0, 35.0
LON_MIN, LON_MAX = -20.0, 75.0

# Years to download (adjust as needed -- each file ~180 MB)
YEARS = list(range(2015, 2024))

CHIRPS_URL_TEMPLATE = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/"
    "chirps-v2.0.{year}.days_p05.nc"
)


def _snap_to_grid(values: np.ndarray, resolution: float = 0.1) -> np.ndarray:
    return np.round(values / resolution) * resolution


def download_raw(years: list[int]) -> list[Path]:
    """Download CHIRPS NetCDF files for requested years. Skip if already present."""
    try:
        import requests
    except ImportError:
        print("  requests not installed -- skipping download. Run: pip install requests")
        return []

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for year in years:
        url = CHIRPS_URL_TEMPLATE.format(year=year)
        dest = RAW_DIR / f"chirps-v2.0.{year}.days_p05.nc"
        if dest.exists():
            print(f"  {dest.name} already present -- skipping")
            downloaded.append(dest)
            continue

        print(f"  Downloading {url} ...")
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            downloaded.append(dest)
            print(f"  Saved {dest.name}")
        except Exception as e:
            print(f"  Failed to download {year}: {e}")

    return downloaded


def process_real(nc_files: list[Path]) -> pd.DataFrame:
    """
    Load CHIRPS NetCDF files, clip to study region, resample to weekly 0.1°.
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray required: pip install xarray netcdf4")

    lats = np.arange(LAT_MIN, LAT_MAX + 0.1, 0.1)
    lons = np.arange(LON_MIN, LON_MAX + 0.1, 0.1)

    records = []
    for nc_path in sorted(nc_files):
        print(f"  Processing {nc_path.name} ...")
        ds = xr.open_dataset(nc_path)

        # CHIRPS uses 'precip' variable, lat/lon dims
        precip = ds["precip"].sel(
            latitude=slice(LAT_MIN, LAT_MAX),
            longitude=slice(LON_MIN, LON_MAX),
        )

        # Regrid from 0.05° to 0.1° via coarsen (2x2 sum)
        precip_01 = precip.coarsen(latitude=2, longitude=2, boundary="trim").sum()

        # Resample daily -> weekly (Monday-anchored)
        precip_weekly = precip_01.resample(time="W-MON").sum()

        df = precip_weekly.to_dataframe(name="rainfall_weekly_mm").reset_index()
        df = df.rename(columns={"time": "week", "latitude": "lat", "longitude": "lon"})
        df["lat"] = _snap_to_grid(df["lat"].values)
        df["lon"] = _snap_to_grid(df["lon"].values)
        records.append(df)
        ds.close()

    df = pd.concat(records, ignore_index=True)
    df["rainfall_weekly_mm"] = df["rainfall_weekly_mm"].clip(lower=0)
    df = df.groupby(["lat", "lon", "week"], as_index=False)["rainfall_weekly_mm"].sum()
    return df


def _make_synthetic() -> pd.DataFrame:
    """
    Generate synthetic weekly rainfall with seasonal structure.
    Uses a simple seasonal sine wave + noise model per grid cell.
    """
    print("  [synthetic] No CHIRPS files found -- generating synthetic rainfall.")
    rng = np.random.default_rng(0)

    lats = np.arange(LAT_MIN, LAT_MAX + 0.1, 0.1).round(1)
    lons = np.arange(LON_MIN, LON_MAX + 0.1, 0.1).round(1)
    weeks = pd.date_range("2015-01-05", "2023-12-25", freq="W-MON")

    # Sample a subset of cells to keep file size manageable
    rng2 = np.random.default_rng(1)
    lat_sample = rng2.choice(lats, size=40, replace=False)
    lon_sample = rng2.choice(lons, size=40, replace=False)

    records = []
    for lat in lat_sample:
        for lon in lon_sample:
            # Season peaks around week 20 (May) and week 40 (Oct) for East Africa
            week_of_year = np.array([w.isocalendar()[1] for w in weeks])
            seasonal = (
                15 * np.sin(2 * np.pi * week_of_year / 52 - np.pi / 3) + 20
                + 5 * np.sin(4 * np.pi * week_of_year / 52)
            ).clip(0)
            noise = rng.exponential(scale=seasonal.clip(min=1))
            rainfall = (seasonal + noise).clip(0)

            records.append(pd.DataFrame({
                "lat": lat,
                "lon": lon,
                "week": weeks,
                "rainfall_weekly_mm": rainfall,
            }))

    return pd.concat(records, ignore_index=True)


def _add_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Add climatological mean and anomaly columns."""
    df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
    clim = (
        df.groupby(["lat", "lon", "week_of_year"])["rainfall_weekly_mm"]
        .mean()
        .rename("rainfall_clim_mm")
        .reset_index()
    )
    df = df.merge(clim, on=["lat", "lon", "week_of_year"])
    df["rainfall_anomaly"] = df["rainfall_weekly_mm"] - df["rainfall_clim_mm"]
    df = df.drop(columns=["week_of_year"])
    return df


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    nc_files = sorted(RAW_DIR.glob("chirps-v2.0.*.nc")) if RAW_DIR.exists() else []

    if not nc_files:
        # Attempt download first
        nc_files = download_raw(YEARS)

    if nc_files:
        df = process_real(nc_files)
    else:
        df = _make_synthetic()

    df = _add_climatology(df)

    print(f"  Rows:       {len(df):,}")
    print(f"  Cells:      {df[['lat','lon']].drop_duplicates().shape[0]:,}")
    print(f"  Weeks:      {df['week'].nunique()} "
          f"({df['week'].min().date()} -> {df['week'].max().date()})")
    print(f"  Rain range: {df['rainfall_weekly_mm'].min():.1f} - "
          f"{df['rainfall_weekly_mm'].max():.1f} mm/week")

    df.to_parquet(OUT_FILE, index=False)
    print(f"  Saved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
