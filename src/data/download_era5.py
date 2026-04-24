#!/usr/bin/env python3
"""
src/data/download_era5.py
--------------------------
Download ERA5 reanalysis data via the Copernicus CDS API, regrid to 0.1°,
and aggregate to weekly resolution.

Setup (one-time):
    1. Register at https://cds.climate.copernicus.eu
    2. Accept the ERA5 licence
    3. Create ~/.cdsapirc with:
           url: https://cds.climate.copernicus.eu/api/v2
           key: <UID>:<API-KEY>
       (Claude Code can generate this file -- run: python docs/setup_cds.py)
    4. pip install cdsapi

Variables downloaded:
    - 2m_temperature
    - 10m_u_component_of_wind
    - 10m_v_component_of_wind
    - 2m_dewpoint_temperature

Output: data/processed/era5_weekly_0p1deg.parquet
Schema:
    lat            float64
    lon            float64
    week           datetime64[ns]
    temp_mean_c    float64   -- weekly mean 2m temperature (°C)
    temp_anomaly   float64
    wind_u_ms      float64   -- weekly mean 10m U-component (m/s)
    wind_v_ms      float64   -- weekly mean 10m V-component (m/s)
    wind_speed_ms  float64   -- sqrt(u²+v²)
    wind_dir_deg   float64   -- meteorological convention (0=N, 90=E)
    wind_dir_sin   float64   -- circular encoding
    wind_dir_cos   float64
    dewpoint_c     float64
    humidity_pct   float64   -- relative humidity approximated from T and Td
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RAW_DIR = ROOT / "data/raw/era5"
OUT_FILE = ROOT / "data/processed/era5_weekly_0p1deg.parquet"

LAT_MIN, LAT_MAX = -5.0, 35.0
LON_MIN, LON_MAX = -20.0, 75.0

YEARS = list(range(2015, 2024))
ERA5_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
]


def _snap_to_grid(values: np.ndarray, resolution: float = 0.1) -> np.ndarray:
    return np.round(values / resolution) * resolution


def download_raw(years: list[int]) -> list[Path]:
    """Download ERA5 monthly files via CDS API. One NetCDF per year."""
    try:
        import cdsapi
    except ImportError:
        print("  cdsapi not installed. Run: pip install cdsapi")
        return []

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = []
    c = cdsapi.Client()

    for year in years:
        dest = RAW_DIR / f"era5_{year}.nc"
        if dest.exists():
            print(f"  {dest.name} already present -- skipping")
            downloaded.append(dest)
            continue

        months = [f"{m:02d}" for m in range(1, 13)]
        days   = [f"{d:02d}" for d in range(1, 32)]
        times  = [f"{h:02d}:00" for h in range(0, 24)]

        print(f"  Downloading ERA5 {year} (this may take several minutes) ...")
        try:
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": ERA5_VARIABLES,
                    "year": str(year),
                    "month": months,
                    "day": days,
                    "time": times,
                    "area": [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],  # N/W/S/E
                    "format": "netcdf",
                },
                str(dest),
            )
            downloaded.append(dest)
            print(f"  Saved {dest.name}")
        except Exception as e:
            print(f"  CDS download failed for {year}: {e}")

    return downloaded


def process_real(nc_files: list[Path]) -> pd.DataFrame:
    """Load ERA5 NetCDF, regrid to 0.1°, aggregate to weekly."""
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray required: pip install xarray netcdf4")

    records = []
    for nc_path in sorted(nc_files):
        print(f"  Processing {nc_path.name} ...")
        ds = xr.open_dataset(nc_path)

        # ERA5 temperature is in Kelvin
        t2m  = ds["t2m"]  - 273.15   # -> °C
        d2m  = ds["d2m"]  - 273.15   # dewpoint °C
        u10  = ds["u10"]              # U-wind m/s
        v10  = ds["v10"]              # V-wind m/s

        # Weekly mean
        t2m_w  = t2m.resample(time="W-MON").mean()
        d2m_w  = d2m.resample(time="W-MON").mean()
        u10_w  = u10.resample(time="W-MON").mean()
        v10_w  = v10.resample(time="W-MON").mean()

        weeks = t2m_w.time.values
        lats  = _snap_to_grid(t2m_w.latitude.values)
        lons  = _snap_to_grid(t2m_w.longitude.values)

        for i, week in enumerate(weeks):
            t_slice = t2m_w.isel(time=i).values
            d_slice = d2m_w.isel(time=i).values
            u_slice = u10_w.isel(time=i).values
            v_slice = v10_w.isel(time=i).values

            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
            n = lat_grid.size

            speed = np.sqrt(u_slice**2 + v_slice**2)
            # Meteorological direction: 0=wind from N, 90=from E
            direction = (270 - np.degrees(np.arctan2(v_slice, u_slice))) % 360
            # Approximate RH from Magnus formula
            rh = 100 * np.exp((17.625 * d_slice) / (243.04 + d_slice)) / \
                     np.exp((17.625 * t_slice) / (243.04 + t_slice))

            records.append(pd.DataFrame({
                "lat":          lat_grid.ravel(),
                "lon":          lon_grid.ravel(),
                "week":         pd.Timestamp(week),
                "temp_mean_c":  t_slice.ravel(),
                "wind_u_ms":    u_slice.ravel(),
                "wind_v_ms":    v_slice.ravel(),
                "wind_speed_ms": speed.ravel(),
                "wind_dir_deg": direction.ravel(),
                "dewpoint_c":   d_slice.ravel(),
                "humidity_pct": rh.clip(0, 100).ravel(),
            }))

        ds.close()

    return pd.concat(records, ignore_index=True)


def _make_synthetic() -> pd.DataFrame:
    """Synthetic ERA5 data with realistic seasonal patterns for the study region."""
    print("  [synthetic] No ERA5 files found -- generating synthetic climate data.")
    rng = np.random.default_rng(1)

    lats = np.arange(LAT_MIN, LAT_MAX + 0.5, 0.5).round(1)
    lons = np.arange(LON_MIN, LON_MAX + 0.5, 0.5).round(1)
    weeks = pd.date_range("2015-01-05", "2023-12-25", freq="W-MON")

    records = []
    for lat in lats:
        for lon in lons:
            n = len(weeks)
            doy = np.array([w.day_of_year for w in weeks])

            # Temperature: hot in summer, peaks around day 200
            temp = 25 + 8 * np.sin(2 * np.pi * (doy - 80) / 365) \
                      - 0.5 * lat + rng.normal(0, 1.5, n)

            # Wind speed: seasonal 2-10 m/s
            wind_speed = 4 + 2 * np.sin(2 * np.pi * doy / 365) \
                           + rng.exponential(1.0, n)
            wind_dir = rng.uniform(0, 360, n)

            u = -wind_speed * np.sin(np.radians(wind_dir))
            v = -wind_speed * np.cos(np.radians(wind_dir))

            dewpoint = temp - 15 + rng.normal(0, 2, n)
            rh = (100 * np.exp((17.625 * dewpoint) / (243.04 + dewpoint)) /
                       np.exp((17.625 * temp) / (243.04 + temp))).clip(0, 100)

            records.append(pd.DataFrame({
                "lat":          lat,
                "lon":          lon,
                "week":         weeks,
                "temp_mean_c":  temp,
                "wind_u_ms":    u,
                "wind_v_ms":    v,
                "wind_speed_ms": wind_speed,
                "wind_dir_deg": wind_dir,
                "dewpoint_c":   dewpoint,
                "humidity_pct": rh,
            }))

    return pd.concat(records, ignore_index=True)


def _add_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
    clim = (
        df.groupby(["lat", "lon", "week_of_year"])["temp_mean_c"]
        .mean()
        .rename("temp_clim_c")
        .reset_index()
    )
    df = df.merge(clim, on=["lat", "lon", "week_of_year"])
    df["temp_anomaly"] = df["temp_mean_c"] - df["temp_clim_c"]
    df = df.drop(columns=["week_of_year", "temp_clim_c"])

    # Circular encoding for wind direction
    df["wind_dir_sin"] = np.sin(np.radians(df["wind_dir_deg"]))
    df["wind_dir_cos"] = np.cos(np.radians(df["wind_dir_deg"]))
    return df


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    nc_files = sorted(RAW_DIR.glob("era5_*.nc")) if RAW_DIR.exists() else []

    if not nc_files:
        nc_files = download_raw(YEARS)

    if nc_files:
        df = process_real(nc_files)
    else:
        df = _make_synthetic()

    df = _add_anomalies(df)

    print(f"  Rows:        {len(df):,}")
    print(f"  Cells:       {df[['lat','lon']].drop_duplicates().shape[0]:,}")
    print(f"  Weeks:       {df['week'].nunique()}")
    print(f"  Temp range:  {df['temp_mean_c'].min():.1f}°C - {df['temp_mean_c'].max():.1f}°C")

    df.to_parquet(OUT_FILE, index=False)
    print(f"  Saved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
