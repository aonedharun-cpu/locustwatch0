#!/usr/bin/env python3
"""
src/data/download_modis.py
--------------------------
Download MODIS MOD13A2 16-day NDVI composites, regrid to 0.1°, and
interpolate to weekly resolution.

Setup (one-time):
    1. Register at https://earthdata.nasa.gov (free)
    2. pip install earthpy pyhdf h5py requests

    earthpy stores credentials in ~/.netrc:
        machine urs.earthdata.nasa.gov login <USER> password <PASS>

    Generate with:
        import earthpy
        earthpy.set_root_dir()  # sets default download directory

Product:  MOD13A2 -- MODIS Terra Vegetation Indices (1 km, 16-day)
Variable: 1 km 16 days NDVI  (scale factor: 0.0001)

Output: data/processed/modis_ndvi_weekly_0p1deg.parquet
Schema:
    lat            float64
    lon            float64
    week           datetime64[ns]
    ndvi           float64   -- NDVI (0-1 scale)
    ndvi_clim      float64   -- climatological mean for that week-of-year
    ndvi_anomaly   float64   -- ndvi - ndvi_clim
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RAW_DIR = ROOT / "data/raw/modis"
OUT_FILE = ROOT / "data/processed/modis_ndvi_weekly_0p1deg.parquet"

LAT_MIN, LAT_MAX = -5.0, 35.0
LON_MIN, LON_MAX = -20.0, 75.0

YEARS = list(range(2015, 2024))


def _snap_to_grid(values: np.ndarray, resolution: float = 0.1) -> np.ndarray:
    return np.round(values / resolution) * resolution


def download_raw(years: list[int]) -> list[Path]:
    """
    Download MOD13A2 HDF files via earthpy / NASA Earthdata.
    Requires ~/.netrc with NASA Earthdata credentials.
    """
    try:
        import requests
        from requests.auth import HTTPBasicAuth
        import netrc
    except ImportError:
        print("  requests not installed: pip install requests")
        return []

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Read credentials from ~/.netrc
    try:
        creds = netrc.netrc()
        auth = creds.authenticators("urs.earthdata.nasa.gov")
        if auth is None:
            raise ValueError("No credentials for urs.earthdata.nasa.gov in ~/.netrc")
        username, _, password = auth
    except Exception as e:
        print(f"  NASA Earthdata credentials not found: {e}")
        print("  Add to ~/.netrc:")
        print("    machine urs.earthdata.nasa.gov login <USER> password <PASS>")
        return []

    # MODIS CMR search API
    CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
    downloaded = []

    for year in years:
        params = {
            "short_name": "MOD13A2",
            "version": "061",
            "temporal": f"{year}-01-01T00:00:00Z,{year}-12-31T23:59:59Z",
            "bounding_box": f"{LON_MIN},{LAT_MIN},{LON_MAX},{LAT_MAX}",
            "page_size": 100,
        }
        print(f"  Querying CMR for MOD13A2 {year} ...")
        try:
            resp = requests.get(CMR_URL, params=params, timeout=30)
            resp.raise_for_status()
            granules = resp.json()["feed"]["entry"]
        except Exception as e:
            print(f"  CMR query failed for {year}: {e}")
            continue

        for granule in granules:
            for link in granule.get("links", []):
                if link.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#":
                    url = link["href"]
                    fname = Path(url).name
                    dest = RAW_DIR / fname
                    if dest.exists():
                        downloaded.append(dest)
                        continue
                    try:
                        r = requests.get(
                            url,
                            auth=HTTPBasicAuth(username, password),
                            stream=True, timeout=60,
                        )
                        r.raise_for_status()
                        with open(dest, "wb") as f:
                            for chunk in r.iter_content(chunk_size=1 << 20):
                                f.write(chunk)
                        downloaded.append(dest)
                        print(f"  Downloaded {fname}")
                    except Exception as e:
                        print(f"  Failed {fname}: {e}")
                    break

    return downloaded


def process_real(hdf_files: list[Path]) -> pd.DataFrame:
    """
    Load MOD13A2 HDF files, extract NDVI, regrid to 0.1°.
    MOD13A2: 16-day composites, 1km sinusoidal projection.
    """
    try:
        import xarray as xr
        import rioxarray  # noqa: F401
        from pyproj import Transformer
    except ImportError:
        raise ImportError("pip install xarray rioxarray rasterio pyproj pyhdf")

    records = []
    for hdf_path in sorted(hdf_files):
        print(f"  Processing {hdf_path.name} ...")
        try:
            ds = xr.open_dataset(hdf_path, engine="rasterio")
            ndvi_raw = ds["1 km 16 days NDVI"]

            # Apply scale factor and mask fill values
            ndvi = ndvi_raw.where(ndvi_raw > -3000) * 0.0001
            ndvi = ndvi.clip(-0.2, 1.0)

            # Reproject to WGS84 and regrid to 0.1°
            ndvi_reproj = ndvi.rio.reproject("EPSG:4326")
            target_lats = np.arange(LAT_MIN, LAT_MAX + 0.1, 0.1)
            target_lons = np.arange(LON_MIN, LON_MAX + 0.1, 0.1)
            ndvi_reproj = ndvi_reproj.interp(
                y=target_lats, x=target_lons, method="linear"
            )

            # Parse acquisition date from filename: MOD13A2.AYYDDD
            doy_str = hdf_path.stem.split(".")[1]  # e.g. A2015001
            date = pd.to_datetime(doy_str[1:], format="%Y%j")
            week = date + pd.offsets.Week(weekday=0)

            df_slice = ndvi_reproj.to_dataframe(name="ndvi").reset_index()
            df_slice = df_slice.rename(columns={"y": "lat", "x": "lon"})
            df_slice["week"] = week
            df_slice["lat"] = _snap_to_grid(df_slice["lat"].values)
            df_slice["lon"] = _snap_to_grid(df_slice["lon"].values)
            records.append(df_slice[["lat", "lon", "week", "ndvi"]])
            ds.close()
        except Exception as e:
            print(f"  Error processing {hdf_path.name}: {e}")

    return pd.concat(records, ignore_index=True)


def _make_synthetic() -> pd.DataFrame:
    """
    Synthetic NDVI with seasonal greening cycle.
    In the Horn of Africa, NDVI peaks during the long rains (Apr-Jun)
    and short rains (Oct-Nov).
    """
    print("  [synthetic] No MODIS files found -- generating synthetic NDVI.")
    rng = np.random.default_rng(2)

    lats = np.arange(LAT_MIN, LAT_MAX + 0.5, 0.5).round(1)
    lons = np.arange(LON_MIN, LON_MAX + 0.5, 0.5).round(1)
    weeks = pd.date_range("2015-01-05", "2023-12-25", freq="W-MON")

    records = []
    for lat in lats:
        for lon in lons:
            n = len(weeks)
            doy = np.array([w.day_of_year for w in weeks])

            # Bimodal seasonal pattern
            ndvi_mean = 0.3 + 0.15 * (
                np.sin(2 * np.pi * (doy - 100) / 365) +
                0.5 * np.sin(4 * np.pi * (doy - 100) / 365)
            )
            # Drier areas have lower NDVI
            ndvi_mean = ndvi_mean * (0.5 + 0.5 * np.exp(-0.01 * abs(lon - 42)))
            ndvi = (ndvi_mean + rng.normal(0, 0.03, n)).clip(0, 1)

            records.append(pd.DataFrame({
                "lat":  lat,
                "lon":  lon,
                "week": weeks,
                "ndvi": ndvi,
            }))

    return pd.concat(records, ignore_index=True)


def _add_climatology(df: pd.DataFrame) -> pd.DataFrame:
    df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
    clim = (
        df.groupby(["lat", "lon", "week_of_year"])["ndvi"]
        .mean()
        .rename("ndvi_clim")
        .reset_index()
    )
    df = df.merge(clim, on=["lat", "lon", "week_of_year"])
    df["ndvi_anomaly"] = df["ndvi"] - df["ndvi_clim"]
    return df.drop(columns=["week_of_year"])


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    hdf_files = sorted(RAW_DIR.glob("MOD13A2.*.hdf")) if RAW_DIR.exists() else []

    if not hdf_files:
        hdf_files = download_raw(YEARS)

    if hdf_files:
        df = process_real(hdf_files)
    else:
        df = _make_synthetic()

    df = _add_climatology(df)

    print(f"  Rows:        {len(df):,}")
    print(f"  Cells:       {df[['lat','lon']].drop_duplicates().shape[0]:,}")
    print(f"  NDVI range:  {df['ndvi'].min():.3f} - {df['ndvi'].max():.3f}")

    df.to_parquet(OUT_FILE, index=False)
    print(f"  Saved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
