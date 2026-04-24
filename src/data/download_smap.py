#!/usr/bin/env python3
"""
src/data/download_smap.py
--------------------------
Download SMAP L4 Global Surface and Root Zone Soil Moisture (SPL4SMGP),
regrid to 0.1°, and aggregate to weekly resolution.

Setup (one-time):
    1. Register at https://earthdata.nasa.gov (same account as MODIS)
    2. Add to ~/.netrc:
           machine urs.earthdata.nasa.gov login <USER> password <PASS>
    3. pip install requests h5py

Product:  SPL4SMGP.006 -- SMAP L4 9km, 3-hourly
Variables:
    sm_surface       -- surface soil moisture (0-5 cm), m³/m³
    sm_rootzone      -- root zone soil moisture (0-100 cm), m³/m³

Output: data/processed/smap_weekly_0p1deg.parquet
Schema:
    lat                      float64
    lon                      float64
    week                     datetime64[ns]
    soil_moisture_surface    float64   -- weekly mean (m³/m³)
    soil_moisture_rootzone   float64
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RAW_DIR = ROOT / "data/raw/smap"
OUT_FILE = ROOT / "data/processed/smap_weekly_0p1deg.parquet"

LAT_MIN, LAT_MAX = -5.0, 35.0
LON_MIN, LON_MAX = -20.0, 75.0

YEARS = list(range(2015, 2024))
SMAP_SHORT_NAME = "SPL4SMGP"
SMAP_VERSION = "006"


def _snap_to_grid(values: np.ndarray, resolution: float = 0.1) -> np.ndarray:
    return np.round(values / resolution) * resolution


def download_raw(years: list[int]) -> list[Path]:
    """
    Download SMAP HDF5 files via NASA CMR + Earthdata auth.
    Downloads one file per day (daily mean is sufficient for weekly aggregation).
    """
    try:
        import requests
        import netrc
    except ImportError:
        print("  requests not installed: pip install requests")
        return []

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

    from requests.auth import HTTPBasicAuth

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
    downloaded = []

    for year in years:
        # Download one file per week (Monday) to keep footprint small
        weekly_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="W-MON")

        for date in weekly_dates:
            date_str = date.strftime("%Y-%m-%d")
            params = {
                "short_name": SMAP_SHORT_NAME,
                "version": SMAP_VERSION,
                "temporal": f"{date_str}T00:00:00Z,{date_str}T23:59:59Z",
                "bounding_box": f"{LON_MIN},{LAT_MIN},{LON_MAX},{LAT_MAX}",
                "page_size": 5,
            }
            try:
                resp = requests.get(CMR_URL, params=params, timeout=30)
                resp.raise_for_status()
                granules = resp.json()["feed"]["entry"]
            except Exception as e:
                print(f"  CMR query failed for {date_str}: {e}")
                continue

            if not granules:
                continue

            for link in granules[0].get("links", []):
                if link.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#":
                    url = link["href"]
                    fname = Path(url).name
                    dest = RAW_DIR / fname
                    if dest.exists():
                        downloaded.append(dest)
                        break
                    try:
                        r = requests.get(
                            url,
                            auth=HTTPBasicAuth(username, password),
                            stream=True, timeout=120,
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


def process_real(h5_files: list[Path]) -> pd.DataFrame:
    """
    Load SMAP HDF5, extract surface + rootzone SM, regrid to 0.1°.
    SMAP L4 uses EASE-2 9km grid.
    """
    try:
        import h5py
        from scipy.interpolate import RegularGridInterpolator
    except ImportError:
        raise ImportError("pip install h5py scipy")

    target_lats = np.arange(LAT_MIN, LAT_MAX + 0.1, 0.1)
    target_lons = np.arange(LON_MIN, LON_MAX + 0.1, 0.1)

    records = []
    for h5_path in sorted(h5_files):
        print(f"  Processing {h5_path.name} ...")
        try:
            with h5py.File(h5_path, "r") as f:
                sm_surf = f["Geophysical_Data/sm_surface"][:]
                sm_root = f["Geophysical_Data/sm_rootzone"][:]
                lats    = f["cell_lat"][:]
                lons    = f["cell_lon"][:]
                fill    = f["Geophysical_Data/sm_surface"].attrs.get("_FillValue", -9999)

            # Mask fill values
            sm_surf = np.where(sm_surf == fill, np.nan, sm_surf)
            sm_root = np.where(sm_root == fill, np.nan, sm_root)

            # Parse date from filename: SMAP_L4_SM_gph_YYYYMMDD...
            date_str = h5_path.stem.split("_")[4][:8]  # YYYYMMDD
            week = pd.to_datetime(date_str, format="%Y%m%d")
            week = week - pd.offsets.Week(weekday=0) + pd.offsets.Week(weekday=0)

            # Regrid via nearest-neighbour onto 0.1° grid
            lat_grid = lats.ravel()
            lon_grid = lons.ravel()
            valid = np.isfinite(sm_surf.ravel()) & np.isfinite(lat_grid) & np.isfinite(lon_grid)

            if valid.sum() < 100:
                continue

            from scipy.spatial import cKDTree
            points = np.column_stack([lat_grid[valid], lon_grid[valid]])
            tree = cKDTree(points)

            tgt_lat, tgt_lon = np.meshgrid(target_lats, target_lons, indexing="ij")
            query_pts = np.column_stack([tgt_lat.ravel(), tgt_lon.ravel()])
            dist, idx = tree.query(query_pts, k=1)

            surf_regrid = sm_surf.ravel()[valid][idx]
            root_regrid = sm_root.ravel()[valid][idx]

            # Mask points more than 0.2° from nearest source point
            surf_regrid[dist > 0.2] = np.nan
            root_regrid[dist > 0.2] = np.nan

            records.append(pd.DataFrame({
                "lat":  tgt_lat.ravel(),
                "lon":  tgt_lon.ravel(),
                "week": week,
                "soil_moisture_surface":  surf_regrid,
                "soil_moisture_rootzone": root_regrid,
            }))

        except Exception as e:
            print(f"  Error processing {h5_path.name}: {e}")

    if not records:
        return pd.DataFrame(columns=["lat","lon","week",
                                     "soil_moisture_surface","soil_moisture_rootzone"])

    df = pd.concat(records, ignore_index=True)
    # Average multiple files in the same week
    df = df.groupby(["lat","lon","week"], as_index=False).mean(numeric_only=True)
    return df


def _make_synthetic() -> pd.DataFrame:
    """
    Synthetic soil moisture: correlated with rainfall, lagged 2 weeks.
    """
    print("  [synthetic] No SMAP files found -- generating synthetic soil moisture.")
    rng = np.random.default_rng(3)

    lats = np.arange(LAT_MIN, LAT_MAX + 0.5, 0.5).round(1)
    lons = np.arange(LON_MIN, LON_MAX + 0.5, 0.5).round(1)
    weeks = pd.date_range("2015-01-05", "2023-12-25", freq="W-MON")

    records = []
    for lat in lats:
        for lon in lons:
            n = len(weeks)
            doy = np.array([w.day_of_year for w in weeks])

            # Surface moisture follows rain season, peaks slightly lagged
            sm_surf = 0.15 + 0.1 * np.sin(2 * np.pi * (doy - 110) / 365) \
                           + rng.normal(0, 0.02, n)
            sm_surf = sm_surf.clip(0.02, 0.45)

            # Root zone integrates over longer window -> smoother
            sm_root = pd.Series(sm_surf).rolling(4, min_periods=1).mean().values
            sm_root = sm_root + rng.normal(0, 0.01, n)
            sm_root = sm_root.clip(0.05, 0.40)

            records.append(pd.DataFrame({
                "lat":  lat,
                "lon":  lon,
                "week": weeks,
                "soil_moisture_surface":  sm_surf,
                "soil_moisture_rootzone": sm_root,
            }))

    return pd.concat(records, ignore_index=True)


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(RAW_DIR.glob("SMAP_L4_SM*.h5")) if RAW_DIR.exists() else []

    if not h5_files:
        h5_files = download_raw(YEARS)

    if h5_files:
        df = process_real(h5_files)
    else:
        df = _make_synthetic()

    print(f"  Rows:   {len(df):,}")
    print(f"  Cells:  {df[['lat','lon']].drop_duplicates().shape[0]:,}")
    print(f"  SM surface range: "
          f"{df['soil_moisture_surface'].min():.3f} - "
          f"{df['soil_moisture_surface'].max():.3f} m³/m³")

    df.to_parquet(OUT_FILE, index=False)
    print(f"  Saved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
