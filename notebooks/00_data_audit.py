#!/usr/bin/env python3
"""
notebooks/00_data_audit.py
--------------------------
Data audit script for Phase 0. Run after all download scripts complete.

For each processed parquet file, reports:
  - Row count and file size
  - Temporal range and number of unique weeks
  - Geographic extent (lat/lon min/max)
  - Number of unique grid cells
  - Missing value rates per column
  - Basic distribution statistics

Also generates three figures saved to outputs/figures/:
  - fig_00a_fao_occurrences_map.png     -- scatter map of outbreak records
  - fig_00b_chirps_rainfall_sample.png  -- time series for 3 sample cells
  - fig_00c_data_completeness.png       -- heatmap of data availability per source/week

Run:
    python notebooks/00_data_audit.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data/processed"
FIGURES   = ROOT / "outputs/figures"
FIGURES.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "fao":    PROCESSED / "fao_clean.parquet",
    "chirps": PROCESSED / "chirps_weekly_0p1deg.parquet",
    "era5":   PROCESSED / "era5_weekly_0p1deg.parquet",
    "modis":  PROCESSED / "modis_ndvi_weekly_0p1deg.parquet",
    "smap":   PROCESSED / "smap_weekly_0p1deg.parquet",
}


# ── Utility ───────────────────────────────────────────────────────────────────

def _hr():
    print("-" * 60)


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def _audit_parquet(name: str, path: Path) -> dict:
    """Load parquet and print audit summary. Returns audit dict."""
    if not path.exists():
        print(f"  {name:<8}  MISSING -- run src/data/download_{name}.py first")
        return {"source": name, "status": "missing"}

    df = pd.read_parquet(path)
    size_mb = _file_size_mb(path)

    # Detect lat/lon column names
    lat_col = next((c for c in df.columns if c in ("lat", "latitude")), None)
    lon_col = next((c for c in df.columns if c in ("lon", "longitude")), None)
    week_col = next((c for c in df.columns if c == "week"), None)
    date_col = next((c for c in df.columns if c == "date"), None)
    time_col = week_col or date_col

    # Missing value summary
    missing_pct = (df.isnull().mean() * 100).round(1)
    high_missing = missing_pct[missing_pct > 5].to_dict()

    audit = {
        "source":     name,
        "status":     "ok",
        "rows":       len(df),
        "cols":       len(df.columns),
        "size_mb":    round(size_mb, 1),
        "columns":    list(df.columns),
        "missing":    high_missing,
    }

    print(f"\n  [{name.upper()}]  {path.name}  ({size_mb:.1f} MB)")
    print(f"    Rows:    {len(df):,}   Cols: {len(df.columns)}")

    if time_col:
        t = pd.to_datetime(df[time_col])
        print(f"    Dates:   {t.min().date()} -> {t.max().date()}  "
              f"({t.nunique()} unique {'weeks' if time_col == 'week' else 'dates'})")
        audit["date_min"] = str(t.min().date())
        audit["date_max"] = str(t.max().date())
        audit["n_periods"] = int(t.nunique())

    if lat_col and lon_col:
        print(f"    Lat:     {df[lat_col].min():.2f} - {df[lat_col].max():.2f}")
        print(f"    Lon:     {df[lon_col].min():.2f} - {df[lon_col].max():.2f}")
        n_cells = df[[lat_col, lon_col]].drop_duplicates().shape[0]
        print(f"    Cells:   {n_cells:,}")
        audit["n_cells"] = n_cells

    if high_missing:
        print(f"    Missing >5%: {high_missing}")
    else:
        print(f"    Missing:  all columns < 5%")

    # Numeric stats for key columns
    num_cols = df.select_dtypes(include=np.number).columns[:6]
    if len(num_cols):
        print(f"    Stats for {list(num_cols)}:")
        print(df[num_cols].describe().round(3).to_string(
            max_cols=6, index=True
        ).replace("\n", "\n    "))

    audit["df"] = df  # keep for plotting
    return audit


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_fao_map(audit: dict):
    """Scatter map of FAO occurrence records coloured by phase."""
    if audit.get("status") != "ok":
        return

    df = audit["df"]
    fig, ax = plt.subplots(figsize=(10, 6))

    phase_colours = {0: "#aaaaaa", 1: "#4dac26", 2: "#f1a340", 3: "#d7191c"}
    phase_labels  = {0: "Unknown", 1: "Solitarious", 2: "Gregarious", 3: "Swarming"}

    lat_col = "latitude" if "latitude" in df.columns else "lat"
    lon_col = "longitude" if "longitude" in df.columns else "lon"

    for phase_id, colour in phase_colours.items():
        mask = df["phase_class"] == phase_id
        if mask.sum() == 0:
            continue
        ax.scatter(
            df.loc[mask, lon_col], df.loc[mask, lat_col],
            c=colour, s=4, alpha=0.5, label=phase_labels[phase_id]
        )

    ax.set_title("FAO Desert Locust Occurrences by Phase")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(markerscale=3, loc="upper right")
    ax.set_xlim(-20, 75)
    ax.set_ylim(-5, 35)
    ax.grid(True, alpha=0.3)

    out = FIGURES / "fig_00a_fao_occurrences_map.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved -> {out.name}")


def fig_chirps_sample(audit: dict):
    """Time series of rainfall for 3 sample cells."""
    if audit.get("status") != "ok":
        return

    df = audit["df"]
    cells = df[["lat", "lon"]].drop_duplicates().sample(3, random_state=42)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("CHIRPS Weekly Rainfall -- 3 Sample Cells")

    for i, (_, row) in enumerate(cells.iterrows()):
        cell_df = df[(df["lat"] == row["lat"]) & (df["lon"] == row["lon"])].sort_values("week")
        ax = axes[i]
        ax.bar(cell_df["week"], cell_df["rainfall_weekly_mm"],
               width=5, color="#2171b5", alpha=0.7)
        ax.plot(cell_df["week"], cell_df["rainfall_clim_mm"],
                color="#e31a1c", linewidth=1.2, label="climatology")
        ax.set_ylabel("mm/week")
        ax.set_title(f"Lat={row['lat']:.1f} Lon={row['lon']:.1f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Week")
    fig.tight_layout()

    out = FIGURES / "fig_00b_chirps_rainfall_sample.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")


def fig_data_completeness(audits: list[dict]):
    """Heatmap showing data availability per source per year."""
    source_order = ["fao", "chirps", "era5", "modis", "smap"]
    available = []
    labels = []

    for name in source_order:
        a = next((x for x in audits if x["source"] == name), None)
        if a and a.get("status") == "ok":
            available.append(1)
        else:
            available.append(0)
        labels.append(name.upper())

    fig, ax = plt.subplots(figsize=(8, 3))
    colours = ["#d73027" if v == 0 else "#1a9850" for v in available]

    for i, (label, colour) in enumerate(zip(labels, colours)):
        ax.barh(i, 1, color=colour, alpha=0.8, edgecolor="white")
        status = "READY" if available[i] else "MISSING"
        ax.text(0.5, i, f"{label}  {status}",
                va="center", ha="center", fontsize=11,
                fontweight="bold", color="white")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(labels) - 0.5)
    ax.axis("off")
    ax.set_title("Data Source Status", fontsize=13, pad=12)

    out = FIGURES / "fig_00c_data_completeness.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  LocustWatch AI -- Phase 0 Data Audit")
    print("=" * 60)

    audits = []
    for name, path in SOURCES.items():
        _hr()
        audit = _audit_parquet(name, path)
        audits.append(audit)

    _hr()
    print("\n  Generating figures ...")

    fao_audit    = next(a for a in audits if a["source"] == "fao")
    chirps_audit = next(a for a in audits if a["source"] == "chirps")

    fig_fao_map(fao_audit)
    fig_chirps_sample(chirps_audit)
    fig_data_completeness(audits)

    _hr()
    n_ready  = sum(1 for a in audits if a.get("status") == "ok")
    n_total  = len(audits)
    print(f"\n  Summary: {n_ready}/{n_total} data sources ready.")

    if n_ready < n_total:
        missing = [a["source"] for a in audits if a.get("status") != "ok"]
        print(f"  Still needed: {missing}")
        print("  Run the corresponding download_*.py scripts first.")
        print("  (Synthetic fallbacks are available -- see each script.)")
    else:
        print("  All sources present. Ready to proceed to Phase 1.")

    print("\n  Figures saved to: outputs/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
