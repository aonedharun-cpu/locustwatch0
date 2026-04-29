#!/usr/bin/env python3
"""
src/dashboard/app.py
----------------------
LocustWatch AI -- Streamlit + Folium interactive dashboard.

Features
--------
  - Interactive Folium risk map with colour-coded cells (watch / warning / emergency)
  - Week selector (time slider)
  - Sub-region weekly risk time series
  - MC Dropout confidence intervals
  - FAO historical locust record overlay
  - Top-feature SHAP importance bar chart
  - Data provenance panel
  - PDF report export

Usage
-----
  streamlit run src/dashboard/app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import folium
from streamlit_folium import st_folium
from io import BytesIO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.dataset import make_datasets, ordered_feat_cols
from src.models.architecture import build_model

# ── Paths ──────────────────────────────────────────────────────────────────────
CHECKPOINT     = ROOT / "outputs/checkpoints/best_model.pt"
TEMP_FILE      = ROOT / "outputs/calibration_temperature.json"
CONFORMAL_FILE = ROOT / "outputs/conformal_threshold.json"
FEATURES_FILE  = ROOT / "data/processed/features.parquet"
FAO_FILE       = ROOT / "data/processed/fao_clean.parquet"
SHAP_CSV       = ROOT / "outputs/shap_importance.csv"
FIGURES_DIR    = ROOT / "outputs/figures"
DEMO_PREDS     = ROOT / "outputs/demo_predictions.parquet"

# ── Risk tier config ───────────────────────────────────────────────────────────
RISK_TIERS = {"watch": 0.30, "warning": 0.60, "emergency": 0.85}
TIER_COLOURS = {
    "emergency": "#7b0000",
    "warning":   "#d7191c",
    "watch":     "#f1a340",
    "none":      "#d4edda",
}
TIER_LABELS = {
    "emergency": "Emergency (>=85%)",
    "warning":   "Warning (60-84%)",
    "watch":     "Watch (30-59%)",
    "none":      "Minimal (<30%)",
}

SUB_REGIONS = {
    "Ethiopia/Somalia": {"lat": (5, 12),   "lon": (38, 48)},
    "Yemen":            {"lat": (12, 20),  "lon": (42, 52)},
    "Kenya":            {"lat": (-2, 5),   "lon": (35, 42)},
    "Pakistan":         {"lat": (23, 35),  "lon": (60, 72)},
    "Arabian Peninsula":{"lat": (15, 25),  "lon": (44, 60)},
}
REGION_COLOURS = {
    "Ethiopia/Somalia":  "#d7191c",
    "Yemen":             "#f1a340",
    "Kenya":             "#1a9850",
    "Pakistan":          "#2c7bb6",
    "Arabian Peninsula": "#7b2d8b",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def tier(prob: float) -> str:
    if prob >= RISK_TIERS["emergency"]:
        return "emergency"
    if prob >= RISK_TIERS["warning"]:
        return "warning"
    if prob >= RISK_TIERS["watch"]:
        return "watch"
    return "none"


@st.cache_resource(show_spinner="Loading LocustNet model...")
def load_model():
    if not CHECKPOINT.exists():
        return None, None, None, None, None
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    meta = ckpt["meta"]
    model = build_model(meta["n_self"], meta["n_nbr"])
    model.load_state_dict(ckpt["model_state"])
    feat_stats = {
        "mean": np.array(meta["feat_mean"], dtype=np.float32),
        "std":  np.array(meta["feat_std"],  dtype=np.float32),
    }
    T = 1.0
    if TEMP_FILE.exists():
        with open(TEMP_FILE) as f:
            T = json.load(f).get("temperature", 1.0)
    q_hat = None
    if CONFORMAL_FILE.exists():
        with open(CONFORMAL_FILE) as f:
            q_hat = json.load(f).get("q_hat")
    return model, feat_stats, meta["feat_cols"], meta["seq_len"], T, q_hat


@st.cache_data(show_spinner="Loading features...")
def load_features():
    if not FEATURES_FILE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(FEATURES_FILE)
    df["week"] = pd.to_datetime(df["week"])
    return df


@st.cache_data(show_spinner="Loading demo predictions...")
def load_demo_preds():
    if not DEMO_PREDS.exists():
        return pd.DataFrame()
    df = pd.read_parquet(DEMO_PREDS)
    df["week"] = pd.to_datetime(df["week"])
    return df


@st.cache_data(show_spinner="Loading FAO records...")
def load_fao():
    if not FAO_FILE.exists():
        return pd.DataFrame()
    fao = pd.read_parquet(FAO_FILE)
    fao["week"] = pd.to_datetime(fao["week"])
    return fao


@torch.no_grad()
def run_inference_batch(model, feat_stats, feat_cols, df, seq_len, T=1.0):
    """Return DataFrame with risk_prob (mean) and risk_std (MC Dropout)."""
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

    valid_arr = np.array(valid_idx)
    seqs = np.stack([normed[e - seq_len + 1:e + 1] for e in valid_arr])
    seq_t = torch.tensor(seqs, dtype=torch.float32)

    # Mean prob (eval mode, no dropout)
    model.eval()
    bl, _, _ = model(seq_t)
    mean_prob = torch.sigmoid(bl / T).cpu().numpy()

    # MC Dropout uncertainty
    mean_mc, std_mc = model.mc_predict(seq_t, n_samples=20)
    mean_mc = mean_mc.cpu().numpy()
    std_mc  = std_mc.cpu().numpy()

    result = df.iloc[valid_arr][["cell_id", "lat", "lon", "week"]].copy()
    result["risk_prob"] = mean_prob
    result["risk_std"]  = std_mc
    return result.reset_index(drop=True)


def build_folium_map(week_preds: pd.DataFrame, fao_week: pd.DataFrame,
                     show_fao: bool, show_uncertainty: bool) -> folium.Map:
    """Build a Folium map for a single week's predictions."""
    m = folium.Map(
        location=[15.0, 40.0],
        zoom_start=4,
        tiles="CartoDB positron",
    )

    # ── Risk cells ────────────────────────────────────────────────────────────
    for _, row in week_preds.iterrows():
        t = tier(row["risk_prob"])
        if t == "none":
            continue
        colour = TIER_COLOURS[t]
        pct    = f"{row['risk_prob']*100:.1f}%"
        ci_lo  = max(0, row["risk_prob"] - 1.96 * row["risk_std"])
        ci_hi  = min(1, row["risk_prob"] + 1.96 * row["risk_std"])
        tooltip = (
            f"Risk: {pct} ({t.upper()})<br>"
            f"95% CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]<br>"
            f"Lat/Lon: {row['lat']:.2f}, {row['lon']:.2f}"
        )
        if show_uncertainty:
            radius = 5 + int(row["risk_std"] * 60)
        else:
            radius = 6

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.75,
            weight=0,
            tooltip=folium.Tooltip(tooltip, sticky=False),
        ).add_to(m)

    # ── FAO records ──────────────────────────────────────────────────────────
    if show_fao and not fao_week.empty:
        for _, row in fao_week.iterrows():
            folium.Marker(
                location=[row["cell_lat"], row["cell_lon"]],
                icon=folium.Icon(color="blue", icon="bug", prefix="fa"),
                tooltip=f"FAO record: {pd.Timestamp(row['week']).date()}",
            ).add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:white;padding:10px 14px;border-radius:8px;
                border:1px solid #ccc;font-size:13px;box-shadow:2px 2px 6px rgba(0,0,0,0.15);">
      <b>Outbreak Risk</b><br>
      <span style="color:#7b0000;">&#9679;</span> Emergency (&ge;85%)<br>
      <span style="color:#d7191c;">&#9679;</span> Warning (60-84%)<br>
      <span style="color:#f1a340;">&#9679;</span> Watch (30-59%)<br>
      <span style="color:#2c7bb6;">&#9873;</span> FAO record
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def fig_timeseries(preds: pd.DataFrame) -> plt.Figure:
    """Return weekly mean risk time series figure for sub-regions."""
    fig, ax = plt.subplots(figsize=(10, 4))

    for name, bounds in SUB_REGIONS.items():
        sub = preds[
            preds["lat"].between(*bounds["lat"]) &
            preds["lon"].between(*bounds["lon"])
        ]
        if sub.empty:
            continue
        ts = sub.groupby("week")["risk_prob"].mean()
        ts_std = sub.groupby("week")["risk_std"].mean()
        ts.index = pd.to_datetime(ts.index)
        color = REGION_COLOURS[name]
        ax.plot(ts.index, ts.values, linewidth=1.5,
                color=color, label=name, alpha=0.9)
        ax.fill_between(ts.index,
                        (ts - 1.96 * ts_std).clip(0, 1),
                        (ts + 1.96 * ts_std).clip(0, 1),
                        color=color, alpha=0.12)

    ax.axhspan(RISK_TIERS["watch"],     RISK_TIERS["warning"],   color="#f1a340", alpha=0.08)
    ax.axhspan(RISK_TIERS["warning"],   RISK_TIERS["emergency"],  color="#d7191c", alpha=0.08)
    ax.axhspan(RISK_TIERS["emergency"], 1.0,                       color="#7b0000", alpha=0.08)

    ax.set_xlabel("Week", fontsize=10)
    ax.set_ylabel("Mean outbreak probability", fontsize=10)
    ax.set_title("Weekly Risk by Sub-Region  (shaded = 95% CI)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def fig_shap_bar() -> plt.Figure | None:
    """Return top-20 SHAP importance bar chart or None if CSV missing."""
    if not SHAP_CSV.exists():
        return None
    df = pd.read_csv(SHAP_CSV).head(20)
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(df))
    ax.barh(y, df["mean_abs_shap"].values[::-1],
            color="#2c7bb6", alpha=0.85, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([f[:28] for f in df["feature"].values[::-1]], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|", fontsize=9)
    ax.set_title("Top-20 Feature Importance (XGBoost SHAP)", fontsize=10)
    ax.grid(True, alpha=0.25, axis="x")
    fig.tight_layout()
    return fig


def generate_pdf_report(selected_week, week_preds, fao_week):
    """
    Generate a simple PDF bytes object summarising the selected week.
    Uses matplotlib's PDF backend (no extra deps).
    """
    from matplotlib.backends.backend_pdf import PdfPages

    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1: summary stats
        fig, ax = plt.subplots(figsize=(8.5, 4))
        ax.axis("off")

        t_counts = week_preds["risk_prob"].apply(tier).value_counts()
        n_watch     = t_counts.get("watch", 0)
        n_warning   = t_counts.get("warning", 0)
        n_emergency = t_counts.get("emergency", 0)
        mean_risk   = week_preds["risk_prob"].mean()

        text = (
            f"LocustWatch AI -- Weekly Risk Report\n"
            f"Week: {pd.Timestamp(selected_week).date()}\n\n"
            f"Cells analysed:  {len(week_preds):,}\n"
            f"Mean risk:        {mean_risk*100:.2f}%\n"
            f"Watch cells:      {n_watch:,}\n"
            f"Warning cells:    {n_warning:,}\n"
            f"Emergency cells:  {n_emergency:,}\n"
            f"FAO records:      {len(fao_week):,}\n"
        )
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=11, verticalalignment="top", family="monospace")
        fig.suptitle("LocustWatch AI Risk Report", fontsize=14, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: risk distribution histogram
        fig, ax = plt.subplots(figsize=(8.5, 4))
        ax.hist(week_preds["risk_prob"], bins=40,
                color="#2c7bb6", alpha=0.8, edgecolor="white")
        ax.axvline(RISK_TIERS["watch"],     color="#f1a340", linestyle="--", label="Watch")
        ax.axvline(RISK_TIERS["warning"],   color="#d7191c", linestyle="--", label="Warning")
        ax.axvline(RISK_TIERS["emergency"], color="#7b0000", linestyle="--", label="Emergency")
        ax.set_xlabel("Predicted outbreak probability")
        ax.set_ylabel("Cell count")
        ax.set_title(f"Risk Distribution -- {pd.Timestamp(selected_week).date()}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: SHAP bar chart if available
        shap_fig = fig_shap_bar()
        if shap_fig is not None:
            pdf.savefig(shap_fig, bbox_inches="tight")
            plt.close(shap_fig)

    buf.seek(0)
    return buf.read()


# ── Shared UI helpers ─────────────────────────────────────────────────────────

def _provenance_table(model, T, q_hat, fao_all, seq_len=12):
    prov_rows = [
        ("Model",           "LocustNet (LSTM + Attention + GNN)"),
        ("Parameters",      "~317K"),
        ("Checkpoint",      CHECKPOINT.name if CHECKPOINT.exists() else "Missing"),
        ("Features file",   FEATURES_FILE.name if FEATURES_FILE.exists() else "Not loaded (demo mode)"),
        ("FAO records",     f"{len(fao_all):,}" if not fao_all.empty else "Missing"),
        ("Calibration T",   f"{T:.4f}"),
        ("Conformal q_hat", f"{q_hat:.4f}" if q_hat else "Not computed"),
        ("Grid resolution", "0.1 deg (~11 km)"),
        ("Temporal split",  "Train<=2017 | Val 2018-20 | Test>=2021"),
        ("Sequence length", f"{seq_len} weeks"),
    ]
    for label, value in prov_rows:
        st.markdown(f"**{label}:** {value}")


def _figures_gallery():
    all_figs = sorted(FIGURES_DIR.glob("fig_*.png")) if FIGURES_DIR.exists() else []
    if not all_figs:
        st.info("No figures found.")
        return
    tabs = st.tabs(["EDA (Phase 0-1)", "Baselines (Phase 2)", "Interpretability (Phase 4)"])
    groups = [
        [f for f in all_figs if f.stem.startswith(("fig_00", "fig_01"))],
        [f for f in all_figs if f.stem.startswith("fig_02")],
        [f for f in all_figs if f.stem.startswith("fig_04")],
    ]
    for tab, fig_files in zip(tabs, groups):
        with tab:
            if not fig_files:
                st.info("No figures in this group.")
                continue
            cols = st.columns(3)
            for i, fp in enumerate(fig_files):
                cols[i % 3].image(str(fp), caption=fp.stem, use_container_width=True)


def static_mode(model, T, q_hat, fao_all):
    """Dashboard shown on Streamlit Cloud where the 2GB features file is absent."""
    preds = load_demo_preds()

    with st.sidebar:
        st.header("Controls")
        show_fao         = st.checkbox("Show FAO records on map", value=True)
        show_uncertainty = st.checkbox("Size circles by uncertainty", value=False)
        min_tier = st.selectbox(
            "Minimum tier on map",
            options=["watch", "warning", "emergency"],
            index=0,
        )
        st.divider()
        st.info(
            "Demo mode: pre-computed predictions (400 cells). "
            "Run locally for full live inference."
        )
        st.caption("LocustWatch AI v0.5 | Phase 5 Dashboard")

    # KPI strip
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model",         "LocustNet")
    col2.metric("Parameters",    "~317K")
    col3.metric("Calibration T", f"{T:.3f}")
    col4.metric("Conformal q",   f"{q_hat:.3f}" if q_hat else "N/A")

    # ── Interactive map ───────────────────────────────────────────────────────
    if not preds.empty:
        st.subheader("Interactive Risk Map")

        available_weeks = sorted(preds["week"].unique())
        week_labels = [str(pd.Timestamp(w).date()) for w in available_weeks]

        col_map, col_ctrl = st.columns([3, 1])
        with col_ctrl:
            week_idx = st.select_slider(
                "Select week",
                options=list(range(len(available_weeks))),
                value=len(available_weeks) - 1,
                format_func=lambda i: week_labels[i],
            )

        selected_week = available_weeks[week_idx]
        week_preds = preds[preds["week"] == selected_week].copy()

        tier_order = {"none": 0, "watch": 1, "warning": 2, "emergency": 3}
        min_val = tier_order[min_tier]
        week_preds = week_preds[
            week_preds["risk_prob"].apply(lambda p: tier_order[tier(p)] >= min_val)
        ].copy()

        fao_week = pd.DataFrame()
        if not fao_all.empty and show_fao:
            fao_week = fao_all[
                (fao_all["week"] >= pd.Timestamp(selected_week) - pd.Timedelta(weeks=2)) &
                (fao_all["week"] <= pd.Timestamp(selected_week) + pd.Timedelta(weeks=2))
            ].copy()

        t_counts = week_preds["risk_prob"].apply(tier).value_counts()
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Cells",     f"{len(week_preds):,}")
        k2.metric("Watch",     f"{t_counts.get('watch', 0):,}")
        k3.metric("Warning",   f"{t_counts.get('warning', 0):,}")
        k4.metric("Emergency", f"{t_counts.get('emergency', 0):,}")
        k5.metric("Mean risk", f"{week_preds['risk_prob'].mean()*100:.1f}%" if len(week_preds) else "N/A")

        with col_map:
            folium_map = build_folium_map(week_preds, fao_week, show_fao, show_uncertainty)
            st_folium(folium_map, width=None, height=480, returned_objects=[])

        # Time series
        st.subheader("Regional Risk Time Series")
        ts_fig = fig_timeseries(preds)
        st.pyplot(ts_fig, use_container_width=True)
        plt.close(ts_fig)

    else:
        st.warning("Demo predictions not found. Run: python scripts/export_demo_predictions.py")
        risk_map = FIGURES_DIR / "fig_04f_risk_maps_2019_2020.png"
        if risk_map.exists():
            st.image(str(risk_map), use_container_width=True)

    # SHAP + provenance
    col_shap, col_prov = st.columns([3, 2])
    with col_shap:
        st.subheader("Feature Importance (SHAP)")
        shap_fig = fig_shap_bar()
        if shap_fig:
            st.pyplot(shap_fig, use_container_width=True)
            plt.close(shap_fig)
        else:
            shap_img = FIGURES_DIR / "fig_04a_shap_importance.png"
            if shap_img.exists():
                st.image(str(shap_img), use_container_width=True)

    with col_prov:
        st.subheader("Data Provenance")
        _provenance_table(model, T, q_hat, fao_all)

    with st.expander("All Analysis Figures", expanded=False):
        _figures_gallery()


# ── Main Streamlit app ─────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="LocustWatch AI",
        page_icon=":bug:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style='text-align:center;padding:8px 0 4px 0;'>
          <h1 style='margin-bottom:0;'>LocustWatch AI</h1>
          <p style='color:#666;margin-top:4px;font-size:15px;'>
            Spatiotemporal desert locust outbreak prediction &mdash;
            Horn of Africa, Arabian Peninsula, South Asia
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Load resources ────────────────────────────────────────────────────────
    result = load_model()
    if len(result) == 5:
        model, feat_stats, feat_cols, seq_len, T = result
        q_hat = None
    else:
        model, feat_stats, feat_cols, seq_len, T, q_hat = result

    df_all = load_features()
    fao_all = load_fao()

    # ── Static mode: no features file (e.g. Streamlit Cloud) ─────────────────
    if df_all.empty:
        static_mode(model, T, q_hat, fao_all)
        return

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Controls")

        min_date = df_all["week"].min().date()
        max_date = df_all["week"].max().date()
        date_range = st.date_input(
            "Date window",
            value=(max_date - pd.Timedelta(weeks=52), max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if len(date_range) == 2:
            start_d, end_d = date_range
        else:
            start_d = end_d = date_range[0]

        n_cells = st.slider(
            "Max cells (for speed)",
            min_value=200, max_value=5000, value=800, step=200,
        )
        show_fao         = st.checkbox("Show FAO records on map", value=True)
        show_uncertainty = st.checkbox("Size circles by uncertainty", value=False)
        min_tier = st.selectbox(
            "Minimum tier on map",
            options=["watch", "warning", "emergency"],
            index=0,
        )
        st.divider()
        st.caption("LocustWatch AI v0.5 | Phase 5 Dashboard")

    # ── Filter & run inference ────────────────────────────────────────────────
    with st.spinner("Running inference..."):
        df_window = df_all[
            (df_all["week"] >= pd.Timestamp(start_d)) &
            (df_all["week"] <= pd.Timestamp(end_d))
        ].copy()

        cells = df_window["cell_id"].unique()
        if len(cells) > n_cells:
            rng = np.random.default_rng(42)
            cells = rng.choice(cells, n_cells, replace=False)
            df_window = df_window[df_window["cell_id"].isin(cells)].copy()

        cutoff = pd.Timestamp(start_d) - pd.Timedelta(weeks=seq_len)
        df_ext = df_all[
            (df_all["week"] >= cutoff) &
            (df_all["week"] <= pd.Timestamp(end_d)) &
            df_all["cell_id"].isin(cells)
        ].copy()

        preds = run_inference_batch(model, feat_stats, feat_cols, df_ext, seq_len, T)
        if preds.empty:
            st.warning("No predictions in this window. Try a wider date range.")
            st.stop()

        preds = preds[preds["week"] >= pd.Timestamp(start_d)].copy()

    # ── Week selector ─────────────────────────────────────────────────────────
    available_weeks = sorted(preds["week"].unique())
    week_labels = [str(pd.Timestamp(w).date()) for w in available_weeks]

    col_map, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        week_idx = st.select_slider(
            "Select week",
            options=list(range(len(available_weeks))),
            value=len(available_weeks) - 1,
            format_func=lambda i: week_labels[i],
        )

    selected_week = available_weeks[week_idx]
    week_preds = preds[preds["week"] == selected_week].copy()

    tier_order = {"none": 0, "watch": 1, "warning": 2, "emergency": 3}
    min_val = tier_order[min_tier]
    week_preds = week_preds[
        week_preds["risk_prob"].apply(lambda p: tier_order[tier(p)] >= min_val)
    ].copy()

    fao_week = pd.DataFrame()
    if not fao_all.empty and show_fao:
        fao_week = fao_all[
            (fao_all["week"] >= pd.Timestamp(selected_week) - pd.Timedelta(weeks=2)) &
            (fao_all["week"] <= pd.Timestamp(selected_week) + pd.Timedelta(weeks=2))
        ].copy()

    # ── Summary KPI strip ─────────────────────────────────────────────────────
    t_counts = week_preds["risk_prob"].apply(tier).value_counts()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Cells analysed", f"{len(week_preds):,}")
    k2.metric("Watch",          f"{t_counts.get('watch', 0):,}")
    k3.metric("Warning",        f"{t_counts.get('warning', 0):,}")
    k4.metric("Emergency",      f"{t_counts.get('emergency', 0):,}")
    k5.metric("Mean risk",      f"{week_preds['risk_prob'].mean()*100:.1f}%")
    st.caption(
        f"Week: **{pd.Timestamp(selected_week).date()}** | "
        f"Calibration T={T:.3f} | "
        + (f"Conformal q_hat={q_hat:.3f}" if q_hat else "No conformal threshold loaded")
    )

    with col_map:
        folium_map = build_folium_map(week_preds, fao_week, show_fao, show_uncertainty)
        st_folium(folium_map, width=None, height=480, returned_objects=[])

    st.subheader("Regional Risk Time Series")
    ts_fig = fig_timeseries(preds)
    st.pyplot(ts_fig, use_container_width=True)
    plt.close(ts_fig)

    col_shap, col_prov = st.columns([3, 2])
    with col_shap:
        st.subheader("Feature Importance (SHAP)")
        shap_fig = fig_shap_bar()
        if shap_fig:
            st.pyplot(shap_fig, use_container_width=True)
            plt.close(shap_fig)
        else:
            st.info("Run: python src/evaluation/shap_analysis.py")

    with col_prov:
        st.subheader("Data Provenance")
        _provenance_table(model, T, q_hat, fao_all, seq_len)

    with st.expander("Analysis Figures", expanded=False):
        _figures_gallery()

    st.divider()
    st.subheader("Export")
    if st.button("Generate PDF report for selected week"):
        with st.spinner("Building PDF..."):
            pdf_bytes = generate_pdf_report(selected_week, week_preds, fao_week)
        fname = f"locustwatch_report_{pd.Timestamp(selected_week).date()}.pdf"
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
        )


if __name__ == "__main__":
    main()
