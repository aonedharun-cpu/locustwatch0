# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Phase 0 — download/generate all 5 data sources (synthetic fallbacks active if no credentials)
python src/data/download_fao.py
python src/data/download_chirps.py
python src/data/download_era5.py
python src/data/download_modis.py
python src/data/download_smap.py

# Phase 1 — build feature matrix (~7M rows, writes data/processed/features.parquet)
python src/features/build_features.py
python src/features/build_features.py --sample 200   # fast dev run (200 cells)

# Phase 0/1 notebooks (write figures to outputs/figures/)
python notebooks/00_data_audit.py
python notebooks/01_feature_eda.py

# Phase 2 — baseline classifiers (LR, RF, XGBoost)
python src/models/baselines.py                  # full run (temporal + spatial CV)
python src/models/baselines.py --sample 500     # fast dev
python src/models/baselines.py --no-spatial-cv  # skip spatial CV
python notebooks/02_baseline_evaluation.py      # figures from baseline_results.csv

# Phase 3 — neural model training
python src/models/train.py --fast               # sanity check (200 cells, 5 epochs)
python src/models/train.py --sample 2000        # mid-size run
python src/models/train.py                      # full run (all cells)

# Phase 4 -- interpretability, calibration, conformal prediction
python src/evaluation/shap_analysis.py --sample 300
python src/evaluation/calibration.py
python src/evaluation/conformal.py
python notebooks/04_east_africa_case_study.py --sample 800

# Phase 5 -- Streamlit dashboard
streamlit run src/dashboard/app.py
# (or via Python if streamlit not on PATH)
python -m streamlit run src/dashboard/app.py
```

No test suite or Makefile exists yet.

## Architecture

The pipeline is strictly sequential across phases. Each phase writes parquet files consumed by the next.

```
Phase 0: src/data/download_*.py        ->  data/processed/*.parquet          DONE
Phase 1: src/features/build_features.py ->  data/processed/features.parquet  DONE
Phase 2: src/models/baselines.py        ->  outputs/baseline_results.csv      DONE
Phase 3: src/models/train.py            ->  outputs/checkpoints/best_model.pt DONE
Phase 4: src/evaluation/               ->  SHAP, calibration, conformal       DONE
Phase 5: src/dashboard/app.py          ->  Streamlit + Folium dashboard        DONE
Phase 6: README, model card, paper scaffold                                    TODO
```

### Phase 3 model details

LocustNet (`src/models/architecture.py`): LSTM (2 layers, hidden=128) -> multi-head attention (4 heads) -> GNN spatial branch (nbr features, hidden=64) -> binary + phase + uncertainty heads. 317K parameters. MC Dropout for uncertainty estimation (`model.mc_predict(x, n_samples=30)`).

Dataset (`src/models/dataset.py`): sliding windows of seq_len=12 weeks per cell. Feature columns ordered self-features first, `_nbr` features last (architecture splits at `n_self` index). Normalised using training-split mean/std; NaN filled with 0 post-normalisation. WeightedRandomSampler oversamples positives.

Checkpoint (`outputs/checkpoints/best_model.pt`) stores model weights + feat_cols + normalisation stats — everything needed to reload for inference.

All configuration lives in `configs/config.yaml` — grid spec, temporal splits, feature params, model hyperparams, dashboard thresholds.

### Grid and time conventions

- **Grid:** 0.1° resolution, lat [-5, 35], lon [-20, 75] (study region: Horn of Africa, Arabian Peninsula, South Asia)
- **Time:** Monday-anchored weekly periods (`freq="W-MON"`). `pd.date_range(..., freq="W-MON")` produces Monday-*end* dates (e.g., 2020-06-15 is the Monday ending the period starting 2020-06-09). All week keys must be computed with `_to_week_key()` in `build_features.py`, NOT `dt.to_period("W-MON").apply(p.start_time)` which produces the Tuesday-start and causes label mismatches.
- **Temporal splits:** train ≤ 2017, val 2018–2020, test ≥ 2021

### Synthetic vs real data

Every `download_*.py` script detects whether the raw file/credential exists and falls back to synthetic generation automatically. Synthetic data uses 0.5° grid steps; real data uses 0.1°. **`build_features.py` auto-detects the actual grid resolution from the data** (via `np.diff` on sorted unique lat/lon values) — never assume a fixed 0.1° offset in spatial joins or neighbourhood computations.

### Feature matrix schema (features.parquet, 61 columns)

| Group | Columns | Notes |
|-------|---------|-------|
| ID | `cell_id, lat, lon, week` | |
| Climate (18) | `rainfall_weekly_mm, rainfall_roll_4w/8w/12w, rainfall_anomaly, temp_mean_c, temp_anomaly, ndvi, ndvi_anomaly, soil_moisture_surface, soil_moisture_rootzone, wind_speed_ms, wind_dir_sin, wind_dir_cos, humidity_pct` | Direct source features |
| Lag (24) | `{feature}_lag4w, {feature}_lag8w` for 12 climate features | |
| Spatial (12) | `{feature}_nbr` for 12 climate features | 8-cell Moore neighbourhood mean |
| Labels (2) | `outbreak_30d` (int8, 0/1), `phase_class` (int8, 0–3) | ~0.05% positive rate on synthetic data |

### Label construction

FAO occurrence records are snapped to the nearest actual grid cell using a **KD-tree** (`scipy.spatial.cKDTree`) — not a fixed rounding formula. `outbreak_30d = 1` if any FAO record falls in that cell within the next 30 days. `phase_class` = highest locust phase (0=none, 1=solitarious, 2=gregarious, 3=swarming).

### Windows / encoding

Running on Windows (Python 3.13, cp1252 terminal). All `print()` statements must use ASCII only — no Unicode box-drawing chars (`─`, `→`, `—`, etc.).
