#!/usr/bin/env python3
"""
src/models/baselines.py
-----------------------
Phase 2: Baseline classifiers for locust outbreak prediction.

Models
------
  - Logistic Regression  (L2, class_weight='balanced')
  - Random Forest        (100 trees, class_weight='balanced')
  - XGBoost              (gradient boosting, scale_pos_weight for imbalance)

Evaluation
----------
  Temporal split  : train<=2017 / val 2018-2020 / test>=2021
  Spatial CV      : leave-one-region-out (5 geographic regions)
  Metrics         : AUC-ROC, F1, PR-AUC, Brier score

Outputs
-------
  outputs/baseline_results.csv       -- metric summary table
  outputs/baseline_curves.parquet    -- ROC/PR curve arrays for plotting

Usage
-----
  python src/models/baselines.py                  # full run
  python src/models/baselines.py --sample 500     # 500 cells (fast dev)
  python src/models/baselines.py --no-spatial-cv  # skip spatial CV
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURES_FILE = ROOT / "data/processed/features.parquet"
RESULTS_FILE  = ROOT / "outputs/baseline_results.csv"
CURVES_FILE   = ROOT / "outputs/baseline_curves.parquet"

LABEL_COL = "outbreak_30d"
SKIP_COLS  = {"cell_id", "lat", "lon", "week", "outbreak_30d", "phase_class"}

# ── 5 geographic regions for spatial CV ──────────────────────────────────────
REGIONS = {
    "horn_of_africa":      lambda lat, lon: (lat < 15) & (lon >= 35) & (lon < 52),
    "arabian_peninsula":   lambda lat, lon: (lat >= 12) & (lat < 30) & (lon >= 42) & (lon < 62),
    "south_asia":          lambda lat, lon: lon >= 60,
    "west_africa":         lambda lat, lon: lon < 20,
    "east_africa_interior":lambda lat, lon: np.ones(len(lat), dtype=bool),  # catch-all
}


def assign_region(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Assign each cell to a region. Priority order matches REGIONS dict."""
    region = np.full(len(lat), "east_africa_interior", dtype=object)
    for name, mask_fn in REGIONS.items():
        if name == "east_africa_interior":
            continue
        m = mask_fn(lat, lon)
        region[m] = name
    return region


# ── Data loading ──────────────────────────────────────────────────────────────

def load_features(sample_cells: int | None = None) -> pd.DataFrame:
    if not FEATURES_FILE.exists():
        print("[ERROR] features.parquet not found.")
        print("  Run: python src/features/build_features.py")
        sys.exit(1)

    print(f"Loading {FEATURES_FILE.name} ...")
    df = pd.read_parquet(FEATURES_FILE)
    print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

    if sample_cells is not None:
        cells = df["cell_id"].unique()
        rng = np.random.default_rng(42)
        chosen = rng.choice(cells, size=min(sample_cells, len(cells)), replace=False)
        df = df[df["cell_id"].isin(chosen)].copy()
        print(f"  Sampled {sample_cells} cells -> {len(df):,} rows")

    print(f"  Positive rate: {df[LABEL_COL].mean()*100:.3f}%  "
          f"({df[LABEL_COL].sum():,} positives)")
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in SKIP_COLS]


# ── Train/val/test temporal split ─────────────────────────────────────────────

def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    week = pd.to_datetime(df["week"])
    train = df[week <= "2017-12-31"].copy()
    val   = df[(week >= "2018-01-01") & (week <= "2020-12-31")].copy()
    test  = df[week >= "2021-01-01"].copy()
    print(f"  Split sizes -> train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")
    return train, val, test


# ── Model definitions ─────────────────────────────────────────────────────────

def build_models(pos_count: int, neg_count: int) -> dict[str, Pipeline]:
    spw = max(1.0, neg_count / pos_count) if pos_count > 0 else 100.0

    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=500, solver="lbfgs", random_state=42,
        )),
    ])

    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestClassifier(
            n_estimators=100, max_depth=15, min_samples_leaf=100,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )),
    ])

    xgb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw,
            n_jobs=-1, random_state=42,
            eval_metric="aucpr",
            verbosity=0,
        )),
    ])

    return {"logistic_regression": lr, "random_forest": rf, "xgboost": xgb}


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = 0.5) -> dict[str, float]:
    if y_true.sum() == 0:
        return {"auc_roc": np.nan, "f1": np.nan, "pr_auc": np.nan, "brier": np.nan}
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "f1":      float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc":  float(average_precision_score(y_true, y_prob)),
        "brier":   float(brier_score_loss(y_true, y_prob)),
    }


def compute_curves(y_true: np.ndarray, y_prob: np.ndarray,
                   model_name: str) -> list[dict]:
    """Return ROC and PR curve rows for storage."""
    rows = []
    if y_true.sum() == 0:
        return rows

    fpr, tpr, thr_roc = roc_curve(y_true, y_prob)
    for x, y, t in zip(fpr, tpr, np.append(thr_roc, np.nan)):
        rows.append({"model": model_name, "curve": "roc", "x": x, "y": y, "threshold": t})

    prec, rec, thr_pr = precision_recall_curve(y_true, y_prob)
    for x, y, t in zip(rec, prec, np.append(thr_pr, np.nan)):
        rows.append({"model": model_name, "curve": "pr", "x": x, "y": y, "threshold": t})

    return rows


# ── Temporal evaluation ───────────────────────────────────────────────────────

def run_temporal_eval(df: pd.DataFrame, feat_cols: list[str]) -> tuple[list[dict], list[dict]]:
    print("\n[Temporal evaluation]")
    train, val, test = temporal_split(df)

    X_train = train[feat_cols].values
    y_train = train[LABEL_COL].values.astype(int)

    models = build_models(y_train.sum(), (y_train == 0).sum())

    results = []
    curves  = []

    for name, pipe in models.items():
        print(f"  Fitting {name} ...")
        t0 = time.time()
        pipe.fit(X_train, y_train)
        print(f"    Trained in {time.time()-t0:.1f}s")

        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            X = split_df[feat_cols].values
            y = split_df[LABEL_COL].values.astype(int)
            proba = pipe.predict_proba(X)[:, 1]
            metrics = compute_metrics(y, proba)
            row = {"eval_type": "temporal", "split": split_name, "model": name}
            row.update(metrics)
            results.append(row)
            print(f"    {split_name:<6}  AUC={metrics['auc_roc']:.4f}  "
                  f"F1={metrics['f1']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
                  f"Brier={metrics['brier']:.4f}")

            if split_name == "test":
                curves.extend(compute_curves(y, proba, name))

    return results, curves


# ── Spatial cross-validation ──────────────────────────────────────────────────

def run_spatial_cv(df: pd.DataFrame, feat_cols: list[str]) -> list[dict]:
    print("\n[Spatial CV -- leave-one-region-out]")

    region_arr = assign_region(df["lat"].values, df["lon"].values)
    df = df.copy()
    df["_region"] = region_arr
    all_regions = sorted(df["_region"].unique())
    print(f"  Regions: {all_regions}")

    # Use only the train+val period to avoid data leakage (test stays hidden)
    week = pd.to_datetime(df["week"])
    df_cv = df[week <= "2020-12-31"].copy()

    results = []

    for held_out in all_regions:
        train_df = df_cv[df_cv["_region"] != held_out]
        test_df  = df_cv[df_cv["_region"] == held_out]

        if test_df[LABEL_COL].sum() == 0:
            print(f"  Region {held_out}: no positives, skipping.")
            continue

        X_tr = train_df[feat_cols].values
        y_tr = train_df[LABEL_COL].values.astype(int)

        models = build_models(y_tr.sum(), (y_tr == 0).sum())
        X_te = test_df[feat_cols].values
        y_te = test_df[LABEL_COL].values.astype(int)

        print(f"  Held out: {held_out}  "
              f"(train {len(train_df):,}  test {len(test_df):,}  "
              f"pos {y_te.sum()})")

        for name, pipe in models.items():
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_te)[:, 1]
            metrics = compute_metrics(y_te, proba)
            row = {"eval_type": "spatial_cv", "split": held_out, "model": name}
            row.update(metrics)
            results.append(row)
            print(f"    {name:<22}  AUC={metrics['auc_roc']:.4f}  "
                  f"F1={metrics['f1']:.4f}  PR-AUC={metrics['pr_auc']:.4f}")

    return results


# ── Save outputs ──────────────────────────────────────────────────────────────

def save_results(results: list[dict], curves: list[dict]):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    df_res = pd.DataFrame(results)
    df_res.to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved metrics -> {RESULTS_FILE}")

    if curves:
        df_curves = pd.DataFrame(curves)
        df_curves.to_parquet(CURVES_FILE, index=False)
        print(f"Saved curves  -> {CURVES_FILE}")

    # Pretty-print summary table
    print("\n--- Temporal evaluation summary (test split) ---")
    test_rows = [r for r in results if r.get("split") == "test"]
    if test_rows:
        summary = pd.DataFrame(test_rows)[["model", "auc_roc", "f1", "pr_auc", "brier"]]
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n--- Spatial CV summary (mean across regions) ---")
    cv_rows = [r for r in results if r.get("eval_type") == "spatial_cv"]
    if cv_rows:
        cv_df = pd.DataFrame(cv_rows)
        cv_mean = cv_df.groupby("model")[["auc_roc", "f1", "pr_auc", "brier"]].mean()
        print(cv_mean.to_string(float_format=lambda x: f"{x:.4f}"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LocustWatch baseline classifiers")
    parser.add_argument("--sample", type=int, default=None,
                        help="Subsample N cells (fast dev; default: all)")
    parser.add_argument("--no-spatial-cv", action="store_true",
                        help="Skip spatial cross-validation")
    args = parser.parse_args()

    print("=" * 60)
    print("  LocustWatch AI -- Phase 2 Baseline Models")
    print("=" * 60)

    df = load_features(sample_cells=args.sample)
    feat_cols = get_feature_cols(df)
    print(f"  Feature columns: {len(feat_cols)}")

    all_results = []
    all_curves  = []

    # Temporal evaluation
    t_results, t_curves = run_temporal_eval(df, feat_cols)
    all_results.extend(t_results)
    all_curves.extend(t_curves)

    # Spatial CV
    if not args.no_spatial_cv:
        s_results = run_spatial_cv(df, feat_cols)
        all_results.extend(s_results)
    else:
        print("\n[Spatial CV skipped (--no-spatial-cv)]")

    save_results(all_results, all_curves)

    print("\n" + "=" * 60)
    print("  Phase 2 complete.")
    print("  Next: python notebooks/02_baseline_evaluation.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
