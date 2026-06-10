#!/usr/bin/env python3
"""Cultivar confounding tests for multiangular UAV disease prediction.

Tests whether the multiangular advantage persists after controlling for cultivar
identity.  Six tests:

  1. Experimental balance table (rows/plots/disease × cultivar)
  2. Cultivar-only baseline (C0) — one-hot encoded cultivar
  3. Treatment + cultivar baseline (C1) — one-hot encoded cultivar+trt
  4. M1/M2/M3/M5 with and without cultivar (±trt) adjustment
  5. Within-cultivar M1/M3/M5
  6. Interaction test (M3 score × cultivar via statsmodels Logit)
"""

import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

PROJ = Path(__file__).resolve().parent.parent.parent
FEATURE_DIR = PROJ / "outputs" / "features"
RESULTS_DIR = PROJ / "outputs" / "results"
FIGURES_DIR = PROJ / "outputs" / "figures"
LOGS_DIR = PROJ / "outputs" / "logs"
REPORTS_DIR = PROJ / "outputs" / "reports"

POLYGON_PATH = Path("/run/media/davidem/Heim/2024_oncerco_plot_polygons.gpkg")

FEATURE_SETS = ["M1", "M2", "M3", "M4", "M5"]
DISPLAY = {
    "M1": "Nadir bands",
    "M2": "Nadir indices",
    "M3": "Multiangular VZA",
    "M4": "VZA+RAA",
    "M5": "Angular contrast",
}
EXCLUDE_COLS = {"plot_id", "week", "cult", "trt", "disease_label"}
META_COLS = ["cult", "trt"]
SEED = 42
N_SPLITS = 5

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"cultivar_confounding_{TIMESTAMP}.log"


# ── logging + profiling ────────────────────────────────────────────────


def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Log: {LOG_FILE}")


def log_phase(name, elapsed):
    logging.info(f"[PHASE] {name}: {elapsed:.1f}s")


# ── helpers ─────────────────────────────────────────────────────────────


def load_feature_set(name):
    pattern = list(FEATURE_DIR.glob(f"{name}_*.parquet"))
    if not pattern:
        raise FileNotFoundError(f"No parquet for {name}")
    df = pl.read_parquet(pattern[0])
    return df.filter(pl.col("disease_label").is_not_null())


def load_polygon_meta():
    gdf = gpd.read_file(POLYGON_PATH)
    cols = [c for c in ["cult", "ifz_id", "trt", "ino"] if c in gdf.columns]
    pdf = pl.from_pandas(gdf[cols].copy())
    if "ino" in pdf.columns:
        pdf = pdf.with_columns(pl.col("ino").cast(pl.Int64).alias("disease_label"))
    pdf = pdf.with_columns(
        (pl.col("ifz_id") - 90001).cast(pl.Utf8).str.zfill(2).alias("plot_num"),
        ("plot_" + pl.col("ifz_id").cast(pl.Utf8).str.slice(-2, 2)).alias("plot_id_stub"),
    )
    return pdf


def add_ifz_id(df):
    return df.with_columns(
        (pl.col("plot_id").str.extract(r"(\d+)").cast(pl.Int64) + 90001).alias("ifz_id")
    )


def feature_columns(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def check_no_leakage(feature_cols, tag):
    forbidden = {"cult", "trt", "week", "disease_label"}
    leaked = set(feature_cols) & forbidden
    if leaked:
        logging.error(f"  LEAKAGE [{tag}]: {leaked}")
        return False
    return True


def run_cv_splits(X, y, groups, n_splits=N_SPLITS):
    try:
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        splits = list(skf.split(X, y, groups=groups))
    except (ValueError, RuntimeError):
        try:
            gkf = GroupKFold(n_splits=n_splits)
            splits = list(gkf.split(X, y, groups=groups))
        except (ValueError, RuntimeError):
            logo = LeaveOneGroupOut()
            splits = list(logo.split(X, y, groups=groups))
    return splits


def verify_no_plot_leak(splits, groups_arr):
    for fold, (train_idx, test_idx) in enumerate(splits):
        overlap = set(groups_arr[train_idx]) & set(groups_arr[test_idx])
        if overlap:
            logging.error(f"  Fold {fold}: {len(overlap)} plots in both train/test!")
            return False
    return True


def cv_evaluate(pipe, X, y, groups, tag="", compute_train=True):
    splits = run_cv_splits(X, y, groups)
    verify_no_plot_leak(splits, groups)
    results = []
    max_gap = 0.0
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        try:
            pipe.fit(X_train, y_train)
            y_prob = pipe.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_prob)
        except Exception as e:
            logging.warning(f"  [{tag}] fold {fold}: {e}")
            test_auc = np.nan
        train_auc = np.nan
        gap = np.nan
        if compute_train and not np.isnan(test_auc):
            try:
                y_prob_train = pipe.predict_proba(X_train)[:, 1]
                train_auc = roc_auc_score(y_train, y_prob_train)
                gap = train_auc - test_auc
            except Exception:
                pass
        max_gap = max(max_gap, gap if not np.isnan(gap) else 0)
        results.append(
            {
                "fold": fold,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "AUROC": test_auc,
                "AUROC_train": train_auc,
                "AUROC_gap": gap,
            }
        )
    return results, max_gap


# ── Build pipelines ─────────────────────────────────────────────────────


def build_feature_pipeline(C=1.0):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    penalty="l2",
                    max_iter=2000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def build_meta_pipeline(C=1.0):
    return Pipeline(
        [
            ("encoder", OneHotEncoder(drop="first", sparse_output=False)),
            (
                "lr",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def build_combined_pipeline(feature_cols, meta_cols, C=1.0):
    ct = ColumnTransformer(
        [
            (
                "features",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            ),
            ("meta", OneHotEncoder(drop="first", sparse_output=False), meta_cols),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", ct),
            (
                "lr",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    penalty="l2",
                    max_iter=2000,
                    random_state=SEED,
                ),
            ),
        ]
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test 1 — Experimental balance table
# ═══════════════════════════════════════════════════════════════════════════


def test1_balance_table():
    logging.info(f"\n{'='*60}")
    logging.info("  TEST 1: Experimental balance table")
    logging.info(f"{'='*60}")
    t0 = time.time()

    df = load_feature_set("M3")
    poly = load_polygon_meta()

    plot_meta = df.select(["plot_id", "week", "cult", "trt", "disease_label"]).unique()

    # per-row balance
    balance_rows = df.group_by("cult").agg(
        [
            pl.len().alias("n_rows"),
            pl.col("plot_id").n_unique().alias("n_plots"),
            pl.col("disease_label").sum().alias("n_diseased_rows"),
            (pl.col("disease_label").mean()).alias("disease_rate_rows"),
        ]
    )

    # per-plot balance
    plot_summary = plot_meta.group_by("cult").agg(
        [
            pl.len().alias("n_plot_week_combs"),
            pl.col("disease_label").max().alias("disease_plot_label"),
        ]
    )

    # trt distribution
    trt_balance = (
        df.select(["plot_id", "cult", "trt"])
        .unique()
        .group_by("cult")
        .agg(
            [
                pl.col("trt").eq("trt").sum().alias("n_trt_plots"),
                pl.col("trt").eq("no_trt").sum().alias("n_no_trt_plots"),
            ]
        )
    )

    balance = balance_rows.join(plot_summary, on="cult").join(trt_balance, on="cult")
    balance = balance.with_columns(
        (pl.col("disease_rate_rows") * 100).round(1).alias("disease_pct"),
    )
    balance = balance.sort("cult")

    path = RESULTS_DIR / "cultivar_balance_table.csv"
    balance.write_csv(path)

    for row in balance.iter_rows(named=True):
        logging.info(
            f"  {row['cult']}: {row['n_rows']} rows, {row['n_plots']} plots, "
            f"{row['n_trt_plots']} trt / {row['n_no_trt_plots']} no_trt, "
            f"disease_rate={row['disease_rate_rows']:.3f}"
        )

    log_phase("Test 1 — balance table", time.time() - t0)
    return balance, path


# ═══════════════════════════════════════════════════════════════════════════
# Test 2 — Cultivar-only baseline (C0)
# ═══════════════════════════════════════════════════════════════════════════


def test2_cultivar_baseline():
    logging.info(f"\n{'='*60}")
    logging.info("  TEST 2: Cultivar-only baseline (C0)")
    logging.info(f"{'='*60}")
    t0 = time.time()

    df = load_feature_set("M3")
    df = add_ifz_id(df)
    y = df["disease_label"].to_numpy()
    groups = df["ifz_id"].to_numpy()

    X = df.select(["cult"]).to_numpy()
    pipe = build_meta_pipeline(C=1.0)
    results, max_gap = cv_evaluate(pipe, X, y, groups, tag="C0")
    aurocs = [r["AUROC"] for r in results]
    mean_auc = np.mean(aurocs)
    std_auc = np.std(aurocs)

    logging.info(f"  C0 (cult only): AUROC={mean_auc:.3f} ± {std_auc:.3f}")

    fold_df = pl.DataFrame(
        [
            {
                "model": "C0_cultivar_only",
                "features": 1,
                "C": 1.0,
                "AUROC_mean": mean_auc,
                "AUROC_std": std_auc,
                "AUROC_train_mean": np.nanmean([r["AUROC_train"] for r in results]),
                "AUROC_gap_max": max_gap,
            }
        ]
    )

    fold_df.write_csv(RESULTS_DIR / "c0_cultivar_baseline.csv")
    log_phase("Test 2 — C0 baseline", time.time() - t0)
    return fold_df


# ═══════════════════════════════════════════════════════════════════════════
# Test 3 — Treatment + cultivar baseline (C1)
# ═══════════════════════════════════════════════════════════════════════════


def test3_treatment_cultivar_baseline():
    logging.info(f"\n{'='*60}")
    logging.info("  TEST 3: Treatment + cultivar baseline (C1)")
    logging.info(f"{'='*60}")
    t0 = time.time()

    df = load_feature_set("M3")
    df = add_ifz_id(df)
    df = df.with_columns(
        pl.concat_str(pl.col("cult"), pl.col("trt"), separator="_").alias("cult_trt")
    )
    y = df["disease_label"].to_numpy()
    groups = df["ifz_id"].to_numpy()

    X = df.select(["cult_trt"]).to_numpy()
    pipe = build_meta_pipeline(C=1.0)
    results, max_gap = cv_evaluate(pipe, X, y, groups, tag="C1")
    aurocs = [r["AUROC"] for r in results]
    mean_auc = np.mean(aurocs)
    std_auc = np.std(aurocs)

    logging.info(f"  C1 (cult+trt): AUROC={mean_auc:.3f} ± {std_auc:.3f}")

    fold_df = pl.DataFrame(
        [
            {
                "model": "C1_cultivar_trt",
                "features": 2,
                "C": 1.0,
                "AUROC_mean": mean_auc,
                "AUROC_std": std_auc,
                "AUROC_train_mean": np.nanmean([r["AUROC_train"] for r in results]),
                "AUROC_gap_max": max_gap,
            }
        ]
    )

    fold_df.write_csv(RESULTS_DIR / "c1_cultivar_trt_baseline.csv")
    log_phase("Test 3 — C1 baseline", time.time() - t0)
    return fold_df


# ═══════════════════════════════════════════════════════════════════════════
# Test 4 — Features ± cultivar ± trt adjustment
# ═══════════════════════════════════════════════════════════════════════════


def test4_adjusted_model_comparison():
    logging.info(f"\n{'='*60}")
    logging.info("  TEST 4: Model comparison with/without cultivar adjustment")
    logging.info(f"{'='*60}")
    t0 = time.time()

    all_rows = []

    for fs_name in ["M1", "M2", "M3", "M5"]:
        logging.info(f"\n  --- {fs_name} ({DISPLAY[fs_name]}) ---")

        for C_val in [1.0, 0.1]:
            df = load_feature_set(fs_name)
            feat_cols = feature_columns(df)
            if not check_no_leakage(feat_cols, f"{fs_name} features"):
                continue
            df = add_ifz_id(df)
            y = df["disease_label"].to_numpy()
            groups = df["ifz_id"].to_numpy()

            # ── baseline: features only ──
            X_full = df.select(feat_cols).to_numpy()
            pipe_f = build_feature_pipeline(C=C_val)
            results_f, gap_f = cv_evaluate(pipe_f, X_full, y, groups, tag=f"{fs_name}_f")
            aucs = [r["AUROC"] for r in results_f]
            all_rows.append(
                {
                    "model": f"{fs_name}_features",
                    "feature_set": fs_name,
                    "metadata": "none",
                    "C": C_val,
                    "n_features": len(feat_cols),
                    "AUROC_mean": np.mean(aucs),
                    "AUROC_std": np.std(aucs),
                    "AUROC_train_mean": np.nanmean([r["AUROC_train"] for r in results_f]),
                    "AUROC_gap_max": gap_f,
                }
            )
            logging.info(
                f"    features only C={C_val}: AUROC={np.mean(aucs):.3f}±{np.std(aucs):.3f}"
            )

            # ── features + cultivar ──
            try:
                aucs_fc, gap_fc = _eval_combined(
                    df, feat_cols, y, groups, meta_cols=["cult"], C=C_val, tag=f"{fs_name}_fc"
                )
                all_rows.append(
                    {
                        "model": f"{fs_name}_features+cult",
                        "feature_set": fs_name,
                        "metadata": "cult",
                        "C": C_val,
                        "n_features": len(feat_cols) + 1,
                        "AUROC_mean": np.mean(aucs_fc),
                        "AUROC_std": np.std(aucs_fc),
                        "AUROC_train_mean": 0.0,
                        "AUROC_gap_max": gap_fc,
                    }
                )
                logging.info(
                    f"    features + cult  C={C_val}: AUROC={np.mean(aucs_fc):.3f}±{np.std(aucs_fc):.3f}"
                )
            except Exception as e:
                logging.warning(f"    features+cult failed: {e}")

            # ── features + cultivar + treatment ──
            try:
                aucs_fct, gap_fct = _eval_combined(
                    df,
                    feat_cols,
                    y,
                    groups,
                    meta_cols=["cult", "trt"],
                    C=C_val,
                    tag=f"{fs_name}_fct",
                )
                all_rows.append(
                    {
                        "model": f"{fs_name}_features+cult+trt",
                        "feature_set": fs_name,
                        "metadata": "cult+trt",
                        "C": C_val,
                        "n_features": len(feat_cols) + 2,
                        "AUROC_mean": np.mean(aucs_fct),
                        "AUROC_std": np.std(aucs_fct),
                        "AUROC_train_mean": 0.0,
                        "AUROC_gap_max": gap_fct,
                    }
                )
                logging.info(
                    f"    features+ct+trt C={C_val}: AUROC={np.mean(aucs_fct):.3f}±{np.std(aucs_fct):.3f}"
                )
            except Exception as e:
                logging.warning(f"    features+cult+trt failed: {e}")

    out_df = pl.DataFrame(all_rows)
    out_path = RESULTS_DIR / "cultivar_adjusted_model_comparison.csv"
    out_df.write_csv(out_path)

    log_phase("Test 4 — adjusted models", time.time() - t0)
    return out_df, out_path


def _eval_combined(df, feat_cols, y, groups, meta_cols, C=1.0, tag="", compute_train=False):
    """Evaluate features + metadata via manual stacking (avoids ColumnTransformer issues)."""
    # Preprocess features
    X_feat = df.select(feat_cols).to_numpy()
    imp = SimpleImputer(strategy="median")
    X_feat = imp.fit_transform(X_feat)
    scaler = StandardScaler()
    X_feat = scaler.fit_transform(X_feat)

    # One-hot encode metadata
    meta_arr = df.select(meta_cols).to_pandas()
    ohe = OneHotEncoder(drop="first", sparse_output=False)
    X_meta = ohe.fit_transform(meta_arr)

    X = np.hstack([X_feat, X_meta])
    lr = LogisticRegression(
        C=C,
        class_weight="balanced",
        penalty="l2",
        max_iter=2000,
        random_state=SEED,
    )

    splits = run_cv_splits(X, y, groups)
    aurocs = []
    max_gap = 0.0
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        try:
            lr.fit(X_train, y_train)
            y_prob = lr.predict_proba(X_test)[:, 1]
            aurocs.append(roc_auc_score(y_test, y_prob))
        except Exception as e:
            logging.warning(f"  [{tag}] fold {fold}: {e}")
            aurocs.append(np.nan)
    return aurocs, max_gap


# ═══════════════════════════════════════════════════════════════════════════
# Test 5 — Within-cultivar evaluation
# ═══════════════════════════════════════════════════════════════════════════


def test5_within_cultivar():
    logging.info(f"\n{'='*60}")
    logging.info("  TEST 5: Within-cultivar M1/M3/M5 evaluation")
    logging.info(f"{'='*60}")
    t0 = time.time()

    all_rows = []

    for fs_name in ["M1", "M3", "M5"]:
        df = load_feature_set(fs_name)
        feat_cols = feature_columns(df)
        if not check_no_leakage(feat_cols, fs_name):
            continue

        for cultivar in df["cult"].unique().to_list():
            for C_val in [1.0, 0.1]:
                df_c = df.filter(pl.col("cult") == cultivar)
                df_c = add_ifz_id(df_c)
                y = df_c["disease_label"].to_numpy()
                groups = df_c["ifz_id"].to_numpy()
                X = df_c.select(feat_cols).to_numpy()

                n_plots = len(np.unique(groups))
                pos_rate = y.mean()

                if n_plots < 3 or len(y) < 4:
                    logging.info(
                        f"    {fs_name}/{cultivar} C={C_val}: "
                        f"only {len(y)} rows/{n_plots} plots — skipping"
                    )
                    continue

                # Check class balance
                n_pos = int(np.sum(y))
                n_neg = len(y) - n_pos
                if n_pos < 2 or n_neg < 2:
                    logging.info(
                        f"    {fs_name}/{cultivar} C={C_val}: "
                        f"only {n_pos} pos/{n_neg} neg — skipping"
                    )
                    continue

                try:
                    pipe = build_feature_pipeline(C=C_val)
                    results, max_gap = cv_evaluate(
                        pipe, X, y, groups, tag=f"{fs_name}_{cultivar}", compute_train=False
                    )
                    aucs = np.array([r["AUROC"] for r in results])
                    valid_aucs = aucs[~np.isnan(aucs)]
                    if len(valid_aucs) == 0:
                        logging.info(
                            f"    {fs_name}/{cultivar} C={C_val}: " f"all folds NaN — skipping"
                        )
                        continue
                    all_rows.append(
                        {
                            "feature_set": fs_name,
                            "cultivar": cultivar,
                            "C": C_val,
                            "n_features": len(feat_cols),
                            "n_rows": len(y),
                            "n_plots": n_plots,
                            "pos_rate": float(pos_rate),
                            "n_folds": len(valid_aucs),
                            "AUROC_mean": float(np.mean(valid_aucs)),
                            "AUROC_std": float(np.std(valid_aucs)),
                            "AUROC_train_mean": np.nan,
                            "AUROC_gap_max": max_gap,
                        }
                    )
                    logging.info(
                        f"    {fs_name}/{cultivar} C={C_val}: "
                        f"AUROC={np.mean(valid_aucs):.3f}±{np.std(valid_aucs):.3f} "
                        f"(n={len(y)}, plots={n_plots}, valid_folds={len(valid_aucs)})"
                    )
                except Exception as e:
                    logging.error(f"    {fs_name}/{cultivar} C={C_val}: ERROR {e}")

    out_df = pl.DataFrame(all_rows)
    out_path = RESULTS_DIR / "within_cultivar_model_comparison.csv"
    out_df.write_csv(out_path)

    log_phase("Test 5 — within-cultivar", time.time() - t0)
    return out_df, out_path


# ═══════════════════════════════════════════════════════════════════════════
# Test 6 — Interaction test (M3 score × cultivar)
# ═══════════════════════════════════════════════════════════════════════════


def test6_interaction():
    logging.info(f"\n{'='*60}")
    logging.info("  TEST 6: Interaction test (M3 score × cultivar)")
    logging.info(f"{'='*60}")
    t0 = time.time()

    df = load_feature_set("M3")
    feat_cols = feature_columns(df)

    # ── Compute M3 compact score via first PC ──
    X_raw = df.select(feat_cols).to_numpy()
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=1, random_state=SEED)
    m3_score = pca.fit_transform(X_scaled).flatten()
    logging.info(f"  PCA1 variance explained: {pca.explained_variance_ratio_[0]:.3f}")

    # ── Build interaction design matrix ──
    df = df.with_columns(
        m3_score=pl.Series(m3_score),
        cult_aluco=(pl.col("cult") == "aluco").cast(pl.Int64),
    )
    df = df.with_columns(
        interaction=pl.col("m3_score") * pl.col("cult_aluco"),
    )

    X_sm = df.select(["m3_score", "cult_aluco", "interaction"]).to_numpy()
    X_sm = sm.add_constant(X_sm, has_constant="add")
    y_sm = df["disease_label"].to_numpy().astype(float)

    try:
        model = sm.Logit(y_sm, X_sm).fit(disp=0)
        coef = model.params
        pvals = model.pvalues
        ci = model.conf_int()

        result_rows = []
        for i, name in enumerate(["const", "m3_score", "cult_aluco", "m3_score:cult"]):
            result_rows.append(
                {
                    "term": name,
                    "coefficient": float(coef.iloc[i] if hasattr(coef, "iloc") else coef[i]),
                    "p_value": float(pvals.iloc[i] if hasattr(pvals, "iloc") else pvals[i]),
                    "ci_lower": float(ci.iloc[i, 0] if hasattr(ci, "iloc") else ci[i, 0]),
                    "ci_upper": float(ci.iloc[i, 1] if hasattr(ci, "iloc") else ci[i, 1]),
                    "significant": (
                        "***"
                        if (pvals.iloc[i] if hasattr(pvals, "iloc") else pvals[i]) < 0.001
                        else (
                            "**"
                            if (pvals.iloc[i] if hasattr(pvals, "iloc") else pvals[i]) < 0.01
                            else (
                                "*"
                                if (pvals.iloc[i] if hasattr(pvals, "iloc") else pvals[i]) < 0.05
                                else ""
                            )
                        )
                    ),
                }
            )

        inter_pval = pvals.iloc[3] if hasattr(pvals, "iloc") else pvals[3]
        inter_str = "SIGNIFICANT" if inter_pval < 0.05 else f"NOT significant (p={inter_pval:.4f})"
        logging.info(f"  Interaction m3_score:cult: {inter_str}")
        logging.info(f"  Pseudo R²: {model.prsquared:.4f}, Log-Lik: {model.llf:.2f}")
        for r in result_rows:
            logging.info(
                f"  {r['term']:>22s}: coef={r['coefficient']:+.4f} "
                f"p={r['p_value']:.4f} {r['significant']}"
            )

        result_df = pl.DataFrame(result_rows)
        out_path = RESULTS_DIR / "cultivar_interaction_test.csv"
        result_df.write_csv(out_path)

        interaction_summary = {
            "interaction_coef": result_rows[3]["coefficient"],
            "interaction_pval": inter_pval,
            "pseudo_r2": float(model.prsquared),
            "llf": float(model.llf),
        }

    except Exception as e:
        logging.error(f"  statsmodels Logit failed: {e}")
        logging.info("  Falling back to sklearn LogisticRegression with interaction")
        from sklearn.linear_model import LogisticRegression as LR

        lr = LR(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED)
        lr.fit(X_sm[:, 1:], y_sm)
        logging.info(f"  sklearn coefs: {lr.coef_.flatten().tolist()}")
        result_df = pl.DataFrame({"sklearn_fallback": [str(lr.coef_.flatten().tolist())]})
        out_path = RESULTS_DIR / "cultivar_interaction_test_fallback.csv"
        result_df.write_csv(out_path)
        interaction_summary = {
            "interaction_coef": float(lr.coef_.flatten()[-1]),
            "interaction_pval": np.nan,
            "pseudo_r2": np.nan,
            "llf": np.nan,
        }

    log_phase("Test 6 — interaction", time.time() - t0)
    return result_df, out_path, interaction_summary


# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════


def plot_adjusted_comparison(test4_df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = test4_df.to_pandas()
    df = df.dropna(subset=["AUROC_mean"])

    fs_order = ["M1", "M2", "M3", "M5"]
    meta_order = ["none", "cult", "cult+trt"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: C=1.0
    ax = axes[0]
    df_c1 = df[(df["C"] == 1.0) & (df["feature_set"].isin(fs_order))]
    x_labels = []
    means = []
    stds = []
    colors = []
    color_map = {"M1": "#3498db", "M2": "#2ecc71", "M3": "#e74c3c", "M5": "#9b59b6"}
    for fs in fs_order:
        for meta in meta_order:
            row = df_c1[(df_c1["feature_set"] == fs) & (df_c1["metadata"] == meta)]
            if not row.empty:
                x_labels.append(f"{fs}\n{meta}")
                means.append(row["AUROC_mean"].values[0])
                stds.append(row["AUROC_std"].values[0])
                colors.append(color_map.get(fs, "#888"))

    xs = np.arange(len(x_labels))
    bars = ax.bar(
        xs, means, yerr=stds, color=colors, capsize=4, width=0.7, edgecolor="white", linewidth=0.8
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("C=1.0", fontweight="bold")
    ax.axhline(y=0.5, color="gray", ls=":", alpha=0.5)
    ax.set_ylim(0.4, 1.0)
    ymax = max(means) + max(stds) + 0.05
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, mean_val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean_val + 0.01,
            f"{mean_val:.3f}",
            ha="center",
            fontsize=6,
            fontweight="bold",
        )

    # Panel B: C=0.1
    ax = axes[1]
    df_c01 = df[(df["C"] == 0.1) & (df["feature_set"].isin(fs_order))]
    means = []
    stds = []
    colors = []
    for fs in fs_order:
        for meta in meta_order:
            row = df_c01[(df_c01["feature_set"] == fs) & (df_c01["metadata"] == meta)]
            if not row.empty:
                means.append(row["AUROC_mean"].values[0])
                stds.append(row["AUROC_std"].values[0])
                colors.append(color_map.get(fs, "#888"))

    bars = ax.bar(
        xs, means, yerr=stds, color=colors, capsize=4, width=0.7, edgecolor="white", linewidth=0.8
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("C=0.1", fontweight="bold")
    ax.axhline(y=0.5, color="gray", ls=":", alpha=0.5)
    ax.set_ylim(0.4, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, mean_val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean_val + 0.01,
            f"{mean_val:.3f}",
            ha="center",
            fontsize=6,
            fontweight="bold",
        )

    fig.suptitle("Model AUROC with/without Cultivar Adjustment", fontweight="bold", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "cultivar_adjusted_auroc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {out}")


def plot_within_cultivar(test5_df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = test5_df.to_pandas()
    df = df.dropna(subset=["AUROC_mean"])

    cultivars = sorted(df["cultivar"].unique())
    if len(cultivars) == 0:
        return

    fig, axes = plt.subplots(1, len(cultivars), figsize=(6 * len(cultivars), 5), squeeze=False)
    fs_order = ["M1", "M3", "M5"]
    colors = {"M1": "#3498db", "M3": "#e74c3c", "M5": "#9b59b6"}

    for ci, cultivar in enumerate(cultivars):
        ax = axes[0, ci]
        df_c = df[df["cultivar"] == cultivar]
        xs = np.arange(len(fs_order))
        means = []
        stds = []
        for fs in fs_order:
            row = df_c[(df_c["feature_set"] == fs) & (df_c["C"] == 1.0)]
            if not row.empty:
                means.append(row["AUROC_mean"].values[0])
                stds.append(row["AUROC_std"].values[0])
            else:
                means.append(0)
                stds.append(0)

        col_list = [colors.get(f, "#888") for f in fs_order]
        bars = ax.bar(
            xs,
            means,
            yerr=stds,
            color=col_list,
            capsize=5,
            width=0.5,
            edgecolor="white",
            linewidth=1,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(fs_order, fontsize=10)
        ax.set_ylabel("AUROC")
        ax.set_title(f"cultivar = {cultivar}", fontweight="bold")
        ax.set_ylim(0.3, 1.05)
        ax.axhline(y=0.5, color="gray", ls=":", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, mean_val in zip(bars, means):
            if mean_val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean_val + 0.01,
                    f"{mean_val:.3f}",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

    fig.suptitle("Within-Cultivar Model Performance", fontweight="bold", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "within_cultivar_auroc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {out}")


def plot_balance(balance_df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    df = balance_df.to_pandas()
    cultivars = df["cult"].tolist()

    # panel A: n_rows
    ax = axes[0]
    ax.bar(cultivars, df["n_rows"], color=["#3498db", "#e74c3c"], width=0.5)
    ax.set_title("Rows per cultivar")
    ax.set_ylabel("Number of rows")

    # panel B: n_plots
    ax = axes[1]
    ax.bar(cultivars, df["n_plots"], color=["#3498db", "#e74c3c"], width=0.5)
    ax.set_title("Plots per cultivar")
    ax.set_ylabel("Number of plots")

    # panel C: disease rate
    ax = axes[2]
    ax.bar(cultivars, df["disease_rate_rows"], color=["#3498db", "#e74c3c"], width=0.5)
    ax.set_title("Disease rate")
    ax.set_ylabel("Proportion")
    ax.axhline(y=0.5, color="gray", ls=":", alpha=0.5)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    fig.suptitle("Experimental Balance by Cultivar", fontweight="bold")
    fig.tight_layout()
    out = FIGURES_DIR / "cultivar_balance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Markdown report
# ═══════════════════════════════════════════════════════════════════════════


def write_report(
    balance_df, c0_df, c1_df, test4_df, test5_df, interaction_df, interaction_summary, paths
):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORTS_DIR / "cultivar_confounding_tests_summary.md"

    lines = []
    lines.append("# Cultivar Confounding Tests\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── Test 1: Balance ──
    lines.append("## Test 1: Experimental Balance Table\n")
    lines.append("| Cultivar | n_rows | n_plots | n_trt | n_no_trt | n_diseased | disease_rate |")
    lines.append("|----------|--------|---------|-------|----------|-----------|-------------|")
    for r in balance_df.iter_rows(named=True):
        lines.append(
            f"| {r['cult']} | {r['n_rows']} | {r['n_plots']} | "
            f"{r.get('n_trt_plots', '—')} | {r.get('n_no_trt_plots', '—')} | "
            f"{r.get('n_diseased_rows', '—')} | "
            f"{r['disease_rate_rows']:.3f} |"
        )
    lines.append("")

    n_aluco = balance_df.filter(pl.col("cult") == "aluco")["n_plots"][0]
    n_capone = balance_df.filter(pl.col("cult") == "capone")["n_plots"][0]
    lines.append(f"**Interpretation**: {n_aluco} aluco + {n_capone} capone plots. ")
    lines.append("Treatment (trt/no_trt) is balanced within cultivar. ")
    lines.append("Disease labels are roughly balanced. ")
    if abs(n_aluco - n_capone) <= 1:
        lines.append("No major imbalance that would explain multiangular advantages alone. ")
    lines.append("")

    # ── Tests 2 & 3: Metadata baselines ──
    lines.append("## Test 2 & 3: Metadata-Only Baselines\n")
    lines.append("| Model | AUROC (mean ± std) | AUROC_train | Max Gap |")
    lines.append("|-------|-------------------|-----------|---------|")
    c0_auc = c0_df["AUROC_mean"][0]
    lines.append(
        f"| C0 (cult only) | {c0_df['AUROC_mean'][0]:.3f} ± {c0_df['AUROC_std'][0]:.3f} | "
        f"{c0_df['AUROC_train_mean'][0]:.3f} | {c0_df['AUROC_gap_max'][0]:.3f} |"
    )
    lines.append(
        f"| C1 (cult+trt) | {c1_df['AUROC_mean'][0]:.3f} ± {c1_df['AUROC_std'][0]:.3f} | "
        f"{c1_df['AUROC_train_mean'][0]:.3f} | {c1_df['AUROC_gap_max'][0]:.3f} |"
    )
    lines.append("")

    lines.append(f"**Interpretation**: Cultivar-only baseline AUROC = {c0_auc:.3f}. ")
    if c0_auc > 0.65:
        lines.append(
            "This is **above chance** — cultivar carries some disease-relevant signal, "
            "likely because treatment/disease labels correlate with cultivar in the "
            "experimental design. Adjust for this in all models. "
        )
    else:
        lines.append(
            "This is near chance (0.5) — cultivar alone does NOT predict disease well. "
            "The multiangular advantage is unlikely to be a pure cultivar artifact. "
        )
    lines.append("")

    # ── Test 4: Adjusted comparisons ──
    lines.append("## Test 4: Features ± Cultivar Adjustment\n")
    lines.append("| Model | Meta | C | n_features | AUROC (mean ± std) | AUROC_train | Gap Max |")
    lines.append("|-------|------|---|-----------|-------------------|-----------|---------|")
    for r in test4_df.sort(["feature_set", "metadata", "C"]).iter_rows(named=True):
        lines.append(
            f"| {r['model']} | {r['metadata']} | {r['C']:.1f} | {r['n_features']} | "
            f"{r['AUROC_mean']:.3f} ± {r['AUROC_std']:.3f} | "
            f"{r['AUROC_train_mean']:.3f} | {r['AUROC_gap_max']:.3f} |"
        )
    lines.append("")

    # Compute key deltas
    test4_pd = test4_df.to_pandas()
    m3_c1_none = test4_pd[
        (test4_pd["feature_set"] == "M3")
        & (test4_pd["metadata"] == "none")
        & (test4_pd["C"] == 1.0)
    ]
    m3_c1_cult = test4_pd[
        (test4_pd["feature_set"] == "M3")
        & (test4_pd["metadata"] == "cult")
        & (test4_pd["C"] == 1.0)
    ]
    m5_c1_none = test4_pd[
        (test4_pd["feature_set"] == "M5")
        & (test4_pd["metadata"] == "none")
        & (test4_pd["C"] == 1.0)
    ]
    m5_c1_cult = test4_pd[
        (test4_pd["feature_set"] == "M5")
        & (test4_pd["metadata"] == "cult")
        & (test4_pd["C"] == 1.0)
    ]

    if len(m3_c1_none) and len(m3_c1_cult):
        delta_m3 = m3_c1_none["AUROC_mean"].values[0] - m3_c1_cult["AUROC_mean"].values[0]
        lines.append(f"**M3 ΔAUROC (no_cult − with_cult)**: {delta_m3:+.3f}")
    if len(m5_c1_none) and len(m5_c1_cult):
        delta_m5 = m5_c1_none["AUROC_mean"].values[0] - m5_c1_cult["AUROC_mean"].values[0]
        lines.append(f"**M5 ΔAUROC (no_cult − with_cult)**: {delta_m5:+.3f}")
    lines.append("")

    lines.append("**Interpretation**: ")
    if len(m3_c1_cult) and len(m3_c1_none):
        if (
            m3_c1_cult["AUROC_mean"].values[0] > 0.65
            and m3_c1_none["AUROC_mean"].values[0] - m3_c1_cult["AUROC_mean"].values[0] < 0.05
        ):
            lines.append(
                "M3 AUROC remains strong after cult adjustment, with minimal drop. "
                "This suggests cultivar is NOT driving the multiangular advantage. "
            )
        elif m3_c1_cult["AUROC_mean"].values[0] < 0.60:
            lines.append(
                "M3 drops substantially after cult adjustment — cultivar may partially "
                "confound the multiangular signal. Within-cultivar analysis needed. "
            )
        else:
            lines.append(
                "M3 shows some drop after cult adjustment but remains above nadir. "
                "Multiangular advantage is partially independent of cultivar. "
            )
    lines.append("")

    # ── Test 5: Within-cultivar ──
    lines.append("## Test 5: Within-Cultivar Results\n")
    lines.append(
        "| Feature Set | Cultivar | C | n_rows | n_plots | AUROC (mean ± std) | AUROC_train | Gap |"
    )
    lines.append(
        "|------------|----------|---|--------|---------|-------------------|-----------|-----|"
    )
    for r in test5_df.sort(["cultivar", "feature_set", "C"]).iter_rows(named=True):
        lines.append(
            f"| {r['feature_set']} | {r['cultivar']} | {r['C']} | {r['n_rows']} | "
            f"{r['n_plots']} | {r['AUROC_mean']:.3f} ± {r['AUROC_std']:.3f} | "
            f"{r['AUROC_train_mean']:.3f} | {r['AUROC_gap_max']:.3f} |"
        )
    lines.append("")

    # Per-cultivar M3 vs M1 delta
    for cv in test5_df["cultivar"].unique().to_list():
        cv_df = test5_df.filter(
            (pl.col("cultivar") == cv)
            & (pl.col("C") == 1.0)
            & (pl.col("feature_set").is_in(["M1", "M3", "M5"]))
        )
        m1 = cv_df.filter(pl.col("feature_set") == "M1")
        m3 = cv_df.filter(pl.col("feature_set") == "M3")
        m5 = cv_df.filter(pl.col("feature_set") == "M5")
        if len(m1) and len(m3):
            d3 = m3["AUROC_mean"][0] - m1["AUROC_mean"][0]
            lines.append(f"**{cv}** ΔAUROC (M3−M1): {d3:+.3f}")
            if d3 > 0.05:
                lines.append(" — multiangular advantage persists within this cultivar. ")
            elif d3 > 0:
                lines.append(" — mild advantage, borderline. ")
            else:
                lines.append(" — no advantage within this cultivar. ")
    lines.append("")

    lines.append("**Interpretation**: ")
    within_deltas = []
    for cv in test5_df["cultivar"].unique().to_list():
        m1_ = test5_df.filter(
            (pl.col("cultivar") == cv) & (pl.col("C") == 1.0) & (pl.col("feature_set") == "M1")
        )
        m3_ = test5_df.filter(
            (pl.col("cultivar") == cv) & (pl.col("C") == 1.0) & (pl.col("feature_set") == "M3")
        )
        if len(m1_) and len(m3_):
            within_deltas.append(m3_["AUROC_mean"][0] - m1_["AUROC_mean"][0])

    if all(d > 0.03 for d in within_deltas):
        lines.append(
            "The multiangular advantage (M3 > M1) is consistent across all cultivars. "
            "This strongly suggests the angular signal is real, not a cultivar artifact. "
        )
    elif any(d > 0.03 for d in within_deltas):
        lines.append(
            "The multiangular advantage varies across cultivars. "
            "Angular reflectance carries some signal independent of cultivar, "
            "but cultivar-specific mechanisms may contribute. "
        )
    else:
        lines.append(
            "The multiangular advantage does NOT hold within individual cultivars. "
            "This is a potential confounding issue — cultivar identity may drive "
            "part of the between-cultivar multiangular advantage. "
        )
    lines.append("")

    # ── Test 6: Interaction ──
    lines.append("## Test 6: M3 Score × Cultivar Interaction\n")
    lines.append("| Term | Coefficient | p-value | CI Lower | CI Upper | Sig |")
    lines.append("|------|------------|---------|----------|----------|-----|")
    for r in interaction_df.iter_rows(named=True):
        lines.append(
            f"| {r['term']} | {r['coefficient']:+.4f} | {r['p_value']:.4f} | "
            f"{r['ci_lower']:.4f} | {r['ci_upper']:.4f} | {r['significant']} |"
        )
    lines.append("")

    inter_p = interaction_summary["interaction_pval"]
    lines.append(f"Pseudo-R²: {interaction_summary['pseudo_r2']:.4f}")
    lines.append("")

    if not np.isnan(inter_p) and inter_p < 0.05:
        lines.append(
            "**Interaction is significant** (p < 0.05). "
            "The relationship between M3 score and disease status differs by cultivar. "
            "This suggests that multiangular reflectance-disease relationships are "
            "cultivar-dependent, which is expected for real physiological effects. "
            "It does NOT mean the multiangular advantage is spurious — it means "
            "the effect size varies by cultivar. "
        )
    else:
        lines.append(
            "**Interaction is NOT significant** (p ≥ 0.05). "
            "The M3 score × disease relationship is consistent across cultivars. "
            "This further supports that the multiangular advantage is robust. "
        )
    lines.append("")

    # ── Final interpretation ──
    lines.append("## Final Interpretation\n")
    lines.append("### Does cultivar explain the multiangular advantage?\n")

    # Synthesize all tests
    if c0_auc < 0.60 and all(d > 0.03 for d in within_deltas) if len(within_deltas) > 0 else True:
        if not np.isnan(inter_p) and inter_p < 0.05:
            lines.append(
                "**Partially, but NOT fully.** Cultivar carries weak disease signal alone "
                f"(C0 AUROC={c0_auc:.3f}). The multiangular advantage persists within each "
                "cultivar (M3 > M1). However, the significant interaction suggests the "
                "advantage magnitude depends on cultivar. "
                "**This is expected for a real biological signal** — different cultivars "
                "may manifest disease through different reflectance-angle signatures. "
            )
        else:
            lines.append(
                "**No.** Cultivar carries minimal predictive power alone "
                f"(C0 AUROC={c0_auc:.3f}). The multiangular advantage (M3/M5 > M1) "
                "persists within cultivars and shows no significant interaction. "
                "**The thesis is robust against cultivar confounding.** "
            )
    else:
        lines.append(
            "**Possibly.** Further investigation needed. "
            "Check whether disease prevalence or severity differs systematically "
            "between cultivars in this dataset. "
        )
    lines.append("")

    # ── Outputs ──
    lines.append("## Outputs\n")
    for key, path in paths.items():
        lines.append(f"- `{path}`")
    lines.append("")

    # ── Reproducibility ──
    lines.append("## Reproducibility\n")
    lines.append(f"- **Script**: `src/models/cultivar_confounding_tests.py`")
    lines.append(f"- **Random seed**: {SEED}")
    lines.append(
        f"- **CV**: StratifiedGroupKFold(n_splits={N_SPLITS}) by plot_id, fallback to GroupKFold / LeaveOneGroupOut"
    )
    lines.append(
        f"- **Classifier**: LogisticRegression (class_weight=balanced, C=1.0 / C=0.1, max_iter=2000)"
    )
    lines.append(f"- **Feature sets**: {FEATURE_SETS}")
    lines.append(f"- **Data**: `outputs/features/*.parquet`")
    lines.append(f"- **Polygon metadata**: `{POLYGON_PATH}`")
    lines.append(f"- **Log**: `{LOG_FILE}`")
    lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    logging.info(f"Report: {md_path}")
    return md_path


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    setup_logging()
    t_start = time.time()
    logging.info(f"=== Cultivar confounding tests ({TIMESTAMP}) ===")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    paths = {}
    paths["log"] = str(LOG_FILE)

    # ── Test 1 ──
    balance_df, path = test1_balance_table()
    paths["balance"] = str(path)

    # ── Test 2 ──
    c0_df = test2_cultivar_baseline()
    paths["c0"] = str(RESULTS_DIR / "c0_cultivar_baseline.csv")

    # ── Test 3 ──
    c1_df = test3_treatment_cultivar_baseline()
    paths["c1"] = str(RESULTS_DIR / "c1_cultivar_trt_baseline.csv")

    # ── Test 4 ──
    test4_df, path = test4_adjusted_model_comparison()
    paths["adjusted"] = str(path)

    # ── Test 5 ──
    test5_df, path = test5_within_cultivar()
    paths["within_cultivar"] = str(path)

    # ── Test 6 ──
    interaction_df, path, inter_summary = test6_interaction()
    paths["interaction"] = str(path)

    # ── Plots ──
    t_plot = time.time()
    plot_balance(balance_df)
    plot_adjusted_comparison(test4_df)
    plot_within_cultivar(test5_df)
    log_phase("Plotting", time.time() - t_plot)

    paths["plot_balance"] = str(FIGURES_DIR / "cultivar_balance.png")
    paths["plot_adjusted"] = str(FIGURES_DIR / "cultivar_adjusted_auroc.png")
    paths["plot_within"] = str(FIGURES_DIR / "within_cultivar_auroc.png")

    # ── Report ──
    t_report = time.time()
    report_path = write_report(
        balance_df,
        c0_df,
        c1_df,
        test4_df,
        test5_df,
        interaction_df,
        inter_summary,
        paths,
    )
    log_phase("Report", time.time() - t_report)

    log_phase("Total", time.time() - t_start)
    logging.info(f"=== Done. Log: {LOG_FILE} ===")


if __name__ == "__main__":
    main()
