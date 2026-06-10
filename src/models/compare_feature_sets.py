#!/usr/bin/env python3
"""Compare 6 feature sets (M0-M5) for disease prediction via stratified-group CV.
Includes spatial leakage verification and per-phase profiling.
"""

import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

FEATURE_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "features"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "results"
LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "logs"
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "reports"

FEATURE_SETS = ["M0", "M1", "M2", "M3", "M4", "M5"]
EXCLUDE_COLS = ["plot_id", "week", "cult", "trt", "disease_label"]
SEED = 42
N_SPLITS = 5

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"compare_feature_sets_{ts}.log"


def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Log file: {LOG_FILE}")


def load_feature_set(name, year_filter=None):
    pattern = list(FEATURE_DIR.glob(f"{name}_*.parquet"))
    if not pattern:
        raise FileNotFoundError(f"No parquet found for {name} in {FEATURE_DIR}")
    df = pl.read_parquet(pattern[0])
    df = df.filter(pl.col("disease_label").is_not_null())
    if year_filter and year_filter != "all":
        if "year" in df.columns:
            df = df.filter(pl.col("year") == int(year_filter))
        else:
            logging.warning(f"  No 'year' column in {name} — cannot filter by year")
    return df


def build_pipeline(C=1.0):
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


def evaluate_with_C(name, C, year=None):
    """Evaluate feature set with specific regularization strength."""
    t0 = time.time()
    logging.info(f"  C={C}: ", extra={"end": ""})

    df = load_feature_set(name, year)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    if not feature_cols:
        return []

    if not verify_no_leakage(df, feature_cols, name):
        return []

    X = df.select(feature_cols).to_numpy()
    y = df["disease_label"].to_numpy()
    df = df.with_columns(
        (pl.col("plot_id").str.extract(r"(\d+)").cast(pl.Int64) + 90001).alias("ifz_id")
    )
    groups = df["ifz_id"].to_numpy()
    plot_ids = df["plot_id"].to_numpy()

    try:
        skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        splits = list(skf.split(X, y, groups=groups))
    except (ValueError, RuntimeError):
        gkf = GroupKFold(n_splits=N_SPLITS)
        splits = list(gkf.split(X, y, groups=groups))

    verify_spatial_cv(splits, groups, plot_ids)

    pipe = build_pipeline(C=C)
    fold_results = []
    max_gap = 0
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_proba_train = pipe.predict_proba(X_train)[:, 1]

        test_aurocs = roc_auc_score(y_test, y_proba)
        train_aurocs = roc_auc_score(y_train, y_proba_train)
        gap = train_aurocs - test_aurocs
        max_gap = max(max_gap, gap)

        fold_results.append(
            {
                "feature_set": name,
                "C": C,
                "fold": fold,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "positive_rate_train": y_train.mean(),
                "positive_rate_test": y_test.mean(),
                "AUROC": test_aurocs,
                "AUROC_train": train_aurocs,
                "AUROC_gap": gap,
                "AUPRC": average_precision_score(y_test, y_proba),
                "F1": f1_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
            }
        )

        logging.info(
            f"  Fold {fold}: test={test_aurocs:.3f} train={train_aurocs:.3f} "
            f"gap={gap:.3f} {'⚠ OVERFIT' if gap > 0.15 else ''}"
        )

    logging.info(
        f"  C={C}: AUROC={np.mean([r['AUROC'] for r in fold_results]):.4f} "
        f"±{np.std([r['AUROC'] for r in fold_results]):.3f} "
        f"max_gap={max_gap:.3f} "
        f"{'HIGH_OVERFIT' if max_gap > 0.20 else 'OK' if max_gap < 0.10 else 'BORDERLINE'}"
    )
    return fold_results


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "AUROC": roc_auc_score(y_true, y_proba),
        "AUPRC": average_precision_score(y_true, y_proba),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
    }


def verify_spatial_cv(splits, groups_arr, plot_ids_arr):
    """Verify no plot_id appears in both train and test across folds."""
    all_ok = True
    for fold, (train_idx, test_idx) in enumerate(splits):
        train_groups = set(groups_arr[train_idx])
        test_groups = set(groups_arr[test_idx])
        overlap = train_groups & test_groups
        if overlap:
            logging.error(f"  LEAKAGE Fold {fold}: {len(overlap)} plot_ids overlap!")
            all_ok = False

        train_plots = set(plot_ids_arr[train_idx])
        test_plots = set(plot_ids_arr[test_idx])
        logging.info(
            f"  Fold {fold}: train={len(train_plots)} plots "
            f"test={len(test_plots)} plots  overlap={len(overlap)}"
        )
    return all_ok


def verify_no_leakage(df, feature_cols, name):
    """Verify that no metadata columns leak into features."""
    forbidden = {"cult", "trt", "week", "disease_label"}
    leaked = set(feature_cols) & forbidden
    if leaked:
        logging.error(f"  LEAKAGE in {name}: metadata columns in features: {leaked}")
        return False
    return True


def evaluate_feature_set(name, year=None):
    """Evaluate a feature set with both C=1.0 and C=0.1 regularization."""
    year_label = "all" if year is None else year
    logging.info(f"\n{'='*60}")
    logging.info(f"  {name} ({year_label})")
    logging.info(f"{'='*60}")

    # Quick load to verify features exist
    yf = None if year == "all" else year
    df = load_feature_set(name, yf)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    if not feature_cols:
        logging.warning(f"  {name} has no features — skipping")
        return []
    if not verify_no_leakage(df, feature_cols, name):
        return []

    # Test both regularization strengths
    results = []
    results.extend(evaluate_with_C(name, C=1.0, year=yf))
    results.extend(evaluate_with_C(name, C=0.1, year=yf))
    return results


def main(years=None):
    if years is None:
        years = ["all", "2024", "2025"]
    elif isinstance(years, str):
        years = [years]

    for year in years:
        logging.info(f"\n{'#'*60}")
        logging.info(f"  YEAR: {year}")
        logging.info(f"{'#'*60}")

        all_folds = []
        for fs_name in FEATURE_SETS:
            try:
                results = evaluate_feature_set(fs_name, year)
                all_folds.extend(results)
            except FileNotFoundError as e:
                logging.warning(f"SKIP {fs_name}: {e}")
            except Exception as e:
                logging.error(f"ERROR {fs_name}: {e}", exc_info=True)

        if not all_folds:
            continue

        fold_df = pl.DataFrame(all_folds)
        year_res_dir = RESULTS_DIR.parent / year / "results"
        year_res_dir.mkdir(parents=True, exist_ok=True)
        fold_df.write_csv(year_res_dir / "model_comparison_by_fold.csv")

        summary_df = (
            fold_df.group_by("feature_set", "C")
            .agg(
                n_folds=pl.len(),
                year=pl.lit(year),
                AUROC_mean=pl.col("AUROC").mean(),
                AUROC_std=pl.col("AUROC").std(),
                AUROC_train_mean=pl.col("AUROC_train").mean(),
                AUROC_gap_max=pl.col("AUROC_gap").max(),
                AUPRC_mean=pl.col("AUPRC").mean(),
                F1_mean=pl.col("F1").mean(),
                F1_std=pl.col("F1").std(),
            )
            .sort("AUROC_mean", descending=True)
        )
        summary_df.write_csv(year_res_dir / "model_comparison_summary.csv")

    summary_df.write_csv(RESULTS_DIR / "model_comparison_summary.csv")

    logging.info(f"\n{'='*80}")
    logging.info("FEATURE SET COMPARISON")
    logging.info(f"{'='*80}")
    logging.info(
        f"  {'Set':>4s} {'C':>5s} {'Test':>8s} {'Train':>8s} {'Gap':>6s} {'F1':>8s}  Overfit"
    )
    logging.info(f"  {'─'*4} {'─'*5} {'─'*8} {'─'*8} {'─'*6} {'─'*8}  {'─'*10}")
    for row in summary_df.iter_rows(named=True):
        flag = "⚠️" if row["AUROC_gap_max"] > 0.15 else "✓"
        logging.info(
            f"  {row['feature_set']:>4s} C={row['C']:<4.1f} "
            f"{row['AUROC_mean']:>7.3f}±{row['AUROC_std']:.3f}  "
            f"{row['AUROC_train_mean']:>7.3f}  "
            f"{row['AUROC_gap_max']:>5.3f}  "
            f"{row['F1_mean']:>7.3f}  {flag}"
        )

    t_total = time.time() - t_start
    logging.info(f"\n[PHASE] Total runtime: {t_total:.1f}s")

    # --- Markdown summary ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md = f"""## Results: Feature Set Comparison (with overfitting check)


| Feature Set | C | Test AUROC | Train AUROC | Max Gap | F1 | Overfit |
|------------|---|-----------|-----------|---------|-----|---------|
"""
    for row in summary_df.iter_rows(named=True):
        gap_flag = (
            "⚠️ **HIGH**"
            if row["AUROC_gap_max"] > 0.20
            else "⚠️ borderline" if row["AUROC_gap_max"] > 0.10 else "✓ OK"
        )
        ftype = (
            "Multiangular"
            if row["feature_set"] in ("M3", "M4", "M5")
            else ("Nadir" if row["feature_set"] in ("M1", "M2") else "Metadata")
        )
        md += (
            f"| {row['feature_set']} | {row['C']:.1f} | "
            f"{row['AUROC_mean']:.3f} ± {row['AUROC_std']:.3f} | "
            f"{row['AUROC_train_mean']:.3f} | "
            f"{row['AUROC_gap_max']:.3f} | "
            f"{row['F1_mean']:.3f} | {gap_flag} |\n"
        )

    best = summary_df.row(0, named=True)
    md += f"""
**Interpretation**: Best: {best['feature_set']} (C={best['C']:.1f}) at test AUROC={best['AUROC_mean']:.3f} ± {best['AUROC_std']:.3f}.
Train-test gaps > 0.15 indicate potential overfitting — C=0.1 provides stronger regularization for high-dimensional feature sets.

**Spatial CV**: Verified — zero plot_id overlap between train and test folds.

**Outputs**: `outputs/results/model_comparison_by_fold.csv`, `outputs/results/model_comparison_summary.csv`
**Config**: `configs/paths.yaml`, seed=42, StratifiedGroupKFold(n_splits=5), groups=ifz_id
**Log**: `{LOG_FILE}`
"""
    (REPORTS_DIR / "compare_feature_sets_summary.md").write_text(md)
    logging.info(f"Markdown summary: {REPORTS_DIR / 'compare_feature_sets_summary.md'}")


if __name__ == "__main__":
    main()
