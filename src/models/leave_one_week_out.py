#!/usr/bin/env python3
"""Leave-one-week-out cross-validation to test temporal generalization.
Train on all weeks except one, test on the held-out week.
"""
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

FEATURE_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "features"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "results"
LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "logs"

EXCLUDE = ["plot_id", "week", "year", "cult", "trt", "disease_label"]

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / f"leave_one_week_out_{ts}.log"),
        logging.StreamHandler(),
    ],
)


def load_features():
    """Load M1, M3, M5; return dict by name."""
    data = {}
    for name in ["M1_nadir_bands", "M3_multiangular_vza", "M5_angular_contrast"]:
        df = pl.read_parquet(FEATURE_DIR / f"{name}.parquet")
        df = df.filter(pl.col("disease_label").is_not_null())
        data[name] = df
        logging.info(
            f"  {name}: {df.height} rows, {len([c for c in df.columns if c not in EXCLUDE])} features"
        )
    return data


def run_lowo(data, year):
    """Leave-one-week-out CV for a specific year."""
    results = []
    for name, df in data.items():
        df_yr = df.filter(pl.col("year") == year)
        weeks = sorted(df_yr["week"].unique().to_list())
        if len(weeks) < 2:
            continue

        feature_cols = [c for c in df_yr.columns if c not in EXCLUDE]
        X = df_yr.select(feature_cols).to_numpy()
        y = df_yr["disease_label"].to_numpy()
        weeks_arr = df_yr["week"].to_numpy()

        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=1.0, class_weight="balanced", max_iter=2000, random_state=42
                    ),
                ),
            ]
        )

        name_results = []
        for test_week in weeks:
            train_mask = weeks_arr != test_week
            test_mask = weeks_arr == test_week

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            if len(np.unique(y_train)) < 2:
                logging.info(f"  {name} {year} wk{test_week}: single class in train - skip")
                continue
            if len(np.unique(y_test)) < 2:
                logging.info(f"  {name} {year} wk{test_week}: single class in test - skip")
                continue

            t0 = time.time()
            pipe.fit(X_train, y_train)
            y_proba = pipe.predict_proba(X_test)[:, 1]
            y_pred = pipe.predict(X_test)
            y_proba_train = pipe.predict_proba(X_train)[:, 1]
            elapsed = time.time() - t0

            auroc_test = roc_auc_score(y_test, y_proba)
            auroc_train = roc_auc_score(y_train, y_proba_train)
            f1 = f1_score(y_test, y_pred)

            row = {
                "feature_set": name,
                "year": year,
                "test_week": test_week,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "pos_train": y_train.mean(),
                "pos_test": y_test.mean(),
                "AUROC": auroc_test,
                "AUROC_train": auroc_train,
                "AUROC_gap": auroc_train - auroc_test,
                "F1": f1,
                "time": elapsed,
            }
            results.append(row)
            name_results.append(row)

            logging.info(
                f"  {name} {year} train→wk{test_week}: "
                f"test={auroc_test:.3f} train={auroc_train:.3f} "
                f"f1={f1:.3f} n_test={len(y_test)}"
            )

        if name_results:
            res_df = pl.DataFrame(name_results)
            mean_auroc = res_df["AUROC"].mean()
            mean_std = res_df["AUROC"].std()
            # Δ vs M1
            logging.info(f"  {name} {year} LOWO mean: AUROC={mean_auroc:.3f} ± {mean_std:.3f}")

    return pl.DataFrame(results) if results else None


def main():
    logging.info("=== Leave-One-Week-Out Validation ===")
    data = load_features()

    all_results = []
    for yr in [2024, 2025]:
        logging.info(f"\n--- {yr} ---")
        r = run_lowo(data, yr)
        if r is not None:
            all_results.append(r)

    if not all_results:
        logging.error("No results")
        return

    final = pl.concat(all_results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    final.write_csv(RESULTS_DIR / "leave_one_week_out.csv")

    # Cross-year summary table
    logging.info(f"\n{'='*70}")
    logging.info("CROSS-YEAR LEAVE-ONE-WEEK-OUT SUMMARY")
    logging.info(f"{'='*70}")
    logging.info(
        f"  {'Set':>4s} {'Year':>4s}  {'Test AUROC':>10s}  {'Train':>7s}  {'Gap':>6s}  {'F1':>7s}"
    )
    logging.info(f"  {'─'*4} {'─'*4}  {'─'*10}  {'─'*7}  {'─'*6}  {'─'*7}")

    for yr in [2024, 2025]:
        for fs in ["M1_nadir_bands", "M3_multiangular_vza", "M5_angular_contrast"]:
            subset = final.filter((pl.col("year") == yr) & (pl.col("feature_set") == fs))
            if subset.height == 0:
                continue
            m_test = subset["AUROC"].mean()
            m_train = subset["AUROC_train"].mean()
            m_f1 = subset["F1"].mean()
            m_gap = (subset["AUROC_train"] - subset["AUROC"]).max()
            logging.info(
                f"  {fs[:5]:>4s} {yr:>4s}  {m_test:>9.3f} ± {subset['AUROC'].std():.4f}  {m_train:>6.3f}  {m_gap:>5.3f}  {m_f1:>6.3f}"
            )

    # ΔAUROC M3-M1 per year
    logging.info(f"\n=== ΔAUROC (M3 − M1) ===")
    for yr in [2024, 2025]:
        m3 = final.filter((pl.col("year") == yr) & (pl.col("feature_set") == "M3_multiangular_vza"))
        m1 = final.filter((pl.col("year") == yr) & (pl.col("feature_set") == "M1_nadir_bands"))
        if m3.height > 0 and m1.height > 0:
            d = m3["AUROC"].mean() - m1["AUROC"].mean()
            logging.info(
                f"  {yr}: M3={m3['AUROC'].mean():.3f}, M1={m1['AUROC'].mean():.3f}, Δ=+{d:.3f}"
            )


if __name__ == "__main__":
    main()
