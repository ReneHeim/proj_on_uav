#!/usr/bin/env python3
"""Permutation test: is multiangular AUROC higher than chance within structural constraints?

Shuffles disease labels within (cultivar, week) groups and re-runs grouped CV
to build a null distribution of AUROC scores.

Compares M5 (best multiangular) and M1 (nadir baseline) against their nulls.
"""

import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.feature_selection import (
    assert_reflectance_only,
    reflectance_feature_columns,
)

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURE_DIR = PROJECT_ROOT / "outputs" / "features"
QUARANTINE_DIR = PROJECT_ROOT / "outputs" / "quarantine_flawed_analysis"
RESULTS_DIR = QUARANTINE_DIR / "results"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
REPORTS_DIR = QUARANTINE_DIR / "reports"
FIGURES_DIR = QUARANTINE_DIR / "figures"

FEATURE_SETS = ["M1", "M5"]
N_PERMUTATIONS = 100
N_SPLITS = 5
SEED = 42

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"permutation_signal_test_{ts}.log"


def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )
    logging.info(f"Log file: {LOG_FILE}")


def load_feature_set(name):
    pattern = list(FEATURE_DIR.glob(f"{name}_*.parquet"))
    if not pattern:
        raise FileNotFoundError(f"No parquet found for {name} in {FEATURE_DIR}")
    df = pl.read_parquet(pattern[0])
    df = df.filter(pl.col("disease_label").is_not_null())
    return df


def shuffle_labels_within_groups(df, rng):
    """Shuffle disease_label within (cult, week) groups. Returns new label array."""
    shuffled = np.zeros(len(df), dtype=np.int64)
    for (cult_val, week_val), group_df in df.group_by(["cult", "week"]):
        idx = group_df.get_column("_row_idx").to_numpy()
        labels = group_df["disease_label"].to_numpy().copy()
        rng.shuffle(labels)
        shuffled[idx] = labels
    return shuffled


def compute_auroc_for_splits(X, y, splits, pipe):
    """Compute AUROC across pre-defined CV splits."""
    scores = []
    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        scores.append(roc_auc_score(y_test, y_proba))
    return np.mean(scores)


def evaluate_permutation_test(name):
    t0 = time.time()
    logging.info(f"\n{'='*60}")
    logging.info(f"  Permutation test: {name}")
    logging.info(f"{'='*60}")

    df = load_feature_set(name)
    t_load = time.time()
    logging.info(f"  [PHASE] data loading: {t_load - t0:.1f}s")

    feature_cols = reflectance_feature_columns(df.columns)
    assert_reflectance_only(feature_cols, f"permutation_signal_test:{name}")
    logging.info(f"  Features: {len(feature_cols)}, samples: {df.shape[0]}")

    X = df.select(feature_cols).to_numpy()
    y_true = df["disease_label"].to_numpy()

    df = df.with_columns(
        pl.Series("_row_idx", np.arange(len(df), dtype=np.int64)),
        (pl.col("plot_id").str.extract(r"(\d+)").cast(pl.Int64) + 90001).alias("ifz_id"),
    )
    groups = df["ifz_id"].to_numpy()

    nan_frac = np.isnan(X).mean()
    logging.info(f"  NaN fraction in features: {nan_frac:.4f}")

    # --- Build pipeline ---
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    penalty="l2",
                    max_iter=2000,
                    random_state=SEED,
                ),
            ),
        ]
    )

    # --- Fixed CV splits (GroupKFold, label-independent) ---
    gkf = GroupKFold(n_splits=N_SPLITS)
    splits = list(gkf.split(X, y_true, groups=groups))
    logging.info(f"  CV splits: {len(splits)} folds (GroupKFold by ifz_id)")

    # Verify spatial CV
    for fold, (train_idx, test_idx) in enumerate(splits):
        train_grp = set(groups[train_idx])
        test_grp = set(groups[test_idx])
        overlap = train_grp & test_grp
        if overlap:
            logging.error(f"  LEAKAGE Fold {fold}: overlap={len(overlap)}")
        logging.info(
            f"  Fold {fold}: train={len(train_idx)} test={len(test_idx)} overlap={len(overlap)}"
        )

    # --- True AUROC ---
    t_true = time.time()
    true_auroc = compute_auroc_for_splits(X, y_true, splits, pipe)
    logging.info(f"  [PHASE] true AUROC: {true_auroc:.4f} ({time.time() - t_true:.1f}s)")

    # --- Permutations ---
    t_perm = time.time()
    rng = np.random.default_rng(SEED)
    null_scores = []
    for i in range(N_PERMUTATIONS):
        y_shuffled = shuffle_labels_within_groups(df, rng)
        score = compute_auroc_for_splits(X, y_shuffled, splits, pipe)
        null_scores.append(score)
        if (i + 1) % 20 == 0:
            logging.info(
                f"  Permutation {i+1}/{N_PERMUTATIONS}: mean null AUROC={np.mean(null_scores):.4f}"
            )

    null_scores = np.array(null_scores)
    logging.info(f"  [PHASE] {N_PERMUTATIONS} permutations: {time.time() - t_perm:.1f}s")

    p_value = np.mean(null_scores >= true_auroc)
    perc_95 = np.percentile(null_scores, 95)
    logging.info(f"  True AUROC: {true_auroc:.4f}")
    logging.info(f"  Null mean: {null_scores.mean():.4f} ± {null_scores.std():.4f}")
    logging.info(f"  Null 95th percentile: {perc_95:.4f}")
    logging.info(
        f"  p-value: {p_value:.4f} ({'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'})"
    )
    logging.info(f"  [PHASE] Total {name}: {time.time() - t0:.1f}s")

    return {
        "name": name,
        "true_auroc": true_auroc,
        "null_scores": null_scores,
        "p_value": p_value,
        "perc_95": perc_95,
        "n_samples": len(y_true),
    }, df


def plot_results(results, df_info):
    t0 = time.time()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, res in zip(axes, results):
        null = res["null_scores"]
        true = res["true_auroc"]

        ax.hist(
            null,
            bins=25,
            alpha=0.7,
            color="steelblue",
            edgecolor="white",
            label="Null distribution",
        )
        ax.axvline(
            true, color="crimson", linewidth=2, linestyle="--", label=f"True AUROC={true:.3f}"
        )
        ax.axvline(
            res["perc_95"],
            color="gray",
            linewidth=1.5,
            linestyle=":",
            label=f"95th pct={res['perc_95']:.3f}",
        )

        ax.set_title(f"{res['name']}  p={res['p_value']:.3f}")
        ax.set_xlabel("AUROC")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Permutation Test: Multiangular vs Nadir\n"
        f"{N_PERMUTATIONS} shuffles within (cultivar, week), GroupKFold by ifz_id",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = FIGURES_DIR / "permutation_test_multiangular_vs_nadir.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  [PHASE] plot: {time.time() - t0:.1f}s -> {out_path}")


def save_results(results):
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for res in results:
        rows.append(
            {
                "feature_set": res["name"],
                "iteration": -1,
                "AUROC": res["true_auroc"],
                "is_true": 1,
            }
        )
        for i, score in enumerate(res["null_scores"]):
            rows.append(
                {
                    "feature_set": res["name"],
                    "iteration": i,
                    "AUROC": score,
                    "is_true": 0,
                }
            )

    out_df = pl.DataFrame(rows)
    csv_path = RESULTS_DIR / "permutation_test_scores.csv"
    out_df.write_csv(csv_path)
    logging.info(f"  [PHASE] save CSV: {time.time() - t0:.1f}s -> {csv_path}")


def write_summary(results, total_time):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = ["## Results: Label Permutation Test", ""]
    lines.append(
        "| Feature Set | True AUROC | Null Mean | Null Std | 95th Pct | p-value | Significant |"
    )
    lines.append(
        "|------------|-----------|-----------|---------|---------|---------|-------------|"
    )

    for res in results:
        sig = "Yes" if res["p_value"] < 0.05 else "No"
        lines.append(
            f"| {res['name']} | {res['true_auroc']:.4f} | {res['null_scores'].mean():.4f} | "
            f"{res['null_scores'].std():.4f} | {res['perc_95']:.4f} | "
            f"{res['p_value']:.4f} | {sig} |"
        )

    delta = results[1]["true_auroc"] - results[0]["true_auroc"] if len(results) >= 2 else 0
    lines.append("")
    lines.append(
        f"**Interpretation**: True AUROC values are compared against a null distribution "
        f"generated by shuffling disease labels within (cultivar, week) groups "
        f"{N_PERMUTATIONS} times. The multiangular feature set (M5) is evaluated "
        f"against the nadir baseline (M1). "
        f"A p-value < 0.05 indicates the model captures signal beyond chance-level "
        f"structure preserved within cultivar and week constraints."
    )
    lines.append("")
    lines.append(
        f"**Outputs**: `outputs/results/permutation_test_scores.csv`, "
        f"`outputs/figures/permutation_test_multiangular_vs_nadir.png`"
    )
    lines.append(
        f"**Config**: CV=GroupKFold(n_splits={N_SPLITS}) by ifz_id, seed={SEED}, "
        f"permutations={N_PERMUTATIONS}, shuffle within (cultivar, week)"
    )
    lines.append(f"**Log**: `{LOG_FILE}`")
    lines.append(f"**Total runtime**: {total_time:.1f}s")

    md_path = REPORTS_DIR / "permutation_signal_test_summary.md"
    md_path.write_text("\n".join(lines) + "\n")
    logging.info(f"  Markdown summary: {md_path}")


def main():
    setup_logging()
    t_start = time.time()

    all_results = []
    df_info = {}

    for fs_name in FEATURE_SETS:
        try:
            res, df = evaluate_permutation_test(fs_name)
            all_results.append(res)
            df_info[fs_name] = df
        except FileNotFoundError as e:
            logging.warning(f"SKIP {fs_name}: {e}")
        except Exception as e:
            logging.error(f"ERROR {fs_name}: {e}", exc_info=True)

    if not all_results:
        logging.error("No results generated.")
        sys.exit(1)

    save_results(all_results)
    plot_results(all_results, df_info)
    t_total = time.time() - t_start
    logging.info(f"\n[PHASE] Total runtime: {t_total:.1f}s")
    write_summary(all_results, t_total)


if __name__ == "__main__":
    main()
