#!/usr/bin/env python3
"""Predict observed week-8 disease status from early-week reflectance features.

Predictors are limited to early weeks, so week-8 reflectance is never used as a
feature. The target must come from observed/non-null disease labels at the target
week; treatment assignment is never converted into an outcome label.
"""

import logging
import math
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
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.feature_selection import (
    assert_reflectance_only,
    reflectance_feature_columns,
)

warnings.filterwarnings("ignore", category=UserWarning)

PROJ = Path(__file__).resolve().parent.parent.parent
FEATURE_DIR = PROJ / "outputs" / "features"
QUARANTINE_DIR = PROJ / "outputs" / "quarantine_flawed_analysis"
RESULTS_DIR = QUARANTINE_DIR / "results"
REPORTS_DIR = QUARANTINE_DIR / "reports"
FIGURES_DIR = QUARANTINE_DIR / "figures"
LOGS_DIR = PROJ / "outputs" / "logs"

FEATURE_SETS = {
    "M1": "Nadir bands",
    "M2": "Nadir indices",
    "M3": "Multiangular VZA",
    "M4": "Multiangular VZA+RAA",
    "M5": "Angular contrast",
}
EARLY_WEEKS = [0, 3, 5]
TARGET_YEAR = 2024
TARGET_WEEK = 8
TARGET_COL = "future_disease_wk8"
EXCLUDE_COLS = {
    "plot_id",
    "week",
    "year",
    "cult",
    "trt",
    "disease_label",
    TARGET_COL,
}
SEED = 42
MAX_SPLITS = 5
N_PERMUTATIONS = 200

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"future_severity_prediction_{TIMESTAMP}.log"


def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )
    logging.info(f"Log file: {LOG_FILE}")


def log_phase(name, start_time):
    elapsed = time.time() - start_time
    logging.info(f"[PHASE] {name}: {elapsed:.1f}s")
    return elapsed


def feature_path(feature_set):
    matches = sorted(FEATURE_DIR.glob(f"{feature_set}_*.parquet"))
    if not matches:
        raise FileNotFoundError(f"No parquet found for {feature_set} in {FEATURE_DIR}")
    return matches[0]


def load_feature_sets():
    t0 = time.time()
    data = {}
    read_times = []
    for name in FEATURE_SETS:
        path = feature_path(name)
        t_read = time.time()
        df = pl.read_parquet(path)
        read_times.append(time.time() - t_read)
        if "year" in df.columns:
            df = df.filter(pl.col("year") == TARGET_YEAR)
        data[name] = df
        logging.info(f"  {name}: {df.height} rows from {path}")

    if read_times:
        logging.info(
            "[PHASE] parquet read summary: "
            f"min={np.min(read_times):.3f}s median={np.median(read_times):.3f}s "
            f"mean={np.mean(read_times):.3f}s max={np.max(read_times):.3f}s"
        )
    log_phase("data loading", t0)
    return data


def load_metadata():
    t0 = time.time()
    path = feature_path("M0")
    metadata = pl.read_parquet(path)
    if "year" in metadata.columns:
        metadata = metadata.filter(pl.col("year") == TARGET_YEAR)
    log_phase("metadata loading", t0)
    return metadata


def build_future_targets(metadata_df):
    """Build future week-8 targets from observed disease labels, never treatment."""
    target = metadata_df
    if "year" in target.columns:
        target = target.filter(pl.col("year") == TARGET_YEAR)
    target = target.filter(pl.col("week") == TARGET_WEEK)
    if "disease_label" not in target.columns:
        raise RuntimeError("Target-week metadata has no observed disease_label column")
    target = (
        target.select(["plot_id", "disease_label"])
        .drop_nulls("disease_label")
        .unique()
        .with_columns(pl.col("disease_label").cast(pl.Int64).alias(TARGET_COL))
        .select(["plot_id", TARGET_COL])
    )
    if target.is_empty():
        raise RuntimeError(
            f"No observed non-null disease_label values are available for year={TARGET_YEAR}, "
            f"target_week={TARGET_WEEK}; refusing to derive labels from treatment assignment."
        )
    if target[TARGET_COL].n_unique() < 2:
        raise RuntimeError(
            f"Observed target labels for year={TARGET_YEAR}, target_week={TARGET_WEEK} contain "
            "fewer than two classes; cannot evaluate prediction."
        )
    logging.info(
        f"  future targets: {target.height} plots, "
        f"positive_rate={target[TARGET_COL].mean():.2f}"
    )
    return target


def feature_columns(df):
    cols = reflectance_feature_columns(df.columns)
    assert_reflectance_only(cols, "future_severity_prediction")
    return cols


def build_feature_audit(aligned_by_week):
    rows = []
    for predictor_week, aligned in aligned_by_week.items():
        for feature_set, (_, cols) in aligned.items():
            for col in cols:
                rows.append(
                    {
                        "predictor_week": predictor_week,
                        "feature_set": feature_set,
                        "feature_type": FEATURE_SETS[feature_set],
                        "predictor": col,
                    }
                )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def prepare_week_data(data, targets, predictor_week):
    """Align all feature sets to the same plots for one early predictor week."""
    per_set = {}
    plot_sets = []
    for name, df in data.items():
        sub = df.filter(pl.col("week") == predictor_week).join(targets, on="plot_id", how="inner")
        cols = feature_columns(sub)
        if not cols:
            logging.warning(f"  {name} wk{predictor_week}: no features")
            continue
        meta_cols = [c for c in ["cult", "trt"] if c in sub.columns]
        sub = sub.select(["plot_id"] + meta_cols + [TARGET_COL] + cols).sort("plot_id")
        per_set[name] = (sub, cols)
        plot_sets.append(set(sub["plot_id"].to_list()))

    if not plot_sets:
        return {}, []

    common_plots = sorted(set.intersection(*plot_sets))
    aligned = {}
    for name, (sub, cols) in per_set.items():
        sub = sub.filter(pl.col("plot_id").is_in(common_plots)).sort("plot_id")
        aligned[name] = (sub, cols)

    logging.info(
        f"  wk{predictor_week}: {len(common_plots)} common plots across {len(aligned)} feature sets"
    )
    return aligned, common_plots


def build_splits(plot_ids, y):
    counts = np.bincount(y.astype(int), minlength=2)
    n_splits = min(MAX_SPLITS, len(plot_ids), int(counts.min()))
    if n_splits < 2:
        return []
    groups = np.array(plot_ids)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(splitter.split(np.zeros((len(y), 1)), y, groups=groups))


def build_pipeline():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED
                ),
            ),
        ]
    )


def evaluate_feature_set(name, df, cols, predictor_week, splits):
    X = df.select(cols).to_numpy()
    y = df[TARGET_COL].to_numpy()
    plot_ids = df["plot_id"].to_numpy()
    pipe = build_pipeline()
    rows = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            logging.info(f"    {name} wk{predictor_week} fold {fold}: single class, skip")
            continue

        t_fit = time.time()
        pipe.fit(X[train_idx], y_train)
        fit_time = time.time() - t_fit

        t_pred = time.time()
        y_pred = pipe.predict(X[test_idx])
        y_prob = pipe.predict_proba(X[test_idx])[:, 1]
        y_prob_train = pipe.predict_proba(X[train_idx])[:, 1]
        predict_time = time.time() - t_pred
        auroc_test = roc_auc_score(y_test, y_prob)
        auroc_train = roc_auc_score(y_train, y_prob_train)

        rows.append(
            {
                "predictor_week": predictor_week,
                "target_week": TARGET_WEEK,
                "feature_set": name,
                "feature_type": FEATURE_SETS[name],
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "test_plots": ";".join(plot_ids[test_idx].tolist()),
                "positive_rate_train": float(y_train.mean()),
                "positive_rate_test": float(y_test.mean()),
                "AUROC": auroc_test,
                "AUROC_train": auroc_train,
                "AUROC_gap": auroc_train - auroc_test,
                "AUPRC": average_precision_score(y_test, y_prob),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "fit_time_s": fit_time,
                "predict_time_s": predict_time,
            }
        )

    if rows:
        logging.info(
            f"    {name} wk{predictor_week}: AUROC="
            f"{np.mean([r['AUROC'] for r in rows]):.3f} +/- {np.std([r['AUROC'] for r in rows]):.3f}"
        )
    return rows


def collect_predictions(name, df, cols, predictor_week, splits):
    X = df.select(cols).to_numpy()
    y = df[TARGET_COL].to_numpy()
    plot_ids = df["plot_id"].to_numpy()
    cultivars = df["cult"].to_numpy() if "cult" in df.columns else np.array([None] * len(df))
    treatments = df["trt"].to_numpy() if "trt" in df.columns else np.array([None] * len(df))
    pipe = build_pipeline()
    rows = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        pipe.fit(X[train_idx], y_train)
        y_prob = pipe.predict_proba(X[test_idx])[:, 1]
        y_pred = pipe.predict(X[test_idx])
        for local_idx, prob, pred in zip(test_idx, y_prob, y_pred):
            rows.append(
                {
                    "predictor_week": predictor_week,
                    "feature_set": name,
                    "fold": fold,
                    "plot_id": plot_ids[local_idx],
                    "cult": cultivars[local_idx],
                    "trt": treatments[local_idx],
                    TARGET_COL: int(y[local_idx]),
                    "predicted_probability": float(prob),
                    "predicted_label": int(pred),
                }
            )
    return rows


def evaluate_cultivar_transfer(name, df, cols, predictor_week):
    if "cult" not in df.columns:
        return []

    X = df.select(cols).to_numpy()
    y = df[TARGET_COL].to_numpy()
    cultivars = df["cult"].to_numpy()
    rows = []
    for holdout in sorted(df["cult"].unique().to_list()):
        train_idx = np.where(cultivars != holdout)[0]
        test_idx = np.where(cultivars == holdout)[0]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        pipe = build_pipeline()
        t_fit = time.time()
        pipe.fit(X[train_idx], y_train)
        fit_time = time.time() - t_fit
        t_pred = time.time()
        y_prob = pipe.predict_proba(X[test_idx])[:, 1]
        y_pred = pipe.predict(X[test_idx])
        predict_time = time.time() - t_pred
        rows.append(
            {
                "predictor_week": predictor_week,
                "feature_set": name,
                "feature_type": FEATURE_SETS[name],
                "holdout_cultivar": holdout,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "positive_rate_train": float(y_train.mean()),
                "positive_rate_test": float(y_test.mean()),
                "AUROC": roc_auc_score(y_test, y_prob),
                "AUPRC": average_precision_score(y_test, y_prob),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "fit_time_s": fit_time,
                "predict_time_s": predict_time,
            }
        )
    return rows


def evaluate_permutation(name, df, cols, predictor_week, splits, n_permutations=N_PERMUTATIONS):
    X = df.select(cols).to_numpy()
    y_true = df[TARGET_COL].to_numpy()
    cultivars = df["cult"].to_numpy() if "cult" in df.columns else np.array(["all"] * len(df))

    true_scores = []
    pipe = build_pipeline()
    for train_idx, test_idx in splits:
        y_train, y_test = y_true[train_idx], y_true[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        pipe.fit(X[train_idx], y_train)
        true_scores.append(roc_auc_score(y_test, pipe.predict_proba(X[test_idx])[:, 1]))
    if not true_scores:
        return None

    rng = np.random.default_rng(SEED)
    null_scores = []
    for _ in range(n_permutations):
        y_perm = y_true.copy()
        for cult in np.unique(cultivars):
            idx = np.where(cultivars == cult)[0]
            shuffled = y_perm[idx].copy()
            rng.shuffle(shuffled)
            y_perm[idx] = shuffled

        fold_scores = []
        for train_idx, test_idx in splits:
            y_train, y_test = y_perm[train_idx], y_perm[test_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            pipe = build_pipeline()
            pipe.fit(X[train_idx], y_train)
            fold_scores.append(roc_auc_score(y_test, pipe.predict_proba(X[test_idx])[:, 1]))
        if fold_scores:
            null_scores.append(float(np.mean(fold_scores)))

    true_mean = float(np.mean(true_scores))
    p_value = float(np.mean(np.array(null_scores) >= true_mean)) if null_scores else np.nan
    return {
        "predictor_week": predictor_week,
        "feature_set": name,
        "feature_type": FEATURE_SETS[name],
        "true_AUROC": true_mean,
        "null_mean": float(np.mean(null_scores)) if null_scores else np.nan,
        "null_std": float(np.std(null_scores)) if null_scores else np.nan,
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


def summarize_results(fold_df):
    return (
        fold_df.group_by(["predictor_week", "feature_set", "feature_type"])
        .agg(
            n_folds=pl.len(),
            AUROC_mean=pl.col("AUROC").mean(),
            AUROC_std=pl.col("AUROC").std(),
            AUROC_train_mean=pl.col("AUROC_train").mean(),
            AUROC_gap_max=pl.col("AUROC_gap").max(),
            AUPRC_mean=pl.col("AUPRC").mean(),
            balanced_accuracy_mean=pl.col("balanced_accuracy").mean(),
            recall_mean=pl.col("recall").mean(),
            fit_time_s=pl.col("fit_time_s").sum(),
            predict_time_s=pl.col("predict_time_s").sum(),
        )
        .sort(["predictor_week", "feature_set"])
    )


def paired_deltas(fold_df):
    rows = []
    for week in sorted(fold_df["predictor_week"].unique().to_list()):
        week_df = fold_df.filter(pl.col("predictor_week") == week)
        for baseline in ["M1", "M2"]:
            base = week_df.filter(pl.col("feature_set") == baseline).select(
                ["fold", pl.col("AUROC").alias("baseline_AUROC")]
            )
            if base.is_empty():
                continue
            for comparator in ["M3", "M4", "M5"]:
                comp = week_df.filter(pl.col("feature_set") == comparator).select(
                    ["fold", pl.col("AUROC").alias("multiangular_AUROC")]
                )
                joined = base.join(comp, on="fold", how="inner")
                if joined.is_empty():
                    continue
                joined = joined.with_columns(
                    (pl.col("multiangular_AUROC") - pl.col("baseline_AUROC")).alias("delta_AUROC")
                )
                rows.append(
                    {
                        "predictor_week": week,
                        "baseline": baseline,
                        "comparator": comparator,
                        "n_paired_folds": joined.height,
                        "baseline_AUROC_mean": joined["baseline_AUROC"].mean(),
                        "multiangular_AUROC_mean": joined["multiangular_AUROC"].mean(),
                        "delta_AUROC_mean": joined["delta_AUROC"].mean(),
                        "delta_AUROC_std": joined["delta_AUROC"].std(),
                    }
                )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def fmt_metric(value, precision=3, signed=False):
    """Format optional numeric report values without crashing on sparse summaries."""
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(numeric):
        return "n/a"
    sign = "+" if signed else ""
    return f"{numeric:{sign}.{precision}f}"


def plot_summary(summary_df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    weeks = sorted(summary_df["predictor_week"].unique().to_list())
    x = np.arange(len(weeks))
    width = 0.14

    for i, feature_set in enumerate(FEATURE_SETS):
        sub = summary_df.filter(pl.col("feature_set") == feature_set).sort("predictor_week")
        if sub.is_empty():
            continue
        means = []
        positions = []
        for wk_pos, wk in enumerate(weeks):
            wk_rows = sub.filter(pl.col("predictor_week") == wk)
            if wk_rows.is_empty():
                continue
            means.append(wk_rows["AUROC_mean"].item())
            positions.append(wk_pos)
        if not means:
            continue
        ax.bar(np.array(positions) + (i - 2) * width, means, width, label=feature_set)

    ax.axhline(0.5, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"week {wk}" for wk in weeks])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title("Predict observed week-8 disease label from early-week features")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    out_path = FIGURES_DIR / "future_severity_auroc_by_week.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved figure: {out_path}")
    return out_path


def write_report(
    summary_df, delta_df, cultivar_df, permutation_df, feature_audit_df, outputs, total_time
):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "future_severity_prediction_summary.md"

    lines = [
        "## Results: Observed Week-8 Disease Prediction",
        "",
        "Predictors are early-week reflectance features from weeks 0, 3, and 5.",
        f"The target is `{TARGET_COL}`, copied from the observed/non-null `disease_label` at week 8.",
        "Week-8 reflectance is never used as a predictor.",
        "Only whitelisted reflectance-derived predictors are allowed in the model. Metadata, treatment, cultivar, plot/year/week identifiers, coordinates, elevation, raw viewing geometry, paths, and labels are excluded from the predictor matrix.",
        "",
        "**Important limitation**: if observed week-8 disease labels are aligned with treatment or spatial blocks, reflectance-only models can still learn treatment/spatial effects indirectly through reflectance. Treatment is never converted into the target label.",
        "",
        "### Model comparison",
        "",
        "| Predictor Week | Feature Set | Type | Folds | Test AUROC | Train AUROC | Max Gap | AUPRC | BalAcc | Recall |",
        "|----------------|-------------|------|-------|------------|-------------|---------|-------|--------|--------|",
    ]
    for row in summary_df.iter_rows(named=True):
        lines.append(
            f"| {row['predictor_week']} | {row['feature_set']} | {row['feature_type']} | "
            f"{row['n_folds']} | {fmt_metric(row['AUROC_mean'])} +/- {fmt_metric(row['AUROC_std'])} | "
            f"{fmt_metric(row['AUROC_train_mean'])} | {fmt_metric(row['AUROC_gap_max'])} | "
            f"{fmt_metric(row['AUPRC_mean'])} | {fmt_metric(row['balanced_accuracy_mean'])} | {fmt_metric(row['recall_mean'])} |"
        )

    lines.extend(
        [
            "",
            "### Paired AUROC deltas",
            "",
            "| Predictor Week | Baseline | Comparator | Folds | Baseline AUROC | Comparator AUROC | Delta AUROC |",
            "|----------------|----------|------------|-------|----------------|------------------|-------------|",
        ]
    )
    if delta_df.is_empty():
        lines.append("| n/a | n/a | n/a | 0 | n/a | n/a | n/a |")
    else:
        for row in delta_df.iter_rows(named=True):
            lines.append(
                f"| {row['predictor_week']} | {row['baseline']} | {row['comparator']} | "
                f"{row['n_paired_folds']} | {fmt_metric(row['baseline_AUROC_mean'])} | "
                f"{fmt_metric(row['multiangular_AUROC_mean'])} | {fmt_metric(row['delta_AUROC_mean'], signed=True)} |"
            )

    lines.extend(
        [
            "",
            "### Leave-one-cultivar-out transfer",
            "",
            "| Predictor Week | Feature Set | Holdout Cultivar | AUROC | AUPRC | BalAcc | Recall |",
            "|----------------|-------------|------------------|-------|-------|--------|--------|",
        ]
    )
    if cultivar_df.is_empty():
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    else:
        for row in cultivar_df.iter_rows(named=True):
            lines.append(
                f"| {row['predictor_week']} | {row['feature_set']} | {row['holdout_cultivar']} | "
                f"{fmt_metric(row['AUROC'])} | {fmt_metric(row['AUPRC'])} | "
                f"{fmt_metric(row['balanced_accuracy'])} | {fmt_metric(row['recall'])} |"
            )

    lines.extend(
        [
            "",
            "### Label permutation sanity check",
            "",
            "Labels are shuffled within cultivar groups while preserving the same CV folds.",
            "",
            "| Predictor Week | Feature Set | True AUROC | Null Mean | Null Std | p-value | Permutations |",
            "|----------------|-------------|------------|-----------|----------|---------|--------------|",
        ]
    )
    if permutation_df.is_empty():
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    else:
        for row in permutation_df.iter_rows(named=True):
            lines.append(
                f"| {row['predictor_week']} | {row['feature_set']} | {fmt_metric(row['true_AUROC'])} | "
                f"{fmt_metric(row['null_mean'])} | {fmt_metric(row['null_std'])} | {fmt_metric(row['p_value'])} | "
                f"{row['n_permutations']} |"
            )

    lines.extend(
        [
            "",
            "### Predictor audit",
            "",
            "| Predictor Week | Feature Set | Predictors Used |",
            "|----------------|-------------|-----------------|",
        ]
    )
    if feature_audit_df.is_empty():
        lines.append("| n/a | n/a | 0 |")
    else:
        audit_summary = (
            feature_audit_df.group_by(["predictor_week", "feature_set"])
            .agg(pl.col("predictor").n_unique().alias("n_predictors"))
            .sort(["predictor_week", "feature_set"])
        )
        for row in audit_summary.iter_rows(named=True):
            lines.append(
                f"| {row['predictor_week']} | {row['feature_set']} | {row['n_predictors']} |"
            )

    best_deltas = (
        delta_df.filter(pl.col("baseline") == "M2") if not delta_df.is_empty() else pl.DataFrame()
    )
    interpretation = "No paired multiangular-vs-nadir-index deltas were available."
    if not best_deltas.is_empty():
        best = best_deltas.sort("delta_AUROC_mean", descending=True).row(0, named=True)
        interpretation = (
            f"The strongest paired gain over the nadir-index baseline is {best['comparator']} at "
            f"week {best['predictor_week']} (delta AUROC {fmt_metric(best['delta_AUROC_mean'], signed=True)}). "
            "This supports the presence of early angular signal, with the treatment-aligned target caveat above."
        )

    lines.extend(["", f"**Interpretation**: {interpretation}", "", "## Outputs", ""])
    for label, path in outputs.items():
        lines.append(f"- {label}: `{path}`")

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Year: {TARGET_YEAR}",
            f"- Predictor weeks: {EARLY_WEEKS}",
            f"- Target week: {TARGET_WEEK}",
            f"- Target source: observed `disease_label` at week {TARGET_WEEK}; no labels are derived from treatment assignment",
            f"- Target column: `{TARGET_COL}`",
            f"- Seed: {SEED}",
            "- CV: StratifiedGroupKFold by plot_id, identical folds for all feature sets within each predictor week",
            "- Model: LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)",
            "- Preprocessing: SimpleImputer(median) + StandardScaler fit inside each fold",
            f"- Log: `{LOG_FILE}`",
            f"- Total runtime: {total_time:.1f}s",
        ]
    )
    with report_path.open("w") as fh:
        fh.write("\n".join(lines) + "\n")
    logging.info(f"Saved report: {report_path}")
    return report_path


def run_analysis():
    setup_logging()
    total_start = time.time()

    data = load_feature_sets()
    metadata = load_metadata()
    targets = build_future_targets(metadata)

    all_rows = []
    all_predictions = []
    all_cultivar_rows = []
    all_permutation_rows = []
    aligned_by_week = {}
    for week in EARLY_WEEKS:
        t_week = time.time()
        logging.info(
            f"\n=== Predictor week {week} -> week {TARGET_WEEK} observed disease label ==="
        )
        aligned, common_plots = prepare_week_data(data, targets, week)
        if not aligned:
            logging.warning(f"  wk{week}: no aligned data")
            continue
        aligned_by_week[week] = aligned
        reference_df = next(iter(aligned.values()))[0]
        y = reference_df[TARGET_COL].to_numpy()
        splits = build_splits(common_plots, y)
        if not splits:
            logging.warning(f"  wk{week}: not enough class balance for CV")
            continue
        logging.info(f"  wk{week}: {len(splits)} paired folds")

        for name, (df, cols) in aligned.items():
            all_rows.extend(evaluate_feature_set(name, df, cols, week, splits))
            all_predictions.extend(collect_predictions(name, df, cols, week, splits))
            all_cultivar_rows.extend(evaluate_cultivar_transfer(name, df, cols, week))
            perm_row = evaluate_permutation(name, df, cols, week, splits)
            if perm_row is not None:
                all_permutation_rows.append(perm_row)
        log_phase(f"week {week} feature evaluation", t_week)

    if not all_rows:
        raise RuntimeError("No observed future disease prediction folds were produced")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_df = pl.DataFrame(all_rows)
    fold_path = RESULTS_DIR / "future_severity_by_fold.csv"
    fold_df.write_csv(fold_path)

    summary_df = summarize_results(fold_df)
    summary_path = RESULTS_DIR / "future_severity_by_week.csv"
    summary_df.write_csv(summary_path)

    delta_df = paired_deltas(fold_df)
    delta_path = RESULTS_DIR / "future_severity_paired_delta.csv"
    delta_df.write_csv(delta_path)

    prediction_df = pl.DataFrame(all_predictions) if all_predictions else pl.DataFrame()
    prediction_path = RESULTS_DIR / "future_severity_predictions.csv"
    prediction_df.write_csv(prediction_path)

    cultivar_df = pl.DataFrame(all_cultivar_rows) if all_cultivar_rows else pl.DataFrame()
    cultivar_path = RESULTS_DIR / "future_severity_leave_one_cultivar.csv"
    cultivar_df.write_csv(cultivar_path)

    permutation_df = pl.DataFrame(all_permutation_rows) if all_permutation_rows else pl.DataFrame()
    permutation_path = RESULTS_DIR / "future_severity_permutation.csv"
    permutation_df.write_csv(permutation_path)

    feature_audit_df = build_feature_audit(aligned_by_week)
    feature_audit_path = RESULTS_DIR / "future_severity_predictor_audit.csv"
    feature_audit_df.write_csv(feature_audit_path)

    figure_path = plot_summary(summary_df)
    total_time = time.time() - total_start
    report_path = write_report(
        summary_df,
        delta_df,
        cultivar_df,
        permutation_df,
        feature_audit_df,
        {
            "Fold results": fold_path,
            "Summary table": summary_path,
            "Paired deltas": delta_path,
            "Per-plot predictions": prediction_path,
            "Leave-one-cultivar transfer": cultivar_path,
            "Permutation test": permutation_path,
            "Predictor audit": feature_audit_path,
            "Figure": figure_path,
        },
        total_time,
    )
    logging.info(f"[PHASE] total: {total_time:.1f}s")
    return fold_df, summary_df, delta_df, report_path


def main():
    run_analysis()


if __name__ == "__main__":
    main()
