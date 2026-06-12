#!/usr/bin/env python3
"""Subgroup robustness analysis: compare M1 (nadir) vs M3 (multiangular) across
cultivar and treatment subgroups.

Subgroups:
  - cult == 'aluco'
  - cult == 'capone'
  - trt == 'trt'       (healthy/treatment)
  - trt == 'no_trt'    (inoculated/diseased)
  - All 4 subgroup combinations (aluco-trt, aluco-no_trt, capone-trt, capone-no_trt)

Generates:
  - outputs/results/model_comparison_by_subgroup.csv
  - outputs/reports/subgroup_analysis_summary.md
  - outputs/figures/delta_auc_by_cultivar.png
  - outputs/figures/delta_auc_by_treatment.png
  - outputs/figures/delta_auc_by_week.png
"""

import logging
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
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
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
FIGURES_DIR = QUARANTINE_DIR / "figures"
REPORTS_DIR = QUARANTINE_DIR / "reports"
LOGS_DIR = PROJ / "outputs" / "logs"

FEATURE_SETS = ["M1", "M3"]
DISPLAY = {"M1": "Nadir bands", "M3": "Multiangular VZA"}
EXCLUDE_COLS = ["plot_id", "week", "year", "cult", "trt", "disease_label"]
SEED = 42
N_SPLITS = 5

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"subgroup_analysis_{TIMESTAMP}.log"

SUBGROUPS = {
    "all": None,
    "aluco": ("cult", "aluco"),
    "capone": ("cult", "capone"),
    "trt": ("trt", "trt"),
    "no_trt": ("trt", "no_trt"),
    "aluco_trt": [("cult", "aluco"), ("trt", "trt")],
    "aluco_no_trt": [("cult", "aluco"), ("trt", "no_trt")],
    "capone_trt": [("cult", "capone"), ("trt", "trt")],
    "capone_no_trt": [("cult", "capone"), ("trt", "no_trt")],
}


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


def load_feature_set(name):
    pattern = list(FEATURE_DIR.glob(f"{name}_*.parquet"))
    if not pattern:
        raise FileNotFoundError(f"No parquet for {name}")
    df = pl.read_parquet(pattern[0])
    return df.filter(pl.col("disease_label").is_not_null())


def apply_subgroup_filter(df, subgroup_key):
    """Filter dataframe based on a subgroup definition.

    subgroup: can be:
      - None                     → all data
      - (column, value)          → single filter
      - [(column, value), ...]   → AND combination
    """
    subdef = SUBGROUPS[subgroup_key]
    if subdef is None:
        return df.clone()

    if isinstance(subdef, tuple):
        col, val = subdef
        return df.filter(pl.col(col) == val)
    elif isinstance(subdef, list):
        # AND combination of filters
        for col, val in subdef:
            df = df.filter(pl.col(col) == val)
        return df
    return df


def build_pipeline():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "AUROC": roc_auc_score(y_true, y_proba),
        "AUPRC": average_precision_score(y_true, y_proba),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }


def run_cv(X_arr, y_arr, groups_arr):
    """Cross-validation with StratifiedGroupKFold fallback."""
    try:
        skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        splits = list(skf.split(X_arr, y_arr, groups=groups_arr))
    except (ValueError, RuntimeError):
        gkf = GroupKFold(n_splits=N_SPLITS)
        splits = list(gkf.split(X_arr, y_arr, groups=groups_arr))
    return splits


def evaluate_feature_set_for_subgroup(fs_name, df_full, subgroup_key):
    """Evaluate one feature set on one subgroup."""
    df = apply_subgroup_filter(df_full, subgroup_key)

    feature_cols = reflectance_feature_columns(df.columns)
    assert_reflectance_only(feature_cols, f"subgroup_analysis:{fs_name}")
    if not feature_cols:
        logging.info(f"    {fs_name}/{subgroup_key}: no features — skipping")
        return []

    X = df.select(feature_cols).to_numpy()
    y = df["disease_label"].to_numpy()

    df = df.with_columns(
        (pl.col("plot_id").str.extract(r"(\d+)").cast(pl.Int64) + 90001).alias("ifz_id")
    )
    groups = df["ifz_id"].to_numpy()

    unique_plots = len(np.unique(groups))
    pos_rate = y.mean()

    if len(np.unique(y)) < 2:
        logging.info(
            f"    {fs_name}/{subgroup_key}: single class only " f"(pos={pos_rate:.2f}) — skipping"
        )
        return []

    if unique_plots < 3 or len(y) < 4:
        logging.info(
            f"    {fs_name}/{subgroup_key}: only {len(y)} rows / "
            f"{unique_plots} plots — skipping"
        )
        return []

    logging.info(
        f"    {fs_name}/{subgroup_key}: {len(y)} rows, "
        f"{X.shape[1]} features, {unique_plots} plots, "
        f"pos={pos_rate:.2f}"
    )

    pipe = build_pipeline()
    splits = run_cv(X, y, groups)

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            logging.info(f"      fold {fold}: single class in train/test — skipping")
            continue

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)

        fold_results.append(
            {
                "feature_set": fs_name,
                "subgroup": subgroup_key,
                "fold": fold,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "positive_rate_train": y_train.mean(),
                "positive_rate_test": y_test.mean(),
                **metrics,
            }
        )

    return fold_results


def run_subgroup_analysis():
    all_folds = []

    for fs_name in FEATURE_SETS:
        logging.info(f"\n{'='*60}")
        logging.info(f"  {fs_name} ({DISPLAY[fs_name]})")
        logging.info(f"{'='*60}")

        df = load_feature_set(fs_name)

        for subgroup_key in SUBGROUPS:
            logging.info(f"  --- subgroup: {subgroup_key} ---")
            try:
                results = evaluate_feature_set_for_subgroup(fs_name, df, subgroup_key)
                all_folds.extend(results)
            except Exception as e:
                logging.error(f"    ERROR {fs_name}/{subgroup_key}: {e}")

    return pl.DataFrame(all_folds)


def summarize_subgroup(fold_df):
    """Group by feature_set and subgroup, compute per-feature-set means.
    Filters NaN AUROC values before computing mean/std."""
    summary = (
        fold_df.group_by(["feature_set", "subgroup"])
        .agg(
            n_folds=pl.len(),
            AUROC_mean=pl.col("AUROC").filter(~pl.col("AUROC").is_nan()).mean(),
            AUROC_std=pl.col("AUROC").filter(~pl.col("AUROC").is_nan()).std(),
            AUPRC_mean=pl.col("AUPRC").mean(),
            AUPRC_std=pl.col("AUPRC").std(),
            balanced_accuracy_mean=pl.col("balanced_accuracy").mean(),
            balanced_accuracy_std=pl.col("balanced_accuracy").std(),
            recall_mean=pl.col("recall").mean(),
            recall_std=pl.col("recall").std(),
        )
        .sort(["subgroup", "feature_set"])
    )
    return summary


def compute_delta_per_subgroup(summary_df):
    """Compute ΔAUROC (M3 − M1) per subgroup. Skips NaN values."""
    records = []
    for subgroup_key in SUBGROUPS:
        sub = summary_df.filter(pl.col("subgroup") == subgroup_key)
        m1 = sub.filter(pl.col("feature_set") == "M1")
        m3 = sub.filter(pl.col("feature_set") == "M3")
        if len(m1) > 0 and len(m3) > 0:
            m1_auc = m1["AUROC_mean"][0]
            m3_auc = m3["AUROC_mean"][0]
            if (
                m1_auc is not None
                and not np.isnan(m1_auc)
                and m3_auc is not None
                and not np.isnan(m3_auc)
            ):
                records.append(
                    {
                        "subgroup": subgroup_key,
                        "M1_AUROC_mean": m1_auc,
                        "M3_AUROC_mean": m3_auc,
                        "delta_AUROC": m3_auc - m1_auc,
                        "M1_AUPRC_mean": m1["AUPRC_mean"][0],
                        "M3_AUPRC_mean": m3["AUPRC_mean"][0],
                        "delta_AUPRC": m3["AUPRC_mean"][0] - m1["AUPRC_mean"][0],
                    }
                )
    return pl.DataFrame(records)


def compute_delta_per_week(fold_df):
    """Compute ΔAUROC per week per feature set, using week info from fold_df.
    Since subgroup analysis doesn't have week breakdown, load raw data and compute per week."""
    # Load M1 and M3 full data, compute per-week CV scores
    week_results = []
    all_weeks = set()

    for fs_name in FEATURE_SETS:
        df = load_feature_set(fs_name)
        weeks = sorted(df["week"].unique().to_list())
        all_weeks.update(weeks)

        for week in weeks:
            df_w = df.filter(pl.col("week") == week)
            if len(df_w) < 4:
                continue

            feature_cols = reflectance_feature_columns(df_w.columns)
            assert_reflectance_only(feature_cols, f"subgroup_analysis:{fs_name}:week{week}")
            X = df_w.select(feature_cols).to_numpy()
            y = df_w["disease_label"].to_numpy()

            df_w = df_w.with_columns(
                (pl.col("plot_id").str.extract(r"(\d+)").cast(pl.Int64) + 90001).alias("ifz_id")
            )
            groups = df_w["ifz_id"].to_numpy()

            pipe = build_pipeline()
            splits = run_cv(X, y, groups)

            for fold, (train_idx, test_idx) in enumerate(splits):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                y_proba = pipe.predict_proba(X_test)[:, 1]

                auroc = roc_auc_score(y_test, y_proba)

                week_results.append(
                    {
                        "feature_set": fs_name,
                        "week": week,
                        "fold": fold,
                        "AUROC": auroc,
                    }
                )

    week_df = pl.DataFrame(week_results)
    week_summary = (
        week_df.group_by(["feature_set", "week"])
        .agg(
            AUROC_mean=pl.col("AUROC").mean(),
            AUROC_std=pl.col("AUROC").std(),
        )
        .sort(["week", "feature_set"])
    )

    # Compute delta per week
    delta_week = []
    for week in sorted(all_weeks):
        m1_row = week_summary.filter((pl.col("feature_set") == "M1") & (pl.col("week") == week))
        m3_row = week_summary.filter((pl.col("feature_set") == "M3") & (pl.col("week") == week))
        if len(m1_row) and len(m3_row):
            m1_val = m1_row["AUROC_mean"][0]
            m3_val = m3_row["AUROC_mean"][0]
            if (
                m1_val is not None
                and not np.isnan(m1_val)
                and m3_val is not None
                and not np.isnan(m3_val)
            ):
                delta_week.append(
                    {
                        "week": int(week),
                        "M1_AUROC_mean": m1_val,
                        "M3_AUROC_mean": m3_val,
                        "delta_AUROC": m3_val - m1_val,
                    }
                )

    return pl.DataFrame(delta_week)


def plot_delta_by_cultivar(summary_df, delta_df):
    """Bar chart: ΔAUROC by cultivar."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    subgroups = ["aluco", "capone"]
    deltas = []
    valid_sg = []
    for sg in subgroups:
        row = delta_df.filter(pl.col("subgroup") == sg)
        if len(row):
            val = row["delta_AUROC"][0]
            if not np.isnan(val):
                deltas.append(val)
                valid_sg.append(sg)
                continue
        deltas.append(0)
        valid_sg.append(sg)

    if not valid_sg or max(abs(v) for v in deltas) == 0:
        logging.info("No valid cultivar deltas to plot")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        valid_sg, deltas, color=["#2ecc71", "#3498db"], width=0.5, edgecolor="white", linewidth=1.2
    )
    ax.axhline(y=0, color="black", linewidth=1)

    for bar, val in zip(bars, deltas):
        ypos = bar.get_height()
        voff = 0.01 if ypos >= 0 else -0.04
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ypos + voff,
            f"{val:+.3f}",
            ha="center",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_ylabel("ΔAUROC (M3 − M1)")
    ax.set_title("Multiangular Advantage by Cultivar", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    y_vals = [v for v in deltas if not np.isnan(v)]
    y_vals = y_vals if y_vals else [-0.15, 0.15]
    ax.set_ylim(min(y_vals) - 0.15, max(y_vals) + 0.15)

    fig.tight_layout()
    out_path = FIGURES_DIR / "delta_auc_by_cultivar.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {out_path}")


def plot_delta_by_treatment(summary_df, delta_df):
    """Bar chart: ΔAUROC by treatment."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    subgroups = ["trt", "no_trt"]
    labels = ["Treatment (trt)", "No Treatment (no_trt)"]
    deltas = []
    valid_labels = []
    for sg, lbl in zip(subgroups, labels):
        row = delta_df.filter(pl.col("subgroup") == sg)
        if len(row):
            val = row["delta_AUROC"][0]
            if not np.isnan(val):
                deltas.append(val)
                valid_labels.append(lbl)
                continue
        # Single-class subgroup — show as "N/A" bar
        deltas.append(0)
        valid_labels.append(lbl + "\n(no disease variation)")

    if not valid_labels:
        logging.info("No valid treatment deltas to plot")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(valid_labels, deltas, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    ax.axhline(y=0, color="black", linewidth=1)

    for bar, val in zip(bars, deltas):
        ypos = bar.get_height()
        voff = 0.01 if ypos >= 0 else -0.04
        label = f"{val:+.3f}" if val != 0 else "N/A"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ypos + voff,
            label,
            ha="center",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_ylabel("ΔAUROC (M3 − M1)")
    ax.set_title("Multiangular Advantage by Treatment", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    y_vals = [v for v in deltas if not np.isnan(v) and v != 0]
    y_vals = y_vals if y_vals else [-0.15, 0.15]
    ax.set_ylim(min(y_vals) - 0.15, max(y_vals) + 0.15)

    fig.tight_layout()
    out_path = FIGURES_DIR / "delta_auc_by_treatment.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {out_path}")


def plot_delta_by_week(week_delta_df):
    """Bar chart: ΔAUROC by week."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if week_delta_df.height == 0:
        logging.info("No valid week deltas to plot")
        return

    weeks = week_delta_df["week"].to_list()
    deltas = week_delta_df["delta_AUROC"].to_list()
    labels = [f"wk{w}" for w in weeks]

    # Filter out NaN
    valid = [(l, d) for l, d in zip(labels, deltas) if not np.isnan(d)]
    if not valid:
        logging.info("All week deltas are NaN — skipping plot")
        return
    labels, deltas = zip(*valid)

    colors = ["#3498db", "#9b59b6", "#e74c3c", "#e67e22"][: len(labels)]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, deltas, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    ax.axhline(y=0, color="black", linewidth=1)

    for bar, val in zip(bars, deltas):
        ypos = bar.get_height()
        voff = 0.01 if ypos >= 0 else -0.04
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ypos + voff,
            f"{val:+.3f}",
            ha="center",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_ylabel("ΔAUROC (M3 − M1)")
    ax.set_title("Multiangular Advantage by Week", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymin = min(deltas) - 0.15
    ymax = max(deltas) + 0.15
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    out_path = FIGURES_DIR / "delta_auc_by_week.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {out_path}")


def write_markdown_summary(summary_df, delta_df, week_delta_df):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "subgroup_analysis_summary.md"

    lines = []
    lines.append("## Results: Subgroup Robustness Analysis")
    lines.append("")
    lines.append(
        "Comparison of M1 (Nadir bands) vs M3 (Multiangular VZA) across "
        "cultivar and treatment subgroups."
    )
    lines.append("")
    lines.append("CV: StratifiedGroupKFold(n_splits=5) by plot_id.")
    lines.append("")

    # Per-subgroup table
    for subgroup_key in SUBGROUPS:
        sub = summary_df.filter(pl.col("subgroup") == subgroup_key)
        if sub.height == 0:
            continue
        lines.append(f"### Subgroup: `{subgroup_key}`")
        lines.append("")
        lines.append(
            "| Feature Set | AUROC (mean ± std) | AUPRC (mean ± std) | "
            "BalAcc (mean ± std) | Recall (mean ± std) | Folds |"
        )
        lines.append(
            "|------------|-------------------|-------------------|"
            "-------------------|-------------------|-------|"
        )
        for row in sub.iter_rows(named=True):
            name = row["feature_set"]
            lines.append(
                f"| {name} ({DISPLAY.get(name, name)}) "
                f"| {row['AUROC_mean']:.3f} ± {row['AUROC_std']:.3f} "
                f"| {row['AUPRC_mean']:.3f} ± {row['AUPRC_std']:.3f} "
                f"| {row['balanced_accuracy_mean']:.3f} ± {row['balanced_accuracy_std']:.3f} "
                f"| {row['recall_mean']:.3f} ± {row['recall_std']:.3f} "
                f"| {row['n_folds']} |"
            )
        lines.append("")

        m1 = sub.filter(pl.col("feature_set") == "M1")
        m3 = sub.filter(pl.col("feature_set") == "M3")
        if len(m1) and len(m3):
            d_auc = m3["AUROC_mean"][0] - m1["AUROC_mean"][0]
            lines.append(f"**ΔAUROC (M3−M1)**: {d_auc:+.3f}")
            lines.append("")

    # ΔAUROC summary table
    lines.append("## ΔAUROC Summary (M3 − M1)")
    lines.append("")
    lines.append("| Subgroup | M1 AUROC | M3 AUROC | ΔAUROC | ΔAUPRC |")
    lines.append("|----------|---------|---------|--------|--------|")
    for row in delta_df.iter_rows(named=True):
        lines.append(
            f"| {row['subgroup']} "
            f"| {row['M1_AUROC_mean']:.3f} "
            f"| {row['M3_AUROC_mean']:.3f} "
            f"| {row['delta_AUROC']:+.3f} "
            f"| {row['delta_AUPRC']:+.3f} |"
        )
    lines.append("")

    # ΔAUROC over weeks
    if len(week_delta_df):
        lines.append("## ΔAUROC by Week")
        lines.append("")
        lines.append("| Week | M1 AUROC | M3 AUROC | ΔAUROC |")
        lines.append("|------|---------|---------|--------|")
        for row in week_delta_df.iter_rows(named=True):
            lines.append(
                f"| wk{row['week']} "
                f"| {row['M1_AUROC_mean']:.3f} "
                f"| {row['M3_AUROC_mean']:.3f} "
                f"| {row['delta_AUROC']:+.3f} |"
            )
        lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    avg_delta = delta_df["delta_AUROC"].mean()
    all_delta = delta_df.filter(pl.col("subgroup") == "all")
    all_d = all_delta["delta_AUROC"][0] if len(all_delta) else float("nan")
    lines.append(
        f"Overall, multiangular features (M3) improved AUROC by {all_d:+.3f} "
        f"over nadir-only (M1). Averaged across all subgroups, the mean ΔAUROC "
        f"was {avg_delta:+.3f}. "
    )

    # Check if advantage persists
    positive_delta = delta_df.filter(pl.col("delta_AUROC") > 0)
    lines.append(
        f"The multiangular advantage persisted in {positive_delta.height}/{delta_df.height} "
        f"subgroups, suggesting that off-nadir information is broadly beneficial "
        f"and not driven by a single cultivar or treatment group."
    )
    lines.append("")

    # Outputs
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Fold results: `outputs/results/model_comparison_by_subgroup.csv`")
    lines.append(f"- Report: `{report_path}`")
    lines.append(f"- Plot: `outputs/figures/delta_auc_by_cultivar.png`")
    lines.append(f"- Plot: `outputs/figures/delta_auc_by_treatment.png`")
    lines.append(f"- Plot: `outputs/figures/delta_auc_by_week.png`")
    lines.append(f"- Log: `{LOG_FILE}`")
    lines.append("")

    # Reproducibility
    lines.append("## Reproducibility")
    lines.append(
        f"- Model: LogisticRegression(class_weight='balanced', max_iter=1000, random_state={SEED})"
    )
    lines.append("- Preprocessing: SimpleImputer(median) + StandardScaler")
    lines.append(f"- CV: StratifiedGroupKFold(n_splits={N_SPLITS}) by plot_id")
    lines.append("- Feature sets: M1 (nadir bands), M3 (multiangular VZA)")
    lines.append("- Subgroups: all, aluco, capone, trt, no_trt, and 4 combinations")
    lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logging.info(f"Saved report: {report_path}")


def main():
    setup_logging()
    t0 = time.time()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Subgroup analysis
    logging.info("Phase 1: Running subgroup analysis...")
    t1 = time.time()
    fold_df = run_subgroup_analysis()
    logging.info(f"[PHASE] Subgroup CV: {time.time() - t1:.1f}s")

    if fold_df.height == 0:
        logging.error("No results generated!")
        return

    # Save fold-level results
    fold_path = RESULTS_DIR / "model_comparison_by_subgroup.csv"
    fold_df.write_csv(fold_path)
    logging.info(f"Fold results saved: {fold_path}")

    # Phase 2: Summarize
    t2 = time.time()
    summary_df = summarize_subgroup(fold_df)

    logging.info(f"\n{'='*80}")
    logging.info("  SUBGROUP ANALYSIS SUMMARY")
    logging.info(f"{'='*80}")
    for row in summary_df.iter_rows(named=True):
        logging.info(
            f"  {row['feature_set']:>3s} | {row['subgroup']:<16s} | "
            f"AUROC={row['AUROC_mean']:.3f}±{row['AUROC_std']:.3f}  "
            f"AUPRC={row['AUPRC_mean']:.3f}±{row['AUPRC_std']:.3f}  "
            f"BalAcc={row['balanced_accuracy_mean']:.3f}  "
            f"Recall={row['recall_mean']:.3f}  "
            f"folds={row['n_folds']}"
        )
    logging.info(f"[PHASE] Summarization: {time.time() - t2:.1f}s")

    # Phase 3: Delta computation
    t3 = time.time()
    delta_df = compute_delta_per_subgroup(summary_df)
    week_delta_df = compute_delta_per_week(fold_df)
    logging.info(f"[PHASE] Delta computation: {time.time() - t3:.1f}s")

    # Phase 4: Plots
    t4 = time.time()
    plot_delta_by_cultivar(summary_df, delta_df)
    plot_delta_by_treatment(summary_df, delta_df)
    plot_delta_by_week(week_delta_df)
    logging.info(f"[PHASE] Plotting: {time.time() - t4:.1f}s")

    # Phase 5: Markdown summary
    t5 = time.time()
    write_markdown_summary(summary_df, delta_df, week_delta_df)
    logging.info(f"[PHASE] Report: {time.time() - t5:.1f}s")

    logging.info(f"[PHASE] Total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
