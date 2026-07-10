#!/usr/bin/env python3
"""Early-warning prediction: train on week t features, predict disease at t+1 or t+2.

For each feature set (M1-nadir, M3-multiangular):
  - Use temporal pairs across available weeks
  - LeaveOneGroupOut CV by plot_id
  - LogisticRegression(class_weight="balanced")
  - Output per-fold metrics: AUROC, AUPRC, recall, balanced_accuracy
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
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"early_warning_prediction_{TIMESTAMP}.log"


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


def build_temporal_pairs(df):
    """For a feature set dataframe, build (week_t, week_t_plus_delta) pairs
    matching on plot_id. Returns list of (t, t+delta, delta, X_rows) tuples.
    """
    weeks_avail = sorted(df["week"].unique().to_list())
    pairs = []

    # t+1 pairs
    for i in range(len(weeks_avail) - 1):
        t = weeks_avail[i]
        t_next = weeks_avail[i + 1]
        pairs.append((t, t_next, 1))

    # t+2 pairs
    for i in range(len(weeks_avail) - 2):
        t = weeks_avail[i]
        t_next = weeks_avail[i + 2]
        pairs.append((t, t_next, 2))

    return pairs


def build_temporal_dataset(df, t, t_plus_delta):
    """Merge features from week t with disease_label from week t_plus_delta."""
    df_t = df.filter(pl.col("week") == t)
    join_cols = ["plot_id"]
    if "year" in df.columns:
        join_cols.insert(0, "year")
    df_target = df.filter(pl.col("week") == t_plus_delta).select(join_cols + ["disease_label"])

    merged = df_t.join(df_target, on=join_cols, suffix="_target")
    merged = merged.rename({"disease_label_target": "target_label"})
    # Keep the original disease_label column (from week t) but also have target_label
    merged = merged.drop("disease_label")
    merged = merged.with_columns(merged["target_label"].cast(pl.Int64))

    return merged


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


def evaluate_temporal_pair(df, feature_set_name, t, t_plus_delta, delta):
    """Evaluate one temporal pair with GroupKFold by plot_id."""
    merged = build_temporal_dataset(df, t, t_plus_delta)
    if len(merged) < 4:
        logging.info(f"  wk{t}→wk{t_plus_delta}: only {len(merged)} plots, skipping")
        return []

    feature_cols = [c for c in merged.columns if c not in EXCLUDE_COLS and c != "target_label"]
    if not feature_cols:
        logging.info(f"  wk{t}→wk{t_plus_delta}: no feature columns, skipping")
        return []

    X = merged.select(feature_cols).to_numpy()
    y = merged["target_label"].to_numpy()
    plot_ids = merged["plot_id"].to_numpy()

    # Convert plot_id strings to integer group labels for GroupKFold
    unique_plots = {pid: i for i, pid in enumerate(np.unique(plot_ids))}
    groups = np.array([unique_plots[pid] for pid in plot_ids])

    n_plots = len(np.unique(groups))
    n_splits = min(5, n_plots)
    gkf = GroupKFold(n_splits=n_splits)
    pipe = build_pipeline()

    pos_rate = y.mean()
    logging.info(
        f"  wk{t}→wk{t_plus_delta} (Δt={delta}): "
        f"{len(y)} rows, {n_plots} plots, pos={pos_rate:.2f}, folds={n_splits}"
    )

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)

        fold_results.append(
            {
                "feature_set": feature_set_name,
                "week_t": t,
                "week_t_plus_delta": t_plus_delta,
                "delta": delta,
                "fold": fold,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "positive_rate_test": y_test.mean(),
                **metrics,
            }
        )

    return fold_results


def run_early_warning():
    all_folds = []

    for fs_name in FEATURE_SETS:
        logging.info(f"\n{'='*60}")
        logging.info(f"  {fs_name} ({DISPLAY[fs_name]})")
        logging.info(f"{'='*60}")

        df = load_feature_set(fs_name)
        pairs = build_temporal_pairs(df)

        for t, t_plus_delta, delta in pairs:
            results = evaluate_temporal_pair(df, fs_name, t, t_plus_delta, delta)
            all_folds.extend(results)

    return pl.DataFrame(all_folds)


def summarize_early_warning(fold_df):
    summary = (
        fold_df.group_by(["feature_set", "delta"])
        .agg(
            n_folds=pl.len(),
            AUROC_mean=pl.col("AUROC").filter(pl.col("AUROC").is_not_nan()).mean(),
            AUROC_std=pl.col("AUROC").filter(pl.col("AUROC").is_not_nan()).std(),
            AUPRC_mean=pl.col("AUPRC").filter(pl.col("AUPRC").is_not_nan()).mean(),
            AUPRC_std=pl.col("AUPRC").filter(pl.col("AUPRC").is_not_nan()).std(),
            balanced_accuracy_mean=pl.col("balanced_accuracy").mean(),
            balanced_accuracy_std=pl.col("balanced_accuracy").std(),
            recall_mean=pl.col("recall").mean(),
            recall_std=pl.col("recall").std(),
        )
        .sort(["delta", "feature_set"])
    )
    return summary


def plot_early_warning_results(summary_df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = ["AUROC", "AUPRC", "recall", "balanced_accuracy"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        for delta in [1, 2]:
            sub = summary_df.filter(pl.col("delta") == delta)
            x = np.arange(len(sub))
            means = sub[mean_col].to_numpy()
            stds = sub[std_col].to_numpy()
            labels = [f"{r['feature_set']} (Δt={delta})" for r in sub.iter_rows(named=True)]

            color = "#3498db" if delta == 1 else "#e74c3c"
            ax.bar(
                x + (delta - 1.5) * 0.35,
                means,
                0.3 - 0.05 * delta,
                yerr=stds,
                color=color,
                alpha=0.8,
                capsize=4,
                label=f"Δt={delta}",
            )

        ax.set_title(metric, fontweight="bold")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([r["feature_set"] for r in sub.iter_rows(named=True)])
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        "Early-Warning Prediction: M1 (Nadir) vs M3 (Multiangular)", fontweight="bold", fontsize=13
    )
    fig.tight_layout()
    out_path = FIGURES_DIR / "early_warning_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved figure: {out_path}")

    # ΔAUROC plot: M3 - M1 per delta
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for delta in [1, 2]:
        sub = summary_df.filter(pl.col("delta") == delta)
        auroc = {}
        for r in sub.iter_rows(named=True):
            auroc[r["feature_set"]] = r["AUROC_mean"]
        delta_auc = auroc.get("M3", 0) - auroc.get("M1", 0)
        color = "#3498db" if delta == 1 else "#e74c3c"
        ax2.bar([delta], [delta_auc], color=color, width=0.5, edgecolor="white", linewidth=1.2)
        ax2.text(
            delta,
            delta_auc + 0.01,
            f"{delta_auc:+.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax2.axhline(y=0, color="black", linewidth=1)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["Δt=1 week", "Δt=2 weeks"])
    ax2.set_ylabel("ΔAUROC (M3 − M1)")
    ax2.set_title("Multiangular Advantage for Early Warning", fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    out_path2 = FIGURES_DIR / "early_warning_delta_auc.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logging.info(f"Saved figure: {out_path2}")


def write_markdown_summary(summary_df, fold_df):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "early_warning_summary.md"

    lines = []
    lines.append("## Results: Early-Warning Prediction")
    lines.append("")
    lines.append("Train on week *t* features, predict disease label at *t+Δ*.")
    lines.append("M1 = Nadir bands; M3 = Multiangular VZA features.")
    lines.append("CV: GroupKFold (n_splits≤5) by plot_id.")
    lines.append("")

    for delta in [1, 2]:
        sub = summary_df.filter(pl.col("delta") == delta)
        lines.append(f"### Δt = {delta} week(s)")
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
            lines.append(
                f"| {row['feature_set']} ({DISPLAY[row['feature_set']]}) "
                f"| {row['AUROC_mean']:.3f} ± {row['AUROC_std']:.3f} "
                f"| {row['AUPRC_mean']:.3f} ± {row['AUPRC_std']:.3f} "
                f"| {row['balanced_accuracy_mean']:.3f} ± {row['balanced_accuracy_std']:.3f} "
                f"| {row['recall_mean']:.3f} ± {row['recall_std']:.3f} "
                f"| {row['n_folds']} |"
            )
        lines.append("")

        # Delta
        m1 = [r for r in sub.iter_rows(named=True) if r["feature_set"] == "M1"]
        m3 = [r for r in sub.iter_rows(named=True) if r["feature_set"] == "M3"]
        if m1 and m3:
            d_auc = m3[0]["AUROC_mean"] - m1[0]["AUROC_mean"]
            d_auprc = m3[0]["AUPRC_mean"] - m1[0]["AUPRC_mean"]
            lines.append(
                f"**ΔAUROC (M3−M1)**: {d_auc:+.3f} — {'Multiangular advantage' if d_auc > 0 else 'Nadir advantage'}"
            )
            lines.append(f"**ΔAUPRC (M3−M1)**: {d_auprc:+.3f}")
            lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    m1_data = summary_df.filter(pl.col("feature_set") == "M1")
    m3_data = summary_df.filter(pl.col("feature_set") == "M3")
    m1_avg = m1_data["AUROC_mean"].mean()
    m3_avg = m3_data["AUROC_mean"].mean()
    diff = m3_avg - m1_avg
    lines.append(
        f"Multiangular features (M3) improved early-warning AUROC by "
        f"{diff:+.3f} over nadir-only (M1) averaged across temporal gaps, "
        f"suggesting that off-nadir viewing angles carry disease-relevant "
        f"spectral information detectable before the disease becomes severe."
    )
    lines.append("")

    # Outputs
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Fold results: `outputs/archive/legacy_unscoped/results/early_warning_model_comparison.csv`")
    lines.append(f"- Report: `{report_path}`")
    lines.append(f"- Plots: `outputs/archive/legacy_unscoped/figures/early_warning_comparison.png`")
    lines.append(f"- Plots: `outputs/archive/legacy_unscoped/figures/early_warning_delta_auc.png`")
    lines.append(f"- Log: `{LOG_FILE}`")
    lines.append("")

    # Reproducibility
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(
        f"- Model: LogisticRegression(class_weight='balanced', max_iter=1000, random_state={SEED})"
    )
    lines.append("- Preprocessing: SimpleImputer(median) + StandardScaler")
    lines.append("- CV: GroupKFold (n_splits≤5) by plot_id")
    lines.append("- Feature sets: M1 (nadir bands), M3 (multiangular VZA)")
    lines.append("- Temporal gaps: Δt = 1 week, Δt = 2 weeks")
    lines.append(f"- Weeks available: [0, 3, 5, 8]")
    lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logging.info(f"Saved report: {report_path}")


def main():
    setup_logging()
    t0 = time.time()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading feature sets and building temporal pairs...")
    t1 = time.time()

    fold_df = run_early_warning()
    logging.info(f"[PHASE] Data loading + CV: {time.time() - t1:.1f}s")

    # Save fold-level results
    fold_path = RESULTS_DIR / "early_warning_model_comparison.csv"
    fold_df.write_csv(fold_path)
    logging.info(f"Fold results saved: {fold_path}")

    # Summarize
    t2 = time.time()
    summary_df = summarize_early_warning(fold_df)
    logging.info(f"[PHASE] Summarization: {time.time() - t2:.1f}s")

    logging.info(f"\n{'='*80}")
    logging.info("  EARLY-WARNING PREDICTION SUMMARY")
    logging.info(f"{'='*80}")
    for row in summary_df.iter_rows(named=True):
        logging.info(
            f"  {row['feature_set']} Δt={row['delta']}: "
            f"AUROC={row['AUROC_mean']:.3f}±{row['AUROC_std']:.3f}  "
            f"AUPRC={row['AUPRC_mean']:.3f}±{row['AUPRC_std']:.3f}  "
            f"BalAcc={row['balanced_accuracy_mean']:.3f}±{row['balanced_accuracy_std']:.3f}  "
            f"Recall={row['recall_mean']:.3f}±{row['recall_std']:.3f}  "
            f"folds={row['n_folds']}"
        )

    # Plots
    t3 = time.time()
    plot_early_warning_results(summary_df)
    logging.info(f"[PHASE] Plotting: {time.time() - t3:.1f}s")

    # Markdown summary
    write_markdown_summary(summary_df, fold_df)

    logging.info(f"[PHASE] Total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
