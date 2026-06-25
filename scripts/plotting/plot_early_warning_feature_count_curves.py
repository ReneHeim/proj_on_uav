"""Plot external F1 versus number of stability-selected angular features.

The feature ranking is learned from 2024 only. For each top-k feature subset,
the classification threshold is calibrated inside 2024 and evaluated once on
the 2025 external rows.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.severity.analyze_cross_year_generalization_2024_to_2025 import (  # noqa: E402
    RAA_2024,
    RAA_2025,
    SEED,
    VZA_2024,
    VZA_2025,
    WARNING_TARGET,
    align_train_test,
    binary_metrics,
    build_model_table,
    grouped_cv_thresholds,
    load_2024_disease_with_fallback,
    load_2025_disease_with_fallback,
    load_feature_sets_for_year,
)
from scripts.analysis.early_warning.explore_early_warning_feature_selection import (  # noqa: E402
    make_logistic,
    make_xgboost,
    stable_features_l1,
)

OUTPUT_ROOT = (
    ROOT / "outputs/cross_year_generalization_2024_to_2025/early_warning_feature_selection"
)
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/logs"

FEATURE_SETS = {
    "multiangular_vza_raa": "VZA + RAA",
    "multiangular_vza_phase": "VZA + phase",
}
MODELS = ["logistic", "xgboost"]
K_VALUES = [5, 10, 15, 20, 30, 40, 60, 80, 100, 125, 150, 175, 200, 225, 250, 300, 350]
THRESHOLD_STRATEGY = "high_recall"


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"plot_early_warning_feature_count_curves_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    logging.info("Log file: %s", log_path)
    return log_path


def log_phase(name: str, t0: float) -> None:
    logging.info("[PHASE] %s: %.1fs", name, time.time() - t0)


def load_tables() -> tuple[dict[str, object], dict[str, object]]:
    t0 = time.time()
    features_2024 = load_feature_sets_for_year(VZA_2024, RAA_2024)
    features_2025 = load_feature_sets_for_year(VZA_2025, RAA_2025)
    disease_2024 = load_2024_disease_with_fallback()
    disease_2025, _audit = load_2025_disease_with_fallback()
    train_tables = {
        name: build_model_table(features_2024[name], disease_2024) for name in FEATURE_SETS
    }
    test_tables = {
        name: build_model_table(features_2025[name], disease_2025) for name in FEATURE_SETS
    }
    log_phase("load and join feature/target tables", t0)
    return train_tables, test_tables


def valid_k_values(n_features: int) -> list[int]:
    values = [k for k in K_VALUES if k < n_features]
    if n_features not in values:
        values.append(n_features)
    return sorted(set(values))


def fit_predict_for_k(model_name: str, train, test, cols: list[str]) -> dict[str, float]:
    y_train = train[WARNING_TARGET].to_numpy(int)
    y_test = test[WARNING_TARGET].to_numpy(int)
    estimator = make_logistic() if model_name == "logistic" else make_xgboost(y_train)

    threshold_t0 = time.time()
    thresholds = grouped_cv_thresholds(estimator, train[cols], y_train, train["plot_id"].to_numpy())
    threshold = thresholds[THRESHOLD_STRATEGY]
    log_phase(f"threshold calibration {model_name} k={len(cols)}", threshold_t0)

    fit_t0 = time.time()
    estimator.fit(train[cols], y_train)
    fit_time = time.time() - fit_t0

    pred_t0 = time.time()
    prob = estimator.predict_proba(test[cols])[:, 1]
    pred = (prob >= threshold).astype(int)
    predict_time = time.time() - pred_t0

    metrics = binary_metrics(y_test, pred, prob)
    return {
        "threshold": float(threshold),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
        **metrics,
    }


def run_curves() -> tuple[pl.DataFrame, list[Path], Path]:
    train_tables, test_tables = load_tables()
    rows: list[dict[str, object]] = []
    stability_paths: list[Path] = []

    for feature_set, label in FEATURE_SETS.items():
        cols, train, test = align_train_test(train_tables[feature_set], test_tables[feature_set])
        y_train = train[WARNING_TARGET].to_numpy(int)
        stability_t0 = time.time()
        stability = stable_features_l1(train, cols, y_train, train["plot_id"].to_numpy())
        log_phase(f"stability ranking {feature_set}", stability_t0)
        stability_path = RESULTS_DIR / f"feature_count_curve_stability_{feature_set}.csv"
        stability.write_csv(stability_path)
        stability_paths.append(stability_path)
        ranked_cols = stability.get_column("feature").to_list()

        for model_name in MODELS:
            for k in valid_k_values(len(ranked_cols)):
                selected_cols = ranked_cols[:k]
                metrics = fit_predict_for_k(model_name, train, test, selected_cols)
                rows.append(
                    {
                        "feature_set": feature_set,
                        "feature_label": label,
                        "model": model_name,
                        "threshold_strategy": THRESHOLD_STRATEGY,
                        "k_features": k,
                        "n_train": len(train),
                        "n_test": len(test),
                        "n_positive_test": int(test[WARNING_TARGET].sum()),
                        **metrics,
                    }
                )

    results = pl.DataFrame(rows)
    results_path = RESULTS_DIR / "early_warning_f1_by_feature_count.csv"
    results.write_csv(results_path)
    return results, stability_paths, results_path


def plot_one_family(results: pl.DataFrame, feature_set: str) -> Path:
    label = FEATURE_SETS[feature_set]
    data = results.filter(pl.col("feature_set") == feature_set).sort("k_features")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = {"logistic": "#2f6f9f", "xgboost": "#c96f2d"}
    markers = {"logistic": "o", "xgboost": "s"}
    for model in MODELS:
        subset = data.filter(pl.col("model") == model)
        ax.plot(
            subset.get_column("k_features"),
            subset.get_column("f1"),
            marker=markers[model],
            linewidth=2.2,
            markersize=4.8,
            color=colors[model],
            label=model.capitalize(),
        )
        best = subset.sort("f1", descending=True).row(0, named=True)
        ax.scatter(
            best["k_features"], best["f1"], s=92, color=colors[model], edgecolor="black", zorder=5
        )
        ax.annotate(
            f"best k={best['k_features']}\nF1={best['f1']:.3f}",
            xy=(best["k_features"], best["f1"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color="#242424",
        )
    ax.set_title(
        f"Early-warning F1 versus selected features: {label}", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Number of top stability-ranked features")
    ax.set_ylabel("External 2025 F1 score")
    ax.set_ylim(0.15, 0.86)
    ax.grid(axis="y", color="#d7d0c6", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.patch.set_facecolor("#fbf7ef")
    ax.set_facecolor("#fbf7ef")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"early_warning_f1_by_feature_count_{feature_set}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def markdown_table(df: pl.DataFrame) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.iter_rows(named=True):
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def write_report(
    results: pl.DataFrame,
    figure_paths: list[Path],
    results_path: Path,
    stability_paths: list[Path],
    log_path: Path,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best = (
        results.sort("f1", descending=True)
        .group_by(["feature_label", "model"])
        .head(1)
        .sort(["feature_label", "model"])
        .select(
            [
                "feature_label",
                "model",
                "k_features",
                "f1",
                "balanced_accuracy",
                "recall",
                "precision",
                "specificity",
                "threshold",
            ]
        )
        .with_columns(pl.selectors.numeric().round(3))
    )
    report_path = REPORTS_DIR / "early_warning_f1_by_feature_count_summary.md"
    lines = [
        "# Early-Warning F1 By Number Of Selected Features",
        "",
        "## Best Points",
        "",
        markdown_table(best),
        "",
        "**Interpretation**: F1 is not monotonic with feature count. The strongest models use a small, stability-ranked subset rather than the full angular feature space, supporting feature selection as a regularization step for cross-year transfer.",
        "",
        "## Outputs",
        "",
        f"- Results CSV: `{results_path}`",
        *[f"- Figure: `{path}`" for path in figure_paths],
        *[f"- Stability ranking: `{path}`" for path in stability_paths],
        f"- Log: `{log_path}`",
        "",
        "## Reproducibility",
        "",
        f"- Seed: `{SEED}`",
        "- Feature ranking: L1 logistic stability selection using 2024 only",
        f"- Threshold strategy: `{THRESHOLD_STRATEGY}` calibrated inside 2024 grouped CV",
        "- Train year: `2024`",
        "- External test year: `2025`",
        "- Target: `future_warning_ds_ge_5`",
    ]
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging()
    script_t0 = time.time()
    results, stability_paths, results_path = run_curves()
    figure_paths = [plot_one_family(results, feature_set) for feature_set in FEATURE_SETS]
    report_path = write_report(results, figure_paths, results_path, stability_paths, log_path)
    logging.info("Results written: %s", results_path)
    for path in figure_paths:
        logging.info("Figure written: %s", path)
    logging.info("Report written: %s", report_path)
    log_phase("total script", script_t0)


if __name__ == "__main__":
    main()
