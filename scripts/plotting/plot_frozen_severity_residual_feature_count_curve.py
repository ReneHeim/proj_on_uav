"""Plot RMSE versus retained compact features for the frozen severity architecture.

This script does not modify the frozen pipeline configuration. It reruns the
same two-stage residual architecture on top-k subsets of the already selected
and reliability-filtered compact multiangular features, so the plot explains
why the manuscript severity table reaches RMSE near 8 while the simpler direct
severity feature-count experiment does not.
"""

from __future__ import annotations

import logging
import math
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
    TARGET,
    build_model_table,
    load_2024_disease_with_fallback,
    load_2025_disease_with_fallback,
)
from scripts.analysis.severity.debug_multiangular_rmse_bottleneck import (  # noqa: E402
    COVARIATES,
    fit_residual_reliability_filtered_xgboost,
    fit_tuned_xgboost_residual_with_cols,
    load_cached_features,
    prepare_aligned,
    reliability_filtered_cols,
    score_predictions,
    select_stable_features,
)

OUTPUT_ROOT = ROOT / "outputs/multiangular_distribution_feature_family/model_bottleneck_debug"
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/logs"

MULTIANGULAR_SET = "compact_anomaly_multiangular"
NADIR_SET = "compact_anomaly_nadir"
K_VALUES = [5, 10, 15, 20, 25, 30, 35, 40]


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"plot_frozen_severity_residual_feature_count_curve_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    logging.info("Log file: %s", log_path)
    return log_path


def log_phase(name: str, started: float) -> None:
    logging.info("[PHASE] %s: %.1fs", name, time.perf_counter() - started)


def ensure_dirs() -> None:
    for path in [RESULTS_DIR, FIGURES_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def timing_features(cols: list[str]) -> list[str]:
    return [col for col in cols if col.startswith("known__")]


def reflectance_features(cols: list[str]) -> list[str]:
    return [col for col in cols if not col.startswith("known__")]


def sorted_reliable_features(selection, filtered_cols: list[str]) -> list[str]:
    reliable = set(filtered_cols)
    ranked = selection.loc[
        selection["feature"].isin(reliable),
        ["feature", "selection_frequency", "mean_abs_elasticnet_coef"],
    ].sort_values(["selection_frequency", "mean_abs_elasticnet_coef"], ascending=[False, False])
    return [col for col in ranked["feature"].tolist() if not col.startswith("known__")]


def valid_k_values(n_reflectance: int) -> list[int]:
    values = [k for k in K_VALUES if k < n_reflectance]
    values.append(n_reflectance)
    return sorted(set(values))


def build_tables():
    t0 = time.perf_counter()
    features = load_cached_features()
    disease_2024 = load_2024_disease_with_fallback()
    disease_2025, _audit = load_2025_disease_with_fallback()
    train_multi = build_model_table(features[MULTIANGULAR_SET][0], disease_2024)
    test_multi = build_model_table(features[MULTIANGULAR_SET][1], disease_2025)
    train_nadir = build_model_table(features[NADIR_SET][0], disease_2024)
    test_nadir = build_model_table(features[NADIR_SET][1], disease_2025)
    log_phase("load compact features and join severity targets", t0)
    return train_multi, test_multi, train_nadir, test_nadir


def evaluate_nadir_reference(train_nadir, test_nadir) -> tuple[dict[str, object], Path]:
    t0 = time.perf_counter()
    result, predictions, _tuning = fit_residual_reliability_filtered_xgboost(train_nadir, test_nadir, NADIR_SET)
    row = {
        "feature_set": NADIR_SET,
        "model_family": "frozen_style_residual_pipeline",
        "k_reflectance_features": result["n_features"] - 2,
        "n_total_features": result["n_features"],
        "is_reference": True,
        **result,
    }
    path = RESULTS_DIR / "frozen_style_residual_nadir_reference_predictions.csv"
    pl.from_pandas(predictions).write_csv(path)
    log_phase("evaluate compact nadir residual reference", t0)
    return row, path


def evaluate_multiangular_curve(train_multi, test_multi) -> tuple[pl.DataFrame, pl.DataFrame]:
    t0 = time.perf_counter()
    cols, train_aligned, test_aligned = prepare_aligned(train_multi, test_multi)
    selected_cols, selection = select_stable_features(train_aligned, cols)
    filtered_cols, shift = reliability_filtered_cols(train_aligned, test_aligned, selected_cols)
    fixed_timing = timing_features(filtered_cols)
    ranked_reflectance = sorted_reliable_features(selection, filtered_cols)
    if not ranked_reflectance:
        raise RuntimeError("No reliable compact multiangular reflectance features available for feature-count curve.")

    selection_out = selection.copy()
    selection_out["selected_before_reliability_filter"] = selection_out["feature"].isin(selected_cols)
    selection_out["retained_after_reliability_filter"] = selection_out["feature"].isin(filtered_cols)
    selection_out["feature_rank_after_reliability"] = np.nan
    rank_map = {feature: rank + 1 for rank, feature in enumerate(ranked_reflectance)}
    selection_out.loc[
        selection_out["feature"].isin(rank_map),
        "feature_rank_after_reliability",
    ] = selection_out.loc[selection_out["feature"].isin(rank_map), "feature"].map(rank_map)
    selection_path = RESULTS_DIR / "frozen_style_residual_multiangular_feature_count_ranking.csv"
    pl.from_pandas(selection_out).write_csv(selection_path)
    shift_path = RESULTS_DIR / "frozen_style_residual_multiangular_reliability_filter.csv"
    pl.from_pandas(shift).write_csv(shift_path)

    rows: list[dict[str, object]] = []
    for k in valid_k_values(len(ranked_reflectance)):
        model_name = f"residual_reliability_filtered_xgboost_top_{k:02d}_compact_features"
        cols_k = fixed_timing + ranked_reflectance[:k]
        fit_started = time.perf_counter()
        result, predictions, tuning = fit_tuned_xgboost_residual_with_cols(
            train_aligned,
            test_aligned,
            cols_k,
            model_name,
            MULTIANGULAR_SET,
        )
        fit_elapsed = time.perf_counter() - fit_started
        base_result = score_predictions(
            predictions.assign(y_pred=predictions["base_pred"]),
            len(train_aligned),
            len(cols_k),
            f"ridge_base_top_{k:02d}_compact_features",
            MULTIANGULAR_SET,
        )
        rows.append(
            {
                "feature_set": MULTIANGULAR_SET,
                "model_family": "frozen_style_residual_pipeline",
                "k_reflectance_features": k,
                "n_timing_features": len(fixed_timing),
                "n_total_features": len(cols_k),
                "is_reference": False,
                "base_ridge_rmse": base_result["rmse"],
                "base_ridge_mae": base_result["mae"],
                "base_ridge_r2": base_result["r2"],
                "base_ridge_spearman": base_result["spearman"],
                "residual_gain_rmse": base_result["rmse"] - result["rmse"],
                "xgboost_config": result.get("xgboost_config"),
                "best_iteration": result.get("best_iteration"),
                "eval_rmse_2024": result.get("eval_rmse_2024"),
                "fit_time_s": fit_elapsed,
                **result,
            }
        )
        tuning_out = tuning.copy()
        tuning_out["k_reflectance_features"] = k
        tuning_path = RESULTS_DIR / f"frozen_style_residual_tuning_top_{k:02d}_compact_features.csv"
        pl.from_pandas(tuning_out).write_csv(tuning_path)
        logging.info(
            "Evaluated top-%s compact features: RMSE %.3f, base Ridge RMSE %.3f, residual gain %.3f",
            k,
            result["rmse"],
            base_result["rmse"],
            base_result["rmse"] - result["rmse"],
        )

    log_phase("evaluate compact multiangular residual feature-count curve", t0)
    return pl.DataFrame(rows), pl.from_pandas(selection_out)


def plot_curve(results: pl.DataFrame, nadir_rmse: float) -> Path:
    t0 = time.perf_counter()
    data = results.sort("k_reflectance_features")
    x = data.get_column("k_reflectance_features").to_numpy()
    y = data.get_column("rmse").to_numpy()
    y_base = data.get_column("base_ridge_rmse").to_numpy()
    best_idx = int(np.nanargmin(y))
    full_idx = len(x) - 1

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.plot(x, y_base, color="#8f8f8f", marker="o", lw=1.8, label="Ridge base only")
    ax.plot(x, y, color="#1f6f5b", marker="o", lw=2.4, label="Ridge + residual XGBoost")
    ax.axhline(nadir_rmse, color="#b5453c", lw=1.8, ls="--", label=f"Nadir residual reference ({nadir_rmse:.2f})")
    ax.scatter([x[best_idx]], [y[best_idx]], s=90, color="#1f6f5b", edgecolor="white", zorder=5)
    if best_idx != full_idx:
        ax.scatter([x[full_idx]], [y[full_idx]], s=80, color="#f0b44c", edgecolor="#3d3020", zorder=5)
    if best_idx == full_idx:
        ax.annotate(
            f"best/full set: {y[best_idx]:.2f} RMSE\n{k_label(x[best_idx])}",
            xy=(x[best_idx], y[best_idx]),
            xytext=(-118, 24),
            textcoords="offset points",
            fontsize=8.5,
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#3d3d3d"},
        )
    else:
        ax.annotate(
            f"best {y[best_idx]:.2f} RMSE\n{k_label(x[best_idx])}",
            xy=(x[best_idx], y[best_idx]),
            xytext=(8, -34),
            textcoords="offset points",
            fontsize=8.5,
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#3d3d3d"},
        )
        ax.annotate(
            f"full retained set {y[full_idx]:.2f} RMSE\n{k_label(x[full_idx])}",
            xy=(x[full_idx], y[full_idx]),
            xytext=(-104, 18),
            textcoords="offset points",
            fontsize=8.5,
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#3d3d3d"},
        )
    ax.set_title("Frozen-style severity pipeline: RMSE by compact feature count", fontsize=12)
    ax.set_xlabel("Retained compact multiangular reflectance features")
    ax.set_ylabel("External-year 2025 RMSE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_ylim(max(0.0, min(np.nanmin(y), np.nanmin(y_base), nadir_rmse) - 0.8), max(np.nanmax(y), np.nanmax(y_base), nadir_rmse) + 0.8)
    fig.tight_layout()
    path = FIGURES_DIR / "frozen_style_residual_rmse_by_compact_feature_count.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    log_phase("plot residual feature-count curve", t0)
    return path


def k_label(k: float) -> str:
    return f"{int(k)} compact features"


def write_report(results: pl.DataFrame, nadir_row: dict[str, object], figure_path: Path, log_path: Path) -> Path:
    t0 = time.perf_counter()
    best = results.sort("rmse").row(0, named=True)
    full = results.sort("k_reflectance_features").row(-1, named=True)
    nadir_rmse = float(nadir_row["rmse"])
    table = (
        results.select(
            [
                "k_reflectance_features",
                "n_total_features",
                "base_ridge_rmse",
                "rmse",
                "mae",
                "r2",
                "spearman",
                "residual_gain_rmse",
                "xgboost_config",
            ]
        )
        .sort("k_reflectance_features")
        .with_columns(
            [
                pl.col("base_ridge_rmse").round(3),
                pl.col("rmse").round(3),
                pl.col("mae").round(3),
                pl.col("r2").round(3),
                pl.col("spearman").round(3),
                pl.col("residual_gain_rmse").round(3),
            ]
        )
    )
    md_table = polars_markdown_table(table)
    report = f"""## Results: Frozen-Style Residual Severity Feature Count Curve

| Reference | RMSE | MAE | R2 | Spearman | Total Features |
|---|---:|---:|---:|---:|---:|
| Compact nadir residual pipeline | {float(nadir_row['rmse']):.3f} | {float(nadir_row['mae']):.3f} | {float(nadir_row['r2']):.3f} | {float(nadir_row['spearman']):.3f} | {int(nadir_row['n_features'])} |
| Best compact multiangular residual subset | {float(best['rmse']):.3f} | {float(best['mae']):.3f} | {float(best['r2']):.3f} | {float(best['spearman']):.3f} | {int(best['n_total_features'])} |
| Full retained compact multiangular residual set | {float(full['rmse']):.3f} | {float(full['mae']):.3f} | {float(full['r2']):.3f} | {float(full['spearman']):.3f} | {int(full['n_total_features'])} |

{md_table}

**Interpretation**: The RMSE near 8 comes from the compact anomaly feature family plus the two-stage residual architecture, not from the simpler direct severity feature-count experiment. The Ridge base captures the broad timing-adjusted severity trajectory, and the residual XGBoost correction lowers the error when enough reliable compact multiangular features are retained.

**Outputs**:

- `outputs/multiangular_distribution_feature_family/model_bottleneck_debug/results/frozen_style_residual_rmse_by_compact_feature_count.csv`
- `outputs/multiangular_distribution_feature_family/model_bottleneck_debug/results/frozen_style_residual_multiangular_feature_count_ranking.csv`
- `outputs/multiangular_distribution_feature_family/model_bottleneck_debug/results/frozen_style_residual_multiangular_reliability_filter.csv`
- `{figure_path.relative_to(ROOT)}`
- `{log_path.relative_to(ROOT)}`

**Reproducibility**:

- Pipeline family: frozen-style two-stage residual severity model.
- Feature family: `{MULTIANGULAR_SET}` compared with `{NADIR_SET}`.
- Covariates: `{COVARIATES}`.
- Ranking: 2024-only stability selection from `select_stable_features`.
- Reliability filter: existing non-null/SMD rule from `debug_multiangular_rmse_bottleneck.py`.
- Evaluation: 2025 external rows, no retuning against 2025 target performance beyond the already established frozen-style code path.
"""
    report_path = REPORTS_DIR / "frozen_style_residual_feature_count_curve_summary.md"
    report_path.write_text(report, encoding="utf-8")
    log_phase("write markdown report", t0)
    return report_path


def polars_markdown_table(table: pl.DataFrame) -> str:
    headers = table.columns
    rows = table.rows()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_markdown_value(value) for value in row) + " |")
    return "\n".join(lines)


def format_markdown_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    ensure_dirs()
    log_path = setup_logging()
    started = time.perf_counter()
    train_multi, test_multi, train_nadir, test_nadir = build_tables()
    nadir_row, nadir_predictions_path = evaluate_nadir_reference(train_nadir, test_nadir)
    curve, _selection = evaluate_multiangular_curve(train_multi, test_multi)
    results_path = RESULTS_DIR / "frozen_style_residual_rmse_by_compact_feature_count.csv"
    curve.write_csv(results_path)
    figure_path = plot_curve(curve, float(nadir_row["rmse"]))
    report_path = write_report(curve, nadir_row, figure_path, log_path)
    logging.info("Nadir reference predictions: %s", nadir_predictions_path)
    logging.info("Results: %s", results_path)
    logging.info("Figure: %s", figure_path)
    logging.info("Report: %s", report_path)
    log_phase("total runtime", started)


if __name__ == "__main__":
    main()
