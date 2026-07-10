#!/usr/bin/env python3
"""Create supplementary model-roster tables for current and future severity."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
MPLCONFIG_DIR = ROOT / "outputs/.matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"
REPORTS_DIR = ROOT / "outputs/archive/legacy_unscoped/reports"

CURRENT_OUTPUT = ROOT / "outputs/runs/analysis/severity/current/2024_to_2025"
CURRENT_RAA_OUTPUT = ROOT / "outputs/runs/analysis/severity/current/raa_geometry_fusion_2024_to_2025"
FUTURE_OUTPUT = ROOT / "outputs/runs/analysis/severity/future/compact_distribution_feature_family/model_bottleneck_debug"
FUTURE_DIRECT_OUTPUT = ROOT / "outputs/runs/analysis/severity/future/current_hurdle_vza_raa_2024_to_2025"
FUTURE_RAA_OUTPUT = ROOT / "outputs/runs/analysis/severity/future/raa_geometry_residual_2024_to_2025"
FUTURE_SELECTION_OUTPUT = ROOT / "outputs/runs/analysis/severity/future/vza_raa_feature_selection_improvement"

CURRENT_PRED = CURRENT_OUTPUT / "results/predictions"
CURRENT_RAA_PRED = CURRENT_RAA_OUTPUT / "results/predictions"
FUTURE_DIRECT_PRED = FUTURE_DIRECT_OUTPUT / "results/predictions"
FUTURE_RAA_PRED = FUTURE_RAA_OUTPUT / "results/predictions"
FUTURE_SELECTION_PRED = FUTURE_SELECTION_OUTPUT / "results/predictions"

PALETTE = {
    "navy": "#0B132B",
    "coral": "#FF6B6B",
    "gold": "#F6C85F",
    "grey": "#6b7280",
    "light_grey": "#e5e7eb",
    "white": "#ffffff",
}


@dataclass(frozen=True)
class ModelRow:
    model: str
    angular_inputs: str
    selection: str
    architecture: str
    prediction_path: Path | None
    n_features: int | None
    note: str = ""


CURRENT_BASELINE = (
    CURRENT_PRED
    / "severity_predictions_current_hurdle_top20_raw_positive_compact_anomaly_nadir_spectral_plus_week.csv"
)
FUTURE_BASELINE = (
    FUTURE_DIRECT_PRED
    / "severity_predictions_current_hurdle_stability_top50_raw_positive_nadir_from_vza_spectral_plus_week_horizon.csv"
)

CURRENT_ROWS = [
    ModelRow("Standard nadir", "Nadir", "selected", "Direct hurdle", CURRENT_BASELINE, 17),
    ModelRow(
        "VZA compact selected",
        "VZA",
        "selected",
        "Direct hurdle",
        CURRENT_PRED
        / "severity_predictions_current_hurdle_stability_top50_raw_positive_compact_anomaly_multiangular_spectral_plus_week.csv",
        80,
    ),
    ModelRow(
        "VZA compact all",
        "VZA",
        "all features",
        "Direct hurdle",
        CURRENT_PRED
        / "severity_predictions_hurdle_probability_times_severity_compact_anomaly_multiangular_spectral_plus_week.csv",
        145,
    ),
    ModelRow(
        "VZA selected",
        "VZA",
        "selected",
        "Direct hurdle",
        CURRENT_RAA_PRED
        / "severity_predictions_current_hurdle_stability_top30_raw_positive_multiangular_vza_spectral_plus_week.csv",
        41,
    ),
    ModelRow(
        "VZA all",
        "VZA",
        "all features",
        "Direct hurdle",
        CURRENT_RAA_PRED
        / "severity_predictions_hurdle_probability_times_severity_multiangular_vza_spectral_plus_week.csv",
        46,
    ),
    ModelRow("RAA only", "RAA", "all/selected", "Not available", None, None, "no true current RAA-only saved model"),
    ModelRow(
        "VZA+RAA selected",
        "VZA+RAA",
        "selected",
        "Direct hurdle",
        CURRENT_RAA_PRED
        / "severity_predictions_current_hurdle_stability_top30_raw_positive_multiangular_vza_raa_spectral_plus_week.csv",
        52,
    ),
    ModelRow(
        "VZA+RAA all",
        "VZA+RAA",
        "all features",
        "Direct hurdle",
        CURRENT_RAA_PRED
        / "severity_predictions_hurdle_probability_times_severity_multiangular_vza_raa_spectral_plus_week.csv",
        206,
    ),
    ModelRow(
        "VZA+phase selected",
        "VZA+phase",
        "selected",
        "Direct hurdle",
        CURRENT_RAA_PRED
        / "severity_predictions_current_hurdle_stability_top30_raw_positive_multiangular_vza_phase_spectral_plus_week.csv",
        54,
    ),
    ModelRow(
        "VZA+phase all",
        "VZA+phase",
        "all features",
        "Direct hurdle",
        CURRENT_RAA_PRED
        / "severity_predictions_hurdle_probability_times_severity_multiangular_vza_phase_spectral_plus_week.csv",
        88,
    ),
    ModelRow(
        "VZA+RAA+phase selected",
        "VZA+RAA+phase",
        "selected",
        "Direct hurdle",
        CURRENT_RAA_PRED
        / "severity_predictions_current_hurdle_stability_top30_raw_positive_multiangular_vza_raa_phase_spectral_plus_week.csv",
        52,
    ),
    ModelRow(
        "VZA+RAA+phase all",
        "VZA+RAA+phase",
        "all features",
        "Direct hurdle",
        CURRENT_RAA_PRED
        / "severity_predictions_hurdle_probability_times_severity_multiangular_vza_raa_phase_spectral_plus_week.csv",
        366,
    ),
]

FUTURE_ROWS = [
    ModelRow("Standard nadir", "Nadir", "selected", "Direct hurdle", FUTURE_BASELINE, 7),
    ModelRow(
        "VZA selected",
        "VZA",
        "selected",
        "Direct hurdle",
        FUTURE_DIRECT_PRED
        / "severity_predictions_current_hurdle_stability_top50_raw_positive_multiangular_vza_spectral_plus_week_horizon.csv",
        47,
    ),
    ModelRow(
        "VZA residual selected",
        "VZA",
        "selected",
        "Ridge + XGBoost residual",
        FUTURE_OUTPUT
        / "results/predictions/severity_predictions_residual_reliability_filtered_xgboost_compact_anomaly_multiangular_spectral_plus_week_horizon.csv",
        43,
    ),
    ModelRow(
        "VZA all",
        "VZA",
        "all features",
        "Huber",
        FUTURE_SELECTION_PRED
        / "severity_predictions_huber_all_features_compact_vza_only_recomputed_spectral_plus_week_horizon.csv",
        146,
    ),
    ModelRow(
        "RAA only subset",
        "RAA",
        "all features",
        "Ridge + XGBoost residual",
        FUTURE_SELECTION_PRED
        / "severity_predictions_oof_raw_raa_all_oof_raw_raa_all_spectral_plus_week_horizon.csv",
        None,
    ),
    ModelRow(
        "VZA+RAA selected",
        "VZA+RAA",
        "selected",
        "Direct hurdle",
        FUTURE_DIRECT_PRED
        / "severity_predictions_current_hurdle_stability_top50_raw_positive_multiangular_vza_raa_spectral_plus_week_horizon.csv",
        84,
    ),
    ModelRow(
        "VZA+RAA residual selected",
        "VZA+RAA",
        "selected",
        "Ridge + XGBoost residual",
        FUTURE_RAA_PRED
        / "severity_predictions_residual_reliability_filtered_xgboost_multiangular_vza_raa_spectral_plus_week_horizon.csv",
        13,
    ),
    ModelRow(
        "VZA+RAA all",
        "VZA+RAA",
        "all features",
        "Ridge + XGBoost residual",
        FUTURE_RAA_PRED
        / "severity_predictions_residual_all_features_xgboost_multiangular_vza_raa_spectral_plus_week_horizon.csv",
        207,
    ),
    ModelRow(
        "VZA+RAA global selected",
        "VZA+RAA global",
        "selected",
        "Ridge + XGBoost residual",
        FUTURE_SELECTION_PRED
        / "severity_predictions_residual_reliability_filtered_xgboost_compact_vza_plus_raa_global_spectral_plus_week_horizon.csv",
        44,
    ),
    ModelRow(
        "VZA+RAA curve selected",
        "VZA+RAA curve",
        "selected",
        "Ridge + XGBoost residual",
        FUTURE_SELECTION_PRED
        / "severity_predictions_residual_reliability_filtered_xgboost_compact_vza_plus_raa_curve_spectral_plus_week_horizon.csv",
        69,
    ),
    ModelRow(
        "VZA+RAA curve all",
        "VZA+RAA curve",
        "all features",
        "Huber",
        FUTURE_SELECTION_PRED
        / "severity_predictions_huber_all_features_compact_vza_plus_raa_curve_spectral_plus_week_horizon.csv",
        182,
    ),
]


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"plot_severity_model_roster_tables_{timestamp}.log"
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


def score_predictions(path: Path) -> dict[str, float | int]:
    frame = pl.read_csv(path)
    y = frame.get_column("y_true").to_numpy().astype(float)
    pred = frame.get_column("y_pred").to_numpy().astype(float)
    rmse = math.sqrt(float(np.mean((y - pred) ** 2)))
    mae = float(np.mean(np.abs(y - pred)))
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2) / denominator) if denominator else float("nan")
    spearman = float(spearmanr(y, pred, nan_policy="omit").correlation)
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "spearman": spearman,
        "n_test_rows": frame.height,
        "n_test_plots": frame.select(pl.col("plot_id").n_unique()).item(),
    }


def bootstrap_rmse_ci(path: Path, *, n_bootstrap: int = 5000, seed: int = 42) -> tuple[float, float]:
    frame = pl.read_csv(path).select(["plot_id", "y_true", "y_pred"])
    groups = [group for _, group in frame.group_by("plot_id", maintain_order=True)]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sampled = rng.integers(0, len(groups), size=len(groups))
        y_sample = np.concatenate(
            [groups[group_idx].get_column("y_true").to_numpy().astype(float) for group_idx in sampled]
        )
        pred_sample = np.concatenate(
            [groups[group_idx].get_column("y_pred").to_numpy().astype(float) for group_idx in sampled]
        )
        boot[idx] = math.sqrt(float(np.mean((y_sample - pred_sample) ** 2)))
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def paired_bootstrap_vs_nadir(
    model_path: Path,
    baseline_path: Path,
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> tuple[float, float]:
    keys = ["plot_id", "predictor_week", "target_week"]
    model = pl.read_csv(model_path).select(keys + ["y_true", "y_pred"])
    baseline = pl.read_csv(baseline_path).select(keys + ["y_pred"])
    joined = model.join(baseline.rename({"y_pred": "y_pred_baseline"}), on=keys, how="inner")
    if joined.height == 0:
        return float("nan"), float("nan")

    y = joined.get_column("y_true").to_numpy().astype(float)
    pred = joined.get_column("y_pred").to_numpy().astype(float)
    baseline_pred = joined.get_column("y_pred_baseline").to_numpy().astype(float)
    observed_delta = float(
        np.sqrt(np.mean((y - baseline_pred) ** 2)) - np.sqrt(np.mean((y - pred) ** 2))
    )

    groups = [group for _, group in joined.group_by("plot_id", maintain_order=True)]
    rng = np.random.default_rng(seed)
    boot_delta = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sampled = rng.integers(0, len(groups), size=len(groups))
        y_sample = np.concatenate(
            [groups[group_idx].get_column("y_true").to_numpy().astype(float) for group_idx in sampled]
        )
        pred_sample = np.concatenate(
            [groups[group_idx].get_column("y_pred").to_numpy().astype(float) for group_idx in sampled]
        )
        baseline_sample = np.concatenate(
            [
                groups[group_idx].get_column("y_pred_baseline").to_numpy().astype(float)
                for group_idx in sampled
            ]
        )
        boot_delta[idx] = np.sqrt(np.mean((y_sample - baseline_sample) ** 2)) - np.sqrt(
            np.mean((y_sample - pred_sample) ** 2)
        )
    p_better = float(np.mean(boot_delta <= 0.0))
    return observed_delta, p_better


def build_table(rows: list[ModelRow], baseline_path: Path) -> pl.DataFrame:
    started = time.perf_counter()
    records = []
    for spec in rows:
        if spec.prediction_path is None:
            records.append(
                {
                    "model": spec.model,
                    "angular_inputs": spec.angular_inputs,
                    "selection": spec.selection,
                    "architecture": spec.architecture,
                    "rmse": None,
                    "rmse_ci_low": None,
                    "rmse_ci_high": None,
                    "delta_rmse_vs_nadir": None,
                    "p_better_than_nadir": None,
                    "mae": None,
                    "r2": None,
                    "spearman": None,
                    "n_features": spec.n_features,
                    "n_test_plots": None,
                    "n_test_rows": None,
                    "prediction_file": None,
                    "note": spec.note,
                }
            )
            continue
        if not spec.prediction_path.exists():
            raise FileNotFoundError(spec.prediction_path)
        scores = score_predictions(spec.prediction_path)
        ci_low, ci_high = bootstrap_rmse_ci(spec.prediction_path)
        if spec.prediction_path == baseline_path:
            delta, p_better = 0.0, None
        else:
            delta, p_better = paired_bootstrap_vs_nadir(spec.prediction_path, baseline_path)
        records.append(
            {
                "model": spec.model,
                "angular_inputs": spec.angular_inputs,
                "selection": spec.selection,
                "architecture": spec.architecture,
                "rmse": scores["rmse"],
                "rmse_ci_low": ci_low,
                "rmse_ci_high": ci_high,
                "delta_rmse_vs_nadir": delta,
                "p_better_than_nadir": p_better,
                "mae": scores["mae"],
                "r2": scores["r2"],
                "spearman": scores["spearman"],
                "n_features": spec.n_features,
                "n_test_plots": scores["n_test_plots"],
                "n_test_rows": scores["n_test_rows"],
                "prediction_file": str(spec.prediction_path.relative_to(ROOT)),
                "note": spec.note,
            }
        )
    out = pl.DataFrame(records)
    log_phase("build model-roster table", started)
    return out


def format_value(value: object, digits: int = 2) -> str:
    if value is None:
        return "NA"
    try:
        if isinstance(value, float) and math.isnan(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def table_for_png(data: pl.DataFrame) -> list[list[str]]:
    rows = []
    for row in data.iter_rows(named=True):
        if row["rmse"] is None:
            rmse_ci = "NA"
        else:
            ci_half_width = (float(row["rmse_ci_high"]) - float(row["rmse_ci_low"])) / 2.0
            rmse_ci = (
                f"{float(row['rmse']):.2f} +- {ci_half_width:.2f}"
            )
        rows.append(
            [
                row["model"],
                row["angular_inputs"],
                row["selection"],
                row["architecture"],
                rmse_ci,
                format_value(row["delta_rmse_vs_nadir"], 2),
                format_value(row["p_better_than_nadir"], 3),
                format_value(row["mae"], 2),
                format_value(row["r2"], 2),
                "NA" if row["n_features"] is None else str(row["n_features"]),
            ]
        )
    return rows


def write_table_figure(data: pl.DataFrame, title: str, output_prefix: Path) -> list[Path]:
    started = time.perf_counter()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "Model",
        "Inputs",
        "Selection",
        "Architecture",
        "RMSE [95% CI]",
        "Delta RMSE\nvs nadir",
        "p better\nthan nadir",
        "MAE",
        "R2",
        "n feat.",
    ]
    body = table_for_png(data)
    fig_height = max(4.8, 1.0 + 0.42 * len(body))
    fig, ax = plt.subplots(figsize=(16, fig_height))
    fig.patch.set_facecolor(PALETTE["white"])
    ax.axis("off")
    ax.set_title(title, fontsize=20, fontweight="bold", color=PALETTE["navy"], pad=12)
    ax.text(
        0.5,
        0.965,
        "Bold = best RMSE; underline = second best; gold row = standard nadir baseline",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.5,
        color=PALETTE["grey"],
    )
    table = ax.table(cellText=body, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1, 1.45)
    widths = [0.17, 0.105, 0.095, 0.165, 0.13, 0.095, 0.085, 0.06, 0.065, 0.055]
    model_names = data.get_column("model").to_list()
    best_row = 1 if model_names else None
    second_best_row = 2 if len(model_names) > 1 else None
    nadir_row = model_names.index("Standard nadir") + 1 if "Standard nadir" in model_names else None
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["light_grey"])
        cell.set_linewidth(0.6)
        if col_idx < len(widths):
            cell.set_width(widths[col_idx])
        if row_idx == 0:
            cell.set_facecolor(PALETTE["navy"])
            cell.set_text_props(color=PALETTE["white"], weight="bold")
        else:
            cell.set_facecolor(PALETTE["white"] if row_idx % 2 else "#f8fafc")
            if col_idx in {0, 1, 2, 3}:
                cell.set_text_props(ha="left")
            if row_idx == nadir_row:
                cell.set_facecolor("#FFF4CC")
                cell.set_edgecolor(PALETTE["gold"])
                cell.set_linewidth(0.9)
            if row_idx == best_row:
                cell.set_text_props(weight="bold")
            if row_idx == second_best_row:
                cell.set_edgecolor(PALETTE["navy"])
                cell.set_linewidth(1.4)
    paths = [output_prefix.with_suffix(ext) for ext in [".png", ".pdf", ".svg"]]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logging.info("Wrote table figure: %s", path)
    plt.close(fig)
    log_phase(f"write {output_prefix.name}", started)
    return paths


def sort_by_rmse(data: pl.DataFrame) -> pl.DataFrame:
    return (
        data.filter(pl.col("rmse").is_not_null() & pl.col("n_features").is_not_null())
        .sort("rmse")
    )


def write_outputs(current: pl.DataFrame, future: pl.DataFrame, log_path: Path) -> list[Path]:
    started = time.perf_counter()
    current = sort_by_rmse(current)
    future = sort_by_rmse(future)
    current_csv = CURRENT_OUTPUT / "results/current_severity_model_roster_table.csv"
    future_csv = FUTURE_OUTPUT / "results/future_severity_model_roster_table.csv"
    current.write_csv(current_csv)
    future.write_csv(future_csv)
    logging.info("Wrote CSV: %s", current_csv)
    logging.info("Wrote CSV: %s", future_csv)

    current_paths = write_table_figure(
        current,
        "Current Severity Prediction Model Roster",
        CURRENT_OUTPUT / "figures/current_severity_model_roster_table",
    )
    future_paths = write_table_figure(
        future,
        "Future Severity Prediction Model Roster",
        FUTURE_OUTPUT / "figures/future_severity_model_roster_table",
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "severity_model_roster_tables_summary.md"
    report = f"""## Results: Severity Model-Roster Tables

Two supplementary model-roster tables were generated for current and future severity prediction.

**Outputs**:
- `{current_csv.relative_to(ROOT)}`
- `{future_csv.relative_to(ROOT)}`
{chr(10).join(f"- `{path.relative_to(ROOT)}`" for path in current_paths + future_paths)}

**Interpretation**: RMSE intervals are absolute plot-level bootstrap 95% CIs for each model. The p-value column is a one-sided paired plot-bootstrap estimate against the matched standard-nadir baseline on overlapping plot/week rows, where smaller values indicate stronger evidence that the model improves RMSE over nadir. The future `RAA only subset` row is not directly comparable with the full-test rows because it has only 4 plots / 28 rows.

**Reproducibility**:
- Train year: `2024`
- Test year: `2025`
- Bootstrap: plot-level resampling, seed=42, n_bootstrap=5000
- Current baseline: `{CURRENT_BASELINE.relative_to(ROOT)}`
- Future baseline: `{FUTURE_BASELINE.relative_to(ROOT)}`
- Log: `{log_path.relative_to(ROOT)}`
"""
    report_path.write_text(report, encoding="utf-8")
    logging.info("Wrote report: %s", report_path)
    log_phase("write model-roster outputs", started)
    return [current_csv, future_csv, *current_paths, *future_paths, report_path]


def main() -> None:
    total_started = time.perf_counter()
    log_path = setup_logging()
    current = build_table(CURRENT_ROWS, CURRENT_BASELINE)
    future = build_table(FUTURE_ROWS, FUTURE_BASELINE)
    write_outputs(current, future, log_path)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
