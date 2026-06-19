"""Report no-week model comparisons for external 2025 testing."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/cross_year_generalization_2024_to_2025"
RESULTS_DIR = BASE / "results"
FIGURES_DIR = BASE / "figures"
REPORTS_DIR = BASE / "reports"
LOGS_DIR = ROOT / "outputs/logs"


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"report_no_week_model_summary_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info("Log file: %s", log_path)
    return log_path


def log_phase(name: str, t0: float) -> None:
    logging.info("[PHASE] %s: %.1fs", name, time.time() - t0)


def pretty_feature(value: str) -> str:
    labels = {
        "nadir": "Nadir",
        "multiangular_vza": "VZA",
        "multiangular_vza_compact": "VZA compact",
        "multiangular_geometry_compact": "Geometry compact",
        "multiangular_vza_raa": "VZA + RAA",
        "multiangular_vza_phase": "VZA + phase",
    }
    return labels.get(value, value)


def markdown_table(df: pd.DataFrame, float_digits: int = 3) -> str:
    columns = list(df.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float | np.floating):
                values.append("" if pd.isna(value) else f"{value:.{float_digits}f}")
            else:
                values.append("" if pd.isna(value) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def build_early_warning_table() -> pd.DataFrame:
    logistic = pd.read_csv(RESULTS_DIR / "warning_external_2024_train_2025_test.csv")
    logistic = logistic.assign(
        model_family="Logistic",
        f1=logistic["f1"],
        auroc=logistic["auroc"],
        auprc=logistic["auprc"],
        precision=logistic["precision"],
        recall=logistic["recall"],
        balanced_accuracy=logistic["balanced_accuracy"],
    )

    xgb = pd.read_csv(RESULTS_DIR / "xgboost_warning_train_eval_2024_test_2025.csv")
    xgb = xgb.assign(
        model_family="XGBoost",
        f1=xgb["external_f1_2025"],
        auroc=xgb["external_auroc_2025"],
        auprc=xgb["external_auprc_2025"],
        precision=xgb["external_precision_2025"],
        recall=xgb["external_recall_2025"],
        balanced_accuracy=xgb["external_balanced_accuracy_2025"],
    )

    table = pd.concat([logistic, xgb], ignore_index=True)
    keep = ["model_family", "feature_set", "f1", "precision", "recall", "balanced_accuracy", "auroc", "auprc"]
    table = table[keep].copy()
    table["feature_label"] = table["feature_set"].map(pretty_feature)
    return table.sort_values("f1", ascending=False)


def build_severity_table() -> pd.DataFrame:
    ridge = pd.read_csv(RESULTS_DIR / "severity_external_2024_train_2025_test.csv")
    ridge = ridge[ridge["covariates"] == "spectral_only"].copy()
    ridge["model_family"] = np.where(ridge["model"] == "log_severity", "Ridge log", "Ridge raw")
    ridge = ridge.rename(
        columns={
            "rmse": "rmse",
            "mae": "mae",
            "r2": "r2",
            "spearman": "spearman",
        }
    )

    xgb = pd.read_csv(RESULTS_DIR / "xgboost_severity_train_eval_2024_test_2025.csv")
    xgb = xgb[xgb["covariates"] == "spectral_only"].copy()
    xgb["model_family"] = "XGBoost early stop"
    xgb = xgb.rename(
        columns={
            "external_rmse_2025": "rmse",
            "external_mae_2025": "mae",
            "external_r2_2025": "r2",
            "external_spearman_2025": "spearman",
        }
    )

    xgb_200_path = RESULTS_DIR / "xgboost_severity_200_trees_train_eval_2024_test_2025.csv"
    frames = [ridge, xgb]
    if xgb_200_path.exists():
        xgb_200 = pd.read_csv(xgb_200_path)
        xgb_200 = xgb_200[xgb_200["covariates"] == "spectral_only"].copy()
        xgb_200["model_family"] = "XGBoost 200 trees"
        xgb_200 = xgb_200.rename(
            columns={
                "external_rmse_2025": "rmse",
                "external_mae_2025": "mae",
                "external_r2_2025": "r2",
                "external_spearman_2025": "spearman",
            }
        )
        frames.append(xgb_200)

    table = pd.concat(frames, ignore_index=True)
    keep = ["model_family", "feature_set", "rmse", "mae", "r2", "spearman"]
    table = table[keep].copy()
    table["feature_label"] = table["feature_set"].map(pretty_feature)
    return table.sort_values("rmse")


def plot_early_warning_f1(table: pd.DataFrame, output_path: Path) -> None:
    t0 = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot = table.copy()
    plot["label"] = plot["model_family"] + "\n" + plot["feature_label"]
    colors = plot["model_family"].map({"Logistic": "#2f6f9f", "XGBoost": "#d08336"}).fillna("#777777")

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    bars = ax.bar(np.arange(len(plot)), plot["f1"], color=colors, edgecolor="white", linewidth=1.1)
    nadir_f1 = float(plot[(plot["model_family"] == "Logistic") & (plot["feature_set"] == "nadir")]["f1"].iloc[0])
    ax.axhline(nadir_f1, color="#323232", linestyle="--", linewidth=1.1, label=f"Logistic nadir F1 = {nadir_f1:.2f}")
    for bar, value in zip(bars, plot["f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("External early-warning F1 without week covariates", fontsize=14, fontweight="bold")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, max(0.82, plot["f1"].max() + 0.09))
    ax.set_xticks(np.arange(len(plot)))
    ax.set_xticklabels(plot["label"], rotation=45, ha="right")
    ax.grid(axis="y", color="#d7d0c6", linewidth=0.8, alpha=0.75)
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("#fbf7ef")
    ax.set_facecolor("#fbf7ef")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log_phase("plot early warning F1", t0)


def plot_severity_no_week(table: pd.DataFrame, output_path: Path) -> None:
    t0 = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot = table.head(12).copy()
    plot["label"] = plot["model_family"] + "\n" + plot["feature_label"]
    colors = plot["model_family"].map(
        {
            "Ridge log": "#4f7f58",
            "Ridge raw": "#7b9e87",
            "XGBoost early stop": "#d08336",
            "XGBoost 200 trees": "#a75d2a",
        }
    ).fillna("#777777")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.8))
    y = np.arange(len(plot))
    bars = axes[0].barh(y, plot["rmse"], color=colors, edgecolor="white", linewidth=1.1)
    for bar, value in zip(bars, plot["rmse"]):
        axes[0].text(value + 0.12, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", ha="left", va="center", fontsize=9)
    axes[0].set_title("Severity RMSE", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("RMSE")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(plot["label"])
    axes[0].invert_yaxis()

    axes[1].barh(y, plot["spearman"], color=colors, edgecolor="white", linewidth=1.1)
    for i, value in enumerate(plot["spearman"]):
        axes[1].text(value + 0.015, i, f"{value:.2f}", ha="left", va="center", fontsize=9)
    axes[1].set_title("Severity rank correlation", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Spearman")
    axes[1].set_xlim(min(-0.05, plot["spearman"].min() - 0.05), max(0.75, plot["spearman"].max() + 0.08))
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()

    for ax in axes:
        ax.grid(axis="x", color="#d7d0c6", linewidth=0.8, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#fbf7ef")
    fig.suptitle("External severity prediction without week covariates: top 12 models by RMSE", fontsize=15, fontweight="bold")
    fig.patch.set_facecolor("#fbf7ef")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log_phase("plot severity no-week", t0)


def write_report(outputs: dict[str, Path], warning: pd.DataFrame, severity: pd.DataFrame, log_path: Path) -> None:
    report = [
        "## Results: No-Week External Model Summary",
        "",
        "These tables and figures exclude week/horizon covariates. They test whether reflectance features alone transfer from 2024 to 2025.",
        "",
        "### Early-warning F1",
        "",
        markdown_table(warning[["model_family", "feature_label", "f1", "precision", "recall", "balanced_accuracy", "auroc", "auprc"]].round(4)),
        "",
        "### Severity Without Week",
        "",
        markdown_table(severity[["model_family", "feature_label", "rmse", "mae", "r2", "spearman"]].round(4)),
        "",
        "**Interpretation**: Without week covariates, severity calibration is weak, but multiangular features improve some models over nadir. The strongest no-week result is early-warning classification, especially XGBoost VZA F1.",
        "",
        "### Outputs",
        "",
    ]
    for label, path in outputs.items():
        report.append(f"- {label}: `{path}`")
    report.append(f"- log: `{log_path}`")
    outputs["report"].write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging()

    t0 = time.time()
    warning = build_early_warning_table()
    severity = build_severity_table()
    log_phase("build tables", t0)

    outputs = {
        "early_warning_table": RESULTS_DIR / "table_early_warning_f1_no_week.csv",
        "severity_table": RESULTS_DIR / "table_severity_no_week.csv",
        "early_warning_figure": FIGURES_DIR / "early_warning_f1_no_week_model_comparison.png",
        "severity_figure": FIGURES_DIR / "severity_no_week_model_comparison.png",
        "report": REPORTS_DIR / "no_week_model_summary.md",
    }
    warning.to_csv(outputs["early_warning_table"], index=False)
    severity.to_csv(outputs["severity_table"], index=False)
    plot_early_warning_f1(warning, outputs["early_warning_figure"])
    plot_severity_no_week(severity, outputs["severity_figure"])
    write_report(outputs, warning, severity, log_path)
    log_phase("total", start)


if __name__ == "__main__":
    main()
