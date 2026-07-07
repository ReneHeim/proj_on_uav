#!/usr/bin/env python3
"""Design-aware mixture diagnostics for current plot severity.

This script does not replace the reflectance-only comparison. It tests a
structured mixture motivated by the observed failure mode: the compact
multiangular model estimates established disease magnitude well, while the
curve FPCA model ranks low-severity early disease better. Known treatment
status is used only as experimental design metadata.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import markdown_table

OUTPUT_ROOT = ROOT / "outputs/current_severity_design_aware_mixture_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/logs"

DISEASE_2024_CLEAN = ROOT / "outputs/disease/clean_disease_scores_2024.csv"
DISEASE_2025_CLEAN = ROOT / "outputs/disease/clean_disease_scores_2025.csv"
CURRENT_RESULTS_DIR = ROOT / "outputs/current_severity_2024_to_2025/results"
CURVE_RESULTS_DIR = ROOT / "outputs/current_severity_curve_embeddings_2024_to_2025/results"

PREDICTION_FILES = {
    "compact_multiangular": CURRENT_RESULTS_DIR
    / "predictions/severity_predictions_current_hurdle_stability_top50_raw_positive_compact_anomaly_multiangular_spectral_plus_week.csv",
    "curve_fpca": CURVE_RESULTS_DIR
    / "predictions/severity_predictions_current_hurdle_stability_top30_raw_positive_curve_embedding_fpca_only_spectral_plus_week.csv",
    "nadir": CURRENT_RESULTS_DIR
    / "predictions/severity_predictions_current_hurdle_top20_raw_positive_compact_anomaly_nadir_spectral_plus_week.csv",
    "all_feature_multiangular": CURRENT_RESULTS_DIR
    / "predictions/severity_predictions_hurdle_probability_times_severity_compact_anomaly_multiangular_spectral_plus_week.csv",
}

TREATED_FLOOR_MAX_SEVERITY = 0.0
EARLY_UNTREATED_MEAN_MAX_SEVERITY = 1.0


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"analyze_current_severity_design_aware_mixture_2024_to_2025_{timestamp}.log"
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


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return math.nan
    value = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    return float(value) if value is not None else math.nan


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else math.nan,
        "spearman": safe_spearman(y_true, y_pred),
    }


def load_prediction_frame() -> pd.DataFrame:
    started = time.perf_counter()
    frames = []
    for name, path in PREDICTION_FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frames.append(
            frame[["plot_id", "predictor_week", "target_week", "y_true", "y_pred"]].rename(
                columns={"y_pred": name}
            )
        )
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(
            frame,
            on=["plot_id", "predictor_week", "target_week", "y_true"],
            how="inner",
        )
    disease_2025 = pd.read_csv(DISEASE_2025_CLEAN)[
        ["plot_id", "week", "cult", "trt", "block", "ino", "ds_leaf_mean", "ds_leaf_sd"]
    ]
    merged = merged.merge(
        disease_2025,
        left_on=["plot_id", "target_week"],
        right_on=["plot_id", "week"],
        how="left",
    ).drop(columns=["week"])
    log_phase("load prediction files and 2025 metadata", started)
    return merged


def training_design_rules() -> tuple[pd.DataFrame, set[int], set[int]]:
    disease_2024 = pd.read_csv(DISEASE_2024_CLEAN)
    treated = (
        disease_2024.loc[disease_2024["trt"].eq("trt")]
        .groupby("week", as_index=False)
        .agg(
            treated_max_severity=("ds_plot", "max"),
            treated_mean_severity=("ds_plot", "mean"),
            treated_n=("ds_plot", "size"),
        )
    )
    untreated = (
        disease_2024.loc[disease_2024["trt"].eq("no_trt")]
        .groupby("week", as_index=False)
        .agg(
            untreated_mean_severity=("ds_plot", "mean"),
            untreated_max_severity=("ds_plot", "max"),
            untreated_n=("ds_plot", "size"),
        )
    )
    rules = treated.merge(untreated, on="week", how="outer").sort_values("week")
    treatment_floor_weeks = set(
        rules.loc[
            rules["treated_max_severity"].fillna(np.inf) <= TREATED_FLOOR_MAX_SEVERITY,
            "week",
        ]
        .astype(int)
        .tolist()
    )
    curve_expert_weeks = set(
        rules.loc[
            rules["untreated_mean_severity"].fillna(np.inf) <= EARLY_UNTREATED_MEAN_MAX_SEVERITY,
            "week",
        ]
        .astype(int)
        .tolist()
    )
    rules["use_treatment_floor_for_treated"] = rules["week"].isin(treatment_floor_weeks)
    rules["use_curve_expert"] = rules["week"].isin(curve_expert_weeks)
    return rules, treatment_floor_weeks, curve_expert_weeks


def build_mixture_predictions(
    predictions: pd.DataFrame,
    treatment_floor_weeks: set[int],
    curve_expert_weeks: set[int],
) -> pd.DataFrame:
    out = predictions.copy()
    use_curve = out["target_week"].isin(curve_expert_weeks)
    use_floor = out["trt"].eq("trt") & out["target_week"].isin(treatment_floor_weeks)
    out["mixture_curve_early_compact_late"] = np.where(
        use_curve, out["curve_fpca"], out["compact_multiangular"]
    )
    out["design_aware_treatment_floor_compact"] = np.where(
        use_floor, 0.0, out["compact_multiangular"]
    )
    out["design_aware_treatment_floor_curve"] = np.where(use_floor, 0.0, out["curve_fpca"])
    out["design_aware_treatment_floor_mixture"] = np.where(
        use_floor, 0.0, out["mixture_curve_early_compact_late"]
    )
    out["simple_average_compact_curve"] = 0.5 * out["compact_multiangular"] + 0.5 * out["curve_fpca"]
    out["design_aware_average_floor"] = np.where(use_floor, 0.0, out["simple_average_compact_curve"])
    out["expert_used"] = np.where(use_curve, "curve_fpca", "compact_multiangular")
    out["treatment_floor_applied"] = use_floor
    return out


def model_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    model_cols = [
        "compact_multiangular",
        "curve_fpca",
        "nadir",
        "all_feature_multiangular",
        "mixture_curve_early_compact_late",
        "design_aware_treatment_floor_compact",
        "design_aware_treatment_floor_curve",
        "design_aware_treatment_floor_mixture",
        "simple_average_compact_curve",
        "design_aware_average_floor",
    ]
    rows = []
    y = predictions["y_true"].to_numpy(float)
    for col in model_cols:
        rows.append(
            {
                "model": col,
                "uses_design_metadata": col.startswith("design_aware"),
                **score(y, predictions[col].to_numpy(float)),
            }
        )
    return pd.DataFrame(rows).sort_values(["rmse", "model"])


def grouped_error_summary(predictions: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    model_cols = [
        "compact_multiangular",
        "curve_fpca",
        "design_aware_treatment_floor_mixture",
        "nadir",
    ]
    rows = []
    for keys, group in predictions.groupby(by, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(by, keys, strict=False))
        y = group["y_true"].to_numpy(float)
        for model in model_cols:
            pred = group[model].to_numpy(float)
            rows.append(
                {
                    **base,
                    "model": model,
                    "n": len(group),
                    "mean_observed": float(np.mean(y)),
                    "mean_predicted": float(np.mean(pred)),
                    "bias": float(np.mean(pred - y)),
                    **score(y, pred),
                }
            )
    return pd.DataFrame(rows).sort_values(by + ["rmse", "model"])


def write_report(
    comparison: pd.DataFrame,
    rules: pd.DataFrame,
    week_summary: pd.DataFrame,
    week_trt_summary: pd.DataFrame,
    paths: dict[str, Path],
    log_path: Path,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "current_severity_design_aware_mixture_summary.md"
    lines = [
        "## Results: Design-Aware Mixture For Current Severity",
        "",
        "This diagnostic model combines two complementary reflectance experts: curve FPCA for early low-severity ranking and compact multiangular bins for established disease magnitude. It also applies a treatment floor only for weeks where treated plots had near-zero severity in the 2024 training data.",
        "",
        "### Model Comparison",
        "",
        markdown_table(comparison.round(4), max_rows=20),
        "",
        "### Rules Learned From 2024 Training Data",
        "",
        markdown_table(rules.round(4), max_rows=20),
        "",
        "### Error By Week",
        "",
        markdown_table(week_summary.round(4), max_rows=40),
        "",
        "### Error By Week And Treatment",
        "",
        markdown_table(week_trt_summary.round(4), max_rows=80),
        "",
        "### Interpretation",
        "",
        "The reflectance-only compact multiangular model estimates severe week-5 magnitude well but misses early week-3 treated-versus-untreated ranking. The curve FPCA model ranks early disease better but underestimates established week-5 severity. The mixture uses each model in the phase where it is strongest and uses known treatment metadata as an experimental-design prior.",
        "",
        "This should be reported separately from reflectance-only results because treatment status is not a spectral measurement.",
        "",
        "### Reproducibility",
        "",
        "- Training year for design rules: 2024.",
        "- Test year: 2025.",
        f"- Treated floor threshold: max treated severity <= {TREATED_FLOOR_MAX_SEVERITY}.",
        f"- Early curve-expert threshold: mean untreated severity <= {EARLY_UNTREATED_MEAN_MAX_SEVERITY}.",
        f"- Log: `{log_path}`",
        "",
        "### Outputs",
        "",
    ]
    lines.extend([f"- {label}: `{path}`" for label, path in paths.items()])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    total = time.perf_counter()
    for directory in [RESULTS_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging()
    predictions = load_prediction_frame()
    rules, treatment_floor_weeks, curve_expert_weeks = training_design_rules()
    logging.info("Treatment-floor weeks from 2024: %s", sorted(treatment_floor_weeks))
    logging.info("Curve-expert weeks from 2024: %s", sorted(curve_expert_weeks))
    predictions = build_mixture_predictions(predictions, treatment_floor_weeks, curve_expert_weeks)
    comparison = model_comparison(predictions)
    week_summary = grouped_error_summary(predictions, ["target_week"])
    week_trt_summary = grouped_error_summary(predictions, ["target_week", "trt"])
    paths = {
        "model_comparison": RESULTS_DIR / "design_aware_mixture_model_comparison.csv",
        "predictions": RESULTS_DIR / "design_aware_mixture_predictions.csv",
        "training_rules": RESULTS_DIR / "design_aware_mixture_training_rules.csv",
        "week_summary": RESULTS_DIR / "design_aware_mixture_week_summary.csv",
        "week_treatment_summary": RESULTS_DIR / "design_aware_mixture_week_treatment_summary.csv",
    }
    comparison.to_csv(paths["model_comparison"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    rules.to_csv(paths["training_rules"], index=False)
    week_summary.to_csv(paths["week_summary"], index=False)
    week_trt_summary.to_csv(paths["week_treatment_summary"], index=False)
    report_path = write_report(comparison, rules, week_summary, week_trt_summary, paths, log_path)
    logging.info("Report: %s", report_path)
    log_phase("total", total)


if __name__ == "__main__":
    main()
