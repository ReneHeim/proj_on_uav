"""Explore 2024-only feature selection for weak early-warning model families.

This script does not modify the frozen severity pipeline or manuscript tables.
It asks whether Logistic VZA and XGBoost VZA+phase improve when feature
selection and threshold calibration are learned using 2024 only, then evaluated
once on 2025.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_cross_year_generalization_2024_to_2025 import (  # noqa: E402
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

OUTPUT_ROOT = (
    ROOT / "outputs/cross_year_generalization_2024_to_2025/early_warning_feature_selection"
)
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/logs"

TARGET_FEATURE_SETS = ["multiangular_vza", "multiangular_vza_phase", "multiangular_vza_raa"]
TARGET_MODEL_FEATURES = {
    ("logistic", "multiangular_vza"),
    ("logistic", "multiangular_vza_phase"),
    ("logistic", "multiangular_vza_raa"),
    ("xgboost", "multiangular_vza_phase"),
    ("xgboost", "multiangular_vza_raa"),
}
STABILITY_REPEATS = 30
STABILITY_SPLITS = 4
STABILITY_THRESHOLD = 0.35


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"explore_early_warning_feature_selection_{timestamp}.log"
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


def safe_name(value: str) -> str:
    return value.replace("+", "plus").replace(" ", "_").replace("/", "_").lower()


def read_inputs() -> tuple[dict[str, object], dict[str, object]]:
    t0 = time.time()
    features_2024 = load_feature_sets_for_year(VZA_2024, RAA_2024)
    features_2025 = load_feature_sets_for_year(VZA_2025, RAA_2025)
    disease_2024 = load_2024_disease_with_fallback()
    disease_2025, _audit = load_2025_disease_with_fallback()
    train_tables = {
        name: build_model_table(features_2024[name], disease_2024) for name in TARGET_FEATURE_SETS
    }
    test_tables = {
        name: build_model_table(features_2025[name], disease_2025) for name in TARGET_FEATURE_SETS
    }
    log_phase("load features and disease tables", t0)
    return train_tables, test_tables


def stable_features_l1(train, cols: list[str], y: np.ndarray, groups: np.ndarray) -> pl.DataFrame:
    """Estimate feature stability from repeated grouped CV using L1 logistic models."""
    t0 = time.time()
    counts = {col: 0 for col in cols}
    total_fits = 0
    for repeat in range(STABILITY_REPEATS):
        splitter = StratifiedGroupKFold(
            n_splits=STABILITY_SPLITS, shuffle=True, random_state=SEED + repeat
        )
        for train_idx, _valid_idx in splitter.split(train[cols], y, groups=groups):
            if len(np.unique(y[train_idx])) < 2:
                continue
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "logreg",
                        LogisticRegression(
                            penalty="l1",
                            C=0.25,
                            solver="liblinear",
                            class_weight="balanced",
                            max_iter=3000,
                            random_state=SEED + repeat,
                        ),
                    ),
                ]
            )
            model.fit(train.iloc[train_idx][cols], y[train_idx])
            coef = model.named_steps["logreg"].coef_[0]
            for col, value in zip(cols, coef, strict=False):
                if abs(value) > 1e-9:
                    counts[col] += 1
            total_fits += 1
    if total_fits == 0:
        raise RuntimeError("No valid stability-selection fits were completed.")
    stability = (
        pl.DataFrame(
            {
                "feature": list(counts),
                "selected_fits": list(counts.values()),
            }
        )
        .with_columns(
            (pl.col("selected_fits") / total_fits).alias("selection_frequency"),
            pl.lit(total_fits).alias("total_fits"),
        )
        .sort("selection_frequency", descending=True)
    )
    log_phase("L1 stability selection", t0)
    return stability


def selected_feature_sets(stability: pl.DataFrame, cols: list[str]) -> dict[str, list[str]]:
    top_frequency = (
        stability.filter(pl.col("selection_frequency") >= STABILITY_THRESHOLD)
        .get_column("feature")
        .to_list()
    )
    if len(top_frequency) < 3:
        top_frequency = stability.head(min(10, len(cols))).get_column("feature").to_list()
    return {
        "all_features": cols,
        "stability_selected": top_frequency,
        "top_10": stability.head(min(10, len(cols))).get_column("feature").to_list(),
        "top_20": stability.head(min(20, len(cols))).get_column("feature").to_list(),
    }


def make_logistic() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegressionCV(
                    Cs=10,
                    cv=3,
                    class_weight="balanced",
                    max_iter=5000,
                    scoring="average_precision",
                    random_state=SEED,
                    solver="liblinear",
                ),
            ),
        ]
    )


def make_xgboost(y_train: np.ndarray) -> Pipeline:
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / positives if positives else 1.0
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "xgb",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    n_estimators=200,
                    learning_rate=0.025,
                    max_depth=2,
                    min_child_weight=5,
                    subsample=0.85,
                    colsample_bytree=0.75,
                    reg_alpha=0.1,
                    reg_lambda=8.0,
                    scale_pos_weight=scale_pos_weight,
                    random_state=SEED,
                    tree_method="hist",
                    n_jobs=4,
                ),
            ),
        ]
    )


def evaluate_model(
    model_name: str, train, test, cols: list[str], feature_set: str, selection_name: str
) -> tuple[list[dict], pl.DataFrame]:
    y_train = train[WARNING_TARGET].to_numpy(int)
    y_test = test[WARNING_TARGET].to_numpy(int)
    groups = train["plot_id"].to_numpy()
    estimator = make_logistic() if model_name == "logistic" else make_xgboost(y_train)

    t0 = time.time()
    thresholds = grouped_cv_thresholds(clone(estimator), train[cols], y_train, groups)
    log_phase(f"calibrate thresholds {model_name} {feature_set} {selection_name}", t0)

    fit_t0 = time.time()
    estimator.fit(train[cols], y_train)
    fit_time = time.time() - fit_t0

    pred_t0 = time.time()
    prob = estimator.predict_proba(test[cols])[:, 1]
    predict_time = time.time() - pred_t0

    rows = []
    pred_rows = []
    for strategy, threshold in thresholds.items():
        pred = (prob >= threshold).astype(int)
        metrics = binary_metrics(y_test, pred, prob)
        rows.append(
            {
                "model": model_name,
                "feature_set": feature_set,
                "selection": selection_name,
                "threshold_strategy": strategy,
                "threshold": float(threshold),
                "n_features": len(cols),
                "n_train": len(train),
                "n_test": len(test),
                "n_positive_test": int(y_test.sum()),
                "fit_time_s": fit_time,
                "predict_time_s": predict_time,
                **metrics,
            }
        )
        for plot_id, predictor_week, target_week, y_true, y_prob, y_pred in zip(
            test["plot_id"],
            test["predictor_week"],
            test["target_week"],
            y_test,
            prob,
            pred,
            strict=False,
        ):
            pred_rows.append(
                {
                    "model": model_name,
                    "feature_set": feature_set,
                    "selection": selection_name,
                    "threshold_strategy": strategy,
                    "threshold": float(threshold),
                    "plot_id": plot_id,
                    "predictor_week": predictor_week,
                    "target_week": target_week,
                    "y_true": int(y_true),
                    "y_prob": float(y_prob),
                    "y_pred": int(y_pred),
                }
            )
    return rows, pl.DataFrame(pred_rows)


def markdown_table(df: pl.DataFrame) -> str:
    if df.is_empty():
        return ""
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.iter_rows(named=True):
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def write_report(results: pl.DataFrame, stability_paths: list[Path], log_path: Path) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "early_warning_feature_selection_summary.md"
    best = (
        results.filter(pl.col("threshold_strategy").is_in(["fixed_0_5", "balanced", "high_recall"]))
        .sort("f1", descending=True)
        .select(
            [
                "model",
                "feature_set",
                "selection",
                "threshold_strategy",
                "n_features",
                "f1",
                "precision",
                "recall",
                "specificity",
                "balanced_accuracy",
                "auroc",
                "auprc",
            ]
        )
        .head(12)
        .with_columns(pl.selectors.numeric().round(3))
    )
    report = [
        "# Early-Warning Feature Selection Exploration",
        "",
        "## Results",
        "",
        markdown_table(best),
        "",
        "**Interpretation**: This is an exploratory 2024-only feature-selection check for two weak model families. "
        "The selection and threshold rules are learned from 2024 only, then evaluated on the 2025 external rows. "
        "Use these results to decide whether a model is worth adding to the supplement; do not replace the main table without freezing the rule.",
        "",
        "## Outputs",
        "",
        f"- Results CSV: `{RESULTS_DIR / 'early_warning_feature_selection_results.csv'}`",
        f"- Predictions CSV: `{RESULTS_DIR / 'early_warning_feature_selection_predictions.csv'}`",
        *[f"- Stability CSV: `{path}`" for path in stability_paths],
        f"- Log: `{log_path}`",
        "",
        "## Reproducibility",
        "",
        f"- Seed: `{SEED}`",
        f"- Stability repeats: `{STABILITY_REPEATS}`",
        f"- Stability grouped CV splits: `{STABILITY_SPLITS}`",
        f"- Stability threshold: `{STABILITY_THRESHOLD}`",
        "- Train year: `2024`",
        "- External test year: `2025`",
        "- Target: `future_warning_ds_ge_5`",
    ]
    report_path.write_text("\n".join(report) + "\n")
    return report_path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging()
    script_t0 = time.time()

    train_tables, test_tables = read_inputs()
    all_results = []
    all_predictions = []
    stability_paths: list[Path] = []

    for feature_set in TARGET_FEATURE_SETS:
        cols, train, test = align_train_test(train_tables[feature_set], test_tables[feature_set])
        y_train = train[WARNING_TARGET].to_numpy(int)
        groups = train["plot_id"].to_numpy()
        stability = stable_features_l1(train, cols, y_train, groups)
        stability_path = RESULTS_DIR / f"stability_{safe_name(feature_set)}.csv"
        stability.write_csv(stability_path)
        stability_paths.append(stability_path)
        selections = selected_feature_sets(stability, cols)

        for model_name in ["logistic", "xgboost"]:
            if (model_name, feature_set) not in TARGET_MODEL_FEATURES:
                continue
            for selection_name, selected_cols in selections.items():
                rows, predictions = evaluate_model(
                    model_name, train, test, selected_cols, feature_set, selection_name
                )
                all_results.extend(rows)
                all_predictions.append(predictions)

    results = pl.DataFrame(all_results)
    predictions = pl.concat(all_predictions, how="vertical")
    results_path = RESULTS_DIR / "early_warning_feature_selection_results.csv"
    predictions_path = RESULTS_DIR / "early_warning_feature_selection_predictions.csv"
    results.write_csv(results_path)
    predictions.write_csv(predictions_path)
    report_path = write_report(results, stability_paths, log_path)

    logging.info("Results written: %s", results_path)
    logging.info("Predictions written: %s", predictions_path)
    logging.info("Report written: %s", report_path)
    log_phase("total script", script_t0)


if __name__ == "__main__":
    main()
