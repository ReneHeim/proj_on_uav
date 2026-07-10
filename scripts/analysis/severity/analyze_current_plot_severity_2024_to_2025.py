#!/usr/bin/env python3
"""Train on 2024 and test on 2025 for current plot-level severity.

This mirrors the compact nadir-vs-multiangular severity comparison, but the
target is same-week `ds_plot` rather than the next later disease observation.
"""

from __future__ import annotations

import logging
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.severity import (
    debug_multiangular_rmse_bottleneck as residual_pipeline,
)
from scripts.analysis.severity.analyze_cross_year_generalization_2024_to_2025 import (
    ALPHAS,
    SEED,
    TARGET,
    TARGET_LOG,
    WARNING_TARGET,
    WARNING_THRESHOLD,
)

OUTPUT_ROOT = ROOT / "outputs/runs/analysis/severity/current/2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"
DISEASE_2024_CLEAN = ROOT / "outputs/shared/disease/clean_disease_scores_2024.csv"
DISEASE_2025_CLEAN = ROOT / "outputs/shared/disease/clean_disease_scores_2025.csv"

MODEL_FUNCTIONS = [
    residual_pipeline.fit_direct_stability_ridge,
    residual_pipeline.fit_phenology_floor_ridge,
    residual_pipeline.fit_hurdle_model,
    "current_hurdle_top20_raw",
    "current_hurdle_top40_raw",
    "current_hurdle_stability_top30_raw",
    "current_hurdle_stability_top50_raw",
    "current_hurdle_top20_log",
    "current_hurdle_top40_log",
    residual_pipeline.fit_reliability_filtered_ridge,
    residual_pipeline.fit_residual_xgboost_stability,
    residual_pipeline.fit_residual_reliability_filtered_xgboost,
]

CURRENT_STABILITY_REPEATS = 20
CURRENT_STABILITY_TEST_SIZE = 0.25


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"analyze_current_plot_severity_2024_to_2025_{timestamp}.log"
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


def configure_reused_pipeline_paths() -> None:
    residual_pipeline.ROOT = ROOT
    residual_pipeline.INPUT_RESULTS_DIR = (
        ROOT / "outputs/runs/analysis/severity/future/compact_distribution_feature_family/results"
    )
    residual_pipeline.FEATURE_SETS = [
        "compact_anomaly_nadir",
        "compact_anomaly_multiangular",
        "compact_anomaly_multiangular_shape",
    ]
    residual_pipeline.COVARIATES = "spectral_plus_week"
    residual_pipeline.OUTPUT_ROOT = OUTPUT_ROOT
    residual_pipeline.RESULTS_DIR = RESULTS_DIR
    residual_pipeline.REPORTS_DIR = REPORTS_DIR
    residual_pipeline.FIGURES_DIR = OUTPUT_ROOT / "figures"
    residual_pipeline.PREDICTIONS_DIR = PREDICTIONS_DIR
    residual_pipeline.FROZEN_MANIFEST_PATH = OUTPUT_ROOT / "current_severity_pipeline_manifest.json"


def load_clean_disease_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.perf_counter()
    if not DISEASE_2024_CLEAN.exists():
        raise FileNotFoundError(DISEASE_2024_CLEAN)
    if not DISEASE_2025_CLEAN.exists():
        raise FileNotFoundError(DISEASE_2025_CLEAN)
    disease_2024 = pd.read_csv(DISEASE_2024_CLEAN)
    disease_2025 = pd.read_csv(DISEASE_2025_CLEAN)
    log_phase("read clean disease CSVs", t0)
    return disease_2024, disease_2025


def build_current_model_table(features: pd.DataFrame, disease: pd.DataFrame) -> pd.DataFrame:
    """Pair plot-week reflectance features with same-week DSDI plot severity."""
    disease_target = disease[["plot_id", "week", "ds_plot"]].rename(columns={"ds_plot": TARGET})
    table = features.merge(disease_target, on=["plot_id", "week"], how="inner")
    table["predictor_week"] = table["week"].astype(int)
    table["target_week"] = table["week"].astype(int)
    table[TARGET_LOG] = np.log1p(table[TARGET])
    table[WARNING_TARGET] = (table[TARGET] >= WARNING_THRESHOLD).astype(int)
    table = table.sort_values(["target_week", "plot_id"]).reset_index(drop=True)
    return table


def ranked_top_columns(
    train_aligned: pd.DataFrame,
    cols: list[str],
    target: np.ndarray,
    top_k: int,
    force_cols: set[str] | None = None,
) -> list[str]:
    force_cols = force_cols or set()
    numeric = train_aligned[cols].copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    scores = []
    for col in cols:
        if col in force_cols:
            continue
        x = numeric[col].to_numpy(float)
        if np.nanstd(x) == 0 or np.nanstd(target) == 0:
            score = 0.0
        else:
            score = abs(float(np.corrcoef(x, target)[0, 1]))
            if not np.isfinite(score):
                score = 0.0
        scores.append((score, col))
    ranked = [col for _, col in sorted(scores, reverse=True)]
    selected = [col for col in cols if col in force_cols]
    selected.extend([col for col in ranked if col not in selected][: max(0, top_k - len(selected))])
    return selected or list(cols)


def stable_ranked_columns(
    train_aligned: pd.DataFrame,
    cols: list[str],
    target: np.ndarray,
    groups: np.ndarray,
    top_k: int,
    force_cols: set[str] | None = None,
    role: str = "feature",
) -> tuple[list[str], pd.DataFrame]:
    """Rank features with grouped repeated ElasticNet selection on 2024 only."""
    force_cols = force_cols or set()
    splitter = GroupShuffleSplit(
        n_splits=CURRENT_STABILITY_REPEATS,
        test_size=CURRENT_STABILITY_TEST_SIZE,
        random_state=SEED,
    )
    counts = pd.Series(0.0, index=cols)
    abs_coef_sum = pd.Series(0.0, index=cols)
    fit_rows = []
    for repeat, (fit_idx, _) in enumerate(splitter.split(train_aligned, target, groups=groups)):
        fit_part = train_aligned.iloc[fit_idx].copy()
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "elasticnet",
                    ElasticNetCV(
                        l1_ratio=residual_pipeline.STABILITY_L1_RATIOS,
                        alphas=residual_pipeline.STABILITY_ALPHAS,
                        cv=3,
                        max_iter=5000,
                        random_state=SEED + repeat,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipeline.fit(fit_part[cols], np.asarray(target)[fit_idx])
        coefs = np.abs(np.asarray(pipeline.named_steps["elasticnet"].coef_, dtype=float))
        selected = coefs > 1e-8
        counts += selected.astype(float)
        abs_coef_sum += coefs
        fit_rows.append(
            {
                "alpha": float(pipeline.named_steps["elasticnet"].alpha_),
                "l1_ratio": float(pipeline.named_steps["elasticnet"].l1_ratio_),
                "selected_features": int(selected.sum()),
            }
        )
    ranking = pd.DataFrame(
        {
            "role": role,
            "feature": cols,
            "selection_frequency": (counts / CURRENT_STABILITY_REPEATS).to_numpy(float),
            "mean_abs_elasticnet_coef": (abs_coef_sum / CURRENT_STABILITY_REPEATS).to_numpy(float),
        }
    ).sort_values(["selection_frequency", "mean_abs_elasticnet_coef"], ascending=[False, False])
    selected_cols = [col for col in cols if col in force_cols]
    selected_cols.extend(
        [col for col in ranking["feature"].tolist() if col not in selected_cols][
            : max(0, top_k - len(selected_cols))
        ]
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking["selected_for_final_model"] = ranking["feature"].isin(selected_cols)
    ranking["selection_strategy"] = f"grouped_elasticnet_top{top_k}"
    ranking["mean_selected_features_per_repeat"] = (
        float(np.mean([row["selected_features"] for row in fit_rows])) if fit_rows else math.nan
    )
    ranking["mean_elasticnet_alpha"] = (
        float(np.mean([row["alpha"] for row in fit_rows])) if fit_rows else math.nan
    )
    ranking["mean_elasticnet_l1_ratio"] = (
        float(np.mean([row["l1_ratio"] for row in fit_rows])) if fit_rows else math.nan
    )
    return selected_cols or list(cols), ranking


def fit_hurdle_with_columns(
    train_aligned: pd.DataFrame,
    test_aligned: pd.DataFrame,
    classifier_cols: list[str],
    regressor_cols: list[str],
    log_positive: bool,
) -> np.ndarray:
    y_train = train_aligned[TARGET].to_numpy(float)
    y_present = (y_train > 0).astype(int)
    classifier = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegressionCV(
                    Cs=np.logspace(-2, 1, 8),
                    cv=3,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )
    if len(np.unique(y_present)) < 2:
        disease_prob = np.full(len(test_aligned), float(y_present.mean()))
    else:
        classifier.fit(train_aligned[classifier_cols], y_present)
        disease_prob = classifier.predict_proba(test_aligned[classifier_cols])[:, 1]

    positive_mask = y_train > 0
    regressor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )
    if positive_mask.sum() < 5:
        severity_pred = np.full(len(test_aligned), float(np.nanmean(y_train)))
    else:
        reg_target = np.log1p(y_train[positive_mask]) if log_positive else y_train[positive_mask]
        regressor.fit(train_aligned.loc[positive_mask, regressor_cols], reg_target)
        severity_pred = regressor.predict(test_aligned[regressor_cols])
        if log_positive:
            severity_pred = np.expm1(severity_pred)
    return residual_pipeline.clip_predictions(disease_prob * severity_pred, y_train)


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else math.nan,
        "spearman": residual_pipeline.safe_spearman(y_true, y_pred),
    }


def grouped_oof_hurdle_predictions(
    train_aligned: pd.DataFrame,
    classifier_cols: list[str],
    regressor_cols: list[str],
    log_positive: bool,
) -> np.ndarray:
    y_train = train_aligned[TARGET].to_numpy(float)
    groups = train_aligned["plot_id"].to_numpy()
    predictions = np.zeros(len(train_aligned), dtype=float)
    n_splits = min(5, len(np.unique(groups)))
    for fit_idx, eval_idx in GroupKFold(n_splits=n_splits).split(
        train_aligned, y_train, groups=groups
    ):
        predictions[eval_idx] = fit_hurdle_with_columns(
            train_aligned.iloc[fit_idx],
            train_aligned.iloc[eval_idx],
            classifier_cols,
            regressor_cols,
            log_positive,
        )
    return predictions


def current_hurdle_stability_topk_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set: str,
    top_k: int,
    log_positive: bool,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cols, train_aligned, test_aligned = residual_pipeline.prepare_aligned(train, test)
    y_train = train_aligned[TARGET].to_numpy(float)
    groups = train_aligned["plot_id"].to_numpy()
    force_cols = {"known__predictor_week"}
    classifier_cols, classifier_ranking = stable_ranked_columns(
        train_aligned,
        cols,
        (y_train > 0).astype(float),
        groups,
        top_k,
        force_cols,
        role="classifier",
    )
    positive_mask = y_train > 0
    if positive_mask.sum() >= 5:
        reg_target = np.log1p(y_train[positive_mask]) if log_positive else y_train[positive_mask]
        regressor_cols, regressor_ranking = stable_ranked_columns(
            train_aligned.loc[positive_mask],
            cols,
            reg_target,
            train_aligned.loc[positive_mask, "plot_id"].to_numpy(),
            top_k,
            force_cols,
            role="positive_severity_regressor",
        )
    else:
        regressor_cols = classifier_cols
        regressor_ranking = classifier_ranking.copy()
        regressor_ranking["role"] = "positive_severity_regressor"

    fit_t0 = time.perf_counter()
    pred = fit_hurdle_with_columns(
        train_aligned, test_aligned, classifier_cols, regressor_cols, log_positive
    )
    fit_time = time.perf_counter() - fit_t0
    model = (
        f"current_hurdle_stability_top{top_k}_{'log_positive' if log_positive else 'raw_positive'}"
    )
    predictions = residual_pipeline.prediction_frame(test_aligned, pred, model, feature_set)
    residual_pipeline.save_predictions(
        predictions, model, feature_set, residual_pipeline.COVARIATES
    )
    result = residual_pipeline.score_predictions(
        predictions,
        len(train_aligned),
        len(sorted(set(classifier_cols + regressor_cols))),
        model,
        feature_set,
    )
    in_sample_pred = fit_hurdle_with_columns(
        train_aligned, train_aligned, classifier_cols, regressor_cols, log_positive
    )
    oof_pred = grouped_oof_hurdle_predictions(
        train_aligned, classifier_cols, regressor_cols, log_positive
    )
    in_sample = regression_scores(y_train, in_sample_pred)
    oof = regression_scores(y_train, oof_pred)
    result["fit_time_s"] = fit_time
    result["classifier_features"] = len(classifier_cols)
    result["regressor_features"] = len(regressor_cols)
    result["feature_selection_strategy"] = f"grouped_elasticnet_top{top_k}_separate_hurdle_roles"
    result["train_in_sample_rmse"] = in_sample["rmse"]
    result["train_grouped_oof_rmse"] = oof["rmse"]
    result["train_oof_minus_in_sample_rmse"] = oof["rmse"] - in_sample["rmse"]
    result["external_minus_oof_rmse"] = result["rmse"] - oof["rmse"]

    selection = pd.concat([classifier_ranking, regressor_ranking], ignore_index=True)
    selection["feature_set"] = feature_set
    selection["model"] = model
    return result, predictions, selection


def current_hurdle_topk_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set: str,
    top_k: int,
    log_positive: bool,
) -> tuple[dict, pd.DataFrame]:
    cols, train_aligned, test_aligned = residual_pipeline.prepare_aligned(train, test)
    y_train = train_aligned[TARGET].to_numpy(float)
    y_present = (y_train > 0).astype(int)
    force_cols = {"known__predictor_week"}
    classifier_cols = ranked_top_columns(
        train_aligned, cols, y_present.astype(float), top_k, force_cols
    )
    positive_mask = y_train > 0
    if positive_mask.sum() >= 5:
        reg_target = np.log1p(y_train[positive_mask]) if log_positive else y_train[positive_mask]
        reg_cols = ranked_top_columns(
            train_aligned.loc[positive_mask], cols, reg_target, top_k, force_cols
        )
    else:
        reg_cols = classifier_cols

    fit_t0 = time.perf_counter()
    classifier = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegressionCV(
                    Cs=np.logspace(-2, 1, 8),
                    cv=3,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )
    if len(np.unique(y_present)) < 2:
        disease_prob = np.full(len(test_aligned), float(y_present.mean()))
    else:
        classifier.fit(train_aligned[classifier_cols], y_present)
        disease_prob = classifier.predict_proba(test_aligned[classifier_cols])[:, 1]

    regressor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )
    if positive_mask.sum() < 5:
        severity_pred = np.full(len(test_aligned), float(np.nanmean(y_train)))
    else:
        target = np.log1p(y_train[positive_mask]) if log_positive else y_train[positive_mask]
        regressor.fit(train_aligned.loc[positive_mask, reg_cols], target)
        severity_pred = regressor.predict(test_aligned[reg_cols])
        if log_positive:
            severity_pred = np.expm1(severity_pred)

    zero_weeks = (
        train_aligned.groupby("target_week")[TARGET].max().loc[lambda s: s <= 0].index.to_numpy()
    )
    pred = disease_prob * severity_pred
    if zero_weeks.size:
        pred[np.isin(test_aligned["target_week"].to_numpy(), zero_weeks)] = 0.0
    pred = residual_pipeline.clip_predictions(pred, y_train)
    fit_time = time.perf_counter() - fit_t0

    model = f"current_hurdle_top{top_k}_{'log_positive' if log_positive else 'raw_positive'}"
    predictions = residual_pipeline.prediction_frame(test_aligned, pred, model, feature_set)
    residual_pipeline.save_predictions(
        predictions, model, feature_set, residual_pipeline.COVARIATES
    )
    result = residual_pipeline.score_predictions(
        predictions,
        len(train_aligned),
        len(sorted(set(classifier_cols + reg_cols))),
        model,
        feature_set,
    )
    result["fit_time_s"] = fit_time
    result["classifier_features"] = len(classifier_cols)
    result["regressor_features"] = len(reg_cols)
    result["zero_target_weeks_from_2024"] = ",".join(map(str, zero_weeks.tolist()))
    return result, predictions


def target_progression(table: pd.DataFrame, year: int, feature_set: str) -> pd.DataFrame:
    return (
        table.groupby("target_week", as_index=False)
        .agg(
            n=("plot_id", "size"),
            n_plots=("plot_id", "nunique"),
            mean_current_ds_plot=(TARGET, "mean"),
            max_current_ds_plot=(TARGET, "max"),
        )
        .assign(year=year, feature_set=feature_set)[
            [
                "year",
                "feature_set",
                "target_week",
                "n",
                "n_plots",
                "mean_current_ds_plot",
                "max_current_ds_plot",
            ]
        ]
    )


def prediction_week_summary(predictions: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (model, feature_set), pred in predictions.items():
        for week, group in pred.groupby("target_week", sort=True):
            y = group["y_true"].to_numpy(float)
            yhat = group["y_pred"].to_numpy(float)
            rows.append(
                {
                    "model": model,
                    "feature_set": feature_set,
                    "target_week": int(week),
                    "n": len(group),
                    "n_plots": group["plot_id"].nunique(),
                    "rmse": math.sqrt(float(np.mean((y - yhat) ** 2))),
                    "mae": float(np.mean(np.abs(y - yhat))),
                    "bias": float(np.mean(yhat - y)),
                    "mean_observed_severity": float(np.mean(y)),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "target_week", "feature_set"])


def write_report(
    results: pd.DataFrame,
    ci: pd.DataFrame,
    progression: pd.DataFrame,
    week_summary: pd.DataFrame,
    paths: dict[str, Path],
    log_path: Path,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "current_plot_severity_2024_to_2025_summary.md"
    lines = [
        "## Results: Current Plot Severity, 2024 Train -> 2025 Test",
        "",
        "This analysis mirrors the compact nadir-vs-multiangular severity pipeline, but pairs reflectance features with same-week `ds_plot` rather than the next later DSDI score.",
        "",
        "### Model Comparison",
        "",
        residual_pipeline.markdown_table(results.round(4)),
        "",
        "### Paired Delta Versus Nadir",
        "",
        "Positive RMSE/MAE reduction means the multiangular model improved over the matched nadir model.",
        "",
        residual_pipeline.markdown_table(ci.round(4)),
        "",
        "### Same-Week Target Support",
        "",
        residual_pipeline.markdown_table(progression.round(4)),
        "",
        "### Target-Week Error Summary",
        "",
        residual_pipeline.markdown_table(week_summary.round(4)),
        "",
        "**Interpretation**: This is a current-severity analysis. It should not be described as early-warning because the predictor and disease-score weeks are identical.",
        "",
        "### Reproducibility",
        "",
        "- Target: same-week plot-level `ds_plot` stored in the existing pipeline target column for model compatibility.",
        "- Training year: 2024.",
        "- Test year: 2025.",
        "- Feature sets: `compact_anomaly_nadir`, `compact_anomaly_multiangular`.",
        "- Covariates: `known__predictor_week` only. Same-week severity does not need a separate target-week covariate because target week equals predictor week.",
        "- Models: stability-selected Ridge, phenology-floor Ridge, hurdle Ridge, top-k hurdle Ridge, reliability-filtered Ridge, and Ridge plus residual XGBoost.",
        f"- Log: `{log_path}`",
        "",
        "### Outputs",
        "",
    ]
    lines.extend([f"- {label}: `{path}`" for label, path in paths.items()])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    total_t0 = time.perf_counter()
    configure_reused_pipeline_paths()
    for directory in [RESULTS_DIR, REPORTS_DIR, PREDICTIONS_DIR, residual_pipeline.FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging()

    disease_t0 = time.perf_counter()
    disease_2024, disease_2025 = load_clean_disease_scores()
    log_phase("load disease scores", disease_t0)

    feature_t0 = time.perf_counter()
    features = residual_pipeline.load_cached_features()
    log_phase("load compact distribution feature sets", feature_t0)

    results = []
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    selection_tables = []
    tuning_tables = []
    progression_tables = []

    model_t0 = time.perf_counter()
    for feature_set, (train_features, test_features) in features.items():
        train = build_current_model_table(train_features, disease_2024)
        test = build_current_model_table(test_features, disease_2025)
        progression_tables.append(target_progression(train, 2024, feature_set))
        progression_tables.append(target_progression(test, 2025, feature_set))
        logging.info("%s current model table: train=%d test=%d", feature_set, len(train), len(test))
        for func in MODEL_FUNCTIONS:
            if func == "current_hurdle_top20_raw":
                result, pred = current_hurdle_topk_model(
                    train, test, feature_set, top_k=20, log_positive=False
                )
            elif func == "current_hurdle_top40_raw":
                result, pred = current_hurdle_topk_model(
                    train, test, feature_set, top_k=40, log_positive=False
                )
            elif func == "current_hurdle_stability_top30_raw":
                result, pred, selection = current_hurdle_stability_topk_model(
                    train, test, feature_set, top_k=30, log_positive=False
                )
                selection_tables.append(selection)
            elif func == "current_hurdle_stability_top50_raw":
                result, pred, selection = current_hurdle_stability_topk_model(
                    train, test, feature_set, top_k=50, log_positive=False
                )
                selection_tables.append(selection)
            elif func == "current_hurdle_top20_log":
                result, pred = current_hurdle_topk_model(
                    train, test, feature_set, top_k=20, log_positive=True
                )
            elif func == "current_hurdle_top40_log":
                result, pred = current_hurdle_topk_model(
                    train, test, feature_set, top_k=40, log_positive=True
                )
            elif func is residual_pipeline.fit_direct_stability_ridge:
                result, pred, selection = func(train, test, feature_set)
                selection_tables.append(selection)
            elif func in {
                residual_pipeline.fit_residual_xgboost_stability,
                residual_pipeline.fit_residual_reliability_filtered_xgboost,
            }:
                result, pred, tuning = func(train, test, feature_set)
                tuning_tables.append(tuning)
            else:
                result, pred = func(train, test, feature_set)
            results.append(result)
            predictions[(result["model"], feature_set)] = pred
    log_phase("fit current-severity models", model_t0)

    results_df = pd.DataFrame(results).sort_values(["rmse", "model"])
    ci_df = residual_pipeline.candidate_delta_ci(results_df)
    progression = pd.concat(progression_tables, ignore_index=True)
    week_summary = prediction_week_summary(predictions)
    selections = (
        pd.concat(selection_tables, ignore_index=True) if selection_tables else pd.DataFrame()
    )
    tuning = pd.concat(tuning_tables, ignore_index=True) if tuning_tables else pd.DataFrame()

    paths = {
        "model_comparison": RESULTS_DIR / "current_severity_model_comparison.csv",
        "paired_delta_ci": RESULTS_DIR / "current_severity_model_comparison_with_paired_ci.csv",
        "target_support": RESULTS_DIR / "current_severity_target_support.csv",
        "week_summary": RESULTS_DIR / "current_severity_week_summary.csv",
        "selected_features": RESULTS_DIR
        / "current_severity_stability_selection_feature_frequencies.csv",
        "xgboost_tuning": RESULTS_DIR / "current_severity_xgboost_tuning_audit.csv",
    }
    results_df.to_csv(paths["model_comparison"], index=False)
    ci_df.to_csv(paths["paired_delta_ci"], index=False)
    progression.to_csv(paths["target_support"], index=False)
    week_summary.to_csv(paths["week_summary"], index=False)
    selections.to_csv(paths["selected_features"], index=False)
    tuning.to_csv(paths["xgboost_tuning"], index=False)
    report_path = write_report(results_df, ci_df, progression, week_summary, paths, log_path)
    logging.info("Report: %s", report_path)
    log_phase("total", total_t0)


if __name__ == "__main__":
    main()
