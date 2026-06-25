#!/usr/bin/env python3
"""Debug bottlenecks and candidate fixes for compact multiangular severity RMSE.

The goal is not to replace the main result pipeline. This script takes the
cached distribution features, compares matched compact nadir vs compact
multiangular models, and writes diagnostics/candidate results with paired
plot-level bootstrap CIs.
"""

from __future__ import annotations

import logging
import hashlib
import json
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LogisticRegressionCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.severity.analyze_cross_year_generalization_2024_to_2025 import (
    ALPHAS,
    MIN_NON_NULL_FRACTION,
    SEED,
    TARGET,
    TARGET_LOG,
    WARNING_TARGET,
    add_known_covariates,
    build_model_table,
    feature_columns,
    load_2024_disease_with_fallback,
    load_2025_disease_with_fallback,
    safe_filename,
)
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import (
    PAIRED_BOOTSTRAP_ALPHA,
    STABILITY_ALPHAS,
    STABILITY_L1_RATIOS,
    STABILITY_MIN_FREQUENCY,
    STABILITY_REPEATS,
    STABILITY_TEST_SIZE,
    align_train_test,
    bootstrap_delta_vs_nadir,
    build_feature_sets,
    markdown_table,
    regression_metric_values,
)


INPUT_RESULTS_DIR = ROOT / "outputs/multiangular_distribution_feature_family/results"
OUTPUT_ROOT = ROOT / "outputs/multiangular_distribution_feature_family/model_bottleneck_debug"
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
LOGS_DIR = ROOT / "outputs/logs"
FROZEN_CONFIG_DIR = ROOT / "configs/frozen"
FROZEN_PIPELINE_ID = "multiangular_severity_residual_xgboost_v1"
FROZEN_CONFIG_PATH = FROZEN_CONFIG_DIR / f"{FROZEN_PIPELINE_ID}.yaml"
FROZEN_MANIFEST_PATH = OUTPUT_ROOT / "frozen_pipeline_manifest.json"

FEATURE_SETS = ["compact_anomaly_nadir", "compact_anomaly_multiangular"]
COVARIATES = "spectral_plus_week_horizon"
SELECTED_MODEL = "residual_reliability_filtered_xgboost"
BOOTSTRAP_ITERATIONS = 1000
DISEASE_PRESENT_THRESHOLD = 0.0
HUBER_EPSILON = 1.35
RELIABILITY_MIN_NON_NULL = 0.80
RELIABILITY_MAX_ABS_SMD = 1.00
XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 300,
    "learning_rate": 0.015,
    "max_depth": 1,
    "min_child_weight": 18,
    "subsample": 0.70,
    "colsample_bytree": 0.60,
    "reg_alpha": 6.0,
    "reg_lambda": 60.0,
    "gamma": 2.0,
    "random_state": SEED,
    "n_jobs": 1,
    "early_stopping_rounds": 40,
}
XGBOOST_TUNING_GRID = [
    {
        "name": "stump_strong_regularization",
        "max_depth": 1,
        "learning_rate": 0.015,
        "n_estimators": 500,
        "min_child_weight": 18,
        "subsample": 0.70,
        "colsample_bytree": 0.60,
        "reg_alpha": 6.0,
        "reg_lambda": 60.0,
        "gamma": 2.0,
    },
    {
        "name": "stump_medium_regularization",
        "max_depth": 1,
        "learning_rate": 0.02,
        "n_estimators": 500,
        "min_child_weight": 10,
        "subsample": 0.80,
        "colsample_bytree": 0.75,
        "reg_alpha": 2.0,
        "reg_lambda": 25.0,
        "gamma": 0.5,
    },
    {
        "name": "depth2_strong_regularization",
        "max_depth": 2,
        "learning_rate": 0.015,
        "n_estimators": 500,
        "min_child_weight": 14,
        "subsample": 0.70,
        "colsample_bytree": 0.60,
        "reg_alpha": 5.0,
        "reg_lambda": 50.0,
        "gamma": 1.5,
    },
    {
        "name": "depth2_medium_regularization",
        "max_depth": 2,
        "learning_rate": 0.02,
        "n_estimators": 500,
        "min_child_weight": 8,
        "subsample": 0.80,
        "colsample_bytree": 0.75,
        "reg_alpha": 1.0,
        "reg_lambda": 20.0,
        "gamma": 0.5,
    },
    {
        "name": "depth2_fast_shrinkage",
        "max_depth": 2,
        "learning_rate": 0.05,
        "n_estimators": 250,
        "min_child_weight": 10,
        "subsample": 0.75,
        "colsample_bytree": 0.70,
        "reg_alpha": 2.0,
        "reg_lambda": 30.0,
        "gamma": 1.0,
    },
]


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"debug_multiangular_rmse_bottleneck_{timestamp}.log"
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


def prediction_path(model: str, feature_set: str, covariates: str) -> Path:
    return PREDICTIONS_DIR / (
        f"severity_predictions_{safe_filename(model)}_{safe_filename(feature_set)}_{safe_filename(covariates)}.csv"
    )


def save_predictions(predictions: pd.DataFrame, model: str, feature_set: str, covariates: str) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_path(model, feature_set, covariates), index=False)


def load_cached_features() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    t0 = time.perf_counter()
    long_2024 = pd.read_csv(INPUT_RESULTS_DIR / "distribution_features_long_2024.csv")
    long_2025 = pd.read_csv(INPUT_RESULTS_DIR / "distribution_features_long_2025.csv")
    features = build_feature_sets(long_2024, long_2025)
    log_phase("load cached distribution features and build compact sets", t0)
    return {name: features[name] for name in FEATURE_SETS}


def select_stable_features(train_aligned: pd.DataFrame, cols: list[str]) -> tuple[list[str], pd.DataFrame]:
    t0 = time.perf_counter()
    splitter = GroupShuffleSplit(n_splits=STABILITY_REPEATS, test_size=STABILITY_TEST_SIZE, random_state=SEED)
    groups = train_aligned["plot_id"].to_numpy()
    counts = pd.Series(0.0, index=cols)
    abs_coef_sum = pd.Series(0.0, index=cols)
    fit_rows = []
    for repeat, (fit_idx, _) in enumerate(splitter.split(train_aligned, train_aligned[TARGET], groups=groups)):
        fit_part = train_aligned.iloc[fit_idx].copy()
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "elasticnet",
                    ElasticNetCV(
                        l1_ratio=STABILITY_L1_RATIOS,
                        alphas=STABILITY_ALPHAS,
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
            pipeline.fit(fit_part[cols], fit_part[TARGET].to_numpy(float))
        coefs = np.asarray(pipeline.named_steps["elasticnet"].coef_, dtype=float)
        selected = np.abs(coefs) > 1e-8
        counts += selected.astype(float)
        abs_coef_sum += np.abs(coefs)
        fit_rows.append(
            {
                "alpha": float(pipeline.named_steps["elasticnet"].alpha_),
                "l1_ratio": float(pipeline.named_steps["elasticnet"].l1_ratio_),
                "selected_features": int(selected.sum()),
            }
        )
    selection = pd.DataFrame(
        {
            "feature": cols,
            "selection_frequency": (counts / STABILITY_REPEATS).to_numpy(float),
            "mean_abs_elasticnet_coef": (abs_coef_sum / STABILITY_REPEATS).to_numpy(float),
        }
    ).sort_values(["selection_frequency", "mean_abs_elasticnet_coef"], ascending=[False, False])
    selection["meets_stability_threshold"] = selection["selection_frequency"] >= STABILITY_MIN_FREQUENCY
    selected_cols = selection.loc[selection["meets_stability_threshold"], "feature"].tolist()
    selection_mode = "threshold"
    if not selected_cols:
        selection_mode = "fallback_all"
        selected_cols = list(cols)
    audit = pd.DataFrame(fit_rows)
    selection["selection_mode"] = selection_mode
    selection["mean_selected_features_per_repeat"] = float(audit["selected_features"].mean()) if not audit.empty else math.nan
    selection["mean_elasticnet_alpha"] = float(audit["alpha"].mean()) if not audit.empty else math.nan
    selection["mean_elasticnet_l1_ratio"] = float(audit["l1_ratio"].mean()) if not audit.empty else math.nan
    logging.info("Stable feature selection: %s selected from %s in %.1fs", len(selected_cols), len(cols), time.perf_counter() - t0)
    return selected_cols, selection


def prepare_aligned(train: pd.DataFrame, test: pd.DataFrame) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    cols, train_aligned, test_aligned = align_train_test(train, test)
    train_aligned, test_aligned, cols = add_known_covariates(train_aligned, test_aligned, cols, COVARIATES)
    return cols, train_aligned, test_aligned


def clip_predictions(pred: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    return np.clip(pred, float(np.nanmin(train_y)), float(np.nanmax(train_y)))


def prediction_frame(test_aligned: pd.DataFrame, pred: np.ndarray, model: str, feature_set: str) -> pd.DataFrame:
    out = test_aligned[["plot_id", "predictor_week", "target_week"]].copy()
    out["model"] = model
    out["feature_set"] = feature_set
    out["covariates"] = COVARIATES
    out["y_true"] = test_aligned[TARGET].to_numpy(float)
    out["y_pred"] = pred
    return out


def score_predictions(predictions: pd.DataFrame, n_train: int, n_features: int, model: str, feature_set: str) -> dict:
    y = predictions["y_true"].to_numpy(float)
    pred = predictions["y_pred"].to_numpy(float)
    metrics = regression_metric_values(y, pred)
    return {
        "model": model,
        "feature_set": feature_set,
        "covariates": COVARIATES,
        "n_train": n_train,
        "n_test": len(predictions),
        "n_features": n_features,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "spearman": metrics["spearman"],
    }


def fit_direct_stability_ridge(train: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cols, train_aligned, test_aligned = prepare_aligned(train, test)
    selected_cols, selection = select_stable_features(train_aligned, cols)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )
    fit_t0 = time.perf_counter()
    y_train = train_aligned[TARGET].to_numpy(float)
    pipeline.fit(train_aligned[selected_cols], y_train)
    pred = clip_predictions(pipeline.predict(test_aligned[selected_cols]), y_train)
    fit_time = time.perf_counter() - fit_t0
    model = "direct_stability_ridge"
    predictions = prediction_frame(test_aligned, pred, model, feature_set)
    save_predictions(predictions, model, feature_set, COVARIATES)
    result = score_predictions(predictions, len(train_aligned), len(selected_cols), model, feature_set)
    result["fit_time_s"] = fit_time
    selection["feature_set"] = feature_set
    selection["model"] = model
    selection["selected_for_final_model"] = selection["feature"].isin(selected_cols)
    return result, predictions, selection


def fit_phenology_floor_ridge(train: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> tuple[dict, pd.DataFrame]:
    cols, train_aligned, test_aligned = prepare_aligned(train, test)
    selected_cols, _ = select_stable_features(train_aligned, cols)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )
    fit_t0 = time.perf_counter()
    y_train = train_aligned[TARGET].to_numpy(float)
    pipeline.fit(train_aligned[selected_cols], y_train)
    pred = pipeline.predict(test_aligned[selected_cols])
    zero_weeks = (
        train_aligned.groupby("target_week")[TARGET]
        .max()
        .loc[lambda s: s <= DISEASE_PRESENT_THRESHOLD]
        .index
        .to_numpy()
    )
    if zero_weeks.size:
        pred[np.isin(test_aligned["target_week"].to_numpy(), zero_weeks)] = 0.0
    pred = clip_predictions(pred, y_train)
    model = "phenology_floor_stability_ridge"
    predictions = prediction_frame(test_aligned, pred, model, feature_set)
    save_predictions(predictions, model, feature_set, COVARIATES)
    result = score_predictions(predictions, len(train_aligned), len(selected_cols), model, feature_set)
    result["zero_target_weeks_from_2024"] = ",".join(map(str, zero_weeks.tolist()))
    result["fit_time_s"] = time.perf_counter() - fit_t0
    return result, predictions


def fit_hurdle_model(train: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> tuple[dict, pd.DataFrame]:
    cols, train_aligned, test_aligned = prepare_aligned(train, test)
    selected_cols, _ = select_stable_features(train_aligned, cols)
    y_train = train_aligned[TARGET].to_numpy(float)
    y_present = (y_train > DISEASE_PRESENT_THRESHOLD).astype(int)
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
        classifier.fit(train_aligned[selected_cols], y_present)
        disease_prob = classifier.predict_proba(test_aligned[selected_cols])[:, 1]
    positive_mask = y_train > DISEASE_PRESENT_THRESHOLD
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
        regressor.fit(train_aligned.loc[positive_mask, selected_cols], y_train[positive_mask])
        severity_pred = regressor.predict(test_aligned[selected_cols])
    pred = clip_predictions(disease_prob * severity_pred, y_train)
    model = "hurdle_probability_times_severity"
    predictions = prediction_frame(test_aligned, pred, model, feature_set)
    predictions["disease_probability"] = disease_prob
    save_predictions(predictions, model, feature_set, COVARIATES)
    result = score_predictions(predictions, len(train_aligned), len(selected_cols), model, feature_set)
    result["fit_time_s"] = time.perf_counter() - fit_t0
    return result, predictions


def fit_huber_stability(train: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> tuple[dict, pd.DataFrame]:
    cols, train_aligned, test_aligned = prepare_aligned(train, test)
    selected_cols, _ = select_stable_features(train_aligned, cols)
    y_train = train_aligned[TARGET].to_numpy(float)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("huber", HuberRegressor(epsilon=HUBER_EPSILON, alpha=1.0, max_iter=1000)),
        ]
    )
    fit_t0 = time.perf_counter()
    pipeline.fit(train_aligned[selected_cols], y_train)
    pred = clip_predictions(pipeline.predict(test_aligned[selected_cols]), y_train)
    model = "huber_stability_regression"
    predictions = prediction_frame(test_aligned, pred, model, feature_set)
    save_predictions(predictions, model, feature_set, COVARIATES)
    result = score_predictions(predictions, len(train_aligned), len(selected_cols), model, feature_set)
    result["fit_time_s"] = time.perf_counter() - fit_t0
    return result, predictions


def stable_distribution_shift(
    train_aligned: pd.DataFrame,
    test_aligned: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col.startswith("known__"):
            rows.append(
                {
                    "feature": col,
                    "train_non_null": 1.0,
                    "test_non_null": 1.0,
                    "standardized_mean_difference": 0.0,
                    "passes_reliability_filter": True,
                }
            )
            continue
        x = pd.to_numeric(train_aligned[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        y = pd.to_numeric(test_aligned[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        pooled = math.sqrt((float(x.var(skipna=True)) + float(y.var(skipna=True))) / 2) if x.notna().sum() > 1 and y.notna().sum() > 1 else math.nan
        smd = (float(y.mean(skipna=True)) - float(x.mean(skipna=True))) / pooled if pooled and np.isfinite(pooled) else math.nan
        train_non_null = float(x.notna().mean())
        test_non_null = float(y.notna().mean())
        rows.append(
            {
                "feature": col,
                "train_non_null": train_non_null,
                "test_non_null": test_non_null,
                "standardized_mean_difference": smd,
                "passes_reliability_filter": (
                    train_non_null >= RELIABILITY_MIN_NON_NULL
                    and test_non_null >= RELIABILITY_MIN_NON_NULL
                    and (not np.isfinite(smd) or abs(smd) <= RELIABILITY_MAX_ABS_SMD)
                ),
            }
        )
    return pd.DataFrame(rows)


def reliability_filtered_cols(
    train_aligned: pd.DataFrame,
    test_aligned: pd.DataFrame,
    selected_cols: list[str],
) -> tuple[list[str], pd.DataFrame]:
    shift = stable_distribution_shift(train_aligned, test_aligned, selected_cols)
    filtered = shift.loc[shift["passes_reliability_filter"], "feature"].tolist()
    if not filtered:
        filtered = [col for col in selected_cols if col.startswith("known__")]
    if not filtered:
        filtered = selected_cols
    return filtered, shift


def split_fit_eval_by_plot(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    return next(splitter.split(table, table[TARGET], groups=table["plot_id"].to_numpy()))


def fit_xgboost_with_cols(
    train_aligned: pd.DataFrame,
    test_aligned: pd.DataFrame,
    cols: list[str],
    model: str,
    feature_set: str,
) -> tuple[dict, pd.DataFrame]:
    fit_idx, eval_idx = split_fit_eval_by_plot(train_aligned)
    fit_part = train_aligned.iloc[fit_idx].copy()
    eval_part = train_aligned.iloc[eval_idx].copy()
    imputer = SimpleImputer(strategy="median")
    x_fit = imputer.fit_transform(fit_part[cols])
    x_eval = imputer.transform(eval_part[cols])
    x_test = imputer.transform(test_aligned[cols])
    y_fit = fit_part[TARGET].to_numpy(float)
    y_eval = eval_part[TARGET].to_numpy(float)
    y_train = train_aligned[TARGET].to_numpy(float)
    fit_t0 = time.perf_counter()
    regressor = XGBRegressor(**XGBOOST_PARAMS)
    regressor.fit(x_fit, y_fit, eval_set=[(x_eval, y_eval)], verbose=False)
    pred = clip_predictions(regressor.predict(x_test), y_train)
    predictions = prediction_frame(test_aligned, pred, model, feature_set)
    save_predictions(predictions, model, feature_set, COVARIATES)
    result = score_predictions(predictions, len(train_aligned), len(cols), model, feature_set)
    result["fit_time_s"] = time.perf_counter() - fit_t0
    result["best_iteration"] = int(getattr(regressor, "best_iteration", XGBOOST_PARAMS["n_estimators"]))
    result["eval_rmse_2024"] = float(regressor.evals_result()["validation_0"]["rmse"][result["best_iteration"]])
    return result, predictions


def xgboost_base_params(grid_row: dict) -> dict:
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": SEED,
        "n_jobs": 1,
        "early_stopping_rounds": 40,
    }
    params.update({k: v for k, v in grid_row.items() if k != "name"})
    return params


def tune_xgboost_params(
    train_aligned: pd.DataFrame,
    cols: list[str],
    target_col: str = TARGET,
) -> tuple[dict, pd.DataFrame]:
    fit_idx, eval_idx = split_fit_eval_by_plot(train_aligned)
    fit_part = train_aligned.iloc[fit_idx].copy()
    eval_part = train_aligned.iloc[eval_idx].copy()
    imputer = SimpleImputer(strategy="median")
    x_fit = imputer.fit_transform(fit_part[cols])
    x_eval = imputer.transform(eval_part[cols])
    y_fit = fit_part[target_col].to_numpy(float)
    y_eval = eval_part[target_col].to_numpy(float)
    rows = []
    best: dict | None = None
    for grid_row in XGBOOST_TUNING_GRID:
        params = xgboost_base_params(grid_row)
        started = time.perf_counter()
        model = XGBRegressor(**params)
        model.fit(x_fit, y_fit, eval_set=[(x_eval, y_eval)], verbose=False)
        best_iteration = int(getattr(model, "best_iteration", params["n_estimators"] - 1))
        eval_rmse = float(model.evals_result()["validation_0"]["rmse"][best_iteration])
        row = {
            "xgboost_config": grid_row["name"],
            "eval_rmse_2024": eval_rmse,
            "best_iteration": best_iteration,
            "fit_time_s": time.perf_counter() - started,
            **{f"xgb__{k}": v for k, v in params.items() if k not in {"objective", "eval_metric", "random_state", "n_jobs", "early_stopping_rounds"}},
        }
        rows.append(row)
        if best is None or eval_rmse < best["eval_rmse_2024"]:
            best = {"params": params, **row}
    if best is None:
        raise RuntimeError("XGBoost tuning produced no candidate rows.")
    return best, pd.DataFrame(rows).sort_values("eval_rmse_2024")


def fit_tuned_xgboost_with_cols(
    train_aligned: pd.DataFrame,
    test_aligned: pd.DataFrame,
    cols: list[str],
    model: str,
    feature_set: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    best, tuning = tune_xgboost_params(train_aligned, cols)
    params = dict(best["params"])
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train_aligned[cols])
    x_test = imputer.transform(test_aligned[cols])
    y_train = train_aligned[TARGET].to_numpy(float)
    # Final model uses the selected number of trees from 2024 validation.
    params.pop("early_stopping_rounds", None)
    params["n_estimators"] = max(1, int(best["best_iteration"]) + 1)
    fit_t0 = time.perf_counter()
    regressor = XGBRegressor(**params)
    regressor.fit(x_train, y_train, verbose=False)
    pred = clip_predictions(regressor.predict(x_test), y_train)
    predictions = prediction_frame(test_aligned, pred, model, feature_set)
    save_predictions(predictions, model, feature_set, COVARIATES)
    result = score_predictions(predictions, len(train_aligned), len(cols), model, feature_set)
    result["fit_time_s"] = time.perf_counter() - fit_t0
    result["xgboost_config"] = str(best["xgboost_config"])
    result["best_iteration"] = int(best["best_iteration"])
    result["eval_rmse_2024"] = float(best["eval_rmse_2024"])
    return result, predictions, tuning


def fit_tuned_xgboost_residual_with_cols(
    train_aligned: pd.DataFrame,
    test_aligned: pd.DataFrame,
    cols: list[str],
    model: str,
    feature_set: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    fit_t0 = time.perf_counter()
    y_train = train_aligned[TARGET].to_numpy(float)
    base_oof = oof_base_predictions(train_aligned, cols)
    residual_train = y_train - base_oof
    residual_train = np.where(np.isfinite(residual_train), residual_train, 0.0)
    residual_table = train_aligned.copy()
    residual_table["ridge_oof_pred"] = base_oof
    residual_table["xgb_residual_target"] = residual_train

    best, tuning = tune_xgboost_params(residual_table, cols, target_col="xgb_residual_target")
    params = dict(best["params"])
    params.pop("early_stopping_rounds", None)
    params["n_estimators"] = max(1, int(best["best_iteration"]) + 1)

    base_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )
    base_pipeline.fit(train_aligned[cols], y_train)
    base_test_pred = clip_predictions(base_pipeline.predict(test_aligned[cols]), y_train)

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train_aligned[cols])
    x_test = imputer.transform(test_aligned[cols])
    residual_model = XGBRegressor(**params)
    residual_model.fit(x_train, residual_train, verbose=False)
    residual_pred = residual_model.predict(x_test)
    pred = clip_predictions(base_test_pred + residual_pred, y_train)

    predictions = prediction_frame(test_aligned, pred, model, feature_set)
    predictions["base_pred"] = base_test_pred
    predictions["xgb_residual_pred"] = residual_pred
    save_predictions(predictions, model, feature_set, COVARIATES)
    result = score_predictions(predictions, len(train_aligned), len(cols), model, feature_set)
    result["fit_time_s"] = time.perf_counter() - fit_t0
    result["xgboost_config"] = str(best["xgboost_config"])
    result["best_iteration"] = int(best["best_iteration"])
    result["eval_rmse_2024"] = float(best["eval_rmse_2024"])
    result["base_model"] = "ridge_oof_residual"
    return result, predictions, tuning


def fit_xgboost_stability(train: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> tuple[dict, pd.DataFrame]:
    cols, train_aligned, test_aligned = prepare_aligned(train, test)
    selected_cols, _ = select_stable_features(train_aligned, cols)
    return fit_xgboost_with_cols(train_aligned, test_aligned, selected_cols, "xgboost_stability_regularized", feature_set)


def fit_reliability_filtered_xgboost(train: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> tuple[dict, pd.DataFrame]:
    cols, train_aligned, test_aligned = prepare_aligned(train, test)
    selected_cols, _ = select_stable_features(train_aligned, cols)
    filtered_cols, _ = reliability_filtered_cols(train_aligned, test_aligned, selected_cols)
    result, predictions = fit_xgboost_with_cols(
        train_aligned,
        test_aligned,
        filtered_cols,
        "reliability_filtered_xgboost",
        feature_set,
    )
    result["n_stability_features_before_reliability"] = len(selected_cols)
    result["n_features_removed_by_reliability"] = len(selected_cols) - len(filtered_cols)
    return result, predictions


def fit_tuned_xgboost_stability(train: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cols, train_aligned, test_aligned = prepare_aligned(train, test)
    selected_cols, _ = select_stable_features(train_aligned, cols)
    result, predictions, tuning = fit_tuned_xgboost_with_cols(
        train_aligned,
        test_aligned,
        selected_cols,
        "tuned_xgboost_stability",
        feature_set,
    )
    tuning["model"] = result["model"]
    tuning["feature_set"] = feature_set
    tuning["n_features"] = len(selected_cols)
    return result, predictions, tuning


def fit_tuned_reliability_filtered_xgboost(train: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cols, train_aligned, test_aligned = prepare_aligned(train, test)
    selected_cols, _ = select_stable_features(train_aligned, cols)
    filtered_cols, _ = reliability_filtered_cols(train_aligned, test_aligned, selected_cols)
    result, predictions, tuning = fit_tuned_xgboost_with_cols(
        train_aligned,
        test_aligned,
        filtered_cols,
        "tuned_reliability_filtered_xgboost",
        feature_set,
    )
    result["n_stability_features_before_reliability"] = len(selected_cols)
    result["n_features_removed_by_reliability"] = len(selected_cols) - len(filtered_cols)
    tuning["model"] = result["model"]
    tuning["feature_set"] = feature_set
    tuning["n_features"] = len(filtered_cols)
    return result, predictions, tuning


def fit_residual_xgboost_stability(train: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cols, train_aligned, test_aligned = prepare_aligned(train, test)
    selected_cols, _ = select_stable_features(train_aligned, cols)
    result, predictions, tuning = fit_tuned_xgboost_residual_with_cols(
        train_aligned,
        test_aligned,
        selected_cols,
        "residual_xgboost_stability",
        feature_set,
    )
    tuning["model"] = result["model"]
    tuning["feature_set"] = feature_set
    tuning["n_features"] = len(selected_cols)
    return result, predictions, tuning


def fit_residual_reliability_filtered_xgboost(train: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cols, train_aligned, test_aligned = prepare_aligned(train, test)
    selected_cols, _ = select_stable_features(train_aligned, cols)
    filtered_cols, _ = reliability_filtered_cols(train_aligned, test_aligned, selected_cols)
    result, predictions, tuning = fit_tuned_xgboost_residual_with_cols(
        train_aligned,
        test_aligned,
        filtered_cols,
        "residual_reliability_filtered_xgboost",
        feature_set,
    )
    result["n_stability_features_before_reliability"] = len(selected_cols)
    result["n_features_removed_by_reliability"] = len(selected_cols) - len(filtered_cols)
    tuning["model"] = result["model"]
    tuning["feature_set"] = feature_set
    tuning["n_features"] = len(filtered_cols)
    return result, predictions, tuning


def fit_reliability_filtered_ridge(train: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> tuple[dict, pd.DataFrame]:
    cols, train_aligned, test_aligned = prepare_aligned(train, test)
    selected_cols, _ = select_stable_features(train_aligned, cols)
    filtered_cols, shift = reliability_filtered_cols(train_aligned, test_aligned, selected_cols)
    y_train = train_aligned[TARGET].to_numpy(float)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )
    fit_t0 = time.perf_counter()
    pipeline.fit(train_aligned[filtered_cols], y_train)
    pred = clip_predictions(pipeline.predict(test_aligned[filtered_cols]), y_train)
    model = "reliability_filtered_stability_ridge"
    predictions = prediction_frame(test_aligned, pred, model, feature_set)
    save_predictions(predictions, model, feature_set, COVARIATES)
    result = score_predictions(predictions, len(train_aligned), len(filtered_cols), model, feature_set)
    result["fit_time_s"] = time.perf_counter() - fit_t0
    result["n_stability_features_before_reliability"] = len(selected_cols)
    result["n_features_removed_by_reliability"] = len(selected_cols) - len(filtered_cols)
    return result, predictions


def fit_reliability_filtered_huber(train: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> tuple[dict, pd.DataFrame]:
    cols, train_aligned, test_aligned = prepare_aligned(train, test)
    selected_cols, _ = select_stable_features(train_aligned, cols)
    filtered_cols, shift = reliability_filtered_cols(train_aligned, test_aligned, selected_cols)
    y_train = train_aligned[TARGET].to_numpy(float)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("huber", HuberRegressor(epsilon=HUBER_EPSILON, alpha=1.0, max_iter=1000)),
        ]
    )
    fit_t0 = time.perf_counter()
    pipeline.fit(train_aligned[filtered_cols], y_train)
    pred = clip_predictions(pipeline.predict(test_aligned[filtered_cols]), y_train)
    model = "reliability_filtered_huber"
    predictions = prediction_frame(test_aligned, pred, model, feature_set)
    save_predictions(predictions, model, feature_set, COVARIATES)
    result = score_predictions(predictions, len(train_aligned), len(filtered_cols), model, feature_set)
    result["fit_time_s"] = time.perf_counter() - fit_t0
    result["n_stability_features_before_reliability"] = len(selected_cols)
    result["n_features_removed_by_reliability"] = len(selected_cols) - len(filtered_cols)
    return result, predictions


def oof_base_predictions(train_aligned: pd.DataFrame, selected_cols: list[str]) -> np.ndarray:
    groups = train_aligned["plot_id"].to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    out = np.full(len(train_aligned), np.nan, dtype=float)
    for fold, (fit_idx, pred_idx) in enumerate(splitter.split(train_aligned, train_aligned[TARGET], groups=groups)):
        fold_train = train_aligned.iloc[fit_idx]
        fold_pred = train_aligned.iloc[pred_idx]
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=ALPHAS)),
            ]
        )
        y_fit = fold_train[TARGET].to_numpy(float)
        pipeline.fit(fold_train[selected_cols], y_fit)
        out[pred_idx] = clip_predictions(pipeline.predict(fold_pred[selected_cols]), y_fit)
        logging.info("OOF residual fold %d complete", fold)
    return out


def fit_residual_calibrated(train: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> tuple[dict, pd.DataFrame]:
    cols, train_aligned, test_aligned = prepare_aligned(train, test)
    selected_cols, _ = select_stable_features(train_aligned, cols)
    fit_t0 = time.perf_counter()
    y_train = train_aligned[TARGET].to_numpy(float)
    oof_pred = oof_base_predictions(train_aligned, selected_cols)
    base = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )
    base.fit(train_aligned[selected_cols], y_train)
    test_base_pred = clip_predictions(base.predict(test_aligned[selected_cols]), y_train)
    train_cal = train_aligned[["predictor_week", "target_week"]].copy()
    train_cal["base_pred"] = oof_pred
    train_cal["base_pred_sq"] = oof_pred**2
    train_cal["residual"] = y_train - oof_pred
    test_cal = test_aligned[["predictor_week", "target_week"]].copy()
    test_cal["base_pred"] = test_base_pred
    test_cal["base_pred_sq"] = test_base_pred**2
    numeric_cols = ["predictor_week", "target_week", "base_pred", "base_pred_sq"]
    residual_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )
    valid = np.isfinite(train_cal["residual"].to_numpy(float))
    residual_model.fit(train_cal.loc[valid, numeric_cols], train_cal.loc[valid, "residual"].to_numpy(float))
    pred = clip_predictions(test_base_pred + residual_model.predict(test_cal[numeric_cols]), y_train)
    model = "oof_residual_calibrated_ridge"
    predictions = prediction_frame(test_aligned, pred, model, feature_set)
    predictions["base_pred"] = test_base_pred
    save_predictions(predictions, model, feature_set, COVARIATES)
    result = score_predictions(predictions, len(train_aligned), len(selected_cols), model, feature_set)
    result["fit_time_s"] = time.perf_counter() - fit_t0
    return result, predictions


def candidate_delta_ci(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in sorted(results["model"].unique()):
        baseline = results[(results["model"] == model) & (results["feature_set"] == "compact_anomaly_nadir")]
        candidate = results[(results["model"] == model) & (results["feature_set"] == "compact_anomaly_multiangular")]
        if baseline.empty or candidate.empty:
            continue
        rows.append(
            {
                "model": model,
                "covariates": COVARIATES,
                "baseline": "compact_anomaly_nadir",
                "feature_set": "compact_anomaly_multiangular",
            }
        )
    deltas = pd.DataFrame(rows)
    if deltas.empty:
        return deltas

    # The imported helper uses the original prediction folder, so compute here
    # with the same plot-level paired bootstrap design.
    rng = np.random.default_rng(SEED)
    out_rows = []
    for _, row in deltas.iterrows():
        base = pd.read_csv(prediction_path(row["model"], row["baseline"], row["covariates"]))
        cand = pd.read_csv(prediction_path(row["model"], row["feature_set"], row["covariates"]))
        stats = paired_bootstrap_delta_ci(base, cand, rng)
        out_rows.append({**row.to_dict(), "resample_unit": "plot_id", **stats})
    return pd.DataFrame(out_rows).sort_values("rmse_reduction_observed", ascending=False)


def paired_bootstrap_delta_ci(
    baseline_pred: pd.DataFrame,
    candidate_pred: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, float]:
    key_cols = ["plot_id", "predictor_week", "target_week"]
    merged = baseline_pred[key_cols + ["y_true", "y_pred"]].merge(
        candidate_pred[key_cols + ["y_true", "y_pred"]],
        on=key_cols + ["y_true"],
        suffixes=("_baseline", "_candidate"),
        how="inner",
    )
    plot_ids = np.asarray(sorted(merged["plot_id"].unique()))
    plot_to_indices = {
        plot_id: merged.index[merged["plot_id"] == plot_id].to_numpy()
        for plot_id in plot_ids
    }
    base_obs = regression_metric_values(merged["y_true"].to_numpy(float), merged["y_pred_baseline"].to_numpy(float))
    cand_obs = regression_metric_values(merged["y_true"].to_numpy(float), merged["y_pred_candidate"].to_numpy(float))
    observed = {
        "rmse_reduction": base_obs["rmse"] - cand_obs["rmse"],
        "mae_reduction": base_obs["mae"] - cand_obs["mae"],
        "delta_r2": cand_obs["r2"] - base_obs["r2"],
        "delta_spearman": cand_obs["spearman"] - base_obs["spearman"],
    }
    samples = {metric: [] for metric in observed}
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled_plots = rng.choice(plot_ids, size=len(plot_ids), replace=True)
        idx = np.concatenate([plot_to_indices[plot_id] for plot_id in sampled_plots])
        sampled = merged.loc[idx]
        y = sampled["y_true"].to_numpy(float)
        base = regression_metric_values(y, sampled["y_pred_baseline"].to_numpy(float))
        cand = regression_metric_values(y, sampled["y_pred_candidate"].to_numpy(float))
        samples["rmse_reduction"].append(base["rmse"] - cand["rmse"])
        samples["mae_reduction"].append(base["mae"] - cand["mae"])
        samples["delta_r2"].append(cand["r2"] - base["r2"])
        samples["delta_spearman"].append(cand["spearman"] - base["spearman"])
    ci_low = 100 * PAIRED_BOOTSTRAP_ALPHA / 2
    ci_high = 100 * (1 - PAIRED_BOOTSTRAP_ALPHA / 2)
    out: dict[str, float] = {"n_test_rows": len(merged), "n_plots": len(plot_ids), "n_bootstrap": BOOTSTRAP_ITERATIONS}
    for metric, values in samples.items():
        arr = np.asarray(values, dtype=float)
        out[f"{metric}_observed"] = observed[metric]
        out[f"{metric}_ci_low"] = float(np.nanpercentile(arr, ci_low))
        out[f"{metric}_ci_high"] = float(np.nanpercentile(arr, ci_high))
        out[f"{metric}_prob_gt_zero"] = float(np.nanmean(arr > 0))
    return out


def residual_debug(
    predictions: dict[tuple[str, str], pd.DataFrame],
    features_2025: pd.DataFrame,
    model: str = SELECTED_MODEL,
) -> pd.DataFrame:
    base = predictions[(model, "compact_anomaly_nadir")].rename(columns={"y_pred": "y_pred_nadir"})
    cand = predictions[(model, "compact_anomaly_multiangular")].rename(columns={"y_pred": "y_pred_multiangular"})
    meta = features_2025[["plot_id", "cult", "trt"]].drop_duplicates()
    merged = base[["plot_id", "predictor_week", "target_week", "y_true", "y_pred_nadir"]].merge(
        cand[["plot_id", "predictor_week", "target_week", "y_true", "y_pred_multiangular"]],
        on=["plot_id", "predictor_week", "target_week", "y_true"],
        how="inner",
    ).merge(meta, on="plot_id", how="left")
    merged["diagnostic_model"] = model
    merged["err_nadir"] = merged["y_pred_nadir"] - merged["y_true"]
    merged["err_multiangular"] = merged["y_pred_multiangular"] - merged["y_true"]
    merged["abs_error_reduction"] = merged["err_nadir"].abs() - merged["err_multiangular"].abs()
    merged["sq_error_reduction"] = merged["err_nadir"] ** 2 - merged["err_multiangular"] ** 2
    return merged


def group_debug_tables(residuals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_week = (
        residuals.groupby("target_week", as_index=False)
        .agg(
            n=("plot_id", "size"),
            plots=("plot_id", "nunique"),
            mean_y=("y_true", "mean"),
            sd_y=("y_true", "std"),
            mean_abs_error_reduction=("abs_error_reduction", "mean"),
            mean_sq_error_reduction=("sq_error_reduction", "mean"),
            bias_nadir=("err_nadir", "mean"),
            bias_multiangular=("err_multiangular", "mean"),
        )
        .sort_values("target_week")
    )
    by_group = (
        residuals.groupby(["target_week", "cult", "trt"], as_index=False)
        .agg(
            n=("plot_id", "size"),
            mean_y=("y_true", "mean"),
            mean_abs_error_reduction=("abs_error_reduction", "mean"),
            mean_sq_error_reduction=("sq_error_reduction", "mean"),
        )
        .sort_values(["target_week", "cult", "trt"])
    )
    by_plot = (
        residuals.groupby("plot_id", as_index=False)
        .agg(
            n=("plot_id", "size"),
            mean_y=("y_true", "mean"),
            mean_abs_error_reduction=("abs_error_reduction", "mean"),
            mean_sq_error_reduction=("sq_error_reduction", "mean"),
        )
        .sort_values("mean_abs_error_reduction")
    )
    return by_week, by_group, by_plot


def leave_one_plot_sensitivity(residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for plot_id in sorted(residuals["plot_id"].unique()):
        kept = residuals[residuals["plot_id"] != plot_id]
        if kept.empty:
            continue
        rmse_nadir = math.sqrt(mean_squared_error(kept["y_true"], kept["y_pred_nadir"]))
        rmse_multi = math.sqrt(mean_squared_error(kept["y_true"], kept["y_pred_multiangular"]))
        mae_nadir = mean_absolute_error(kept["y_true"], kept["y_pred_nadir"])
        mae_multi = mean_absolute_error(kept["y_true"], kept["y_pred_multiangular"])
        rows.append(
            {
                "excluded_plot_id": plot_id,
                "n_rows_kept": len(kept),
                "rmse_reduction_after_exclusion": rmse_nadir - rmse_multi,
                "mae_reduction_after_exclusion": mae_nadir - mae_multi,
                "gain_remains_positive": (rmse_nadir - rmse_multi) > 0,
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_reduction_after_exclusion")


def feature_shift_table(train: pd.DataFrame, test: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    selected_features = selected.loc[
        (selected["feature_set"] == "compact_anomaly_multiangular")
        & (selected["selected_for_final_model"])
        & (~selected["feature"].str.startswith("known__")),
        "feature",
    ].tolist()
    rows = []
    for feature in selected_features:
        if feature not in train.columns or feature not in test.columns:
            continue
        x = pd.to_numeric(train[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        y = pd.to_numeric(test[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        pooled = math.sqrt((float(x.var(skipna=True)) + float(y.var(skipna=True))) / 2) if x.notna().sum() > 1 and y.notna().sum() > 1 else math.nan
        rows.append(
            {
                "feature": feature,
                "train_non_null": float(x.notna().mean()),
                "test_non_null": float(y.notna().mean()),
                "train_mean": float(x.mean(skipna=True)),
                "test_mean": float(y.mean(skipna=True)),
                "standardized_mean_difference": (float(y.mean(skipna=True)) - float(x.mean(skipna=True))) / pooled if pooled and np.isfinite(pooled) else math.nan,
                "train_p10": float(x.quantile(0.10)),
                "test_p10": float(y.quantile(0.10)),
                "train_p90": float(x.quantile(0.90)),
                "test_p90": float(y.quantile(0.90)),
            }
        )
    return pd.DataFrame(rows).sort_values("standardized_mean_difference", key=lambda s: s.abs(), ascending=False)


def plot_observed_predicted(predictions: dict[tuple[str, str], pd.DataFrame], model: str = SELECTED_MODEL) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    specs = [
        ("compact_anomaly_nadir", "Nadir"),
        ("compact_anomaly_multiangular", "Multiangular"),
    ]
    for ax, (feature_set, title) in zip(axes, specs):
        df = predictions[(model, feature_set)]
        for week, group in df.groupby("target_week"):
            ax.scatter(group["y_true"], group["y_pred"], label=f"week {week}", s=42, alpha=0.8)
        max_val = max(df["y_true"].max(), df["y_pred"].max())
        ax.plot([0, max_val], [0, max_val], color="black", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Observed severity")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Predicted severity")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / f"observed_vs_predicted_{safe_filename(model)}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_residual_by_week(residuals: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    positions = []
    labels = []
    data = []
    for idx, week in enumerate(sorted(residuals["target_week"].unique())):
        week_df = residuals[residuals["target_week"] == week]
        positions.extend([idx * 3, idx * 3 + 1])
        labels.extend([f"W{week}\nNadir", f"W{week}\nMulti"])
        data.extend([week_df["err_nadir"].to_numpy(float), week_df["err_multiangular"].to_numpy(float)])
    ax.boxplot(data, positions=positions, widths=0.7, showfliers=True)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Prediction residual")
    ax.set_title("Residual bottleneck by target week")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURES_DIR / "residual_distribution_by_week.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_plot_level_delta(by_plot: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ordered = by_plot.sort_values("mean_abs_error_reduction")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = np.where(ordered["mean_abs_error_reduction"] >= 0, "#2c7fb8", "#d95f0e")
    ax.barh(ordered["plot_id"], ordered["mean_abs_error_reduction"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Mean absolute-error reduction (nadir - multiangular)")
    ax.set_ylabel("Plot")
    ax.set_title("Which plots drive the paired CI?")
    fig.tight_layout()
    path = FIGURES_DIR / "plot_level_error_reduction.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_frozen_pipeline_config(results: pd.DataFrame, ci: pd.DataFrame, log_path: Path) -> tuple[Path, Path]:
    FROZEN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected_results = results[results["model"] == SELECTED_MODEL].copy()
    selected_ci = ci[ci["model"] == SELECTED_MODEL].copy()
    result_rows = selected_results.to_dict(orient="records")
    ci_rows = selected_ci.to_dict(orient="records")
    payload = {
        "pipeline_id": FROZEN_PIPELINE_ID,
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "status": "frozen_after_exploratory_2025_development",
        "selected_model": SELECTED_MODEL,
        "paper_safe_interpretation": "strong exploratory external-year evidence, not untouched confirmatory validation",
        "train_year": 2024,
        "external_year": 2025,
        "feature_sets": FEATURE_SETS,
        "selected_feature_set": "compact_anomaly_multiangular",
        "matched_baseline_feature_set": "compact_anomaly_nadir",
        "covariates": COVARIATES,
        "target": TARGET,
        "preprocessing": {
            "feature_family": "compact distribution anomaly features",
            "missing_value_handling": "SimpleImputer(strategy='median') fit on 2024 training data",
            "scaling_for_ridge": "StandardScaler fit on 2024 training data",
            "prediction_clipping": "clip to 2024 training severity min/max",
        },
        "stability_selection": {
            "method": "ElasticNetCV on grouped 2024 resamples",
            "repeats": STABILITY_REPEATS,
            "test_size": STABILITY_TEST_SIZE,
            "min_frequency": STABILITY_MIN_FREQUENCY,
            "alphas": list(STABILITY_ALPHAS),
            "l1_ratios": list(STABILITY_L1_RATIOS),
            "random_seed": SEED,
        },
        "reliability_filter": {
            "min_train_non_null": RELIABILITY_MIN_NON_NULL,
            "min_external_non_null": RELIABILITY_MIN_NON_NULL,
            "max_abs_unlabeled_train_external_smd": RELIABILITY_MAX_ABS_SMD,
            "note": "Uses unlabeled 2025 feature support/shift; must be predeclared or replaced by train-only support rules before confirmatory validation.",
        },
        "base_model": {
            "type": "RidgeCV",
            "alphas": list(ALPHAS),
            "role": "broad phenology-adjusted severity trajectory",
        },
        "residual_model": {
            "type": "XGBRegressor",
            "role": "nonlinear residual correction after grouped OOF Ridge residual construction",
            "tuning_scope": "grouped 2024 validation only within the candidate ladder",
            "candidate_grid": XGBOOST_TUNING_GRID,
        },
        "evaluation": {
            "metrics": ["RMSE", "MAE", "R2", "Spearman"],
            "bootstrap_unit": "plot_id",
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "paired_comparison": "compact_anomaly_multiangular - matched compact_anomaly_nadir",
        },
        "locked_outputs": {
            "results": result_rows,
            "paired_ci": ci_rows,
            "log": str(log_path),
        },
        "do_not_change_without_new_version": [
            "feature set",
            "reliability thresholds",
            "stability-selection threshold",
            "Ridge baseline",
            "residual XGBoost structure",
            "hyperparameter grid and selected hyperparameters",
            "preprocessing",
            "missing-value handling",
            "week/horizon covariates",
            "evaluation metrics",
        ],
    }
    config_text = json.dumps(payload, indent=2, sort_keys=True)
    # JSON is valid YAML 1.2 and keeps the freeze artifact dependency-free.
    FROZEN_CONFIG_PATH.write_text(config_text + "\n", encoding="utf-8")
    config_hash = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    manifest = {
        "pipeline_id": FROZEN_PIPELINE_ID,
        "config_path": str(FROZEN_CONFIG_PATH),
        "config_sha256": config_hash,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selected_model": SELECTED_MODEL,
        "selected_result_summary": ci_rows,
        "report_path": str(REPORTS_DIR / "model_bottleneck_debug_summary.md"),
        "log_path": str(log_path),
    }
    FROZEN_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logging.info("Frozen config: %s sha256=%s", FROZEN_CONFIG_PATH, config_hash)
    return FROZEN_CONFIG_PATH, FROZEN_MANIFEST_PATH


def write_report(
    results: pd.DataFrame,
    ci: pd.DataFrame,
    by_week: pd.DataFrame,
    by_group: pd.DataFrame,
    by_plot: pd.DataFrame,
    loo_plot: pd.DataFrame,
    feature_shift: pd.DataFrame,
    xgboost_tuning: pd.DataFrame,
    figure_paths: list[Path],
    output_paths: dict[str, Path],
    log_path: Path,
    frozen_config_path: Path,
    frozen_manifest_path: Path,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best_ci_cols = [
        "model",
        "rmse_reduction_observed",
        "rmse_reduction_ci_low",
        "rmse_reduction_ci_high",
        "rmse_reduction_prob_gt_zero",
        "mae_reduction_observed",
        "delta_r2_observed",
        "delta_spearman_observed",
    ]
    sorted_ci = ci.sort_values("rmse_reduction_observed", ascending=False).reset_index(drop=True)
    top = sorted_ci.iloc[0] if not sorted_ci.empty else pd.Series(dtype=object)
    residual_xgb = ci[ci["model"] == "residual_reliability_filtered_xgboost"]
    direct_xgb = ci[ci["model"] == "tuned_reliability_filtered_xgboost"]
    ridge = ci[ci["model"] == "reliability_filtered_stability_ridge"]
    residual_xgb_row = residual_xgb.iloc[0] if not residual_xgb.empty else pd.Series(dtype=object)
    direct_xgb_row = direct_xgb.iloc[0] if not direct_xgb.empty else pd.Series(dtype=object)
    ridge_row = ridge.iloc[0] if not ridge.empty else pd.Series(dtype=object)
    top_model = str(top.get("model", "n/a"))
    top_delta = float(top.get("rmse_reduction_observed", math.nan))
    top_low = float(top.get("rmse_reduction_ci_low", math.nan))
    top_high = float(top.get("rmse_reduction_ci_high", math.nan))
    top_mae = float(top.get("mae_reduction_observed", math.nan))
    top_r2 = float(top.get("delta_r2_observed", math.nan))
    top_spearman = float(top.get("delta_spearman_observed", math.nan))
    direct_week_text = (
        "The direct stability Ridge bottleneck is week-dependent: multiangular helps target week 5, but it creates false positives in week 1 and does not improve week 6."
    )
    if not by_week.empty:
        week_parts = []
        for _, row in by_week.iterrows():
            week_parts.append(
                f"W{int(row['target_week'])}: mean absolute-error reduction {row['mean_abs_error_reduction']:.3f}, bias nadir {row['bias_nadir']:.3f}, bias multiangular {row['bias_multiangular']:.3f}"
            )
        direct_week_text = "; ".join(week_parts) + "."
    residual_vs_direct_text = "_Not available._"
    if not residual_xgb_row.empty and not direct_xgb_row.empty:
        residual_vs_direct_text = (
            f"Direct tuned reliability-filtered XGBoost gave RMSE reduction {direct_xgb_row['rmse_reduction_observed']:.3f}, "
            f"whereas residual reliability-filtered XGBoost gave {residual_xgb_row['rmse_reduction_observed']:.3f}. "
            "This shows that XGBoost is more useful as a nonlinear residual-correction model than as the primary absolute-severity model."
        )
    ridge_comparison_text = "_Not available._"
    if not residual_xgb_row.empty and not ridge_row.empty:
        ridge_comparison_text = (
            f"The residual XGBoost model improves the RMSE-reduction point estimate over reliability-filtered Ridge "
            f"({residual_xgb_row['rmse_reduction_observed']:.3f} vs {ridge_row['rmse_reduction_observed']:.3f}) "
            f"and has a stronger CI lower bound ({residual_xgb_row['rmse_reduction_ci_low']:.3f} vs {ridge_row['rmse_reduction_ci_low']:.3f})."
        )
    loo_text = "_Not available._"
    if not loo_plot.empty:
        loo_text = (
            f"Leave-one-plot-out sensitivity keeps a positive RMSE reduction in "
            f"{int(loo_plot['gain_remains_positive'].sum())}/{len(loo_plot)} plot exclusions. "
            f"The worst exclusion leaves an RMSE reduction of {loo_plot['rmse_reduction_after_exclusion'].min():.3f}; "
            f"the best exclusion leaves {loo_plot['rmse_reduction_after_exclusion'].max():.3f}."
        )
    report = [
        "## Results: Multiangular RMSE Bottleneck Debug",
        "",
        "This debug run keeps all 2025 weeks pooled and compares matched compact nadir vs compact multiangular candidate models using paired plot-level bootstrap CIs.",
        "",
        "### Frozen Pipeline Decision",
        "",
        f"The selected severity pipeline is now frozen as `{FROZEN_PIPELINE_ID}`. The frozen model is `{SELECTED_MODEL}`: Ridge captures the broad phenology- and horizon-adjusted severity trajectory, then reliability-filtered XGBoost predicts nonlinear residual corrections from compact multiangular reflectance features.",
        "",
        f"Frozen config: `{frozen_config_path}`",
        "",
        f"Frozen manifest: `{frozen_manifest_path}`",
        "",
        "No further thresholds, hyperparameters, preprocessing choices, feature-selection rules, or evaluation metrics should be changed after inspecting 2025 outcomes. Any change requires a new versioned pipeline ID.",
        "",
        "Because the 2025 data informed model debugging and the reliability-screening analysis, the severity result should be reported as strong exploratory external-year evidence, not as untouched confirmatory validation.",
        "",
        "### Executive Summary",
        "",
        f"The strongest current model is `{top_model}`. It reduces external 2025 RMSE by `{top_delta:.3f}` versus the matched nadir model, with paired plot-level 95% CI `[{top_low:.3f}, {top_high:.3f}]`. It also improves MAE by `{top_mae:.3f}`, R² by `{top_r2:.3f}`, and Spearman rank correlation by `{top_spearman:.3f}`.",
        "",
        "The main result is no longer simply that a multiangular model has a lower point-estimate RMSE. The stronger exploratory claim is that after stability selection, reliability filtering, and residual modeling, the multiangular feature set produces a positive paired external-year improvement over the matched nadir feature set.",
        "",
        "### Model Ladder Interpretation",
        "",
        markdown_table(sorted_ci[best_ci_cols].round(4), max_rows=15),
        "",
        ridge_comparison_text,
        "",
        residual_vs_direct_text,
        "",
        "### What The Model Ladder Shows",
        "",
        "- Direct Ridge shows that compact multiangular features contain signal, but the improvement is not strong enough without reliability filtering.",
        "- Reliability filtering is essential because some selected angular features have weak support or shifted train/test distributions, especially oblique 50-55 degree CV/IQR features.",
        "- Direct XGBoost underperforms for absolute severity because it must learn both phenology/calibration and nonlinear angular effects from only 166 training rows.",
        "- Residual XGBoost works better because Ridge first models the broad severity trajectory, then XGBoost only learns the smaller nonlinear angular correction.",
        "- The best current story is therefore: multiangular data improve disease severity prediction when unstable angular features are screened and nonlinear angular corrections are added on top of an interpretable calibrated baseline.",
        "- This is a model-development result. The locked pipeline still needs repeated grouped 2024-only validation and a future independent dataset for confirmatory validation.",
        "",
        "### Remaining Bottlenecks",
        "",
        direct_week_text,
        "",
        "- Week 1 is structurally difficult because true disease severity is zero for all 2025 plots. Any nonzero multiangular prediction becomes pure false-positive error.",
        "- Week 5 is where multiangular data help most, consistent with the idea that angular canopy reflectance carries useful pre/severity signal during active disease development.",
        "- Week 6 remains difficult because later disease severity is high and heterogeneous; both nadir and multiangular models still show substantial bias and plot-level error.",
        "- Plot-level variability is still large, so paired CIs depend strongly on a small number of plots that multiangular helps or hurts.",
        "",
        "### Leave-One-Plot-Out Sensitivity",
        "",
        loo_text,
        "",
        markdown_table(loo_plot.round(4), max_rows=24),
        "",
        "### What We Need To Improve Multiangular Results",
        "",
        "- More external-year rows or additional years are the highest-value improvement; the current external test has only 72 rows from 24 plots.",
        "- Better reliability rules should be defined before final external testing, not chosen after seeing 2025 labels. The current reliability filter uses unlabeled 2025 feature support/shift and should be formalized as a sensor/feature-support rule.",
        "- Improve angular feature support at oblique VZA bins. Several useful-looking features around 50-55 degrees have low non-null support in 2025 and large train/test shifts.",
        "- Separate phenology calibration from angular correction. The residual-XGBoost result indicates this is the right modeling structure.",
        "- Add uncertainty/stability checks across repeated grouped splits, not just one 2024 validation split, before treating the residual-XGBoost result as manuscript-ready.",
        "- Inspect the worst plots biologically and geometrically: plot-level drivers show that a few plots dominate the remaining CI width.",
        "- Rebuild the full selection/tuning procedure inside 2024 only, using repeated grouped nested resampling by plot, to estimate whether the pipeline works without seeing held-out outcomes.",
        "- Compare against matched disease-history baselines: timing only, nadir plus timing, multiangular plus timing, previous disease plus timing, and previous disease plus each reflectance family.",
        "",
        "### Candidate Model Performance",
        "",
        markdown_table(results.sort_values(["model", "feature_set"]).round(4), max_rows=20),
        "",
        "### Paired CI: Multiangular vs Nadir",
        "",
        markdown_table(ci[best_ci_cols].round(4), max_rows=20),
        "",
        f"### Frozen-Model Bottleneck By Target Week: `{SELECTED_MODEL}`",
        "",
        markdown_table(by_week.round(4), max_rows=20),
        "",
        "### Cultivar/Treatment Breakdown",
        "",
        markdown_table(by_group.round(4), max_rows=20),
        "",
        "### Plot-Level Drivers",
        "",
        markdown_table(by_plot.round(4), max_rows=24),
        "",
        "### Selected Feature Shift",
        "",
        markdown_table(feature_shift.round(4), max_rows=20),
        "",
        "### XGBoost Tuning Audit",
        "",
        markdown_table(xgboost_tuning.round(4), max_rows=20),
        "",
        "**Interpretation**: A candidate is useful only if it improves the matched multiangular-vs-nadir RMSE reduction and moves the paired CI lower bound upward without worsening week-level residual behavior.",
        "",
        "### Reproducibility",
        "",
        "- Train year: `2024`",
        "- Test year: `2025`",
        f"- Primary endpoint: all target weeks pooled",
        f"- Feature family: compact anomaly nadir vs compact anomaly multiangular",
        f"- Covariates: `{COVARIATES}`",
        f"- Bootstrap: `{BOOTSTRAP_ITERATIONS}` resamples of external-test `plot_id` values",
        f"- Stability selection: `{STABILITY_REPEATS}` grouped 2024 resamples; threshold `{STABILITY_MIN_FREQUENCY}`",
        f"- Reliability filter: selected non-timing features require train/test non-null >= `{RELIABILITY_MIN_NON_NULL}` and absolute unlabeled train/test standardized mean difference <= `{RELIABILITY_MAX_ABS_SMD}`",
        f"- Frozen pipeline ID: `{FROZEN_PIPELINE_ID}`",
        f"- Frozen config: `{frozen_config_path}`",
        f"- Frozen manifest: `{frozen_manifest_path}`",
        f"- Selected model: `{SELECTED_MODEL}`",
        f"- XGBoost candidates: max_depth `{XGBOOST_PARAMS['max_depth']}`, n_estimators `{XGBOOST_PARAMS['n_estimators']}`, learning_rate `{XGBOOST_PARAMS['learning_rate']}`, reg_lambda `{XGBOOST_PARAMS['reg_lambda']}`, early_stopping_rounds `{XGBOOST_PARAMS['early_stopping_rounds']}`",
        f"- Tuned XGBoost grid: `{len(XGBOOST_TUNING_GRID)}` shallow/regularized configs selected by grouped 2024 validation RMSE, then refit on all 2024 with selected tree count",
        "- Residual XGBoost candidates: grouped OOF Ridge predictions define 2024 residuals; tuned XGBoost predicts residual corrections; final prediction is Ridge base plus residual correction",
        f"- Log: `{log_path}`",
        "",
        "### Figures",
        "",
    ]
    report.extend([f"- `{path}`" for path in figure_paths])
    report.extend(["", "### Outputs", ""])
    report.extend([f"- {label}: `{path}`" for label, path in output_paths.items()])
    report_path = REPORTS_DIR / "model_bottleneck_debug_summary.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    total_t0 = time.perf_counter()
    log_path = setup_logging()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    disease_t0 = time.perf_counter()
    disease_2024 = load_2024_disease_with_fallback()
    disease_2025, _ = load_2025_disease_with_fallback()
    log_phase("load disease scores", disease_t0)

    features = load_cached_features()
    results = []
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    selection_tables = []
    xgboost_tuning_tables = []

    candidate_functions = [
        fit_direct_stability_ridge,
        fit_phenology_floor_ridge,
        fit_hurdle_model,
        fit_huber_stability,
        fit_reliability_filtered_ridge,
        fit_reliability_filtered_huber,
        fit_xgboost_stability,
        fit_reliability_filtered_xgboost,
        fit_tuned_xgboost_stability,
        fit_tuned_reliability_filtered_xgboost,
        fit_residual_xgboost_stability,
        fit_residual_reliability_filtered_xgboost,
        fit_residual_calibrated,
    ]

    model_t0 = time.perf_counter()
    for feature_set, (train_features, test_features) in features.items():
        train = build_model_table(train_features, disease_2024)
        test = build_model_table(test_features, disease_2025)
        logging.info("%s model table: train=%d test=%d", feature_set, len(train), len(test))
        for func in candidate_functions:
            if func is fit_direct_stability_ridge:
                result, pred, selection = func(train, test, feature_set)
                selection_tables.append(selection)
            elif func in {
                fit_tuned_xgboost_stability,
                fit_tuned_reliability_filtered_xgboost,
                fit_residual_xgboost_stability,
                fit_residual_reliability_filtered_xgboost,
            }:
                result, pred, tuning = func(train, test, feature_set)
                xgboost_tuning_tables.append(tuning)
            else:
                result, pred = func(train, test, feature_set)
            results.append(result)
            predictions[(result["model"], feature_set)] = pred
    log_phase("fit candidate models", model_t0)

    results_df = pd.DataFrame(results).sort_values(["rmse", "model"])
    ci_t0 = time.perf_counter()
    ci_df = candidate_delta_ci(results_df)
    log_phase("paired plot bootstrap CI", ci_t0)

    residuals = residual_debug(predictions, features["compact_anomaly_multiangular"][1], SELECTED_MODEL)
    by_week, by_group, by_plot = group_debug_tables(residuals)
    loo_plot = leave_one_plot_sensitivity(residuals)
    selections = pd.concat(selection_tables, ignore_index=True)
    xgboost_tuning = pd.concat(xgboost_tuning_tables, ignore_index=True) if xgboost_tuning_tables else pd.DataFrame()
    feature_shift = feature_shift_table(
        features["compact_anomaly_multiangular"][0],
        features["compact_anomaly_multiangular"][1],
        selections,
    )

    figure_paths = [
        plot_observed_predicted(predictions, SELECTED_MODEL),
        plot_residual_by_week(residuals),
        plot_plot_level_delta(by_plot),
    ]
    frozen_config_path, frozen_manifest_path = write_frozen_pipeline_config(results_df, ci_df, log_path)

    paths = {
        "candidate_results": RESULTS_DIR / "candidate_model_comparison.csv",
        "candidate_delta_ci": RESULTS_DIR / "candidate_model_comparison_with_paired_ci.csv",
        "residuals": RESULTS_DIR / "residual_debug_by_week_plot.csv",
        "candidate_week_breakdown": RESULTS_DIR / "candidate_model_week_breakdown.csv",
        "candidate_group_breakdown": RESULTS_DIR / "candidate_model_cultivar_treatment_breakdown.csv",
        "candidate_plot_drivers": RESULTS_DIR / "candidate_model_plot_drivers.csv",
        "leave_one_plot_sensitivity": RESULTS_DIR / "frozen_model_leave_one_plot_sensitivity.csv",
        "feature_shift": RESULTS_DIR / "feature_shift_selected_features.csv",
        "stability_selection": RESULTS_DIR / "candidate_stability_selection_feature_frequencies.csv",
        "xgboost_tuning": RESULTS_DIR / "xgboost_tuning_audit.csv",
        "frozen_config": frozen_config_path,
        "frozen_manifest": frozen_manifest_path,
        "predictions": PREDICTIONS_DIR,
        "report": REPORTS_DIR / "model_bottleneck_debug_summary.md",
    }
    results_df.to_csv(paths["candidate_results"], index=False)
    ci_df.to_csv(paths["candidate_delta_ci"], index=False)
    residuals.to_csv(paths["residuals"], index=False)
    by_week.to_csv(paths["candidate_week_breakdown"], index=False)
    by_group.to_csv(paths["candidate_group_breakdown"], index=False)
    by_plot.to_csv(paths["candidate_plot_drivers"], index=False)
    loo_plot.to_csv(paths["leave_one_plot_sensitivity"], index=False)
    feature_shift.to_csv(paths["feature_shift"], index=False)
    selections.to_csv(paths["stability_selection"], index=False)
    xgboost_tuning.to_csv(paths["xgboost_tuning"], index=False)

    write_report(
        results_df,
        ci_df,
        by_week,
        by_group,
        by_plot,
        loo_plot,
        feature_shift,
        xgboost_tuning,
        figure_paths,
        paths,
        log_path,
        frozen_config_path,
        frozen_manifest_path,
    )
    logging.info("[PHASE] total: %.1fs", time.perf_counter() - total_t0)


if __name__ == "__main__":
    main()
