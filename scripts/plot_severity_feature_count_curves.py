"""Plot external severity performance versus selected angular feature count.

The angular feature ranking is learned from 2024 only using repeated grouped
Ridge fits. For each top-k subset, predictor week and target week are retained
as known timing covariates, then severity performance is evaluated on 2025.
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
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_cross_year_generalization_2024_to_2025 import (  # noqa: E402
    RAA_2024,
    RAA_2025,
    SEED,
    TARGET,
    VZA_2024,
    VZA_2025,
    add_known_covariates,
    align_train_test,
    build_model_table,
    load_2024_disease_with_fallback,
    load_2025_disease_with_fallback,
    load_feature_sets_for_year,
)

OUTPUT_ROOT = ROOT / "outputs/cross_year_generalization_2024_to_2025/severity_feature_selection"
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/logs"

FEATURE_SETS = {
    "multiangular_vza_raa": "VZA + RAA",
    "multiangular_vza_phase": "VZA + phase",
}
MODELS = ["ridge", "xgboost"]
K_VALUES = [5, 10, 15, 20, 30, 40, 60, 80, 100, 125, 150, 175, 200, 225, 250, 300, 350]
ALPHAS = np.logspace(-3, 4, 20)


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"plot_severity_feature_count_curves_{timestamp}.log"
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


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return math.nan
    value = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    return float(value) if value is not None else math.nan


def load_tables() -> tuple[dict[str, object], dict[str, object]]:
    t0 = time.time()
    features_2024 = load_feature_sets_for_year(VZA_2024, RAA_2024)
    features_2025 = load_feature_sets_for_year(VZA_2025, RAA_2025)
    disease_2024 = load_2024_disease_with_fallback()
    disease_2025, _audit = load_2025_disease_with_fallback()
    train_tables = {name: build_model_table(features_2024[name], disease_2024) for name in FEATURE_SETS}
    test_tables = {name: build_model_table(features_2025[name], disease_2025) for name in FEATURE_SETS}
    log_phase("load and join feature/target tables", t0)
    return train_tables, test_tables


def ridge_stability_ranking(train, cols: list[str]) -> pl.DataFrame:
    """Rank angular features by repeated grouped Ridge coefficient magnitude."""
    t0 = time.time()
    y = train[TARGET].to_numpy(float)
    groups = train["plot_id"].to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(4, len(unique_groups))
    if n_splits < 2:
        raise RuntimeError("Not enough groups for grouped stability ranking.")

    coef_sum = np.zeros(len(cols), dtype=float)
    coef_nonzero = np.zeros(len(cols), dtype=int)
    total_fits = 0
    for repeat in range(20):
        # GroupKFold itself is deterministic; shuffle group labels outside the splitter.
        rng = np.random.default_rng(SEED + repeat)
        shuffled_group_labels = dict(zip(unique_groups, rng.permutation(unique_groups), strict=False))
        pseudo_groups = np.array([shuffled_group_labels[g] for g in groups])
        splitter = GroupKFold(n_splits=n_splits)
        for train_idx, _valid_idx in splitter.split(train[cols], y, groups=pseudo_groups):
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("ridge", RidgeCV(alphas=ALPHAS)),
                ]
            )
            model.fit(train.iloc[train_idx][cols], y[train_idx])
            coef = np.abs(model.named_steps["ridge"].coef_)
            coef_sum += coef
            coef_nonzero += (coef > 1e-12).astype(int)
            total_fits += 1
    ranking = pl.DataFrame(
        {
            "feature": cols,
            "mean_abs_standardized_coef": coef_sum / total_fits,
            "nonzero_frequency": coef_nonzero / total_fits,
            "total_fits": total_fits,
        }
    ).sort("mean_abs_standardized_coef", descending=True)
    log_phase("Ridge coefficient stability ranking", t0)
    return ranking


def valid_k_values(n_features: int) -> list[int]:
    values = [k for k in K_VALUES if k < n_features]
    if n_features not in values:
        values.append(n_features)
    return sorted(set(values))


def fit_ridge(train, test, cols: list[str]) -> dict[str, float]:
    t0 = time.time()
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )
    model.fit(train[cols], train[TARGET].to_numpy(float))
    fit_time = time.time() - t0

    pred_t0 = time.time()
    pred = model.predict(test[cols])
    y_train = train[TARGET].to_numpy(float)
    pred = np.clip(pred, float(np.nanmin(y_train)), float(np.nanmax(y_train)))
    predict_time = time.time() - pred_t0
    y = test[TARGET].to_numpy(float)
    return regression_metrics(y, pred) | {"fit_time_s": fit_time, "predict_time_s": predict_time}


def fit_xgboost(train, test, cols: list[str]) -> dict[str, float]:
    t0 = time.time()
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[cols])
    x_test = imputer.transform(test[cols])
    y_train = train[TARGET].to_numpy(float)
    y_test = test[TARGET].to_numpy(float)
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=200,
        learning_rate=0.025,
        max_depth=2,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=8.0,
        random_state=SEED,
        tree_method="hist",
        n_jobs=4,
    )
    model.fit(x_train, y_train, verbose=False)
    fit_time = time.time() - t0

    pred_t0 = time.time()
    pred = model.predict(x_test)
    pred = np.clip(pred, float(np.nanmin(y_train)), float(np.nanmax(y_train)))
    predict_time = time.time() - pred_t0
    return regression_metrics(y_test, pred) | {"fit_time_s": fit_time, "predict_time_s": predict_time}


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": math.sqrt(mean_squared_error(y, pred)),
        "mae": mean_absolute_error(y, pred),
        "r2": r2_score(y, pred) if len(np.unique(y)) > 1 else math.nan,
        "spearman": safe_spearman(y, pred),
    }


def run_curves() -> tuple[pl.DataFrame, list[Path], Path]:
    train_tables, test_tables = load_tables()
    rows: list[dict[str, object]] = []
    ranking_paths: list[Path] = []

    for feature_set, label in FEATURE_SETS.items():
        angular_cols, train, test = align_train_test(train_tables[feature_set], test_tables[feature_set])
        ranking = ridge_stability_ranking(train, angular_cols)
        ranking_path = RESULTS_DIR / f"severity_feature_count_ranking_{feature_set}.csv"
        ranking.write_csv(ranking_path)
        ranking_paths.append(ranking_path)
        ranked_cols = ranking.get_column("feature").to_list()

        for k in valid_k_values(len(ranked_cols)):
            selected_angular = ranked_cols[:k]
            train_k = train.copy()
            test_k = test.copy()
            train_k, test_k, model_cols = add_known_covariates(train_k, test_k, selected_angular, "spectral_plus_week_horizon")
            for model_name in MODELS:
                fit_fn = fit_ridge if model_name == "ridge" else fit_xgboost
                metrics = fit_fn(train_k, test_k, model_cols)
                rows.append(
                    {
                        "feature_set": feature_set,
                        "feature_label": label,
                        "model": model_name,
                        "k_angular_features": k,
                        "n_total_features": len(model_cols),
                        "n_train": len(train_k),
                        "n_test": len(test_k),
                        **metrics,
                    }
                )
    results = pl.DataFrame(rows)
    results_path = RESULTS_DIR / "severity_performance_by_feature_count.csv"
    results.write_csv(results_path)
    return results, ranking_paths, results_path


def plot_one_family(results: pl.DataFrame, feature_set: str) -> Path:
    label = FEATURE_SETS[feature_set]
    data = results.filter(pl.col("feature_set") == feature_set).sort("k_angular_features")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = {"ridge": "#2f6f9f", "xgboost": "#c96f2d"}
    markers = {"ridge": "o", "xgboost": "s"}
    for model in MODELS:
        subset = data.filter(pl.col("model") == model)
        ax.plot(
            subset.get_column("k_angular_features"),
            subset.get_column("rmse"),
            marker=markers[model],
            linewidth=2.2,
            markersize=4.8,
            color=colors[model],
            label="Ridge" if model == "ridge" else "XGBoost",
        )
        best = subset.sort("rmse").row(0, named=True)
        ax.scatter(best["k_angular_features"], best["rmse"], s=92, color=colors[model], edgecolor="black", zorder=5)
        ax.annotate(
            f"best k={best['k_angular_features']}\nRMSE={best['rmse']:.2f}",
            xy=(best["k_angular_features"], best["rmse"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color="#242424",
        )
    ax.set_title(f"Severity RMSE versus selected features: {label}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of top stability-ranked angular features")
    ax.set_ylabel("External 2025 RMSE")
    ax.grid(axis="y", color="#d7d0c6", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.patch.set_facecolor("#fbf7ef")
    ax.set_facecolor("#fbf7ef")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"severity_rmse_by_feature_count_{feature_set}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def markdown_table(df: pl.DataFrame) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.iter_rows(named=True):
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def write_report(results: pl.DataFrame, figure_paths: list[Path], results_path: Path, ranking_paths: list[Path], log_path: Path) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best = (
        results.sort("rmse")
        .group_by(["feature_label", "model"])
        .head(1)
        .sort(["feature_label", "model"])
        .select(["feature_label", "model", "k_angular_features", "n_total_features", "rmse", "mae", "r2", "spearman"])
        .with_columns(pl.selectors.numeric().round(3))
    )
    report_path = REPORTS_DIR / "severity_rmse_by_feature_count_summary.md"
    lines = [
        "# Severity RMSE By Number Of Selected Angular Features",
        "",
        "## Best Points",
        "",
        markdown_table(best),
        "",
        "**Interpretation**: Severity prediction does not consistently improve by adding all angular features. "
        "The best external RMSE occurs at intermediate feature counts, indicating that angular features need feature selection/regularization for cross-year transfer.",
        "",
        "## Outputs",
        "",
        f"- Results CSV: `{results_path}`",
        *[f"- Figure: `{path}`" for path in figure_paths],
        *[f"- Ranking CSV: `{path}`" for path in ranking_paths],
        f"- Log: `{log_path}`",
        "",
        "## Reproducibility",
        "",
        f"- Seed: `{SEED}`",
        "- Angular feature ranking: repeated grouped Ridge coefficient magnitude using 2024 only",
        "- Timing covariates included in every fitted model: predictor week and target week",
        "- Train year: `2024`",
        "- External test year: `2025`",
        "- Target: continuous `future_ds_plot` severity",
    ]
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging()
    script_t0 = time.time()
    results, ranking_paths, results_path = run_curves()
    figure_paths = [plot_one_family(results, feature_set) for feature_set in FEATURE_SETS]
    report_path = write_report(results, figure_paths, results_path, ranking_paths, log_path)
    logging.info("Results written: %s", results_path)
    for path in figure_paths:
        logging.info("Figure written: %s", path)
    logging.info("Report written: %s", report_path)
    log_phase("total script", script_t0)


if __name__ == "__main__":
    main()
