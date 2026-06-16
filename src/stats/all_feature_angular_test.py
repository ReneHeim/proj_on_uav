#!/usr/bin/env python3
"""Nested-CV test using every supported reflectance-derived angular feature."""

import argparse
import logging
import time
from datetime import datetime

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.models.angular_support_sensitivity import retained_cells_overall, wide_band_table
from src.models.matched_angular_validation import (
    BANDS,
    LOGS_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    SEED,
    TARGET_COL,
    VZA_STEP,
    build_splits,
    load_or_build_week_cells,
    load_targets,
)


PREDICTOR_WEEK = 5
TARGET_WEEK = 8
SUPPORT_THRESHOLD = 0.70
INNER_SPLITS = 3
NADIR_FEATURES = len(BANDS)


def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOGS_DIR / f"all_feature_angular_test_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
        force=True,
    )
    logging.info("Log file: %s", path)
    return path


def angular_contrasts(values):
    nadir = values[:, :NADIR_FEATURES]
    off_nadir = values[:, NADIR_FEATURES:]
    band_indices = np.arange(off_nadir.shape[1]) // (off_nadir.shape[1] // NADIR_FEATURES)
    return np.column_stack([nadir, off_nadir - nadir[:, band_indices]])


def build_outer_features(cells, targets, training_plot_ids):
    plot_ids = targets["plot_id"].to_list()
    support = retained_cells_overall(cells, training_plot_ids, SUPPORT_THRESHOLD).filter(
        pl.col("vza_cell") + VZA_STEP / 2 >= 15
    )
    if support.is_empty():
        raise RuntimeError("No training-supported off-nadir angular cells")

    base = targets.select(["plot_id", TARGET_COL]).sort("plot_id")
    nadir = (
        cells.filter(
            pl.col("plot_id").is_in(plot_ids)
            & (pl.col("vza_cell") + VZA_STEP / 2 < 15)
        )
        .group_by("plot_id")
        .agg(*[pl.col(band).mean().alias(f"{band}_nadir") for band in BANDS])
    )
    feature_table = base.join(nadir, on="plot_id", how="left")
    matched = cells.filter(pl.col("plot_id").is_in(plot_ids)).join(
        support, on=["vza_cell", "raa_cell"], how="inner"
    )
    off_cols = []
    for band in BANDS:
        wide = wide_band_table(matched, support, band)
        renamed = {column: f"off_{column}" for column in wide.columns if column != "plot_id"}
        wide = wide.rename(renamed)
        band_cols = sorted(column for column in wide.columns if column != "plot_id")
        off_cols.extend(band_cols)
        feature_table = feature_table.join(wide, on="plot_id", how="left")
    nadir_cols = [f"{band}_nadir" for band in BANDS]
    return feature_table, nadir_cols, off_cols, support.height


def model_pipeline(multiangular):
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if multiangular:
        steps.extend(
            [
                ("contrasts", FunctionTransformer(angular_contrasts, validate=False)),
                ("scaler", StandardScaler()),
                ("pca", PCA(random_state=SEED)),
            ]
        )
    else:
        steps.append(("scaler", StandardScaler()))
    steps.append(
        (
            "classifier",
            LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED),
        )
    )
    return Pipeline(steps)


def tune_and_score(train_x, test_x, train_y, test_y, multiangular, seed):
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    grid = {"classifier__C": [0.01, 0.1, 1.0, 10.0]}
    if multiangular:
        grid["pca__n_components"] = [0.80, 0.90, 0.95, None]
    search = GridSearchCV(
        model_pipeline(multiangular),
        grid,
        scoring="roc_auc",
        cv=inner,
        n_jobs=1,
        refit=True,
    )
    fit_started = time.time()
    search.fit(train_x, train_y)
    fit_time = time.time() - fit_started
    predict_started = time.time()
    probability = search.predict_proba(test_x)[:, 1]
    predict_time = time.time() - predict_started
    return roc_auc_score(test_y, probability), search.best_params_, fit_time, predict_time


def evaluate_effect(cells, targets, seed):
    roster = targets.sort("plot_id")
    y = roster[TARGET_COL].to_numpy()
    rows = []
    for fold, (train_idx, test_idx) in enumerate(build_splits(roster, seed=seed)):
        training_plot_ids = roster["plot_id"].gather(train_idx).to_list()
        features, nadir_cols, off_cols, n_cells = build_outer_features(
            cells, targets, training_plot_ids
        )
        nadir_x = features.select(nadir_cols).to_numpy()
        multi_x = features.select(nadir_cols + off_cols).to_numpy()
        if not np.isfinite(nadir_x[train_idx].astype(float)).any():
            raise RuntimeError(f"Fold {fold} has no observed nadir reflectance in training plots")
        fold_scores = {}
        for name, x, is_multi in [
            ("nadir", nadir_x, False),
            ("multiangular", multi_x, True),
        ]:
            score, params, fit_time, predict_time = tune_and_score(
                x[train_idx], x[test_idx], y[train_idx], y[test_idx], is_multi, seed + fold
            )
            fold_scores[name] = score
            rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "model": name,
                    "AUROC": score,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_angular_cells": n_cells,
                    "n_input_features": x.shape[1],
                    "best_C": params["classifier__C"],
                    "best_pca": str(params.get("pca__n_components", "not_used")),
                    "fit_time_s": fit_time,
                    "predict_time_s": predict_time,
                }
            )
        rows.append(
            {
                "seed": seed,
                "fold": fold,
                "model": "delta",
                "AUROC": fold_scores["multiangular"] - fold_scores["nadir"],
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_angular_cells": n_cells,
                "n_input_features": len(nadir_cols) + len(off_cols),
                "best_C": None,
                "best_pca": None,
                "fit_time_s": 0.0,
                "predict_time_s": 0.0,
            }
        )
    folds = pl.DataFrame(rows)
    scores = folds.filter(pl.col("model") != "delta").group_by("model").agg(pl.col("AUROC").mean())
    return folds, {
        "nadir": scores.filter(pl.col("model") == "nadir")["AUROC"].item(),
        "multiangular": scores.filter(pl.col("model") == "multiangular")["AUROC"].item(),
        "delta": folds.filter(pl.col("model") == "delta")["AUROC"].mean(),
    }


def permuted_targets(targets, rng):
    labels = targets[TARGET_COL].to_numpy().copy()
    rng.shuffle(labels)
    return targets.with_columns(pl.Series(TARGET_COL, labels))


def run_analysis(n_repeats=30, n_permutations=200, seed=SEED):
    log_path = setup_logging()
    total_started = time.time()
    load_started = time.time()
    targets = load_targets().select(["plot_id", TARGET_COL])
    cells, _ = load_or_build_week_cells(PREDICTOR_WEEK)
    logging.info("[PHASE] data loading: %.1fs", time.time() - load_started)

    observed_started = time.time()
    observed_folds, observed = evaluate_effect(cells, targets, seed)
    logging.info(
        "Observed: nadir=%.3f multiangular=%.3f delta=%+.3f",
        observed["nadir"], observed["multiangular"], observed["delta"],
    )
    logging.info("[PHASE] observed fit and predict: %.1fs", time.time() - observed_started)

    repeat_started = time.time()
    repeats = []
    for repeat in range(n_repeats):
        _, effect = evaluate_effect(cells, targets, seed + repeat)
        repeats.append({"repeat": repeat, "seed": seed + repeat, **effect})
    repeat_df = pl.DataFrame(repeats)
    logging.info("[PHASE] repeated nested CV: %.1fs", time.time() - repeat_started)

    permutation_started = time.time()
    rng = np.random.default_rng(seed)
    permutations = []
    for permutation in range(n_permutations):
        _, effect = evaluate_effect(cells, permuted_targets(targets, rng), seed)
        permutations.append({"permutation": permutation, "delta": effect["delta"]})
    permutation_df = pl.DataFrame(permutations)
    logging.info("[PHASE] permutation fit and predict: %.1fs", time.time() - permutation_started)

    repeat_deltas = repeat_df["delta"].to_numpy()
    null_deltas = permutation_df["delta"].to_numpy()
    summary = pl.DataFrame(
        [{
            "predictor_week": PREDICTOR_WEEK,
            "target_week": TARGET_WEEK,
            "n_plots": targets.height,
            "observed_nadir_AUROC": observed["nadir"],
            "observed_multiangular_AUROC": observed["multiangular"],
            "observed_delta_AUROC": observed["delta"],
            "n_repeats": n_repeats,
            "repeated_delta_mean": float(np.mean(repeat_deltas)),
            "repeated_delta_sd": float(np.std(repeat_deltas, ddof=1)),
            "repeated_delta_p2_5": float(np.percentile(repeat_deltas, 2.5)),
            "repeated_delta_p97_5": float(np.percentile(repeat_deltas, 97.5)),
            "repeat_fraction_positive": float(np.mean(repeat_deltas > 0)),
            "n_permutations": n_permutations,
            "permutation_p_multi_superiority": float(
                (1 + np.sum(null_deltas >= observed["delta"])) / (1 + n_permutations)
            ),
            "permutation_p_two_sided": float(
                (1 + np.sum(np.abs(null_deltas) >= abs(observed["delta"]))) / (1 + n_permutations)
            ),
        }]
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    observed_folds.write_csv(RESULTS_DIR / "all_feature_angular_by_fold.csv")
    repeat_df.write_csv(RESULTS_DIR / "all_feature_angular_repeated_cv.csv")
    permutation_df.write_csv(RESULTS_DIR / "all_feature_angular_permutations.csv")
    summary.write_csv(RESULTS_DIR / "all_feature_angular_summary.csv")
    report = write_report(summary.row(0, named=True), log_path, time.time() - total_started)
    logging.info("[PHASE] total: %.1fs", time.time() - total_started)
    logging.info("Report: %s", report)
    return summary, repeat_df, permutation_df


def write_report(row, log_path, total_time):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "all_feature_angular_test_summary.md"
    interpretation = (
        "Adding the complete supported angular reflectance profile improved prediction beyond nadir."
        if row["observed_delta_AUROC"] > 0 and row["permutation_p_multi_superiority"] < 0.05
        else "The complete angular profile did not show statistically conclusive improvement beyond nadir."
    )
    text = f"""## Results: All-Feature Angular Test

| Plots | Nadir AUROC | Multiangular AUROC | Delta | Repeated delta | Repeated SD | Repeated 2.5%-97.5% | Positive repeats | Permutation p | Two-sided p |
|------:|-------------:|-------------------:|------:|---------------:|------------:|--------------------:|-----------------:|--------------:|------------:|
| {row['n_plots']} | {row['observed_nadir_AUROC']:.3f} | {row['observed_multiangular_AUROC']:.3f} | {row['observed_delta_AUROC']:+.3f} | {row['repeated_delta_mean']:+.3f} | {row['repeated_delta_sd']:.3f} | [{row['repeated_delta_p2_5']:+.3f}, {row['repeated_delta_p97_5']:+.3f}] | {row['repeat_fraction_positive']:.1%} | {row['permutation_p_multi_superiority']:.4f} | {row['permutation_p_two_sided']:.4f} |

**Interpretation**: {interpretation}

## Outputs

- `outputs/results/all_feature_angular_by_fold.csv`
- `outputs/results/all_feature_angular_repeated_cv.csv`
- `outputs/results/all_feature_angular_permutations.csv`
- `outputs/results/all_feature_angular_summary.csv`

## Reproducibility

- Predictor week: {PREDICTOR_WEEK}
- Target week: {TARGET_WEEK}
- Nadir model: all five nadir reflectance bands
- Multiangular model: the same nadir bands plus every training-supported off-nadir cell as a within-band nadir contrast
- No treatment, cultivar, coordinates, identifiers, raw geometry, or angular-presence predictors
- Outer CV: stratified by plot
- Inner CV: {INNER_SPLITS}-fold tuning of logistic C and PCA variance retention
- Angular support: at least {SUPPORT_THRESHOLD:.0%} of outer-training plots
- Missing-value imputation, scaling, PCA, and model tuning: fit inside CV
- Seed: {SEED}
- Log: `{log_path}`
- Runtime: {total_time:.1f}s
"""
    with path.open("w") as handle:
        handle.write(text)
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    run_analysis(args.repeats, args.permutations, args.seed)


if __name__ == "__main__":
    main()
