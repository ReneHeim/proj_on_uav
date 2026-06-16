#!/usr/bin/env python3
"""Repeated-CV stability and full-pipeline permutation test for week-5 angular gain."""

import argparse
import logging
import time
from datetime import datetime

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from src.models.angular_support_sensitivity import retained_cells_overall
from src.models.matched_angular_validation import (
    BANDS,
    LOGS_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    SEED,
    TARGET_COL,
    VZA_STEP,
    build_splits,
    classifier,
    load_or_build_week_cells,
    load_targets,
    residualize,
)

PREDICTOR_WEEK = 5
SUPPORT_THRESHOLD = 0.70
GEOMETRY_COLS = [
    "n_pixels",
    "n_images",
    "vza_mean",
    "vza_std",
    "vza_min",
    "vza_max",
    "raa_mean",
    "raa_std",
]


def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"fair_multiangular_significance_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    logging.info("Log file: %s", log_path)
    return log_path


def compact_features(cells, geometry, targets, training_plot_ids):
    support = retained_cells_overall(cells, training_plot_ids, SUPPORT_THRESHOLD).filter(
        pl.col("vza_cell") + VZA_STEP / 2 >= 15
    )
    if support.is_empty():
        raise RuntimeError("No training-supported off-nadir cells")
    plot_ids = targets["plot_id"].to_list()
    nadir = (
        cells.filter(pl.col("plot_id").is_in(plot_ids) & (pl.col("vza_cell") + VZA_STEP / 2 < 15))
        .group_by("plot_id")
        .agg(*[pl.col(band).mean().alias(f"{band}_nadir") for band in BANDS])
    )
    off_nadir = (
        cells.filter(pl.col("plot_id").is_in(plot_ids))
        .join(support, on=["vza_cell", "raa_cell"], how="inner")
        .group_by("plot_id")
        .agg(*[pl.col(band).mean().alias(f"{band}_off") for band in BANDS])
    )
    return (
        targets.join(geometry, on="plot_id", how="left")
        .join(nadir, on="plot_id", how="left")
        .join(off_nadir, on="plot_id", how="left")
        .sort("plot_id")
    )


def imputed_compact_arrays(features, train_idx, test_idx):
    source_cols = [f"{band}_{view}" for view in ("nadir", "off") for band in BANDS]
    values = features.select(source_cols).to_numpy()
    if not np.isfinite(values[np.asarray(train_idx), : len(BANDS)].astype(float)).any():
        raise RuntimeError("no observed nadir reflectance in training fold")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train = imputer.fit_transform(values[train_idx])
    test = imputer.transform(values[test_idx])
    n_bands = len(BANDS)
    train_nadir, train_off = train[:, :n_bands], train[:, n_bands:]
    test_nadir, test_off = test[:, :n_bands], test[:, n_bands:]
    train_multi = np.column_stack([train_nadir, train_off - train_nadir])
    test_multi = np.column_stack([test_nadir, test_off - test_nadir])
    return train_nadir, test_nadir, train_multi, test_multi


def evaluate_effect(cells, geometry, targets, seed):
    roster = targets.join(geometry, on="plot_id", how="left").sort("plot_id")
    splits = build_splits(roster, seed=seed)
    y = roster[TARGET_COL].to_numpy()
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        training_plot_ids = roster["plot_id"].gather(train_idx).to_list()
        features = compact_features(cells, geometry, targets, training_plot_ids)
        train_nadir, test_nadir, train_multi, test_multi = imputed_compact_arrays(
            features, train_idx, test_idx
        )
        geometry_values = features.select(GEOMETRY_COLS).to_numpy()
        train_nadir, test_nadir = residualize(
            train_nadir, test_nadir, geometry_values[train_idx], geometry_values[test_idx]
        )
        train_multi, test_multi = residualize(
            train_multi, test_multi, geometry_values[train_idx], geometry_values[test_idx]
        )
        scores = {}
        for name, train_x, test_x in [
            ("nadir", train_nadir, test_nadir),
            ("multiangular", train_multi, test_multi),
        ]:
            model = classifier()
            model.fit(train_x, y[train_idx])
            scores[name] = roc_auc_score(y[test_idx], model.predict_proba(test_x)[:, 1])
        fold_rows.append(
            {
                "fold": fold,
                "nadir_AUROC": scores["nadir"],
                "multiangular_AUROC": scores["multiangular"],
                "delta_AUROC": scores["multiangular"] - scores["nadir"],
            }
        )
    folds = pl.DataFrame(fold_rows)
    return {
        "n_paired_folds": folds.height,
        "baseline_AUROC_mean": float(folds["nadir_AUROC"].mean()),
        "comparator_AUROC_mean": float(folds["multiangular_AUROC"].mean()),
        "delta_AUROC_mean": float(folds["delta_AUROC"].mean()),
        "folds_improved": int((folds["delta_AUROC"] > 0).sum()),
    }


def permuted_targets(targets, rng):
    labels = targets[TARGET_COL].to_numpy().copy()
    rng.shuffle(labels)
    return targets.with_columns(pl.Series(TARGET_COL, labels))


def duration_summary(durations):
    values = np.asarray(durations, dtype=float)
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def run_analysis(n_repeats=50, n_permutations=200, seed=SEED):
    log_path = setup_logging()
    total_started = time.time()

    load_started = time.time()
    targets = load_targets()
    cells, geometry = load_or_build_week_cells(PREDICTOR_WEEK)
    logging.info("[PHASE] data loading: %.1fs", time.time() - load_started)

    observed_started = time.time()
    observed = evaluate_effect(cells, geometry, targets, seed)
    observed_delta = observed["delta_AUROC_mean"]
    logging.info(
        "Observed effect: nadir=%.3f multiangular=%.3f delta=%+.3f",
        observed["baseline_AUROC_mean"],
        observed["comparator_AUROC_mean"],
        observed_delta,
    )
    logging.info(
        "[PHASE] observed model fitting and prediction: %.1fs", time.time() - observed_started
    )

    repeat_started = time.time()
    repeat_rows = []
    repeat_durations = []
    for repeat in range(n_repeats):
        iteration_started = time.time()
        split_seed = seed + repeat
        effect = evaluate_effect(cells, geometry, targets, split_seed)
        repeat_rows.append(
            {
                "repeat": repeat,
                "seed": split_seed,
                "nadir_AUROC": effect["baseline_AUROC_mean"],
                "multiangular_AUROC": effect["comparator_AUROC_mean"],
                "delta_AUROC": effect["delta_AUROC_mean"],
                "folds_improved": effect["folds_improved"],
            }
        )
        repeat_durations.append(time.time() - iteration_started)
    repeat_df = pl.DataFrame(repeat_rows)
    repeat_timing = duration_summary(repeat_durations)
    logging.info(
        "Repeated CV times: min=%.2fs median=%.2fs mean=%.2fs max=%.2fs",
        repeat_timing["min"],
        repeat_timing["median"],
        repeat_timing["mean"],
        repeat_timing["max"],
    )
    logging.info("[PHASE] repeated CV: %.1fs", time.time() - repeat_started)

    permutation_started = time.time()
    rng = np.random.default_rng(seed)
    permutation_rows = []
    permutation_durations = []
    failures = 0
    for permutation in range(n_permutations):
        iteration_started = time.time()
        try:
            effect = evaluate_effect(cells, geometry, permuted_targets(targets, rng), seed)
        except RuntimeError as exc:
            failures += 1
            logging.warning("Permutation %d skipped: %s", permutation, exc)
            continue
        permutation_rows.append(
            {"permutation": permutation, "delta_AUROC": effect["delta_AUROC_mean"]}
        )
        permutation_durations.append(time.time() - iteration_started)
    if not permutation_rows:
        raise RuntimeError("No valid label permutations completed")
    permutation_df = pl.DataFrame(permutation_rows)
    permutation_timing = duration_summary(permutation_durations)
    logging.info(
        "Permutation times: min=%.2fs median=%.2fs mean=%.2fs max=%.2fs",
        permutation_timing["min"],
        permutation_timing["median"],
        permutation_timing["mean"],
        permutation_timing["max"],
    )
    logging.info(
        "[PHASE] permutation fitting and prediction: %.1fs", time.time() - permutation_started
    )

    null_deltas = permutation_df["delta_AUROC"].to_numpy()
    permutation_p_multi = float(
        (1 + np.sum(null_deltas >= observed_delta)) / (1 + len(null_deltas))
    )
    permutation_p_nadir = float(
        (1 + np.sum(null_deltas <= observed_delta)) / (1 + len(null_deltas))
    )
    permutation_p_two_sided = float(
        (1 + np.sum(np.abs(null_deltas) >= abs(observed_delta))) / (1 + len(null_deltas))
    )
    repeat_deltas = repeat_df["delta_AUROC"].to_numpy()
    summary = pl.DataFrame(
        [
            {
                "predictor_week": PREDICTOR_WEEK,
                "target_week": 8,
                "support_threshold": SUPPORT_THRESHOLD,
                "n_plots": targets.height,
                "n_positive": int(targets[TARGET_COL].sum()),
                "n_negative": int(targets.height - targets[TARGET_COL].sum()),
                "observed_nadir_AUROC": observed["baseline_AUROC_mean"],
                "observed_multiangular_AUROC": observed["comparator_AUROC_mean"],
                "observed_delta_AUROC": observed_delta,
                "n_repeats": n_repeats,
                "repeated_delta_mean": float(np.mean(repeat_deltas)),
                "repeated_delta_sd": float(np.std(repeat_deltas, ddof=1)),
                "repeated_delta_p2_5": float(np.percentile(repeat_deltas, 2.5)),
                "repeated_delta_p97_5": float(np.percentile(repeat_deltas, 97.5)),
                "repeat_fraction_positive": float(np.mean(repeat_deltas > 0)),
                "n_valid_permutations": len(null_deltas),
                "n_failed_permutations": failures,
                "permutation_p_multi_superiority": permutation_p_multi,
                "permutation_p_nadir_superiority": permutation_p_nadir,
                "permutation_p_two_sided": permutation_p_two_sided,
            }
        ]
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    repeat_path = RESULTS_DIR / "fair_multiangular_repeated_cv.csv"
    permutation_path = RESULTS_DIR / "fair_multiangular_permutations.csv"
    summary_path = RESULTS_DIR / "fair_multiangular_significance_summary.csv"
    repeat_df.write_csv(repeat_path)
    permutation_df.write_csv(permutation_path)
    summary.write_csv(summary_path)
    report_path = write_report(
        summary.row(0, named=True),
        repeat_path,
        permutation_path,
        summary_path,
        log_path,
        time.time() - total_started,
        seed,
    )
    logging.info("[PHASE] total: %.1fs", time.time() - total_started)
    logging.info("Report: %s", report_path)
    return summary, repeat_df, permutation_df


def write_report(row, repeat_path, permutation_path, summary_path, log_path, total_time, seed):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "fair_multiangular_significance_summary.md"
    if row["observed_delta_AUROC"] > 0 and row["permutation_p_multi_superiority"] < 0.05:
        interpretation = "The full-pipeline permutation test supports an incremental week-5 multiangular effect beyond adjusted nadir."
    elif row["observed_delta_AUROC"] < 0 and row["permutation_p_nadir_superiority"] < 0.05:
        interpretation = "The full-pipeline permutation test supports better week-5 performance from adjusted nadir than adjusted multiangular reflectance."
    elif row["observed_delta_AUROC"] < 0:
        interpretation = "Multiangular reflectance performed consistently worse across repeated splits, but the permutation evidence for nadir superiority did not reach p < 0.05."
    else:
        interpretation = "The estimated week-5 multiangular gain is not statistically conclusive."
    text = f"""## Results: Fair Multiangular Significance Test

| N plots | Positive | Negative | Adjusted nadir AUROC | Adjusted multiangular AUROC | Delta AUROC | Repeated delta mean | Repeated delta SD | Repeated 2.5%-97.5% | Positive repeats | p multi > nadir | p nadir > multi | Two-sided p |
|--------:|---------:|---------:|----------------------:|----------------------------:|-------------:|--------------------:|------------------:|--------------------:|-----------------:|----------------:|----------------:|------------:|
| {row['n_plots']} | {row['n_positive']} | {row['n_negative']} | {row['observed_nadir_AUROC']:.3f} | {row['observed_multiangular_AUROC']:.3f} | {row['observed_delta_AUROC']:+.3f} | {row['repeated_delta_mean']:+.3f} | {row['repeated_delta_sd']:.3f} | [{row['repeated_delta_p2_5']:+.3f}, {row['repeated_delta_p97_5']:+.3f}] | {row['repeat_fraction_positive']:.1%} | {row['permutation_p_multi_superiority']:.4f} | {row['permutation_p_nadir_superiority']:.4f} | {row['permutation_p_two_sided']:.4f} |

**Interpretation**: {interpretation} Repeated cross-validation measures split sensitivity and does not create additional independent samples.

## Outputs

- `{repeat_path}`
- `{permutation_path}`
- `{summary_path}`
- `{report_path}`

## Reproducibility

- Predictor week: {PREDICTOR_WEEK}
- Target week: 8
- Support threshold: {SUPPORT_THRESHOLD:.0%}
- Repeated nested CV runs: {row['n_repeats']}
- Valid full-pipeline label permutations: {row['n_valid_permutations']}
- Failed permutations: {row['n_failed_permutations']}
- Support selection: label-independent off-nadir coverage selected within each training fold
- Nadir model: 5 geometry-adjusted band means from VZA < 15 degrees
- Multiangular model: the same 5 nadir bands plus 5 geometry-adjusted off-nadir-minus-nadir band contrasts
- Seed: {seed}
- Log: `{log_path}`
- Runtime: {total_time:.1f}s
"""
    with report_path.open("w") as handle:
        handle.write(text)
    return report_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    run_analysis(args.repeats, args.permutations, args.seed)


if __name__ == "__main__":
    main()
