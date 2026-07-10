#!/usr/bin/env python3
"""Sensitivity analysis for relaxed angular-support thresholds.

Retains fine angular cells represented in at least 70%, 80%, 90%, or 100% of
plots within each observed target class. Missing reflectance cells are median
imputed inside each training fold without missingness indicators. A separate
presence-only baseline tests whether the pattern of available angles predicts
the outcome.
"""

import logging
import math
import time
from datetime import datetime
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from src.models.matched_angular_validation import (
    ANGLE_ZONES,
    BANDS,
    FIGURES_DIR,
    LOGS_DIR,
    MAX_SPLITS,
    RAA_STEP,
    REPORTS_DIR,
    RESULTS_DIR,
    SEED,
    TARGET_COL,
    VZA_STEP,
    WEEK_DIRS,
    build_splits,
    classifier,
    load_or_build_week_cells,
    load_targets,
    residualize,
)

SUPPORT_THRESHOLDS = [0.70, 0.80, 0.90, 1.00]
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"angular_support_sensitivity_{TIMESTAMP}.log"


def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )
    logging.info("Log file: %s", LOG_FILE)


def retained_cells(cells, targets, plot_ids, threshold):
    labelled = cells.filter(pl.col("plot_id").is_in(plot_ids)).join(
        targets.select(["plot_id", TARGET_COL]), on="plot_id", how="inner"
    )
    class_sizes = targets.filter(pl.col("plot_id").is_in(plot_ids)).group_by(TARGET_COL).len()
    required = {
        int(row[TARGET_COL]): math.ceil(threshold * row["len"])
        for row in class_sizes.iter_rows(named=True)
    }
    coverage = (
        labelled.group_by(["vza_cell", "raa_cell", TARGET_COL])
        .agg(pl.col("plot_id").n_unique().alias("n_plots"))
        .pivot(on=TARGET_COL, index=["vza_cell", "raa_cell"], values="n_plots")
        .fill_null(0)
    )
    for class_label, minimum in required.items():
        column = str(class_label)
        if column not in coverage.columns:
            return pl.DataFrame(schema={"vza_cell": pl.Int16, "raa_cell": pl.Int16})
        coverage = coverage.filter(pl.col(column) >= minimum)
    return coverage.select(["vza_cell", "raa_cell"]).sort(["vza_cell", "raa_cell"])


def retained_cells_overall(cells, plot_ids, threshold):
    minimum = math.ceil(threshold * len(plot_ids))
    return (
        cells.filter(pl.col("plot_id").is_in(plot_ids))
        .group_by(["vza_cell", "raa_cell"])
        .agg(pl.col("plot_id").n_unique().alias("n_plots"))
        .filter(pl.col("n_plots") >= minimum)
        .select(["vza_cell", "raa_cell"])
        .sort(["vza_cell", "raa_cell"])
    )


def cell_name(vza, raa):
    return f"v{int(vza):02d}_r{int(raa):03d}"


def wide_band_table(matched, support, band):
    mapping = {
        (row["vza_cell"], row["raa_cell"]): f"{band}_{cell_name(row['vza_cell'], row['raa_cell'])}"
        for row in support.iter_rows(named=True)
    }
    table = matched.select(["plot_id", "vza_cell", "raa_cell", band]).with_columns(
        pl.struct(["vza_cell", "raa_cell"])
        .map_elements(
            lambda value: mapping[(value["vza_cell"], value["raa_cell"])],
            return_dtype=pl.Utf8,
        )
        .alias("feature")
    )
    return table.pivot(on="feature", index="plot_id", values=band)


def build_threshold_features(
    week,
    cells,
    geometry,
    targets,
    threshold,
    support_plot_ids=None,
    class_balanced=True,
):
    started = time.time()
    plot_ids = sorted(targets["plot_id"].unique().to_list())
    support_plot_ids = plot_ids if support_plot_ids is None else support_plot_ids
    support = (
        retained_cells(cells, targets, support_plot_ids, threshold)
        if class_balanced
        else retained_cells_overall(cells, support_plot_ids, threshold)
    )
    if support.is_empty():
        raise RuntimeError("no cells satisfy class-balanced support threshold")

    matched = cells.filter(pl.col("plot_id").is_in(plot_ids)).join(
        support, on=["vza_cell", "raa_cell"], how="inner"
    )
    base = targets.filter(pl.col("plot_id").is_in(plot_ids)).join(
        geometry, on="plot_id", how="left"
    )

    feature_table = base
    absolute_cols = []
    for band in BANDS:
        wide = wide_band_table(matched, support, band)
        absolute_cols.extend(c for c in wide.columns if c != "plot_id")
        feature_table = feature_table.join(wide, on="plot_id", how="left")

    presence = (
        matched.select(["plot_id", "vza_cell", "raa_cell"])
        .with_columns(
            pl.struct(["vza_cell", "raa_cell"])
            .map_elements(
                lambda value: f"present_{cell_name(value['vza_cell'], value['raa_cell'])}",
                return_dtype=pl.Utf8,
            )
            .alias("feature"),
            pl.lit(1.0).alias("present"),
        )
        .pivot(on="feature", index="plot_id", values="present")
    )
    presence_cols = [c for c in presence.columns if c != "plot_id"]
    feature_table = feature_table.join(presence, on="plot_id", how="left").with_columns(
        [pl.col(c).fill_null(0.0) for c in presence_cols]
    )

    candidate_nadir_cols = [f"{band}_nadir_reference" for band in BANDS]
    nadir_table = (
        cells.filter(pl.col("plot_id").is_in(plot_ids) & (pl.col("vza_cell") + VZA_STEP / 2 < 15))
        .group_by("plot_id")
        .agg(*[pl.col(band).mean().alias(f"{band}_nadir_reference") for band in BANDS])
    )
    feature_table = feature_table.join(nadir_table, on="plot_id", how="left")
    nadir_cols = candidate_nadir_cols if nadir_table.height else []
    off_nadir_keys = {
        cell_name(row["vza_cell"], row["raa_cell"])
        for row in support.filter(pl.col("vza_cell") + VZA_STEP / 2 >= 15).iter_rows(named=True)
    }
    contrast_cols = []
    contrast_sources = {}
    for band in BANDS:
        band_nadir = [f"{band}_nadir_reference"] if nadir_cols else []
        if not band_nadir:
            continue
        for col in [
            c
            for c in absolute_cols
            if c.startswith(f"{band}_") and c.split("_", 1)[1] in off_nadir_keys
        ]:
            contrast_col = col.replace(f"{band}_", f"{band}_contrast_", 1)
            contrast_cols.append(contrast_col)
            contrast_sources[contrast_col] = {"off_nadir": col, "nadir": band_nadir}

    geometry_cols = [
        "n_pixels",
        "n_images",
        "vza_mean",
        "vza_std",
        "vza_min",
        "vza_max",
        "raa_mean",
        "raa_std",
    ]
    missing_fraction = float(
        feature_table.select(absolute_cols).null_count().to_numpy().sum()
        / (feature_table.height * len(absolute_cols))
    )
    logging.info(
        "  week %d threshold %.0f%%: %d plots, %d cells, %d absolute features, missing=%.3f",
        week,
        threshold * 100,
        feature_table.height,
        support.height,
        len(absolute_cols),
        missing_fraction,
    )
    logging.info(
        "[PHASE] week %d threshold %.0f%% features: %.1fs",
        week,
        threshold * 100,
        time.time() - started,
    )
    return (
        feature_table,
        {
            "geometry": geometry_cols,
            "presence": presence_cols,
            "nadir": nadir_cols,
            "absolute": absolute_cols,
            "contrast": contrast_cols,
            "contrast_sources": contrast_sources,
        },
        support.height,
        missing_fraction,
    )


def feature_sets(columns):
    sets = {
        "G_geometry": (columns["geometry"], False),
        "P_presence": (columns["presence"], False),
        "A_fine_absolute": (columns["absolute"], False),
        "A_geometry_residual": (columns["absolute"], True),
    }
    if columns["nadir"]:
        sets["N_fine_nadir"] = (columns["nadir"], False)
        sets["N_geometry_residual"] = (columns["nadir"], True)
    if columns["contrast"]:
        sets["C_fine_contrast"] = (columns["contrast"], False)
        sets["C_geometry_residual"] = (columns["contrast"], True)
    return sets


def fold_contrast_arrays(features, contrast_sources, train_idx, test_idx):
    """Impute source reflectances on training data before deriving contrasts."""
    source_cols = sorted(
        {
            source
            for spec in contrast_sources.values()
            for source in [spec["off_nadir"], *spec["nadir"]]
        }
    )
    source_index = {name: index for index, name in enumerate(source_cols)}
    source_values = features.select(source_cols).to_numpy().astype(float)
    nadir_cols = sorted({column for spec in contrast_sources.values() for column in spec["nadir"]})
    nadir_indices = [source_index[column] for column in nadir_cols]
    if not np.isfinite(source_values[np.ix_(np.asarray(train_idx), nadir_indices)]).any():
        raise RuntimeError("no observed nadir reflectance in training fold")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_sources = imputer.fit_transform(source_values[train_idx])
    test_sources = imputer.transform(source_values[test_idx])

    def derive(values):
        columns = []
        for name in contrast_sources:
            spec = contrast_sources[name]
            off_nadir = values[:, source_index[spec["off_nadir"]]]
            nadir = values[:, [source_index[column] for column in spec["nadir"]]].mean(axis=1)
            columns.append(off_nadir - nadir)
        return np.column_stack(columns)

    return derive(train_sources), derive(test_sources)


def evaluate_split(
    features,
    columns,
    train_idx,
    test_idx,
    week,
    threshold,
    fold,
    n_cells,
    missing_fraction,
):
    sets = feature_sets(columns)
    y = features[TARGET_COL].to_numpy()
    geometry = features.select(columns["geometry"]).to_numpy()
    rows = []
    for name, (feature_cols, use_residuals) in sets.items():
        if not feature_cols:
            continue
        is_contrast = name in {"C_fine_contrast", "C_geometry_residual"}
        if is_contrast:
            try:
                x_train, x_test = fold_contrast_arrays(
                    features,
                    columns["contrast_sources"],
                    train_idx,
                    test_idx,
                )
            except RuntimeError as exc:
                logging.warning(
                    "week %d threshold %.0f%% fold %d skipped %s: %s",
                    week,
                    threshold * 100,
                    fold,
                    name,
                    exc,
                )
                continue
        else:
            x = features.select(feature_cols).to_numpy().astype(float)
            x_train, x_test = x[train_idx], x[test_idx]
            if name in {"N_fine_nadir", "N_geometry_residual"} and not np.isfinite(x_train).any():
                logging.warning(
                    "week %d threshold %.0f%% fold %d skipped %s: no observed nadir reflectance",
                    week,
                    threshold * 100,
                    fold,
                    name,
                )
                continue
        if use_residuals:
            x_train, x_test = residualize(x_train, x_test, geometry[train_idx], geometry[test_idx])
        model = classifier()
        fit_started = time.time()
        model.fit(x_train, y[train_idx])
        fit_time = time.time() - fit_started
        predict_started = time.time()
        probability = model.predict_proba(x_test)[:, 1]
        predicted = model.predict(x_test)
        predict_time = time.time() - predict_started
        rows.append(
            {
                "week": week,
                "support_threshold": threshold,
                "feature_set": name,
                "fold": fold,
                "n_plots": features.height,
                "n_cells": n_cells,
                "n_features": len(feature_cols),
                "missing_fraction": (
                    missing_fraction if name not in {"G_geometry", "P_presence"} else 0.0
                ),
                "AUROC": roc_auc_score(y[test_idx], probability),
                "AUPRC": average_precision_score(y[test_idx], probability),
                "balanced_accuracy": balanced_accuracy_score(y[test_idx], predicted),
                "fit_time_s": fit_time,
                "predict_time_s": predict_time,
            }
        )
    return rows


def evaluate(features, columns, week, threshold, n_cells, missing_fraction, seed=SEED):
    splits = build_splits(features, seed=seed)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        rows.extend(
            evaluate_split(
                features,
                columns,
                train_idx,
                test_idx,
                week,
                threshold,
                fold,
                n_cells,
                missing_fraction,
            )
        )
    return pl.DataFrame(rows)


def evaluate_nested_support(
    week,
    cells,
    geometry,
    targets,
    threshold,
    seed=SEED,
    class_balanced=True,
):
    roster = targets.join(geometry, on="plot_id", how="left").sort("plot_id")
    splits = build_splits(roster, seed=seed)
    rows = []
    skipped_folds = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        train_plot_ids = roster["plot_id"].gather(train_idx).to_list()
        try:
            features, columns, n_cells, missing_fraction = build_threshold_features(
                week,
                cells,
                geometry,
                targets,
                threshold,
                support_plot_ids=train_plot_ids,
                class_balanced=class_balanced,
            )
        except RuntimeError as exc:
            skipped_folds.append(f"fold {fold}: {exc}")
            continue
        features = features.sort("plot_id")
        rows.extend(
            evaluate_split(
                features,
                columns,
                train_idx,
                test_idx,
                week,
                threshold,
                fold,
                n_cells,
                missing_fraction,
            )
        )
    if not rows:
        detail = "; ".join(skipped_folds) or "no valid folds"
        raise RuntimeError(detail)
    if skipped_folds:
        logging.warning(
            "week %d threshold %.0f%% skipped support selection in %s",
            week,
            threshold * 100,
            "; ".join(skipped_folds),
        )
    return pl.DataFrame(rows)


def summarize(folds):
    return (
        folds.group_by(["week", "support_threshold", "feature_set"])
        .agg(
            pl.len().alias("n_folds"),
            pl.col("n_plots").first(),
            pl.col("n_cells").mean().alias("n_cells_mean"),
            pl.col("n_cells").min().alias("n_cells_min"),
            pl.col("n_cells").max().alias("n_cells_max"),
            pl.col("n_features").mean().alias("n_features_mean"),
            pl.col("n_features").min().alias("n_features_min"),
            pl.col("n_features").max().alias("n_features_max"),
            pl.col("missing_fraction").mean().alias("missing_fraction_mean"),
            pl.col("AUROC").mean().alias("AUROC_mean"),
            pl.col("AUROC").std().alias("AUROC_std"),
            pl.col("AUPRC").mean().alias("AUPRC_mean"),
            pl.col("balanced_accuracy").mean().alias("balanced_accuracy_mean"),
        )
        .sort(["week", "support_threshold", "feature_set"])
    )


def paired_multiangular_effects(folds):
    """Compare adjusted multiangular models against adjusted nadir on identical folds."""
    comparisons = {
        "A_geometry_residual": "N_geometry_residual",
        "C_geometry_residual": "N_geometry_residual",
    }
    rows = []
    for comparator, baseline in comparisons.items():
        comparator_rows = folds.filter(pl.col("feature_set") == comparator).select(
            ["week", "support_threshold", "fold", pl.col("AUROC").alias("comparator_AUROC")]
        )
        baseline_rows = folds.filter(pl.col("feature_set") == baseline).select(
            ["week", "support_threshold", "fold", pl.col("AUROC").alias("baseline_AUROC")]
        )
        paired = comparator_rows.join(
            baseline_rows,
            on=["week", "support_threshold", "fold"],
            how="inner",
        ).with_columns((pl.col("comparator_AUROC") - pl.col("baseline_AUROC")).alias("delta_AUROC"))
        for key, group in paired.group_by(["week", "support_threshold"], maintain_order=True):
            week, threshold = key
            deltas = group["delta_AUROC"].to_numpy()
            observed = float(np.mean(deltas))
            null_means = np.array(
                [
                    np.mean(deltas * np.asarray(signs))
                    for signs in product([-1.0, 1.0], repeat=len(deltas))
                ]
            )
            p_value = float(np.sum(null_means >= observed - 1e-12) / len(null_means))
            rows.append(
                {
                    "week": int(week),
                    "support_threshold": float(threshold),
                    "baseline": baseline,
                    "comparator": comparator,
                    "n_paired_folds": group.height,
                    "baseline_AUROC_mean": float(group["baseline_AUROC"].mean()),
                    "comparator_AUROC_mean": float(group["comparator_AUROC"].mean()),
                    "delta_AUROC_mean": observed,
                    "delta_AUROC_std": (
                        float(group["delta_AUROC"].std()) if group.height > 1 else None
                    ),
                    "folds_improved": int((group["delta_AUROC"] > 0).sum()),
                    "one_sided_sign_flip_p": p_value,
                }
            )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def fmt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.3f}"


def interpretation_from_results(summary, paired, skipped):
    statements = []
    skipped_by_week = {}
    for item in skipped:
        skipped_by_week.setdefault(item["week"], []).append(item["threshold"])
    evaluated_weeks = set(summary["week"].unique().to_list())
    for week in sorted(set(WEEK_DIRS) | evaluated_weeks):
        week_summary = summary.filter(pl.col("week") == week)
        if week_summary.is_empty():
            thresholds = skipped_by_week.get(week, [])
            detail = (
                f" at {', '.join(f'{value:.0%}' for value in sorted(thresholds))} support"
                if thresholds
                else ""
            )
            statements.append(f"Week {week} was not evaluable{detail}.")
            continue
        geometry_all = week_summary.filter(pl.col("feature_set") == "G_geometry")
        expected_folds = (
            int(geometry_all["n_folds"].max())
            if geometry_all.height
            else int(week_summary["n_folds"].max())
        )
        complete = week_summary.filter(pl.col("n_folds") == expected_folds)
        geometry = complete.filter(pl.col("feature_set") == "G_geometry")
        if geometry.height:
            statements.append(
                f"Week {week} geometry alone reached AUROC {fmt(geometry['AUROC_mean'].max())}."
            )
        fair = (
            paired.filter(
                (pl.col("week") == week)
                & (pl.col("comparator") == "A_geometry_residual")
                & (pl.col("n_paired_folds") == expected_folds)
            )
            if not paired.is_empty()
            else pl.DataFrame()
        )
        if fair.height:
            best = fair.sort("delta_AUROC_mean", descending=True).row(0, named=True)
            statements.append(
                f"At {best['support_threshold']:.0%} support, geometry-adjusted multiangular AUROC was "
                f"{best['comparator_AUROC_mean']:.3f} versus {best['baseline_AUROC_mean']:.3f} for "
                f"geometry-adjusted nadir (paired delta {best['delta_AUROC_mean']:+.3f}, "
                f"one-sided sign-flip p={best['one_sided_sign_flip_p']:.3f})."
            )
    return " ".join(statements)


def write_report(summary, paired, skipped, total_time):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "angular_support_sensitivity_summary.md"
    lines = [
        "## Results: Angular-Support Sensitivity",
        "",
        "Fine 2-degree VZA by 15-degree RAA cells were retained at class-balanced support thresholds of 70%, 80%, 90%, and 100%.",
        "Missing reflectance cells were median-imputed inside each training fold without missingness indicators. `P_presence` explicitly tests whether angular-cell availability alone predicts the target.",
        "",
        "| Week | Support | Model | Folds | Plots | Cells | Features | Missing | AUROC | AUPRC | BalAcc |",
        "|------|---------|-------|-------|-------|-------|----------|---------|-------|-------|--------|",
    ]
    for row in summary.iter_rows(named=True):
        lines.append(
            f"| {row['week']} | {row['support_threshold']:.0%} | {row['feature_set']} | {row['n_folds']} | "
            f"{row['n_plots']} | {row['n_cells_mean']:.1f} ({row['n_cells_min']}-{row['n_cells_max']}) | "
            f"{row['n_features_mean']:.1f} ({row['n_features_min']}-{row['n_features_max']}) | "
            f"{row['missing_fraction_mean']:.3f} | "
            f"{fmt(row['AUROC_mean'])} +/- {fmt(row['AUROC_std'])} | {fmt(row['AUPRC_mean'])} | "
            f"{fmt(row['balanced_accuracy_mean'])} |"
        )
    if skipped:
        lines.extend(["", "### Skipped combinations", ""])
        for item in skipped:
            lines.append(
                f"- Week {item['week']}, support {item['threshold']:.0%}: {item['reason']}"
            )
    if not paired.is_empty():
        lines.extend(
            [
                "",
                "### Paired incremental multiangular effect",
                "",
                "| Week | Support | Baseline | Comparator | Folds | Baseline AUROC | Comparator AUROC | Delta | Wins | One-sided p |",
                "|------|---------|----------|------------|-------|-----------------|------------------|-------|------|-------------|",
            ]
        )
        for row in paired.iter_rows(named=True):
            lines.append(
                f"| {row['week']} | {row['support_threshold']:.0%} | {row['baseline']} | {row['comparator']} | "
                f"{row['n_paired_folds']} | {row['baseline_AUROC_mean']:.3f} | "
                f"{row['comparator_AUROC_mean']:.3f} | {row['delta_AUROC_mean']:+.3f} | "
                f"{row['folds_improved']}/{row['n_paired_folds']} | {row['one_sided_sign_flip_p']:.3f} |"
            )

    interpretation = interpretation_from_results(summary, paired, skipped)

    lines.extend(
        [
            "",
            f"**Interpretation**: {interpretation}",
            "",
            "## Outputs",
            "",
            "- Fold results: `outputs/archive/legacy_unscoped/results/angular_support_sensitivity_by_fold.csv`",
            "- Summary: `outputs/archive/legacy_unscoped/results/angular_support_sensitivity_summary.csv`",
            "- Paired effect: `outputs/archive/legacy_unscoped/results/angular_support_paired_effect.csv`",
            "- Figure: `outputs/archive/legacy_unscoped/figures/angular_support_sensitivity_auroc.png`",
            "",
            "## Reproducibility",
            "",
            f"- Support thresholds: {SUPPORT_THRESHOLDS}",
            f"- Fine cells: {VZA_STEP}-degree VZA x {RAA_STEP}-degree RAA",
            "- Coverage requirement: threshold met separately within both observed target classes using training plots only",
            "- Feature selection: angular support is reselected independently inside every CV training fold",
            "- Missing values: median imputation fit within each CV training fold; no missingness indicators in reflectance models",
            "- CV: StratifiedGroupKFold by plot_id with identical folds across models within week/threshold",
            "- Classifier: LogisticRegression(C=0.1, class_weight='balanced')",
            "- Geometry residualization: Ridge(alpha=10), fit within each training fold",
            "- Primary comparison: paired fold AUROC for geometry-adjusted multiangular versus geometry-adjusted nadir",
            "- Uncertainty diagnostic: one-sided sign-flip test over paired fold differences",
            "- Caveat: CV folds have overlapping training sets, so the fold sign-flip p-value is descriptive rather than confirmatory",
            f"- Seed: {SEED}",
            f"- Log: `{LOG_FILE}`",
            f"- Total runtime: {total_time:.1f}s",
        ]
    )
    with path.open("w") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def plot_summary(summary):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    models = [
        "G_geometry",
        "P_presence",
        "N_fine_nadir",
        "N_geometry_residual",
        "A_fine_absolute",
        "C_fine_contrast",
        "A_geometry_residual",
        "C_geometry_residual",
    ]
    weeks = sorted(summary["week"].unique().to_list())
    fig, axes = plt.subplots(1, len(weeks), figsize=(6 * len(weeks), 5), squeeze=False)
    for axis, week in zip(axes[0], weeks):
        week_df = summary.filter(pl.col("week") == week)
        for model in models:
            sub = week_df.filter(pl.col("feature_set") == model).sort("support_threshold")
            if sub.is_empty():
                continue
            axis.plot(
                sub["support_threshold"].to_numpy() * 100,
                sub["AUROC_mean"].to_numpy(),
                marker="o",
                label=model,
            )
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"Week {week}")
        axis.set_xlabel("Required support (%)")
        axis.set_ylim(0, 1)
        axis.set_ylabel("AUROC")
    axes[0][-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    path = FIGURES_DIR / "angular_support_sensitivity_auroc.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def run_analysis():
    setup_logging()
    started = time.time()
    targets = load_targets()
    parts = []
    skipped = []
    for week in sorted(WEEK_DIRS):
        cells, geometry = load_or_build_week_cells(week)
        for threshold in SUPPORT_THRESHOLDS:
            try:
                parts.append(evaluate_nested_support(week, cells, geometry, targets, threshold))
            except (RuntimeError, ValueError) as exc:
                logging.warning("week %d threshold %.0f%% skipped: %s", week, threshold * 100, exc)
                skipped.append({"week": week, "threshold": threshold, "reason": str(exc)})
    if not parts:
        raise RuntimeError("No support-threshold evaluations were possible")
    folds = pl.concat(parts)
    summary = summarize(folds)
    paired = paired_multiangular_effects(folds)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    folds.write_csv(RESULTS_DIR / "angular_support_sensitivity_by_fold.csv")
    summary.write_csv(RESULTS_DIR / "angular_support_sensitivity_summary.csv")
    paired.write_csv(RESULTS_DIR / "angular_support_paired_effect.csv")
    plot_summary(summary)
    report = write_report(summary, paired, skipped, time.time() - started)
    logging.info("[PHASE] total: %.1fs", time.time() - started)
    logging.info("Report: %s", report)
    return folds, summary, report


def main():
    run_analysis()


if __name__ == "__main__":
    main()
