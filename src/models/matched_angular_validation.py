#!/usr/bin/env python3
"""Validate multiangular signal after matching angular sampling across plots.

Raw pixels are aggregated into narrow VZA/RAA cells. Only cells represented in
every plot of a predictor week are retained, and each retained cell receives
equal weight when constructing broad-angle reflectance features. The analysis
also evaluates a geometry-only baseline and reflectance residuals after fitting
geometry effects inside each training fold.
"""

import logging
import math
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJ = Path(__file__).resolve().parent.parent.parent
FEATURE_DIR = PROJ / "outputs" / "features"
RESULTS_DIR = PROJ / "outputs" / "results"
REPORTS_DIR = PROJ / "outputs" / "reports"
FIGURES_DIR = PROJ / "outputs" / "figures"
LOGS_DIR = PROJ / "outputs" / "logs"
CACHE_DIR = PROJ / "outputs" / "cache" / "matched_angular"

WEEK_DIRS = {
    0: Path("/run/media/davidem/Heim/2024/20240603_week0/metashape/20241205_products_uav_data/output/extract/polygon_df"),
    3: Path("/run/media/davidem/Heim/2024/20240624_week3/metashape/20241206_week3_products_uav_data/output/plots"),
    5: Path("/run/media/davidem/Heim/2024/20240715_week5/metashape/20241207_week5_products_uav_data/output/plots"),
}
BANDS = [f"band{i}" for i in range(1, 6)]
VZA_STEP = 2
RAA_STEP = 15
MIN_CELL_PIXELS = 500
ANGLE_ZONES = [(0, 15), (15, 25), (25, 35), (35, 45), (45, 60)]
TARGET_YEAR = 2024
TARGET_WEEK = 8
TARGET_COL = "future_disease_wk8"
SEED = 42
MAX_SPLITS = 5

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"matched_angular_validation_{TIMESTAMP}.log"


def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )
    logging.info("Log file: %s", LOG_FILE)


def log_phase(name, started):
    elapsed = time.time() - started
    logging.info("[PHASE] %s: %.1fs", name, elapsed)
    return elapsed


def load_targets():
    started = time.time()
    metadata = pl.read_parquet(FEATURE_DIR / "M0_metadata.parquet")
    target = (
        metadata.filter((pl.col("year") == TARGET_YEAR) & (pl.col("week") == TARGET_WEEK))
        .select(["plot_id", "cult", "trt", "disease_label"])
        .drop_nulls("disease_label")
        .unique()
        .with_columns(pl.col("disease_label").cast(pl.Int64).alias(TARGET_COL))
        .drop("disease_label")
    )
    if target.is_empty() or target[TARGET_COL].n_unique() < 2:
        raise RuntimeError("Observed week-8 disease labels are missing or contain fewer than two classes")
    logging.info(
        "Observed targets: %d plots, positive_rate=%.3f",
        target.height,
        target[TARGET_COL].mean(),
    )
    log_phase("target loading", started)
    return target


def _raa_expr():
    return (((pl.col("saa").cast(pl.Float64) - pl.col("vaa").cast(pl.Float64) + 180) % 360) - 180).abs()


def aggregate_plot(path):
    started = time.time()
    lf = (
        pl.scan_parquet(path)
        .select(BANDS + ["vza", "vaa", "saa", "path"])
        .filter(
            pl.col("vza").is_finite()
            & (pl.col("vza") >= 0)
            & (pl.col("vza") < 60)
            & pl.all_horizontal([pl.col(b).is_finite() & (pl.col(b) > 0) for b in BANDS])
        )
        .with_columns(_raa_expr().alias("raa"))
        .filter(pl.col("raa").is_finite() & (pl.col("raa") < 180))
        .with_columns(
            ((pl.col("vza") / VZA_STEP).floor() * VZA_STEP).cast(pl.Int16).alias("vza_cell"),
            ((pl.col("raa") / RAA_STEP).floor() * RAA_STEP).cast(pl.Int16).alias("raa_cell"),
        )
    )

    cell_query = (
        lf.group_by(["vza_cell", "raa_cell"])
        .agg(
            pl.len().alias("n_pixels"),
            pl.col("vza").mean().alias("vza_mean"),
            pl.col("raa").mean().alias("raa_mean"),
            *[pl.col(b).mean().alias(b) for b in BANDS],
        )
        .filter(pl.col("n_pixels") >= MIN_CELL_PIXELS)
        .sort(["vza_cell", "raa_cell"])
    )
    geometry_query = lf.select(
        pl.len().alias("n_pixels"),
        pl.col("path").n_unique().alias("n_images"),
        pl.col("vza").mean().alias("vza_mean"),
        pl.col("vza").std().alias("vza_std"),
        pl.col("vza").min().alias("vza_min"),
        pl.col("vza").max().alias("vza_max"),
        pl.col("raa").mean().alias("raa_mean"),
        pl.col("raa").std().alias("raa_std"),
    )
    cells, geometry = pl.collect_all([cell_query, geometry_query], engine="streaming")
    logging.info(
        "  %s: %.1fs, %d shared-cell candidates, %d pixels",
        path.name,
        time.time() - started,
        cells.height,
        geometry["n_pixels"].item(),
    )
    return path.stem, cells, geometry.row(0, named=True), time.time() - started


def load_or_build_week_cells(week):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cell_cache = CACHE_DIR / f"week{week}_cells.parquet"
    geometry_cache = CACHE_DIR / f"week{week}_geometry.parquet"
    if cell_cache.exists() and geometry_cache.exists():
        started = time.time()
        cells = pl.read_parquet(cell_cache)
        geometry = pl.read_parquet(geometry_cache)
        logging.info("week %d: loaded cached cell aggregates", week)
        log_phase(f"week {week} cache loading", started)
        return cells, geometry

    raw_dir = WEEK_DIRS[week]
    files = sorted(raw_dir.glob("plot_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No plot parquet files in {raw_dir}")

    started = time.time()
    cell_parts = []
    geometry_rows = []
    read_times = []
    for path in files:
        plot_id, cells, geometry, elapsed = aggregate_plot(path)
        cell_parts.append(cells.with_columns(pl.lit(plot_id).alias("plot_id")))
        geometry_rows.append({"plot_id": plot_id, **geometry})
        read_times.append(elapsed)

    cells = pl.concat(cell_parts, how="diagonal_relaxed")
    geometry = pl.DataFrame(geometry_rows)
    cells.write_parquet(cell_cache)
    geometry.write_parquet(geometry_cache)
    logging.info(
        "[PHASE] week %d file processing: min=%.1fs median=%.1fs mean=%.1fs max=%.1fs",
        week,
        np.min(read_times),
        np.median(read_times),
        np.mean(read_times),
        np.max(read_times),
    )
    log_phase(f"week {week} raw aggregation", started)
    return cells, geometry


def common_cells(cells, plot_ids):
    n_plots = len(plot_ids)
    common = (
        cells.filter(pl.col("plot_id").is_in(plot_ids))
        .group_by(["vza_cell", "raa_cell"])
        .agg(pl.col("plot_id").n_unique().alias("n_plots"))
        .filter(pl.col("n_plots") == n_plots)
        .select(["vza_cell", "raa_cell"])
    )
    if common.is_empty():
        raise RuntimeError("No narrow angular cells are shared by every target plot")
    logging.info("  common angular support: %d cells across %d plots", common.height, n_plots)
    return common


def zone_for_vza(vza_cell):
    center = float(vza_cell) + VZA_STEP / 2
    for lo, hi in ANGLE_ZONES:
        if lo <= center < hi:
            return f"{lo}_{hi}"
    return None


def build_matched_features(week, cells, geometry, targets):
    started = time.time()
    plot_ids = sorted(targets["plot_id"].unique().to_list())
    common = common_cells(cells, plot_ids)
    matched = cells.filter(pl.col("plot_id").is_in(plot_ids)).join(
        common, on=["vza_cell", "raa_cell"], how="inner"
    )
    matched = matched.with_columns(
        pl.col("vza_cell")
        .map_elements(zone_for_vza, return_dtype=pl.Utf8)
        .alias("angle_zone")
    ).drop_nulls("angle_zone")

    profiles = matched.group_by(["plot_id", "angle_zone"]).agg(
        pl.len().alias("n_equal_cells"),
        *[pl.col(b).mean().alias(b) for b in BANDS],
    )

    rows = []
    for plot_id in plot_ids:
        row = {"plot_id": plot_id, "week": week}
        plot_profile = profiles.filter(pl.col("plot_id") == plot_id)
        zone_values = {}
        for item in plot_profile.iter_rows(named=True):
            zone = item["angle_zone"]
            zone_values[zone] = item
            for band in BANDS:
                row[f"{band}_matched_{zone}"] = item[band]
        nadir = zone_values.get("0_15")
        if nadir is not None:
            for zone in ["15_25", "25_35", "35_45", "45_60"]:
                if zone not in zone_values:
                    continue
                for band in BANDS:
                    row[f"{band}_contrast_{zone}"] = zone_values[zone][band] - nadir[band]
        rows.append(row)

    features = pl.DataFrame(rows).join(geometry, on="plot_id", how="inner").join(
        targets, on="plot_id", how="inner"
    )
    for lo, hi in ANGLE_ZONES:
        counts = (
            matched.filter(
                (pl.col("vza_cell") + VZA_STEP / 2 >= lo)
                & (pl.col("vza_cell") + VZA_STEP / 2 < hi)
            )
            .group_by("plot_id")
            .agg(pl.col("n_pixels").sum().alias(f"pixels_vza_{lo}_{hi}"))
        )
        features = features.join(counts, on="plot_id", how="left")

    logging.info("  week %d matched feature table: %d plots x %d columns", week, *features.shape)
    log_phase(f"week {week} matched feature building", started)
    return features, common


def build_splits(df, seed=SEED):
    y = df[TARGET_COL].to_numpy()
    counts = np.bincount(y.astype(int), minlength=2)
    groups = df["plot_id"].to_numpy()
    max_splits = min(MAX_SPLITS, int(counts.min()), df.height)
    for n_splits in range(max_splits, 1, -1):
        if np.unique(groups).size == df.height:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(splitter.split(np.zeros((df.height, 1)), y))
        else:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(splitter.split(np.zeros((df.height, 1)), y, groups=groups))
        if all(
            np.unique(y[train_idx]).size == 2 and np.unique(y[test_idx]).size == 2
            for train_idx, test_idx in splits
        ):
            return splits
    raise RuntimeError("Not enough observed labels for grouped CV with both classes in every fold")


def classifier():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=SEED)),
        ]
    )


def residualize(train_x, test_x, train_g, test_g):
    x_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    g_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train = x_imputer.fit_transform(train_x)
    x_test = x_imputer.transform(test_x)
    g_train = g_imputer.fit_transform(train_g)
    g_test = g_imputer.transform(test_g)
    g_scaler = StandardScaler()
    g_train = g_scaler.fit_transform(g_train)
    g_test = g_scaler.transform(g_test)
    ridge = Ridge(alpha=10.0)
    ridge.fit(g_train, x_train)
    train_prediction = ridge.predict(g_train)
    test_prediction = ridge.predict(g_test)
    if train_prediction.ndim == 1:
        train_prediction = train_prediction[:, np.newaxis]
        test_prediction = test_prediction[:, np.newaxis]
    return x_train - train_prediction, x_test - test_prediction


def evaluate_week(features):
    geometry_cols = [
        "n_pixels", "n_images", "vza_mean", "vza_std", "vza_min", "vza_max", "raa_mean", "raa_std",
        *[f"pixels_vza_{lo}_{hi}" for lo, hi in ANGLE_ZONES],
    ]
    nadir_cols = [f"{band}_matched_0_15" for band in BANDS if f"{band}_matched_0_15" in features.columns]
    angular_cols = [c for c in features.columns if "_matched_" in c]
    contrast_cols = [c for c in features.columns if "_contrast_" in c]
    feature_sets = {"G_geometry": (geometry_cols, False)}
    if nadir_cols:
        feature_sets["N_matched_nadir"] = (nadir_cols, False)
    if angular_cols:
        feature_sets["A_matched_absolute"] = (angular_cols, False)
        feature_sets["A_geometry_residual"] = (angular_cols, True)
    if contrast_cols:
        feature_sets["C_matched_contrast"] = (contrast_cols, False)
        feature_sets["C_geometry_residual"] = (contrast_cols, True)
    splits = build_splits(features)
    y = features[TARGET_COL].to_numpy()
    g = features.select(geometry_cols).to_numpy()
    rows = []
    predictions = []
    for name, (cols, use_residuals) in feature_sets.items():
        x = features.select(cols).to_numpy()
        for fold, (train_idx, test_idx) in enumerate(splits):
            y_train, y_test = y[train_idx], y[test_idx]
            if use_residuals:
                x_train, x_test = residualize(
                    x[train_idx], x[test_idx], g[train_idx], g[test_idx]
                )
            else:
                x_train, x_test = x[train_idx], x[test_idx]
            model = classifier()
            fit_started = time.time()
            model.fit(x_train, y_train)
            fit_time = time.time() - fit_started
            predict_started = time.time()
            probability = model.predict_proba(x_test)[:, 1]
            predicted = model.predict(x_test)
            predict_time = time.time() - predict_started
            rows.append(
                {
                    "week": features["week"].item(0),
                    "feature_set": name,
                    "fold": fold,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_features": len(cols),
                    "AUROC": roc_auc_score(y_test, probability),
                    "AUPRC": average_precision_score(y_test, probability),
                    "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
                    "fit_time_s": fit_time,
                    "predict_time_s": predict_time,
                }
            )
            for idx, prob in zip(test_idx, probability):
                predictions.append(
                    {
                        "week": features["week"].item(0),
                        "feature_set": name,
                        "fold": fold,
                        "plot_id": features["plot_id"][int(idx)],
                        TARGET_COL: int(y[idx]),
                        "probability": float(prob),
                    }
                )
    return pl.DataFrame(rows), pl.DataFrame(predictions)


def summarize(folds):
    return (
        folds.group_by(["week", "feature_set"])
        .agg(
            pl.len().alias("n_folds"),
            pl.col("n_test").sum().alias("n_test_predictions"),
            pl.col("n_features").first().alias("n_features"),
            pl.col("AUROC").mean().alias("AUROC_mean"),
            pl.col("AUROC").std().alias("AUROC_std"),
            pl.col("AUPRC").mean().alias("AUPRC_mean"),
            pl.col("balanced_accuracy").mean().alias("balanced_accuracy_mean"),
            pl.col("fit_time_s").sum().alias("fit_time_s"),
            pl.col("predict_time_s").sum().alias("predict_time_s"),
        )
        .sort(["week", "feature_set"])
    )


def fmt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.3f}"


def write_report(summary, common_support, skipped_weeks, outputs, total_time):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "matched_angular_validation_summary.md"
    lines = [
        "## Results: Matched-Angular Validation",
        "",
        "Reflectance was aggregated in 2-degree VZA by 15-degree RAA cells. Only cells present in every evaluated plot were retained, and cells were weighted equally inside each reporting VZA zone.",
        "The target is the observed week-8 `disease_label`; treatment is not used as a predictor or target source.",
        "",
        "| Week | Feature Set | Features | Folds | Test Predictions | AUROC | AUPRC | BalAcc |",
        "|------|-------------|----------|-------|------------------|-------|-------|--------|",
    ]
    for row in summary.iter_rows(named=True):
        lines.append(
            f"| {row['week']} | {row['feature_set']} | {row['n_features']} | {row['n_folds']} | "
            f"{row['n_test_predictions']} | {fmt(row['AUROC_mean'])} +/- {fmt(row['AUROC_std'])} | "
            f"{fmt(row['AUPRC_mean'])} | {fmt(row['balanced_accuracy_mean'])} |"
        )
    lines.extend(["", "### Common angular support", ""])
    for week, support in common_support.items():
        lines.append(f"- Week {week}: {support.height} shared 2-degree x 15-degree cells")
    for week, reason in skipped_weeks.items():
        lines.append(f"- Week {week}: not evaluable ({reason})")

    week0 = summary.filter(pl.col("week") == 0)

    def score(name):
        row = week0.filter(pl.col("feature_set") == name)
        return row["AUROC_mean"].item() if row.height else None

    geometry_auc = score("G_geometry")
    absolute_auc = score("A_matched_absolute")
    contrast_auc = score("C_matched_contrast")
    absolute_residual_auc = score("A_geometry_residual")
    contrast_residual_auc = score("C_geometry_residual")
    interpretation = (
        f"At week 0, geometry alone achieved AUROC {fmt(geometry_auc)}, while matched absolute "
        f"multiangular reflectance achieved {fmt(absolute_auc)} and matched angular contrast achieved "
        f"{fmt(contrast_auc)}. After removing geometry-predictable reflectance variation inside each "
        f"training fold, AUROC fell to {fmt(absolute_residual_auc)} for absolute reflectance and "
        f"{fmt(contrast_residual_auc)} for angular contrast. The previous perfect early result is therefore "
        "not robust to flight-geometry control and should not be presented as independent evidence of early "
        "disease prediction. Weeks 3 and 5 lacked angular cells shared by every plot under the strict matching rule."
    )

    lines.extend(
        [
            "",
            f"**Interpretation**: {interpretation}",
            "",
            "## Outputs",
            "",
        ]
    )
    for label, path in outputs.items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Year: {TARGET_YEAR}",
            f"- Predictor weeks: {sorted(WEEK_DIRS)}",
            f"- Target: observed disease_label at week {TARGET_WEEK}",
            f"- VZA cell width: {VZA_STEP} degrees",
            f"- RAA cell width: {RAA_STEP} degrees",
            f"- Minimum pixels per plot-cell: {MIN_CELL_PIXELS}",
            "- Cell weighting: equal weight per shared angular cell",
            "- CV: StratifiedGroupKFold by plot_id, identical folds for every feature set within a week",
            "- Classifier: LogisticRegression(C=0.1, class_weight='balanced')",
            "- Geometry residualization: Ridge(alpha=10), fit inside each training fold",
            f"- Seed: {SEED}",
            f"- Log: `{LOG_FILE}`",
            f"- Total runtime: {total_time:.1f}s",
        ]
    )
    with report_path.open("w") as handle:
        handle.write("\n".join(lines) + "\n")
    return report_path


def plot_results(summary):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    feature_sets = summary["feature_set"].unique(maintain_order=True).to_list()
    weeks = sorted(summary["week"].unique().to_list())
    x = np.arange(len(weeks))
    width = 0.12
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for index, name in enumerate(feature_sets):
        values = []
        for week in weeks:
            row = summary.filter((pl.col("week") == week) & (pl.col("feature_set") == name))
            values.append(row["AUROC_mean"].item() if row.height else np.nan)
        ax.bar(x + (index - (len(feature_sets) - 1) / 2) * width, values, width, label=name)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"week {week}" for week in weeks])
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC")
    ax.set_title("Observed week-8 disease prediction after angular-support matching")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = FIGURES_DIR / "matched_angular_validation_auroc.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def run_analysis():
    setup_logging()
    total_started = time.time()
    targets = load_targets()
    fold_parts = []
    prediction_parts = []
    common_support = {}
    skipped_weeks = {}
    feature_paths = {}

    for week in sorted(WEEK_DIRS):
        cells, geometry = load_or_build_week_cells(week)
        try:
            features, support = build_matched_features(week, cells, geometry, targets)
        except RuntimeError as exc:
            logging.warning("week %d skipped: %s", week, exc)
            skipped_weeks[week] = str(exc)
            continue
        common_support[week] = support
        feature_path = RESULTS_DIR / f"matched_angular_features_week{week}.parquet"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        features.write_parquet(feature_path)
        feature_paths[f"Matched features week {week}"] = feature_path
        fold_df, prediction_df = evaluate_week(features)
        fold_parts.append(fold_df)
        prediction_parts.append(prediction_df)

    if not fold_parts:
        raise RuntimeError("No predictor week had sufficient common angular support")
    folds = pl.concat(fold_parts)
    predictions = pl.concat(prediction_parts)
    summary = summarize(folds)
    fold_path = RESULTS_DIR / "matched_angular_validation_by_fold.csv"
    summary_path = RESULTS_DIR / "matched_angular_validation_summary.csv"
    prediction_path = RESULTS_DIR / "matched_angular_validation_predictions.csv"
    folds.write_csv(fold_path)
    summary.write_csv(summary_path)
    predictions.write_csv(prediction_path)
    figure_path = plot_results(summary)
    total_time = time.time() - total_started
    report_path = write_report(
        summary,
        common_support,
        skipped_weeks,
        {
            **feature_paths,
            "Fold results": fold_path,
            "Summary": summary_path,
            "Predictions": prediction_path,
            "Figure": figure_path,
        },
        total_time,
    )
    logging.info("[PHASE] total: %.1fs", total_time)
    logging.info("Report: %s", report_path)
    return folds, summary, report_path


def main():
    run_analysis()


if __name__ == "__main__":
    main()
