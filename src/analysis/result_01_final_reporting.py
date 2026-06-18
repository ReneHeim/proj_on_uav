#!/usr/bin/env python3
"""Build final paper-facing reports for Result 1 multiangular reflectance."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from src.analysis.result_01_reflectance_distributions import BANDS, DEFAULT_OUTPUT, SEED, style_axis

ROOT = Path(__file__).resolve().parents[2]
VZA_OUTPUT = DEFAULT_OUTPUT
RAA_OUTPUT = ROOT / "outputs/result_01_raa_sun_geometry"
FINAL_RESULTS = VZA_OUTPUT / "results"
FINAL_REPORTS = VZA_OUTPUT / "reports"
FINAL_FIGURES = VZA_OUTPUT / "figures/main"
MIN_HEADLINE_PLOTS = 10
YEARS = [2024, 2025]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vza-output", type=Path, default=VZA_OUTPUT)
    parser.add_argument("--raa-output", type=Path, default=RAA_OUTPUT)
    parser.add_argument("--filter-state", default="ground_filtered", choices=["ground_filtered", "unfiltered"])
    return parser.parse_args()


def configure_logging() -> Path:
    log_dir = ROOT / "outputs/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"result_01_final_reporting_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
        force=True,
    )
    return path


def log_phase(name: str, started: float) -> None:
    logging.info("[PHASE] %s: %.2fs", name, time.perf_counter() - started)


def save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".png", ".svg", ".tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix in {".png", ".tiff"}:
            kwargs["dpi"] = 600
        figure.savefig(stem.with_suffix(suffix), **kwargs)
    plt.close(figure)


def read_csv(path: Path) -> pl.DataFrame:
    started = time.perf_counter()
    frame = pl.read_csv(path)
    logging.info("[IO] read %s: %.3fs rows=%d", path, time.perf_counter() - started, frame.height)
    return frame


def vza_results_dir(base: Path, year: int, filter_state: str) -> Path:
    return base / str(year) / filter_state / "results"


def raa_results_dir(base: Path, year: int, filter_state: str) -> Path:
    return base / str(year) / filter_state / "results"


def build_vza_support(vza_output: Path, filter_state: str) -> pl.DataFrame:
    rows = []
    for year in YEARS:
        summary = read_csv(vza_results_dir(vza_output, year, filter_state) / "reflectance_by_vza_summary.csv")
        rows.append(
            summary.group_by("week", "vza_class", "vza_midpoint")
            .agg(
                pl.lit(year).alias("year"),
                pl.col("plots").min().alias("plots"),
                pl.col("observations").sum().alias("band_observations"),
            )
            .select("year", "week", "vza_class", "vza_midpoint", "plots", "band_observations")
        )
    support = pl.concat(rows).with_columns((pl.col("plots") < MIN_HEADLINE_PLOTS).alias("is_sparse"))
    return support.sort("year", "week", "vza_midpoint")


def build_raa_support(raa_output: Path, filter_state: str) -> pl.DataFrame:
    rows = []
    for year in YEARS:
        support = read_csv(raa_results_dir(raa_output, year, filter_state) / f"raa_support_by_week_vza_{year}.csv")
        rows.append(
            support.group_by("year", "week", "vza_class", "vza_midpoint", "raa_class", "raa_midpoint")
            .agg(
                pl.col("plots").min().alias("plots"),
                pl.col("pixels").sum().alias("pixels"),
                pl.col("images").sum().alias("images"),
            )
            .with_columns((pl.col("plots") < MIN_HEADLINE_PLOTS).alias("is_sparse"))
        )
    return pl.concat(rows).sort("year", "week", "vza_midpoint", "raa_midpoint")


def plot_angular_support(vza_support: pl.DataFrame, raa_support: pl.DataFrame) -> None:
    started = time.perf_counter()
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), constrained_layout=True)
    for col, year in enumerate(YEARS):
        data = vza_support.filter(pl.col("year") == year)
        matrix = (
            data.pivot(on="vza_class", index="week", values="plots", aggregate_function="mean")
            .sort("week")
        )
        vza_cols = sorted([c for c in matrix.columns if c != "week"], key=lambda x: int(x.split("-")[0]))
        image = axes[0, col].imshow(matrix.select(vza_cols).to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=24)
        axes[0, col].set_title(f"{year}: VZA support", fontweight="bold")
        axes[0, col].set_yticks(range(matrix.height), matrix["week"].to_list())
        axes[0, col].set_xticks(range(len(vza_cols)), vza_cols, rotation=45, ha="right")
        axes[0, col].set_ylabel("Week")
        axes[0, col].set_xlabel("VZA class")
        for row_i in range(matrix.height):
            for col_i, value in enumerate(matrix.select(vza_cols).to_numpy()[row_i]):
                axes[0, col].text(col_i, row_i, f"{int(value)}", ha="center", va="center", fontsize=7)
        raa = (
            raa_support.filter(pl.col("year") == year)
            .group_by("vza_class", "vza_midpoint", "raa_class", "raa_midpoint")
            .agg(pl.col("plots").median().alias("median_plots"))
            .sort("vza_midpoint", "raa_midpoint")
        )
        raa_matrix = raa.pivot(on="raa_class", index="vza_class", values="median_plots", aggregate_function="mean")
        raa_matrix = raa_matrix.sort("vza_class")
        raa_cols = sorted([c for c in raa_matrix.columns if c != "vza_class"], key=lambda x: int(x.split("-")[0]))
        axes[1, col].imshow(raa_matrix.select(raa_cols).to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=24)
        axes[1, col].set_title(f"{year}: median RAA support within VZA", fontweight="bold")
        axes[1, col].set_yticks(range(raa_matrix.height), raa_matrix["vza_class"].to_list())
        axes[1, col].set_xticks(range(len(raa_cols)), raa_cols, rotation=45, ha="right")
        axes[1, col].set_ylabel("VZA class")
        axes[1, col].set_xlabel("RAA class")
        vals = raa_matrix.select(raa_cols).to_numpy()
        for row_i in range(raa_matrix.height):
            for col_i, value in enumerate(vals[row_i]):
                axes[1, col].text(col_i, row_i, f"{value:.0f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axes, shrink=0.70, label="Plots")
    save_figure(figure, FINAL_FIGURES / "angular_support_heatmap")
    save_figure(figure_from_geometry(vza_support, raa_support), FINAL_FIGURES / "observation_geometry_distribution")
    log_phase("support_figures", started)


def figure_from_geometry(vza_support: pl.DataFrame, raa_support: pl.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for year in YEARS:
        data = (
            vza_support.filter(pl.col("year") == year)
            .group_by("vza_class", "vza_midpoint")
            .agg(pl.col("plots").mean().alias("mean_plots"))
            .sort("vza_midpoint")
        )
        axes[0].plot(data["vza_midpoint"].to_numpy(), data["mean_plots"].to_numpy(), marker="o", label=str(year))
    axes[0].axhline(MIN_HEADLINE_PLOTS, color="#8B1A1A", linestyle="--", linewidth=1.0, label="headline threshold")
    axes[0].set_title("VZA sampling support", loc="left", fontweight="bold")
    axes[0].set_xlabel("VZA midpoint (degrees)")
    axes[0].set_ylabel("Mean plots per week")
    style_axis(axes[0])
    axes[0].legend(frameon=False)
    for year in YEARS:
        data = (
            raa_support.filter(pl.col("year") == year)
            .group_by("raa_class", "raa_midpoint")
            .agg(pl.col("plots").mean().alias("mean_plots"))
            .sort("raa_midpoint")
        )
        axes[1].plot(data["raa_midpoint"].to_numpy(), data["mean_plots"].to_numpy(), marker="o", label=str(year))
    axes[1].axhline(MIN_HEADLINE_PLOTS, color="#8B1A1A", linestyle="--", linewidth=1.0, label="headline threshold")
    axes[1].set_title("RAA sampling support", loc="left", fontweight="bold")
    axes[1].set_xlabel("RAA midpoint (degrees)")
    axes[1].set_ylabel("Mean plots per VZA/week")
    style_axis(axes[1])
    axes[1].legend(frameon=False)
    return figure


def plot_vza_year_comparison(vza_output: Path, filter_state: str) -> None:
    started = time.perf_counter()
    figure, axes = plt.subplots(len(BANDS), len(YEARS), figsize=(8.8, 10.8), constrained_layout=True, sharex=True)
    contrasts_by_year = {}
    band_limits: dict[str, tuple[float, float]] = {}
    for col, year in enumerate(YEARS):
        contrasts = read_csv(vza_results_dir(vza_output, year, filter_state) / "matched_angular_contrasts.csv")
        contrasts_by_year[year] = contrasts.filter(pl.col("matched_plots") >= MIN_HEADLINE_PLOTS)
    for band in BANDS:
        limits = []
        for contrasts in contrasts_by_year.values():
            data = contrasts.filter(pl.col("band") == band)
            if data.is_empty():
                continue
            limits.extend([float(data["ci_low"].min()), float(data["ci_high"].max())])
        if limits:
            low, high = min(limits), max(limits)
            pad = max((high - low) * 0.08, 0.005)
            band_limits[band] = (low - pad, high + pad)
    for col, year in enumerate(YEARS):
        contrasts = contrasts_by_year[year]
        for row, (band, band_name) in enumerate(BANDS.items()):
            axis = axes[row, col]
            data = (
                contrasts.filter(pl.col("band") == band)
                .group_by("vza_class", "vza_midpoint")
                .agg(
                    pl.col("median_absolute_contrast").median().alias("median_contrast"),
                    pl.col("ci_low").median().alias("ci_low"),
                    pl.col("ci_high").median().alias("ci_high"),
                )
                .sort("vza_midpoint")
            )
            x = data["vza_midpoint"].to_numpy()
            y = data["median_contrast"].to_numpy()
            axis.plot(x, y, marker="o", linewidth=1.5, color="#176B6B")
            axis.fill_between(x, data["ci_low"].to_numpy(), data["ci_high"].to_numpy(), color="#76A9A9", alpha=0.22)
            axis.axhline(0, color="#333333", linewidth=0.8)
            if band in band_limits:
                axis.set_ylim(*band_limits[band])
            if row == 0:
                axis.set_title(str(year), fontweight="bold")
            if col == 0:
                axis.set_ylabel(f"{band_name}\ncontrast")
            if row == len(BANDS) - 1:
                axis.set_xlabel("VZA midpoint")
            style_axis(axis)
    figure.suptitle("Matched VZA contrasts by year", fontsize=12, fontweight="bold")
    save_figure(figure, FINAL_FIGURES / "vza_contrast_2024_2025_comparison")
    log_phase("vza_year_comparison", started)


def plot_workflow_schematic() -> None:
    started = time.perf_counter()
    figure, axis = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    axis.axis("off")
    boxes = [
        (0.06, 0.60, "UAV overlapping images\nnadir + off-nadir views"),
        (0.38, 0.76, "Nadir/orthomosaic workflow\nkeeps one blended view"),
        (0.38, 0.38, "Multiangular workflow\nkeeps repeated views"),
        (0.70, 0.76, "Reference/nadir features\nVZA 10-15"),
        (0.70, 0.38, "Geometry features\nVZA contrasts + RAA/phase"),
    ]
    for x, y, text in boxes:
        axis.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F4F7F7", "edgecolor": "#176B6B", "linewidth": 1.2},
        )
    arrows = [
        ((0.19, 0.60), (0.31, 0.76)),
        ((0.19, 0.60), (0.31, 0.38)),
        ((0.51, 0.76), (0.61, 0.76)),
        ((0.51, 0.38), (0.61, 0.38)),
    ]
    for start, end in arrows:
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#333333"})
    axis.text(
        0.50,
        0.10,
        "Result 1 tests whether off-nadir geometry changes reflectance. Result 2 should test disease prediction.",
        ha="center",
        va="center",
        fontsize=10,
        color="#333333",
    )
    save_figure(figure, FINAL_FIGURES / "multiangular_workflow_schematic")
    log_phase("workflow_schematic", started)


def build_evidence_table(vza_output: Path, raa_output: Path, filter_state: str) -> pl.DataFrame:
    rows = []
    for year in YEARS:
        vza = (
            read_csv(vza_results_dir(vza_output, year, filter_state) / "matched_angular_contrasts.csv")
            .filter(pl.col("matched_plots") >= MIN_HEADLINE_PLOTS)
            .with_columns(pl.col("median_absolute_contrast").abs().alias("magnitude"))
            .sort("magnitude", descending=True)
            .head(8)
        )
        vza_model = read_csv(vza_results_dir(vza_output, year, filter_state) / f"vza_model_comparison_{year}.csv")
        for record in vza.iter_rows(named=True):
            model = vza_model.filter(pl.col("band") == record["band"]).row(0, named=True)
            rows.append(
                {
                    "year": year,
                    "geometry": "VZA",
                    "band": record["band"],
                    "band_name": record["band_name"],
                    "week": record["week"],
                    "angle_comparison": f"{record['vza_class']} vs 10-15 VZA",
                    "median_contrast": record["median_absolute_contrast"],
                    "ci_low": record["ci_low"],
                    "ci_high": record["ci_high"],
                    "relative_contrast": record["median_relative_contrast"],
                    "matched_plots": record["matched_plots"],
                    "delta_r2": model["delta_r2_vza_vs_control"],
                    "delta_aic": model["delta_aic_vza_vs_control"],
                    "delta_bic": model["delta_bic_vza_vs_control"],
                    "interpretation": "Off-reference VZA changes plot-level reflectance versus the same plot/week reference view.",
                }
            )
        raa = (
            read_csv(raa_results_dir(raa_output, year, filter_state) / f"matched_raa_contrasts_{year}.csv")
            .filter(pl.col("matched_plots") >= MIN_HEADLINE_PLOTS)
            .with_columns(pl.col("median_absolute_contrast").abs().alias("magnitude"))
            .sort("magnitude", descending=True)
            .head(8)
        )
        raa_model = read_csv(raa_results_dir(raa_output, year, filter_state) / f"raa_vza_model_comparison_{year}.csv")
        for record in raa.iter_rows(named=True):
            model = raa_model.filter(pl.col("band") == record["band"]).row(0, named=True)
            rows.append(
                {
                    "year": year,
                    "geometry": "RAA",
                    "band": record["band"],
                    "band_name": record["band_name"],
                    "week": record["week"],
                    "angle_comparison": (
                        f"{record['raa_class']} vs {record['reference_raa_class']} RAA "
                        f"within {record['vza_class']} VZA"
                    ),
                    "median_contrast": record["median_absolute_contrast"],
                    "ci_low": record["ci_low"],
                    "ci_high": record["ci_high"],
                    "relative_contrast": record["median_relative_contrast"],
                    "matched_plots": record["matched_plots"],
                    "delta_r2": model["delta_r2_raa_vs_vza"],
                    "delta_aic": model["delta_aic_raa_vs_vza"],
                    "delta_bic": model["delta_bic_raa_vs_vza"],
                    "interpretation": "Sun-relative viewing direction changes reflectance even within matched VZA bins.",
                }
            )
    table = pl.DataFrame(rows).sort("year", "geometry", "band", "week")
    FINAL_RESULTS.mkdir(parents=True, exist_ok=True)
    table.write_csv(FINAL_RESULTS / "multiangular_evidence_summary.csv")
    return table


def build_robustness(vza_output: Path, raa_output: Path, filter_state: str) -> pl.DataFrame:
    rows = []
    for year in YEARS:
        vza_summary = read_csv(vza_results_dir(vza_output, year, filter_state) / "reflectance_by_vza_summary.csv")
        mean_median = vza_summary.with_columns(
            (pl.col("mean_reflectance") - pl.col("median_reflectance")).abs().alias("abs_mean_median_gap")
        )
        rows.append(
            {
                "year": year,
                "diagnostic": "mean_vs_median_reflectance",
                "scope": "VZA summaries",
                "n_rows": mean_median.height,
                "max_abs_difference": float(mean_median["abs_mean_median_gap"].max()),
                "median_abs_difference": float(mean_median["abs_mean_median_gap"].median()),
                "status": "pass" if float(mean_median["abs_mean_median_gap"].median()) < 0.01 else "review",
            }
        )
        features = pl.read_parquet(vza_results_dir(vza_output, year, filter_state) / f"plot_week_angle_features_{year}.parquet")
        weeks = sorted(features["week"].unique().to_list())
        common_plots = (
            features.select("plot_id", "week")
            .unique()
            .group_by("plot_id")
            .agg(pl.col("week").n_unique().alias("n_weeks"))
            .filter(pl.col("n_weeks") == len(weeks))
        )
        rows.append(
            {
                "year": year,
                "diagnostic": "plots_present_all_weeks",
                "scope": "VZA features",
                "n_rows": common_plots.height,
                "max_abs_difference": None,
                "median_abs_difference": None,
                "status": "pass" if common_plots.height >= 10 else "review",
            }
        )
        raa_support = read_csv(raa_results_dir(raa_output, year, filter_state) / f"raa_support_by_week_vza_{year}.csv")
        sparse_fraction = float((raa_support["plots"] < MIN_HEADLINE_PLOTS).sum() / raa_support.height)
        rows.append(
            {
                "year": year,
                "diagnostic": "raa_sparse_fraction",
                "scope": "RAA support cells",
                "n_rows": raa_support.height,
                "max_abs_difference": sparse_fraction,
                "median_abs_difference": None,
                "status": "review" if sparse_fraction > 0.25 else "pass",
            }
        )
        for model_name, path in [
            ("VZA", vza_results_dir(vza_output, year, filter_state) / f"vza_model_comparison_{year}.csv"),
            ("RAA", raa_results_dir(raa_output, year, filter_state) / f"raa_vza_model_comparison_{year}.csv"),
        ]:
            model = read_csv(path)
            rows.append(
                {
                    "year": year,
                    "diagnostic": f"{model_name.lower()}_model_rows",
                    "scope": "model comparison",
                    "n_rows": model.height,
                    "max_abs_difference": None,
                    "median_abs_difference": None,
                    "status": "pass" if model.height == len(BANDS) else "review",
                }
            )
    diagnostics = pl.DataFrame(rows)
    diagnostics.write_csv(FINAL_RESULTS / "robustness_diagnostics.csv")
    return diagnostics


def write_markdown_table(path: Path, title: str, table: pl.DataFrame, max_rows: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = table.head(max_rows) if max_rows else table
    lines = [f"# {title}", ""]
    if data.is_empty():
        lines.append("No rows.")
    else:
        columns = data.columns
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join(["---"] * len(columns)) + "|")
        for row in data.iter_rows(named=True):
            lines.append("| " + " | ".join(format_value(row[col]) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_reports(
    evidence: pl.DataFrame,
    vza_support: pl.DataFrame,
    raa_support: pl.DataFrame,
    robustness: pl.DataFrame,
    log_path: Path,
) -> None:
    started = time.perf_counter()
    FINAL_REPORTS.mkdir(parents=True, exist_ok=True)
    write_markdown_table(FINAL_REPORTS / "multiangular_evidence_summary.md", "Multiangular Evidence Summary", evidence)
    support_summary = pl.concat(
        [
            vza_support.select(
                pl.col("year").cast(pl.Int64),
                pl.col("week").cast(pl.Int64),
                pl.lit("VZA").alias("geometry"),
                pl.col("vza_class").alias("vza_class"),
                pl.col("vza_class").alias("angle_bin"),
                pl.col("plots").cast(pl.Int64),
                "is_sparse",
            ),
            raa_support.select(
                pl.col("year").cast(pl.Int64),
                pl.col("week").cast(pl.Int64),
                pl.lit("RAA").alias("geometry"),
                pl.col("vza_class").alias("vza_class"),
                pl.col("raa_class").alias("angle_bin"),
                pl.col("plots").cast(pl.Int64),
                "is_sparse",
            ),
        ],
        how="vertical",
    )
    support_summary.write_csv(FINAL_RESULTS / "angular_support_summary.csv")
    write_markdown_table(FINAL_REPORTS / "angular_support_summary.md", "Angular Support Summary", support_summary, max_rows=80)
    write_markdown_table(FINAL_REPORTS / "robustness_diagnostics.md", "Robustness Diagnostics", robustness)
    story = [
        "# Result 1 Final Story: Multiangular Reflectance",
        "",
        "## Main Claim",
        "",
        "Multiangular UAV observations produce structured, repeatable changes in sugar beet canopy reflectance. The strongest effects occur in NIR and red edge, vary through the season, persist after ground/background filtering, and are also influenced by sun-relative viewing geometry.",
        "",
        "This result demonstrates a reflectance-geometry effect. It does not yet demonstrate disease-prediction improvement.",
        "",
        "## Evidence Chain",
        "",
        "1. VZA contrasts show that off-reference viewing angles change reflectance relative to the same plot/week reference view.",
        "2. VZA model ladders show that VZA adds explanatory value beyond week, cultivar, and treatment controls.",
        "3. VZA x week improvements show that the angular response changes with canopy development.",
        "4. RAA contrasts show that sun-relative geometry changes reflectance within matched VZA bins.",
        "5. Ground-filtered outputs use `OSAVI > 0.2`, so the main story is not only bare-soil/background exposure.",
        "",
        "## Paper Figures",
        "",
        f"- Angular support heatmap: `{FINAL_FIGURES / 'angular_support_heatmap.pdf'}`",
        f"- Observation geometry distribution: `{FINAL_FIGURES / 'observation_geometry_distribution.pdf'}`",
        f"- VZA 2024/2025 comparison: `{FINAL_FIGURES / 'vza_contrast_2024_2025_comparison.pdf'}`",
        f"- Workflow schematic: `{FINAL_FIGURES / 'multiangular_workflow_schematic.pdf'}`",
        "",
        "## Paper Tables",
        "",
        f"- Evidence table: `{FINAL_RESULTS / 'multiangular_evidence_summary.csv'}`",
        f"- Angular support table: `{FINAL_RESULTS / 'angular_support_summary.csv'}`",
        f"- Robustness diagnostics: `{FINAL_RESULTS / 'robustness_diagnostics.csv'}`",
        "",
        "## Methods Text",
        "",
        "Reflectance was analyzed at the plot-week-angle level. VZA matched contrasts compare each off-reference VZA bin with the `10-15` degree reference view from the same plot, week, and band. RAA matched contrasts compare relative-azimuth classes within the same plot, week, band, and VZA bin. Headline rows require at least 10 matched plots. The main analysis uses ground-filtered data with `OSAVI > 0.2`.",
        "",
        "## Results Text",
        "",
        "The VZA analysis showed large matched NIR contrasts in both seasons, reaching approximately 0.07 reflectance units in the strongest supported cells. Red edge also showed consistent angular effects. VZA model ladders indicated that adding VZA improved fit beyond week, cultivar, and treatment controls, and VZA-by-week terms further improved fit, indicating seasonal changes in the angular response. RAA analyses showed additional directional reflectance variation within VZA bins, with strongest effects again in NIR and red edge.",
        "",
        "## Limitations",
        "",
        "RAA support is less balanced than VZA support, and categorical RAA models are parameter-heavy. Coefficient tables should be treated as diagnostics; matched contrasts are the primary interpretable effect sizes. Result 1 should not be used to claim disease-prediction improvement.",
        "",
        "## Reproducibility",
        "",
        f"- Log: `{log_path}`",
        f"- Random seed: `{SEED}`",
        "- Source outputs: current ground-filtered VZA and RAA result folders.",
        "",
    ]
    (FINAL_REPORTS / "result_01_final_story.md").write_text("\n".join(story), encoding="utf-8")
    log_phase("write_reports", started)


def validate_outputs(evidence: pl.DataFrame) -> None:
    if evidence.filter(pl.col("matched_plots") < MIN_HEADLINE_PLOTS).height:
        raise RuntimeError("Evidence table contains headline rows below matched plot threshold.")
    required = [
        FINAL_RESULTS / "multiangular_evidence_summary.csv",
        FINAL_RESULTS / "angular_support_summary.csv",
        FINAL_RESULTS / "robustness_diagnostics.csv",
        FINAL_REPORTS / "result_01_final_story.md",
        FINAL_FIGURES / "angular_support_heatmap.pdf",
        FINAL_FIGURES / "vza_contrast_2024_2025_comparison.pdf",
        FINAL_FIGURES / "observation_geometry_distribution.pdf",
        FINAL_FIGURES / "multiangular_workflow_schematic.pdf",
    ]
    missing = [path for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing or empty final reporting outputs: {missing}")


def main() -> None:
    args = parse_args()
    log_path = configure_logging()
    total_started = time.perf_counter()
    logging.info("Starting final Result 1 reporting for filter_state=%s", args.filter_state)
    FINAL_RESULTS.mkdir(parents=True, exist_ok=True)
    FINAL_REPORTS.mkdir(parents=True, exist_ok=True)
    FINAL_FIGURES.mkdir(parents=True, exist_ok=True)
    vza_support = build_vza_support(args.vza_output, args.filter_state)
    raa_support = build_raa_support(args.raa_output, args.filter_state)
    vza_support.write_csv(FINAL_RESULTS / "vza_angular_support_summary.csv")
    raa_support.write_csv(FINAL_RESULTS / "raa_angular_support_summary.csv")
    plot_angular_support(vza_support, raa_support)
    plot_vza_year_comparison(args.vza_output, args.filter_state)
    plot_workflow_schematic()
    evidence = build_evidence_table(args.vza_output, args.raa_output, args.filter_state)
    robustness = build_robustness(args.vza_output, args.raa_output, args.filter_state)
    write_reports(evidence, vza_support, raa_support, robustness, log_path)
    validate_outputs(evidence)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
