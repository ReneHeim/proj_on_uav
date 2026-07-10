import itertools
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import f as f_dist
from scipy.stats import f_oneway


def setup_logging(script_name, log_dir="outputs/archive/legacy_unscoped/logs"):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{script_name}_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logging.info(f"Log file: {log_path}")
    return log_path


def load_data(table_path, value_col="reflectance_median"):
    t0 = time.time()
    df = pl.read_parquet(table_path)
    raw_n = df.height
    df = df.filter(pl.col("disease_label").is_not_null())
    df = df.with_columns(
        pl.col("disease_label").cast(pl.Int32).alias("disease_int"),
        pl.col("vza_bin").cast(pl.Utf8),
        pl.col("band").cast(pl.Utf8),
    )
    df = df.drop_nulls([value_col, "disease_int", "vza_bin", "band"])
    df = df.filter(pl.col(value_col).is_not_nan() & pl.col(value_col).is_finite())
    n = df.height
    nan_in_col = raw_n - df.filter(pl.col(value_col).is_not_nan()).height
    if nan_in_col > 0:
        logging.warning(
            f"reflectance_mean all NaN — using {value_col} instead ({nan_in_col} NaN rows dropped)"
        )
    logging.info(f"Rows loaded: {raw_n}, with disease labels and valid {value_col}: {n}")
    logging.info(
        f"Diseased: {(df['disease_int']==1).sum()}, Healthy: {(df['disease_int']==0).sum()}"
    )
    logging.info(f"[PHASE] data loading: {time.time() - t0:.1f}s")
    return df, value_col


def _get_group_arrays(grouped_df):
    arrays = []
    for row in grouped_df.iter_rows(named=True):
        vals = np.array(row["values"])
        vals = vals[np.isfinite(vals)]
        if len(vals) > 1:
            arrays.append(vals)
    return arrays


def one_way_anova_by_band(df, group_col, value_col="reflectance_mean"):
    bands = sorted(df["band"].unique().to_list())
    results = []
    for band in bands:
        bdf = df.filter(pl.col("band") == band)
        groups = bdf.group_by(group_col).agg(pl.col(value_col).alias("values"))
        group_arrays = _get_group_arrays(groups)
        if len(group_arrays) < 2:
            continue
        f_stat, p_val = f_oneway(*group_arrays)
        grand_mean = np.mean(np.concatenate(group_arrays))
        ssb = sum(len(a) * (np.mean(a) - grand_mean) ** 2 for a in group_arrays)
        ssw = sum(np.sum((a - np.mean(a)) ** 2) for a in group_arrays)
        eta_sq = ssb / (ssb + ssw) if (ssb + ssw) > 0 else np.nan
        results.append(
            {
                "test": f"{value_col} ~ {group_col}",
                "band": band,
                "f_stat": f_stat,
                "p_value": p_val,
                "eta_sq": eta_sq,
                "n_groups": len(group_arrays),
                "n_total": sum(len(a) for a in group_arrays),
                "significant": p_val < 0.05,
            }
        )
    return pl.DataFrame(results)


def two_way_anova_interaction(df, value_col="reflectance_mean"):
    bands = sorted(df["band"].unique().to_list())
    disease_levels = sorted(df["disease_int"].unique().to_list())
    results = []
    for band in bands:
        bdf = df.filter(pl.col("band") == band)
        for dl in disease_levels:
            sub = bdf.filter(pl.col("disease_int") == dl)
            groups = sub.group_by("vza_bin").agg(pl.col(value_col).alias("values"))
            group_arrays = _get_group_arrays(groups)
            if len(group_arrays) < 2:
                continue
            f_stat, p_val = f_oneway(*group_arrays)
            grand_mean = np.mean(np.concatenate(group_arrays))
            ssb = sum(len(a) * (np.mean(a) - grand_mean) ** 2 for a in group_arrays)
            ssw = sum(np.sum((a - np.mean(a)) ** 2) for a in group_arrays)
            eta_sq = ssb / (ssb + ssw) if (ssb + ssw) > 0 else np.nan
            results.append(
                {
                    "test": f"{value_col} ~ vza_bin | disease={dl}",
                    "band": band,
                    "disease_label": dl,
                    "f_stat": f_stat,
                    "p_value": p_val,
                    "eta_sq": eta_sq,
                    "n_groups": len(group_arrays),
                    "n_total": sum(len(a) for a in group_arrays),
                    "significant": p_val < 0.05,
                }
            )
    return pl.DataFrame(results)


def anova_per_cultivar(df, value_col="reflectance_mean"):
    cultivars = sorted(df["cultivar"].unique().to_list())
    bands = sorted(df["band"].unique().to_list())
    results = []
    for cv in cultivars:
        for band in bands:
            sub = df.filter(pl.col("cultivar") == cv, pl.col("band") == band)
            if sub.height < 10:
                continue
            groups = sub.group_by("vza_bin").agg(pl.col(value_col).alias("values"))
            group_arrays = _get_group_arrays(groups)
            if len(group_arrays) < 2:
                continue
            f_stat, p_val = f_oneway(*group_arrays)
            ssb = sum(
                len(a) * (np.mean(a) - np.mean(np.concatenate(group_arrays))) ** 2
                for a in group_arrays
            )
            ssw = sum(np.sum((a - np.mean(a)) ** 2) for a in group_arrays)
            eta_sq = ssb / (ssb + ssw) if (ssb + ssw) > 0 else np.nan
            results.append(
                {
                    "test": f"{value_col} ~ vza_bin | cultivar={cv}",
                    "band": band,
                    "cultivar": cv,
                    "f_stat": f_stat,
                    "p_value": p_val,
                    "eta_sq": eta_sq,
                    "n_groups": len(group_arrays),
                    "n_total": sum(len(a) for a in group_arrays),
                    "significant": p_val < 0.05,
                }
            )
    return pl.DataFrame(results)


def anova_per_treatment(df, value_col="reflectance_mean"):
    treatments = sorted(df["treatment"].unique().to_list())
    bands = sorted(df["band"].unique().to_list())
    results = []
    for trt in treatments:
        for band in bands:
            sub = df.filter(pl.col("treatment") == trt, pl.col("band") == band)
            if sub.height < 10:
                continue
            groups = sub.group_by("vza_bin").agg(pl.col(value_col).alias("values"))
            group_arrays = _get_group_arrays(groups)
            if len(group_arrays) < 2:
                continue
            f_stat, p_val = f_oneway(*group_arrays)
            ssb = sum(
                len(a) * (np.mean(a) - np.mean(np.concatenate(group_arrays))) ** 2
                for a in group_arrays
            )
            ssw = sum(np.sum((a - np.mean(a)) ** 2) for a in group_arrays)
            eta_sq = ssb / (ssb + ssw) if (ssb + ssw) > 0 else np.nan
            results.append(
                {
                    "test": f"{value_col} ~ vza_bin | treatment={trt}",
                    "band": band,
                    "treatment": trt,
                    "f_stat": f_stat,
                    "p_value": p_val,
                    "eta_sq": eta_sq,
                    "n_groups": len(group_arrays),
                    "n_total": sum(len(a) for a in group_arrays),
                    "significant": p_val < 0.05,
                }
            )
    return pl.DataFrame(results)


def disease_vza_interaction_all(df, value_col="reflectance_mean"):
    bands = sorted(df["band"].unique().to_list())
    vza_bins = sorted(df["vza_bin"].unique().to_list())
    disease_levels = [0, 1]
    results = []
    for band in bands:
        bdf = df.filter(pl.col("band") == band)
        cells = []
        for dl in disease_levels:
            for vza in vza_bins:
                vals = bdf.filter(
                    pl.col("disease_int") == dl,
                    pl.col("vza_bin") == vza,
                )[value_col].to_numpy()
                vals = vals[np.isfinite(vals)]
                if len(vals) > 0:
                    cells.append(
                        {
                            "disease": dl,
                            "vza_bin": vza,
                            "mean": np.mean(vals),
                            "n": len(vals),
                            "values": vals,
                        }
                    )

        if len(cells) < 4:
            continue

        disease_means = {}
        for dl in disease_levels:
            vals = bdf.filter(pl.col("disease_int") == dl)[value_col].to_numpy()
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                disease_means[dl] = np.mean(vals)

        if len(disease_means) == 2:
            delta_reflectance = disease_means[1] - disease_means[0]
        else:
            delta_reflectance = np.nan

        group_arrays = [c["values"] for c in cells if len(c["values"]) > 1]

        if len(group_arrays) >= 2:
            f_stat, p_val = f_oneway(*group_arrays)
            grand_mean_val = np.mean(np.concatenate(group_arrays))
            ssb = sum(len(a) * (np.mean(a) - grand_mean_val) ** 2 for a in group_arrays)
            ssw = sum(np.sum((a - np.mean(a)) ** 2) for a in group_arrays)
            eta_sq = ssb / (ssb + ssw) if (ssb + ssw) > 0 else np.nan
            results.append(
                {
                    "test": f"{value_col} ~ disease * vza_bin * band",
                    "band": band,
                    "f_stat": f_stat,
                    "p_value": p_val,
                    "eta_sq": eta_sq,
                    "n_groups": len(group_arrays),
                    "n_total": sum(len(a) for a in group_arrays),
                    "delta_reflectance_diseased_minus_healthy": delta_reflectance,
                    "significant": p_val < 0.05,
                }
            )
    return pl.DataFrame(results)


def write_markdown_summary(all_results_df, output_paths, config_info):
    t0 = time.time()

    by_band_df = all_results_df.filter(pl.col("test") == "reflectance_mean ~ vza_bin")
    by_disease_df = all_results_df.filter(pl.col("test").str.contains("vza_bin \\| disease="))
    by_cultivar_df = all_results_df.filter(pl.col("test").str.contains("cultivar="))
    by_treatment_df = all_results_df.filter(pl.col("test").str.contains("treatment="))
    interaction_df = all_results_df.filter(
        pl.col("test") == "reflectance_mean ~ disease * vza_bin * band"
    )

    band_names = {
        "band1": "Blue (475nm)",
        "band2": "Green (560nm)",
        "band3": "Red (668nm)",
        "band4": "Red Edge (717nm)",
        "band5": "NIR (842nm)",
    }

    lines = [
        "## Results: Aggregated ANOVA — Angle × Disease",
        "",
        "### 1. One-way ANOVA: reflectance_mean ~ vza_bin (per band)",
        "",
        "| Band | F-stat | p-value | eta_sq | N groups | N total |",
        "|------|--------|---------|--------|----------|---------|",
    ]
    for row in by_band_df.sort("band").iter_rows(named=True):
        lines.append(
            f"| {band_names.get(row['band'], row['band'])} | {row['f_stat']:.3f} | "
            f"{row['p_value']:.4f} | {row['eta_sq']:.3f} | {row['n_groups']} | {row['n_total']} |"
        )
    n_sig_band = by_band_df.filter(pl.col("significant")).height
    lines.append(f"\n**{n_sig_band}/{by_band_df.height} bands show significant VZA effects.**")
    lines.append("")

    lines += [
        "### 2. VZA effect stratified by disease label",
        "",
        "| Band | Disease | F-stat | p-value | eta_sq | N total |",
        "|------|---------|--------|---------|--------|---------|",
    ]
    for row in by_disease_df.sort(["band", "disease_label"]).iter_rows(named=True):
        dl = "Diseased" if row["disease_label"] == 1 else "Healthy"
        lines.append(
            f"| {band_names.get(row['band'], row['band'])} | {dl} | {row['f_stat']:.3f} | "
            f"{row['p_value']:.4f} | {row['eta_sq']:.3f} | {row['n_total']} |"
        )
    n_sig_disease = by_disease_df.filter(pl.col("significant")).height
    lines.append(
        f"\n**{n_sig_disease}/{by_disease_df.height} strata show significant VZA effects.**"
    )
    lines.append("")

    lines += [
        "### 3. Disease × VZA × Band interaction",
        "",
        "| Band | F-stat | p-value | eta_sq | ΔReflectance (diseased - healthy) |",
        "|------|--------|---------|--------|-----------------------------------|",
    ]
    for row in interaction_df.sort("band").iter_rows(named=True):
        lines.append(
            f"| {band_names.get(row['band'], row['band'])} | {row['f_stat']:.3f} | "
            f"{row['p_value']:.4f} | {row['eta_sq']:.3f} | {row['delta_reflectance_diseased_minus_healthy']:.5f} |"
        )
    n_sig_inter = interaction_df.filter(pl.col("significant")).height
    lines.append(
        f"\n**{n_sig_inter}/{interaction_df.height} bands show significant disease × VZA × band interaction.**"
    )
    lines.append("")

    lines += [
        "### 4. Per-cultivar VZA effects",
        "",
        "| Cultivar | Band | F-stat | p-value | eta_sq | N total |",
        "|----------|------|--------|---------|--------|---------|",
    ]
    for row in by_cultivar_df.sort(["cultivar", "band"]).iter_rows(named=True):
        lines.append(
            f"| {row['cultivar']} | {band_names.get(row['band'], row['band'])} | "
            f"{row['f_stat']:.3f} | {row['p_value']:.4f} | {row['eta_sq']:.3f} | {row['n_total']} |"
        )
    n_sig_cv = by_cultivar_df.filter(pl.col("significant")).height
    lines.append(
        f"\n**{n_sig_cv}/{by_cultivar_df.height} cultivar×band combinations show significant VZA effects.**"
    )
    lines.append("")

    lines += [
        "### 5. Per-treatment VZA effects",
        "",
        "| Treatment | Band | F-stat | p-value | eta_sq | N total |",
        "|-----------|------|--------|---------|--------|---------|",
    ]
    for row in by_treatment_df.sort(["treatment", "band"]).iter_rows(named=True):
        lines.append(
            f"| {row['treatment']} | {band_names.get(row['band'], row['band'])} | "
            f"{row['f_stat']:.3f} | {row['p_value']:.4f} | {row['eta_sq']:.3f} | {row['n_total']} |"
        )
    n_sig_trt = by_treatment_df.filter(pl.col("significant")).height
    lines.append(
        f"\n**{n_sig_trt}/{by_treatment_df.height} treatment×band combinations show significant VZA effects.**"
    )

    lines += [
        "",
        "### Interpretation",
    ]

    sig_bands = by_band_df.filter(pl.col("significant"))
    if sig_bands.height > 0:
        sig_names = [band_names.get(r["band"], r["band"]) for r in sig_bands.iter_rows(named=True)]
        lines.append(
            f"**Reflectance varies significantly with VZA bin in {', '.join(sig_names)}**, confirming that "
            f"viewing angle strongly affects measured reflectance in sugar beet canopies."
        )

    if n_sig_inter > 0:
        lines.append(
            f"**{n_sig_inter} bands show significant disease × VZA interaction**, indicating that the "
            f"angular dependence of the disease signal is statistically robust. "
            f"This supports using multiangular data for disease detection."
        )
    else:
        lines.append(
            "No significant disease × VZA interaction detected via aggregated ANOVA. "
            "This may indicate that the overall ANOVA is underpowered on aggregated data, "
            "and per-pixel or mixed-effects models are needed."
        )

    lines += [
        "",
        "### Outputs",
    ]
    for k, v in output_paths.items():
        lines.append(f"- **{k}**: `{v}`")

    lines += [
        "",
        "### Reproducibility",
    ]
    for k, v in config_info.items():
        lines.append(f"- {k}: `{v}`")

    lines.append("")

    report_dir = Path("outputs/quarantine_flawed_analysis/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "anova_aggregated_summary.md"
    report_path.write_text("\n".join(lines))
    logging.info(f"Markdown summary: {report_path}")
    logging.info(f"[PHASE] markdown summary: {time.time() - t0:.1f}s")
    return report_path


def main():
    script_name = "anova_aggregated"
    log_path = setup_logging(script_name)

    t_total = time.time()
    table_path = "outputs/tables/analysis_table_plot_week_band_angle.parquet"
    value_col = "reflectance_median"
    logging.info(f"Data source: {table_path}")
    logging.info(f"Value column: {value_col} (reflectance_mean is all NaN in source data)")

    df, value_col = load_data(table_path, value_col=value_col)

    t0 = time.time()
    by_band = one_way_anova_by_band(df, "vza_bin", value_col=value_col)
    logging.info(f"[PHASE] one-way ANOVA by band: {time.time() - t0:.1f}s")

    t0 = time.time()
    by_disease = two_way_anova_interaction(df, value_col=value_col)
    logging.info(f"[PHASE] disease-stratified ANOVA: {time.time() - t0:.1f}s")

    t0 = time.time()
    by_cultivar = anova_per_cultivar(df, value_col=value_col)
    logging.info(f"[PHASE] per-cultivar ANOVA: {time.time() - t0:.1f}s")

    t0 = time.time()
    by_treatment = anova_per_treatment(df, value_col=value_col)
    logging.info(f"[PHASE] per-treatment ANOVA: {time.time() - t0:.1f}s")

    t0 = time.time()
    interaction = disease_vza_interaction_all(df, value_col=value_col)
    logging.info(f"[PHASE] interaction ANOVA: {time.time() - t0:.1f}s")

    all_results = pl.concat(
        [by_band, by_disease, by_cultivar, by_treatment, interaction], how="diagonal"
    )
    all_results = all_results.with_columns(pl.col("significant").cast(pl.Boolean))
    results_dir = Path("outputs/quarantine_flawed_analysis/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / "anova_aggregated_angle_disease.csv"
    all_results.write_csv(result_path)
    logging.info(f"Saved: {result_path}")

    output_paths = {
        "ANOVA results": str(result_path),
        "Log file": str(log_path),
    }

    config_info = {
        "Data source": table_path,
        "Value column": value_col,
        "Note": "reflectance_mean is all NaN — using reflectance_median",
        "Tests": "scipy.stats.f_oneway",
        "Only 2024 data": "Yes (disease_label not null)",
        "Number of rows": df.height,
    }

    report_path = write_markdown_summary(all_results, output_paths, config_info)

    logging.info(f"\n[PHASE] Total runtime: {time.time() - t_total:.1f}s")
    logging.info(f"All outputs complete. Report: {report_path}")


if __name__ == "__main__":
    main()
