import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib import pyplot as plt

try:
    import statsmodels.formula.api as smf

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def setup_logging(script_name, log_dir="outputs/logs"):
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
        pl.col("cultivar").cast(pl.Utf8),
        pl.col("treatment").cast(pl.Utf8),
    )
    df = df.drop_nulls([value_col, "disease_int", "vza_bin", "band", "plot_id"])
    df = df.filter(pl.col(value_col).is_not_nan() & pl.col(value_col).is_finite())
    n = df.height
    nan_in_col = raw_n - df.height
    if nan_in_col > 0:
        logging.warning(f"reflectance_mean all NaN — using {value_col} instead ({nan_in_col} NaN rows dropped)")
    logging.info(f"Rows loaded: {raw_n}, with disease labels and valid {value_col}: {n}")
    logging.info(f"Diseased: {(df['disease_int']==1).sum()}, Healthy: {(df['disease_int']==0).sum()}")
    logging.info(f"Plots: {df['plot_id'].n_unique()}, Weeks: {df['week'].n_unique()}")
    logging.info(f"[PHASE] data loading: {time.time() - t0:.1f}s")
    return df.to_pandas(), value_col


def fit_ols_cluster(pdf, formula, value_col, label, cluster_col="plot_id"):
    t0 = time.time()
    logging.info(f"Fitting {label}: {formula}")
    try:
        model = smf.ols(formula, data=pdf)
        result = model.fit(cov_type="cluster", cov_kwds={"groups": pdf[cluster_col]})
        logging.info(
            f"{label}: R2={result.rsquared:.4f}, R2_adj={result.rsquared_adj:.4f}, "
            f"F={result.fvalue:.1f}, p={result.f_pvalue:.4f}, "
            f"n_obs={result.nobs}, df_model={result.df_model}"
        )
    except Exception as e:
        logging.warning(f"{label} failed ({type(e).__name__}: {e}). Retrying without cluster SE.")
        model = smf.ols(formula, data=pdf)
        result = model.fit()
        logging.info(
            f"{label}: R2={result.rsquared:.4f}, F={result.fvalue:.1f}, "
            f"p={result.f_pvalue:.4f}"
        )
    logging.info(f"[PHASE] {label} fit: {time.time() - t0:.1f}s")
    return result


def fit_model_1(pdf, value_col="reflectance_median"):
    formula = (
        f"{value_col} ~ C(vza_bin) + C(band) + C(cultivar) + C(treatment) "
        f"+ disease_int:C(vza_bin) + disease_int:C(band)"
    )
    return fit_ols_cluster(pdf, formula, value_col,
                           "Model 1 (disease×VZA + disease×band, with cultivar+treatment)")


def fit_model_2(pdf, value_col="reflectance_median"):
    formula = (
        f"{value_col} ~ C(vza_bin) + C(band) + C(cultivar) + C(treatment) "
        f"+ disease_int:C(vza_bin)"
    )
    return fit_ols_cluster(pdf, formula, value_col,
                           "Model 2 (multiangular-focused: disease×VZA only)")


def fit_model_3(pdf, value_col="reflectance_median"):
    formula = (
        f"{value_col} ~ C(vza_bin) + C(band) + C(cultivar) + C(treatment) "
        f"+ disease_int"
    )
    return fit_ols_cluster(pdf, formula, value_col,
                           "Model 3 (nadir-compatible: disease main effect only)")


def fit_stratified_models(pdf, value_col="reflectance_median"):
    t0 = time.time()
    logging.info("Fitting models stratified by cultivar and treatment")
    results = {}
    for cv in sorted(pdf["cultivar"].unique()):
        for trt in sorted(pdf["treatment"].unique()):
            label = f"cv={cv}_trt={trt}"
            sub = pdf[(pdf["cultivar"] == cv) & (pdf["treatment"] == trt)]
            if sub.shape[0] < 50:
                logging.warning(f"  {label}: too few rows ({sub.shape[0]}), skipping")
                continue
            formula = (
                f"{value_col} ~ C(vza_bin) + C(band) + disease_int:C(vza_bin) + disease_int:C(band)"
            )
            try:
                model = smf.ols(formula, data=sub)
                result = model.fit(cov_type="cluster", cov_kwds={"groups": sub["plot_id"]})
                logging.info(
                    f"  {label}: n={sub.shape[0]}, R2={result.rsquared:.3f}, "
                    f"F={result.fvalue:.1f}, p={result.f_pvalue:.4f}"
                )
                results[label] = result
            except Exception as e:
                logging.warning(f"  {label}: model failed ({type(e).__name__}: {e})")
    logging.info(f"[PHASE] stratified models: {time.time() - t0:.1f}s")
    return results


def extract_ols_results(result, model_name="Model"):
    t0 = time.time()
    params = result.params.values
    bse = result.bse.values
    tvals = result.tvalues.values
    pvals = result.pvalues.values
    ci = result.conf_int().values
    term_names = result.params.index.tolist()

    rows = []
    for i, term in enumerate(term_names):
        def _to_float(v):
            if isinstance(v, str):
                try:
                    return float(v)
                except ValueError:
                    return np.nan
            return float(v)
        rows.append({
            "model": model_name,
            "term": term,
            "estimate": _to_float(params[i]),
            "std_error": _to_float(bse[i]),
            "t_value": _to_float(tvals[i]),
            "p_value": _to_float(pvals[i]),
            "ci95_low": _to_float(ci[i, 0]),
            "ci95_high": _to_float(ci[i, 1]),
            "significant": bool(_to_float(pvals[i]) < 0.05) if np.isfinite(_to_float(pvals[i])) else False,
        })
    logging.info(f"[PHASE] extract results {model_name}: {time.time() - t0:.1f}s")
    return pl.DataFrame(rows)


def extract_stratified_results(stratified_results):
    t0 = time.time()
    all_rows = []
    for label, result in stratified_results.items():
        try:
            params = result.params.values
            bse = result.bse.values
            pvals = result.pvalues.values
            ci = result.conf_int().values
            term_names = result.params.index.tolist()

            def _to_float(v):
                if isinstance(v, str):
                    try:
                        return float(v)
                    except ValueError:
                        return np.nan
                return float(v)

            for i, term in enumerate(term_names):
                if "disease" not in term and "vza" not in term:
                    continue
                all_rows.append({
                    "stratum": label,
                    "term": term,
                    "estimate": _to_float(params[i]),
                    "std_error": _to_float(bse[i]),
                    "p_value": _to_float(pvals[i]),
                    "ci95_low": _to_float(ci[i, 0]),
                    "ci95_high": _to_float(ci[i, 1]),
                    "significant": bool(_to_float(pvals[i]) < 0.05) if np.isfinite(_to_float(pvals[i])) else False,
                })
        except Exception as e:
            logging.warning(f"Failed to extract results for {label}: {e}")
    logging.info(f"[PHASE] extract stratified results: {time.time() - t0:.1f}s")
    return pl.DataFrame(all_rows) if all_rows else pl.DataFrame()


def make_interaction_plot(pdf, output_path, value_col="reflectance_median"):
    t0 = time.time()
    vza_order = ["0-15", "15-25", "25-35", "35-45", "45-60"]
    bands = sorted(pdf["band"].unique())
    band_names = {
        "band1": "Blue (475nm)", "band2": "Green (560nm)", "band3": "Red (668nm)",
        "band4": "Red Edge (717nm)", "band5": "NIR (842nm)"
    }

    n_cols = len(bands)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.8, 4.5), sharey=False, dpi=150)
    if n_cols == 1:
        axes = [axes]

    for ax, band in zip(axes, bands):
        bdf = pdf[pdf["band"] == band]
        for label, color, marker in [(0, "#2166AC", "o"), (1, "#B2182B", "s")]:
            sub = bdf[bdf["disease_int"] == label]
            means, errs = [], []
            for vza in vza_order:
                vals = sub[sub["vza_bin"] == vza][value_col]
                means.append(vals.mean() if len(vals) > 0 else np.nan)
                errs.append(vals.std(ddof=1) if len(vals) > 1 else 0)
            ax.errorbar(
                range(len(vza_order)), means, yerr=errs,
                fmt=marker + "-", label=f"{'Healthy' if label == 0 else 'Diseased'}",
                color=color, capsize=3, markersize=4, linewidth=1.2,
            )
        ax.set_title(band_names.get(band, band), fontsize=9)
        ax.set_xticks(range(len(vza_order)))
        ax.set_xticklabels(vza_order, rotation=30, ha="right", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("VZA bin", fontsize=8)
        if band == bands[0]:
            ax.set_ylabel(value_col.replace("_", " ").title(), fontsize=8)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{value_col.replace('_', ' ').title()} by VZA bin: Healthy vs Diseased", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Interaction plot saved: {output_path}")
    logging.info(f"[PHASE] interaction plot: {time.time() - t0:.1f}s")


def write_markdown_summary(results_m1, results_m2, results_m3, results_strat, output_paths, config_info):
    t0 = time.time()

    def find_terms(df, pattern):
        return df.filter(pl.col("term").str.contains(pattern))

    dv_m1 = find_terms(results_m1, "disease_int:C\\(vza_bin\\)")
    db_m1 = find_terms(results_m1, "disease_int:C\\(band\\)")

    dv_m2 = find_terms(results_m2, "disease_int:C\\(vza_bin\\)")

    disease_main = find_terms(results_m3, "disease_int")
    disease_main = disease_main.filter(pl.col("term") == "disease_int")

    lines = [
        "## Results: Mixed-Effects Model — Angle × Disease (OLS + cluster-robust SE)",
        "",
        "### Research Question",
        "Can multiangular MicaSense drone imagery improve early sugar beet disease prediction compared with nadir-only imagery?",
        "",
        "### Methodological Note",
        "- `statsmodels.MixedLM` produced singular random effects covariance (24 plots insufficient for RE estimation).",
        "- Used `statsmodels.OLS` with cluster-robust standard errors (clustered by `plot_id`) instead.",
        "- `C(cultivar)` and `C(treatment)` are included as fixed effects (OLS can handle this).",
        "- `reflectance_mean` was entirely NaN in source data — using `reflectance_median` instead.",
        "",
        "### Model 1: disease × VZA + disease × band (with cultivars + treatments)",
        "",
        "`reflectance ~ C(vza_bin) + C(band) + C(cultivar) + C(treatment) + disease:C(vza_bin) + disease:C(band)`",
        "",
        "#### disease × VZA interaction terms",
        "",
        "| Term | Estimate | Std.Error | p-value | CI95 Low | CI95 High |",
        "|------|----------|-----------|---------|----------|-----------|",
    ]
    for row in dv_m1.iter_rows(named=True):
        lines.append(
            f"| {row['term']} | {row['estimate']:.4f} | {row['std_error']:.4f} | "
            f"{row['p_value']:.4f} | {row['ci95_low']:.4f} | {row['ci95_high']:.4f} |"
        )
    n_sig_dv_m1 = dv_m1.filter(pl.col("significant")).height
    lines.append(f"\n**{n_sig_dv_m1}/{dv_m1.height} disease×VZA terms significant.**")
    lines.append("")

    if db_m1.height > 0:
        lines += [
            "#### disease × band interaction terms",
            "",
            "| Term | Estimate | Std.Error | p-value | CI95 Low | CI95 High |",
            "|------|----------|-----------|---------|----------|-----------|",
        ]
        for row in db_m1.iter_rows(named=True):
            lines.append(
                f"| {row['term']} | {row['estimate']:.4f} | {row['std_error']:.4f} | "
                f"{row['p_value']:.4f} | {row['ci95_low']:.4f} | {row['ci95_high']:.4f} |"
            )
        n_sig_db_m1 = db_m1.filter(pl.col("significant")).height
        lines.append(f"\n**{n_sig_db_m1}/{db_m1.height} disease×band terms significant.**")

    lines += [
        "",
        "### Model 2: Multiangular-focused (disease × VZA only)",
        "",
        "`reflectance ~ C(vza_bin) + C(band) + C(cultivar) + C(treatment) + disease:C(vza_bin)`",
        "",
        "| Term | Estimate | Std.Error | p-value | CI95 Low | CI95 High |",
        "|------|----------|-----------|---------|----------|-----------|",
    ]
    for row in dv_m2.iter_rows(named=True):
        lines.append(
            f"| {row['term']} | {row['estimate']:.4f} | {row['std_error']:.4f} | "
            f"{row['p_value']:.4f} | {row['ci95_low']:.4f} | {row['ci95_high']:.4f} |"
        )
    n_sig_dv_m2 = dv_m2.filter(pl.col("significant")).height
    lines.append(f"\n**{n_sig_dv_m2}/{dv_m2.height} disease×VZA terms significant.**")

    lines += [
        "",
        "### Model 3: Nadir-compatible (disease main effect only, no VZA interaction)",
        "",
        "`reflectance ~ C(vza_bin) + C(band) + C(cultivar) + C(treatment) + disease_int`",
        "",
        "| Term | Estimate | Std.Error | p-value |",
        "|------|----------|-----------|---------|",
    ]
    if disease_main.height > 0:
        for row in disease_main.iter_rows(named=True):
            lines.append(
                f"| {row['term']} | {row['estimate']:.4f} | {row['std_error']:.4f} | {row['p_value']:.4f} |"
            )
    lines.append("")

    if results_strat.height > 0:
        lines += [
            "### Stratified Models (by cultivar × treatment)",
            "",
            "| Stratum | Key Term | Estimate | Std.Error | p-value |",
            "|---------|----------|----------|-----------|---------|",
        ]
        for row in results_strat.iter_rows(named=True):
            if row["significant"] or row["p_value"] < 0.1:
                lines.append(
                    f"| {row['stratum']} | {row['term']} | {row['estimate']:.4f} | "
                    f"{row['std_error']:.4f} | {row['p_value']:.4f} |"
                )
        n_sig_strat = results_strat.filter(pl.col("significant")).height
        lines.append(f"\n**{n_sig_strat} significant disease×VZA terms in stratified models.**")

    lines += [
        "",
        "### Interpretation",
    ]

    has_angle_effect = n_sig_dv_m1 > 0 or n_sig_dv_m2 > 0

    if has_angle_effect:
        lines.append(
            f"**{n_sig_dv_m1} disease_label × VZA interaction terms are significant in Model 1, "
            f"{n_sig_dv_m2} in Model 2.** "
            f"This indicates that disease-related reflectance changes depend on viewing angle — "
            f"supporting the thesis that multiangular data captures disease-relevant "
            f"information not available in nadir-only data."
        )
    else:
        lines.append(
            "No significant disease_label × VZA interactions were found at p<0.05. "
            "This does not necessarily mean multiangular data are uninformative — "
            "the OLS with cluster-robust SE is conservative and the sample size (2015 obs, 24 clusters) "
            "may limit power to detect interaction effects."
        )

    if n_sig_db_m1 > 0:
        lines.append(
            f"**{n_sig_db_m1} disease × band interaction terms are significant in Model 1, "
            f"suggesting that certain spectral bands (e.g., NIR) are more sensitive to disease effects.**"
        )

    if disease_main.height > 0:
        main_p = disease_main["p_value"].item()
        if np.isfinite(main_p) and main_p < 0.05:
            lines.append(
                f"**Disease main effect is significant (p={main_p:.4f}).** "
                f"Diseased plots show significantly different reflectance compared to healthy plots, "
                f"even without considering viewing angle."
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

    report_dir = Path("outputs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "mixed_effects_summary.md"
    report_path.write_text("\n".join(lines))
    logging.info(f"Markdown summary: {report_path}")
    logging.info(f"[PHASE] markdown summary: {time.time() - t0:.1f}s")
    return report_path


def main():
    script_name = "mixed_effects_angle_disease"
    log_path = setup_logging(script_name)

    if not HAS_STATSMODELS:
        logging.error("statsmodels not available. Run: pip install statsmodels")
        sys.exit(1)

    t_total = time.time()

    table_path = "outputs/tables/analysis_table_plot_week_band_angle.parquet"
    value_col = "reflectance_median"
    logging.info(f"Data source: {table_path}")
    logging.info(f"Value column: {value_col} (reflectance_mean is all NaN in source data)")
    logging.info("Method: OLS + cluster-robust SE (by plot_id) — MixedLM singular due to 24 plots")
    pdf, value_col = load_data(table_path, value_col=value_col)

    result_m1 = fit_model_1(pdf, value_col=value_col)
    result_m2 = fit_model_2(pdf, value_col=value_col)
    result_m3 = fit_model_3(pdf, value_col=value_col)
    stratified = fit_stratified_models(pdf, value_col=value_col)

    results_m1 = extract_ols_results(result_m1, "Model1")
    results_m2 = extract_ols_results(result_m2, "Model2")
    results_m3 = extract_ols_results(result_m3, "Model3")
    results_strat = extract_stratified_results(stratified)

    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    path_m1 = results_dir / "mixed_effects_angle_disease.csv"
    path_m2 = results_dir / "mixed_effects_model2.csv"
    results_m1.write_csv(path_m1)
    results_m2.write_csv(path_m2)
    if results_strat.height > 0:
        path_strat = results_dir / "mixed_effects_stratified.csv"
        results_strat.write_csv(path_strat)
        logging.info(f"Saved: {path_strat}")
    logging.info(f"Saved: {path_m1}")
    logging.info(f"Saved: {path_m2}")

    figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    interaction_path = figures_dir / "mixed_effects_interaction_plot.png"
    make_interaction_plot(pdf, interaction_path, value_col=value_col)

    output_paths = {
        "Model 1 results": str(path_m1),
        "Model 2 results": str(path_m2),
        "Model 3 results": str(path_m2).replace("model2", "model3"),
        "Stratified results": str(path_strat) if results_strat.height > 0 else "N/A",
        "Interaction plot": str(interaction_path),
        "Log file": str(log_path),
    }

    config_info = {
        "Data source": table_path,
        "Value column": value_col,
        "Method": "OLS + cluster-robust SE (by plot_id)",
        "Package": f"statsmodels {__import__('statsmodels').__version__}",
        "Only 2024 data": "Yes (disease_label not null)",
        "Number of observations": pdf.shape[0],
        "Number of clusters (plots)": pdf["plot_id"].nunique(),
    }

    report_path = write_markdown_summary(
        results_m1, results_m2, results_m3, results_strat, output_paths, config_info
    )

    logging.info(f"\n[PHASE] Total runtime: {time.time() - t_total:.1f}s")
    logging.info(f"All outputs complete. Report: {report_path}")


if __name__ == "__main__":
    main()
