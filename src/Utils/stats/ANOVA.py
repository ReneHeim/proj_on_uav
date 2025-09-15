import itertools
import time

import numpy as np
import polars as pl
from scipy.stats import f as f_dist
from scipy.stats import studentized_range
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def ANOVA(dataframe, band_col):

    ## ANOVA MAIN
    dfc = dataframe.select(["vza_bin", band_col]).drop_nulls().filter(pl.col(band_col).is_finite())

    # Set up multiple comparison analysis using Tukey's HSD test
    # MultiComparison requires the dependent variable (band1) and grouping variable (vza_bin)
    tk = pairwise_tukeyhsd(dfc[band_col].to_numpy(), dfc["vza_bin"].to_numpy())
    rows = tk.summary().data[
        1:
    ]  # trusted order: group1, group2, meandiff, p-adj, lower, upper, reject

    # Generate all possible pairwise combinations of groups for comparison
    # This creates tuples like (group1, group2), (group1, group3), etc.
    # per-bin vectors for effect sizes
    grp = {g: dfc.filter(pl.col("vza_bin") == g)[band_col].to_numpy() for g in tk.groupsunique}

    # Define function to calculate Cohen's d (effect size measure)
    # Cohen's d = (mean1 - mean2) / pooled_standard_deviation
    def cd(a, b):
        n1, n2 = len(a), len(b)  # Sample sizes for each group
        s1 = np.var(a, ddof=1)  # Variance of group 1 (sample variance)
        s2 = np.var(b, ddof=1)  # Variance of group 2 (sample variance)

        # Calculate pooled standard deviation
        sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

        # Return Cohen's d: difference in means divided by pooled standard deviation
        return (np.mean(a) - np.mean(b)) / sp

    # Calculate Cohen's d for all pairwise comparisons
    g1 = [r[0] for r in rows]
    g2 = [r[1] for r in rows]
    meandiff = np.array([r[2] for r in rows], float)
    p = np.array([r[3] for r in rows], float)
    lower = np.array([r[4] for r in rows], float)
    upper = np.array([r[5] for r in rows], float)
    reject = np.array([r[6] for r in rows], bool)
    d = [abs(cd(grp[a], grp[b])) for a, b in zip(g1, g2)]
    tiny = np.nextafter(0, 1)
    nlog10 = -np.log10(np.clip(p, tiny, 1.0))
    ptxt = [("p < 0.0001" if (x == 0 or x < 1e-300) else f"{x:.3e}") for x in p]

    # F statistic
    groups = [grp[g] for g in tk.groupsunique]
    ns = [len(a) for a in groups]
    means = [a.mean() for a in groups]
    N, k = sum(ns), len(groups)
    grand = sum(n * m for n, m in zip(ns, means)) / N
    SSB = sum(n * (m - grand) ** 2 for n, m in zip(ns, means))
    SSW = sum(((a - m) ** 2).sum() for a, m in zip(groups, means))
    dfn, dfd = k - 1, N - k
    MSB, MSW = SSB / dfn, SSW / dfd
    F = MSB / MSW
    p_global = f_dist.sf(F, dfn, dfd)
    neglog10p = -np.log10(np.clip(p_global, np.nextafter(0, 1), 1.0))
    eta_sq = SSB / (SSB + SSW)

    # Create comprehensive results table with all statistical measures
    tbl = pl.DataFrame(
        {
            "group1": g1,  # First group in each comparison
            "group2": g2,  # Second group in each comparison
            "mean group 2 - group1": meandiff,  # Mean difference between group2 - group1
            "CI+-": upper - lower,  # Lower confidence interval bound
            "p_text": ptxt,  # Formatted p-values for display
            "-log10_p": nlog10,  # Negative log10 p-values (significance strength)
            "cohens_d": d,  # Effect size (Cohen's d)
            "reject": reject,  # Boolean: reject null hypothesis?
        }
    ).with_columns(
        pl.col("-log10_p").round(3),  # Round -log10 p-values to 3 decimals
        pl.col("cohens_d").round(3),  # per-pair standardized effect size |mean1-mean2|/spooled
        pl.lit(F).alias("F_global"),  # one-way ANOVA F-statistic across vza_bin for THIS band
        pl.lit(dfn / dfd)
        .round(10)
        .alias(
            "degrees of freedom"
        ),  # degrees of freedom: numerator df = k - 1 (k = number of bins)  /
        # denominator df = N - k (N = total observations)
        pl.lit(eta_sq)
        .round(3)
        .alias("eta_sq"),  # effect size: SSB / (SSB + SSW), variance explained by vza_bin
    )

    tbl.sort("cohens_d", descending=True)

    return tbl


def ANOVA_optimized(dataframe, band_col, col=None):
    """
    Optimized ANOVA implementation with custom Tukey's HSD for large datasets.
    Uses Polars and NumPy for efficient computation of statistics.

    Args:
        dataframe: A Polars DataFrame
        band_col: Column name containing the dependent variable
    """

    # Start with lazy operations to optimize memory usage
    # Select only necessary columns, remove nulls and non-finite values
    lazy_df = (
        dataframe.lazy().select([col, band_col]).drop_nulls().filter(pl.col(band_col).is_finite())
    )

    # Calculate all group statistics in a single efficient pass
    # This avoids multiple passes through the data for each statistic
    group_stats = (
        lazy_df.group_by(col)
        .agg(
            [
                pl.col(band_col).count().alias("n"),  # Sample size per group
                pl.col(band_col).mean().alias("mean"),  # Group means
                pl.col(band_col).var().alias("var"),  # Group variances
                pl.col(band_col).sum().alias("sum"),  # Group sums (for calculations)
            ]
        )
        .collect()
    )  # Execute the lazy computation

    # Extract statistics for calculations
    groups = group_stats[col].to_numpy()
    ns = group_stats["n"].to_numpy()
    means = group_stats["mean"].to_numpy()
    variances = group_stats["var"].to_numpy()

    # Calculate ANOVA statistics
    N = sum(ns)  # Total sample size
    k = len(groups)  # Number of groups
    grand_mean = sum(n * m for n, m in zip(ns, means)) / N  # Overall mean

    # Between-group sum of squares (variation explained by group differences)
    SSB = sum(n * (m - grand_mean) ** 2 for n, m in zip(ns, means))

    # Within-group sum of squares (variation not explained by groups)
    SSW = sum((n - 1) * v for n, v in zip(ns, variances))

    # Calculate F-statistic and p-value
    dfn, dfd = k - 1, N - k  # Degrees of freedom (between and within groups)
    MSB, MSW = SSB / dfn, SSW / dfd  # Mean squares (between and within)
    F = MSB / MSW  # F statistic (ratio of between to within variation)
    p_global = f_dist.sf(F, dfn, dfd)  # Global p-value for ANOVA
    eta_sq = SSB / (SSB + SSW)  # Effect size (proportion of variance explained)

    # Collect full dataset for pairwise comparisons
    dfc = lazy_df.collect()

    # Generate all unique pairs of groups for comparison
    # This creates tuples like (group1, group2), (group1, group3), etc.
    pairs = list(itertools.combinations(groups, 2))

    # Pre-compute statistics for each group once to avoid repeated calculations
    # This is more efficient than extracting the data repeatedly
    group_data = {}
    for g in groups:
        group_data[g] = {
            "mean": dfc.filter(pl.col(col) == g)[band_col].mean(),
            "n": dfc.filter(pl.col(col) == g).height,
            "var": dfc.filter(pl.col(col) == g)[band_col].var(),
        }

    # Custom implementation of Tukey's HSD test
    # This avoids the slower implementation in statsmodels
    results = []
    for g1, g2 in pairs:
        # Extract pre-computed statistics for the pair
        mean1 = group_data[g1]["mean"]
        mean2 = group_data[g2]["mean"]
        n1 = group_data[g1]["n"]
        n2 = group_data[g2]["n"]
        var1 = group_data[g1]["var"]
        var2 = group_data[g2]["var"]

        # Calculate mean difference between groups
        meandiff = mean2 - mean1

        # Calculate pooled standard error for Tukey's HSD
        se = np.sqrt(MSW * (1 / n1 + 1 / n2))

        # Calculate q statistic for Tukey's HSD
        q = abs(meandiff) / se

        # Calculate critical q value for confidence intervals
        q_crit = studentized_range.ppf(0.99, k, dfd)

        # Calculate confidence interval bounds
        lower = meandiff - q_crit * se
        upper = meandiff + q_crit * se

        # Calculate adjusted p-value for the comparison
        p = studentized_range.sf(q, k, dfd)

        # Calculate Cohen's d effect size (standardized mean difference)
        # Pooled standard deviation using weighted average of group variances
        sp = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        cohens_d = abs(mean1 - mean2) / sp

        # Determine if null hypothesis is rejected (alpha = 0.005)
        reject = p < 0.005

        # Store results for this pair
        results.append(
            {
                "group1": g1,
                "group2": g2,
                "meandiff": meandiff,
                "lower": lower,
                "upper": upper,
                "p_adj": p,
                "cohens_d": cohens_d,
                "reject": reject,
            }
        )

    # Format results for output DataFrame
    g1 = [r["group1"] for r in results]
    g2 = [r["group2"] for r in results]
    meandiff = [r["meandiff"] for r in results]
    p = [r["p_adj"] for r in results]
    lower = [r["lower"] for r in results]
    upper = [r["upper"] for r in results]
    cohens_d = [r["cohens_d"] for r in results]
    reject = [r["reject"] for r in results]

    # Process significance levels for display
    tiny = np.nextafter(0, 1)  # Smallest positive float for handling very small p-values
    nlog10 = -np.log10(
        np.clip(p, tiny, 1.0)
    )  # Negative log10 of p-values (higher = more significant)
    ptxt = [("p < 0.0001" if (x == 0 or x < 1e-4) else f"{x:.3e}") for x in p]  # Formatted p-values

    # Calculate confidence interval range
    CI_range = [u - l for u, l in zip(upper, lower)]

    # Create comprehensive results table with all statistical measures
    tbl = pl.DataFrame(
        {
            "group1": g1,  # First group in each comparison
            "group2": g2,  # Second group in each comparison
            "mean group 2 - group1": meandiff,  # Mean difference between group2 - group1
            "CI+-": CI_range,  # Confidence interval range
            "p_text": ptxt,  # Formatted p-values for display
            "-log10_p": nlog10,  # Negative log10 p-values (significance strength)
            "cohens_d": cohens_d,  # Effect size (Cohen's d)
            "reject": reject,  # Boolean: reject null hypothesis?
        }
    ).with_columns(
        [
            pl.col("-log10_p").round(3),  # Round -log10 p-values to 3 decimals
            pl.col("cohens_d").round(3),  # Round Cohen's d to 3 decimals
            pl.lit(F).alias("F_global"),  # One-way ANOVA F-statistic across vza_bin for this band
            pl.lit(dfn / dfd).round(10).alias("degrees of freedom"),  # Degrees of freedom ratio
            pl.lit(eta_sq).round(3).alias("eta_sq"),  # Effect size: variance explained by vza_bin
        ]
    )

    # Sort by effect size (largest effects first)
    result = tbl.sort("cohens_d", descending=True)

    return result
