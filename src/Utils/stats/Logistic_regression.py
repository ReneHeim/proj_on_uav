import pandas as pd
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import patsy


def preprocess_healthy_diesel(df_h, df_d, sample_size=500_000, random_state=42, raa_edges=list(range(-360, 361, 90)),
                              vza_edges=[0, 20, 40, 60, 80]):
    """
    Preprocess data for logistic regression analysis.

    Args:
        df_h: Healthy samples dataframe
        df_d: Diseased samples dataframe
        sample_size: Maximum number of samples to use
        random_state: Seed for reproducibility
        raa_edges: Bin edges for relative azimuth angle
        vza_edges: Bin edges for viewing zenith angle

    Returns:
        a combined pandas DataFrame with balanced classes and necessary preprocessing.
    """
    # vectorized binning with polars (no Python UDFs)

    # Convert sun elevation to zenith angle
    df_d = df_d.with_columns((90 - pl.col("sunelev")).alias("sza"))
    df_h = df_h.with_columns((90 - pl.col("sunelev")).alias("sza"))


    # Calculate relative azimuth angle constrained to [0,180]
    df_d = df_d.with_columns( (((pl.col("saa") - pl.col("vaa") + 180) % 360) - 180).alias("RAA"))
    df_h = df_h.with_columns( (((pl.col("saa") - pl.col("vaa") + 180) % 360) - 180).alias("RAA"))

    ## === Create bin labels ===
    np.random.seed(random_state)
    raa_labels = [f"{lo}-{hi}" for lo, hi in zip(raa_edges[:-1], raa_edges[1:])]
    vza_labels = [f"{lo}-{hi}" for lo, hi in zip(vza_edges[:-1], vza_edges[1:])]

    df_h = (
        df_h
        .with_columns([
            pl.cut(pl.col("RAA"), bins=raa_edges, labels=raa_labels).alias("raa_bin"),
            pl.cut(pl.col("vza"), bins=vza_edges, labels=vza_labels).alias("vza_bin"),
        ])
        .drop_nulls(["raa_bin", "vza_bin"])  # keep only rows that fell into both sets of bins
    )
    df_d = (
        df_d
        .with_columns([
            pl.cut(pl.col("RAA"), bins=raa_edges, labels=raa_labels).alias("raa_bin"),
            pl.cut(pl.col("vza"), bins=vza_edges, labels=vza_labels).alias("vza_bin"),
        ])
        .drop_nulls(["raa_bin", "vza_bin"])  # keep only rows that fell into both sets of bins
    )

    ###=== Combine datasets =====

    common = set(df_h.columns) & set(df_d.columns)  # align schemas
    df = pl.concat([
        df_h.select(common).with_columns(pl.lit("healthy").alias("status")),
        df_d.select(common).with_columns(pl.lit("diseased").alias("status"))
    ])
    df = df.with_columns([pl.col("status").cast(pl.Categorical),
                          pl.col("vza_bin").cast(pl.Categorical),
                          pl.col("raa_bin").cast(pl.Categorical)])
    n = min(sample_size, df.height)

    df = df.sample(n, with_replacement=False, shuffle=True)
    return df.to_pandas()


def OLS(df):
    """
    Perform OLS analysis comparing main effects vs full interaction model.

    Args:
        df: Preprocessed dataframe

    Returns:
        Prints comparison statistics
    """
    m0 = smf.ols("band5 ~ C(status) + C(vza_bin) + C(raa_bin)", data=df).fit()
    m1 = smf.ols("band5 ~ C(status)*C(vza_bin)*C(raa_bin)", data=df).fit()  # add all interactions
    lrt = 2 * (m1.llf - m0.llf)
    df_diff = int(m1.df_model - m0.df_model)
    p = chi2.sf(lrt, df_diff)
    print({"LRT": lrt, "df": df_diff, "p": p, "ΔAIC": m0.aic - m1.aic, "ΔBIC": m0.bic - m1.bic})


def logistic_regression(df):
    """
    Perform comprehensive statistical analysis comparing different models:
    - Nadir-only (vza < 20°) vs main effects model
    - Main effects vs full interaction model
    - AUROC comparisons between models
    - Effect size analysis using Cohen's d

    Args:
        df: Preprocessed dataframe with 'status', 'band5', 'vza', 'vza_bin', 'raa_bin' columns

    Returns:
        A polars DataFrame with all analysis results
    """
    # Create a copy to avoid modifying the original dataframe
    d = df.copy()

    # Run all analyses
    ols_results = _compare_ols_models(d)
    auroc_results = _calculate_auroc_metrics(d)
    cohens_d_results = _calculate_cohens_d(d)

    # Combine all results into a single dictionary
    all_results = {
        "OLS_comparisons": ols_results,
        "AUROC_metrics": auroc_results,
        "Effect_size": cohens_d_results
    }

    # Convert results to polars DataFrame for return
    return pl.from_dict(all_results)


def _compare_ols_models(d):
    """
    Compare OLS models of increasing complexity using likelihood ratio tests.

    Args:
        d: Preprocessed dataframe

    Returns:
        Dictionary of model comparison metrics
    """
    # --- OLS models with different complexities ---

    # Nadir subset (vza < 20°)
    nd = d.query("vza < 20").copy()

    # Fit models with increasing complexity
    m_nadir = smf.ols("band5 ~ C(status)", data=nd).fit()
    m_main = smf.ols("band5 ~ C(status) + C(vza_bin) + C(raa_bin)", data=d).fit()
    m1 = smf.ols("band5 ~ C(status)*C(vza_bin)*C(raa_bin)", data=d).fit()

    # Helper function for model comparison using likelihood ratio test
    def cmp(a, b):
        LRT = 2 * (b.llf - a.llf)
        df = int(b.df_model - a.df_model)
        p = chi2.sf(LRT, df)
        return {"LRT": LRT, "df": df, "p": p, "ΔAIC": a.aic - b.aic, "ΔBIC": a.bic - b.bic}

    # Compare models pairwise
    results = {
        "nadir→main": cmp(m_nadir, m_main),
        "main→full": cmp(m_main, m1),
        "nadir→full": cmp(m_nadir, m1)
    }

    return results


def _calculate_auroc_metrics(d):
    """
    Calculate AUROC metrics for different models using cross-validation.

    Args:
        d: Preprocessed dataframe

    Returns:
        Dictionary of AUROC values and differences between models
    """
    # --- AUROC calculations for different models ---

    # Prepare target variable
    y = (d["status"].astype(str).str.lower() == "diseased").astype(int)

    # Nadir subset (vza < 20°)
    nd = d.query("vza < 20")
    y_nadir = (nd["status"].astype(str).str.lower() == "diseased").astype(int)

    # AUROC for nadir-only model (baseline)
    au_nadir = cross_val_score(
        LogisticRegression(max_iter=2000),
        nd[["band5"]],
        y_nadir,
        cv=5,
        scoring="roc_auc"
    ).mean()

    # AUROC for main effects model (band5 + geometry)
    X_main = patsy.dmatrix("band5 + C(vza_bin) + C(raa_bin)", d, return_type='dataframe')
    au_main = cross_val_score(
        LogisticRegression(max_iter=2000),
        X_main,
        y,
        cv=5,
        scoring="roc_auc"
    ).mean()

    # AUROC for full interaction model
    X_full = patsy.dmatrix("band5 * C(vza_bin) * C(raa_bin)", d, return_type='dataframe')
    au_full = cross_val_score(
        LogisticRegression(max_iter=2000),
        X_full,
        y,
        cv=5,
        scoring="roc_auc"
    ).mean()

    # Geometry-aware model using dummy variables
    X_geo = pd.get_dummies(d[["band5", "vza_bin", "raa_bin"]], drop_first=True)
    au_geo = cross_val_score(
        LogisticRegression(max_iter=2000),
        X_geo,
        y,
        cv=5,
        scoring="roc_auc"
    ).mean()

    # Collect results
    results = {
        "AUROC_nadir": au_nadir,
        "AUROC_main": au_main,
        "AUROC_full": au_full,
        "AUROC_angle": au_geo,
        "Δ_main−nadir": au_main - au_nadir,
        "Δ_full−main": au_full - au_main,
        "ΔAUROC_geo−nadir": au_geo - au_nadir
    }

    return results


def _calculate_cohens_d(d):
    """
    Calculate Cohen's d effect size across different viewing angle bins.

    Args:
        d: Preprocessed dataframe

    Returns:
        Dictionary with Cohen's d values
    """

    # --- Cohen's d effect size calculations ---

    # Helper function to compute Cohen's d (pooled standard deviation)
    def cohen_d(sub):
        """Calculate pooled-SD d (diseased - healthy)"""
        g = sub.groupby("status")["band5"].agg(["mean", "std", "count"])
        m1, m0 = g.loc["diseased", "mean"], g.loc["healthy", "mean"]
        s1, s0 = g.loc["diseased", "std"], g.loc["healthy", "std"]
        n1, n0 = g.loc["diseased", "count"], g.loc["healthy", "count"]
        sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n0 - 1) * s0 ** 2) / (n1 + n0 - 2))
        return (m1 - m0) / sp

    # Calculate baseline Cohen's d for nadir viewing angles
    baseline = cohen_d(d.query("vza < 20"))

    # Calculate Cohen's d for each viewing geometry bin combination
    perbin = d.groupby(["vza_bin", "raa_bin"]).apply(cohen_d).rename("d")

    # Get top 5 bins by absolute effect size
    top = perbin.reindex(perbin.abs().sort_values(ascending=False).index)[:5]

    # Collect results
    results = {
        "Cohen_d_nadir": baseline,
        "top_bins_by_|d|": top.to_dict()
    }

    return results