import pandas as pd
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import patsy

def preprocess_healthy_diseased(df_h, df_d, sample_size=500_000, random_state=42,
                                raa_edges=list(range(-360, 361, 90)),
                                vza_edges=[0, 20, 40, 60, 80]):
    np.random.seed(random_state)

    # compute SZA and RAA (RAA in [-180, 180])
    def add_angles(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            (90 - pl.col("sunelev")).alias("sza"),
            (((pl.col("saa") - pl.col("vaa") + 180) % 360) - 180).alias("RAA"),
        ])

    raa_labels = [f"{lo}-{hi}" for lo, hi in zip(raa_edges[:-1], raa_edges[1:])]
    vza_labels = [f"{lo}-{hi}" for lo, hi in zip(vza_edges[:-1], vza_edges[1:])]

    # expression-level `cut` (labels must be len(breaks)-1)
    def bin_two(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            # use internal cut points; labels remain pairwise edge names
            pl.col("RAA").cut(breaks=raa_edges[1:-1], labels=raa_labels).alias("raa_bin"),
            pl.col("vza").cut(breaks=vza_edges[1:-1], labels=vza_labels).alias("vza_bin"),
        ]).drop_nulls(["raa_bin", "vza_bin"])

    df_h2 = bin_two(add_angles(df_h))
    df_d2 = bin_two(add_angles(df_d))

    common = set(df_h2.columns) & set(df_d2.columns)
    df = pl.concat([
        df_h2.select(common).with_columns(pl.lit("healthy").alias("status")),
        df_d2.select(common).with_columns(pl.lit("diseased").alias("status")),
    ]).with_columns(pl.col("status").cast(pl.Categorical))




    return df.sample(min(sample_size, df.height), shuffle=True).to_pandas()



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
    df: Preprocessed dataframe containing both healthy and diseased samples, with columns: 'status', 'band5', 'vza', 'vza_bin', and 'raa_bin'

    Returns:
        A polars DataFrame with all analysis results
    """
    # Create a copy to avoid modifying the original dataframe
    d = df.copy()

    # Run all analyses
    ols_results = _compare_ols_fast(d)
    auroc_results = _auroc_fast(d)
    cohens_d_results = _calculate_cohens_d(d)

    # Combine all results into a single dictionary
    all_results = {
        "OLS_comparisons": ols_results,
        "AUROC_metrics": auroc_results,
        "Effect_size": cohens_d_results
    }

    # Convert results to polars DataFrame for return
    return pl.from_dict(all_results)


def _compare_ols_fast(d):
    """
    Compare OLS-like models (Gaussian) using sparse encodings and RSS-based LR/AIC/BIC.
    Matches the science of:
      m_nadir: band5 ~ C(status)           on vza<20
      m_main : band5 ~ C(status)+C(vza_bin)+C(raa_bin)
      m_full : band5 ~ C(status)*C(vza_bin)*C(raa_bin)
    Returns dict with LRT, df, p, ΔAIC, ΔBIC for each pair.
    """
    time_start = time.time()
    import numpy as np
    from scipy.sparse import csr_matrix, hstack
    from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
    from sklearn.linear_model import Ridge
    from scipy.stats import chi2

    y = d["band5"].to_numpy(dtype=np.float32)
    n  = y.shape[0]

    # --- helper: fit linear model on CSR X and compute RSS & k (≈ OLS via tiny ridge) ---
    def rss_and_k(X_csr, y_):
        # tiny alpha approximates OLS but enables sparse solvers
        model = Ridge(alpha=1e-8, fit_intercept=True, solver="auto", max_iter=None)
        model.fit(X_csr, y_)
        resid = y_ - model.predict(X_csr)
        rss = float(np.dot(resid, resid))
        # k: #coeffs incl. intercept = nnz columns + 1 (intercept)
        k = int(X_csr.shape[1] + 1)
        return rss, k

    # --- Nadir model: band5 ~ C(status) on vza<20 ---
    nd_mask = d["vza"] < 20
    y_nd = y[nd_mask]
    enc_nd = OneHotEncoder(drop="first", sparse_output=True, dtype=np.float32, handle_unknown="ignore")
    Xn = enc_nd.fit_transform(d.loc[nd_mask, ["status"]])  # CSR
    rss_nadir, k_nadir = rss_and_k(Xn, y_nd)
    n_nd = y_nd.shape[0]

    # --- Main and Full on all rows ---
    # One encoder over all three factors; drop='first' to avoid collinearities.
    enc = OneHotEncoder(
        drop="first", sparse_output=True, dtype=np.float32, handle_unknown="ignore"
    ).fit(d[["status", "vza_bin", "raa_bin"]])
    Xcat = enc.transform(d[["status", "vza_bin", "raa_bin"]])  # CSR

    # Main effects == the one-hot block itself
    X_main = Xcat

    # Full interactions: 2-way + 3-way of categorical indicators
    poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)
    X_full = poly.fit_transform(Xcat)  # stays CSR

    rss_main, k_main = rss_and_k(X_main, y)
    rss_full, k_full = rss_and_k(X_full, y)

    # --- Metrics via Gaussian identities (no dense likelihood needed) ---
    def lr_cmp(rss_a, n_a, k_a, rss_b, n_b, k_b):
        # assume same data for a vs b unless 'nadir' (then use n_a,n_b separately)
        # LRT with Gaussian ML: n * log(RSS_a/RSS_b)
        if n_a != n_b:
            # different samples: use the appropriate n for each log-lik term
            # log L = -n/2 [log(2π) + 1 + log(RSS/n)]
            ll_a = -(n_a/2) * (np.log(2*np.pi) + 1 + np.log(rss_a / n_a))
            ll_b = -(n_b/2) * (np.log(2*np.pi) + 1 + np.log(rss_b / n_b))
            LRT = 2 * (ll_b - ll_a)
        else:
            n = n_a
            LRT = n * np.log(rss_a / rss_b)

        df  = int(k_b - k_a)
        p   = float(chi2.sf(LRT, df))

        # AIC/BIC deltas (same-sample preferred; when samples differ, report w.r.t. own n)
        def aic_bic(rss, n, k):
            ll = -(n/2) * (np.log(2*np.pi) + 1 + np.log(rss / n))
            return 2*k - 2*ll, k*np.log(n) - 2*ll

        AIC_a, BIC_a = aic_bic(rss_a, n_a, k_a)
        AIC_b, BIC_b = aic_bic(rss_b, n_b, k_b)
        return {"LRT": float(LRT), "df": df, "p": p, "ΔAIC": AIC_a - AIC_b, "ΔBIC": BIC_a - BIC_b}

    logging.info(f"Model fits done in {time.time() - time_start:.2f}s")

    results = {
        "nadir→main": lr_cmp(rss_nadir, n_nd, k_nadir, rss_main, n, k_main),
        "main→full":  lr_cmp(rss_main,  n,    k_main,  rss_full, n, k_full),
        "nadir→full": lr_cmp(rss_nadir, n_nd, k_nadir, rss_full, n, k_full),
    }
    return results

import logging
import time

def _auroc_fast(d, same_size=False, random_state=0):
    """
    Compute AUROC for four models under strict anti-leakage protocols:

    Models
    ------
    1) Nadir baseline: Logistic regression on band5, restricted to vza < 20°.
    2) Main effects : band5 + one-hot(vza_bin, raa_bin).
    3) Full interactions: main effects + pairwise + triple interactions (degree=3,
       interaction_only=True, include_bias=False).
    4) Angle-aware (geo): identical to main effects; kept for naming continuity.

    Key safeguards
    --------------
    - All preprocessing inside Pipelines (no leakage).
    - Stratified CV with fixed seed for reproducibility.
    - Optional `same_size=True` forces non-nadir sample N == nadir N.

    Returns
    -------
    dict with AUROCs and differences (Δ) between models.
    """
    import logging, time
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    # --- Target variable: binary 0/1 as int8 to reduce memory footprint ---
    y = (d["status"].astype(str).str.lower() == "diseased").astype("int8").to_numpy()

    # --- Nadir mask (baseline science preserved exactly) ---
    nd_mask = (d["vza"] < 20).to_numpy()

    # --- Single design table; column selectors will pick subsets later ---
    # Keep only columns actually used by the models.
    X = d[["band5", "vza_bin", "raa_bin"]]

    # Sanity checks (optional but helpful in a paper/repro context)
    if X.isnull().any().any():
        logging.warning("NaNs detected in X; consider imputing before modeling.")
    if y.shape[0] != X.shape[0]:
        raise ValueError("X and y length mismatch, check preprocessing.")

    # --- Preprocessing primitives (fit only on train folds via Pipeline) ---
    scale_band = ("scale", StandardScaler(with_mean=False), ["band5"])
    onehot_geo = ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"), ["vza_bin", "raa_bin"])

    # Main-effects ColumnTransformer: [scaled band5 | one-hot(vza_bin, raa_bin)]
    ct_main = ColumnTransformer([scale_band, onehot_geo])

    # Classifier with explicit regularization and tolerance (stable across folds)
    lr = LogisticRegression(penalty="l2", max_iter=300, tol=1e-3, random_state=0)

    # Pipelines:
    # 1) Nadir baseline: band5 only, evaluated on nadir rows.
    pipe_nadir = Pipeline([("prep", ColumnTransformer([scale_band])), ("clf", lr)])

    # 2) Main effects
    pipe_main = Pipeline([("prep", ct_main), ("clf", lr)])

    # 3) Full interactions: WARNING - PolynomialFeatures densifies; mind RAM.
    pipe_full = Pipeline([
        ("prep", ct_main),
        ("poly", PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)),
        ("clf", lr)
    ])

    # 4) Angle-aware == main effects (kept for naming continuity/plots)
    pipe_geo = pipe_main

    # --- same_size protocol: match non-nadir N to nadir N for fair comparison ---
    # We downsample (or bootstrap) ONLY non-nadir rows; baseline remains pristine.
    if same_size:
        rng = np.random.default_rng(random_state)
        N = int(nd_mask.sum())                       # nadir count
        nn_idx = np.flatnonzero(~nd_mask)           # indices of non-nadir rows
        if nn_idx.size == 0:
            raise ValueError("No non-nadir rows available for same_size sampling.")
        replace = (nn_idx.size < N)                 # bootstrap if too few non-nadir
        take = rng.choice(nn_idx, size=N, replace=replace)
        X_non_nadir, y_non_nadir = X.iloc[take], y[take]
    else:
        X_non_nadir, y_non_nadir = X.loc[~nd_mask], y[~nd_mask]


    # --- Cross-validation: aligned splits across models for fair comparisons ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    def auroc(pipe, X_, y_, label):
        """
        Compute cross-validated AUROC for a given pipeline.
        All preprocessing happens *within* each fold (no leakage).
        """
        start = time.time()
        score = float(cross_val_score(
            pipe, X_, y_,
            cv=cv, scoring="roc_auc",
            n_jobs=-1, pre_dispatch="2*n_jobs"  # robust parallelism
        ).mean())
        logging.info(f"[AUROC] {label:16s} = {score:.4f}  ({time.time()-start:.2f}s)")
        return score

    # --- Evaluate baseline on nadir rows only ---
    au_nadir = auroc(pipe_nadir, X.loc[nd_mask], y[nd_mask], "nadir baseline")

    # --- Evaluate comparison models on (possibly size-matched) non-nadir rows ---
    au_main  = auroc(pipe_main, X_non_nadir, y_non_nadir, "main effects")
    au_full  = auroc(pipe_full, X_non_nadir, y_non_nadir, "full interactions")
    au_geo   = auroc(pipe_geo,  X_non_nadir, y_non_nadir, "angle-aware")

    # --- Report in a way that's easy to cite in text and tables ---
    results = {
        "AUROC_nadir": au_nadir,
        "AUROC_main":  au_main,
        "AUROC_full":  au_full,
        "AUROC_angle": au_geo,
        "Δ_full−nadir": au_full - au_nadir,
        "Δ_main−nadir": au_main - au_nadir,
        "Δ_full−main":  au_full - au_main,
        "ΔAUROC_geo−nadir": au_geo - au_nadir,
    }
    return results







def _auroc_fast(d, same_size=True, random_state=0):
    """
    Compute AUROC metrics for baseline (nadir-only), main effects,
    full interaction, and angle-aware models.

    Args:
        d : preprocessed dataframe
        same_size : bool
            If True, force the non-nadir sample (main/full/geo models)
            to be exactly the same size as the nadir baseline.
            Sampling is done randomly with replacement if needed.
        random_state : int
            Seed for reproducible sampling.
    """

    # --- Target variable (binary; int8 to save memory) ---
    y = (d["status"].astype(str).str.lower() == "diseased").astype("int8").to_numpy()

    # --- Nadir subset (vza < 20°) for baseline ---
    nd_mask = (d["vza"] < 20).to_numpy()
    X_nadir = d.loc[nd_mask, ["band5"]].to_numpy(dtype="float32")
    y_nadir = y[nd_mask]

    # --- Sparse feature engineering (shared by all non-nadir models) ---
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
    from scipy.sparse import csr_matrix, hstack

    # Continuous band5, scaled (CSR-friendly: with_mean=False avoids dense centering)
    band = d[["band5"]].to_numpy(dtype="float32")
    band = StandardScaler(with_mean=False).fit_transform(band)
    band = csr_matrix(band)  # make sparse for easy hstack later

    # Geometry bins encoded as sparse one-hot (drop first to avoid collinearities)
    enc = OneHotEncoder(
        categories=[
            sorted(d["vza_bin"].unique()),
            sorted(d["raa_bin"].unique())
        ],
        drop="first",
        handle_unknown="ignore",
        sparse_output=True,
        dtype="float32",
    ).fit(d[["vza_bin", "raa_bin"]])
    cats = enc.transform(d[["vza_bin", "raa_bin"]])  # CSR matrix with dummy vars

    # Main-effects design matrix = [band5 | vza_bin | raa_bin]
    X_main = hstack([band, cats], format="csr")

    # Full interaction model (two-way + triple interactions)
    poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)
    X_full = poly.fit_transform(X_main)

    # Angle-aware "geo" model = same as main effects in this setup
    X_geo = X_main

    # --- Optional size matching ---
    if same_size:
        import numpy as np
        rng = np.random.default_rng(random_state)

        # Count nadir samples
        N = nd_mask.sum()

        # Indices for non-nadir observations
        non_nadir_idx = np.flatnonzero(~nd_mask)

        # Sample exactly N rows from non-nadir (with replacement if fewer than N available)
        chosen = rng.choice(non_nadir_idx, size=int(N), replace=(non_nadir_idx.size < N))

        # Subset design matrices + labels for main/full/geo models
        X_main = X_main[chosen]
        X_full = X_full[chosen]
        X_geo  = X_geo[chosen]
        y      = y[chosen]

    # --- AUROC via cross-validation ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import time, logging

    # Logistic regression classifier
    lr = LogisticRegression(
        penalty="l2",
        max_iter=300,
        tol=1e-3,
        random_state=0,
    )

    # 5-fold stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    def auc(X, y_, label):
        """Helper: compute AUROC with timing + logging"""
        start = time.time()
        score = float(
            cross_val_score(
                lr, X, y_, cv=cv,
                scoring="roc_auc",
                n_jobs=-1, pre_dispatch="2*n_jobs"
            ).mean()
        )
        logging.info(f"AUROC for {label} computed in {time.time() - start:.2f}s")
        return score

    # Compute AUROC for all models
    au_nadir = auc(X_nadir, y_nadir, "nadir baseline")
    au_main  = auc(X_main,  y, "main effects")
    au_full  = auc(X_full,  y, "full interactions")
    au_geo   = auc(X_geo,   y, "angle-aware")

    # --- Results dictionary ---
    return {
        "AUROC_nadir": au_nadir,
        "AUROC_main":  au_main,
        "AUROC_full":  au_full,
        "AUROC_angle": au_geo,
        "Δ_full−nadir": au_full - au_nadir,
        "Δ_main−nadir": au_main - au_nadir,
        "Δ_full−main":  au_full - au_main,
        "ΔAUROC_geo−nadir": au_geo - au_nadir,
    }


def _calculate_cohens_d(d):
    """
    Calculate Cohen's d effect size across different viewing angle bins.

    Args:
        d: Preprocessed pandas.DataFrame with columns:
           - status in {"diseased", "healthy"}
           - band5 (numeric)
           - vza (numeric)
           - vza_bin, raa_bin (categorical/integers)
    Returns:
        Dictionary with Cohen's d values; all keys are strings.
    """
    import numpy as np
    import pandas as pd

    # --- Cohen's d effect size calculations ---

    # Helper function to compute Cohen's d (pooled standard deviation)
    def cohen_d(sub: pd.DataFrame) -> float:
        """Calculate pooled-SD d (diseased - healthy), return NaN if not computable."""
        g = sub.groupby("status", observed=True)["band5"].agg(mean="mean", std="std", count="count")
        # Need both groups present
        if not {"diseased", "healthy"}.issubset(g.index):
            return np.nan

        m1, m0 = g.loc["diseased", "mean"], g.loc["healthy", "mean"]
        s1, s0 = g.loc["diseased", "std"], g.loc["healthy", "std"]
        n1, n0 = g.loc["diseased", "count"], g.loc["healthy", "count"]

        dof = (n1 - 1) + (n0 - 1)
        if dof <= 0:
            return np.nan

        sp2 = ((n1 - 1) * (s1 ** 2) + (n0 - 1) * (s0 ** 2)) / dof
        if not np.isfinite(sp2) or sp2 <= 0:
            return np.nan

        sp = np.sqrt(sp2)
        return float((m1 - m0) / sp)

    # Calculate baseline Cohen's d for nadir viewing angles
    baseline = cohen_d(d.query("vza < 20"))

    # Calculate Cohen's d for each viewing geometry bin combination
    # - observed=True to silence future deprecation warning on observed default
    # - include_groups=False to silence future change in .apply behavior
    perbin = (
        d.groupby(["vza_bin", "raa_bin"], observed=True)
         .apply(cohen_d, include_groups=False)
         .rename("d")
    )

    # Get top 5 bins by absolute effect size
    top = perbin.reindex(perbin.abs().sort_values(ascending=False).index)[:5]

    # Convert tuple index keys to strings to keep Polars happy
    # e.g., "vza_bin=3_raa_bin=120": 0.87
    top_str_keys = {f"vza_bin={vb}_raa_bin={rb}": float(val)
                    for (vb, rb), val in top.items()}

    # Collect results (only string keys at top level)
    results = {
        "Cohen_d_nadir": float(baseline) if np.isfinite(baseline) else np.nan,
        "top_bins_by_|d|": top_str_keys,
    }
    return results




import re
from typing import Any, Dict, List, Tuple, Union

import polars as pl


def _extract_result_dict(lr_out: Union[dict, pl.DataFrame]) -> Dict[str, Any]:
    """
    Normalize the output of logistic_regression into a plain Python dict:
      {"AUROC_metrics": {...}, "Effect_size": {"Cohen_d_nadir": ..., "top_bins_by_|d|": {...}}}
    """
    if isinstance(lr_out, dict):
        return lr_out

    if isinstance(lr_out, pl.DataFrame):
        if lr_out.height != 1:
            raise ValueError("Expected a single-row DataFrame from logistic_regression.")
        row = lr_out.row(0, named=True)
        # Row values for struct columns arrive as nested dicts already.
        # Ensure keys we expect exist, or default to empty.
        auroc = row.get("AUROC_metrics") or {}
        eff = row.get("Effect_size") or {}
        return {"AUROC_metrics": auroc, "Effect_size": eff}

    raise TypeError("lr_out must be a dict or a single-row Polars DataFrame.")


def _parse_top_bin_key(key: Any) -> Tuple[str, str, str]:
    """
    Parse keys like 'vza_bin=0-20_raa_bin=-90-0' into ('0-20', '-90-0', original_key)
    Falls back gracefully if the format is unexpected.
    """
    if isinstance(key, tuple):
        # Shouldn't occur if you used stringified keys, but handle just in case.
        vb, rb = key
        return str(vb), str(rb), f"vza_bin={vb}_raa_bin={rb}"

    s = str(key)
    m_vza = re.search(r"vza_bin=([^_]+)", s)
    m_raa = re.search(r"raa_bin=(.+)$", s)
    vb = m_vza.group(1) if m_vza else ""
    rb = m_raa.group(1) if m_raa else ""
    return vb, rb, s


def format_logistic_results(
    lr_out: Union[dict, pl.DataFrame],
    *,
    shape: str = "wide",
    top_k: int = 5
) -> pl.DataFrame:
    """
    Make a nice Polars DataFrame from the output of logistic_regression.

    Parameters:
    - lr_out: dict or single-row Polars DataFrame returned by logistic_regression.
    - shape:
        - "wide": returns a single-row DataFrame with AUROC metrics, Cohen_d_nadir,
                  and top-k bins expanded into columns.
        - "long": returns a tidy DataFrame with one metric per row; top bins appear
                  as separate rows with vza_bin/raa_bin/d and rank.
    - top_k: how many top bins to include (if available) for effect sizes.

    Returns:
    - Polars DataFrame in the requested shape.
    """
    data = _extract_result_dict(lr_out)
    au = dict(data.get("AUROC_metrics") or {})
    eff = dict(data.get("Effect_size") or {})

    baseline = eff.get("Cohen_d_nadir", None)
    top_bins_dict = dict(eff.get("top_bins_by_|d|") or {})

    # Turn top bins into a sorted list by |d| descending
    top_bins: List[Tuple[str, str, float, float, str]] = []
    for k, v in top_bins_dict.items():
        try:
            val = float(v)
        except Exception:
            continue
        vb, rb, label = _parse_top_bin_key(k)
        top_bins.append((vb, rb, val, abs(val), label))

    top_bins.sort(key=lambda t: t[3], reverse=True)  # by |d|
    top_bins = top_bins[:top_k]

    if shape == "wide":
        # Build a single-row dict
        row: Dict[str, Any] = {}
        # AUROC summary
        row.update({k: float(v) for k, v in au.items()})
        # Cohen's d baseline
        if baseline is not None:
            try:
                row["Cohen_d_nadir"] = float(baseline)
            except Exception:
                row["Cohen_d_nadir"] = None

        # Add top-k bins as columns
        for i, (vb, rb, dval, _, label) in enumerate(top_bins, start=1):
            row[f"top{i}_vza_bin"] = vb
            row[f"top{i}_raa_bin"] = rb
            row[f"top{i}_d"] = dval
            row[f"top{i}_label"] = label

        # Ensure stable column order (optional)
        preferred_order = [
            "AUROC_nadir", "AUROC_main", "AUROC_full", "AUROC_angle",
            "Δ_main−nadir", "Δ_full−main", "ΔAUROC_geo−nadir", "Cohen_d_nadir",
        ]
        # Then top bins in numeric order
        for i in range(1, top_k + 1):
            preferred_order += [
                f"top{i}_vza_bin", f"top{i}_raa_bin", f"top{i}_d", f"top{i}_label"
            ]

        # Keep existing keys; ignore those not present
        ordered = {k: row.get(k, None) for k in preferred_order if k in row}
        # Add any extra keys at the end
        for k, v in row.items():
            if k not in ordered:
                ordered[k] = v

        return pl.DataFrame([ordered])

    elif shape == "long":
        rows: List[Dict[str, Any]] = []
        # AUROC metrics
        for k, v in au.items():
            try:
                rows.append({
                    "section": "AUROC",
                    "metric": k,
                    "value": float(v),
                    "vza_bin": None,
                    "raa_bin": None,
                    "rank": None,
                })
            except Exception:
                continue

        # Cohen's d baseline
        if baseline is not None:
            try:
                rows.append({
                    "section": "EffectSize",
                    "metric": "Cohen_d_nadir",
                    "value": float(baseline),
                    "vza_bin": None,
                    "raa_bin": None,
                    "rank": None,
                })
            except Exception:
                pass

        # Top bins as separate rows
        for i, (vb, rb, dval, _, _label) in enumerate(top_bins, start=1):
            rows.append({
                "section": "EffectSize",
                "metric": "Cohen_d_bin",
                "value": dval,
                "vza_bin": vb,
                "raa_bin": rb,
                "rank": i,
            })

        return pl.DataFrame(rows)

    else:
        raise ValueError("shape must be 'wide' or 'long'")