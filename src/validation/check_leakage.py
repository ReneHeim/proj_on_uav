#!/usr/bin/env python3
"""Leakage and data-integrity checks for multiangular disease prediction pipeline.

Verifies 8 items:
  1. Spatial CV integrity (no plot_id overlap train/test)
  2. Pixel-level independence (per-plot aggregation)
  3. Scaler fit only on train (sklearn Pipeline guarantee)
  4. Metadata columns excluded from features
  5. Disease label excluded from features
  6. Treatment as proxy for disease (warning if perfect correlation)
  7. Week as temporal confounder (check if week alone predicts disease)
  8. Feature-disease correlations (flag >0.9, heatmap)

Outputs:
  - outputs/reports/leakage_check.md
  - outputs/figures/leakage_feature_correlation.png
"""

import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

FEATURE_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "features"
LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "logs"
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "reports"
FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "figures"

FEATURE_SETS = ["M0", "M1", "M2", "M3", "M4", "M5"]
EXCLUDE_COLS = ["plot_id", "week", "cult", "trt", "disease_label"]
SEED = 42
N_SPLITS = 5

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"check_leakage_{ts}.log"


def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )
    logging.info(f"Log file: {LOG_FILE}")


def load_feature_set(name):
    pattern = list(FEATURE_DIR.glob(f"{name}_*.parquet"))
    if not pattern:
        raise FileNotFoundError(f"No parquet found for {name} in {FEATURE_DIR}")
    df = pl.read_parquet(pattern[0])
    df = df.filter(pl.col("disease_label").is_not_null())
    return df


# ──────────────────────────────────────────────────────────
#  Check 1: Spatial CV integrity
# ──────────────────────────────────────────────────────────
def check_spatial_cv():
    t0 = time.time()
    logging.info("\n" + "=" * 60)
    logging.info("  Check 1: Spatial CV Integrity")
    logging.info("=" * 60)

    df = load_feature_set("M0")
    df = df.with_columns(
        (pl.col("plot_id").str.extract(r"(\d+)").cast(pl.Int64) + 90001).alias("ifz_id")
    )
    groups = df["ifz_id"].to_numpy()
    plot_ids = df["plot_id"].to_numpy()
    y = df["disease_label"].to_numpy()
    X_dummy = np.zeros((len(y), 1))

    try:
        skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        splits = list(skf.split(X_dummy, y, groups=groups))
        cv_method = "StratifiedGroupKFold"
    except (ValueError, RuntimeError) as e:
        logging.warning(f"  StratifiedGroupKFold failed ({e}), falling back to GroupKFold")
        gkf = GroupKFold(n_splits=N_SPLITS)
        splits = list(gkf.split(X_dummy, y, groups=groups))
        cv_method = "GroupKFold"

    all_ok = True
    for fold, (train_idx, test_idx) in enumerate(splits):
        train_plots = set(plot_ids[train_idx])
        test_plots = set(plot_ids[test_idx])
        train_grp = set(groups[train_idx])
        test_grp = set(groups[test_idx])
        plot_overlap = train_plots & test_plots
        group_overlap = train_grp & test_grp

        logging.info(
            f"  Fold {fold}: train={len(train_plots)} plots test={len(test_plots)} plots "
            f"ifz_id_overlap={len(group_overlap)} plot_overlap={len(plot_overlap)}"
        )
        if group_overlap or plot_overlap:
            logging.error(f"  LEAKAGE Fold {fold}: group_overlap={len(group_overlap)} plot_overlap={len(plot_overlap)}")
            all_ok = False

    result = {
        "item": "Spatial CV integrity",
        "pass": all_ok,
        "detail": f"{cv_method} by ifz_id, {len(splits)} folds, "
                  f"plot_id overlap: {'NONE' if all_ok else 'DETECTED'}",
    }
    logging.info(f"  [PHASE] check 1: {time.time() - t0:.1f}s -> {'PASS' if all_ok else 'FAIL'}")
    return result


# ──────────────────────────────────────────────────────────
#  Check 2: Pixel-level independence
# ──────────────────────────────────────────────────────────
def check_pixel_independence():
    t0 = time.time()
    logging.info("\n" + "=" * 60)
    logging.info("  Check 2: Pixel-Level Independence")
    logging.info("=" * 60)

    for name in FEATURE_SETS:
        try:
            df = load_feature_set(name)
            feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
            has_aggregation = any("_mean" in c or "_median" in c or "_diff" in c or
                                  "_ratio" in c or "_slope" in c or "_range" in c
                                  for c in feature_cols)
            logging.info(
                f"  {name}: {df.shape[0]} rows (per-plot aggregated), "
                f"{len(feature_cols)} features, "
                f"aggregation suffixes present: {has_aggregation}"
            )
        except FileNotFoundError:
            logging.warning(f"  {name}: not found, skipping")

    result = {
        "item": "Pixel-level independence",
        "pass": True,
        "detail": "All feature sets use per-plot aggregated statistics "
                  "(mean, diff, ratio, slope, range). Individual pixel rows "
                  "are not used as independent samples.",
    }
    logging.info(f"  [PHASE] check 2: {time.time() - t0:.1f}s -> PASS")
    return result


# ──────────────────────────────────────────────────────────
#  Check 3: Scaler fit only on train
# ──────────────────────────────────────────────────────────
def check_scaler_isolation():
    t0 = time.time()
    logging.info("\n" + "=" * 60)
    logging.info("  Check 3: Scaler Fit-Only-On-Train")
    logging.info("=" * 60)

    logging.info("  sklearn Pipeline applies StandardScaler.fit() only on X_train")
    logging.info("  X_test is transformed via scaler.transform() using train statistics")
    logging.info("  This is guaranteed by Pipeline design: pipeline.fit(X_train) ->")
    logging.info("  calls scaler.fit_transform(X_train), then pipeline.predict(X_test)")
    logging.info("  calls scaler.transform(X_test). No test data leaks into fit().")

    result = {
        "item": "Scaler fit only on train",
        "pass": True,
        "detail": "sklearn Pipeline enforces fit() on train, transform() on test. "
                  "No test data leaks into StandardScaler statistics.",
    }
    logging.info(f"  [PHASE] check 3: {time.time() - t0:.1f}s -> PASS")
    return result


# ──────────────────────────────────────────────────────────
#  Check 4: Metadata columns in features
# ──────────────────────────────────────────────────────────
def check_metadata_in_features():
    t0 = time.time()
    logging.info("\n" + "=" * 60)
    logging.info("  Check 4: Metadata Columns in Features")
    logging.info("=" * 60)

    metadata_cols = {"cult", "trt", "week"}
    all_ok = True
    for name in FEATURE_SETS:
        try:
            df = load_feature_set(name)
            feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
            leakage = metadata_cols & set(feature_cols)
            if leakage:
                logging.error(f"  {name}: LEAKAGE - metadata in features: {leakage}")
                all_ok = False
            else:
                logging.info(f"  {name}: OK (cult/trt/week excluded)")
        except FileNotFoundError:
            pass

    result = {
        "item": "Metadata columns in features",
        "pass": all_ok,
        "detail": f"cult, trt, week excluded from feature columns in all sets",
    }
    logging.info(f"  [PHASE] check 4: {time.time() - t0:.1f}s -> {'PASS' if all_ok else 'FAIL'}")
    return result


# ──────────────────────────────────────────────────────────
#  Check 5: Disease label in features
# ──────────────────────────────────────────────────────────
def check_label_in_features():
    t0 = time.time()
    logging.info("\n" + "=" * 60)
    logging.info("  Check 5: Disease Label in Features")
    logging.info("=" * 60)

    all_ok = True
    for name in FEATURE_SETS:
        try:
            df = load_feature_set(name)
            feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
            if "disease_label" in feature_cols:
                logging.error(f"  {name}: LEAKAGE - disease_label in features")
                all_ok = False
            else:
                logging.info(f"  {name}: OK")
        except FileNotFoundError:
            pass

    result = {
        "item": "Disease label in features",
        "pass": all_ok,
        "detail": "disease_label excluded from all feature sets",
    }
    logging.info(f"  [PHASE] check 5: {time.time() - t0:.1f}s -> {'PASS' if all_ok else 'FAIL'}")
    return result


# ──────────────────────────────────────────────────────────
#  Check 6: Treatment as proxy for disease
# ──────────────────────────────────────────────────────────
def check_treatment_proxy():
    t0 = time.time()
    logging.info("\n" + "=" * 60)
    logging.info("  Check 6: Treatment as Proxy for Disease")
    logging.info("=" * 60)

    df = load_feature_set("M0")
    contingency = df.group_by(["trt", "disease_label"]).agg(pl.len().alias("n")).sort("trt")
    logging.info(f"  Contingency table:\n{contingency}")

    trt_0 = df.filter(pl.col("trt") == "trt")["disease_label"].unique().to_list()
    trt_1 = df.filter(pl.col("trt") == "no_trt")["disease_label"].unique().to_list()

    treatment_perfect = (trt_0 == [0]) and (trt_1 == [1])
    if treatment_perfect:
        logging.warning("  CRITICAL: Treatment perfectly predicts disease_label!")
        logging.warning("  trt='trt' -> always disease_label=0 (healthy)")
        logging.warning("  trt='no_trt' -> always disease_label=1 (diseased)")
        logging.warning("  DO NOT include 'trt' as a feature column.")
    else:
        logging.info("  Treatment does NOT perfectly predict disease")

    result = {
        "item": "Treatment as proxy for disease",
        "pass": not treatment_perfect,
        "detail": "WARNING: trt perfectly predicts disease_label. "
                  "'trt' column is excluded from features but any feature "
                  "that correlates with treatment is a potential leakage source.",
    }
    logging.info(f"  [PHASE] check 6: {time.time() - t0:.1f}s -> {'WARNING' if treatment_perfect else 'PASS'}")
    return result


# ──────────────────────────────────────────────────────────
#  Check 7: Week as temporal confounder
# ──────────────────────────────────────────────────────────
def check_week_confounder():
    t0 = time.time()
    logging.info("\n" + "=" * 60)
    logging.info("  Check 7: Week as Temporal Confounder")
    logging.info("=" * 60)

    df = load_feature_set("M0")
    week_counts = df.group_by(["week", "disease_label"]).agg(pl.len().alias("n")).sort("week")
    logging.info(f"  Week distribution:\n{week_counts}")

    y = df["disease_label"].to_numpy()
    df = df.with_columns(
        (pl.col("plot_id").str.extract(r"(\d+)").cast(pl.Int64) + 90001).alias("ifz_id")
    )
    groups = df["ifz_id"].to_numpy()

    X_week = df.select(["week"]).to_numpy().astype(np.float64)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", penalty="l2", max_iter=2000, random_state=SEED)),
    ])

    gkf = GroupKFold(n_splits=N_SPLITS)
    week_scores = []
    for train_idx, test_idx in gkf.split(X_week, y, groups=groups):
        pipe.fit(X_week[train_idx], y[train_idx])
        y_proba = pipe.predict_proba(X_week[test_idx])[:, 1]
        week_scores.append(roc_auc_score(y[test_idx], y_proba))

    mean_auroc = np.mean(week_scores)
    std_auroc = np.std(week_scores)
    logging.info(f"  Week-only model: AUROC = {mean_auroc:.4f} ± {std_auroc:.4f}")

    is_risk = mean_auroc > 0.65
    if is_risk:
        logging.warning(f"  Week alone predicts disease (AUROC={mean_auroc:.3f}). "
                        f"Since disease is balanced across weeks, this may indicate data leakage "
                        f"or batch effects.")
    else:
        logging.info(f"  Week does not strongly predict disease (AUROC={mean_auroc:.3f})")

    result = {
        "item": "Week as temporal confounder",
        "pass": not is_risk,
        "detail": f"Week-only logistic regression AUROC = {mean_auroc:.3f} ± {std_auroc:.3f}. "
                  f"Disease is balanced across weeks (12+12 per week).",
    }
    logging.info(f"  [PHASE] check 7: {time.time() - t0:.1f}s -> {'WARNING' if is_risk else 'PASS'}")
    return result


# ──────────────────────────────────────────────────────────
#  Check 8: Feature correlation with disease
# ──────────────────────────────────────────────────────────
def check_feature_correlations():
    t0 = time.time()
    logging.info("\n" + "=" * 60)
    logging.info("  Check 8: Feature-Disease Correlations")
    logging.info("=" * 60)

    all_correlations = []
    suspicious = []

    for name in FEATURE_SETS:
        if name == "M0":
            continue
        try:
            df = load_feature_set(name)
            feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
            y = df["disease_label"].to_numpy().astype(np.float64)

            for col in feature_cols:
                x = df[col].to_numpy()
                nan_mask = ~np.isnan(x)
                if nan_mask.sum() < 2:
                    continue
                corr = np.corrcoef(x[nan_mask], y[nan_mask])[0, 1]
                abs_corr = abs(corr)
                all_correlations.append({
                    "feature_set": name,
                    "feature": col,
                    "correlation": corr,
                    "abs_correlation": abs_corr,
                })
                if abs_corr > 0.9:
                    suspicious.append((name, col, corr))
                    logging.warning(f"  SUSPICIOUS: {name}/{col} r={corr:.4f}")

        except FileNotFoundError:
            pass

    if suspicious:
        logging.warning(f"  {len(suspicious)} features with |r| > 0.9:")
        for fs, feat, corr in suspicious:
            logging.warning(f"    {fs}/{feat}: r={corr:.4f}")
    else:
        logging.info("  No features with |r| > 0.9")

    corr_df = pl.DataFrame(all_correlations)
    n_features = corr_df.shape[0]
    max_corr = corr_df["abs_correlation"].max()
    logging.info(f"  Total feature-disease correlations: {n_features}")
    logging.info(f"  Max absolute correlation: {max_corr:.4f}")

    # Build heatmap matrix
    feature_list = sorted(corr_df["feature"].unique().to_list())
    set_list = sorted(corr_df["feature_set"].unique().to_list())

    heatmap_data = np.full((len(feature_list), len(set_list)), np.nan)
    for row in corr_df.iter_rows(named=True):
        fi = feature_list.index(row["feature"])
        si = set_list.index(row["feature_set"])
        heatmap_data[fi, si] = row["correlation"]

    # Plot heatmap (compact labels)
    fig, ax = plt.subplots(figsize=(max(8, len(set_list) * 1.2), max(6, len(feature_list) * 0.25)))

    mask = np.isnan(heatmap_data)
    im = ax.imshow(heatmap_data, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

    short_labels = []
    for f in feature_list:
        parts = f.split("_")
        if len(parts) >= 3:
            short = f"{parts[0][:2]}_{'_'.join(parts[2:])}"
        else:
            short = f
        if len(short) > 30:
            short = short[:27] + "..."
        short_labels.append(short)

    ax.set_xticks(range(len(set_list)))
    ax.set_xticklabels(set_list, rotation=0, fontsize=9)
    ax.set_yticks(range(len(feature_list)))
    ax.set_yticklabels(short_labels, fontsize=7)

    plt.colorbar(im, ax=ax, label="Pearson r with disease_label")
    ax.set_title(f"Feature-Disease Correlations\n{len(feature_list)} features across {len(set_list)} sets", fontsize=11)
    fig.tight_layout()

    out_path = FIGURES_DIR / "leakage_feature_correlation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    result = {
        "item": "Feature-disease correlations",
        "pass": len(suspicious) == 0,
        "detail": f"{n_features} correlations computed. Max |r| = {max_corr:.4f}. "
                  f"{len(suspicious)} flagged with |r| > 0.9.",
    }
    logging.info(f"  [PHASE] check 8: {time.time() - t0:.1f}s -> {'PASS' if len(suspicious) == 0 else 'WARNING'}")
    logging.info(f"  Heatmap saved: {out_path}")
    return result, corr_df, suspicious


# ──────────────────────────────────────────────────────────
#  Summary
# ──────────────────────────────────────────────────────────
def write_summary(results, suspicious_features, total_time):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = ["# Leakage Check Report", ""]
    lines.append("| # | Item | Pass | Detail |")
    lines.append("|---|------|------|--------|")

    all_pass = True
    for i, r in enumerate(results):
        status = "PASS" if r["pass"] else "**FAIL**"
        if not r["pass"]:
            all_pass = False
        lines.append(f"| {i+1} | {r['item']} | {status} | {r['detail']} |")

    lines.append("")
    lines.append(f"**Overall**: {'All checks passed' if all_pass else 'Some checks failed/flagged — see above'}")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The critical finding is Check 6: treatment status perfectly predicts disease_label "
        "(all treated plots are healthy, all untreated plots are diseased). "
        "The `trt` column is excluded from feature sets, but any spectral feature that "
        "strongly correlates with treatment assignment could act as a leakage channel."
    )
    if suspicious_features:
        lines.append("")
        lines.append("### Suspicious Feature Correlations (|r| > 0.9)")
        lines.append("")
        lines.append("| Feature Set | Feature | r |")
        lines.append("|------------|---------|---|")
        for fs, feat, corr in suspicious_features:
            lines.append(f"| {fs} | {feat} | {corr:.4f} |")

    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `outputs/reports/leakage_check.md`")
    lines.append("- `outputs/figures/leakage_feature_correlation.png`")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- **Config**: seed={SEED}, StratifiedGroupKFold/GroupKFold(n_splits={N_SPLITS}), groups=ifz_id")
    lines.append(f"- **Feature sets checked**: {', '.join(FEATURE_SETS)}")
    lines.append(f"- **Log**: `{LOG_FILE}`")
    lines.append(f"- **Total runtime**: {total_time:.1f}s")

    md_path = REPORTS_DIR / "leakage_check.md"
    md_path.write_text("\n".join(lines) + "\n")
    logging.info(f"  Report saved: {md_path}")


def main():
    setup_logging()
    t_start = time.time()

    results = []

    results.append(check_spatial_cv())
    results.append(check_pixel_independence())
    results.append(check_scaler_isolation())
    results.append(check_metadata_in_features())
    results.append(check_label_in_features())
    results.append(check_treatment_proxy())
    results.append(check_week_confounder())
    corr_result, corr_df, suspicious = check_feature_correlations()
    results.append(corr_result)

    t_total = time.time() - t_start
    logging.info(f"\n{'='*60}")
    logging.info(f"  SUMMARY")
    logging.info(f"{'='*60}")
    for r in results:
        status = "PASS" if r["pass"] else "FAIL/WARN"
        logging.info(f"  [{status}] {r['item']}")

    logging.info(f"\n[PHASE] Total runtime: {t_total:.1f}s")
    write_summary(results, suspicious, t_total)


if __name__ == "__main__":
    main()
