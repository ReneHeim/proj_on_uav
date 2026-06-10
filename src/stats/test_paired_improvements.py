#!/usr/bin/env python3
"""Paired ΔAUROC bootstrap tests comparing multiangular vs nadir feature sets."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import wilcoxon

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "results"
N_BOOTSTRAP = 1000
SEED = 42

COMPARISONS = [
    ("M3", "M1", "multiangular VZA vs nadir bands"),
    ("M4", "M1", "multiangular VZA+RAA vs nadir bands"),
    ("M5", "M1", "angular contrast vs nadir bands"),
    ("M3", "M2", "multiangular VZA vs nadir indices"),
    ("M4", "M2", "multiangular VZA+RAA vs nadir indices"),
    ("M5", "M2", "angular contrast vs nadir indices"),
]


def load_fold_results():
    path = RESULTS_DIR / "model_comparison_by_fold.csv"
    if not path.exists():
        raise FileNotFoundError(f"Fold results not found: {path}")
    return pl.read_csv(path)


def bootstrap_ci(deltas, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(deltas, size=len(deltas), replace=True)
        means.append(np.mean(sample))
    ci_low = np.percentile(means, 2.5)
    ci_high = np.percentile(means, 97.5)
    return ci_low, ci_high


def interpret(mean_delta, ci_low, ci_high, p_value, n_pos, n_neg):
    if mean_delta > 0 and ci_low > 0 and p_value < 0.05:
        return "supported"
    elif mean_delta > 0 and n_pos > n_neg:
        return "promising"
    elif ci_low <= 0 <= ci_high:
        return "inconclusive"
    else:
        return "unsupported"


def run_comparison(fold_df, m_a, m_b, label):
    df_a = fold_df.filter(pl.col("feature_set") == m_a).sort("fold")
    df_b = fold_df.filter(pl.col("feature_set") == m_b).sort("fold")

    folds_a = set(df_a["fold"].to_list())
    folds_b = set(df_b["fold"].to_list())
    common_folds = sorted(folds_a & folds_b)

    if not common_folds:
        print(f"  SKIP {m_a} vs {m_b}: no common folds")
        return None

    df_a = df_a.filter(pl.col("fold").is_in(common_folds)).sort("fold")
    df_b = df_b.filter(pl.col("fold").is_in(common_folds)).sort("fold")

    auroc_a = df_a["AUROC"].to_numpy()
    auroc_b = df_b["AUROC"].to_numpy()
    deltas = auroc_a - auroc_b

    mean_delta = np.mean(deltas)
    median_delta = np.median(deltas)
    ci_low, ci_high = bootstrap_ci(deltas)

    try:
        if np.all(deltas == 0):
            w_p = 1.0
        else:
            _, w_p = wilcoxon(auroc_a, auroc_b)
    except (ValueError, TypeError):
        w_p = np.nan

    n_pos = int(np.sum(deltas > 0))
    n_neg = int(np.sum(deltas < 0))
    conclusion = interpret(mean_delta, ci_low, ci_high, w_p, n_pos, n_neg)

    return {
        "comparison": label,
        "n_folds": len(deltas),
        "mean_delta_AUROC": mean_delta,
        "median_delta_AUROC": median_delta,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "wilcoxon_p": w_p,
        "n_positive_deltas": n_pos,
        "n_negative_deltas": n_neg,
        "conclusion": conclusion,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading fold results...")
    fold_df = load_fold_results()

    print("\nPaired ΔAUROC tests:")
    print(f"{'='*90}")
    all_results = []
    for m_a, m_b, label in COMPARISONS:
        print(f"\n  {m_a} − {m_b}  ({label})")
        result = run_comparison(fold_df, m_a, m_b, label)
        if result is None:
            continue
        all_results.append(result)
        print(
            f"    ΔAUROC = {result['mean_delta_AUROC']:+.4f} "
            f"(CI95: [{result['ci95_low']:+.4f}, {result['ci95_high']:+.4f}])"
        )
        print(
            f"    Wilcoxon p = {result['wilcoxon_p']:.4f},  "
            f"pos/neg folds: {result['n_positive_deltas']}/{result['n_negative_deltas']}"
        )
        print(f"    Conclusion: {result['conclusion']}")

    if all_results:
        out_df = pl.DataFrame(all_results)
        out_path = RESULTS_DIR / "paired_delta_auc_tests.csv"
        out_df.write_csv(out_path)
        print(f"\nSaved: {out_path}")

        print(f"\n{'='*90}")
        print("  SUMMARY")
        print(f"{'='*90}")
        for r in all_results:
            flag = (
                "✓"
                if r["conclusion"] == "supported"
                else ("?" if r["conclusion"] == "promising" else " ")
            )
            print(
                f"  [{flag}] {r['comparison']:<48s}  "
                f"Δ={r['mean_delta_AUROC']:+.4f}  "
                f"CI=[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]  "
                f"p={r['wilcoxon_p']:.4f}  "
                f"→ {r['conclusion']}"
            )
    else:
        print("No comparisons evaluated.")


if __name__ == "__main__":
    main()
