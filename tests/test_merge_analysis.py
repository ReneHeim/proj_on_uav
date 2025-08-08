import polars as pl

from src.Common.merge_analysis import analyze_kdtree_matching


def test_kdtree_matching_no_crash():
    # synthetic small dataframes with overlapping coordinates
    df_dem = pl.DataFrame({
        "Xw": [0.0, 1.0, 2.0],
        "Yw": [0.0, 1.0, 2.0],
    })
    df_all = pl.DataFrame({
        "Xw": [0.0, 1.0, 2.0],
        "Yw": [0.0, 1.0, 2.0],
    })

    stats = analyze_kdtree_matching(df_dem, df_all, precision=0)
    assert "exact_matches" in stats
    assert stats["exact_matches"] >= 3

