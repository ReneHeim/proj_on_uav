import polars as pl

from src.Utils.filtering.filters import OSAVI_index_filtering, excess_green_filter


def test_osavi_and_excess_green():
    df = pl.DataFrame(
        {
            "band1": [0.2, 0.1],
            "band2": [0.3, 0.4],
            "band3": [0.1, 0.1],
            "band5": [0.6, 0.5],
        }
    )
    df = OSAVI_index_filtering(df)
    assert "OSAVI" in df.columns
    df = excess_green_filter(df)
    assert "ExcessGreen" in df.columns
