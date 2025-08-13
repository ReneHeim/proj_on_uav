import geopandas as gpd
import polars as pl
from shapely.geometry import Polygon

from src.Common.polygon_filtering import check_data_polygon_overlap


def test_overlap_detection():
    # Create a tiny dataframe with points in [0, 1]x[0, 1]
    df = pl.DataFrame(
        {
            "Xw": [0.2, 0.8],
            "Yw": [0.2, 0.8],
            "band1": [1, 1],
            "elev": [100, 100],
        }
    )

    # Polygon that overlaps the points region
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf_poly = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:32632")

    has_overlap, data_bounds = check_data_polygon_overlap(df, gdf_poly)
    assert has_overlap is True
    assert len(data_bounds) == 4
