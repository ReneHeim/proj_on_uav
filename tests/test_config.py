from pathlib import Path
import yaml

from src.Common.config_object import config_object


def test_config_minimal_tmp(tmp_path):
    """Test config loading with minimal valid config."""
    cfg = {
        "base_path": str(tmp_path),
        "inputs": {
            "date_time": {"start": "2024-01-01 12:00:00", "time_zone": "UTC"},
            "paths": {
                "cam_path": str(tmp_path / "cam.txt"),
                "dem_path": str(tmp_path / "dem.tif"),
                "orthophoto_path": str(tmp_path / "*.tif"),
                "ori": [str(tmp_path)],
                "mosaic_path": str(tmp_path / "mosaic.tif"),
                "ground_truth_coordinates": str(tmp_path / "gps.csv"),
                "polygon_file_path": str(tmp_path / "polygons.gpkg"),
            },
            "settings": {
                "number_of_processor": 1,
                "filter_radius": 1,
                "file_name": "test",
                "bands": 5,
                "target_crs": "EPSG:32632",
            },
        },
        "outputs": {
            "paths": {
                "main_out": str(tmp_path / "out"),
                "plot_out": str(tmp_path / "out" / "plots"),
            }
        },
    }
    
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    
    # Should not raise
    c = config_object(str(cfg_path))
    assert c.base_path == str(tmp_path)

