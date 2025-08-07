from pathlib import Path
import yaml

from src.Common.config_object import config_object


def test_config_minimal_tmp(tmp_path: Path):
    cfg = {
        "base_path": str(tmp_path),
        "inputs": {
            "date_time": {"start": "2024-01-01 00:00:00", "time_zone": "UTC"},
            "paths": {
                "cam_path": f"{tmp_path}/cams.txt",
                "dem_path": f"{tmp_path}/dem.tif",
                "orthophoto_path": f"{tmp_path}/orthos/*.tif",
                "ori": [f"{tmp_path}/orthos"],
                "mosaic_path": f"{tmp_path}/mosaic.tif",
                "ground_truth_coordinates": f"{tmp_path}/gps.csv",
            },
            "settings": {
                "number_of_processor": 1,
                "filter_radius": 1,
                "file_name": "demo",
                "bands": 5,
                "target_crs": "EPSG:32632",
            },
        },
        "outputs": {"paths": {"main_out": f"{tmp_path}/out", "plot_out": f"{tmp_path}/out/plots"}},
    }

    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    # create files the config validation expects
    for f in ["cams.txt", "dem.tif", "mosaic.tif", "gps.csv"]:
        (tmp_path / f).write_text("")

    c = config_object(str(cfg_path))
    assert c.main_extract_out.exists()
    assert c.plot_out.exists()

