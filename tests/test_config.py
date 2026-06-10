from pathlib import Path

import pytest
import yaml

from src.core.config_object import AttrDict, ConfigValidator, config_object, load_config

# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_config_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        config_object("/nonexistent/path/config.yml")


def test_config_invalid_yaml(tmp_path):
    cfg_path = tmp_path / "bad.yml"
    cfg_path.write_text(": invalid: yaml: :::")
    with pytest.raises(yaml.YAMLError):
        config_object(str(cfg_path))


def test_config_non_dict_yaml_raises(tmp_path):
    cfg_path = tmp_path / "list_config.yml"
    cfg_path.write_text(yaml.safe_dump([1, 2, 3]))
    with pytest.raises(ValueError, match="must contain a dictionary"):
        config_object(str(cfg_path))


def test_config_missing_required_fields(tmp_path):
    cfg = {"base_path": str(tmp_path)}
    cfg_path = tmp_path / "minimal.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="Configuration validation failed"):
        config_object(str(cfg_path))


def test_config_wrong_field_types(tmp_path):
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
                "bands": "five",  # wrong type
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
    cfg_path = tmp_path / "wrong_type.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="must be int"):
        config_object(str(cfg_path))


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


def _make_minimal_config_dict(tmp_path):
    return {
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


def test_config_minimal_tmp(tmp_path):
    """Test config loading with minimal valid config."""
    cfg = _make_minimal_config_dict(tmp_path)
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    c = config_object(str(cfg_path))
    assert c.base_path == str(tmp_path)


def test_config_base_path_substitution(tmp_path):
    cfg = _make_minimal_config_dict(tmp_path)
    cfg["inputs"]["paths"]["cam_path"] = "{base_path}/data/cam.txt"
    cfg["inputs"]["paths"]["dem_path"] = "{base_path}/data/dem.tif"
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    c = config_object(str(cfg_path))
    assert c.inputs.paths.cam_path == f"{tmp_path}/data/cam.txt"
    assert c.inputs.paths.dem_path == f"{tmp_path}/data/dem.tif"


def test_config_nested_attribute_access(tmp_path):
    cfg = _make_minimal_config_dict(tmp_path)
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    c = config_object(str(cfg_path))
    assert c.inputs.date_time.start == "2024-01-01 12:00:00"
    assert c.inputs.date_time.time_zone == "UTC"
    assert c.inputs.settings.bands == 5


def test_config_override_properties(tmp_path):
    cfg = _make_minimal_config_dict(tmp_path)
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    c = config_object(str(cfg_path))
    assert c.start_date == "2024-01-01 12:00:00"
    assert c.bands == 5
    assert c.target_crs == "EPSG:32632"
    assert c.time_zone == "UTC"
    assert c.main_extract_name == "test"


# ---------------------------------------------------------------------------
# AttrDict
# ---------------------------------------------------------------------------


def test_attrdict_attribute_access():
    ad = AttrDict({"a": 1, "b": 2})
    assert ad.a == 1
    assert ad.b == 2


def test_attrdict_dict_access():
    ad = AttrDict({"a": 1, "b": 2})
    assert ad["a"] == 1
    assert ad["b"] == 2


def test_attrdict_set_attribute():
    ad = AttrDict({"a": 1})
    ad.a = 10
    assert ad.a == 10


def test_attrdict_setitem():
    ad = AttrDict({"a": 1})
    ad["a"] = 20
    assert ad["a"] == 20
    assert ad.a == 20


def test_attrdict_get_with_default():
    ad = AttrDict({"a": 1})
    assert ad.get("a") == 1
    assert ad.get("missing") is None
    assert ad.get("missing", 42) == 42


def test_attrdict_keys():
    ad = AttrDict({"a": 1, "b": 2})
    assert set(ad.keys()) == {"a", "b"}


def test_attrdict_values():
    ad = AttrDict({"a": 1, "b": 2})
    assert set(ad.values()) == {1, 2}


def test_attrdict_items():
    ad = AttrDict({"a": 1, "b": 2})
    items = dict(ad.items())
    assert items == {"a": 1, "b": 2}


def test_attrdict_contains():
    ad = AttrDict({"a": 1})
    assert "a" in ad
    assert "b" not in ad


def test_attrdict_nested_creation():
    ad = AttrDict({"outer": {"inner": 42}})
    assert isinstance(ad.outer, AttrDict)
    assert ad.outer.inner == 42
    assert ad["outer"]["inner"] == 42


# ---------------------------------------------------------------------------
# ConfigValidator
# ---------------------------------------------------------------------------


def test_get_nested_value_valid():
    ad = AttrDict({"a": {"b": {"c": 99}}})
    val = ConfigValidator._get_nested_value(ad, "a.b.c")
    assert val == 99


def test_get_nested_value_missing():
    ad = AttrDict({"a": {"b": 1}})
    val = ConfigValidator._get_nested_value(ad, "a.x.y")
    assert val is None


def test_validate_config_valid(tmp_path):
    cfg = _make_minimal_config_dict(tmp_path)
    ad = AttrDict(cfg)
    errors = ConfigValidator.validate_config(ad)
    assert errors == []


def test_validate_config_invalid():
    ad = AttrDict({"base_path": "/tmp"})
    errors = ConfigValidator.validate_config(ad)
    assert len(errors) > 0
    assert any("Missing required field" in e for e in errors)
