"""
Configuration management for UAV reflectance extraction pipeline.
Handles loading, parsing, and validating YAML configuration files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class AttrDict:
    """A dictionary that allows attribute access to its keys."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, AttrDict(value))
            elif isinstance(value, list):
                setattr(
                    self,
                    key,
                    [AttrDict(item) if isinstance(item, dict) else item for item in value],
                )
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def values(self) -> List[Any]:
        return list(self._data.values())

    def items(self) -> List[tuple]:
        return list(self._data.items())

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"AttrDict({self._data})"


class ConfigValidator:
    """Validates configuration files for required fields and data types."""

    REQUIRED_FIELDS = {
        "base_path": str,
        "inputs.date_time.start": str,
        "inputs.date_time.time_zone": str,
        "inputs.paths.cam_path": str,
        "inputs.paths.dem_path": str,
        "inputs.paths.orthophoto_path": str,
        "inputs.paths.ori": list,
        "inputs.paths.mosaic_path": str,
        "inputs.paths.ground_truth_coordinates": str,
        "inputs.paths.polygon_file_path": str,
        "inputs.settings.number_of_processor": int,
        "inputs.settings.filter_radius": (int, float),
        "inputs.settings.file_name": str,
        "inputs.settings.bands": int,
        "inputs.settings.target_crs": str,
        "outputs.paths.main_out": str,
        "outputs.paths.plot_out": str,
    }

    @classmethod
    def validate_config(cls, config: AttrDict) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        for field_path, expected_type in cls.REQUIRED_FIELDS.items():
            value = cls._get_nested_value(config, field_path)

            if value is None:
                errors.append(f"Missing required field: {field_path}")
            elif not isinstance(value, expected_type):
                errors.append(
                    f"Field {field_path} must be {expected_type.__name__}, got {type(value).__name__}"
                )

        return errors

    @staticmethod
    def _get_nested_value(obj: Any, path: str) -> Any:
        """Get nested value from object using dot notation."""
        keys = path.split(".")
        current = obj

        for key in keys:
            if hasattr(current, key):
                current = getattr(current, key)
            elif isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current


def load_config(config_path: Union[str, Path]) -> AttrDict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        AttrDict: Configuration object with attribute access

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
        ValueError: If config validation fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}")

    if not isinstance(config_data, dict):
        raise ValueError(f"Configuration file must contain a dictionary, got {type(config_data)}")

    config = AttrDict(config_data)

    # Validate configuration
    errors = ConfigValidator.validate_config(config)
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        raise ValueError(error_msg)

    return config


def config_object(config_path: Union[str, Path]) -> AttrDict:
    """
    Create configuration object from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        AttrDict: Configuration object with processed paths and settings
    """
    config = load_config(config_path)

    # Process base_path substitution
    base_path = config.base_path

    # Helper function to substitute base_path in strings
    def substitute_base_path(value: str) -> str:
        if isinstance(value, str) and "{base_path}" in value:
            return value.format(base_path=base_path)
        return value

    # Process paths with base_path substitution
    for path_key in [
        "cam_path",
        "dem_path",
        "orthophoto_path",
        "mosaic_path",
        "ground_truth_coordinates",
        "polygon_file_path",
    ]:
        if hasattr(config.inputs.paths, path_key):
            path_value = getattr(config.inputs.paths, path_key)
            setattr(config.inputs.paths, path_key, substitute_base_path(path_value))

    # Process ori paths
    if hasattr(config.inputs.paths, "ori"):
        ori_paths = []
        for ori_path in config.inputs.paths.ori:
            ori_paths.append(substitute_base_path(ori_path))
        config.inputs.paths.ori = ori_paths

    # Process output paths
    for output_key in ["main_out", "plot_out"]:
        if hasattr(config.outputs.paths, output_key):
            output_value = getattr(config.outputs.paths, output_key)
            setattr(config.outputs.paths, output_key, substitute_base_path(output_value))

    # Create output directories
    Path(config.outputs.paths.main_out).mkdir(parents=True, exist_ok=True)
    Path(config.outputs.paths.plot_out).mkdir(parents=True, exist_ok=True)

    # Add convenience properties for backward compatibility
    config.main_extract_out = config.outputs.paths.main_out
    config.main_extract_cam_path = config.inputs.paths.cam_path
    config.main_extract_dem_path = config.inputs.paths.dem_path
    config.main_extract_ori = config.inputs.paths.ori
    config.main_extract_name = config.inputs.settings.file_name
    config.main_extract_path_list_tag = config.inputs.paths.orthophoto_path
    config.main_polygon_path = config.inputs.paths.polygon_file_path
    config.main_extract_out_polygons_df = config.outputs.paths.plot_out
    config.start_date = config.inputs.date_time.start
    config.time_zone = config.inputs.date_time.time_zone
    config.plot_out = Path(config.outputs.paths.plot_out)
    config.target_crs = config.inputs.settings.target_crs
    config.bands = config.inputs.settings.bands

    logging.info(f"Configuration loaded successfully from {config_path}")
    logging.info(f"Base path: {base_path}")
    logging.info(f"Output directory: {config.outputs.paths.main_out}")

    return config
