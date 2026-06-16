"""Strict feature selection for reflectance-only modelling.

The modelling scripts keep metadata columns in their dataframes for joining,
grouping, and diagnostics. This module is the single gate for predictor
selection so metadata, labels, geometry, and identifiers cannot silently enter a
model just because they are numeric.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


FORBIDDEN_PREDICTOR_COLUMNS = {
    "plot_id",
    "ifz_id",
    "week",
    "year",
    "cult",
    "cultivar",
    "trt",
    "treatment",
    "disease_label",
    "severity_wk8",
    "future_severe_wk8",
    "target_label",
    "Xw",
    "Yw",
    "x",
    "y",
    "cx",
    "cy",
    "lon",
    "lat",
    "elev",
    "vza",
    "vaa",
    "sza",
    "saa",
    "raa",
    "delta_z",
    "delta_x",
    "delta_y",
    "distance_xy",
    "path",
}

REFLECTANCE_FEATURE_PATTERNS = [
    re.compile(r"^band[1-5]_nadir_mean$"),
    re.compile(r"^band[1-5]_vza\d+_\d+$"),
    re.compile(r"^band[1-5]_vza\d+_\d+_raa\d+_\d+$"),
    re.compile(r"^band[1-5]_vza_diff_\d+_\d+$"),
    re.compile(r"^band[1-5]_vza_ratio_\d+$"),
    re.compile(r"^band[1-5]_angular_slope$"),
    re.compile(r"^band[1-5]_range$"),
    re.compile(r"^(ndvi|gndvi|ndre)_nadir$"),
    re.compile(r"^(red_edge_ratio|nir_red_ratio)_nadir$"),
]


def is_reflectance_feature(column: str) -> bool:
    """Return True only for approved reflectance-derived predictor columns."""
    return any(pattern.match(column) for pattern in REFLECTANCE_FEATURE_PATTERNS)


def reflectance_feature_columns(columns: Iterable[str]) -> list[str]:
    """Select model predictors using a strict reflectance-only whitelist."""
    return [column for column in columns if is_reflectance_feature(column)]


def assert_reflectance_only(feature_cols: Iterable[str], context: str = "model") -> None:
    """Fail loudly if a predictor list contains metadata, labels, or geometry."""
    feature_cols = list(feature_cols)
    forbidden = sorted(set(feature_cols) & FORBIDDEN_PREDICTOR_COLUMNS)
    non_reflectance = sorted(column for column in feature_cols if not is_reflectance_feature(column))
    if forbidden or non_reflectance:
        details = []
        if forbidden:
            details.append(f"forbidden={forbidden}")
        if non_reflectance:
            details.append(f"non_reflectance={non_reflectance}")
        raise ValueError(f"Non-reflectance predictors in {context}: " + "; ".join(details))
