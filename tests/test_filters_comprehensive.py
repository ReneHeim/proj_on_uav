"""
Comprehensive unit tests for the filters module.
Tests all functions including plotting and masking capabilities.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.Common.filters import (
    OSAVI_index_filtering,
    add_mask_and_plot,
    excess_green_filter,
    plot_heatmap,
    plot_spectrogram,
)


class TestOSAVIFiltering:
    """Test OSAVI index filtering functionality."""

    def test_osavi_basic_calculation(self):
        """Test basic OSAVI calculation without filtering."""
        df = pl.DataFrame(
            {
                "band5": [0.8, 0.6, 0.4, 0.2],  # NIR
                "band3": [0.2, 0.3, 0.4, 0.5],  # Red
                "Xw": [1, 2, 3, 4],
                "Yw": [1, 2, 3, 4],
            }
        )

        result = OSAVI_index_filtering(df)

        assert "OSAVI" in result.columns
        assert len(result) == 4

        # Check OSAVI values are reasonable (should be positive for vegetation)
        osavi_values = result["OSAVI"].to_list()
        assert all(not np.isnan(val) for val in osavi_values)
        assert all(not np.isinf(val) for val in osavi_values)

    def test_osavi_with_threshold_filtering(self):
        """Test OSAVI filtering with threshold."""
        df = pl.DataFrame(
            {
                "band5": [0.8, 0.6, 0.4, 0.2],
                "band3": [0.2, 0.3, 0.4, 0.5],
                "Xw": [1, 2, 3, 4],
                "Yw": [1, 2, 3, 4],
            }
        )

        result = OSAVI_index_filtering(df, removal_threshold=0.1)

        assert "OSAVI" in result.columns
        assert len(result) <= 4  # Some rows might be filtered out

        if len(result) > 0:
            assert all(result["OSAVI"] > 0.1)

    def test_osavi_missing_columns(self):
        """Test OSAVI with missing required columns."""
        df = pl.DataFrame(
            {"band1": [0.1, 0.2, 0.3], "band2": [0.2, 0.3, 0.4], "Xw": [1, 2, 3], "Yw": [1, 2, 3]}
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            OSAVI_index_filtering(df)

    def test_osavi_edge_cases(self):
        """Test OSAVI with edge cases (zero values, negative values)."""
        df = pl.DataFrame(
            {
                "band5": [0.0, 0.1, -0.1, 0.5],
                "band3": [0.0, 0.1, 0.1, 0.2],
                "Xw": [1, 2, 3, 4],
                "Yw": [1, 2, 3, 4],
            }
        )

        result = OSAVI_index_filtering(df)

        assert "OSAVI" in result.columns
        assert len(result) == 4


class TestExcessGreenFilter:
    """Test Excess Green filtering functionality."""

    def test_excess_green_basic_calculation(self):
        """Test basic Excess Green calculation without filtering."""
        df = pl.DataFrame(
            {
                "band1": [0.1, 0.2, 0.3, 0.4],  # Blue
                "band2": [0.3, 0.4, 0.5, 0.6],  # Green
                "band3": [0.2, 0.3, 0.4, 0.5],  # Red
                "Xw": [1, 2, 3, 4],
                "Yw": [1, 2, 3, 4],
            }
        )

        result = excess_green_filter(df)

        assert "ExcessGreen" in result.columns
        assert len(result) == 4

        # Check ExcessGreen values
        eg_values = result["ExcessGreen"].to_list()
        assert all(not np.isnan(val) for val in eg_values)
        assert all(not np.isinf(val) for val in eg_values)

    def test_excess_green_with_threshold_filtering(self):
        """Test Excess Green filtering with threshold."""
        df = pl.DataFrame(
            {
                "band1": [0.1, 0.2, 0.3, 0.4],
                "band2": [0.3, 0.4, 0.5, 0.6],
                "band3": [0.2, 0.3, 0.4, 0.5],
                "Xw": [1, 2, 3, 4],
                "Yw": [1, 2, 3, 4],
            }
        )

        result = excess_green_filter(df, removal_threshold=0.1)

        assert "ExcessGreen" in result.columns
        assert len(result) <= 4

        if len(result) > 0:
            assert all(result["ExcessGreen"] > 0.1)

    def test_excess_green_missing_columns(self):
        """Test Excess Green with missing required columns."""
        df = pl.DataFrame(
            {"band1": [0.1, 0.2, 0.3], "band5": [0.2, 0.3, 0.4], "Xw": [1, 2, 3], "Yw": [1, 2, 3]}
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            excess_green_filter(df)

    def test_excess_green_edge_cases(self):
        """Test Excess Green with edge cases."""
        df = pl.DataFrame(
            {
                "band1": [0.0, 0.1, -0.1, 0.5],
                "band2": [0.0, 0.1, 0.1, 0.2],
                "band3": [0.0, 0.1, 0.1, 0.1],
                "Xw": [1, 2, 3, 4],
                "Yw": [1, 2, 3, 4],
            }
        )

        result = excess_green_filter(df)

        assert "ExcessGreen" in result.columns
        assert len(result) == 4


class TestPlottingFunctions:
    """Test plotting functions."""

    def test_plot_heatmap_basic(self):
        """Test basic heatmap plotting."""
        df = pl.DataFrame(
            {"Xw": [1, 2, 3, 4, 5], "Yw": [1, 2, 3, 4, 5], "OSAVI": [0.1, 0.2, 0.3, 0.4, 0.5]}
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "plots"
            output_path.mkdir(exist_ok=True)

            # Should not raise
            plot_heatmap(df, "OSAVI", str(output_path), sample_size=5)

            # Check if file was created
            expected_file = output_path / "heatmap_OSAVI.png"
            assert expected_file.exists()

    def test_plot_heatmap_large_dataset(self):
        """Test heatmap with large dataset (should sample)."""
        # Create large dataset
        n_points = 200000
        df = pl.DataFrame(
            {
                "Xw": np.random.rand(n_points),
                "Yw": np.random.rand(n_points),
                "OSAVI": np.random.rand(n_points),
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "plots"
            output_path.mkdir(exist_ok=True)

            # Should not raise and should sample
            plot_heatmap(df, "OSAVI", str(output_path), sample_size=1000)

            expected_file = output_path / "heatmap_OSAVI.png"
            assert expected_file.exists()

    def test_plot_spectrogram_basic(self):
        """Test basic spectrogram plotting."""
        df = pl.DataFrame(
            {
                "band1": [0.1, 0.2, 0.3, 0.4, 0.5],
                "band2": [0.2, 0.3, 0.4, 0.5, 0.6],
                "band3": [0.3, 0.4, 0.5, 0.6, 0.7],
                "band4": [0.4, 0.5, 0.6, 0.7, 0.8],
                "band5": [0.5, 0.6, 0.7, 0.8, 0.9],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "plots"
            output_path.mkdir(exist_ok=True)
            wavelengths = [475, 560, 668, 717, 842]

            # Should not raise
            plot_spectrogram(df, 5, wavelengths, sample_size=5, output_path=str(output_path))

            expected_file = output_path / "spectrogram.png"
            assert expected_file.exists()

    def test_add_mask_and_plot_above_threshold(self):
        """Test masking and plotting above threshold."""
        df = pl.DataFrame(
            {"Xw": [1, 2, 3, 4, 5], "Yw": [1, 2, 3, 4, 5], "OSAVI": [0.1, 0.2, 0.3, 0.4, 0.5]}
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "plots"
            output_path.mkdir(exist_ok=True)

            # Should not raise
            add_mask_and_plot(df, "OSAVI", 0.3, above=True, output_path=str(output_path))

            expected_file = output_path / "mask_plot_OSAVI.png"
            assert expected_file.exists()

    def test_add_mask_and_plot_below_threshold(self):
        """Test masking and plotting below threshold."""
        df = pl.DataFrame(
            {"Xw": [1, 2, 3, 4, 5], "Yw": [1, 2, 3, 4, 5], "OSAVI": [0.1, 0.2, 0.3, 0.4, 0.5]}
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "plots"
            output_path.mkdir(exist_ok=True)

            # Should not raise
            add_mask_and_plot(df, "OSAVI", 0.3, above=False, output_path=str(output_path))

            expected_file = output_path / "mask_plot_OSAVI.png"
            assert expected_file.exists()


class TestFilterIntegration:
    """Test integration of multiple filters."""

    def test_osavi_and_excess_green_combined(self):
        """Test applying both OSAVI and Excess Green filters."""
        df = pl.DataFrame(
            {
                "band1": [0.1, 0.2, 0.3, 0.4],  # Blue
                "band2": [0.3, 0.4, 0.5, 0.6],  # Green
                "band3": [0.2, 0.3, 0.4, 0.5],  # Red
                "band5": [0.8, 0.6, 0.4, 0.2],  # NIR
                "Xw": [1, 2, 3, 4],
                "Yw": [1, 2, 3, 4],
            }
        )

        # Apply both filters
        result = OSAVI_index_filtering(df, removal_threshold=0.1)
        result = excess_green_filter(result, removal_threshold=0.1)

        assert "OSAVI" in result.columns
        assert "ExcessGreen" in result.columns
        assert len(result) <= 4

    def test_filter_preserves_original_columns(self):
        """Test that filtering preserves original data columns."""
        df = pl.DataFrame(
            {
                "band1": [0.1, 0.2, 0.3, 0.4],
                "band2": [0.3, 0.4, 0.5, 0.6],
                "band3": [0.2, 0.3, 0.4, 0.5],
                "band5": [0.8, 0.6, 0.4, 0.2],
                "Xw": [1, 2, 3, 4],
                "Yw": [1, 2, 3, 4],
                "elev": [100, 101, 102, 103],
            }
        )

        result = OSAVI_index_filtering(df)
        result = excess_green_filter(result)

        # Check original columns are preserved
        for col in ["band1", "band2", "band3", "band5", "Xw", "Yw", "elev"]:
            assert col in result.columns


if __name__ == "__main__":
    pytest.main([__file__])
