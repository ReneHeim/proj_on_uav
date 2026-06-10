# Multi-angular UAV Reflectance Extractor

[![CI](https://github.com/ReneHeim/proj_on_uav/workflows/CI/badge.svg)](https://github.com/ReneHeim/proj_on_uav/actions)
[![Codecov](https://codecov.io/gh/ReneHeim/proj_on_uav/graph/badge.svg)](https://app.codecov.io/gh/ReneHeim/proj_on_uav)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202-blue)](LICENSE)

## Overview

This repository provides a Python pipeline to extract multi-angular reflectance and geometry from UAV orthophotos, filter data spatially using polygons, and fit RPV models per plot and week.

version: 0.11.1

## Features

- **Multi-angular reflectance extraction** from UAV orthophotos
- **Spatial filtering** using polygon boundaries
- **RPV model fitting** for vegetation analysis
- **Comprehensive testing** with unit and E2E tests
- **Modern development tools** (pre-commit, CI/CD, type hints)
- **Cross-platform compatibility** (Linux, macOS, Windows)

## Quickstart

### Prerequisites

- Python 3.10 or higher
- [Agisoft Metashape Professional](https://www.agisoft.com/) (for ortophoto alignment)
### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ReneHeim/proj_on_uav.git
   cd proj_on_uav
   ```

2. **Install dependencies:**
   ```bash
   # Using pip
   pip install -r requirements.txt

   # Or using the Makefile
   make install
   ```

3. **Install pre-commit hooks (optional but recommended):**
   ```bash
   pre-commit install
   ```

### Configuration

1. **Copy and edit the configuration file:**
   ```bash
   cp src/config_file_example.yml my_config.yml
   ```

2. **Edit `my_config.yml` with your data paths:**
   ```yaml
   base_path: '/path/to/your/data'
   inputs:
     date_time:
       start: "2024-12-07 12:00:00"
       time_zone: "Europe/Berlin"
     paths:
       cam_path: "{base_path}/cameras.txt"
       dem_path: "{base_path}/dem.tif"
       orthophoto_path: "{base_path}/orthophotos/*.tif"
       # ... other paths
   ```

## Input Data Requirements

### Required Files
- **DEM**: Digital Elevation Model as GeoTIFF
- **Orthophotos**: Multi-band orthophotos as GeoTIFF files
- **Camera positions**: Text file with camera metadata
- **Polygon file**: GeoPackage with plot boundaries
- **Ground truth coordinates**: CSV with sample locations

### Data Format
- **Orthophotos**: Multi-band GeoTIFF (typically 5 bands)
- **DEM**: Single-band GeoTIFF with same extent as orthophotos
- **Camera file**: Tab-separated with columns: PhotoID, X, Y, Z, Omega, Phi, Kappa, etc.
- **Polygons**: GeoPackage (.gpkg) with plot geometries
- **Coordinates**: CSV with columns: id, lon, lat

## Output

### Generated Files
- **Parquet files**: Per-image extracted data with reflectance and geometry
- **Filtered data**: Spectral-filtered datasets split by polygon
- **RPV results**: CSV files with fitted RPV parameters per plot and week
- **Plots**: Visualization of angles, bands, and filtering results

### Output Structure
```
output/
├── extract/              # Extracted per-pixel data
│   ├── IMG_0001_0.tif.parquet
│   └── ...
├── plots/                # Generated plots
│   ├── angles_data/      # Viewing angle plots
│   ├── bands_data/       # Band reflectance plots
│   └── ...
└── RPV_Results/          # RPV model results
    └── V5/
        └── rpv_week1_band1_results.csv
```


### Usage

#### Step 1: Extract Data
```bash
# Extract per-pixel data from orthophotos and DEM
python -m src.pipeline_extract_data --config my_config.yml

# With optional co-registration
python -m src.pipeline_extract_data --config my_config.yml --alignment

# Without polygon filtering
python -m src.pipeline_extract_data --config my_config.yml --no-polygon
```

#### Step 2: Apply Filters
```bash
# Apply spectral filters and split by polygon
python -m src.pipeline_filtering --config my_config.yml
```

#### Step 3: Fit RPV Models
```bash
# Fit RPV models for a specific band
python -m src.pipeline_modelling --config my_config.yml --band band1
```

#### Using Makefile (Alternative)
```bash
make extract    # Run extraction
make filter     # Run filtering
make rpv        # Run RPV modeling
```

## Testing

### Run Tests
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Run specific test categories
python -m pytest tests/           # Unit tests
python -m pytest tests/e2e/       # End-to-end tests
python -m pytest tests/test_smoke.py  # CLI smoke tests
```

### Development Tools
```bash
make lint       # Run linting
make format     # Format code
make install    # Install dependencies
```

## Project Structure

```
proj_on_uav/
├── src/                        # Source code
│   ├── core/                   # Core utilities
│   │   ├── config_object.py    # Configuration management
│   │   ├── logging.py          # Logging setup
│   │   ├── preprocess.py       # Data preprocessing
│   │   └── search.py           # File search utilities
│   ├── extract/                # Data extraction modules
│   │   ├── camera.py           # Camera position & angle calculations
│   │   ├── date_time.py        # Timezone & datetime handling
│   │   ├── merge_analysis.py   # DEM/orthophoto merge & KD-tree matching
│   │   ├── polygon_filtering.py# Spatial filtering by polygon
│   │   └── raster.py           # Raster I/O, alignment, plotting
│   ├── filter/                 # Spectral filtering modules
│   │   ├── data_loader.py      # Polygon-based data loading
│   │   └── filters.py          # OSAVI, Excess Green, spectral indices
│   ├── modelling/              # RPV modelling modules
│   │   ├── rpv.py              # RPV model fitting (least squares)
│   │   ├── processing.py       # Weekly RPV batch processing
│   │   └── plotting.py         # RPV result visualization
│   ├── stats/                  # Statistical analysis modules
│   │   ├── ANOVA.py            # ANOVA & Tukey's HSD
│   │   ├── Logistic_regression.py # Logistic regression & AUROC
│   │   ├── plotting.py         # Statistical plotting (KDE)
│   │   └── processing.py       # Weekly stats batch processing
│   ├── pipeline_extract_data.py# Step 1: Extract per-pixel data
│   ├── pipeline_filtering.py   # Step 2: Apply spectral filters
│   ├── pipeline_modelling.py   # Step 3: Fit RPV models
│   └── config_file_example.yml # Example configuration
├── main_extract.py             # Entry point wrapper (uav-extract)
├── filtering.py                # Entry point wrapper (uav-filter)
├── rpv_modelling.py            # Entry point wrapper (uav-rpv)
├── tests/                      # Test suite
│   ├── e2e/                    # End-to-end tests
│   ├── test_*.py               # Unit tests
│   └── test_smoke.py           # CLI smoke tests
├── Documentation/              # Documentation
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project configuration
├── Makefile                    # Development commands
└── README.md                   # This file
```


## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `python -m pytest`
5. Format code: `make format`
6. Commit: `git commit -m 'Add feature'`
7. Push: `git push origin feature-name`
8. Create a Pull Request

### Development Setup
```bash
# Install in development mode
pip install -e .

# Install pre-commit hooks
pre-commit install

# Run tests before committing
python -m pytest
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're running from the project root
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **File not found**: Check paths in your config file
4. **Memory issues**: Reduce `number_of_processor` in config
5. **Alignment errors**: Use `--alignment` flag for co-registration

### Getting Help

- Check the [Documentation](Documentation/) folder
- Review existing [Issues](https://github.com/ReneHeim/proj_on_uav/issues)
- Create a new issue with detailed error information

## Citation

If you use this software in your research, please cite:

```
Heim, R. HJ., Okole, N., Steppe, K., van Labeke, M.C., Geedicke, I., & Maes, W. H. (2024).
An applied framework to unlocking multi‑angular UAV reflectance data:
A case study for classification of plant parameters in maize (Zea mays).
Precision Agriculture. (accepted)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Development supported by [Your Institution]
- Built with [Polars](https://pola.rs/), [GeoPandas](https://geopandas.org/), and [Rasterio](https://rasterio.readthedocs.io/)
