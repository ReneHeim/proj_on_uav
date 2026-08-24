# ONCERCO UAV reflectance tools

[![CI](https://github.com/ReneHeim/proj_on_uav/actions/workflows/ci.yml/badge.svg)](https://github.com/ReneHeim/proj_on_uav/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/ReneHeim/proj_on_uav/branch/main/graph/badge.svg)](https://codecov.io/gh/ReneHeim/proj_on_uav)

`oncerco-uav` is a public Python package for extracting and modeling
multi-angular multispectral UAV reflectance. It provides polygon-aware pixel
extraction, spectral filtering, and RPV modeling for MicaSense-compatible
five-band stacks.

## Public package layout

- `src/oncerco_uav/core/`: configuration, validation, preprocessing, and search
  helpers.
- `src/oncerco_uav/extract/`: camera, raster, geometry, and polygon extraction.
- `src/oncerco_uav/filter/`: parquet loading and spectral-index filtering.
- `src/oncerco_uav/modelling/`: RPV fitting and modeling helpers.
- `src/oncerco_uav/pipelines/`: extraction, filtering, and RPV command modules.
- `scripts/preprocessing/`: generic RedEdge-P and orthorectification tools.

The package does not include project data, private analysis, research results,
presentations, or machine-local configurations.

## Install

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run the pipelines

Copy `examples/config_file_example.yml`, set its input/output paths, and run:

```bash
python -m oncerco_uav.pipelines.extract_data --config my_config.yml
python -m oncerco_uav.pipelines.filtering --config my_config.yml
python -m oncerco_uav.pipelines.modelling --config my_config.yml --band band1
```

The same commands are available as `uav-extract`, `uav-filter`, and `uav-rpv`.

For RedEdge-P stacks the output order is Blue, Green, Red, Red edge, NIR.
Reflectance stacks use `uint16` values with `reflectance = pixel / 32767`.

## Development

```bash
make test
make test-e2e
make coverage
make build
```

CI enforces at least **95% line coverage** for the public `oncerco_uav`
package. The current local baseline is **95.06%** (220 non-e2e tests passed;
the six e2e tests are run separately).

Use synthetic or user-provided data paths in tests. Do not commit raw imagery,
large derived products, parquets, TIFFs, or local mount paths.
