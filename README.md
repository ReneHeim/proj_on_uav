# ONCERCO UAV reflectance tools

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
make build
```

Use synthetic or user-provided data paths in tests. Do not commit raw imagery,
large derived products, parquets, TIFFs, or local mount paths.
