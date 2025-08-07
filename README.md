## Overview

This repository provides a Python pipeline to extract multi-angular reflectance and geometry from UAV orthophotos, filter data spatially using polygons, and fit RPV models per plot and week.

## Quickstart

- Install Python 3.10+.
- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

- Configure paths by copying `src/config_file_example.yml` and editing to your data.

  ```bash
  cp src/config_file_example.yml my_config.yml
  ```

- Run extraction:

  ```bash
  python src/01_main_extract_data.py --config my_config.yml
  ```

- Run filtering and per‑polygon splitting:

  ```bash
  python src/02_filtering.py --config my_config.yml
  ```

- Run RPV fitting:

  ```bash
  python src/03_RPV_modelling.py --config my_config.yml --band band1
  ```

Alternatively, use the provided Makefile targets (`make install`, `make extract`, `make filter`, `make rpv`).

## Notes

- Outputs are created under `outputs.paths.main_out` defined in your YAML config.
- DEM and orthophotos must cover the same extent; optional co‑registration can be enabled with `--alignment` in the extract step.

## Citation

Heim, R. HJ., Okole, N., Steppe, K., van Labeke, M.C., Geedicke, I., & Maes, W. H. (2024). An applied framework to unlocking multi‑angular UAV reflectance data: A case study for classification of plant parameters in maize (Zea mays). Precision Agriculture. (accepted)
