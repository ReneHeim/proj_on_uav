# Contributing

`oncerco-uav` is the public extraction and RPV package. Contributions should
remain generic and reproducible: do not add field data, unpublished analysis,
presentations, machine-local paths, or generated products.

## Development setup

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the checks before opening a pull request:

```bash
make lint
make test-unit
make build
```

The end-to-end tests require user-supplied imagery/configuration and are kept
separate from the unit-test command.

## Repository layout

- `src/oncerco_uav/` contains the installable package.
- `src/oncerco_uav/pipelines/` contains the three command-line pipelines.
- `scripts/preprocessing/` contains generic RedEdge-P and orthorectification
  helpers.
- `tests/` contains package tests and synthetic fixtures.
- `examples/` contains path-neutral configuration examples.

New reusable implementation belongs under `src/oncerco_uav/`; root-level
commands are compatibility wrappers only. Keep input and output paths in
configuration or command-line arguments rather than hard-coding local mounts.

## Pull requests

Explain the behavior changed, include or update tests, and report the commands
used for verification. Keep commits focused and avoid committing files that
match the repository's data/artifact exclusions.
