"""Compatibility entry point for the public extraction pipeline."""

from oncerco_uav.pipelines.extract_data import main


if __name__ == "__main__":
    raise SystemExit(main())
