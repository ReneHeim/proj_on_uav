"""
Entry point wrapper for the UAV RPV modelling pipeline.

This module serves as the console_scripts entry point (uav-rpv) defined
in pyproject.toml. It dynamically loads and executes the RPV modelling
pipeline located at src/pipeline_modelling.py.

Usage:
    uav-rpv --config path/to/config.yml --band band1
    python -m rpv_modelling --config path/to/config.yml --band band1
"""

import importlib.util
import sys
from pathlib import Path


def main():
    """Entry point for the uav-rpv command."""
    script_path = Path(__file__).parent / "src/pipeline_modelling.py"
    if not script_path.exists():
        print(f"Error: Pipeline script not found at {script_path}", file=sys.stderr)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("modelling_script", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["modelling_script"] = module
    spec.loader.exec_module(module)

    return module.main()


if __name__ == "__main__":
    main()
