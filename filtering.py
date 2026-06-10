"""
Entry point wrapper for the UAV spectral filtering pipeline.

This module serves as the console_scripts entry point (uav-filter) defined
in pyproject.toml. It dynamically loads and executes the filtering pipeline
located at src/pipeline_filtering.py.

Usage:
    uav-filter --config path/to/config.yml
    python -m filtering --config path/to/config.yml
"""

import importlib.util
import sys
from pathlib import Path


def main():
    """Entry point for the uav-filter command."""
    script_path = Path(__file__).parent / "src/pipeline_filtering.py"
    if not script_path.exists():
        print(f"Error: Pipeline script not found at {script_path}", file=sys.stderr)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("filtering_script", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["filtering_script"] = module
    spec.loader.exec_module(module)

    return module.main()


if __name__ == "__main__":
    main()
