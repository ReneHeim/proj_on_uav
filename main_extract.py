"""
Entry point wrapper for the UAV reflectance extraction pipeline.

This module serves as the console_scripts entry point (uav-extract) defined
in pyproject.toml. It dynamically loads and executes the main extraction
pipeline located at src/pipeline_extract_data.py.

Usage:
    uav-extract --config path/to/config.yml
    python -m main_extract --config path/to/config.yml
"""

import importlib.util
import sys
from pathlib import Path


def main():
    """Entry point for the uav-extract command."""
    script_path = Path(__file__).parent / "src/pipeline_extract_data.py"
    if not script_path.exists():
        print(f"Error: Pipeline script not found at {script_path}", file=sys.stderr)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("main_extract_script", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["main_extract_script"] = module
    spec.loader.exec_module(module)

    return module.main()


if __name__ == "__main__":
    main()
