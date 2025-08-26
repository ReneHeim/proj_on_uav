"""
Wrapper module for the RPV modelling script.
This provides a valid module name for the entry point.
"""

import importlib.util
import sys
from pathlib import Path


def main():
    """Entry point for the uav-rpv command."""
    # Import the main function from the script file
    script_path = Path(__file__).parent / "pipeline_RPV_modelling.py"
    spec = importlib.util.spec_from_file_location("rpv_modelling_script", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rpv_modelling_script"] = module
    spec.loader.exec_module(module)

    # Call the main function
    return module.main()


if __name__ == "__main__":
    main()
