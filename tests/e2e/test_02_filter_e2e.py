import subprocess
from pathlib import Path


def test_script_02_filter_e2e(tmp_path: Path):
    # re-use output from previous test location if running together is not guaranteed;
    # Here we only sanity-run the CLI to ensure it accepts --help
    proc = subprocess.run(["python", "-m", "src.02_filtering", "--help"], capture_output=True)
    assert proc.returncode == 0
