import subprocess


def test_script_03_rpv_e2e():
    # Sanity-run CLI --help; full E2E requires real data produced by step 02
    proc = subprocess.run(["python", "-m", "src.03_RPV_modelling", "--help"], capture_output=True)
    assert proc.returncode == 0
