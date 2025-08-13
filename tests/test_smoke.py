import subprocess


def test_scripts_help():
    for script in [
        ["python", "-m", "src.01_main_extract_data", "--help"],
        ["python", "-m", "src.02_filtering", "--help"],
        ["python", "-m", "src.03_RPV_modelling", "--help"],
    ]:
        proc = subprocess.run(script, capture_output=True)
        assert proc.returncode == 0
        assert b"--config" in proc.stdout or b"--config" in proc.stderr
