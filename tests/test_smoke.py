import subprocess


def test_scripts_help():
    for script in [
        ["python", "src/01_main_extract_data.py", "--help"],
        ["python", "src/02_filtering.py", "--help"],
        ["python", "src/03_RPV_modelling.py", "--help"],
    ]:
        proc = subprocess.run(script, capture_output=True)
        assert proc.returncode == 0
        assert b"--config" in proc.stdout or b"--config" in proc.stderr

