import subprocess


def test_scripts_help():
    for script in [
        ["python", "-m", "main_extract_data", "--help"],
        ["python", "-m", "filtering", "--help"],
        ["python", "-m", "RPV_modelling", "--help"],
    ]:
        proc = subprocess.run(script, capture_output=True)
        assert proc.returncode == 0
        assert b"--config" in proc.stdout or b"--config" in proc.stderr
