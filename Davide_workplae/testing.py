import subprocess

exiftool_path = r"C:\ExifTool\exiftool.exe"

try:
    result = subprocess.run([exiftool_path, "-ver"], check=True, text=True, capture_output=True)
    print(f"ExifTool version: {result.stdout.strip()}")
except FileNotFoundError:
    print(f"ExifTool not found at {exiftool_path}")
except Exception as e:
    print(f"Error: {e}")
