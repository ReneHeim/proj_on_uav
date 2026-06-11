#!/usr/bin/env python3
"""Run feature building + model comparison for a specific year."""
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def load_feature_builder():
    """Load build_feature_sets.py so its globals can be scoped by year."""
    builder_path = PROJECT_ROOT / "src" / "features" / "build_feature_sets.py"
    spec = importlib.util.spec_from_file_location("build_feature_sets_year", builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load feature builder from {builder_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_year(year: str) -> None:
    if year not in {"2024", "2025"}:
        raise ValueError(f"Invalid year: {year}")

    feat_dir = PROJECT_ROOT / "outputs" / "features"
    res_dir = PROJECT_ROOT / "outputs" / year / "results"
    log_dir = PROJECT_ROOT / "outputs" / year / "logs"
    res_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  BUILDING FEATURES FOR {year}")
    print(f"{'=' * 60}")

    bfs = load_feature_builder()
    bfs.WEEK_DIRS = {k: v for k, v in bfs.WEEK_DIRS.items() if k.startswith(year)}
    bfs.OUT_DIR = feat_dir
    bfs.main()

    print(f"\n{'=' * 60}")
    print(f"  RUNNING MODEL COMPARISON FOR {year}")
    print(f"{'=' * 60}")

    proc = subprocess.run(
        [sys.executable, "-m", "src.models.compare_feature_sets"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=300,
        check=False,
    )

    with (log_dir / f"compare_{year}.log").open("w") as fh:
        fh.write(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Model comparison failed for {year}; see outputs/{year}/logs/compare_{year}.log"
        )

    for line in proc.stdout.splitlines():
        if any(
            kw in line
            for kw in ["Set", "AUROC", "M1", "M2", "M3", "M4", "M5", "OVERFIT", "HIGH", "Overfit"]
        ):
            print(line)

    for f in res_dir.glob("*.csv"):
        f.unlink()
    for f in (PROJECT_ROOT / "outputs" / "results").glob("*.csv"):
        shutil.copy(f, res_dir / f.name)

    feat_year_dir = PROJECT_ROOT / "outputs" / year / "features"
    feat_year_dir.mkdir(parents=True, exist_ok=True)
    for f in feat_dir.glob("*.parquet"):
        shutil.copy(f, feat_year_dir / f.name)

    print(f"\nResults saved to outputs/{year}/")
    print(f"Features saved to outputs/{year}/features/")


def main() -> None:
    year = sys.argv[1] if len(sys.argv) > 1 else "2024"
    run_year(year)


if __name__ == "__main__":
    main()
