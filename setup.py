#!/usr/bin/env python3
"""Setup script for proj_on_uav package."""

from setuptools import find_packages, setup

setup(
    name="proj_on_uav",
    version="0.1.0",
    description="Multi-angular UAV reflectance extraction, filtering, and RPV modeling",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    entry_points={
        "console_scripts": [
            "uav-extract=main_extract:main",
            "uav-filter=filtering:main",
            "uav-rpv=rpv_modelling:main",
        ],
    },
)
