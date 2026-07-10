"""Shared, reproducible helpers for ONCERCO research workflows.

This package contains reusable analysis infrastructure.  Model-specific logic
belongs in subpackages such as :mod:`src.research.severity`; command-line
scripts should only orchestrate a named workflow.
"""

from src.research.common import (
    RunPaths,
    artifact_path,
    configure_logging,
    load_artifact_catalog,
    log_phase,
    markdown_table,
    project_root,
    regression_metrics,
    safe_spearman,
)

__all__ = [
    "RunPaths",
    "artifact_path",
    "configure_logging",
    "load_artifact_catalog",
    "log_phase",
    "markdown_table",
    "project_root",
    "regression_metrics",
    "safe_spearman",
]
