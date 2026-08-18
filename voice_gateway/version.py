"""Resolve the exact source revision used by the running gateway."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


def source_version(project_dir: Path | None = None) -> str:
    """Return an operator override or the current Git source revision."""
    override = os.environ.get("VOICE_GATEWAY_SOURCE_VERSION", "").strip()
    if override:
        return override

    root = project_dir or Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "describe", "--always", "--dirty"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"
