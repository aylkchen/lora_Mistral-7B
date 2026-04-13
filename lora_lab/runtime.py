from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_utf8_runtime() -> None:
    """Re-exec on Windows so TRL can import chat templates with UTF-8."""
    if os.name != "nt":
        return
    if os.environ.get("PYTHONUTF8") == "1":
        return

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    completed = subprocess.run([sys.executable, *sys.argv], env=env, check=False)
    raise SystemExit(completed.returncode)
