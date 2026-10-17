"""PyInstaller hook: make the frozen app behave like a folder-based launch."""

import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    os.chdir(Path(sys.executable).resolve().parent)
