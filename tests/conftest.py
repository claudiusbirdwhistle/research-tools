"""Shared pytest configuration and fixtures for research-tools-lib tests."""

import sys
from pathlib import Path

# Ensure the lib package is importable when running tests from /tools/
sys.path.insert(0, str(Path(__file__).parent.parent))
