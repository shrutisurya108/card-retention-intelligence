# conftest.py
# -----------
# Pytest configuration — project root.
# Explicitly inserts the project root into sys.path so that
# `from src.ingestion import ...` resolves correctly on all Python versions.

import sys
import os
from pathlib import Path

# Absolute path to project root (the folder containing this file)
PROJECT_ROOT = Path(__file__).resolve().parent

# Only add if not already present (avoids duplicates on repeated imports)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Also set as environment variable for subprocess-based tools
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))
