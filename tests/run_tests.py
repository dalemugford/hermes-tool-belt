#!/usr/bin/env python3
"""Tiny entry point: ``python tests/run_tests.py``.

The plugin directory name contains a hyphen, so ``python -m unittest
dynamic-tools.tests.…`` doesn't work. This script registers the package
under an importable alias and then discovers/runs the test suite.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # so `tests` package itself loads
sys.path.insert(0, str(HERE.parent))         # make sibling modules visible

# Trigger conftest's package registration before anything else runs.
import tests.conftest  # noqa: F401,E402

loader = unittest.TestLoader()
suite = loader.discover(start_dir=str(HERE), pattern="test_*.py")
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
