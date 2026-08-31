"""Test bootstrap for the tool-belt plugin.

The plugin directory is named with a hyphen, so we can't ``import tool-belt``
directly. ``scripts/_plugin_loader.py`` owns that import dance and is the same
loader the shipped operator scripts use; this module is a thin caller of it so
the tests and the scripts never drift apart — and so pruning ``tests/`` from a
distribution cannot break the scripts.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Suite-wide sandbox: the runtime resolves its state dir from $HERMES_HOME at
# call time, so any test that reaches a real logging path without pinning its
# own home would append fixture rows to the OPERATOR'S live telemetry (found
# live: 40 model='m' rows in ~/.hermes predictions.jsonl). Tests that need a
# specific home still override this in their own setUp.
_SANDBOX_HOME = tempfile.mkdtemp(prefix="tool-belt-tests-home-")
os.environ["HERMES_HOME"] = _SANDBOX_HOME

_PLUGIN_DIR = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _PLUGIN_DIR / "scripts"

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _plugin_loader import load_plugin_package  # noqa: E402

# ``logger_io`` and ``analyze`` are loaded eagerly so tests can address them as
# ``sys.modules["tool_belt_plugin.<name>"]``.
load_plugin_package(eager_submodules=("logger_io", "analyze"))

# Pristine in-code defaults, snapshotted BEFORE any test mutates the module
# config. Tier-0/wire tests restore from this so "deployed default" assertions
# are immune to sibling tests that seed _CONFIG without cleanup.
import sys as _sys
PRISTINE_CONFIG = dict(_sys.modules["tool_belt_plugin"]._CONFIG)
