"""Test bootstrap for the tool-belt plugin.

The plugin directory is named with a hyphen, so we can't ``import tool-belt``
directly. ``scripts/_plugin_loader.py`` owns that import dance and is the same
loader the shipped operator scripts use; this module is a thin caller of it so
the tests and the scripts never drift apart — and so pruning ``tests/`` from a
distribution cannot break the scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _PLUGIN_DIR / "scripts"

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _plugin_loader import load_plugin_package  # noqa: E402

# ``logger_io`` and ``analyze`` are loaded eagerly so tests can address them as
# ``sys.modules["tool_belt_plugin.<name>"]``.
load_plugin_package(eager_submodules=("logger_io", "analyze"))
