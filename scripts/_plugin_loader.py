"""Import helper: register the plugin directory as ``tool_belt_plugin``.

The plugin directory is named with a hyphen, so ``import tool-belt`` is not
valid Python. Everything that needs the plugin from the outside — the shipped
operator scripts and the test suite — loads it here instead: the package is
built with :func:`importlib.util.spec_from_file_location`, given the plugin
directory as its search path, and registered under the importable alias
``tool_belt_plugin``. Submodules (``logger_io``, ``predictor`` …) then resolve
normally through the package machinery, so the internal ``from . import …``
statements keep working.

This module lives under ``scripts/`` so that the shipped commands do not depend
on ``tests/`` being present: a distribution may prune the test directory.
``tests/conftest.py`` calls into here rather than the other way round.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

PLUGIN_DIR = Path(__file__).resolve().parent.parent
PACKAGE_NAME = "tool_belt_plugin"


def load_plugin_package(eager_submodules: Iterable[str] = ()) -> ModuleType:
    """Return the plugin package, importing and registering it if needed.

    Idempotent: a package already in ``sys.modules`` is reused, so callers that
    run inside the test suite share its module objects rather than loading a
    second copy.

    ``eager_submodules`` names submodules to import immediately, which makes
    them addressable as ``sys.modules["tool_belt_plugin.<name>"]``.
    """
    module = sys.modules.get(PACKAGE_NAME)
    if module is None:
        if str(PLUGIN_DIR) not in sys.path:
            sys.path.insert(0, str(PLUGIN_DIR))
        spec = importlib.util.spec_from_file_location(
            PACKAGE_NAME,
            PLUGIN_DIR / "__init__.py",
            submodule_search_locations=[str(PLUGIN_DIR)],
        )
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise ImportError(f"cannot load the tool-belt package from {PLUGIN_DIR}")
        module = importlib.util.module_from_spec(spec)
        # Registered before execution so the package's own relative imports
        # resolve against it.
        sys.modules[PACKAGE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(PACKAGE_NAME, None)
            raise
    for sub in eager_submodules:
        importlib.import_module(f"{PACKAGE_NAME}.{sub}")
    return module
