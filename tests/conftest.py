"""Test bootstrap for the dynamic-tools plugin.

The plugin directory is named with a hyphen, so we can't ``import
dynamic-tools`` directly — Python won't let us. Instead we load the
package via ``importlib.util.spec_from_file_location`` with the plugin
directory wired up as the package's search path, then register it under
the importable name ``dynamic_tools_plugin``. Submodules (``logger_io``,
``analyze`` …) load automatically through the package mechanism so the
existing ``from . import …`` statements keep working.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parent.parent
_PACKAGE_NAME = "dynamic_tools_plugin"


def _load_plugin_package() -> None:
    if _PACKAGE_NAME in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        _PACKAGE_NAME,
        _PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(_PLUGIN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    # Eagerly load submodules used by tests so they're addressable via
    # sys.modules["dynamic_tools_plugin.<name>"].
    for sub in ("logger_io", "analyze"):
        importlib.import_module(f"{_PACKAGE_NAME}.{sub}")


_load_plugin_package()
