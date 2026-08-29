"""Regression: expand_tools must never be deferred behind the Tool Search bridge.

Hermes' native tiered-disclosure bridge (``tools.tool_search`` →
``assemble_tool_defs``) runs at tool-definition time, *upstream* of tool-belt's
``_build_api_kwargs`` narrowing. Its classifier, ``is_deferrable_tool_name``,
would defer ``expand_tools`` — a non-core, non-MCP plugin tool — behind
``tool_search``/``describe``/``call``, hiding the one tool whose whole job is
expanding tools. Our narrowing only ever subtracts, so once the bridge removes
``expand_tools`` we can never add it back, and ``always_carry`` (policy.yaml)
never gets a chance to protect it (``A = always_carry ∩ E`` is empty when the
tool is absent from the enabled ceiling).

``_pin_expand_tools_visible()`` wraps that classifier so ``expand_tools`` is
never classified deferrable — and nothing else changes. These tests guard the
wrap's contract against a controlled stand-in for ``tools.tool_search`` (the
suite stays decoupled from the hermes-agent import graph; the live bridge
integration is exercised manually against the venv):

  * expand_tools → not deferrable, every other tool → unchanged,
  * the wrap is idempotent (a second install is a no-op),
  * a missing/unimportable ``tools.tool_search`` fails open (no raise, no-op).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

# Reuse conftest's plugin-loader bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401
import unittest

plugin = sys.modules["tool_belt_plugin"]
EXPAND_TOOLS_NAME = plugin.expand_tools_mod.SCHEMA["name"]


def _make_fake_tool_search() -> types.ModuleType:
    """A stand-in ``tools.tool_search`` whose classifier defers *everything*.

    Mirrors the real module's contract for a plain plugin tool: without the
    pin, ``expand_tools`` is deferrable. The pin must flip only ``expand_tools``
    and pass every other name through untouched.
    """
    mod = types.ModuleType("tools.tool_search")
    seen: list[str] = []

    def is_deferrable_tool_name(name, *args, **kwargs):  # noqa: ANN001
        seen.append(name)
        return True

    mod.is_deferrable_tool_name = is_deferrable_tool_name
    mod.seen = seen  # type: ignore[attr-defined]
    return mod


class ExpandToolsPinTest(unittest.TestCase):
    def _install_fake(self) -> types.ModuleType:
        """Inject a fake ``tools.tool_search`` and auto-restore after the test."""
        fake_ts = _make_fake_tool_search()
        fake_tools = types.ModuleType("tools")
        fake_tools.tool_search = fake_ts  # type: ignore[attr-defined]
        patcher = mock.patch.dict(
            sys.modules, {"tools": fake_tools, "tools.tool_search": fake_ts}
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return fake_ts

    def test_pins_only_expand_tools(self) -> None:
        ts = self._install_fake()
        original = ts.is_deferrable_tool_name

        plugin._pin_expand_tools_visible()

        # expand_tools is pinned visible; everything else keeps the default.
        self.assertFalse(ts.is_deferrable_tool_name(EXPAND_TOOLS_NAME))
        self.assertTrue(ts.is_deferrable_tool_name("mcp__probe__x"))
        self.assertTrue(ts.is_deferrable_tool_name("some_plugin_tool"))

        # The wrapper is marked and keeps the original reachable for unwrapping.
        self.assertTrue(
            getattr(ts.is_deferrable_tool_name, "_tool_belt_expand_pin", False)
        )
        self.assertIs(ts.is_deferrable_tool_name.__wrapped__, original)

    def test_idempotent(self) -> None:
        ts = self._install_fake()

        plugin._pin_expand_tools_visible()
        wrapped_once = ts.is_deferrable_tool_name
        self.assertTrue(getattr(wrapped_once, "_tool_belt_expand_pin", False))

        # A second install must not re-wrap (no stacking of wrappers).
        plugin._pin_expand_tools_visible()
        self.assertIs(ts.is_deferrable_tool_name, wrapped_once)

    def test_fail_open_when_tool_search_unimportable(self) -> None:
        # A non-package ``tools`` with no importable ``tool_search`` submodule
        # makes ``import tools.tool_search`` raise — the pin must swallow it.
        fake_tools = types.ModuleType("tools")  # no __path__ → not a package
        patcher = mock.patch.dict(sys.modules, {"tools": fake_tools})
        patcher.start()
        self.addCleanup(patcher.stop)
        sys.modules.pop("tools.tool_search", None)

        try:
            plugin._pin_expand_tools_visible()
        except Exception as exc:  # pragma: no cover — the point of the test
            self.fail(f"_pin_expand_tools_visible must fail open, raised: {exc!r}")


if __name__ == "__main__":
    unittest.main()
