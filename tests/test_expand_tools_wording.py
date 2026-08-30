"""expand_tools response clarity — the auto-management wave's wording fixes.

Locks two live-feedback fixes:

  · "N tools not enabled" must name the CAUSE (the operator's
    ``platform_toolsets`` ceiling) and its FIXABILITY (Tool Belt cannot
    restore them; enabling requires a Hermes config change);
  · persistence wording is honest per cache mode: cache-on (no sticky key)
    says the expansion "persists for this session" and never emits a
    misleading ``sticky: {enabled: false}`` block; cache-off keeps the
    sticky-residency block and its wording.
"""

from __future__ import annotations

import contextvars
import importlib
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import conftest  # noqa: F401,E402 — registers the tool_belt_plugin package

expand_tools = importlib.import_module("tool_belt_plugin.expand_tools")

if "toolsets" not in sys.modules:  # bare checkout stub (see handler tests)
    import types

    _toolsets_stub = types.ModuleType("toolsets")
    _toolsets_stub.resolve_toolset = lambda name: []  # type: ignore[attr-defined]
    _toolsets_stub.get_toolset_names = lambda: []  # type: ignore[attr-defined]
    sys.modules["toolsets"] = _toolsets_stub


def _make_state(*, sticky_key="", enabled_ceiling=None):
    state = {
        "initial_active_tools": ["read_file"],
        "expansions": set(),
        "sticky_key": sticky_key,
        "scope": "test:cli",
        "triggers_fired": [],
    }
    if enabled_ceiling is not None:
        state["enabled_ceiling"] = list(enabled_ceiling)
    return state


def _invoke(args, state, sticky_refresh_fn=None):
    cv: contextvars.ContextVar = contextvars.ContextVar("test_pred", default=None)
    cv.set(state)
    handler = expand_tools.make_handler(cv, sticky_refresh_fn=sticky_refresh_fn)
    return json.loads(handler(args))


def _patched(resolved):
    return mock.patch.object(
        expand_tools, "_resolve_category",
        return_value=("browser", list(resolved), ["browser"]),
    )


class CeilingCauseWordingTests(unittest.TestCase):
    def test_unavailable_note_names_cause_and_fixability(self):
        state = _make_state(enabled_ceiling=["read_file", "browser_navigate"])
        with _patched(["browser_navigate", "browser_click"]):
            payload = _invoke({"category": "browser"}, state)
        msg = payload["message"]
        self.assertIn("platform_toolsets", msg)
        self.assertIn("Tool Belt cannot restore them", msg)
        self.assertIn("Hermes config change", msg)

    def test_all_unavailable_message_names_cause_too(self):
        state = _make_state(enabled_ceiling=["read_file"])
        with _patched(["browser_navigate", "browser_click"]):
            payload = _invoke({"category": "browser"}, state)
        msg = payload["message"]
        self.assertIn("platform_toolsets", msg)
        self.assertIn("Hermes config change", msg)


class PersistenceWordingTests(unittest.TestCase):
    def test_cache_on_says_persists_for_this_session_and_omits_sticky(self):
        # Cache-on = no sticky key (frozen monotonic expansion set).
        state = _make_state(sticky_key="")

        def sticky_fn(key, category, tools, policy_scope=""):
            return {"enabled": False}

        with _patched(["browser_navigate"]):
            payload = _invoke({"category": "browser"}, state,
                              sticky_refresh_fn=sticky_fn)
        self.assertIn("persists for this session", payload["message"])
        self.assertNotIn("sticky", payload)

    def test_cache_off_keeps_sticky_block_and_reserves_sticky_wording(self):
        state = _make_state(sticky_key="sticky-abc")

        def sticky_fn(key, category, tools, policy_scope=""):
            return {"enabled": True, "refreshed": True, "sticky_tools": list(tools),
                    "sticky_categories": [category], "sticky_remaining_turns": {}}

        with _patched(["browser_navigate"]):
            payload = _invoke({"category": "browser"}, state,
                              sticky_refresh_fn=sticky_fn)
        self.assertIn("sticky", payload)
        self.assertTrue(payload["sticky"]["refreshed"])
        self.assertNotIn("persists for this session", payload["message"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
