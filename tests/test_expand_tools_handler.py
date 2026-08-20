"""Direct tests for the ``expand_tools`` handler response contract.

These tests bypass Hermes' tool-dispatch layer and call the handler closure
the plugin registers, so the asserts pin the JSON payload the model would
actually see.
"""

from __future__ import annotations

import contextvars
import importlib
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

if "tool_belt_plugin" not in sys.modules:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent.parent))
    sys.path.insert(0, str(here.parent))
    from tests import conftest  # noqa: F401 — side-effect: register package

expand_tools = importlib.import_module("tool_belt_plugin.expand_tools")


def _make_state(*, initial_allowed=(), cut_tools=(), expansions=None):
    return {
        "initial_allowed_tools": list(initial_allowed),
        "cut_tools": list(cut_tools),
        "expansions": set(expansions or set()),
        "sticky_key": "",
        "scope": "test:cli",
        "triggers_fired": [],
    }


def _invoke(args, state):
    """Run the handler against a fresh ContextVar holding ``state``."""
    cv: contextvars.ContextVar = contextvars.ContextVar("test_pred", default=None)
    cv.set(state)
    handler = expand_tools.make_handler(cv, sticky_refresh_fn=None)
    return json.loads(handler(args))


class AlreadyAvailableCategoryTests(unittest.TestCase):
    """If the resolved tools were already in initial_allowed, the response
    must not claim they were 'added' — and the message must say so."""

    def test_fully_already_available_reports_no_additions(self):
        # Pretend the live toolset table maps "file" to a set of names that
        # are entirely contained in the model's initial allowed list.
        resolved = ["read_file", "write_file"]
        state = _make_state(initial_allowed=resolved)

        with mock.patch.object(expand_tools, "_resolve_category",
                               return_value=("file", list(resolved), ["file", "browser"])):
            payload = _invoke({"category": "file"}, state)

        self.assertTrue(payload["success"], payload)
        self.assertEqual(payload["tools_added"], [],
                         "no tool counts as 'added' when all were already available")
        self.assertEqual(sorted(payload["already_available_tools"]), sorted(resolved))
        self.assertEqual(sorted(payload["resolved_tools"]), sorted(resolved))
        self.assertIn("already loaded", payload["message"].lower())

    def test_partial_overlap_only_credits_genuinely_new_tools(self):
        resolved = ["read_file", "write_file", "patch"]
        state = _make_state(initial_allowed=["read_file"])

        with mock.patch.object(expand_tools, "_resolve_category",
                               return_value=("file", list(resolved), ["file"])):
            payload = _invoke({"category": "file"}, state)

        self.assertTrue(payload["success"])
        self.assertEqual(sorted(payload["tools_added"]), ["patch", "write_file"])
        self.assertEqual(payload["already_available_tools"], ["read_file"])

    def test_re_expanding_same_category_does_not_double_count(self):
        resolved = ["browser_navigate", "browser_click"]
        # First call: nothing was initially available; both tools are added.
        state = _make_state(initial_allowed=[])
        with mock.patch.object(expand_tools, "_resolve_category",
                               return_value=("browser", list(resolved), ["browser"])):
            first = _invoke({"category": "browser"}, state)
            self.assertEqual(sorted(first["tools_added"]), sorted(resolved))
            # Second call against the SAME state should report zero new tools.
            second = _invoke({"category": "browser"}, state)
        self.assertEqual(second["tools_added"], [])


class CaseInsensitiveCategoryInputTests(unittest.TestCase):
    """The handler should forgive simple case + whitespace differences."""

    def test_uppercase_resolves_to_canonical_name(self):
        # Live table only knows "browser"; handler should accept "BROWSER".
        fake_available = ["browser", "file", "terminal"]
        def fake_resolve(name):
            return ["browser_navigate"] if name == "browser" else []

        with mock.patch.object(expand_tools, "_available_toolset_names",
                               return_value=fake_available), \
             mock.patch("toolsets.resolve_toolset", side_effect=fake_resolve, create=True):
            state = _make_state()
            payload = _invoke({"category": "BROWSER"}, state)

        self.assertTrue(payload["success"], payload)
        self.assertEqual(payload["category"], "browser",
                         "canonical-cased name should be returned in the response")
        self.assertEqual(payload["resolved_tools"], ["browser_navigate"])

    def test_whitespace_is_stripped(self):
        fake_available = ["browser"]
        def fake_resolve(name):
            return ["browser_navigate"] if name == "browser" else []

        with mock.patch.object(expand_tools, "_available_toolset_names",
                               return_value=fake_available), \
             mock.patch("toolsets.resolve_toolset", side_effect=fake_resolve, create=True):
            payload = _invoke({"category": "  browser  "}, _make_state())

        self.assertTrue(payload["success"], payload)
        self.assertEqual(payload["category"], "browser")


class InvalidCategoryErrorTests(unittest.TestCase):
    """Bad categories should not pretend success; the error should be helpful."""

    def test_missing_category_argument(self):
        payload = _invoke({}, _make_state())
        self.assertFalse(payload["success"])
        self.assertIn("required", payload["error"].lower())

    def test_unknown_category_lists_dynamic_available_set(self):
        fake_available = ["browser", "file", "terminal", "cronjob"]
        def fake_resolve(name):
            return []  # nothing resolves — true "unknown" path

        with mock.patch.object(expand_tools, "_available_toolset_names",
                               return_value=fake_available), \
             mock.patch("toolsets.resolve_toolset", side_effect=fake_resolve, create=True):
            payload = _invoke({"category": "no-such-thing"}, _make_state())

        self.assertFalse(payload["success"])
        err = payload["error"]
        self.assertIn("'no-such-thing'", err)
        # The dynamic set, not the old hardcoded list, should be referenced.
        for name in fake_available:
            self.assertIn(name, err, f"expected dynamic category {name!r} in error")

    def test_close_match_suggestion_when_typo(self):
        fake_available = ["browser", "cronjob", "terminal"]
        def fake_resolve(name):
            return []

        with mock.patch.object(expand_tools, "_available_toolset_names",
                               return_value=fake_available), \
             mock.patch("toolsets.resolve_toolset", side_effect=fake_resolve, create=True):
            payload = _invoke({"category": "browsr"}, _make_state())

        self.assertFalse(payload["success"])
        self.assertIn("Did you mean", payload["error"])
        self.assertIn("browser", payload["error"])

    def test_no_prediction_context_returns_graceful_error(self):
        cv: contextvars.ContextVar = contextvars.ContextVar("test_pred", default=None)
        # Intentionally do not set state.
        handler = expand_tools.make_handler(cv, sticky_refresh_fn=None)
        payload = json.loads(handler({"category": "browser"}))
        self.assertFalse(payload["success"])
        self.assertIn("no prediction context", payload["error"])


class ResponsePayloadSymmetryTests(unittest.TestCase):
    """The success response should expose the same resolved_tools the
    telemetry event records — source-side and result-side telemetry agree."""

    def test_resolved_tools_present_in_response(self):
        resolved = ["browser_navigate", "browser_click"]
        with mock.patch.object(expand_tools, "_resolve_category",
                               return_value=("browser", list(resolved), ["browser"])):
            payload = _invoke({"category": "browser"}, _make_state())

        self.assertIn("resolved_tools", payload)
        self.assertEqual(sorted(payload["resolved_tools"]), sorted(resolved))


if __name__ == "__main__":
    unittest.main()
