"""Regression coverage for cache_replay session grouping across `/new`.

The freeze simulation must group calls by the Hermes session id
(``hermes_session_id``), which rotates on ``/new``, and fall back to the legacy
transport ``session_id`` only when the Hermes id is absent. If it grouped by the
stable transport ``session_id`` alone, two Hermes sessions that share the same
underlying chat/transport id would be fused into one freeze cohort: the second
session would inherit the first's frozen tool-list hash (its "frozen active
set") and its accumulated trigger-mutation history, so a fresh `/new` session
would spuriously "break" the stale frozen prefix.

These tests import the real module functions (``_session_key``,
``stability_simulation``) rather than re-deriving their logic.
"""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = PLUGIN_DIR / "tests"
sys.path.insert(0, str(TESTS_DIR))
import conftest  # noqa: F401,E402 — registers tool_belt_plugin, puts plugin on sys.path

cache_replay = importlib.import_module("tool_belt_plugin.cache_replay")


def _pred(prediction_id, session_id, ts, *, hermes_session_id=None,
          scope="assistant-a:telegram", triggers=None):
    row = {
        "prediction_id": prediction_id,
        "session_id": session_id,
        "ts": ts,
        "scope": scope,
        "trigger_activated_tools": list(triggers or []),
    }
    if hermes_session_id is not None:
        row["hermes_session_id"] = hermes_session_id
    return row


def _call(prediction_id, session_id, ts, api_call_idx, tool_list_hash, *,
          hermes_session_id=None, scope="assistant-a:telegram", cache_read_tokens=1000):
    row = {
        "prediction_id": prediction_id,
        "session_id": session_id,
        "ts": ts,
        "api_call_idx": api_call_idx,
        "tool_list_hash": tool_list_hash,
        "cache_read_tokens": cache_read_tokens,
        "scope": scope,
    }
    if hermes_session_id is not None:
        row["hermes_session_id"] = hermes_session_id
    return row


class FreezeSimulationNewBoundaryTests(unittest.TestCase):
    def test_new_boundary_does_not_reuse_frozen_active_set(self):
        """Two Hermes sessions share the transport session_id but differ in
        hermes_session_id (a `/new` reset). The second must open a fresh freeze
        cohort: its first call is a first_call, not a break against the first
        session's frozen hash."""
        shared_transport = "chat-transport-shared"
        preds = [
            _pred("P1", shared_transport, ts=1, hermes_session_id="H1"),
            _pred("P2", shared_transport, ts=10, hermes_session_id="H2"),
        ]
        calls = [
            # Hermes session H1: frozen hash "A", second call matches.
            _call("P1", shared_transport, ts=1, api_call_idx=0,
                  tool_list_hash="A", hermes_session_id="H1"),
            _call("P1", shared_transport, ts=2, api_call_idx=1,
                  tool_list_hash="A", hermes_session_id="H1"),
            # Hermes session H2 (post-/new): different frozen hash "B".
            _call("P2", shared_transport, ts=10, api_call_idx=0,
                  tool_list_hash="B", hermes_session_id="H2"),
            _call("P2", shared_transport, ts=11, api_call_idx=1,
                  tool_list_hash="B", hermes_session_id="H2"),
        ]
        result = cache_replay.stability_simulation(preds, calls, tool_calls=[])
        # One freeze cohort per Hermes session, each with its own first call.
        self.assertEqual(result["sessions"], 2)
        self.assertEqual(result["first_calls_per_session"], 2)
        self.assertEqual(result["matches_stable"], 2)
        # The post-/new hash change must NOT read as an avoidable freeze break;
        # grouping by transport session_id alone would produce would_break==2.
        self.assertEqual(result["would_break_mutations"], 0)

    def test_new_boundary_does_not_reuse_trigger_mutation_history(self):
        """The same trigger tool activating in a post-/new session must count as
        a fresh trigger-driven mutation, not be swallowed as already-seen from
        the prior session (which would misclassify it as an avoidable break)."""
        shared_transport = "chat-transport-shared"
        preds = [
            _pred("P1", shared_transport, ts=1, hermes_session_id="H1",
                  triggers=["mnemosyne_recall"]),
            _pred("P2", shared_transport, ts=10, hermes_session_id="H2",
                  triggers=["mnemosyne_recall"]),
        ]
        calls = [
            # H1: first call freezes hash "A"; mutation to "B" is trigger-driven.
            _call("P1", shared_transport, ts=1, api_call_idx=0,
                  tool_list_hash="A", hermes_session_id="H1"),
            _call("P1", shared_transport, ts=2, api_call_idx=1,
                  tool_list_hash="B", hermes_session_id="H1"),
            # H2 (post-/new): first call freezes "C"; the same trigger re-fires
            # and mutates to "D" — a fresh trigger-driven mutation for H2.
            _call("P2", shared_transport, ts=10, api_call_idx=0,
                  tool_list_hash="C", hermes_session_id="H2"),
            _call("P2", shared_transport, ts=11, api_call_idx=1,
                  tool_list_hash="D", hermes_session_id="H2"),
        ]
        result = cache_replay.stability_simulation(preds, calls, tool_calls=[])
        self.assertEqual(result["sessions"], 2)
        self.assertEqual(result["first_calls_per_session"], 2)
        # Both mutations are trigger-driven; none is an avoidable break. Grouping
        # by transport session_id would carry H1's seen trigger tool into H2,
        # demoting H2's mutation to would_break.
        self.assertEqual(result["trigger_driven_mutations"], 2)
        self.assertEqual(result["would_break_mutations"], 0)

    def test_session_id_fallback_groups_turns_without_hermes_id(self):
        """When rows carry no hermes_session_id, _session_key falls back to
        session_id, so turns sharing that id land in one freeze cohort."""
        sid = "legacy-chat"
        preds = [_pred("P1", sid, ts=1)]  # no hermes_session_id
        calls = [
            _call("P1", sid, ts=1, api_call_idx=0, tool_list_hash="A"),
            _call("P1", sid, ts=2, api_call_idx=1, tool_list_hash="A"),
            _call("P1", sid, ts=3, api_call_idx=2, tool_list_hash="A"),
        ]
        result = cache_replay.stability_simulation(preds, calls, tool_calls=[])
        self.assertEqual(result["sessions"], 1)
        self.assertEqual(result["first_calls_per_session"], 1)
        self.assertEqual(result["matches_stable"], 2)
        self.assertEqual(result["would_break_mutations"], 0)


if __name__ == "__main__":
    unittest.main()
