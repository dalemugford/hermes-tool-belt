"""In-process periodic auto-shaping — the automatic-application promise.

Dale's product order: "shaping should be applied automatically, periodically,
without needing system-level scheduled tasks, when not in observe mode."
These tests lock that behavior:

  · an apply-mode scope past the evidence threshold and past the debounce
    interval gets a real ``learned.json`` write (``source: "auto"``) from the
    session-end path — THE promise test;
  · recommend/observe scopes are never auto-written;
  · the per-scope debounce holds;
  · a below-threshold apply scope records the attempt and writes nothing else;
  · an engine exception never propagates out of the session-end hook;
  · ``auto_shape: false`` disables the whole path;
  · a live frozen session is never mutated by an auto-apply.

Everything runs against a throwaway ``HERMES_HOME``; no live state is touched.
"""

from __future__ import annotations

import copy
import importlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import conftest  # noqa: F401,E402 — registers the tool_belt_plugin package

plugin = sys.modules["tool_belt_plugin"]

SCOPE = "agent-a:telegram"


def _shaping_mod():
    """The shared shaping core — imported lazily so this module still loads
    (and its tests FAIL rather than error the whole collection) on a tree
    that predates the extraction."""
    return importlib.import_module("tool_belt_plugin.shaping")


def seed_state_dir(state_dir: Path, *, scope: str = SCOPE, sessions: int = 3,
                   policy_source: str = "preset") -> None:
    """Telemetry yielding exactly one promote candidate (``grep_files``)."""
    state_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    preds, calls = [], []
    for i in range(sessions):
        pid = f"pred-{i}"
        preds.append({
            "policy_source": policy_source,
            "ts": now + i,
            "schema_version": 2,
            "scope": scope,
            "session_id": "key",
            "hermes_session_id": f"sess-{i}",
            "prediction_id": pid,
            "always_carry_tools": ["read_file"],
            "carry_tools": ["read_file"],
            "expand_only_tools": ["grep_files"],
            "active_tools": ["read_file"],
            "ceiling_tools": ["read_file", "grep_files"],
        })
        calls.append({
            "ts": now + i,
            "schema_version": 2,
            "prediction_id": pid,
            "tool_name": "grep_files",
            "activated_by_expansion": True,
        })
    (state_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8")
    (state_dir / "tool_calls.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in calls), encoding="utf-8")


class AutoShapeBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.state_dir = self.home / "state" / "tool-belt"
        env = mock.patch.dict(os.environ, {"HERMES_HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)
        # Isolated plugin config: enabled, apply-mode scope, auto_shape on.
        self.config = {
            "enabled": True,
            "log": False,
            "auto_shape": True,
            "learned_mode": "recommend",
            "channels": {SCOPE: {"learned_mode": "apply"}},
        }
        cfg = mock.patch.dict(plugin._CONFIG, self.config)
        cfg.start()
        self.addCleanup(cfg.stop)
        # Reset the in-process throttle so each test's first pass runs.
        self._reset_throttle()

    def _reset_throttle(self):
        auto = getattr(plugin, "_AUTO_SHAPE", None)
        if isinstance(auto, dict):
            auto["not_before"] = 0.0

    def learned_doc(self):
        path = self.state_dir / "learned.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


class SessionEndAutoApplyTests(AutoShapeBase):
    """The promise: session end auto-applies shaping for apply-mode scopes."""

    def test_apply_scope_past_threshold_is_shaped_on_session_end(self):
        seed_state_dir(self.state_dir)
        plugin._on_session_end(session_id="sess-x")
        doc = self.learned_doc()
        self.assertIsNotNone(doc, "session end must have written learned.json")
        entry = doc["scopes"][SCOPE]
        self.assertIn("grep_files", entry["carry"])
        shaping = entry["shaping"]
        self.assertEqual(shaping["source"], "auto")
        self.assertTrue(shaping["applied_at"])
        self.assertTrue(shaping["last_auto_shape_at"])

    def test_recommend_scope_is_never_auto_written(self):
        seed_state_dir(self.state_dir)
        plugin._CONFIG["channels"] = {SCOPE: {"learned_mode": "recommend"}}
        plugin._on_session_end(session_id="sess-x")
        self.assertIsNone(self.learned_doc())

    def test_default_mode_scope_is_never_auto_written(self):
        # No channel entry at all: global default learned_mode "recommend".
        seed_state_dir(self.state_dir)
        plugin._CONFIG["channels"] = {}
        plugin._on_session_end(session_id="sess-x")
        self.assertIsNone(self.learned_doc())

    def test_auto_shape_false_disables_the_path(self):
        seed_state_dir(self.state_dir)
        plugin._CONFIG["auto_shape"] = False
        plugin._on_session_end(session_id="sess-x")
        self.assertIsNone(self.learned_doc())

    def test_engine_exception_never_escapes_session_end(self):
        seed_state_dir(self.state_dir)
        shaping = _shaping_mod()
        with mock.patch.object(
            shaping, "auto_shape_run", side_effect=RuntimeError("boom")
        ):
            # Must not raise — the gateway path is unaffected.
            self.assertIsNone(plugin._on_session_end(session_id="sess-x"))
        # And the lock must have been released: a later pass still works.
        self._reset_throttle()
        plugin._on_session_end(session_id="sess-x")
        self.assertIsNotNone(self.learned_doc())

    def test_second_session_end_within_interval_is_debounced(self):
        seed_state_dir(self.state_dir)
        plugin._on_session_end(session_id="sess-x")
        first = (self.state_dir / "learned.json").read_text(encoding="utf-8")
        # New evidence lands, throttle cleared — but the per-scope 24h
        # debounce (persisted last_auto_shape_at) must still hold.
        seed_state_dir(self.state_dir, sessions=5)
        self._reset_throttle()
        plugin._on_session_end(session_id="sess-y")
        second = (self.state_dir / "learned.json").read_text(encoding="utf-8")
        self.assertEqual(first, second)

    def test_live_session_posture_is_never_mutated(self):
        # The session-end auto-shape pass rewrites learned.json; it must not
        # touch a live session's posture pin (D3, 2026-09-02 — replaces the
        # removed frozen-snapshot state this guarded).
        seed_state_dir(self.state_dir)
        pin = {"mode": "on", "provider": "openai-codex"}
        plugin._CACHE_DECISION_BY_SESSION["live-key"] = pin
        self.addCleanup(plugin._CACHE_DECISION_BY_SESSION.pop, "live-key", None)
        snapshot = copy.deepcopy(pin)
        plugin._on_session_end(session_id="sess-x")
        self.assertIsNotNone(self.learned_doc())  # the apply really happened
        self.assertIs(plugin._CACHE_DECISION_BY_SESSION["live-key"], pin)
        self.assertEqual(plugin._CACHE_DECISION_BY_SESSION["live-key"], snapshot)


class CarryAllTrafficTests(AutoShapeBase):
    """A scope whose whole window ran carry-all has nothing to learn from:
    the pass reports it and leaves it UNSTAMPED, so it re-qualifies the
    moment a session lands on an uncached route."""

    def test_carry_all_only_scope_is_skipped_and_left_unstamped(self):
        seed_state_dir(self.state_dir, policy_source="cache_on_carry_all")
        shaping = _shaping_mod()
        summary = shaping.auto_shape_run(self.config, self.state_dir)
        self.assertEqual(summary.get("skipped_no_narrowed_sessions"), [SCOPE])
        self.assertIsNone(self.learned_doc(),
                          "no evidence — nothing may be written")
        stamps_path = self.state_dir / shaping.AUTO_SHAPE_STAMP_FILE
        stamps = (json.loads(stamps_path.read_text(encoding="utf-8"))
                  if stamps_path.exists() else {})
        self.assertNotIn(SCOPE, stamps,
                         "an unshaped scope must not start its debounce clock")

    def test_narrowed_traffic_arriving_later_shapes_at_once(self):
        # Same scope, now with narrowed sessions: no stale debounce stamp
        # stands in the way.
        seed_state_dir(self.state_dir, policy_source="cache_on_carry_all")
        shaping = _shaping_mod()
        shaping.auto_shape_run(self.config, self.state_dir)
        seed_state_dir(self.state_dir)
        summary = shaping.auto_shape_run(self.config, self.state_dir)
        self.assertEqual(summary.get("skipped_no_narrowed_sessions"), [])
        doc = self.learned_doc()
        self.assertIsNotNone(doc)
        self.assertIn("grep_files", doc["scopes"][SCOPE]["carry"])


class FutureSchemaGuardTests(AutoShapeBase):
    """load_state refuses a learned.json newer than this build; the WRITERS
    must refuse symmetrically or a future-schema file would be re-stamped v2
    on the next pass, destroying its version marker."""

    def test_auto_pass_never_rewrites_a_future_schema_doc(self):
        seed_state_dir(self.state_dir)
        path = self.state_dir / "learned.json"
        original = json.dumps({"version": 3, "scopes": {},
                               "from_the_future": True})
        path.write_text(original, encoding="utf-8")
        shaping = _shaping_mod()
        result = shaping.auto_shape_run(self.config, self.state_dir)
        self.assertEqual(result.get("reason"), "learned_schema_too_new")
        self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_configure_merge_refuses_a_future_schema_doc(self):
        path = self.state_dir / "learned.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        original = json.dumps({"version": 3, "scopes": {}})
        path.write_text(original, encoding="utf-8")
        shaping = _shaping_mod()
        _state, changed = shaping.merge_into_learned(
            self.state_dir,
            {SCOPE: {"promote": [{"tool": "read_file"}], "demote": []}},
            dry_run=False)
        self.assertFalse(changed)
        self.assertEqual(path.read_text(encoding="utf-8"), original)


class EngineDebounceTests(AutoShapeBase):
    """Engine-level clock control for the debounce and attempt-recording."""

    def test_debounce_blocks_then_expires(self):
        seed_state_dir(self.state_dir)
        shaping = _shaping_mod()
        t0 = time.time()
        first = shaping.auto_shape_run(self.config, self.state_dir, now=t0)
        self.assertTrue(first["ran"])
        self.assertIn(SCOPE, first["applied"])
        # Within the interval: not even attempted.
        again = shaping.auto_shape_run(self.config, self.state_dir, now=t0 + 3600)
        self.assertFalse(again["ran"])
        self.assertEqual(again["reason"], "no_eligible_scopes")
        # Past 24h: attempted again (no structural change this time).
        later = shaping.auto_shape_run(
            self.config, self.state_dir, now=t0 + 25 * 3600
        )
        self.assertTrue(later["ran"])
        self.assertIn(SCOPE, later["attempted"])
        self.assertEqual(later["applied"], {})

    def test_interval_override_is_honored(self):
        seed_state_dir(self.state_dir)
        shaping = _shaping_mod()
        cfg = dict(self.config)
        cfg["channels"] = {
            SCOPE: {"learned_mode": "apply", "auto_shape_interval_hours": 1}
        }
        t0 = time.time()
        shaping.auto_shape_run(cfg, self.state_dir, now=t0)
        later = shaping.auto_shape_run(cfg, self.state_dir, now=t0 + 3700)
        self.assertTrue(later["ran"])
        self.assertIn(SCOPE, later["attempted"])

    def test_below_threshold_leaves_learned_untouched_but_still_debounces(self):
        # One session: below promote_min_sessions (2) — no recommendation.
        # A nothing-changed pass must NOT rewrite learned.json (rewriting
        # made the cross-process last-writer-wins window a routine event);
        # the attempt lands in the debounce sidecar and still gates the
        # next run.
        seed_state_dir(self.state_dir, sessions=1)
        shaping = _shaping_mod()
        result = shaping.auto_shape_run(self.config, self.state_dir)
        self.assertTrue(result["ran"])
        self.assertEqual(result["applied"], {})
        self.assertIsNone(self.learned_doc(),
                          "a quiet pass writes no learned.json")
        stamps = shaping._read_auto_shape_stamps(self.state_dir)
        self.assertIn(SCOPE, stamps)
        # The sidecar stamp is exactly what debounces the next run.
        again = shaping.auto_shape_run(self.config, self.state_dir)
        self.assertFalse(again["ran"])


if __name__ == "__main__":
    unittest.main()
