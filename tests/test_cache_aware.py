"""Unit tests for the cache-aware refactor (Phases 0-5 of doc 16).

Companion to scripts/smoke-test.py, which covers integration-level
behavior. These are focused unit tests for the individual helpers
that make the freeze model work, plus regression guards for bugs
that have already been fixed once:

  · _resolve_cache_mode_for_session — forced modes + auto with the
    cross-session detection cache hit/miss
  · _freeze_session_snapshot / _build_state_from_frozen — snapshot
    shape; reuse counter increments; expansions carry forward
  · _update_cache_mode_detection — forced lock, provider blocklist,
    threshold met/failed, divergence detection
  · _persist_detection_lock / _cached_detection_mode — cross-session
    persistence; provider-blocklist locks not persisted
  · on_session_end DOES NOT evict freeze (Hermes hook semantics —
    fires per-turn, the bug we hit during Phase 1 verification)
  · on_session_reset DOES evict freeze
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Reuse conftest's plugin-loader bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401

plugin = sys.modules["dynamic_tools_plugin"]


def _seed_plugin_config(**overrides) -> None:
    plugin._CONFIG.clear()
    plugin._CONFIG.update({
        "enabled": True,
        "log": False,  # no JSONL writes from unit tests
        "agent": "bernard",
        "bypass_rate": 0.0,
        "cache_mode": "auto",
        "channels": {},
        "cache_off": {
            "sticky": {"enabled": True, "ttl_turns": 3, "categories": ["*"]},
            "predictor": {"lookback_turns": 0},
        },
        "cache_auto": {
            "detect_calls": 5,
            "detect_min_input_tokens": 5000,
            "on_threshold": 0.40,
            "providers_off_models": ["kimi-k2.6:cloud", "gpt-5.4-mini"],
        },
    })


def _reset_plugin_state() -> None:
    plugin._STICKY_BY_KEY.clear()
    plugin._POLICY_TURN_BY_SCOPE.clear()
    plugin._PRIOR_MESSAGES_BY_SESSION.clear()
    plugin._FROZEN_BY_SESSION.clear()
    plugin._CACHE_MODE_BY_SESSION.clear()
    plugin._DETECTION_CACHE.clear()
    plugin._DETECTION_CACHE_LOADED = False


class ResolveCacheModeTests(unittest.TestCase):
    """_resolve_cache_mode_for_session: dispatch-time decision."""

    def setUp(self):
        _seed_plugin_config()
        _reset_plugin_state()

    def test_forced_off_returns_off(self):
        plugin._CONFIG["cache_mode"] = "off"
        self.assertEqual(plugin._resolve_cache_mode_for_session("sid", scope="bernard:telegram"), "off")

    def test_forced_on_returns_on(self):
        plugin._CONFIG["cache_mode"] = "on"
        self.assertEqual(plugin._resolve_cache_mode_for_session("sid", scope="bernard:telegram"), "on")

    def test_auto_defaults_on_when_no_cached_entry(self):
        # Safe-default per the pivot doc — assume caching is on unless
        # explicitly known otherwise.
        plugin._CONFIG["cache_mode"] = "auto"
        self.assertEqual(plugin._resolve_cache_mode_for_session("sid", scope="bernard:telegram"), "on")

    def test_auto_honors_cached_off(self):
        plugin._CONFIG["cache_mode"] = "auto"
        plugin._DETECTION_CACHE["bernard:telegram"] = {"mode": "off"}
        self.assertEqual(plugin._resolve_cache_mode_for_session("sid", scope="bernard:telegram"), "off")

    def test_auto_honors_cached_on(self):
        plugin._CONFIG["cache_mode"] = "auto"
        plugin._DETECTION_CACHE["bernard:telegram"] = {"mode": "on"}
        self.assertEqual(plugin._resolve_cache_mode_for_session("sid", scope="bernard:telegram"), "on")


class FreezeSnapshotTests(unittest.TestCase):
    """_freeze_session_snapshot + _build_state_from_frozen: snapshot shape."""

    def setUp(self):
        _seed_plugin_config()
        _reset_plugin_state()

    def test_snapshot_starts_with_zero_reuses(self):
        plugin._freeze_session_snapshot(
            "sid",
            allowed_tool_names=["read_file"],
            baseline_allowed_tools=["read_file"],
            known_tool_names={"read_file", "write_file"},
            preset_name="aggressive",
            always_on_count=1,
            always_on_tools=["read_file"],
            trigger_tools_by_group={},
            triggers_fired=[],
            triggers_suppressed=[],
            policy_source="preset",
            policy_version="v1",
            learned_mode="off",
            learned_scope="",
            learned_changes=[],
        )
        self.assertEqual(plugin._FROZEN_BY_SESSION["sid"]["reuses"], 0)
        self.assertEqual(plugin._FROZEN_BY_SESSION["sid"]["expansions"], set())

    def test_build_from_frozen_increments_reuses(self):
        plugin._freeze_session_snapshot(
            "sid",
            allowed_tool_names=["read_file"],
            baseline_allowed_tools=["read_file"],
            known_tool_names={"read_file"},
            preset_name="aggressive",
            always_on_count=1,
            always_on_tools=["read_file"],
            trigger_tools_by_group={},
            triggers_fired=[],
            triggers_suppressed=[],
            policy_source="preset",
            policy_version="v1",
            learned_mode="off",
            learned_scope="",
            learned_changes=[],
        )
        frozen = plugin._FROZEN_BY_SESSION["sid"]

        state1 = plugin._build_state_from_frozen(
            frozen, session_id="sid", scope="bernard:telegram",
            channel="bernard:telegram", agent="bernard", platform="telegram",
            message="first reuse",
        )
        self.assertTrue(state1["frozen_reuse"])
        self.assertEqual(state1["frozen_reuse_count"], 1)

        state2 = plugin._build_state_from_frozen(
            frozen, session_id="sid", scope="bernard:telegram",
            channel="bernard:telegram", agent="bernard", platform="telegram",
            message="second reuse",
        )
        self.assertEqual(state2["frozen_reuse_count"], 2)

    def test_build_from_frozen_carries_expansions(self):
        plugin._freeze_session_snapshot(
            "sid",
            allowed_tool_names=["read_file"],
            baseline_allowed_tools=["read_file"],
            known_tool_names={"read_file"},
            preset_name="aggressive",
            always_on_count=1,
            always_on_tools=["read_file"],
            trigger_tools_by_group={},
            triggers_fired=[],
            triggers_suppressed=[],
            policy_source="preset",
            policy_version="v1",
            learned_mode="off",
            learned_scope="",
            learned_changes=[],
        )
        plugin._FROZEN_BY_SESSION["sid"]["expansions"] = {"browser_navigate", "browser_click"}

        state = plugin._build_state_from_frozen(
            plugin._FROZEN_BY_SESSION["sid"],
            session_id="sid", scope="bernard:telegram",
            channel="bernard:telegram", agent="bernard", platform="telegram",
            message="hi",
        )
        self.assertEqual(state["expansions"], {"browser_navigate", "browser_click"})

    def test_build_from_frozen_disables_sticky_and_lookback(self):
        plugin._freeze_session_snapshot(
            "sid",
            allowed_tool_names=["read_file"],
            baseline_allowed_tools=["read_file"],
            known_tool_names={"read_file"},
            preset_name="aggressive",
            always_on_count=1,
            always_on_tools=["read_file"],
            trigger_tools_by_group={},
            triggers_fired=[],
            triggers_suppressed=[],
            policy_source="preset",
            policy_version="v1",
            learned_mode="off",
            learned_scope="",
            learned_changes=[],
        )
        state = plugin._build_state_from_frozen(
            plugin._FROZEN_BY_SESSION["sid"],
            session_id="sid", scope="bernard:telegram",
            channel="bernard:telegram", agent="bernard", platform="telegram",
            message="hi",
        )
        self.assertEqual(state["sticky_key"], "")
        self.assertEqual(state["sticky_tools"], [])
        self.assertEqual(state["lookback_used"], 0)
        self.assertEqual(state["lookback_turns_config"], 0)


class DetectionStateMachineTests(unittest.TestCase):
    """_update_cache_mode_detection: pending → locked transitions."""

    def setUp(self):
        _seed_plugin_config()
        _reset_plugin_state()

    def test_forced_on_locks_immediately(self):
        plugin._CONFIG["cache_mode"] = "on"
        mode = plugin._update_cache_mode_detection(
            session_key="sid", model="claude-sonnet-4-6",
            cache_read=0, cache_write=0, input_tokens=100,
            scope="bernard:telegram",
        )
        self.assertEqual(mode, "on")

    def test_provider_blocklist_locks_off_on_first_call(self):
        mode = plugin._update_cache_mode_detection(
            session_key="sid", model="kimi-k2.6:cloud",
            cache_read=0, cache_write=0, input_tokens=100,
            scope="bernard:telegram",
        )
        self.assertEqual(mode, "off")
        self.assertEqual(
            plugin._CACHE_MODE_BY_SESSION["sid"]["lock_reason"],
            "provider_blocklist",
        )

    def test_pending_until_call_and_token_thresholds_met(self):
        # 4 calls with tiny inputs → still pending (needs 5 calls + 5K)
        for _ in range(4):
            mode = plugin._update_cache_mode_detection(
                session_key="sid", model="gpt-5.4",
                cache_read=100, cache_write=0, input_tokens=100,
                scope="bernard:telegram",
            )
            self.assertEqual(mode, "pending")

    def test_threshold_met_locks_on(self):
        # 5 calls × (cache=8000, fresh=2000) → hit_rate=0.80 > 0.40 → "on"
        for _ in range(5):
            mode = plugin._update_cache_mode_detection(
                session_key="sid", model="gpt-5.4",
                cache_read=8000, cache_write=0, input_tokens=2000,
                scope="bernard:telegram",
            )
        self.assertEqual(mode, "on")
        self.assertEqual(
            plugin._CACHE_MODE_BY_SESSION["sid"]["lock_reason"],
            "threshold_met",
        )

    def test_threshold_failed_locks_off(self):
        # 5 calls × (cache=100, fresh=10000) → hit_rate≈0.01 < 0.40 → "off"
        for _ in range(5):
            mode = plugin._update_cache_mode_detection(
                session_key="sid", model="gpt-5.4",
                cache_read=100, cache_write=0, input_tokens=10000,
                scope="bernard:telegram",
            )
        self.assertEqual(mode, "off")
        self.assertEqual(
            plugin._CACHE_MODE_BY_SESSION["sid"]["lock_reason"],
            "threshold_failed",
        )


class DetectionCachePersistenceTests(unittest.TestCase):
    """_persist_detection_lock / _cached_detection_mode: cross-session memory."""

    def setUp(self):
        _seed_plugin_config()
        _reset_plugin_state()

    def test_persists_threshold_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"HERMES_HOME": tmp}, clear=False):
                state = {
                    "mode": "on",
                    "lock_reason": "threshold_met",
                    "hit_rate_at_lock": 0.85,
                }
                plugin._persist_detection_lock("bernard:telegram", state, "claude-sonnet-4-6")
                self.assertIn("bernard:telegram", plugin._DETECTION_CACHE)
                self.assertEqual(plugin._DETECTION_CACHE["bernard:telegram"]["mode"], "on")
                self.assertEqual(plugin._DETECTION_CACHE["bernard:telegram"]["sessions_locked"], 1)

    def test_does_not_persist_provider_blocklist_lock(self):
        # Blocklist is config, not observation — shouldn't pollute the cache.
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"HERMES_HOME": tmp}, clear=False):
                state = {
                    "mode": "off",
                    "lock_reason": "provider_blocklist",
                }
                plugin._persist_detection_lock("bernard:telegram", state, "kimi-k2.6:cloud")
                self.assertNotIn("bernard:telegram", plugin._DETECTION_CACHE)

    def test_consecutive_same_mode_locks_increment_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"HERMES_HOME": tmp}, clear=False):
                state = {"mode": "on", "lock_reason": "threshold_met"}
                plugin._persist_detection_lock("bernard:telegram", state, "model-a")
                plugin._persist_detection_lock("bernard:telegram", state, "model-a")
                plugin._persist_detection_lock("bernard:telegram", state, "model-a")
                self.assertEqual(
                    plugin._DETECTION_CACHE["bernard:telegram"]["sessions_locked"], 3,
                )

    def test_disagreeing_lock_resets_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"HERMES_HOME": tmp}, clear=False):
                plugin._persist_detection_lock(
                    "bernard:telegram",
                    {"mode": "on", "lock_reason": "threshold_met"},
                    "model-a",
                )
                plugin._persist_detection_lock(
                    "bernard:telegram",
                    {"mode": "off", "lock_reason": "threshold_failed"},
                    "model-a",
                )
                self.assertEqual(plugin._DETECTION_CACHE["bernard:telegram"]["mode"], "off")
                self.assertEqual(
                    plugin._DETECTION_CACHE["bernard:telegram"]["sessions_locked"], 1,
                )


class SessionHookSemanticsTests(unittest.TestCase):
    """Regression guards for the on_session_end vs on_session_reset bug.

    During Phase 1 verification we discovered Hermes fires on_session_end
    at the end of every run_conversation call — once per user message in
    multi-turn sessions, NOT at actual session end. Evicting the freeze
    there nukes it between turns and defeats the freeze.
    """

    def setUp(self):
        _seed_plugin_config()
        _reset_plugin_state()
        # Seed a frozen snapshot.
        plugin._freeze_session_snapshot(
            "sid",
            allowed_tool_names=["read_file"],
            baseline_allowed_tools=["read_file"],
            known_tool_names={"read_file"},
            preset_name="aggressive",
            always_on_count=1,
            always_on_tools=["read_file"],
            trigger_tools_by_group={},
            triggers_fired=[],
            triggers_suppressed=[],
            policy_source="preset",
            policy_version="v1",
            learned_mode="off",
            learned_scope="",
            learned_changes=[],
        )
        plugin._CACHE_MODE_BY_SESSION["sid"] = {"mode": "on", "calls_observed": 3}

    def test_on_session_end_does_not_evict_freeze(self):
        with mock.patch.dict(os.environ, {"HERMES_SESSION_KEY": "sid"}, clear=False):
            plugin._on_session_end(session_id="sid", session_key="sid")
        self.assertIn("sid", plugin._FROZEN_BY_SESSION)
        self.assertIn("sid", plugin._CACHE_MODE_BY_SESSION)

    def test_on_session_end_evicts_sticky_and_lookback(self):
        # Sticky is keyed by a hash derived from session_id, not by sid
        # directly. Populate the canonical sticky key the handler will look up.
        sticky_key = plugin._sticky_key_for_session("sid")
        self.assertTrue(sticky_key, "sanity: derivation produces non-empty key")
        plugin._STICKY_BY_KEY[sticky_key] = {"terminal": {"remaining_turns": 2}}
        plugin._PRIOR_MESSAGES_BY_SESSION["sid"] = ["msg1"]
        with mock.patch.dict(os.environ, {"HERMES_SESSION_KEY": "sid"}, clear=False):
            plugin._on_session_end(session_id="sid", session_key="sid")
        self.assertNotIn("sid", plugin._PRIOR_MESSAGES_BY_SESSION)
        self.assertNotIn(sticky_key, plugin._STICKY_BY_KEY)

    def test_on_session_reset_evicts_freeze(self):
        plugin._on_session_reset(session_id="sid", session_key="sid")
        self.assertNotIn("sid", plugin._FROZEN_BY_SESSION)
        self.assertNotIn("sid", plugin._CACHE_MODE_BY_SESSION)


if __name__ == "__main__":
    unittest.main()
