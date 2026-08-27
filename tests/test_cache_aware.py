"""Unit tests for cache-aware Tool Belt behavior.

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
  · on_session_end DOES NOT evict freeze because Hermes fires it per turn
  · on_session_reset DOES evict freeze
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Reuse conftest's plugin-loader bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401

plugin = sys.modules["tool_belt_plugin"]


def _seed_plugin_config(**overrides) -> None:
    plugin._CONFIG.clear()
    plugin._CONFIG.update({
        "enabled": True,
        "log": False,  # no JSONL writes from unit tests
        "agent": "assistant-a",
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
    plugin._LAST_CANONICAL_BY_PLATFORM.clear()
    plugin._DETECTION_CACHE.clear()
    plugin._DETECTION_CACHE_LOADED = False


class ResolveCacheModeTests(unittest.TestCase):
    """_resolve_cache_mode_for_session: dispatch-time decision."""

    def setUp(self):
        _seed_plugin_config()
        _reset_plugin_state()

    def test_forced_off_returns_off(self):
        plugin._CONFIG["cache_mode"] = "off"
        self.assertEqual(plugin._resolve_cache_mode_for_session("sid", scope="assistant-a:telegram"), "off")

    def test_forced_on_returns_on(self):
        plugin._CONFIG["cache_mode"] = "on"
        self.assertEqual(plugin._resolve_cache_mode_for_session("sid", scope="assistant-a:telegram"), "on")

    def test_auto_defaults_on_when_no_cached_entry(self):
        # Protect prefix stability unless telemetry has established that
        # provider-side caching is unavailable.
        plugin._CONFIG["cache_mode"] = "auto"
        self.assertEqual(plugin._resolve_cache_mode_for_session("sid", scope="assistant-a:telegram"), "on")

    def test_auto_honors_cached_off(self):
        plugin._CONFIG["cache_mode"] = "auto"
        plugin._DETECTION_CACHE["assistant-a:telegram"] = {"mode": "off"}
        self.assertEqual(plugin._resolve_cache_mode_for_session("sid", scope="assistant-a:telegram"), "off")

    def test_auto_honors_cached_on(self):
        plugin._CONFIG["cache_mode"] = "auto"
        plugin._DETECTION_CACHE["assistant-a:telegram"] = {"mode": "on"}
        self.assertEqual(plugin._resolve_cache_mode_for_session("sid", scope="assistant-a:telegram"), "on")


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
            frozen, session_id="sid", scope="assistant-a:telegram",
            channel="assistant-a:telegram", agent="assistant-a", platform="telegram",
            message="first reuse",
        )
        self.assertTrue(state1["frozen_reuse"])
        self.assertEqual(state1["frozen_reuse_count"], 1)

        state2 = plugin._build_state_from_frozen(
            frozen, session_id="sid", scope="assistant-a:telegram",
            channel="assistant-a:telegram", agent="assistant-a", platform="telegram",
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
            session_id="sid", scope="assistant-a:telegram",
            channel="assistant-a:telegram", agent="assistant-a", platform="telegram",
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
            session_id="sid", scope="assistant-a:telegram",
            channel="assistant-a:telegram", agent="assistant-a", platform="telegram",
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
            scope="assistant-a:telegram",
        )
        self.assertEqual(mode, "on")

    def test_provider_blocklist_locks_off_on_first_call(self):
        mode = plugin._update_cache_mode_detection(
            session_key="sid", model="kimi-k2.6:cloud",
            cache_read=0, cache_write=0, input_tokens=100,
            scope="assistant-a:telegram",
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
                scope="assistant-a:telegram",
            )
            self.assertEqual(mode, "pending")

    def test_threshold_met_locks_on(self):
        # 5 calls × (cache=8000, fresh=2000) → hit_rate=0.80 > 0.40 → "on"
        for _ in range(5):
            mode = plugin._update_cache_mode_detection(
                session_key="sid", model="gpt-5.4",
                cache_read=8000, cache_write=0, input_tokens=2000,
                scope="assistant-a:telegram",
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
                scope="assistant-a:telegram",
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
                plugin._persist_detection_lock("assistant-a:telegram", state, "claude-sonnet-4-6")
                self.assertIn("assistant-a:telegram", plugin._DETECTION_CACHE)
                self.assertEqual(plugin._DETECTION_CACHE["assistant-a:telegram"]["mode"], "on")
                self.assertEqual(plugin._DETECTION_CACHE["assistant-a:telegram"]["sessions_locked"], 1)

    def test_does_not_persist_provider_blocklist_lock(self):
        # Blocklist is config, not observation — shouldn't pollute the cache.
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"HERMES_HOME": tmp}, clear=False):
                state = {
                    "mode": "off",
                    "lock_reason": "provider_blocklist",
                }
                plugin._persist_detection_lock("assistant-a:telegram", state, "kimi-k2.6:cloud")
                self.assertNotIn("assistant-a:telegram", plugin._DETECTION_CACHE)

    def test_consecutive_same_mode_locks_increment_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"HERMES_HOME": tmp}, clear=False):
                state = {"mode": "on", "lock_reason": "threshold_met"}
                plugin._persist_detection_lock("assistant-a:telegram", state, "model-a")
                plugin._persist_detection_lock("assistant-a:telegram", state, "model-a")
                plugin._persist_detection_lock("assistant-a:telegram", state, "model-a")
                self.assertEqual(
                    plugin._DETECTION_CACHE["assistant-a:telegram"]["sessions_locked"], 3,
                )

    def test_disagreeing_lock_resets_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"HERMES_HOME": tmp}, clear=False):
                plugin._persist_detection_lock(
                    "assistant-a:telegram",
                    {"mode": "on", "lock_reason": "threshold_met"},
                    "model-a",
                )
                plugin._persist_detection_lock(
                    "assistant-a:telegram",
                    {"mode": "off", "lock_reason": "threshold_failed"},
                    "model-a",
                )
                self.assertEqual(plugin._DETECTION_CACHE["assistant-a:telegram"]["mode"], "off")
                self.assertEqual(
                    plugin._DETECTION_CACHE["assistant-a:telegram"]["sessions_locked"], 1,
                )


class SessionHookSemanticsTests(unittest.TestCase):
    """Verify Tool Belt follows Hermes' session-hook contracts.

    Hermes fires on_session_end once per user message. Frozen state therefore
    survives that hook and is evicted by on_session_reset instead.
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


class SlashCommandBypassTests(unittest.TestCase):
    """Regression guards for the /new freeze-pollution bug.

    Hermes' gateway fires pre_gateway_dispatch BEFORE routing slash
    commands to their handlers. Without the bypass, the predictor runs
    on the command text and creates a freeze that's never reached by
    an LLM call but pollutes the next real-message dispatch with
    frozen_reuse=true. Then on_session_reset (which Hermes fires with
    only the NEW session UUID + platform) can't evict the freeze
    because the canonical key isn't in the hook kwargs.

    The fix is two-part:
      1. Track the canonical session_key per platform during dispatch
      2. Skip the build pipeline for slash-command messages
      3. on_session_reset falls back to the platform-tracked canonical
    """

    PLATFORM = "telegram"
    CANONICAL = "agent:main:telegram:dm:100000001"

    def setUp(self):
        _seed_plugin_config()
        _reset_plugin_state()

    def _make_event(self, text: str):
        from types import SimpleNamespace
        # Use SimpleNamespace so undefined attrs return None — MagicMock
        # would auto-create infinite-recursion-ready stubs that confuse
        # the platform extractor.
        platform_obj = SimpleNamespace(value=self.PLATFORM)
        source = SimpleNamespace(platform=platform_obj, chat_id="100000001", user_id="user-a")
        return SimpleNamespace(text=text, message=text, source=source, platform=None)

    def _make_session_store(self):
        store = mock.MagicMock()
        store._generate_session_key.return_value = self.CANONICAL
        return store

    def test_pre_gateway_dispatch_skips_slash_command(self):
        """A /new message must not build a freeze."""
        event = self._make_event("/new")
        store = self._make_session_store()
        plugin._on_pre_gateway_dispatch(event=event, session_store=store)
        self.assertNotIn(self.CANONICAL, plugin._FROZEN_BY_SESSION,
                         "freeze should not be created for slash-command messages")

    def test_pre_gateway_dispatch_records_canonical_for_slash_command(self):
        """Even though /new is bypassed, the canonical key must be tracked
        so the subsequent on_session_reset can find it."""
        event = self._make_event("/new")
        store = self._make_session_store()
        plugin._on_pre_gateway_dispatch(event=event, session_store=store)
        self.assertEqual(
            plugin._LAST_CANONICAL_BY_PLATFORM.get(self.PLATFORM),
            self.CANONICAL,
            "canonical key should be recorded even on bypassed slash commands"
        )

    def test_on_session_reset_evicts_via_platform_fallback(self):
        """Hermes fires on_session_reset with only session_id (new UUID)
        and platform — never the canonical key. Eviction must succeed by
        falling back to _LAST_CANONICAL_BY_PLATFORM."""
        # Seed: a freeze under the canonical key, with the platform back-ref
        plugin._freeze_session_snapshot(
            self.CANONICAL,
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
        plugin._LAST_CANONICAL_BY_PLATFORM[self.PLATFORM] = self.CANONICAL

        # Call _on_session_reset with the new session UUID Hermes hands
        # us — NOT the canonical key — exactly as Hermes' gateway does
        # after /new.
        plugin._on_session_reset(
            session_id="20260601_162555_d16739f9",
            platform=self.PLATFORM,
        )

        self.assertNotIn(self.CANONICAL, plugin._FROZEN_BY_SESSION,
                         "freeze under canonical key should be evicted via platform fallback")

    def test_new_then_message_does_not_reuse_freeze(self):
        """End-to-end: /new arrives, then a real message. The real message
        must produce frozen_reuse=False because /new should have skipped
        building any freeze, and any pre-existing freeze must have been
        evicted by on_session_reset's platform fallback."""
        store = self._make_session_store()

        # Step 1: /new arrives. pre_gateway_dispatch fires.
        plugin._on_pre_gateway_dispatch(event=self._make_event("/new"), session_store=store)
        self.assertNotIn(self.CANONICAL, plugin._FROZEN_BY_SESSION)

        # Step 2: Gateway routes /new → on_session_reset with new UUID.
        plugin._on_session_reset(
            session_id="20260601_162555_d16739f9",
            platform=self.PLATFORM,
        )
        self.assertNotIn(self.CANONICAL, plugin._FROZEN_BY_SESSION)

        # Step 3: Real user message arrives. With cache_mode=on, this
        # should build a fresh freeze (reuses=0), not reuse one.
        plugin._CONFIG["cache_mode"] = "on"
        plugin._on_pre_gateway_dispatch(event=self._make_event("Run the communication test"), session_store=store)
        frozen = plugin._FROZEN_BY_SESSION.get(self.CANONICAL)
        self.assertIsNotNone(frozen, "real message should build a fresh freeze")
        self.assertEqual(frozen["reuses"], 0,
                         "fresh freeze should have reuses=0 (not a reuse of a stale snapshot)")


class BypassCohortTests(unittest.TestCase):
    """Regression guard: bypass-cohort rows must not pollute cache-on/off
    savings figures.

    The A/B baseline cohort (config: bypass_rate > 0) deterministically
    ships the full toolset for a fraction of sessions so the analyzer
    can compare narrowed vs unnarrowed cohorts. These rows write
    policy_source: "bypass" and ceiling_count == narrowed_count
    (i.e. tokens_saved == 0). Including them in the headline cache-on
    or cache-off "tokens saved" figure would tank the average with rows
    that NEVER narrowed by design.
    """

    def setUp(self):
        # Import the savings-report module by file path — it lives under
        # scripts/ not the importable plugin package, so we load it
        # explicitly. This mirrors how the script is invoked in practice.
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        spec = importlib.util.spec_from_file_location(
            "savings_report", scripts_dir / "savings-report.py"
        )
        assert spec and spec.loader, "could not load savings-report.py"
        self.report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.report)

    def _bypass_row(self, prediction_id: str = "bp1") -> dict[str, Any]:
        return {
            "prediction_id": prediction_id,
            "session_id": "agent:main:slack:dm:D0X:1780000000",
            "policy_source": "bypass",
            "ceiling_count": 46, "narrowed_count": 46,
            "ceiling_tokens": 12521, "narrowed_tokens": 12521,
        }

    def _narrowed_row(self, prediction_id: str = "nx1") -> dict[str, Any]:
        return {
            "prediction_id": prediction_id,
            "session_id": "agent:main:slack:dm:D0Y:1780000001",
            "policy_source": "preset",
            "ceiling_count": 46, "narrowed_count": 29,
            "ceiling_tokens": 12521, "narrowed_tokens": 7137,
            "frozen_reuse": False,
        }

    def test_bypass_row_classifies_as_bypass(self):
        api_last = {"bp1": {"cache_mode": "on"}}  # api says on, but bypass wins
        self.assertEqual(
            self.report.classify_prediction_mode(self._bypass_row(), api_last),
            "bypass",
        )

    def test_bypass_row_excluded_from_cache_on_cohort(self):
        """The cache-on cohort must not include bypass rows even when
        the underlying api_call cache_mode says 'on'."""
        api_last = {"bp1": {"cache_mode": "on"}, "nx1": {"cache_mode": "on"}}
        rows = [self._bypass_row(), self._narrowed_row()]
        on_stats = self.report.cohort_stats(rows, api_last, [], mode_filter="on")
        self.assertEqual(on_stats["n_predictions"], 1,
                         "cache-on cohort should contain only the narrowed row")
        self.assertEqual(on_stats["saved_tokens_total"], 5384,
                         "saved total should reflect ONLY the narrowed row")

    def test_bypass_cohort_collected_separately(self):
        """Bypass rows are still counted — just under their own cohort."""
        api_last = {"bp1": {"cache_mode": "on"}}
        bypass_stats = self.report.cohort_stats(
            [self._bypass_row()], api_last, [], mode_filter="bypass"
        )
        self.assertEqual(bypass_stats["n_predictions"], 1)
        # Bypass shipped the full ceiling — "saved" should be 0
        # honestly (no narrowing happened by design).
        self.assertEqual(bypass_stats["saved_tokens_total"], 0)
        self.assertEqual(bypass_stats["ceiling_tokens_total"], 12521)


class TokenEstimatorTests(unittest.TestCase):
    """Verify estimate_tokens uses tiktoken when available, chars/4 when not.

    The soft-dep pattern: tiktoken IS imported lazily inside _get_encoder,
    cached via functools.lru_cache. The estimator name is exposed via
    token_estimator_name() so per-row provenance is honest.
    """

    def setUp(self):
        # Import logger_io fresh and clear the encoder cache so each test
        # exercises whichever environment we set up here.
        self.logger_io = importlib.import_module("tool_belt_plugin.logger_io")
        self.logger_io._get_encoder.cache_clear()

    def tearDown(self):
        # Restore the natural state of the cache (re-detect on next call).
        self.logger_io._get_encoder.cache_clear()

    def test_chars_div_4_fallback_when_tiktoken_missing(self):
        """When tiktoken can't be imported, fall back to chars/4 and stamp
        the row with the fallback name."""
        original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

        def block_tiktoken(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("tiktoken not installed (simulated)")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=block_tiktoken):
            self.assertEqual(self.logger_io.token_estimator_name(), "chars-div-4")
            # Sample payload: 16 chars of JSON → 4 tokens
            self.assertEqual(self.logger_io.estimate_tokens(["abcdef"]), len('["abcdef"]') // 4)

    def test_tiktoken_path_when_available(self):
        """When tiktoken IS importable, prefer it and stamp the row."""
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            self.skipTest("tiktoken not installed in this environment")

        self.assertEqual(self.logger_io.token_estimator_name(), "tiktoken-cl100k")
        # Sanity: a non-trivial payload returns a positive count, and that
        # count is plausibly different from the chars/4 heuristic.
        payload = [{"name": "read_file",
                    "description": "Read a file from disk.",
                    "parameters": {"type": "object",
                                   "properties": {"path": {"type": "string"}},
                                   "required": ["path"]}}]
        n = self.logger_io.estimate_tokens(payload)
        self.assertGreater(n, 0)

    def test_encoder_cached_across_calls(self):
        """The encoder load is one-time per process (lru_cache)."""
        name1 = self.logger_io.token_estimator_name()
        name2 = self.logger_io.token_estimator_name()
        self.assertEqual(name1, name2)
        # Calling _get_encoder directly should also hit the cache:
        info_before = self.logger_io._get_encoder.cache_info()
        self.logger_io._get_encoder()
        info_after = self.logger_io._get_encoder.cache_info()
        self.assertEqual(info_after.hits, info_before.hits + 1)


if __name__ == "__main__":
    unittest.main()
