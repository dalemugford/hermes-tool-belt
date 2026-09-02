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
    plugin._CACHE_MODE_BY_SESSION.clear()
    plugin._CACHE_DECISION_BY_SESSION.clear()
    plugin._LAST_CANONICAL_BY_PLATFORM.clear()
    plugin._DETECTION_CACHE.clear()
    plugin._DETECTION_CACHE_LOADED = False
    # Host facts (configured primary provider/model) live outside _CONFIG;
    # clear so a test that sets a primary can't leak into the next.
    plugin._HOST_MODEL.update(provider="", model="")


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

    def test_decision_is_pinned_for_the_sessions_lifetime(self):
        # A detection lock landing MID-SESSION must not flip the current
        # session's path: granted expansions would vanish and the freeze
        # entry would orphan. The lock applies to the next session only.
        plugin._CONFIG["cache_mode"] = "auto"
        scope = "assistant-a:telegram"
        self.assertEqual(
            plugin._resolve_cache_mode_for_session("pin-sid", scope=scope), "on")
        plugin._DETECTION_CACHE[scope] = {"mode": "off"}
        self.assertEqual(
            plugin._resolve_cache_mode_for_session("pin-sid", scope=scope), "on",
            "mid-session lock must not flip the live session")
        self.assertEqual(
            plugin._resolve_cache_mode_for_session("next-sid", scope=scope), "off",
            "the NEXT session honors the lock")


class CarryAllStateTests(unittest.TestCase):
    """_build_carry_all_state: the cache-on (caching-provider) dispatch state.

    Replaces the removed freeze model (D1, 2026-09-02): caching providers
    carry the whole ceiling, ship no ``expand_tools``, and never mutate the
    tool list mid-session, so there is no per-session snapshot to reuse.
    """

    class _Preset:
        name = "aggressive"
        policy_version = "v1"
        always_carry = ["clarify"]
        carry = ["read_file", "write_file"]
        demoted: list = []
        config_always_carry: list = []
        learned_mode = "apply"
        learned_scope = ""
        learned_changes: list = []

    def setUp(self):
        _seed_plugin_config()
        _reset_plugin_state()

    def _build(self, message="hi"):
        return plugin._build_carry_all_state(
            self._Preset(),
            session_id="sid", scope="assistant-a:telegram",
            channel="assistant-a:telegram", agent="assistant-a",
            platform="telegram", message=message,
        )

    def test_carry_all_state_is_no_narrowing(self):
        state = self._build()
        self.assertIs(state["active_tool_names"], plugin.NO_NARROWING)
        self.assertIs(state["baseline_active_tools"], plugin.NO_NARROWING)

    def test_carry_all_stamps_distinct_policy_source(self):
        # Must NOT be "bypass" — that is the internal A/B control cohort;
        # the analyzer has to tell the two unnarrowed cohorts apart.
        self.assertEqual(self._build()["policy_source"], "cache_on_carry_all")

    def test_carry_all_disables_sticky_and_lookback(self):
        state = self._build()
        self.assertEqual(state["sticky_key"], "")
        self.assertEqual(state["sticky_tools"], [])
        self.assertEqual(state["lookback_used"], 0)
        self.assertEqual(state["lookback_turns_config"], 0)

    def test_carry_all_starts_with_no_expansions(self):
        # Nothing is admitted mid-session; expand_tools is not even shipped.
        self.assertEqual(self._build()["expansions"], set())


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
            plugin._CACHE_MODE_BY_SESSION["sid"]["buckets"][""]["lock_reason"],
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
            plugin._CACHE_MODE_BY_SESSION["sid"]["buckets"][""]["lock_reason"],
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
            plugin._CACHE_MODE_BY_SESSION["sid"]["buckets"][""]["lock_reason"],
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

    Hermes fires on_session_end once per user message, so the session's
    posture pin (_CACHE_DECISION_BY_SESSION) and detection state
    (_CACHE_MODE_BY_SESSION) must survive it; on_session_reset — /new,
    /reset — evicts them through _evict_session_cache_state (D3, 2026-09-02;
    the old freeze snapshot was removed with the carry-all posture).
    """

    def setUp(self):
        _seed_plugin_config()
        _reset_plugin_state()
        # Seed a posture pin + its detection bucket, as a live cache-on
        # session would carry after its first dispatch.
        plugin._CACHE_DECISION_BY_SESSION["sid"] = {
            "mode": "on", "provider": "openai-codex",
        }
        plugin._CACHE_MODE_BY_SESSION["sid"] = {
            "observed_provider": "openai-codex",
            "buckets": {"openai-codex": {"mode": "on", "calls_observed": 3}},
        }

    def test_on_session_end_does_not_evict_posture(self):
        with mock.patch.dict(os.environ, {"HERMES_SESSION_KEY": "sid"}, clear=False):
            plugin._on_session_end(session_id="sid", session_key="sid")
        self.assertIn("sid", plugin._CACHE_DECISION_BY_SESSION)
        self.assertIn("sid", plugin._CACHE_MODE_BY_SESSION)

    def test_on_session_end_keeps_sticky_and_lookback(self):
        # The hook fires per turn; sticky (3-turn TTL) and lookback are
        # session-scoped and must survive it, or the features are inert.
        sticky_key = plugin._sticky_key_for_session("sid")
        self.assertTrue(sticky_key, "sanity: derivation produces non-empty key")
        plugin._STICKY_BY_KEY[sticky_key] = {"terminal": {"remaining_turns": 2}}
        plugin._PRIOR_MESSAGES_BY_SESSION["sid"] = ["msg1"]
        with mock.patch.dict(os.environ, {"HERMES_SESSION_KEY": "sid"}, clear=False):
            plugin._on_session_end(session_id="sid", session_key="sid")
        self.assertIn("sid", plugin._PRIOR_MESSAGES_BY_SESSION)
        self.assertIn(sticky_key, plugin._STICKY_BY_KEY)

    def test_on_session_reset_evicts_posture(self):
        plugin._on_session_reset(session_id="sid", session_key="sid")
        self.assertNotIn("sid", plugin._CACHE_DECISION_BY_SESSION)
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
        """A /new message must not pin a posture for the session."""
        event = self._make_event("/new")
        store = self._make_session_store()
        plugin._on_pre_gateway_dispatch(event=event, session_store=store)
        self.assertNotIn(self.CANONICAL, plugin._CACHE_DECISION_BY_SESSION,
                         "no posture pin should be created for slash-command messages")

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
        # Seed: a posture pin under the canonical key, with the platform back-ref.
        plugin._CACHE_DECISION_BY_SESSION[self.CANONICAL] = {
            "mode": "on", "provider": "openai-codex",
        }
        plugin._LAST_CANONICAL_BY_PLATFORM[self.PLATFORM] = self.CANONICAL

        # Call _on_session_reset with the new session UUID Hermes hands
        # us — NOT the canonical key — exactly as Hermes' gateway does
        # after /new.
        plugin._on_session_reset(
            session_id="20260601_162555_d16739f9",
            platform=self.PLATFORM,
        )

        self.assertNotIn(self.CANONICAL, plugin._CACHE_DECISION_BY_SESSION,
                         "posture pin under canonical key should be evicted via platform fallback")

    def test_new_then_message_builds_fresh_carry_all(self):
        """End-to-end: /new arrives, then a real message. /new must not pin
        a posture; on_session_reset evicts any stale pin via the platform
        fallback; the real message then builds a fresh carry-all state under
        cache_mode=on (policy_source cache_on_carry_all), not a stale reuse."""
        store = self._make_session_store()

        # Step 1: /new arrives. pre_gateway_dispatch fires.
        plugin._on_pre_gateway_dispatch(event=self._make_event("/new"), session_store=store)
        self.assertNotIn(self.CANONICAL, plugin._CACHE_DECISION_BY_SESSION)

        # Step 2: Gateway routes /new → on_session_reset with new UUID.
        plugin._on_session_reset(
            session_id="20260601_162555_d16739f9",
            platform=self.PLATFORM,
        )
        self.assertNotIn(self.CANONICAL, plugin._CACHE_DECISION_BY_SESSION)

        # Step 3: Real user message arrives. With cache_mode=on, this builds
        # a carry-all dispatch state stamped cache_on_carry_all.
        plugin._CONFIG["cache_mode"] = "on"
        plugin._on_pre_gateway_dispatch(
            event=self._make_event("Run the communication test"), session_store=store)
        state = plugin._PREDICTION_CV.get()
        self.assertIsNotNone(state, "real message should build a dispatch state")
        self.assertEqual(state.get("policy_source"), "cache_on_carry_all")
        self.assertIs(state.get("active_tool_names"), plugin.NO_NARROWING)


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


class EnabledCeilingCaptureTests(unittest.TestCase):
    """The first request of a turn captures the enabled built-in ceiling into
    prediction state — before expand_tools can report any activation — and the
    partition filter cuts untriggered expand_only tools while passing MCP
    tools through untouched."""

    def setUp(self):
        _seed_plugin_config()
        _reset_plugin_state()

    def test_first_request_captures_ceiling_and_partitions(self):
        state = {
            "active_tool_names": ["clarify"],
            "resolved_always_carry": ["clarify"],
            "resolved_carry": [],
            # Full-start: narrowing on the wire happens only through the
            # demoted loadout — this also pins that learned demotions REACH
            # the resolve path (a missing wire would keep read_file active).
            "resolved_demoted": ["read_file"],
            "triggered_tools": [],
            "expansions": set(),
            "logged": False,
            "session_id": "",
        }
        token = plugin._PREDICTION_CV.set(state)
        try:
            def original(self_, msgs):
                return {
                    "tools": [
                        {"name": "clarify"},
                        {"name": "read_file"},
                        {"name": "mcp__github__create_issue"},
                    ],
                    "model": "claude-test",
                }
            wrapped = plugin._wrap_build_api_kwargs(original)
            with mock.patch.dict(plugin._CONFIG, {"enabled": True}), \
                    mock.patch.object(plugin, "_maybe_log_prediction", lambda *a, **k: None):
                result = wrapped(object(), [])
        finally:
            plugin._PREDICTION_CV.reset(token)

        # Ceiling captured, MCP excluded from the built-in partition domain.
        self.assertIn("clarify", state["enabled_ceiling"])
        self.assertIn("read_file", state["enabled_ceiling"])
        self.assertNotIn("mcp__github__create_issue", state["enabled_ceiling"])
        self.assertIn("mcp__github__create_issue", state["mcp_passthrough_tools"])

        kept = [plugin._tool_name(t) for t in result["tools"]]
        self.assertIn("clarify", kept, "resident stays active")
        self.assertNotIn("read_file", kept,
                         "untriggered expand_only tool is cut")
        self.assertIn("mcp__github__create_issue", kept,
                      "MCP tool passes through untouched")


class ExpandOnlyManifestInjectionTests(unittest.TestCase):
    """Phase 4: the request builder appends a compact expand-only manifest to a
    CLONE of the carried ``expand_tools`` schema — naming enabled expand-only
    built-ins the model would otherwise not know exist, without carrying their
    full schemas."""

    # category -> member tool names, injected so grouping is deterministic
    # without the live ``toolsets`` table.
    _INDEX = {
        "web": {"web_extract"},
        "browser": {"browser_exec", "browser_navigate"},
    }

    def setUp(self):
        _seed_plugin_config()
        _reset_plugin_state()

    def _tools(self):
        # A live tool list: two residents (clarify, expand_tools), one adaptive
        # resident (read_file), three expand-only built-ins (two categorized,
        # one unknown), and one MCP pass-through.
        return [
            {"name": "clarify"},
            {"name": "expand_tools", "description": expand_tools_SCHEMA_DESC},
            {"name": "read_file"},
            {"name": "web_extract"},
            {"name": "browser_exec"},
            {"name": "brand_new_builtin"},
            {"name": "mcp__github__create_issue"},
        ]

    def _run(self, *, tools=None, triggered=(), expand_tools_schema=None,
             manifest_side_effect=None):
        """Run the wrapped builder and return (result_kwargs, expand_tools_desc)."""
        if tools is None:
            tools = self._tools()
        if expand_tools_schema is not None:
            tools = [expand_tools_schema if plugin._tool_name(t) == "expand_tools"
                     else t for t in tools]
        state = {
            "active_tool_names": ["clarify", "expand_tools", "read_file"],
            "resolved_always_carry": ["clarify", "expand_tools"],
            "resolved_carry": ["read_file"],
            # Full-start: X materializes only from the demoted loadout.
            "resolved_demoted": ["web_extract", "browser_exec", "brand_new_builtin"],
            "triggered_tools": list(triggered),
            "expansions": set(),
            "logged": False,
            "session_id": "",
        }
        token = plugin._PREDICTION_CV.set(state)

        def original(self_, msgs):
            return {"tools": list(tools), "model": "claude-test"}

        wrapped = plugin._wrap_build_api_kwargs(original)
        patches = [
            mock.patch.dict(plugin._CONFIG, {"enabled": True}),
            mock.patch.object(plugin, "_maybe_log_prediction", lambda *a, **k: None),
            mock.patch.object(plugin.expand_tools_mod, "_build_category_index",
                              return_value={k: set(v) for k, v in self._INDEX.items()}),
        ]
        if manifest_side_effect is not None:
            patches.append(mock.patch.object(
                plugin.expand_tools_mod, "build_expand_only_manifest",
                side_effect=manifest_side_effect))
        try:
            with patches[0], patches[1], patches[2]:
                if manifest_side_effect is not None:
                    with patches[3]:
                        result = wrapped(object(), [])
                else:
                    result = wrapped(object(), [])
        finally:
            plugin._PREDICTION_CV.reset(token)

        desc = ""
        for t in result["tools"]:
            if plugin._tool_name(t) == "expand_tools":
                desc = t.get("description", "")
        return result, desc

    def test_manifest_names_unlisted_enabled_expand_only_tools(self):
        _result, desc = self._run()
        # The model discovers the expand-only built-ins by name, grouped.
        self.assertIn("web: web_extract", desc)
        self.assertIn("browser: browser_exec", desc)
        self.assertIn("(ungrouped): brand_new_builtin", desc)
        # The recovery affordance is explained.
        self.assertIn("expand_tools(tool=", desc)
        self.assertIn("trigger", desc.lower())

    def test_manifest_omits_full_schemas_of_expand_only_tools(self):
        result, _desc = self._run()
        kept = [plugin._tool_name(t) for t in result["tools"]]
        # The expand-only tools are named in the manifest but their full
        # schemas are NOT carried in the tool list.
        self.assertNotIn("web_extract", kept)
        self.assertNotIn("browser_exec", kept)
        self.assertNotIn("brand_new_builtin", kept)

    def test_carried_and_always_carried_names_absent_from_manifest(self):
        _result, desc = self._run()
        # clarify + expand_tools (always_carry) and read_file (carry) are
        # residents — they must not appear in the expand-only manifest.
        manifest = desc.split("Grouped by toolset:", 1)[-1]
        self.assertNotIn("clarify", manifest)
        self.assertNotIn("read_file", manifest)
        # 'expand_tools' appears in the recovery prose but never as a listed
        # expand-only entry (no line naming it as a group member).
        for line in manifest.splitlines():
            if ":" in line:
                self.assertNotIn("expand_tools", line.split(":", 1)[1])

    def test_mcp_passthrough_names_absent_from_manifest(self):
        _result, desc = self._run()
        self.assertNotIn("mcp__github__create_issue", desc)
        self.assertNotIn("github", desc)

    def test_ceiling_absent_trigger_name_absent_from_manifest(self):
        # A trigger references a tool that is NOT in the enabled ceiling E.
        _result, desc = self._run(triggered=["ghost_tool"])
        self.assertNotIn("ghost_tool", desc)

    def test_original_registered_schema_unchanged_after_build(self):
        import copy
        snapshot = copy.deepcopy(plugin.expand_tools_mod.SCHEMA)
        _result, desc = self._run(expand_tools_schema=plugin.expand_tools_mod.SCHEMA)
        # The registered object is byte/deep-equal after request construction...
        self.assertEqual(plugin.expand_tools_mod.SCHEMA, snapshot,
                         "the registered expand_tools schema must not be mutated")
        # ...but the per-request clone carried the manifest.
        self.assertIn("web: web_extract", desc)
        self.assertNotIn(
            "web: web_extract",
            plugin.expand_tools_mod.SCHEMA["description"],
            "the manifest must live only on the per-request clone",
        )

    def test_cache_on_manifest_identical_before_and_after_trigger(self):
        # Residency (hence X) is identical across turns; a later trigger only
        # grows the active set. The manifest must stay byte-identical.
        _r1, before = self._run(triggered=())
        _r2, after = self._run(triggered=["web_extract"])
        self.assertEqual(before, after,
                         "manifest is stable across trigger activation (cache-on)")

    def test_manifest_construction_failure_preserves_original_schema(self):
        # If manifest construction raises, the build fails OPEN: the expand_tools
        # tool keeps its original description and the tool list is still returned.
        result, desc = self._run(
            manifest_side_effect=RuntimeError("boom"))
        self.assertEqual(desc, expand_tools_SCHEMA_DESC,
                         "on failure the original expand_tools description survives")
        # The rest of the narrowing still happened.
        kept = [plugin._tool_name(t) for t in result["tools"]]
        self.assertIn("clarify", kept)
        self.assertNotIn("web_extract", kept)


# The real schema description, captured once so injection tests can compare
# against it without hardcoding prose.
expand_tools_SCHEMA_DESC = plugin.expand_tools_mod.SCHEMA["description"]


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

if __name__ == "__main__":
    unittest.main()


def tearDownModule():
    # This module's config seeders clear-and-replace the module _CONFIG; put
    # the pristine in-code defaults back so later files see the real deploy
    # state (hygiene debt found by the Tier-0 rebuild).
    plugin._CONFIG.clear()
    plugin._CONFIG.update(conftest.PRISTINE_CONFIG)
