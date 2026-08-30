"""Focused tests for the telemetry-attribution fixes.

Covers:
  1. Prediction ``session_id`` no longer blank for gateway-dispatched
     events when ``session_store`` is reachable.
  2. ``_should_bypass`` activates once ``session_id`` exists (it was a
     no-op when the field came through empty).
  3. Real Hermes session-key shape parsing — ``agent:main:{platform}:…``,
     where position 1 is the *literal* string ``"main"``, not the agent
     name. Recovering ``assistant-a:telegram`` from a real gateway key
     requires combining the canonical platform with a profile-derived
     agent.
  4. Blank attribution preserved for genuinely untrackable rows.
  5. ``_on_session_end`` evicts sticky/lookback state keyed by the
     canonical session key, even when Hermes hands us the AIAgent's
     uuid-style ``session_id``.
  6. Analyzer warns clearly when PyYAML / preset excludes can't be
     loaded.
"""

from __future__ import annotations

import importlib
from collections import Counter
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# conftest.py is loaded by the run_tests.py entry; if a user invokes the
# test module directly (``python tests/test_session_attribution.py``)
# we still need to make sure the plugin package is registered.
if "tool_belt_plugin" not in sys.modules:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent.parent))
    sys.path.insert(0, str(here.parent))
    from tests import conftest  # noqa: F401 — side-effect: register package

plugin = sys.modules["tool_belt_plugin"]
logger_io = importlib.import_module("tool_belt_plugin.logger_io")
analyze = importlib.import_module("tool_belt_plugin.analyze")


# The real Hermes session key shape from ``gateway/session.py::build_session_key``.
# Position 1 is always the literal string "main" — NOT the per-profile
# agent identity. This is the exact bug the earlier tests missed.
REAL_KEY_TELEGRAM = "agent:main:telegram:dm:12345"
REAL_KEY_WHATSAPP = "agent:main:whatsapp:dm:5551234567@s.whatsapp.net"


def _make_event(text: str = "please open the browser") -> SimpleNamespace:
    """Minimal stand-in for a Hermes MessageEvent.

    The real ``MessageEvent`` dataclass deliberately has no ``session_id``
    attribute — that's exactly the bug we're regression-testing — so we
    mirror that here and only supply ``source``.
    """
    source = SimpleNamespace(platform=SimpleNamespace(value="telegram"), chat_id="12345")
    return SimpleNamespace(source=source, text=text, attachments=[])


def _fake_session_store(key: str = REAL_KEY_TELEGRAM):
    """Stand-in for the gateway SessionStore exposing ``_generate_session_key``."""
    store = mock.MagicMock()
    store._generate_session_key.return_value = key
    return store


def _make_profile_dir(parent: str, agent_name: str) -> str:
    """Create ``<parent>/profiles/<agent_name>`` so HERMES_HOME parsing
    resolves to ``agent_name``."""
    path = Path(parent) / "profiles" / agent_name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


class CanonicalSessionKeyTests(unittest.TestCase):
    def test_uses_session_store_generator(self):
        event = _make_event()
        store = _fake_session_store()
        result = plugin._canonical_session_key(event, store, {})
        store._generate_session_key.assert_called_once_with(event.source)
        self.assertEqual(result, REAL_KEY_TELEGRAM)

    def test_falls_back_to_env_when_store_absent(self):
        event = _make_event()
        with mock.patch.dict(os.environ, {"HERMES_SESSION_KEY": REAL_KEY_TELEGRAM}, clear=False):
            result = plugin._canonical_session_key(event, None, {})
        self.assertEqual(result, REAL_KEY_TELEGRAM)

    def test_returns_blank_when_nothing_available(self):
        event = _make_event()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(plugin._canonical_session_key(event, None, {}), "")


class SessionKeyParsingTests(unittest.TestCase):
    """Regression tests for the ``parts[1] == "main"`` parsing bug."""

    def setUp(self):
        self._original_config = dict(plugin._CONFIG)
        self.addCleanup(self._restore_config)
        plugin._CONFIG["agent"] = ""

    def _restore_config(self) -> None:
        plugin._CONFIG.clear()
        plugin._CONFIG.update(self._original_config)

    def test_platform_from_real_session_key(self):
        self.assertEqual(plugin._platform_from_session_key(REAL_KEY_TELEGRAM), "telegram")
        self.assertEqual(plugin._platform_from_session_key(REAL_KEY_WHATSAPP), "whatsapp")

    def test_platform_from_session_key_ignores_uuid(self):
        # AIAgent.session_id has no "agent:" prefix; the helper must
        # return blank rather than guessing.
        self.assertEqual(plugin._platform_from_session_key("20260516_123456_deadbeef"), "")

    def test_profile_agent_from_hermes_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile_dir = _make_profile_dir(tmp, "assistant-b")
            with mock.patch.dict(os.environ, {"HERMES_HOME": profile_dir}, clear=False):
                # Ensure _CONFIG['agent'] doesn't shadow HERMES_HOME for this case.
                plugin._CONFIG["agent"] = ""
                self.assertEqual(plugin._profile_agent_name(), "assistant-b")

    def test_root_home_uses_explicit_profile_env_before_default(self):
        with mock.patch.dict(
            os.environ,
            {"HERMES_HOME": "/tmp/custom hermes root", "HERMES_PROFILE": "operator"},
            clear=False,
        ):
            self.assertEqual(plugin._profile_agent_name(), "operator")

    def test_profileless_context_stays_blank(self):
        with mock.patch.dict(
            os.environ, {"HERMES_HOME": "", "HERMES_PROFILE": ""}, clear=False
        ):
            self.assertEqual(plugin._profile_agent_name(), "")


class PreGatewayDispatchSessionIdTests(unittest.TestCase):
    """Drives ``_on_pre_gateway_dispatch`` end-to-end with a temp state dir."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._env_patch = mock.patch.dict(
            os.environ,
            {
                "HERMES_HOME": self.tmp.name,
                "HERMES_SESSION_KEY": REAL_KEY_TELEGRAM,
            },
            clear=False,
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

        self._original_config = dict(plugin._CONFIG)
        plugin._CONFIG["enabled"] = True
        plugin._CONFIG["log"] = True
        plugin._CONFIG["agent"] = "assistant-a"
        self.addCleanup(self._restore_config)

        plugin._PREDICTION_CV.set(None)
        plugin._STICKY_BY_KEY.clear()
        plugin._PRIOR_MESSAGES_BY_SESSION.clear()
        plugin._POLICY_TURN_BY_SCOPE.clear()

    def _restore_config(self) -> None:
        plugin._CONFIG.clear()
        plugin._CONFIG.update(self._original_config)

    def test_session_id_populated_from_session_store(self):
        event = _make_event()
        store = _fake_session_store()
        plugin._on_pre_gateway_dispatch(event=event, gateway=None, session_store=store)
        state = plugin._PREDICTION_CV.get()
        self.assertIsNotNone(state, "pre_gateway_dispatch should set the contextvar")
        self.assertEqual(state["session_id"], REAL_KEY_TELEGRAM)
        self.assertEqual(state["agent"], "assistant-a")
        self.assertEqual(state["platform"], "telegram")
        self.assertEqual(state["scope"], "assistant-a:telegram")
        predictions = (Path(self.tmp.name) / "state" / "tool-belt" / "predictions.jsonl")
        plugin._maybe_log_prediction(state, ceiling=[], narrowed=[])
        rows = [json.loads(line) for line in predictions.read_text().splitlines() if line]
        self.assertTrue(rows, "expected at least one prediction row")
        self.assertEqual(rows[-1]["session_id"], REAL_KEY_TELEGRAM)
        self.assertEqual(rows[-1]["scope"], "assistant-a:telegram")


class BypassEligibilityTests(unittest.TestCase):
    def test_should_bypass_false_when_session_id_blank(self):
        self.assertFalse(plugin._should_bypass("assistant-a:telegram", ""))

    def test_should_bypass_can_fire_with_real_session_id(self):
        original_rate = plugin._CONFIG.get("bypass_rate")
        plugin._CONFIG["bypass_rate"] = 1.0
        try:
            self.assertTrue(plugin._should_bypass("assistant-a:telegram", REAL_KEY_TELEGRAM))
        finally:
            plugin._CONFIG["bypass_rate"] = original_rate


class PostToolCallAttributionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Make HERMES_HOME a real profile dir so _profile_agent_name()
        # resolves to "assistant-a" — mirrors the live ``hermes -p assistant-a
        # gateway`` shape.
        self.profile_home = _make_profile_dir(self.tmp.name, "assistant-a")
        self._env_patch = mock.patch.dict(
            os.environ,
            {"HERMES_HOME": self.profile_home, "HERMES_SESSION_KEY": ""},
            clear=False,
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

        self._original_config = dict(plugin._CONFIG)
        plugin._CONFIG["enabled"] = True
        plugin._CONFIG["log"] = True
        plugin._CONFIG["agent"] = ""  # force HERMES_HOME resolution
        self.addCleanup(self._restore_config)

        plugin._PREDICTION_CV.set(None)

    def _restore_config(self) -> None:
        plugin._CONFIG.clear()
        plugin._CONFIG.update(self._original_config)

    def test_recover_from_real_session_key_uses_profile_agent(self):
        # The earlier code would have returned ("main", "telegram", ...).
        # The fix derives the agent from the HERMES_HOME profile.
        agent, platform, scope = plugin._recover_attribution_without_state(
            REAL_KEY_TELEGRAM, kwargs={}
        )
        self.assertEqual(agent, "assistant-a")
        self.assertEqual(platform, "telegram")
        self.assertEqual(scope, "assistant-a:telegram")

    def test_recover_from_whatsapp_session_key(self):
        # Confirms the parser works across platforms, not just telegram.
        agent, platform, scope = plugin._recover_attribution_without_state(
            REAL_KEY_WHATSAPP, kwargs={}
        )
        self.assertEqual((agent, platform, scope), ("assistant-a", "whatsapp", "assistant-a:whatsapp"))

    def test_recover_returns_blanks_for_uuid_style_session_id(self):
        # AIAgent.session_id is a uuid/timestamp string. With no env-var
        # session key and no kwargs hints, we can't fill in platform —
        # so attribution stays blank rather than synthesizing defaults.
        with mock.patch.dict(os.environ, {"HERMES_SESSION_KEY": ""}, clear=False):
            agent, platform, scope = plugin._recover_attribution_without_state(
                "20260516_123456_deadbeef", kwargs={}
            )
        self.assertEqual((agent, platform, scope), ("", "", ""))

    def test_recover_blank_without_profile_context(self):
        # Even with a real session key, blank profile context means we
        # can't responsibly stamp an agent — leave the row blank.
        with mock.patch.dict(
            os.environ,
            {"HERMES_HOME": "", "HERMES_PROFILE": "", "HERMES_SESSION_KEY": ""},
            clear=False,
        ):
            plugin._CONFIG["agent"] = ""
            agent, platform, scope = plugin._recover_attribution_without_state(
                REAL_KEY_TELEGRAM, kwargs={}
            )
        self.assertEqual((agent, platform, scope), ("", "", ""))

    def test_context_uses_default_for_root_profile(self):
        # Hermes reserves "default" for the root profile. A custom root
        # HERMES_HOME therefore remains attributable without requiring a
        # private plugin-specific `agent:` setting.
        with mock.patch.dict(
            os.environ,
            {"HERMES_HOME": "/tmp/custom hermes root", "HERMES_PROFILE": "",
             "HERMES_SESSION_KEY": REAL_KEY_TELEGRAM},
            clear=False,
        ):
            plugin._CONFIG["agent"] = ""
            agent, platform, scope = plugin._agent_platform_from_context(event=None, kwargs={})
        self.assertEqual((agent, platform, scope),
                         ("default", "telegram", "default:telegram"))

    def test_context_resolves_when_config_agent_set(self):
        # An explicit agent remains the highest-precedence override even when
        # HERMES_HOME identifies the root profile.
        with mock.patch.dict(
            os.environ,
            {"HERMES_HOME": "/tmp/custom hermes root", "HERMES_PROFILE": "",
             "HERMES_SESSION_KEY": REAL_KEY_TELEGRAM},
            clear=False,
        ):
            plugin._CONFIG["agent"] = "assistant-a"
            agent, platform, scope = plugin._agent_platform_from_context(event=None, kwargs={})
        self.assertEqual((agent, platform, scope),
                         ("assistant-a", "telegram", "assistant-a:telegram"))

    def test_post_tool_call_logs_recovered_attribution(self):
        plugin._on_post_tool_call(
            tool_name="terminal",
            args={"command": "ls"},
            result="ok",
            task_id="t1",
            session_id=REAL_KEY_TELEGRAM,
        )
        tool_calls_path = Path(self.profile_home) / "state" / "tool-belt" / "tool_calls.jsonl"
        rows = [json.loads(l) for l in tool_calls_path.read_text().splitlines() if l]
        self.assertTrue(rows, "expected one tool-call row")
        row = rows[-1]
        self.assertEqual(row["agent"], "assistant-a")
        self.assertEqual(row["platform"], "telegram")
        self.assertEqual(row["scope"], "assistant-a:telegram")


class SessionEndCleanupTests(unittest.TestCase):
    """The pre-dispatch path keys sticky/lookback state by the canonical
    session key. Hermes' ``on_session_end`` hands us the AIAgent's
    uuid-style ``session_id`` — without the env-var fallback the plugin
    would leave stale per-session entries behind."""

    AGENT_SESSION_ID = "20260516_123456_deadbeef"

    def setUp(self):
        self._original_config = dict(plugin._CONFIG)
        plugin._CONFIG["enabled"] = True
        self.addCleanup(self._restore_config)
        plugin._STICKY_BY_KEY.clear()
        plugin._PRIOR_MESSAGES_BY_SESSION.clear()
        plugin._PREDICTION_CV.set(None)

    def _restore_config(self) -> None:
        plugin._CONFIG.clear()
        plugin._CONFIG.update(self._original_config)

    def _seed_state(self, canonical_key: str) -> str:
        sticky_key = plugin._sticky_key_for_session(canonical_key)
        plugin._STICKY_BY_KEY[sticky_key] = {
            "browser": {"tools": {"open_browser"}, "remaining_turns": 3}
        }
        plugin._PRIOR_MESSAGES_BY_SESSION[canonical_key] = ["hello"]
        return sticky_key

    def test_evicts_canonical_state_when_env_key_present(self):
        sticky_key = self._seed_state(REAL_KEY_TELEGRAM)
        with mock.patch.dict(
            os.environ, {"HERMES_SESSION_KEY": REAL_KEY_TELEGRAM}, clear=False
        ):
            plugin._on_session_end(session_id=self.AGENT_SESSION_ID)
        self.assertNotIn(sticky_key, plugin._STICKY_BY_KEY)
        self.assertNotIn(REAL_KEY_TELEGRAM, plugin._PRIOR_MESSAGES_BY_SESSION)

    def test_evicts_canonical_state_via_explicit_kwarg(self):
        sticky_key = self._seed_state(REAL_KEY_TELEGRAM)
        with mock.patch.dict(os.environ, {"HERMES_SESSION_KEY": ""}, clear=False):
            plugin._on_session_end(
                session_id=self.AGENT_SESSION_ID,
                session_key=REAL_KEY_TELEGRAM,
            )
        self.assertNotIn(sticky_key, plugin._STICKY_BY_KEY)
        self.assertNotIn(REAL_KEY_TELEGRAM, plugin._PRIOR_MESSAGES_BY_SESSION)

    def test_also_evicts_uuid_keyed_state(self):
        # UUID-keyed state is a supported defensive fallback and must be
        # cleaned up alongside canonical-key state.
        uuid_sticky = self._seed_state(self.AGENT_SESSION_ID)
        with mock.patch.dict(os.environ, {"HERMES_SESSION_KEY": ""}, clear=False):
            plugin._on_session_end(session_id=self.AGENT_SESSION_ID)
        self.assertNotIn(uuid_sticky, plugin._STICKY_BY_KEY)
        self.assertNotIn(self.AGENT_SESSION_ID, plugin._PRIOR_MESSAGES_BY_SESSION)


class AnalyzerExcludesDegradedModeTests(unittest.TestCase):
    def test_load_preset_excludes_returns_status_no_policy(self):
        out, status = analyze._load_preset_excludes(Path(tempfile.mkdtemp()))
        self.assertEqual(out, {})
        self.assertEqual(status, "no_policy")

    def test_load_preset_always_carry_reads_the_shipped_policy(self):
        # (Formerly named for a no-PyYAML fallback it never exercised; the
        # fallback is gone — see tests/test_shaper_porcelain.py for the
        # loud-exit path.)
        tools, status = analyze._load_preset_always_carry(Path(plugin.__file__).parent)
        self.assertEqual(status, "ok")
        # always_carry holds only the immutable residents; adaptive carry tools
        # (mnemosyne_recall, process) are NOT part of the immutable set.
        self.assertIn("clarify", tools)
        self.assertIn("expand_tools", tools)
        self.assertNotIn("mnemosyne_recall", tools)
        self.assertNotIn("process", tools)

    def test_dampener_candidates_emits_warning_when_degraded(self):
        stat = analyze.ScopeStats(scope="assistant-a:telegram")
        stat.trigger_fp_previews["browser"] = [
            "please open browser please",
            "please open browser please",
            "please open browser please",
        ]
        args = SimpleNamespace(
            suggest_dampeners=True,
            dampener_min_support=2,
            dampener_min_n=2,
            dampener_max_n=3,
            dampener_min_precision=0.5,
            dampener_max_candidates=5,
        )

        def fake_load_excludes(_plugin_dir):
            return {}, "no_yaml"

        with mock.patch.object(analyze, "_load_preset_excludes", fake_load_excludes):
            with mock.patch.object(sys, "stderr") as stderr_mock:
                rows = analyze.dampener_candidates({"assistant-a:telegram": stat}, args)
        printed = "".join(
            call.args[0] for call in stderr_mock.write.call_args_list if call.args
        )
        self.assertIn("PyYAML", printed)
        self.assertTrue(rows, "expected at least one dampener row to be produced")
        self.assertEqual(rows[0]["preset_excludes_status"], "no_yaml")


class TriggerFpLateBoundTpTests(unittest.TestCase):
    """The analyzer must credit a trigger as TP when the matching tool
    is called within a few turns of the trigger firing in the same
    session — sticky residency carries expansions across turns, so a
    "fired here, used three turns later" pattern is correct prediction,
    not a false positive.

    When ``session_id`` is blank, the analyzer must fall back to
    same-prediction behavior to avoid cross-session leakage.
    """

    def _row_prediction(self, *, pid, sid, trigger, tools_for_trigger, ts):
        return {
            "ts": ts,
            "prediction_id": pid,
            "session_id": sid,
            "scope": "assistant-a:telegram",
            "agent": "assistant-a",
            "platform": "telegram",
            "policy_source": "preset",
            "message_preview": f"msg-{pid}",
            "triggers_fired": [trigger] if trigger else [],
            "trigger_tools_by_group": {trigger: tools_for_trigger} if trigger else {},
            "tokens_saved": 0,
            "ceiling_tokens": 0,
            "narrowed_tokens": 0,
            "always_on_tools": [],
            "cut_tools": [],
        }

    def _row_call(self, *, pid, tool):
        return {"prediction_id": pid, "tool_name": tool}

    def test_late_bound_call_within_session_window_counts_as_tp(self):
        preds = [
            self._row_prediction(pid="p1", sid="s1", trigger="shell",
                                 tools_for_trigger=["terminal"], ts=1.0),
            self._row_prediction(pid="p2", sid="s1", trigger=None,
                                 tools_for_trigger=[], ts=2.0),
        ]
        calls = [self._row_call(pid="p2", tool="terminal")]
        stats = analyze.collect_stats(preds, calls)
        scope = stats["assistant-a:telegram"]
        self.assertEqual(scope.trigger_hits["shell"], 1,
            "tool called in next prediction (within window) must count as TP")
        self.assertEqual(scope.trigger_false_positives.get("shell", 0), 0)

    def test_call_beyond_window_counts_as_fp(self):
        preds = [
            self._row_prediction(pid="p1", sid="s1", trigger="shell",
                                 tools_for_trigger=["terminal"], ts=1.0),
            self._row_prediction(pid="p2", sid="s1", trigger=None,
                                 tools_for_trigger=[], ts=2.0),
            self._row_prediction(pid="p3", sid="s1", trigger=None,
                                 tools_for_trigger=[], ts=3.0),
            self._row_prediction(pid="p4", sid="s1", trigger=None,
                                 tools_for_trigger=[], ts=4.0),
            self._row_prediction(pid="p5", sid="s1", trigger=None,
                                 tools_for_trigger=[], ts=5.0),
        ]
        # Window is 3 — p5 is 4 turns out, beyond the window.
        calls = [self._row_call(pid="p5", tool="terminal")]
        stats = analyze.collect_stats(preds, calls)
        scope = stats["assistant-a:telegram"]
        self.assertEqual(scope.trigger_false_positives.get("shell", 0), 1,
            "tool called beyond window must remain FP")

    def test_cross_session_call_does_not_leak(self):
        preds = [
            self._row_prediction(pid="p1", sid="s1", trigger="shell",
                                 tools_for_trigger=["terminal"], ts=1.0),
            # Different session — even if chronologically adjacent, must
            # not credit s1's trigger.
            self._row_prediction(pid="p2", sid="s2", trigger=None,
                                 tools_for_trigger=[], ts=2.0),
        ]
        calls = [self._row_call(pid="p2", tool="terminal")]
        stats = analyze.collect_stats(preds, calls)
        scope = stats["assistant-a:telegram"]
        self.assertEqual(scope.trigger_false_positives.get("shell", 0), 1,
            "different-session tool call must not satisfy the trigger")

    def test_blank_session_falls_back_to_same_prediction(self):
        # Historical telemetry with blank session_id — analyzer must
        # not synthesize a session and must not look ahead across
        # adjacent rows (would cross-contaminate).
        preds = [
            self._row_prediction(pid="p1", sid="", trigger="shell",
                                 tools_for_trigger=["terminal"], ts=1.0),
            self._row_prediction(pid="p2", sid="", trigger=None,
                                 tools_for_trigger=[], ts=2.0),
        ]
        calls = [self._row_call(pid="p2", tool="terminal")]
        stats = analyze.collect_stats(preds, calls)
        scope = stats["assistant-a:telegram"]
        self.assertEqual(scope.trigger_false_positives.get("shell", 0), 1,
            "blank session_id must fall back to same-prediction classification")


class TriggerKeywordSuggesterTests(unittest.TestCase):
    """The trigger-keyword suggester mines expand_only-tool message previews
    for candidate keywords that would have fired a trigger to activate the
    tool. Symmetric inverse of dampener mining; must filter the same
    pollution patterns (stop-words, existing-pattern overlap)."""

    def _args(self, **overrides):
        defaults = dict(
            suggest_trigger_keywords=True,
            dampener_min_support=2,
            dampener_min_n=2,
            dampener_max_n=4,
            dampener_min_precision=0.6,
            dampener_max_candidates=5,
            harvest_min_expand_only_calls=2,
            expand_round_trip_tokens=1500,
            per_tool_tokens=388,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _make_stats(self, *, scope="assistant-a:telegram", expand_only_previews,
                    all_previews, was_expand_only_count):
        from collections import Counter
        stat = analyze.ScopeStats(scope=scope)
        stat.harvest_predictions = len(all_previews)
        stat.harvest_all_previews = list(all_previews)
        for tool, previews in expand_only_previews.items():
            stat.harvest_expand_only_previews[tool] = list(previews)
            stat.harvest_was_expand_only[tool] = was_expand_only_count.get(tool, len(previews))
        return {scope: stat}

    def test_surfaces_content_words_filters_stopwords(self):
        # 5 cut messages mentioning "deploy the app", 20 noise messages
        # that talk about everything else. Expect "deploy the app" or
        # subset to surface; pure stopword n-grams ("and the", "for a")
        # must not surface even if frequent.
        cut_msgs = [
            "please deploy the app to staging",
            "can you deploy the app today",
            "deploy the app and confirm it ran",
            "i want to deploy the app",
            "deploy the app right now",
        ]
        noise = [
            "how are you doing today",
            "what time is the meeting and the call",
            "for a moment let me think",
            "and the next thing is to ask",
        ] * 5
        stats = self._make_stats(
            expand_only_previews={"deploy_tool": cut_msgs},
            all_previews=cut_msgs + noise,
            was_expand_only_count={"deploy_tool": 5},
        )
        with mock.patch.object(analyze, "_load_preset_triggers",
                               return_value=({}, "ok")):
            rows = analyze.trigger_keyword_candidates(stats, self._args())
        self.assertTrue(rows, "expected at least one candidate row")
        patterns = [c["pattern"] for c in rows[0]["candidates"]]
        self.assertTrue(any("deploy" in p for p in patterns),
            f"expected 'deploy' in candidate patterns; got {patterns}")
        # Stopword-only patterns must not appear
        for p in patterns:
            self.assertTrue(analyze._has_content_word(p),
                f"pure-stopword pattern leaked into candidates: {p!r}")

    def test_existing_trigger_group_targeted(self):
        # When the cut tool is already listed under an existing trigger
        # group's tools, action must be "add_keywords_to_trigger" and
        # target_trigger must name that group.
        cut_msgs = ["run the build script", "execute the build script",
                    "kick off the build script", "build script again"]
        triggers = {
            "shell": {"tools": ["terminal"], "keyword_patterns": []},
        }
        stats = self._make_stats(
            expand_only_previews={"terminal": cut_msgs},
            all_previews=cut_msgs + ["random unrelated message"] * 10,
            was_expand_only_count={"terminal": 4},
        )
        with mock.patch.object(analyze, "_load_preset_triggers",
                               return_value=(triggers, "ok")):
            rows = analyze.trigger_keyword_candidates(stats, self._args())
        self.assertTrue(rows)
        self.assertEqual(rows[0]["action"], "add_keywords_to_trigger")
        self.assertEqual(rows[0]["target_trigger"], "shell")

    def test_no_existing_trigger_suggests_new_group(self):
        # No trigger group claims this tool — must suggest creating a
        # new group, named by the tool's underscore prefix.
        cut_msgs = ["please screenshot the page"] * 4
        stats = self._make_stats(
            expand_only_previews={"browser_snapshot": cut_msgs},
            all_previews=cut_msgs + ["chat about lunch"] * 10,
            was_expand_only_count={"browser_snapshot": 4},
        )
        with mock.patch.object(analyze, "_load_preset_triggers",
                               return_value=({}, "ok")):
            rows = analyze.trigger_keyword_candidates(stats, self._args())
        self.assertTrue(rows)
        self.assertEqual(rows[0]["action"], "create_new_trigger")
        self.assertEqual(rows[0]["target_trigger"], "browser")

    def test_existing_keyword_match_excluded(self):
        # If an existing trigger pattern already matches a candidate
        # n-gram, the candidate must be filtered out (no duplicate
        # suggestions).
        import re as _re
        cut_msgs = ["deploy the app now"] * 4
        triggers = {
            "deploy": {
                "tools": ["deploy_tool"],
                "keyword_patterns": [_re.compile(r"deploy the", _re.IGNORECASE)],
            },
        }
        stats = self._make_stats(
            expand_only_previews={"deploy_tool": cut_msgs},
            all_previews=cut_msgs + ["random"] * 10,
            was_expand_only_count={"deploy_tool": 4},
        )
        with mock.patch.object(analyze, "_load_preset_triggers",
                               return_value=(triggers, "ok")):
            rows = analyze.trigger_keyword_candidates(stats, self._args())
        # All cut messages contain "deploy the" which is already covered.
        # Candidates list (if any) must not include "deploy the" or
        # substrings that the existing pattern matches.
        if rows:
            for c in rows[0]["candidates"]:
                self.assertFalse(triggers["deploy"]["keyword_patterns"][0].search(c["pattern"]),
                    f"existing-pattern overlap leaked: {c['pattern']!r}")


class RecommendationRowProtectionTests(unittest.TestCase):
    def _args(self, **overrides):
        defaults = dict(
            min_expansions=2,
            promote_expand_rate=0.5,
            promote_use_rate=0.8,
            expand_round_trip_tokens=1500,
            per_tool_tokens=388,
            unused_carry_turns=10,
            trigger_min_fires=3,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_policy_always_carry_tool_is_not_flagged_for_demote(self):
        stat = analyze.ScopeStats(scope="assistant-a:telegram")
        stat.predictions = 25
        stat.carry_turns = Counter({"mnemosyne_recall": 25})
        rows = analyze.recommendation_rows(
            {"assistant-a:telegram": stat},
            self._args(),
            immutable_always_carry={"mnemosyne_recall"},
        )
        self.assertFalse([r for r in rows if r.get("item") == "mnemosyne_recall"])

    def test_category_recommendation_never_writes_category_into_carry(self):
        """An ``expanded_category`` recommendation carries a toolset/category
        name in ``item`` — never a per-tool identifier — so its proposed learned
        patch must not inject that category string into a per-tool ``carry`` or
        ``expand_only`` list (Phase-6 invariant). The row stays advisory-only."""
        stat = analyze.ScopeStats(scope="assistant-a:telegram")
        stat.predictions = 10
        # Strong promote signal: expands often, always used downstream.
        stat.expansions_by_category = Counter({"filesystem": 8})
        stat.used_expansion_event_ids_by_category["filesystem"] = {
            f"e{i}" for i in range(8)
        }
        rows = analyze.recommendation_rows(
            {"assistant-a:telegram": stat},
            self._args(),
            immutable_always_carry=set(),
        )
        category_rows = [r for r in rows if r.get("kind") == "expanded_category"]
        self.assertTrue(category_rows, "expected an expanded_category row")
        row = category_rows[0]
        self.assertEqual(row["item"], "filesystem")
        # Verify the promote branch actually fired (the branch that previously
        # leaked the category string into the carry patch).
        self.assertEqual(row["action"], "promote_to_carry")
        scopes = row["proposed_learned_patch"]["scopes"]
        for scope_patch in scopes.values():
            self.assertNotIn("filesystem", scope_patch.get("carry", []))
            self.assertNotIn("filesystem", scope_patch.get("expand_only", []))


class ExpandToolsUsedAttributionTests(unittest.TestCase):
    """The ``expand_tools_used`` flag must only fire when expansion
    actually provided the tool — not when the model happens to call a
    tool that was already in the initial allowed set and is also
    coincidentally covered by an active sticky entry.

    The bug this guards against: live telemetry showed 94% of
    ``expand_tools_used`` flags coinciding with ``was_initially_available
    == True``, inflating analyzer promotion signal ~15×.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.profile_home = _make_profile_dir(self.tmp.name, "assistant-a")
        self._env_patch = mock.patch.dict(
            os.environ,
            {"HERMES_HOME": self.profile_home, "HERMES_SESSION_KEY": ""},
            clear=False,
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

        self._original_config = dict(plugin._CONFIG)
        plugin._CONFIG["enabled"] = True
        plugin._CONFIG["log"] = True
        plugin._CONFIG["agent"] = "assistant-a"
        self.addCleanup(self._restore_config)

        # Clear the sticky table so prior tests' state can't leak in.
        plugin._STICKY_BY_KEY.clear()
        plugin._POLICY_TURN_BY_SCOPE.clear()
        plugin._PREDICTION_CV.set(None)

    def _restore_config(self) -> None:
        plugin._CONFIG.clear()
        plugin._CONFIG.update(self._original_config)
        plugin._STICKY_BY_KEY.clear()
        plugin._POLICY_TURN_BY_SCOPE.clear()

    def _seed_sticky(self, sticky_key: str, category: str, tools: list[str]) -> None:
        """Insert a live sticky entry by hand so the test doesn't depend
        on running a full expand_tools round-trip."""
        plugin._STICKY_BY_KEY[sticky_key] = {
            category: {
                "tools": {str(t) for t in tools},
                "remaining_turns": 3,
                "updated_at": 0.0,
                "expanded_at_turn": 0,
                "prediction_id": "pred-prior",
            }
        }

    def _set_prediction_state(self, *, initial_allowed: list[str], sticky_key: str) -> None:
        plugin._PREDICTION_CV.set({
            "prediction_id": "pred-current",
            "agent": "assistant-a",
            "platform": "telegram",
            "scope": "assistant-a:telegram",
            "sticky_key": sticky_key,
            "session_id": REAL_KEY_TELEGRAM,
            "initial_active_tools": initial_allowed,
            "expand_only_tools": [],
            "expansions": set(),
            "pending_expansion": None,
            "ceiling_tools": list(initial_allowed),
        })

    def _latest_row(self) -> dict:
        tool_calls_path = Path(self.profile_home) / "state" / "tool-belt" / "tool_calls.jsonl"
        rows = [json.loads(l) for l in tool_calls_path.read_text().splitlines() if l]
        self.assertTrue(rows, "expected at least one tool-call row")
        return rows[-1]

    def test_already_available_tool_does_not_credit_sticky_expansion(self):
        # The bug case: terminal is already in the initial allowed set
        # AND covered by an active sticky entry. The flag must NOT fire.
        sticky_key = plugin._sticky_key_for_session(REAL_KEY_TELEGRAM)
        self._seed_sticky(sticky_key, "terminal", ["terminal"])
        self._set_prediction_state(initial_allowed=["terminal", "memory"], sticky_key=sticky_key)

        plugin._on_post_tool_call(
            tool_name="terminal", args={}, result="ok", task_id="t1",
            session_id=REAL_KEY_TELEGRAM,
        )
        row = self._latest_row()
        self.assertTrue(row.get("was_initially_active"))
        self.assertNotEqual(row.get("expansion_provided_access"), True,
            "tool already in initial allowed set must not be credited as expansion-driven")

    def test_expand_only_tool_with_active_sticky_credits_expansion(self):
        # The legitimate case: terminal was expand-only in the initial active
        # set; sticky residency from a prior expansion is what made it
        # callable. The flag must fire.
        sticky_key = plugin._sticky_key_for_session(REAL_KEY_TELEGRAM)
        self._seed_sticky(sticky_key, "terminal", ["terminal"])
        self._set_prediction_state(initial_allowed=["memory"], sticky_key=sticky_key)
        # Mark the tool expand-only so the canonical flag reflects reality (the predictor
        # would have set this when narrowing).
        state = plugin._PREDICTION_CV.get()
        state["expand_only_tools"] = ["terminal"]
        plugin._PREDICTION_CV.set(state)

        plugin._on_post_tool_call(
            tool_name="terminal", args={}, result="ok", task_id="t1",
            session_id=REAL_KEY_TELEGRAM,
        )
        row = self._latest_row()
        self.assertFalse(row.get("was_initially_active"))
        self.assertTrue(row.get("expansion_provided_access"),
            "expand-only tool reachable via sticky must be credited")
        self.assertEqual(row.get("expand_category"), "terminal")

    def test_sticky_carried_tool_in_initial_allowed_still_credits_expansion(self):
        # Regression: in production, sticky_tools get merged into the
        # narrowed active set BEFORE filtering, so a sticky-carried tool
        # ends up in initial_active_tools. Without a pre-sticky baseline,
        # the credit decision saw was_initially_available=True and skipped
        # the sticky-expansion branch, zeroing out expand_tools_used.
        sticky_key = plugin._sticky_key_for_session(REAL_KEY_TELEGRAM)
        self._seed_sticky(sticky_key, "terminal", ["terminal"])
        # Terminal IS in initial_allowed (sticky merged in upstream)…
        self._set_prediction_state(initial_allowed=["terminal", "memory"], sticky_key=sticky_key)
        # …but the pre-sticky baseline does NOT include it.
        state = plugin._PREDICTION_CV.get()
        state["baseline_active_tools"] = ["memory"]
        plugin._PREDICTION_CV.set(state)

        plugin._on_post_tool_call(
            tool_name="terminal", args={}, result="ok", task_id="t1",
            session_id=REAL_KEY_TELEGRAM,
        )
        row = self._latest_row()
        self.assertFalse(row.get("was_initially_active"),
            "baseline (pre-sticky) determines was_initially_available")
        self.assertTrue(row.get("expansion_provided_access"),
            "sticky-carried tool must be credited as expansion-driven")
        self.assertEqual(row.get("expand_category"), "terminal")

    def test_pending_expansion_skipped_when_already_available(self):
        # Same-turn expansion (pending_expansion) for a tool that was
        # already always-on: the round-trip happened but provided
        # nothing. after_expand_tools still records the round-trip, but
        # expand_tools_used must stay false.
        sticky_key = plugin._sticky_key_for_session(REAL_KEY_TELEGRAM)
        self._set_prediction_state(initial_allowed=["terminal"], sticky_key=sticky_key)
        state = plugin._PREDICTION_CV.get()
        state["pending_expansion"] = {
            "category": "terminal",
            "resolved_tools": ["terminal"],
            "tools_added": [],
        }
        plugin._PREDICTION_CV.set(state)

        plugin._on_post_tool_call(
            tool_name="terminal", args={}, result="ok", task_id="t1",
            session_id=REAL_KEY_TELEGRAM,
        )
        row = self._latest_row()
        self.assertTrue(row.get("was_initially_active"))
        self.assertTrue(row.get("after_expand_tools"),
            "round-trip happened — should still record after_expand_tools")
        self.assertNotEqual(row.get("expansion_provided_access"), True,
            "round-trip provided nothing new — must not be credited as expansion-driven")


class BuildApiKwargsSnapshotTests(unittest.TestCase):
    """``initial_active_tools`` must be a *snapshot* captured on the first
    ``_build_api_kwargs`` call of a prediction, NOT a moving window that
    follows post-expansion state. If it moves, expansion-driven tool calls
    later in the same turn look like they were initially available, which
    zeros out the expansion-success signal the analyzer (and apply.py)
    depends on.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.profile_home = _make_profile_dir(self.tmp.name, "assistant-a")
        self._env_patch = mock.patch.dict(
            os.environ,
            {"HERMES_HOME": self.profile_home, "HERMES_SESSION_KEY": ""},
            clear=False,
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)
        self._original_config = dict(plugin._CONFIG)
        plugin._CONFIG["enabled"] = True
        plugin._CONFIG["log"] = False  # disable prediction-log writes
        plugin._CONFIG["agent"] = "assistant-a"
        self.addCleanup(self._restore_config)
        plugin._PREDICTION_CV.set(None)

    def _restore_config(self) -> None:
        plugin._CONFIG.clear()
        plugin._CONFIG.update(self._original_config)

    def _run_wrapper_with_tools(self, tool_names: list[str]) -> dict:
        tools = [{"name": n} for n in tool_names]
        original = lambda self_, msgs: {"tools": list(tools)}
        wrapped = plugin._wrap_build_api_kwargs(original)
        return wrapped(SimpleNamespace(), [])

    def test_initial_allowed_tools_snapshot_survives_post_expansion_call(self):
        # Predictor narrowed to {memory, send_message}; terminal was cut.
        plugin._PREDICTION_CV.set({
            "prediction_id": "pred-1",
            "active_tool_names": ["memory", "send_message"],
            "baseline_active_tools": ["memory", "send_message"],
            "resolved_always_carry": [],
            "resolved_carry": ["memory", "send_message"],
            # Full-start: terminal/file_read are cut via the demoted loadout.
            "resolved_demoted": ["terminal", "file_read"],
            "triggered_tools": [],
            "expansions": set(),
            "agent": "assistant-a", "platform": "telegram", "scope": "assistant-a:telegram",
            "sticky_key": "k", "session_id": REAL_KEY_TELEGRAM,
            "sticky_tools": [], "sticky_categories": [], "sticky_remaining_turns": {},
        })

        # First call: model sees the narrowed view.
        kwargs1 = self._run_wrapper_with_tools(["memory", "send_message", "terminal", "file_read"])
        state = plugin._PREDICTION_CV.get()
        snapshot = list(state["initial_active_tools"])
        self.assertEqual(sorted(snapshot), ["memory", "send_message"])
        self.assertEqual(sorted(_tool_names_in(kwargs1["tools"])), ["memory", "send_message"])

        # Model called expand_tools(terminal); expansions now non-empty.
        state["expansions"] = {"terminal"}
        plugin._PREDICTION_CV.set(state)

        # Second call within the same prediction.
        kwargs2 = self._run_wrapper_with_tools(["memory", "send_message", "terminal", "file_read"])
        state = plugin._PREDICTION_CV.get()

        # The SDK view widened (terminal is now allowed)…
        self.assertIn("terminal", _tool_names_in(kwargs2["tools"]))
        # …but the snapshot must NOT have moved.
        self.assertEqual(
            sorted(state["initial_active_tools"]), sorted(snapshot),
            "initial_active_tools must be a one-time snapshot per prediction",
        )

    def test_end_to_end_recovered_tool_credits_expand_tools_used(self):
        # Full flow: prediction with baseline excluding terminal, sticky
        # carries terminal forward from a prior turn. The model calls
        # terminal. The row must be credited as expansion-driven.
        sticky_key = plugin._sticky_key_for_session(REAL_KEY_TELEGRAM)
        plugin._STICKY_BY_KEY[sticky_key] = {
            "terminal": {
                "tools": {"terminal"},
                "remaining_turns": 3,
                "updated_at": 0.0,
                "expanded_at_turn": 0,
                "prediction_id": "pred-prior",
            }
        }
        self.addCleanup(plugin._STICKY_BY_KEY.clear)

        plugin._CONFIG["log"] = True  # need writes to verify the row
        plugin._PREDICTION_CV.set({
            "prediction_id": "pred-current",
            "active_tool_names": ["memory", "terminal"],  # post-sticky merge
            "baseline_active_tools": ["memory"],           # pre-sticky
            "resolved_always_carry": [],
            "resolved_carry": ["memory"],
            # terminal was demoted — that's why sticky recovery matters here.
            "resolved_demoted": ["terminal"],
            "triggered_tools": [],
            "expansions": set(),
            "agent": "assistant-a", "platform": "telegram", "scope": "assistant-a:telegram",
            "sticky_key": sticky_key, "session_id": REAL_KEY_TELEGRAM,
            "sticky_tools": ["terminal"], "sticky_categories": ["terminal"],
            "sticky_remaining_turns": {"terminal": 3},
            "expand_only_tools": ["terminal"], "pending_expansion": None,
        })

        # First _build_api_kwargs call — establishes initial_active_tools.
        self._run_wrapper_with_tools(["memory", "terminal"])

        # Model calls terminal.
        plugin._on_post_tool_call(
            tool_name="terminal", args={}, result="ok",
            task_id="t1", session_id=REAL_KEY_TELEGRAM,
        )

        tool_calls_path = Path(self.profile_home) / "state" / "tool-belt" / "tool_calls.jsonl"
        rows = [json.loads(l) for l in tool_calls_path.read_text().splitlines() if l]
        row = rows[-1]
        self.assertFalse(row.get("was_initially_active"),
            "baseline excluded terminal — it was only callable via sticky")
        self.assertTrue(row.get("expansion_provided_access"),
            "sticky-recovered tool must be credited as expansion-driven")

    def test_already_available_tool_never_credits_expansion(self):
        # Tool present in baseline (no expansion involved) must not be
        # credited, even with sticky residency coincidentally covering it.
        sticky_key = plugin._sticky_key_for_session(REAL_KEY_TELEGRAM)
        plugin._STICKY_BY_KEY[sticky_key] = {
            "terminal": {
                "tools": {"terminal"},
                "remaining_turns": 3,
                "updated_at": 0.0,
                "expanded_at_turn": 0,
                "prediction_id": "pred-prior",
            }
        }
        self.addCleanup(plugin._STICKY_BY_KEY.clear)

        plugin._CONFIG["log"] = True
        plugin._PREDICTION_CV.set({
            "prediction_id": "pred-current",
            "active_tool_names": ["memory", "terminal"],
            "baseline_active_tools": ["memory", "terminal"],  # terminal IS in baseline
            "resolved_always_carry": [],
            "resolved_carry": ["memory", "terminal"],
            "triggered_tools": [],
            "expansions": set(),
            "agent": "assistant-a", "platform": "telegram", "scope": "assistant-a:telegram",
            "sticky_key": sticky_key, "session_id": REAL_KEY_TELEGRAM,
            "sticky_tools": ["terminal"], "sticky_categories": ["terminal"],
            "sticky_remaining_turns": {"terminal": 3},
            "expand_only_tools": [], "pending_expansion": None,
        })

        self._run_wrapper_with_tools(["memory", "terminal"])
        plugin._on_post_tool_call(
            tool_name="terminal", args={}, result="ok",
            task_id="t1", session_id=REAL_KEY_TELEGRAM,
        )

        tool_calls_path = Path(self.profile_home) / "state" / "tool-belt" / "tool_calls.jsonl"
        rows = [json.loads(l) for l in tool_calls_path.read_text().splitlines() if l]
        row = rows[-1]
        self.assertTrue(row.get("was_initially_active"))
        self.assertNotEqual(row.get("expansion_provided_access"), True,
            "tool in baseline must never be credited as expansion-driven")


def _tool_names_in(tools: list[dict]) -> list[str]:
    out = []
    for t in tools:
        n = t.get("name") if isinstance(t, dict) else None
        if n:
            out.append(n)
    return out


if __name__ == "__main__":
    unittest.main()
