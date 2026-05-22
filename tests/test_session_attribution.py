"""Focused tests for the telemetry-attribution fixes.

Covers:
  1. Prediction ``session_id`` no longer blank for gateway-dispatched
     events when ``session_store`` is reachable.
  2. ``_should_bypass`` activates once ``session_id`` exists (it was a
     no-op when the field came through empty).
  3. Real Hermes session-key shape parsing — ``agent:main:{platform}:…``,
     where position 1 is the *literal* string ``"main"``, not the agent
     name. Recovering ``bernard:telegram`` from a real gateway key
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
if "dynamic_tools_plugin" not in sys.modules:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent.parent))
    sys.path.insert(0, str(here.parent))
    from tests import conftest  # noqa: F401 — side-effect: register package

plugin = sys.modules["dynamic_tools_plugin"]
logger_io = importlib.import_module("dynamic_tools_plugin.logger_io")
analyze = importlib.import_module("dynamic_tools_plugin.analyze")


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

    def test_parse_session_key_scope_uses_profile_agent_not_parts_1(self):
        # The bug to regress: parts[1] is "main", and the old code would
        # surface ("main", "telegram", "main:telegram"). With a profile
        # in scope, we should now see the profile agent instead.
        plugin._CONFIG["agent"] = "bernard"
        agent, platform, scope = plugin._parse_session_key_scope(REAL_KEY_TELEGRAM)
        self.assertEqual((agent, platform, scope), ("bernard", "telegram", "bernard:telegram"))

    def test_parse_session_key_scope_blank_without_profile(self):
        # Without a profile-derived agent, the helper must return blanks
        # — falling back to "main:telegram" would be a regression.
        with mock.patch.dict(os.environ, {}, clear=True):
            agent, platform, scope = plugin._parse_session_key_scope(REAL_KEY_TELEGRAM)
        self.assertEqual((agent, platform, scope), ("", "", ""))

    def test_profile_agent_from_hermes_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile_dir = _make_profile_dir(tmp, "sue")
            with mock.patch.dict(os.environ, {"HERMES_HOME": profile_dir}, clear=False):
                # Ensure _CONFIG['agent'] doesn't shadow HERMES_HOME for this case.
                plugin._CONFIG["agent"] = ""
                self.assertEqual(plugin._profile_agent_name(), "sue")


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
        plugin._CONFIG["agent"] = "bernard"
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
        self.assertEqual(state["agent"], "bernard")
        self.assertEqual(state["platform"], "telegram")
        self.assertEqual(state["scope"], "bernard:telegram")
        predictions = (Path(self.tmp.name) / "state" / "dynamic-tools" / "predictions.jsonl")
        plugin._maybe_log_prediction(state, ceiling=[], narrowed=[])
        rows = [json.loads(line) for line in predictions.read_text().splitlines() if line]
        self.assertTrue(rows, "expected at least one prediction row")
        self.assertEqual(rows[-1]["session_id"], REAL_KEY_TELEGRAM)
        self.assertEqual(rows[-1]["scope"], "bernard:telegram")


class BypassEligibilityTests(unittest.TestCase):
    def test_should_bypass_false_when_session_id_blank(self):
        self.assertFalse(plugin._should_bypass("bernard:telegram", ""))

    def test_should_bypass_can_fire_with_real_session_id(self):
        original_rate = plugin._CONFIG.get("bypass_rate")
        plugin._CONFIG["bypass_rate"] = 1.0
        try:
            self.assertTrue(plugin._should_bypass("bernard:telegram", REAL_KEY_TELEGRAM))
        finally:
            plugin._CONFIG["bypass_rate"] = original_rate


class PostToolCallAttributionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Make HERMES_HOME a real profile dir so _profile_agent_name()
        # resolves to "bernard" — mirrors the live ``hermes -p bernard
        # gateway`` shape.
        self.profile_home = _make_profile_dir(self.tmp.name, "bernard")
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
        self.assertEqual(agent, "bernard")
        self.assertEqual(platform, "telegram")
        self.assertEqual(scope, "bernard:telegram")

    def test_recover_from_whatsapp_session_key(self):
        # Confirms the parser works across platforms, not just telegram.
        agent, platform, scope = plugin._recover_attribution_without_state(
            REAL_KEY_WHATSAPP, kwargs={}
        )
        self.assertEqual((agent, platform, scope), ("bernard", "whatsapp", "bernard:whatsapp"))

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

    def test_context_returns_blank_when_agent_unresolved(self):
        # Regression: a config without `agent:` and HERMES_HOME pointing
        # at the Hermes root (not a profiles/<name> subdir) used to be
        # silently labelled "default:telegram", masking the missing
        # config line for four days in production. Blank attribution is
        # the honest signal — analyzer's normalize_scope buckets these
        # as "unknown" so the regression surfaces in reports.
        with mock.patch.dict(
            os.environ,
            {"HERMES_HOME": "/Users/macmini/.hermes", "HERMES_PROFILE": "",
             "HERMES_SESSION_KEY": REAL_KEY_TELEGRAM},
            clear=False,
        ):
            plugin._CONFIG["agent"] = ""
            agent, platform, scope = plugin._agent_platform_from_context(event=None, kwargs={})
        self.assertEqual((agent, platform, scope), ("", "", ""),
            "blank agent + real platform must NOT synthesize default:telegram")

    def test_context_resolves_when_config_agent_set(self):
        # The supported fix path: explicit `agent: bernard` in plugin
        # config recovers attribution even when HERMES_HOME is the root.
        with mock.patch.dict(
            os.environ,
            {"HERMES_HOME": "/Users/macmini/.hermes", "HERMES_PROFILE": "",
             "HERMES_SESSION_KEY": REAL_KEY_TELEGRAM},
            clear=False,
        ):
            plugin._CONFIG["agent"] = "bernard"
            agent, platform, scope = plugin._agent_platform_from_context(event=None, kwargs={})
        self.assertEqual((agent, platform, scope), ("bernard", "telegram", "bernard:telegram"))

    def test_post_tool_call_logs_recovered_attribution(self):
        plugin._on_post_tool_call(
            tool_name="terminal",
            args={"command": "ls"},
            result="ok",
            task_id="t1",
            session_id=REAL_KEY_TELEGRAM,
        )
        tool_calls_path = Path(self.profile_home) / "state" / "dynamic-tools" / "tool_calls.jsonl"
        rows = [json.loads(l) for l in tool_calls_path.read_text().splitlines() if l]
        self.assertTrue(rows, "expected one tool-call row")
        row = rows[-1]
        self.assertEqual(row["agent"], "bernard")
        self.assertEqual(row["platform"], "telegram")
        self.assertEqual(row["scope"], "bernard:telegram")


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

    def test_also_evicts_legacy_uuid_keyed_state(self):
        # Rows written under the uuid form before the fix should also
        # be cleaned up — covers in-flight upgrade.
        legacy_sticky = self._seed_state(self.AGENT_SESSION_ID)
        with mock.patch.dict(os.environ, {"HERMES_SESSION_KEY": ""}, clear=False):
            plugin._on_session_end(session_id=self.AGENT_SESSION_ID)
        self.assertNotIn(legacy_sticky, plugin._STICKY_BY_KEY)
        self.assertNotIn(self.AGENT_SESSION_ID, plugin._PRIOR_MESSAGES_BY_SESSION)


class AnalyzerExcludesDegradedModeTests(unittest.TestCase):
    def test_load_preset_excludes_returns_status_no_yaml(self):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("yaml unavailable")
            return real_import(name, *args, **kwargs)

        with mock.patch.dict(sys.modules, {"yaml": None}):
            with mock.patch.object(builtins, "__import__", fake_import):
                out, status = analyze._load_preset_excludes(Path(plugin.__file__).parent)
        self.assertEqual(out, {})
        self.assertEqual(status, "no_yaml")
        self.assertIn("PyYAML is not installed", analyze._EXCLUDES_STATUS_MESSAGE["no_yaml"])

    def test_load_preset_excludes_returns_status_no_policy(self):
        out, status = analyze._load_preset_excludes(Path(tempfile.mkdtemp()))
        self.assertEqual(out, {})
        self.assertEqual(status, "no_policy")

    def test_dampener_candidates_emits_warning_when_degraded(self):
        stat = analyze.ScopeStats(scope="bernard:telegram")
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
                rows = analyze.dampener_candidates({"bernard:telegram": stat}, args)
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

    When ``session_id`` is blank (pre-fix telemetry), the analyzer must
    fall back to same-prediction behavior to avoid cross-session leakage.
    """

    def _row_prediction(self, *, pid, sid, trigger, tools_for_trigger, ts):
        return {
            "ts": ts,
            "prediction_id": pid,
            "session_id": sid,
            "scope": "bernard:telegram",
            "agent": "bernard",
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
        scope = stats["bernard:telegram"]
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
        scope = stats["bernard:telegram"]
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
        scope = stats["bernard:telegram"]
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
        scope = stats["bernard:telegram"]
        self.assertEqual(scope.trigger_false_positives.get("shell", 0), 1,
            "blank session_id must fall back to same-prediction classification")


class TriggerKeywordSuggesterTests(unittest.TestCase):
    """The trigger-keyword suggester mines cut-tool message previews
    for candidate keywords that would have fired a trigger to cover the
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
            harvest_min_cuts=2,
            expand_round_trip_tokens=1500,
            per_tool_tokens=388,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _make_stats(self, *, scope="bernard:telegram", cut_previews,
                    all_previews, was_cut_count):
        from collections import Counter
        stat = analyze.ScopeStats(scope=scope)
        stat.harvest_predictions = len(all_previews)
        stat.harvest_all_previews = list(all_previews)
        for tool, previews in cut_previews.items():
            stat.harvest_cut_previews[tool] = list(previews)
            stat.harvest_was_cut[tool] = was_cut_count.get(tool, len(previews))
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
            cut_previews={"deploy_tool": cut_msgs},
            all_previews=cut_msgs + noise,
            was_cut_count={"deploy_tool": 5},
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
            cut_previews={"terminal": cut_msgs},
            all_previews=cut_msgs + ["random unrelated message"] * 10,
            was_cut_count={"terminal": 4},
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
            cut_previews={"browser_snapshot": cut_msgs},
            all_previews=cut_msgs + ["chat about lunch"] * 10,
            was_cut_count={"browser_snapshot": 4},
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
            cut_previews={"deploy_tool": cut_msgs},
            all_previews=cut_msgs + ["random"] * 10,
            was_cut_count={"deploy_tool": 4},
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


class HarvestRecommendationTests(unittest.TestCase):
    """The analyzer's harvest-aware recommendation generator should
    surface per-tool promotion candidates from was_cut signal, with
    clear actions for the three regimes (net-positive promote,
    frequent-but-net-negative broaden, sparse keep)."""

    def _harvest_row(self, *, pid, scope, tool=None, was_cut=False):
        if tool is None:
            return {
                "prediction_id": pid,
                "session_id": f"harvest:{pid[:4]}",
                "scope": scope,
                "agent": scope.split(":")[0],
                "platform": scope.split(":")[1],
                "policy_source": "harvest",
                "message_preview": f"msg-{pid}",
                "ceiling_count": 30, "narrowed_count": 15,
                "ceiling_tokens": 100, "narrowed_tokens": 50,
            }
        return {
            "prediction_id": pid,
            "session_id": f"harvest:{pid[:4]}",
            "scope": scope,
            "tool_name": tool,
            "was_cut": was_cut,
            "policy_source": "harvest",
        }

    def _args(self, **overrides):
        defaults = dict(
            expand_round_trip_tokens=1500,
            per_tool_tokens=388,
            harvest_min_cuts=3,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_net_positive_surfaces_as_promote(self):
        # 100 predictions, 50 was_cut for tool X → cost = 50*1500 = 75k
        # carry = 100*388 = 38.8k → net +36.2k → promote
        preds = [self._harvest_row(pid=f"p{i:03d}", scope="bernard:telegram") for i in range(100)]
        calls = [self._harvest_row(pid=f"p{i:03d}", scope="bernard:telegram",
                                   tool="terminal", was_cut=True) for i in range(50)]
        stats = analyze.collect_stats(preds, calls)
        rows = analyze.harvest_recommendation_rows(stats, self._args())
        terminal_rec = next((r for r in rows if r["item"] == "terminal"), None)
        self.assertIsNotNone(terminal_rec)
        self.assertEqual(terminal_rec["action"], "promote_always_on")
        self.assertGreater(terminal_rec["metrics"]["net_savings_tokens"], 0)

    def test_frequent_but_net_negative_surfaces_as_broaden(self):
        # 1000 predictions (carry = 388k), 150 was_cut for X (rt = 225k)
        # → net -163k but cut_rate = 15% (above 10% threshold) → broaden
        preds = [self._harvest_row(pid=f"p{i:04d}", scope="bernard:telegram") for i in range(1000)]
        calls = [self._harvest_row(pid=f"p{i:04d}", scope="bernard:telegram",
                                   tool="patch", was_cut=True) for i in range(150)]
        stats = analyze.collect_stats(preds, calls)
        rows = analyze.harvest_recommendation_rows(stats, self._args())
        patch_rec = next((r for r in rows if r["item"] == "patch"), None)
        self.assertIsNotNone(patch_rec)
        self.assertEqual(patch_rec["action"], "broaden_trigger_recall")
        self.assertLess(patch_rec["metrics"]["net_savings_tokens"], 0)

    def test_sparse_cuts_keep_gated(self):
        # 1000 predictions, only 5 was_cut → cut_rate 0.5%, net very
        # negative → keep_gated
        preds = [self._harvest_row(pid=f"p{i:04d}", scope="bernard:telegram") for i in range(1000)]
        calls = [self._harvest_row(pid=f"p{i:04d}", scope="bernard:telegram",
                                   tool="browser_vision", was_cut=True) for i in range(5)]
        stats = analyze.collect_stats(preds, calls)
        rows = analyze.harvest_recommendation_rows(stats, self._args())
        rec = next((r for r in rows if r["item"] == "browser_vision"), None)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["action"], "keep_gated")

    def test_below_min_cuts_does_not_surface(self):
        # 1 was_cut < harvest_min_cuts=3 → no rec emitted
        preds = [self._harvest_row(pid="p001", scope="bernard:telegram")]
        calls = [self._harvest_row(pid="p001", scope="bernard:telegram",
                                   tool="cronjob", was_cut=True)]
        stats = analyze.collect_stats(preds, calls)
        rows = analyze.harvest_recommendation_rows(stats, self._args())
        self.assertFalse([r for r in rows if r["item"] == "cronjob"])

    def test_no_harvest_rows_no_output(self):
        # Live data only (no policy_source=harvest) → empty output
        preds = [{
            "prediction_id": "p1", "session_id": "live-s1", "scope": "bernard:telegram",
            "policy_source": "preset", "message_preview": "x",
            "ceiling_count": 1, "narrowed_count": 1, "ceiling_tokens": 1, "narrowed_tokens": 1,
        }]
        calls = [{"prediction_id": "p1", "session_id": "live-s1", "scope": "bernard:telegram",
                  "tool_name": "terminal", "was_cut": True, "policy_source": "preset"}]
        stats = analyze.collect_stats(preds, calls)
        rows = analyze.harvest_recommendation_rows(stats, self._args())
        self.assertFalse(rows, "live-only telemetry must not produce harvest recs")


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
        self.profile_home = _make_profile_dir(self.tmp.name, "bernard")
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
        plugin._CONFIG["agent"] = "bernard"
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
            "agent": "bernard",
            "platform": "telegram",
            "scope": "bernard:telegram",
            "sticky_key": sticky_key,
            "session_id": REAL_KEY_TELEGRAM,
            "initial_allowed_tools": initial_allowed,
            "cut_tools": [],
            "expansions": set(),
            "pending_expansion": None,
            "ceiling_tools": list(initial_allowed),
            "unknown_kept_tools": [],
        })

    def _latest_row(self) -> dict:
        tool_calls_path = Path(self.profile_home) / "state" / "dynamic-tools" / "tool_calls.jsonl"
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
        self.assertTrue(row.get("was_initially_available"))
        self.assertNotEqual(row.get("expand_tools_used"), True,
            "tool already in initial allowed set must not be credited as expansion-driven")

    def test_cut_tool_with_active_sticky_credits_expansion(self):
        # The legitimate case: terminal was cut from the initial allowed
        # set; sticky residency from a prior expansion is what made it
        # callable. The flag must fire.
        sticky_key = plugin._sticky_key_for_session(REAL_KEY_TELEGRAM)
        self._seed_sticky(sticky_key, "terminal", ["terminal"])
        self._set_prediction_state(initial_allowed=["memory"], sticky_key=sticky_key)
        # Mark the tool as cut so was_cut reflects reality (the predictor
        # would have set this when narrowing).
        state = plugin._PREDICTION_CV.get()
        state["cut_tools"] = ["terminal"]
        plugin._PREDICTION_CV.set(state)

        plugin._on_post_tool_call(
            tool_name="terminal", args={}, result="ok", task_id="t1",
            session_id=REAL_KEY_TELEGRAM,
        )
        row = self._latest_row()
        self.assertFalse(row.get("was_initially_available"))
        self.assertTrue(row.get("expand_tools_used"),
            "tool cut from initial allowed set but reachable via sticky must be credited")
        self.assertEqual(row.get("expand_category"), "terminal")

    def test_sticky_carried_tool_in_initial_allowed_still_credits_expansion(self):
        # Regression: in production, sticky_tools get merged into the
        # narrowed allowed set BEFORE filtering, so a sticky-carried tool
        # ends up in initial_allowed_tools. Without a pre-sticky baseline,
        # the credit decision saw was_initially_available=True and skipped
        # the sticky-expansion branch, zeroing out expand_tools_used.
        sticky_key = plugin._sticky_key_for_session(REAL_KEY_TELEGRAM)
        self._seed_sticky(sticky_key, "terminal", ["terminal"])
        # Terminal IS in initial_allowed (sticky merged in upstream)…
        self._set_prediction_state(initial_allowed=["terminal", "memory"], sticky_key=sticky_key)
        # …but the pre-sticky baseline does NOT include it.
        state = plugin._PREDICTION_CV.get()
        state["baseline_allowed_tools"] = ["memory"]
        plugin._PREDICTION_CV.set(state)

        plugin._on_post_tool_call(
            tool_name="terminal", args={}, result="ok", task_id="t1",
            session_id=REAL_KEY_TELEGRAM,
        )
        row = self._latest_row()
        self.assertFalse(row.get("was_initially_available"),
            "baseline (pre-sticky) determines was_initially_available")
        self.assertTrue(row.get("expand_tools_used"),
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
        self.assertTrue(row.get("was_initially_available"))
        self.assertTrue(row.get("after_expand_tools"),
            "round-trip happened — should still record after_expand_tools")
        self.assertNotEqual(row.get("expand_tools_used"), True,
            "round-trip provided nothing new — must not be credited as expansion-driven")


class BuildApiKwargsSnapshotTests(unittest.TestCase):
    """``initial_allowed_tools`` must be a *snapshot* captured on the first
    ``_build_api_kwargs`` call of a prediction, NOT a moving window that
    follows post-expansion state. If it moves, expansion-driven tool calls
    later in the same turn look like they were initially available, which
    zeros out the expansion-success signal the analyzer (and apply.py)
    depends on.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.profile_home = _make_profile_dir(self.tmp.name, "bernard")
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
        plugin._CONFIG["agent"] = "bernard"
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
            "allowed_tool_names": ["memory", "send_message"],
            "baseline_allowed_tools": ["memory", "send_message"],
            "known_tool_names": {"memory", "send_message", "terminal", "file_read"},
            "expansions": set(),
            "agent": "bernard", "platform": "telegram", "scope": "bernard:telegram",
            "sticky_key": "k", "session_id": REAL_KEY_TELEGRAM,
            "sticky_tools": [], "sticky_categories": [], "sticky_remaining_turns": {},
        })

        # First call: model sees the narrowed view.
        kwargs1 = self._run_wrapper_with_tools(["memory", "send_message", "terminal", "file_read"])
        state = plugin._PREDICTION_CV.get()
        snapshot = list(state["initial_allowed_tools"])
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
            sorted(state["initial_allowed_tools"]), sorted(snapshot),
            "initial_allowed_tools must be a one-time snapshot per prediction",
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
            "allowed_tool_names": ["memory", "terminal"],  # post-sticky merge
            "baseline_allowed_tools": ["memory"],           # pre-sticky
            "known_tool_names": {"memory", "terminal"},
            "expansions": set(),
            "agent": "bernard", "platform": "telegram", "scope": "bernard:telegram",
            "sticky_key": sticky_key, "session_id": REAL_KEY_TELEGRAM,
            "sticky_tools": ["terminal"], "sticky_categories": ["terminal"],
            "sticky_remaining_turns": {"terminal": 3},
            "cut_tools": ["terminal"], "pending_expansion": None,
        })

        # First _build_api_kwargs call — establishes initial_allowed_tools.
        self._run_wrapper_with_tools(["memory", "terminal"])

        # Model calls terminal.
        plugin._on_post_tool_call(
            tool_name="terminal", args={}, result="ok",
            task_id="t1", session_id=REAL_KEY_TELEGRAM,
        )

        tool_calls_path = Path(self.profile_home) / "state" / "dynamic-tools" / "tool_calls.jsonl"
        rows = [json.loads(l) for l in tool_calls_path.read_text().splitlines() if l]
        row = rows[-1]
        self.assertFalse(row.get("was_initially_available"),
            "baseline excluded terminal — it was only callable via sticky")
        self.assertTrue(row.get("expand_tools_used"),
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
            "allowed_tool_names": ["memory", "terminal"],
            "baseline_allowed_tools": ["memory", "terminal"],  # terminal IS in baseline
            "known_tool_names": {"memory", "terminal"},
            "expansions": set(),
            "agent": "bernard", "platform": "telegram", "scope": "bernard:telegram",
            "sticky_key": sticky_key, "session_id": REAL_KEY_TELEGRAM,
            "sticky_tools": ["terminal"], "sticky_categories": ["terminal"],
            "sticky_remaining_turns": {"terminal": 3},
            "cut_tools": [], "pending_expansion": None,
        })

        self._run_wrapper_with_tools(["memory", "terminal"])
        plugin._on_post_tool_call(
            tool_name="terminal", args={}, result="ok",
            task_id="t1", session_id=REAL_KEY_TELEGRAM,
        )

        tool_calls_path = Path(self.profile_home) / "state" / "dynamic-tools" / "tool_calls.jsonl"
        rows = [json.loads(l) for l in tool_calls_path.read_text().splitlines() if l]
        row = rows[-1]
        self.assertTrue(row.get("was_initially_available"))
        self.assertNotEqual(row.get("expand_tools_used"), True,
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
