"""Focused tests for telemetry attribution.

Covers:
  1. Prediction ``session_id`` is populated for gateway-dispatched events
     when ``session_store`` is reachable.
  2. ``_should_bypass`` — the blank-session guard and the deterministic
     (scope, session_id) cohort hash.
  3. Real Hermes session-key shape parsing — ``agent:main:{platform}:…``,
     where position 1 is the *literal* string ``"main"``, not the agent
     name. Recovering ``assistant-a:telegram`` from a real gateway key
     requires combining the canonical platform with a profile-derived
     agent.
  4. Blank attribution preserved for genuinely untrackable rows.
  5. ``_on_session_reset`` evicts sticky/lookback state keyed by the
     canonical session key, even when Hermes hands us the AIAgent's
     uuid-style ``session_id``.
  6. Analyzer degraded modes, trigger TP/FP windowing and keyword mining.
  7. The tool-call row's credit rules: the primary-dispatch gate, the
     ``initial_active_tools`` snapshot, and the credit baseline that keeps
     ``expansion_provided_access`` honest.
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

    def test_platform_parsed_only_from_the_canonical_agent_main_shape(self):
        # Position 1 is the LITERAL "main" in every gateway key. Accepting an
        # arbitrary second segment would make the helper read position 2 of a
        # key it does not understand and stamp a bogus platform on the row —
        # so the "main" check is asserted directly, not just implied by the
        # happy path.
        cases = [
            (REAL_KEY_TELEGRAM, "telegram"),
            (REAL_KEY_WHATSAPP, "whatsapp"),
            # AIAgent.session_id has no "agent:" prefix.
            ("20260516_123456_deadbeef", ""),
            # An "agent:"-prefixed key whose position 1 is NOT "main": the
            # shape is not the gateway's, so nothing may be parsed out of it.
            ("agent:assistant-a:telegram:dm:1", ""),
            ("agent:sub:whatsapp:dm:1", ""),
            ("", ""),
        ]
        for key, expected in cases:
            with self.subTest(key=key):
                self.assertEqual(plugin._platform_from_session_key(key), expected)

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
    """``_should_bypass`` is the A/B cohort assignment. Two properties make the
    cohort usable: a session without an id can never be assigned (its turns
    would flip cohort mid-session and poison both arms), and assignment is a
    deterministic hash of (scope, session_id) rather than a coin flip."""

    SCOPE = "assistant-a:telegram"
    # Buckets under the (scope, sid) sha1: 0.369 and 0.828 respectively, so at
    # rate 0.5 the first is inside the cohort and the second is outside.
    IN_COHORT = REAL_KEY_TELEGRAM
    OUT_OF_COHORT = "agent:main:telegram:dm:3"

    def setUp(self):
        self._original_rate = plugin._CONFIG.get("bypass_rate")
        self.addCleanup(self._restore_rate)

    def _restore_rate(self):
        if self._original_rate is None:
            plugin._CONFIG.pop("bypass_rate", None)
        else:
            plugin._CONFIG["bypass_rate"] = self._original_rate

    def test_blank_session_id_never_bypasses_even_at_full_rate(self):
        # Rate 1.0 so the "rate <= 0" early return cannot be what produces the
        # False — only the blank-session guard can.
        plugin._CONFIG["bypass_rate"] = 1.0
        self.assertTrue(plugin._should_bypass(self.SCOPE, REAL_KEY_TELEGRAM),
                        "PRECONDITION: at rate 1.0 an identified session bypasses")
        self.assertFalse(plugin._should_bypass(self.SCOPE, ""),
                         "a session with no id can never join the bypass cohort")

    def test_cohort_assignment_is_a_stable_hash_not_a_coin_flip(self):
        plugin._CONFIG["bypass_rate"] = 0.5
        first = plugin._should_bypass(self.SCOPE, self.IN_COHORT)
        self.assertIs(first, plugin._should_bypass(self.SCOPE, self.IN_COHORT),
                      "the same session must get the same answer every turn")
        self.assertTrue(first, "this session's bucket falls inside a 0.5 rate")
        self.assertFalse(
            plugin._should_bypass(self.SCOPE, self.OUT_OF_COHORT),
            "a session whose bucket exceeds the rate stays out of the cohort",
        )


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
        # The fix derives the agent from the HERMES_HOME profile, and the
        # platform from the key — across platforms, not just telegram.
        for key, platform in ((REAL_KEY_TELEGRAM, "telegram"),
                              (REAL_KEY_WHATSAPP, "whatsapp")):
            with self.subTest(platform=platform):
                recovered = plugin._recover_attribution_without_state(key, kwargs={})
                self.assertEqual(
                    recovered,
                    ("assistant-a", platform, f"assistant-a:{platform}"),
                )

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
            tool_call_id="tc-primary",
        )
        tool_calls_path = Path(self.profile_home) / "state" / "tool-belt" / "tool_calls.jsonl"
        rows = [json.loads(l) for l in tool_calls_path.read_text().splitlines() if l]
        self.assertTrue(rows, "expected one tool-call row")
        row = rows[-1]
        self.assertEqual(row["agent"], "assistant-a")
        self.assertEqual(row["platform"], "telegram")
        self.assertEqual(row["scope"], "assistant-a:telegram")


class SessionResetEvictsSessionStateTests(unittest.TestCase):
    """True cleanup of sticky residency and lookback history belongs to
    ``_on_session_reset`` (/new, /reset) alone — and it must find the state
    even though Hermes hands the hook a fresh AIAgent uuid rather than the
    canonical key the state is filed under. (The other half of the contract —
    that the per-turn ``on_session_end`` hook does NOT evict them — is pinned
    by ``test_cache_aware.SessionHookSemanticsTests``.)"""

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

    def test_session_reset_still_evicts_them(self):
        sticky_key = self._seed_state(REAL_KEY_TELEGRAM)
        plugin._on_session_reset(session_id="new-uuid",
                                 session_key=REAL_KEY_TELEGRAM)
        self.assertNotIn(sticky_key, plugin._STICKY_BY_KEY)
        self.assertNotIn(REAL_KEY_TELEGRAM, plugin._PRIOR_MESSAGES_BY_SESSION)


class AnalyzerExcludesDegradedModeTests(unittest.TestCase):
    def test_load_preset_excludes_returns_status_no_policy(self):
        out, status = analyze._load_preset_excludes(Path(tempfile.mkdtemp()))
        self.assertEqual(out, {})
        self.assertEqual(status, "no_policy")

    def test_load_preset_always_carry_reads_the_shipped_policy(self):
        tools, status = analyze._load_preset_always_carry(Path(plugin.__file__).parent)
        self.assertEqual(status, "ok")
        # always_carry holds only the immutable residents; adaptive carry tools
        # (mnemosyne_recall, process) are NOT part of the immutable set.
        self.assertIn("clarify", tools)
        self.assertIn("expand_tools", tools)
        self.assertNotIn("mnemosyne_recall", tools)
        self.assertNotIn("process", tools)


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
            "schema_version": 2,
            "ts": ts,
            "prediction_id": pid,
            "session_id": sid,
            "hermes_session_id": sid,
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
            "carry_tools": [],
            "expand_only_tools": [],
        }

    def _row_call(self, *, pid, tool):
        return {"schema_version": 2, "prediction_id": pid, "tool_name": tool,
                "source": "gateway"}

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
        # One turn past the shipped lookahead, whatever that is — the fixture
        # is generated from the constant so a change to the window moves this
        # test with it instead of silently making it assert nothing.
        window = analyze.DEFAULT_STICKY_LOOKAHEAD_TURNS
        preds = [
            self._row_prediction(pid="p1", sid="s1", trigger="shell",
                                 tools_for_trigger=["terminal"], ts=1.0),
        ]
        preds += [
            self._row_prediction(pid=f"p{n}", sid="s1", trigger=None,
                                 tools_for_trigger=[], ts=float(n))
            for n in range(2, window + 3)
        ]
        beyond = preds[-1]["prediction_id"]
        calls = [self._row_call(pid=beyond, tool="terminal")]
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
        # The same fixture WITHOUT the covering pattern must produce
        # candidates — otherwise "no overlap leaked" would be vacuously true
        # because the miner found nothing at all.
        uncovered = {"deploy": {"tools": ["deploy_tool"], "keyword_patterns": []}}
        with mock.patch.object(analyze, "_load_preset_triggers",
                               return_value=(uncovered, "ok")):
            baseline = analyze.trigger_keyword_candidates(stats, self._args())
        covered = triggers["deploy"]["keyword_patterns"][0]
        baseline_patterns = [c["pattern"] for r in baseline for c in r["candidates"]]
        self.assertTrue(any(covered.search(p) for p in baseline_patterns),
                        "PRECONDITION: without the existing pattern this "
                        f"fixture mines an overlapping candidate; got "
                        f"{baseline_patterns}")

        with mock.patch.object(analyze, "_load_preset_triggers",
                               return_value=(triggers, "ok")):
            rows = analyze.trigger_keyword_candidates(stats, self._args())
        # Every candidate the existing pattern already covers must be gone —
        # the miner may still offer a narrower, uncovered n-gram.
        for row in rows:
            for c in row["candidates"]:
                self.assertFalse(covered.search(c["pattern"]),
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
        # The promote branch must fire without leaking the category string
        # into the carry patch.
        self.assertEqual(row["action"], "promote_to_carry")
        scopes = row["proposed_learned_patch"]["scopes"]
        for scope_patch in scopes.values():
            self.assertNotIn("filesystem", scope_patch.get("carry", []))
            self.assertNotIn("filesystem", scope_patch.get("expand_only", []))


class ExpansionProvidedAccessAttributionTests(unittest.TestCase):
    """The ``expansion_provided_access`` flag must only fire when expansion
    actually provided the tool. This class holds the same-turn round-trip
    half: an ``expand_tools`` call that resolved a tool the model already had
    still records the round-trip cost but credits nothing. (The sticky /
    cross-turn half is pinned through the real wrapper in
    ``BuildApiKwargsSnapshotTests``.)

    The bug this guards against: live telemetry showed 94% of
    ``expansion_provided_access`` flags coinciding with
    ``was_initially_active == True``, inflating analyzer promotion signal ~15×.
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

    def test_pending_expansion_skipped_when_already_active(self):
        # Same-turn expansion (pending_expansion) for a tool that was
        # already resident: the round-trip happened but provided
        # nothing. after_expand_tools still records the round-trip, but
        # expansion_provided_access must stay false.
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
            session_id=REAL_KEY_TELEGRAM, tool_call_id="tc-primary",
        )
        row = self._latest_row()
        self.assertTrue(row.get("was_initially_active"))
        self.assertTrue(row.get("after_expand_tools"),
            "round-trip happened — should still record after_expand_tools")
        self.assertNotEqual(row.get("expansion_provided_access"), True,
            "round-trip provided nothing new — must not be credited as expansion-driven")


class PrimaryDispatchGateTests(unittest.TestCase):
    """A ``tool_calls.jsonl`` row means "the model dispatched this tool" —
    nothing else. Hermes emits ``post_tool_call`` both for the model's
    assistant tool_use blocks (carrying the provider ``tool_call_id``) and
    for nested/secondary dispatches routed straight through
    ``model_tools.handle_function_call`` — code-execution sandboxes, the
    code kernel, the MCP tools server, and memory/mnemosyne batch fan-out.
    Those nested calls carry an empty ``tool_call_id`` and never faced
    per-turn narrowing; logging them inflated ``uses_in_window`` in the
    economic demotion engine (a resolved-but-not-dispatched tool looked
    heavily used, so it was systematically over-carried).
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

        plugin._STICKY_BY_KEY.clear()
        plugin._POLICY_TURN_BY_SCOPE.clear()
        plugin._PREDICTION_CV.set(None)

    def _restore_config(self) -> None:
        plugin._CONFIG.clear()
        plugin._CONFIG.update(self._original_config)
        plugin._STICKY_BY_KEY.clear()
        plugin._POLICY_TURN_BY_SCOPE.clear()
        plugin._PREDICTION_CV.set(None)

    def _rows(self) -> list[dict]:
        path = Path(self.profile_home) / "state" / "tool-belt" / "tool_calls.jsonl"
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text().splitlines() if l]

    def _set_post_expansion_state(self) -> None:
        # Mirror production after the model ran expand_tools(coding): memory
        # is now in the expansions set and pending_expansion lingers. A later
        # nested memory write in this context is exactly the polluting case.
        plugin._PREDICTION_CV.set({
            "prediction_id": "pred-current",
            "agent": "assistant-a",
            "platform": "telegram",
            "scope": "assistant-a:telegram",
            "sticky_key": plugin._sticky_key_for_session(REAL_KEY_TELEGRAM),
            "session_id": REAL_KEY_TELEGRAM,
            "initial_active_tools": ["terminal"],
            "baseline_active_tools": ["terminal"],
            "expand_only_tools": [],
            "expansions": {"coding", "memory"},
            "pending_expansion": {
                "category": "coding",
                "resolved_tools": ["memory", "read_file", "write_file"],
                "tools_added": ["memory", "read_file", "write_file"],
            },
            "ceiling_tools": ["terminal", "memory"],
        })

    def test_expansion_resolution_does_not_log_nested_tool(self):
        # The exact live-file signature: a memory row with
        # activation_source=expansion / after_expand_tools=true that the
        # model never dispatched. It reaches the hook as a nested dispatch
        # (empty tool_call_id) — it must NOT create a row.
        self._set_post_expansion_state()
        plugin._on_post_tool_call(
            tool_name="memory", args={}, result="ok",
            task_id=REAL_KEY_TELEGRAM, session_id="", tool_call_id="",
        )
        rows = self._rows()
        self.assertEqual(
            [r for r in rows if r.get("tool_name") == "memory"], [],
            "a nested (id-less) memory dispatch must not create a call row",
        )

    def test_primary_dispatch_is_logged_with_tool_call_id(self):
        # A genuine model dispatch carries the provider tool_call_id and must
        # still be recorded — including the id, for dedup/debug.
        self._set_post_expansion_state()
        plugin._on_post_tool_call(
            tool_name="memory", args={}, result="ok",
            task_id="t1", session_id=REAL_KEY_TELEGRAM, tool_call_id="toolu_abc123",
        )
        mem = [r for r in self._rows() if r.get("tool_name") == "memory"]
        self.assertEqual(len(mem), 1, "a real model dispatch must be logged")
        self.assertEqual(mem[0].get("tool_call_id"), "toolu_abc123")


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

    def test_initial_active_tools_snapshot_survives_post_expansion_call(self):
        # Predictor narrowed to {memory, send_message}; terminal is expand_only.
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

    def test_end_to_end_recovered_tool_credits_expansion_provided_access(self):
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
            task_id="t1", session_id=REAL_KEY_TELEGRAM, tool_call_id="tc-primary",
        )

        tool_calls_path = Path(self.profile_home) / "state" / "tool-belt" / "tool_calls.jsonl"
        rows = [json.loads(l) for l in tool_calls_path.read_text().splitlines() if l]
        row = rows[-1]
        self.assertFalse(row.get("was_initially_active"),
            "baseline excluded terminal — it was only callable via sticky")
        self.assertTrue(row.get("expansion_provided_access"),
            "sticky-recovered tool must be credited as expansion-driven")

    def test_implicit_carry_resident_widens_the_credit_baseline(self):
        """Class C is mostly IMPLICIT (E − A − demoted), so the predictor's
        pre-ceiling baseline understates what the model actually saw.

        ``_build_api_kwargs`` widens ``baseline_active_tools`` with the
        resolved resident classes once the live ceiling is known. Without that
        widening a tool that was carried anyway looks absent from the
        baseline, and a coincidental sticky entry then credits
        ``expansion_provided_access`` to it — the exact 15× promotion-signal
        inflation the credit rules exist to prevent. No other test drives an
        implicitly-carried tool (one named in NO resident list) through the
        real wrapper and then calls it.
        """
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
            "active_tool_names": ["memory"],
            # The predictor's pre-ceiling baseline: terminal is NOT in it, and
            # terminal is named in no resident list either — it becomes a
            # class-C resident only once the ceiling is resolved.
            "baseline_active_tools": ["memory"],
            "resolved_always_carry": [],
            "resolved_carry": ["memory"],
            "resolved_demoted": [],
            "triggered_tools": [],
            "expansions": set(),
            "agent": "assistant-a", "platform": "telegram",
            "scope": "assistant-a:telegram",
            "sticky_key": sticky_key, "session_id": REAL_KEY_TELEGRAM,
            "sticky_tools": [], "sticky_categories": [],
            "sticky_remaining_turns": {},
            "expand_only_tools": [], "pending_expansion": None,
        })

        self._run_wrapper_with_tools(["memory", "terminal"])
        state = plugin._PREDICTION_CV.get()
        self.assertIn("terminal", state["carry_carry"],
                      "PRECONDITION: terminal resolves to an implicit class-C "
                      "resident against the live ceiling")
        self.assertIn("terminal", state["baseline_active_tools"],
                      "the resolved residents widen the credit baseline")

        plugin._on_post_tool_call(
            tool_name="terminal", args={}, result="ok",
            task_id="t1", session_id=REAL_KEY_TELEGRAM, tool_call_id="tc-primary",
        )
        tool_calls_path = Path(self.profile_home) / "state" / "tool-belt" / "tool_calls.jsonl"
        row = [json.loads(l) for l in tool_calls_path.read_text().splitlines() if l][-1]
        self.assertTrue(row.get("was_initially_active"),
                        "an implicitly carried tool was available all along")
        self.assertNotEqual(row.get("expansion_provided_access"), True,
                            "a tool that was carried anyway must never be "
                            "credited to expansion, sticky entry or not")

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
            task_id="t1", session_id=REAL_KEY_TELEGRAM, tool_call_id="tc-primary",
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
