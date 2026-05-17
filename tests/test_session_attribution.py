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


if __name__ == "__main__":
    unittest.main()
