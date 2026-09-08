"""Shell-bypass detection: the pattern table, the runtime gate, the writer's
privacy contract, and the analyzer's promote-to-carry path.

The promotion machinery only learned from tools the model *recovered* via
``expand_tools`` (the harvest expand-only path). A latent tool the model
bypassed by shelling out — because it never thought to expand it — was
invisible. These tests lock the new detector that makes such bypasses
observable, and the guarantees around it:

  1. ``bypass_map.detect_bypass`` — match / no-match / malformed-entry skip /
     bad-regex skip, and the privacy property that only (pattern_id, tool)
     ever leave the detector.
  2. The ``_maybe_log_shell_bypass`` gate — fires only for a latent tool on a
     narrowed gateway turn; silent when the tool was initially active, when no
     ceiling is resolved, when the tool isn't in the ceiling, and deduped per
     prediction+tool so failed retries count once.
  3. The writer's privacy contract — the on-disk row carries no command text
     and no command-derived field, only the matched tool, the pattern id, and
     the reused message_preview.
  4. ``analyze.bypass_recommendation_rows`` — fires above the min-bypass floor,
     stays silent below it, and drives a promote_to_carry patch.
  5. The real 2026-09-08 incident replay: prediction 99effab4707a4990 with
     cronjob latent, three ``hermes cron create`` shell calls → exactly one
     cronjob bypass event.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

if "tool_belt_plugin" not in sys.modules:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent.parent))
    sys.path.insert(0, str(here.parent))
    from tests import conftest  # noqa: F401 — side-effect: register package

plugin = sys.modules["tool_belt_plugin"]
bypass_map = importlib.import_module("tool_belt_plugin.bypass_map")
logger_io = importlib.import_module("tool_belt_plugin.logger_io")
analyze = importlib.import_module("tool_belt_plugin.analyze")

PLUGIN_DIR = Path(__file__).resolve().parent.parent


def _args(*extra: str):
    """Real analyzer defaults, so the tests track the shipped thresholds."""
    with mock.patch.object(sys, "argv", ["analyze.py", *extra]):
        return analyze.parse_args()


# ─── 1. Pattern table ────────────────────────────────────────────────────────

class BypassMapTests(unittest.TestCase):
    def test_shipped_cron_pattern_matches(self):
        # The real shipped policy.yaml table must map `hermes cron create` to
        # cronjob — the incident that motivated the feature.
        for cmd in (
            "hermes cron create --name x --schedule '@daily'",
            "HERMES CRON RUN 9286d021",
            "cd /tmp && hermes cron list",
        ):
            match = bypass_map.detect_bypass(cmd)
            self.assertIsNotNone(match, f"expected a match for {cmd!r}")
            self.assertEqual(match, ("hermes_cron_subcommand", "cronjob"))

    def test_no_match_for_unrelated_shell(self):
        for cmd in ("ls -la", "git status", "hermes status",
                    "echo hermes cron is a thing"):
            self.assertIsNone(bypass_map.detect_bypass(cmd),
                               f"unexpected match for {cmd!r}")

    def test_empty_and_nonstring_command(self):
        self.assertIsNone(bypass_map.detect_bypass(""))
        self.assertIsNone(bypass_map.detect_bypass(None))  # type: ignore[arg-type]

    def _table_from_yaml(self, body: str) -> list:
        tmp = Path(tempfile.mkdtemp()) / "policy.yaml"
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp.parent, ignore_errors=True))
        tmp.write_text(body, encoding="utf-8")
        return bypass_map.load_bypass_patterns(tmp)

    def test_malformed_entries_are_skipped_individually(self):
        # A non-dict entry, an entry missing `tool`, and one missing `pattern`
        # are each dropped, but the one good entry survives.
        table = self._table_from_yaml(
            "bypass_patterns:\n"
            "  - just a string\n"
            "  - id: no_tool\n"
            "    pattern: 'x'\n"
            "  - id: no_pattern\n"
            "    tool: cronjob\n"
            "  - id: good\n"
            "    tool: cronjob\n"
            "    pattern: 'hermes cron'\n"
        )
        self.assertEqual([r.id for r in table], ["good"])

    def test_bad_regex_is_skipped_not_fatal(self):
        # An uncompilable regex must not take down the rest of the table.
        table = self._table_from_yaml(
            "bypass_patterns:\n"
            "  - id: broken\n"
            "    tool: cronjob\n"
            "    pattern: '([unterminated'\n"
            "  - id: fine\n"
            "    tool: cronjob\n"
            "    pattern: 'hermes cron'\n"
        )
        self.assertEqual([r.id for r in table], ["fine"])

    def test_missing_file_yields_empty_table(self):
        missing = Path(tempfile.mkdtemp()) / "nope.yaml"
        self.addCleanup(lambda: __import__("shutil").rmtree(missing.parent, ignore_errors=True))
        self.assertEqual(bypass_map.load_bypass_patterns(missing), [])

    def test_no_bypass_section_yields_empty_table(self):
        self.assertEqual(self._table_from_yaml("name: aggressive\n"), [])


# ─── 2 + 3. The runtime gate and the writer's privacy contract ───────────────

class ShellBypassGateTests(unittest.TestCase):
    SECRET = "SECRET_CANARY_job_name_should_never_be_logged_XYZ789"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._env = mock.patch.dict(
            os.environ, {"HERMES_HOME": self.tmp.name}, clear=False)
        self._env.start()
        self.addCleanup(self._env.stop)

    def _events(self) -> list[dict]:
        path = logger_io.bypass_events_path()
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def _latent_state(self, **over) -> dict:
        """A narrowed gateway turn with cronjob latent (in ceiling, not active)."""
        state = {
            "prediction_id": "pred1",
            "scope": "bernard:telegram",
            "ceiling_tools": ["terminal", "cronjob", "todo"],
            "baseline_active_tools": ["terminal", "todo"],
            "initial_active_tools": ["terminal", "todo"],
            "message_preview": "When you say left yourself a note you mean scheduled",
        }
        state.update(over)
        return state

    def _cmd(self, args):
        return {"command": args}

    def test_fires_for_latent_tool(self):
        state = self._latent_state()
        plugin._maybe_log_shell_bypass(
            "terminal", self._cmd(f"hermes cron create --name {self.SECRET}"),
            state, "sess1", "gateway")
        events = self._events()
        self.assertEqual(len(events), 1)
        row = events[0]
        self.assertEqual(row["matched_tool"], "cronjob")
        self.assertEqual(row["pattern_id"], "hermes_cron_subcommand")
        self.assertEqual(row["shell_tool"], "terminal")
        self.assertTrue(row["tool_in_ceiling"])
        self.assertFalse(row["tool_was_initially_active"])

    def test_privacy_no_command_text_or_field(self):
        state = self._latent_state()
        plugin._maybe_log_shell_bypass(
            "terminal", self._cmd(f"hermes cron create --name {self.SECRET}"),
            state, "sess1", "gateway")
        events = self._events()
        self.assertEqual(len(events), 1)
        row = events[0]
        serialized = json.dumps(row)
        # No command text, no command-derived field.
        self.assertNotIn(self.SECRET, serialized)
        self.assertNotIn("hermes cron", serialized)
        self.assertNotIn("command", row)
        self.assertNotIn("args", row)
        # The only disclosure is the reused prediction preview.
        self.assertEqual(row["message_preview"], state["message_preview"])

    def test_silent_when_initially_active(self):
        # An active tool driven from the shell is operator preference, not a
        # learning signal.
        state = self._latent_state(
            baseline_active_tools=["terminal", "cronjob", "todo"],
            initial_active_tools=["terminal", "cronjob", "todo"])
        plugin._maybe_log_shell_bypass(
            "terminal", self._cmd("hermes cron create --name x"),
            state, "sess1", "gateway")
        self.assertEqual(self._events(), [])

    def test_silent_without_ceiling(self):
        state = self._latent_state(ceiling_tools=[])
        plugin._maybe_log_shell_bypass(
            "terminal", self._cmd("hermes cron create --name x"),
            state, "sess1", "gateway")
        self.assertEqual(self._events(), [])

    def test_silent_when_tool_not_in_ceiling(self):
        state = self._latent_state(ceiling_tools=["terminal", "todo"])
        plugin._maybe_log_shell_bypass(
            "terminal", self._cmd("hermes cron create --name x"),
            state, "sess1", "gateway")
        self.assertEqual(self._events(), [])

    def test_silent_for_non_gateway_source(self):
        for source in ("cron", "subagent"):
            plugin._maybe_log_shell_bypass(
                "terminal", self._cmd("hermes cron create --name x"),
                self._latent_state(), "sess1", source)
        self.assertEqual(self._events(), [])

    def test_silent_for_non_shell_tool(self):
        plugin._maybe_log_shell_bypass(
            "execute_code", self._cmd("hermes cron create --name x"),
            self._latent_state(), "sess1", "gateway")
        self.assertEqual(self._events(), [])

    def test_dedup_per_prediction_and_tool(self):
        # Two failed retries + one success within one turn (one shared state
        # object) = one bypass decision.
        state = self._latent_state()
        for cmd in ("hermes cron create --wrong-order",
                    "hermes cron create --still-wrong",
                    "hermes cron create --name ok"):
            plugin._maybe_log_shell_bypass(
                "terminal", self._cmd(cmd), state, "sess1", "gateway")
        self.assertEqual(len(self._events()), 1)


# ─── 4. Analyzer promote-to-carry path ───────────────────────────────────────

class BypassAnalyzerTests(unittest.TestCase):
    def _stats(self, n_events: int, scope: str = "bernard:telegram"):
        preds = [
            {"scope": scope, "prediction_id": f"p{i}", "ts": i,
             "hermes_session_id": "s1", "policy_source": "learned"}
            for i in range(n_events)
        ]
        bypass = [
            {"scope": scope, "prediction_id": f"p{i}", "ts": i,
             "matched_tool": "cronjob", "pattern_id": "hermes_cron_subcommand",
             "shell_tool": "terminal", "tool_in_ceiling": True,
             "tool_was_initially_active": False}
            for i in range(n_events)
        ]
        return analyze.collect_stats(preds, [], bypass)

    def test_fires_above_min(self):
        stats = self._stats(3)  # default floor is 3
        rows = analyze.bypass_recommendation_rows(stats, _args(), set())
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["kind"], "shell_bypass_promotion")
        self.assertEqual(row["item"], "cronjob")
        self.assertEqual(row["action"], "promote_to_carry")
        self.assertEqual(row["metrics"]["shell_bypass_events"], 3)
        self.assertEqual(row["metrics"]["shell_bypass_predictions"], 3)
        self.assertEqual(
            row["proposed_learned_patch"],
            {"scopes": {"bernard:telegram": {"carry": ["cronjob"]}}})

    def test_silent_below_min(self):
        stats = self._stats(2)
        self.assertEqual(analyze.bypass_recommendation_rows(stats, _args(), set()), [])

    def test_custom_min_floor(self):
        stats = self._stats(4)
        below = analyze.bypass_recommendation_rows(
            stats, _args("--harvest-min-bypass-calls", "5"), set())
        self.assertEqual(below, [])
        above = analyze.bypass_recommendation_rows(
            stats, _args("--harvest-min-bypass-calls", "4"), set())
        self.assertEqual(len(above), 1)

    def test_immutable_tool_never_promoted(self):
        stats = self._stats(5)
        rows = analyze.bypass_recommendation_rows(stats, _args(), {"cronjob"})
        self.assertEqual(rows, [])

    def test_separate_counter_from_expand_only(self):
        # A bypass event must NOT feed harvest_was_expand_only, and vice-versa —
        # the two paths are distinct signals.
        stats = self._stats(3)
        stat = stats["bernard:telegram"]
        self.assertEqual(sum(stat.harvest_was_expand_only.values()), 0)
        self.assertEqual(stat.shell_bypass_events["cronjob"], 3)


# ─── 5. Real-incident replay ─────────────────────────────────────────────────

class IncidentReplayTests(unittest.TestCase):
    """Replay the 2026-09-08 evidence through the live detector path."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._env = mock.patch.dict(
            os.environ, {"HERMES_HOME": self.tmp.name}, clear=False)
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_incident_produces_exactly_one_bypass_event(self):
        # The decisive prediction: triggers_fired [], cronjob in ceiling, not
        # narrowed (not in active/baseline).
        state = {
            "prediction_id": "99effab4707a4990",
            "scope": "bernard:telegram",
            "ceiling_tools": ["terminal", "cronjob", "todo", "web_search"],
            "baseline_active_tools": ["terminal", "todo"],
            "initial_active_tools": ["terminal", "todo"],
            "message_preview": 'When you say "left yourself a note," you mean you scheduled',
        }
        session = "20260908_110426_40613908"
        # The real sequence: two failed arg-order attempts, then the success
        # that created job 9286d021d2f5 — all in the one decisive turn.
        for cmd in (
            "hermes cron create sue-surface-fix-followup '@daily'",
            "hermes cron create --name sue-surface-fix-followup --at 16:45",
            "hermes cron create --name sue-surface-fix-followup --schedule '45 16 * * *'",
        ):
            plugin._maybe_log_shell_bypass(
                "terminal", {"command": cmd}, state, session, "gateway")

        path = logger_io.bypass_events_path()
        events = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        self.assertEqual(len(events), 1, "dedup must collapse the retries")
        row = events[0]
        self.assertEqual(row["matched_tool"], "cronjob")
        self.assertEqual(row["prediction_id"], "99effab4707a4990")
        self.assertEqual(row["scope"], "bernard:telegram")

        # And it must surface as exactly one promote-to-carry candidate.
        stats = analyze.collect_stats(
            [{"scope": "bernard:telegram", "prediction_id": "99effab4707a4990",
              "ts": 1, "hermes_session_id": session, "policy_source": "learned"}],
            [], events)
        rows = analyze.bypass_recommendation_rows(
            stats, _args("--harvest-min-bypass-calls", "1"), set())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["item"], "cronjob")


if __name__ == "__main__":
    unittest.main()
