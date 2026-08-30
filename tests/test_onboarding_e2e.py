"""End-to-end onboarding, driven by scripted conversations.

Where ``tests/test_configure.py`` unit-tests ``scripts/configure.py`` against
hand-built fixture rows, this file drives the whole onboarding arc over
telemetry produced by ``tests/seed_sessions.py`` — the real policy resolver
and the real predictor, run over the scripted conversations in
``tests/scripts/``.

Isolation, in every test:

* a temporary ``HERMES_HOME`` whose path contains a space
  (``test_configure.TempHomeTestCase``);
* ``shutil.which`` and ``configure._default_runner`` both patched, so no real
  ``hermes`` binary is ever reachable — and so the overlay write is reached
  at all: ``flow_shape`` returns before
  writing ``learned.json`` when ``hermes`` is missing;
* ``learned._CACHE`` cleared, so one test's overlay cannot survive into the
  next through the mtime cache;
* no network, no provider key, no model call — the seeder never opens a socket.

Config reads and writes do not round-trip through ``FakeRunner``, and
``main()`` reads config once before any flow runs (``configure.py:1083``), so
a state transition is asserted by **re-invoking** ``main()`` with the new
values seeded into ``FakeRunner.get_values`` — never by expecting one call to
observe its own writes.
"""
from __future__ import annotations

import contextlib
import json
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))
import conftest  # noqa: F401,E402 — registers tool_belt_plugin

try:  # discovery imports this file as ``tests.test_onboarding_e2e``
    from tests import seed_sessions, test_configure as tc  # noqa: E402
except ImportError:  # …but ``tests/`` alone on sys.path also has to work
    import seed_sessions  # type: ignore[no-redef]  # noqa: E402
    import test_configure as tc  # type: ignore[no-redef]  # noqa: E402

configure = tc.configure
SCRIPTS_DIR = TESTS_DIR / "scripts"


def _load(name: str) -> dict:
    return seed_sessions.load_script(SCRIPTS_DIR / f"{name}.yaml")


class OnboardingTestCase(tc.TempHomeTestCase):
    """Temp home, cleared learned cache, and a scripted-seed helper."""

    def setUp(self) -> None:
        super().setUp()
        # learned.py caches learned.json by mtime; two tests writing different
        # overlays inside one mtime tick would otherwise cross-contaminate.
        seed_sessions.learned._CACHE.clear()
        self.addCleanup(seed_sessions.learned._CACHE.clear)

    def seed(self, script_name: str, **kwargs) -> "seed_sessions.SeedResult":
        return seed_sessions.seed(_load(script_name), self.home, **kwargs)

    def run_main(self, argv: list[str], runner: "tc.FakeRunner", which: str | None = "/usr/bin/hermes"):
        """Invoke ``configure.main`` with every path to the real machine cut."""
        lines: list[str] = []
        with contextlib.ExitStack() as stack:
            for patch in tc.isolate(runner, which, lines):
                stack.enter_context(patch)
            rc = configure.main(argv + ["--hermes-home", str(self.home)])
        return rc, "\n".join(lines)

    def fs_snapshot(self) -> set[str]:
        return {str(p.relative_to(self.home)) for p in self.home.rglob("*")}

    def learned(self, state_dir: Path | None = None) -> dict:
        path = (state_dir or self.root_state) / "learned.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def observing_config(scope: str) -> dict[str, str]:
        """``FakeRunner`` reads that put ``scope`` in observation mode."""
        return {
            f"plugins.tool-belt.channels.{scope}.bypass_rate": "1.0",
            f"plugins.tool-belt.channels.{scope}.learned_mode": "recommend",
        }


class SeederTests(OnboardingTestCase):
    """The seeder itself: schema fidelity, determinism, trigger verification."""

    def test_seeded_rows_carry_the_production_prediction_schema(self) -> None:
        """Rows are built through ``PredictionRecord``, so the keys must match.

        This is the guard for the one structural risk in the approach: if
        ``PredictionRecord.to_dict()`` gains, renames, or drops a field the
        shaper reads, a hand-rolled writer would drift silently.
        """
        result = self.seed("observing")
        rows = [
            json.loads(line)
            for line in (result.state_dir / "predictions.jsonl").read_text().splitlines()
        ]
        reference = seed_sessions.logger_io.PredictionRecord(
            ts=0.0,
            prediction_id="",
            session_id="",
            channel="",
            message_hash="",
            message_preview="",
            preset="",
            triggers_fired=[],
            always_carry_count=0,
            carry_count=0,
            ceiling_count=0,
            narrowed_count=0,
            ceiling_tokens=0,
            narrowed_tokens=0,
        ).to_dict()
        self.assertEqual(set(rows[0]), set(reference))
        # …and every field the shaper actually reads is populated, not just present.
        for field in ("scope", "hermes_session_id", "ts", "prediction_id"):
            self.assertTrue(rows[0][field], f"{field} should be non-blank")
        for field in ("always_carry_tools", "carry_tools", "expand_only_tools"):
            self.assertIn(field, rows[0])

    def test_seeding_is_deterministic_and_uses_no_wall_clock(self) -> None:
        first = self.seed("observing")
        text = (first.state_dir / "predictions.jsonl").read_text()
        (first.state_dir / "predictions.jsonl").unlink()
        (first.state_dir / "tool_calls.jsonl").unlink()
        second = self.seed("observing")
        self.assertEqual((second.state_dir / "predictions.jsonl").read_text(), text)

        rows = [json.loads(line) for line in text.splitlines()]
        self.assertTrue(all(r["ts"] >= seed_sessions.BASE_TS for r in rows))
        self.assertTrue(all(r["ts"] < seed_sessions.BASE_TS + 1e6 for r in rows))
        self.assertTrue(all(r["tokens_estimator"] == "seeded" for r in rows))
        self.assertEqual(len({r["hermes_session_id"] for r in rows}), 5)

    def test_a_trigger_regression_fails_the_seed(self) -> None:
        script = _load("observing")
        script["turns"][0]["expect_triggers"] = ["browser"]
        with self.assertRaises(seed_sessions.ScriptMismatch):
            seed_sessions.seed(script, self.home)


class OnboardingArcTests(OnboardingTestCase):
    """The four-state machine and both onboarding paths, end to end."""

    def test_fresh_reporting(self) -> None:
        """A home with no telemetry reports nothing and invents nothing."""
        runner = tc.FakeRunner()
        rc, output = self.run_main(["--status"], runner)
        self.assertEqual(rc, 0)
        self.assertIn("No agent scopes found yet", output)
        self.assertEqual(runner.writes, [])

    def test_recommend_to_observing(self) -> None:
        """recommend → the two config writes, the sidecar, then ``observing``."""
        result = self.seed("observing")
        self.assertLess(result.sessions, self.needed)

        runner = tc.FakeRunner()
        rc, _out = self.run_main(["--path", "recommend", "--yes"], runner)
        self.assertEqual(rc, 0)
        emitted = {c[3]: c[4] for c in runner.writes}
        self.assertEqual(
            emitted,
            {
                f"plugins.tool-belt.channels.{result.scope}.learned_mode": "recommend",
                f"plugins.tool-belt.channels.{result.scope}.bypass_rate": "1.0",
            },
        )
        # The sidecar remembers the bypass value observation mode replaced.
        sidecar = self.root_state / configure.CONFIGURE_STATE_FILE
        self.assertTrue(sidecar.exists())
        self.assertIn(result.scope, json.loads(sidecar.read_text())["scopes"])

        # main() reads config once, before the flow runs — so the transition is
        # asserted on a fresh invocation with those values fed back in.
        _rc, output = self.run_main(
            ["--status"], tc.FakeRunner(self.observing_config(result.scope))
        )
        self.assertIn(configure.STATE_OBSERVING, output)
        self.assertIn(f"{result.sessions}/{self.needed} sessions", output)

    def test_threshold_crossing_to_ready(self) -> None:
        """The same config at 20 sessions reads ``ready`` instead."""
        result = self.seed("chat-heavy")
        self.assertGreaterEqual(result.sessions, self.needed)

        runner = tc.FakeRunner(self.observing_config(result.scope))
        rc, output = self.run_main(["--status"], runner)
        self.assertEqual(rc, 0)
        self.assertIn(configure.STATE_READY, output)
        self.assertIn(f"{result.sessions}/{self.needed} sessions", output)
        self.assertEqual(runner.writes, [])

    def test_shape_spine(self) -> None:
        """shape → overlay on disk, ``learned_mode: apply``, bypass off, ``shaped``."""
        result = self.seed("terminal-heavy")
        runner = tc.FakeRunner(self.observing_config(result.scope))
        rc, output = self.run_main(["--path", "shape", "--yes"], runner)
        self.assertEqual(rc, 0)

        entry = self.learned()["scopes"][result.scope]
        self.assertIn("execute_code", entry["carry"])
        self.assertEqual(entry["shaping"]["scope"], result.scope)
        self.assertIn("execute_code", output)

        # The confirmed-apply epilogue speaks the 1.0 vocabulary: the retired
        # "moved to on-demand" phrasing must never resurface in any flow.
        self.assertIn("moved to expand-only", output)
        self.assertNotIn("moved to on-demand", output)

        self.assertEqual(
            {c[3]: c[4] for c in runner.writes},
            {
                f"plugins.tool-belt.channels.{result.scope}.learned_mode": "apply",
                f"plugins.tool-belt.channels.{result.scope}.bypass_rate": "0.0",
            },
        )

        _rc, status = self.run_main(
            ["--status"],
            tc.FakeRunner({f"plugins.tool-belt.channels.{result.scope}.learned_mode": "apply"}),
        )
        self.assertIn(configure.STATE_SHAPED, status)

    def test_overlay_write_is_disclosed_and_matches_what_lands_on_disk(self) -> None:
        """The terminal diff names every tool the overlay write then contains."""
        result = self.seed("terminal-heavy")
        runner = tc.FakeRunner(self.observing_config(result.scope))
        rc, output = self.run_main(["--path", "shape", "--yes"], runner)
        self.assertEqual(rc, 0)

        entry = self.learned()["scopes"][result.scope]
        lines = output.splitlines()
        carry_line = next(
            l for l in lines if f"learned.json[{result.scope}].carry:" in l
        )
        expand_line = next(
            l for l in lines if f"learned.json[{result.scope}].expand_only:" in l
        )
        self.assertIn(f"→ {len(entry['carry'])} (", carry_line)
        self.assertIn(f"→ {len(entry['expand_only'])} (", expand_line)
        for tool in entry["carry"]:
            self.assertIn(f"+{tool}", carry_line)
        for tool in entry["expand_only"]:
            self.assertIn(f"+{tool}", expand_line)

    def test_demote_arm_named_profile(self) -> None:
        """A chat-only agent in a named profile demotes its unused carry residents.

        Under the 1.0 carrying model shaping moves enabled built-ins only between
        the adaptive ``carry`` class and ``expand_only``: every ``carry`` resident
        that goes unused across the window is a demote candidate (there is no
        protected policy-mirror shielding them anymore). The immutable
        ``always_carry`` surface is never a candidate. The overlay lands in the
        named profile's state dir and is read through the v2 normalizer.
        """
        result = self.seed("chat-heavy", profile_override="assistant-b")
        profile_state = self.home / "profiles" / "assistant-b" / "state" / "tool-belt"
        self.assertEqual(result.state_dir, profile_state)

        runner = tc.FakeRunner(self.observing_config(result.scope))
        rc, _out = self.run_main(["--agent", "assistant-b", "--path", "shape", "--yes"], runner)
        self.assertEqual(rc, 0)

        # The overlay lands in the profile's state dir, not the root's.
        self.assertFalse((self.root_state / "learned.json").exists())
        entry = seed_sessions.learned.normalize_state(
            self.learned(profile_state)
        )["scopes"][result.scope]
        expand_only = set(entry["expand_only"])

        # The injected unused residents demote to expand_only …
        self.assertLessEqual({"unit_convert", "weather_lookup"}, expand_only)
        # … alongside every other unused policy carry resident, while the one
        # carry tool the agent actually used (mnemosyne_recall) stays resident.
        self.assertNotIn("mnemosyne_recall", expand_only)
        # Nothing was promoted into carry (no expansion evidence in this arm).
        self.assertEqual(entry["carry"], [])

        # The immutable always_carry surface can never be demoted, by construction.
        always_carry = set(seed_sessions.presets.load_base_policy().always_carry)
        self.assertEqual(expand_only & always_carry, set())

    def test_multi_scope_independence(self) -> None:
        """Two scopes in one home; shaping one leaves the other untouched."""
        terminal = self.seed("terminal-heavy")
        browser = self.seed("browser-heavy")
        self.assertEqual(terminal.state_dir, browser.state_dir)

        infos = {i.scope: i for i in configure.discover_scopes(self.home)}
        self.assertEqual(set(infos), {terminal.scope, browser.scope})

        runner = tc.FakeRunner()
        ctx = tc.make_ctx(self.home, runner, assume_yes=True, thresholds=self.thresholds)
        self.assertEqual(configure.flow_shape(ctx, [infos[terminal.scope]]), 0)

        self.assertEqual(list(self.learned()["scopes"]), [terminal.scope])
        self.assertEqual([c[3] for c in runner.writes],
                         [f"plugins.tool-belt.channels.{terminal.scope}.learned_mode"])

        _rc, status = self.run_main(
            ["--status"],
            tc.FakeRunner({f"plugins.tool-belt.channels.{terminal.scope}.learned_mode": "apply"}),
        )
        self.assertIn(f"{terminal.scope:<28} {configure.STATE_SHAPED}", status)
        self.assertIn(f"{browser.scope:<28} {configure.STATE_READY}", status)

if __name__ == "__main__":
    unittest.main()
