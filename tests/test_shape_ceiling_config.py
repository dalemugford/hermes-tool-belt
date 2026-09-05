"""User-configurable shaper thresholds via ``config.yaml``.

The five ``learning.shape_ceiling`` keys resolve across three layers, per
key, highest first: ``config.yaml`` (the user layer, passed as
``plugin_config``) → ``policy.yaml`` (shipped preset) → ``shaping.DEFAULTS``.

These tests lock the precedence contract and its fail-open behaviour (a bad
user value degrades to the policy layer, never raises), that the shipped
DEFAULTS are exactly the aggressive-shaping contract, and — end to end — that a
``window_days`` set through ``plugin_config`` actually reaches the shaper's
analysis and changes how much history it considers.

Everything runs against throwaway files; no live state is touched.
"""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import conftest  # noqa: F401,E402 — registers the tool_belt_plugin package

from tool_belt_plugin import shaping  # noqa: E402

_DAY = 86400

# A policy layer deliberately distinct from both DEFAULTS and the config
# layer, so a value can be traced to exactly one source.
_POLICY_YAML = """\
name: test-policy
learning:
  shape_ceiling:
    window_days: 77
    promote_min_sessions: 4
    promote_min_calls: 6
    demote_min_sessions_no_use: 12
    demote_k: 2.5
"""


def _write_policy(dir_path: Path, body: str = _POLICY_YAML) -> Path:
    path = dir_path / "policy.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _cfg(shape_ceiling: dict) -> dict:
    """A plugin_config carrying only a learning.shape_ceiling override."""
    return {"learning": {"shape_ceiling": shape_ceiling}}


class PrecedenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.policy = _write_policy(Path(self.tmp.name))

    def test_config_over_policy_over_defaults(self):
        # config.yaml wins where set (window_days, demote_k); the other
        # three fall through to the policy layer, none to DEFAULTS.
        resolved = shaping.load_shape_ceiling_defaults(
            policy_path=self.policy,
            plugin_config=_cfg({"window_days": 30, "demote_k": 1.1}),
        )
        self.assertEqual(resolved["window_days"], 30)      # config layer
        self.assertEqual(resolved["demote_k"], 1.1)        # config layer
        self.assertEqual(resolved["promote_min_sessions"], 4)   # policy layer
        self.assertEqual(resolved["promote_min_calls"], 6)      # policy layer
        self.assertEqual(resolved["demote_min_sessions_no_use"], 12)  # policy
        # And these differ from the hardcoded fallback, proving the layering.
        self.assertNotEqual(resolved["window_days"],
                            shaping.DEFAULTS["window_days"])
        self.assertNotEqual(resolved["promote_min_calls"],
                            shaping.DEFAULTS["promote_min_calls"])

    def test_empty_learning_block_is_policy_layer(self):
        for cfg in ({}, {"learning": {}}, {"learning": {"shape_ceiling": {}}}):
            resolved = shaping.load_shape_ceiling_defaults(
                policy_path=self.policy, plugin_config=cfg)
            self.assertEqual(resolved["window_days"], 77, cfg)
            self.assertEqual(resolved["demote_k"], 2.5, cfg)

    def test_bad_int_values_fall_back_to_policy_without_raising(self):
        # Zero, negative, non-numeric, and wrong-type all fail the positive-int
        # gate and degrade to the policy value (77) — never raising.
        for bad in (0, -5, "nope", None, [30], {"x": 1}):
            resolved = shaping.load_shape_ceiling_defaults(
                policy_path=self.policy,
                plugin_config=_cfg({"window_days": bad}),
            )
            self.assertEqual(resolved["window_days"], 77, bad)

    def test_bad_demote_k_values_fall_back_to_policy_without_raising(self):
        # demote_k accepts a positive float; anything else degrades to policy.
        for bad in (0, 0.0, -1.5, "nope", None, [1.5], {"x": 1}):
            resolved = shaping.load_shape_ceiling_defaults(
                policy_path=self.policy,
                plugin_config=_cfg({"demote_k": bad}),
            )
            self.assertEqual(resolved["demote_k"], 2.5, bad)

    def test_malformed_learning_shapes_never_raise(self):
        for cfg in (
            {"learning": "not-a-dict"},
            {"learning": {"shape_ceiling": "not-a-dict"}},
            {"learning": None},
            "not-a-dict",
            None,
        ):
            resolved = shaping.load_shape_ceiling_defaults(
                policy_path=self.policy, plugin_config=cfg)
            self.assertEqual(resolved["window_days"], 77, cfg)

    def test_config_over_defaults_when_policy_missing(self):
        # Policy file absent → DEFAULTS is the base; config still overrides it.
        missing = Path(self.tmp.name) / "does-not-exist.yaml"
        resolved = shaping.load_shape_ceiling_defaults(
            policy_path=missing, plugin_config=_cfg({"window_days": 25}))
        self.assertEqual(resolved["window_days"], 25)
        # unset key comes from DEFAULTS (no policy layer to consult)
        self.assertEqual(resolved["demote_k"], shaping.DEFAULTS["demote_k"])


class DefaultsContractTests(unittest.TestCase):
    """``shaping.DEFAULTS`` is the last-resort layer every other layer falls
    back to, so its exact content is the shipped promise when policy.yaml is
    unreadable. Fails on any silent re-tune of a threshold or a stray extra
    key."""

    def test_defaults_are_exactly_the_contract(self):
        self.assertEqual(shaping.DEFAULTS, {
            "window_days": 7,
            "promote_min_sessions": 1,
            "promote_min_calls": 2,
            "demote_min_sessions_no_use": 2,
            "demote_k": 1.5,
        })

    def test_window_days_falls_all_the_way_through_to_defaults(self):
        # No policy file, no user layer → the DEFAULTS value is what resolves.
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "no-policy.yaml"
            resolved = shaping.load_shape_ceiling_defaults(policy_path=missing)
        self.assertEqual(resolved, dict(shaping.DEFAULTS))


def _seed_sessions(state_dir: Path, scope: str, n_sessions: int,
                   *, spacing: float = 1.0, newest_ts: float | None = None) -> None:
    """N distinct sessions, each with one prediction + one expansion call for
    ``grep_files`` — enough for the shaper to run its analysis per scope.

    ``spacing`` is the gap in seconds between consecutive sessions (session i
    sits ``(n - 1 - i) * spacing`` seconds before the newest), so a caller can
    lay history out across days instead of seconds.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    newest = time.time() if newest_ts is None else newest_ts
    preds, calls = [], []
    for i in range(n_sessions):
        pid = f"pred-{i}"
        ts = newest - (n_sessions - 1 - i) * spacing
        preds.append({
            "ts": ts,
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
            "ts": ts,
            "schema_version": 2,
            "prediction_id": pid,
            "tool_name": "grep_files",
            "activated_by_expansion": True,
        })
    (state_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8")
    (state_dir / "tool_calls.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in calls), encoding="utf-8")


class EndToEndWindowThreadingTests(unittest.TestCase):
    """``window_days`` set via plugin_config reaches the shaper's analysis: it
    is the value ``auto_shape_run`` passes to ``compute_scope_recommendations``
    AND it changes how many sessions that analysis considers.

    Fails if the auto-shape engine stops threading the resolved threshold
    through (the 2026-09-03 ``learning``-block regression), and — because the
    history here is laid out one session per day — if the window is ever
    re-interpreted as a session count again."""

    SCOPE = "agent-a:telegram"
    #: One session per day, 12 days of history.
    SESSIONS = 12

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _run_and_capture_window(self, plugin_config: dict) -> tuple[int, int]:
        """Seed a FRESH state dir (so the per-scope debounce never suppresses a
        run), run auto_shape_run, and return (window_days,
        sessions_considered) that reached compute_scope_recommendations."""
        state_dir = Path(tempfile.mkdtemp(dir=self.tmp.name)) / "tool-belt"
        _seed_sessions(state_dir, self.SCOPE, self.SESSIONS, spacing=_DAY)
        captured: dict[str, int] = {}
        real = shaping.compute_scope_recommendations

        def spy(*args, **kwargs):
            captured["window_days"] = kwargs["window_days"]
            result = real(*args, **kwargs)
            captured["sessions_considered"] = result.get("sessions_considered")
            return result

        with mock.patch.object(shaping, "compute_scope_recommendations", spy):
            shaping.auto_shape_run(plugin_config, state_dir=state_dir)
        return captured["window_days"], captured["sessions_considered"]

    def test_configured_window_days_changes_the_analysis(self):
        base = {
            "auto_shape": True,
            "channels": {self.SCOPE: {"learned_mode": "apply"}},
        }
        # Default (no override): the shipped 7-day window over 12 days of
        # one-session-per-day history → the newest 8 sessions (day 0 through
        # day 7 inclusive of the cutoff).
        window_default, considered_default = self._run_and_capture_window(base)
        self.assertEqual(window_default, shaping.DEFAULTS["window_days"])
        self.assertEqual(considered_default, 8)

        # Widened via plugin_config: 30 days covers the whole history.
        widened = dict(base)
        widened["learning"] = {"shape_ceiling": {"window_days": 30}}
        window_wide, considered_wide = self._run_and_capture_window(widened)
        self.assertEqual(window_wide, 30)
        self.assertEqual(considered_wide, self.SESSIONS)

        # Tightened: 2 days keeps only the newest 3 sessions.
        tightened = dict(base)
        tightened["learning"] = {"shape_ceiling": {"window_days": 2}}
        window_tight, considered_tight = self._run_and_capture_window(tightened)
        self.assertEqual(window_tight, 2)
        self.assertEqual(considered_tight, 3)


def _seed_unused_carry(state_dir: Path, scope: str, tool: str,
                       n_sessions: int = 20) -> None:
    """N sessions carrying ``tool`` adaptively and never calling it — the
    shape of telemetry that makes it a demote candidate."""
    state_dir.mkdir(parents=True, exist_ok=True)
    newest = time.time()
    preds = []
    for i in range(n_sessions):
        preds.append({
            "ts": newest - (n_sessions - 1 - i) * 60.0,
            "schema_version": 2,
            "scope": scope,
            "session_id": "key",
            "hermes_session_id": f"sess-{i}",
            "prediction_id": f"pred-{i}",
            "always_carry_tools": ["read_file"],
            "carry_tools": [tool],
            "expand_only_tools": [],
            "active_tools": ["read_file", tool],
            "ceiling_tools": ["read_file", tool],
        })
    (state_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8")
    (state_dir / "tool_calls.jsonl").write_text("", encoding="utf-8")


class CliConfigPinTests(unittest.TestCase):
    """``config.yaml`` ``always_carry`` pins must protect a tool on the CLI
    path exactly as they do in process.

    The operator scripts run under whichever interpreter started them, so
    ``hermes_cli.config`` — the runtime's config reader — is usually absent.
    When the loader gave up there, the user layer resolved to ``{}``: the
    protected surface collapsed to the shipped policy baseline and a
    config-pinned tool was reported as demotable.
    """

    SCOPE = "agent-a:telegram"
    PINNED = "pinned_tool"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        (self.home / "config.yaml").write_text(
            "plugins:\n"
            "  entries:\n"
            "    tool-belt:\n"
            "      settings:\n"
            "        always_carry:\n"
            f"          - {self.PINNED}\n"
            "        channels:\n"
            "          agent-a:\n"
            "            telegram:\n"
            "              learned_mode: apply\n",
            encoding="utf-8",
        )
        # The host package must be unavailable, as it is for any script not
        # started with the Hermes interpreter: that is the path under test.
        self._patches = [
            mock.patch.dict(sys.modules, {"hermes_cli": None}),
            mock.patch.dict(os.environ, {"HERMES_HOME": str(self.home)}),
        ]
        for patch in self._patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_config_pins_reach_the_cli_plugin_config(self):
        config = shaping.load_cli_plugin_config()
        self.assertEqual(config.get("always_carry"), [self.PINNED])
        self.assertIn(self.SCOPE, config.get("channels") or {},
                      "channels must arrive flattened to agent:platform keys")

    def test_config_pin_is_protected_from_demotion_on_the_cli_path(self):
        protected = shaping.effective_always_carry(
            shaping.load_cli_plugin_config(), self.SCOPE)
        self.assertIn(self.PINNED, protected)

        state_dir = Path(self.tmp.name) / "state" / "tool-belt"
        _seed_unused_carry(state_dir, self.SCOPE, self.PINNED)
        argv = ["shape-ceiling.py", "--state-dir", str(state_dir),
                "--dry-run", "--json"]
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", argv), redirect_stdout(buf):
            rc = _load_shape_ceiling().main()
        self.assertEqual(rc, 0)
        doc = json.loads(buf.getvalue())
        scopes = {s["scope"]: s for s in doc["scopes"]}
        self.assertIn(self.SCOPE, scopes)
        demoted = {d["tool"] for d in scopes[self.SCOPE]["demote"]}
        self.assertNotIn(
            self.PINNED, demoted,
            "a config.yaml always_carry pin is undemotable on the CLI path too",
        )


def _load_shape_ceiling():
    """The ``shape-ceiling.py`` script as an importable module."""
    name = "tool_belt_shape_ceiling_cli"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent.parent / "scripts" / "shape-ceiling.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
