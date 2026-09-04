"""User-configurable shaper thresholds via ``config.yaml``.

The five ``learning.shape_ceiling`` keys resolve across three layers, per
key, highest first: ``config.yaml`` (the user layer, passed as
``plugin_config``) → ``policy.yaml`` (shipped preset) → ``shaping.DEFAULTS``.

These tests lock the precedence contract and its fail-open behaviour (a bad
user value degrades to the policy layer, never raises), that the legacy
no-arg call site keeps working unchanged, and — end to end — that a tighter
``session_window`` set through ``plugin_config`` actually reaches the
shaper's analysis and shrinks the window it considers.

Everything runs against throwaway files; no live state is touched.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import conftest  # noqa: F401,E402 — registers the tool_belt_plugin package

plugin = sys.modules["tool_belt_plugin"]
from tool_belt_plugin import shaping  # noqa: E402

# A policy layer deliberately distinct from both DEFAULTS and the config
# layer, so a value can be traced to exactly one source.
_POLICY_YAML = """\
name: test-policy
learning:
  shape_ceiling:
    session_window: 77
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
        # config.yaml wins where set (session_window, demote_k); the other
        # three fall through to the policy layer, none to DEFAULTS.
        resolved = shaping.load_shape_ceiling_defaults(
            policy_path=self.policy,
            plugin_config=_cfg({"session_window": 30, "demote_k": 1.1}),
        )
        self.assertEqual(resolved["session_window"], 30)   # config layer
        self.assertEqual(resolved["demote_k"], 1.1)        # config layer
        self.assertEqual(resolved["promote_min_sessions"], 4)   # policy layer
        self.assertEqual(resolved["promote_min_calls"], 6)      # policy layer
        self.assertEqual(resolved["demote_min_sessions_no_use"], 12)  # policy
        # And these differ from the hardcoded fallback, proving the layering.
        self.assertNotEqual(resolved["session_window"],
                            shaping.DEFAULTS["session_window"])
        self.assertNotEqual(resolved["promote_min_calls"],
                            shaping.DEFAULTS["promote_min_calls"])

    def test_no_config_resolves_policy_layer(self):
        resolved = shaping.load_shape_ceiling_defaults(policy_path=self.policy)
        self.assertEqual(resolved["session_window"], 77)
        self.assertEqual(resolved["demote_k"], 2.5)

    def test_empty_learning_block_is_policy_layer(self):
        for cfg in ({}, {"learning": {}}, {"learning": {"shape_ceiling": {}}}):
            resolved = shaping.load_shape_ceiling_defaults(
                policy_path=self.policy, plugin_config=cfg)
            self.assertEqual(resolved["session_window"], 77, cfg)
            self.assertEqual(resolved["demote_k"], 2.5, cfg)

    def test_bad_int_values_fall_back_to_policy_without_raising(self):
        # Zero, negative, non-numeric, and wrong-type all fail the positive-int
        # gate and degrade to the policy value (77) — never raising.
        for bad in (0, -5, "nope", None, [30], {"x": 1}):
            resolved = shaping.load_shape_ceiling_defaults(
                policy_path=self.policy,
                plugin_config=_cfg({"session_window": bad}),
            )
            self.assertEqual(resolved["session_window"], 77, bad)

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
            self.assertEqual(resolved["session_window"], 77, cfg)

    def test_config_over_defaults_when_policy_missing(self):
        # Policy file absent → DEFAULTS is the base; config still overrides it.
        missing = Path(self.tmp.name) / "does-not-exist.yaml"
        resolved = shaping.load_shape_ceiling_defaults(
            policy_path=missing, plugin_config=_cfg({"session_window": 25}))
        self.assertEqual(resolved["session_window"], 25)
        # unset key comes from DEFAULTS (no policy layer to consult)
        self.assertEqual(resolved["demote_k"], shaping.DEFAULTS["demote_k"])


class BackwardCompatTests(unittest.TestCase):
    def test_noarg_call_matches_shipped_policy(self):
        # The legacy no-arg signature keeps its exact prior behaviour: the
        # shipped policy.yaml thresholds, no user layer.
        self.assertEqual(shaping.load_shape_ceiling_defaults(), {
            "session_window": 100, "promote_min_sessions": 2,
            "promote_min_calls": 3, "demote_min_sessions_no_use": 20,
            "demote_k": 1.5,
        })

    def test_policy_path_only_call_still_works(self):
        # A caller passing only policy_path (the old keyword) is unaffected.
        with tempfile.TemporaryDirectory() as tmp:
            policy = _write_policy(Path(tmp))
            resolved = shaping.load_shape_ceiling_defaults(policy_path=policy)
        self.assertEqual(resolved["session_window"], 77)


def _seed_sessions(state_dir: Path, scope: str, n_sessions: int) -> None:
    """N distinct sessions, each with one prediction + one expansion call for
    ``grep_files`` — enough for the shaper to run its analysis per scope."""
    state_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    preds, calls = [], []
    for i in range(n_sessions):
        pid = f"pred-{i}"
        preds.append({
            "ts": now + i,
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
            "ts": now + i,
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
    """A tightened session_window set via plugin_config reaches the shaper's
    analysis: the window it actually applies shrinks below the history size."""

    SCOPE = "agent-a:telegram"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _run_and_capture_window(self, plugin_config: dict) -> tuple[int, int]:
        """Seed a FRESH state dir (so the per-scope debounce never suppresses a
        run), run auto_shape_run, and return (window_requested,
        sessions_considered) that reached compute_scope_recommendations."""
        state_dir = Path(tempfile.mkdtemp(dir=self.tmp.name)) / "tool-belt"
        _seed_sessions(state_dir, self.SCOPE, n_sessions=40)
        captured: dict[str, int] = {}
        real = shaping.compute_scope_recommendations

        def spy(*args, **kwargs):
            captured["window"] = kwargs["window"]
            result = real(*args, **kwargs)
            captured["sessions_considered"] = result.get("sessions_considered")
            return result

        with mock.patch.object(shaping, "compute_scope_recommendations", spy):
            shaping.auto_shape_run(plugin_config, state_dir=state_dir)
        return captured["window"], captured["sessions_considered"]

    def test_tightened_window_shrinks_the_analysis(self):
        base = {
            "auto_shape": True,
            "channels": {self.SCOPE: {"learned_mode": "apply"}},
        }
        # Default (no override): the full history of 40 sessions is considered.
        window_default, considered_default = self._run_and_capture_window(base)
        self.assertEqual(window_default, shaping.DEFAULTS["session_window"])
        self.assertEqual(considered_default, 40)

        # Tightened via plugin_config: the window drops to 30 and the shaper
        # now considers only the 30 most recent sessions.
        tightened = dict(base)
        tightened["learning"] = {"shape_ceiling": {"session_window": 30}}
        window_tight, considered_tight = self._run_and_capture_window(tightened)
        self.assertEqual(window_tight, 30)
        self.assertEqual(considered_tight, 30)


if __name__ == "__main__":
    unittest.main()
