"""Crown-jewel wire contracts: the deployed promises, end-to-end, zero config.

Every prior incident shared one shape: something upstream removed a capability
before Tool Belt's own logic ran, and no test asserted the end-to-end promise —
the config-default flip (registered-but-inert, 90 min), the demotable bridge
(silent total MCP loss), the deferred expand_tools. These tests drive the REAL
hook path — ``_on_pre_gateway_dispatch`` building state from an empty
HERMES_HOME with no config.yaml and no learned.json, then the real wrapped
``_build_api_kwargs`` — with Hermes faked only at its import seams
(``toolsets``). Telemetry is NOT patched away: the rows and sidecars these
tests assert on are the real writers' output.

Justifications (the sentence each test lives by):
  · ZeroConfigFullStart — protects promises #1/#3/#6/#7 at once: a fresh
    install ships everything, pins and bridge and MCP survive, telemetry is
    honest — and it is the only test that would catch a code-default or
    registration drift that leaves the plugin silently inert or lossy.
  · DemotionRecoveryOnTheWire — protects promise #3 end-to-end: the ONLY test
    where a real learned demotion is recovered through the real registered
    expand_tools handler and re-ships on the wire; every other expansion test
    stops at the handler payload or hand-builds the state.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # — registers the package; holds PRISTINE_CONFIG

plugin = sys.modules["tool_belt_plugin"]
from tool_belt_plugin import expand_tools, learned  # noqa: E402

BRIDGE = ("tool_search", "tool_describe", "tool_call")
SESSION_KEY = "telegram_555000111"


def _tool_def(name: str) -> dict:
    return {"name": name, "description": f"{name} does things",
            "input_schema": {"type": "object", "properties": {}}}


class _WireHome(unittest.TestCase):
    """Fresh Hermes home (path contains a space), real hooks, seam-only fakes."""

    # The full wire ceiling: policy residents, plain built-ins, the bridge,
    # and an MCP passthrough tool.
    NAMES = ["clarify", "expand_tools", "skills_list", "skill_view",
             "read_file", "write_file", "terminal", *BRIDGE, "mcp__probe__x"]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / "fresh hermes home"
        self.home.mkdir()
        env = mock.patch.dict(os.environ, {
            "HERMES_HOME": str(self.home),
            "HERMES_SESSION_KEY": SESSION_KEY,
        }, clear=False)
        env.start(); self.addCleanup(env.stop)

        # Assert against PRISTINE in-code defaults: restore the snapshot
        # conftest took before any sibling test could seed _CONFIG (two older
        # files clear-and-replace it without cleanup — found by this tier).
        self._orig = dict(plugin._CONFIG)
        self.addCleanup(self._restore)
        plugin._CONFIG.clear()
        plugin._CONFIG.update(conftest.PRISTINE_CONFIG)
        plugin._PREDICTION_CV.set(None)
        plugin._STICKY_BY_KEY.clear()
        plugin._PRIOR_MESSAGES_BY_SESSION.clear()
        # A fresh install has no detection evidence and no host-config
        # primary: clear the cross-session/cache-mode/pin state a sibling
        # test may have left, or posture resolution would inherit it and a
        # caching-provider "off" lock would flip this install off carry-all.
        plugin._DETECTION_CACHE.clear()
        plugin._DETECTION_CACHE_LOADED = False
        plugin._CACHE_MODE_BY_SESSION.clear()
        plugin._CACHE_DECISION_BY_SESSION.clear()
        plugin._LAST_CANONICAL_BY_PLATFORM.clear()
        plugin._HOST_MODEL.update(provider="", model="")

        # Hermes toolsets seam — the one allowed fake.
        ts = types.ModuleType("toolsets")
        ts.resolve_toolset = lambda cat: {"file": ["read_file", "write_file"]}.get(cat)
        ts.get_toolset_names = lambda: ["file"]
        seam = mock.patch.dict(sys.modules, {"toolsets": ts})
        seam.start(); self.addCleanup(seam.stop)

    def _restore(self):
        plugin._CONFIG.clear()
        plugin._CONFIG.update(self._orig)

    def _dispatch(self, text="hello there", session_key=SESSION_KEY):
        source = SimpleNamespace(platform=SimpleNamespace(value="telegram"),
                                 chat_id="555")
        event = SimpleNamespace(source=source, text=text, attachments=[])
        store = mock.MagicMock()
        store._generate_session_key.return_value = session_key
        with mock.patch.dict(os.environ, {"HERMES_SESSION_KEY": session_key}):
            plugin._on_pre_gateway_dispatch(event=event, gateway=None,
                                            session_store=store)
        state = plugin._PREDICTION_CV.get()
        self.assertIsNotNone(state, "pre_gateway_dispatch must set state")
        return state

    def _run_wire(self):
        defs = [_tool_def(n) for n in self.NAMES]

        def original(_self, _msgs):
            return {"tools": list(defs), "model": "claude-test"}

        wrapped = plugin._wrap_build_api_kwargs(original)
        result = wrapped(object(), [])
        return {plugin._tool_name(t) for t in result["tools"]}


class ZeroConfigFullStart(_WireHome):
    def test_fresh_install_ships_everything_and_logs_honestly(self):
        # The deployed default itself is part of the assertion — no test
        # setup may have turned the plugin on.
        self.assertIs(plugin._CONFIG.get("enabled"), True,
                      "zero-config install must be ON by default (the "
                      "90-minute registered-but-inert incident)")

        # Zero-config default is cache_mode: auto with no detection evidence,
        # which resolves to the caching-provider carry-all posture (D1): ship
        # the whole ceiling EXCEPT expand_tools (nothing to expand when
        # everything is already carried), and never narrow.
        state = self._dispatch()
        kept = self._run_wire()

        self.assertEqual(kept, set(self.NAMES) - {"expand_tools"},
                         "fresh install carries the entire ceiling — pins, "
                         "unknowns, bridge, and MCP — minus expand_tools")
        self.assertEqual(state.get("policy_source"), "cache_on_carry_all",
                         "the carry-all cohort is stamped distinctly from bypass")
        self.assertIs(state.get("active_tool_names"), plugin.NO_NARROWING)
        self.assertEqual(list(state.get("expand_only_tools") or []), [],
                         "nothing is expand-only under carry-all")

        # Honest telemetry, from the REAL writers into the fresh home.
        pred_file = self.home / "state" / "tool-belt" / "predictions.jsonl"
        self.assertTrue(pred_file.exists(), "prediction row written")
        row = json.loads(pred_file.read_text().splitlines()[-1])
        self.assertEqual(row.get("schema_version"), 2)
        self.assertTrue(row.get("tokens_estimator"),
                        "every row names the estimator that measured it")
        self.assertTrue(row.get("ceiling_tools"))
        sizes_file = self.home / "state" / "tool-belt" / "schema_sizes.json"
        self.assertTrue(sizes_file.exists(),
                        "per-tool schema sizes snapshotted on the hot path")
        self.assertIn("read_file",
                      json.loads(sizes_file.read_text())["tools"])

    def test_fresh_install_on_noncaching_provider_partitions_honestly(self):
        # On a non-caching ("off") provider the full-start partition still
        # applies: nothing demoted before evidence, unknowns in adaptive
        # carry, pins immutable, bridge outside the partition, expand_tools
        # SHIPPED as the recovery valve.
        plugin._CONFIG["cache_mode"] = "off"
        state = self._dispatch()
        kept = self._run_wire()

        self.assertEqual(kept, set(self.NAMES),
                         "non-caching full start ships the entire ceiling, "
                         "expand_tools included")
        self.assertEqual(list(state.get("carry_expand_only") or
                              state.get("expand_only_tools") or []), [],
                         "nothing is expand-only before any evidence exists")
        carry = set(state.get("carry_carry") or [])
        self.assertIn("read_file", carry, "unknown tools start as adaptive carry")
        always = set(state.get("carry_always_carry") or [])
        self.assertIn("clarify", always)
        self.assertIn("expand_tools", always)
        for name in BRIDGE:
            self.assertIn(name, state.get("mcp_passthrough_tools") or [],
                          "bridge rides outside the partition")


class DemotionRecoveryOnTheWire(_WireHome):
    def test_demoted_tool_absent_then_recovered_via_real_handler(self):
        # Demotion and expand_tools recovery are the non-caching ("off")
        # engine (D1: caching providers carry everything and ship no
        # expand_tools), so this exercises the off posture explicitly.
        plugin._CONFIG["cache_mode"] = "off"
        # Learn the real scope string from a first dispatch, then demote
        # read_file through the production writer.
        scope = self._dispatch()["scope"]
        self.assertTrue(scope, "scope must resolve for learned state to apply")
        learned.write_state(
            {"version": 2,
             "scopes": {scope: {"carry": [], "expand_only": ["read_file"]}}},
            self.home / "state" / "tool-belt" / "learned.json")

        # A NEW session: per-session state must not leak between sessions.
        state = self._dispatch("plain message, no triggers",
                               session_key="telegram_555000222")
        kept = self._run_wire()
        # Precondition asserted (meta-rule B): the demotion really bit.
        self.assertIn("read_file", self.NAMES)
        self.assertNotIn("read_file", kept,
                         "PRECONDITION: learned demotion reaches the wire")
        self.assertIn("write_file", kept, "only the demoted tool is absent")

        # Recovery through the REAL registered handler wiring.
        handler = expand_tools.make_handler(plugin._PREDICTION_CV,
                                            sticky_refresh_fn=None)
        payload = json.loads(handler({"tool": "read_file"}))
        self.assertTrue(payload.get("success"), payload)

        kept_after = self._run_wire()
        self.assertIn("read_file", kept_after,
                      "an expanded tool re-ships on the wire for this session")


if __name__ == "__main__":
    unittest.main()
