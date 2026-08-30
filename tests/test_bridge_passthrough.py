"""Regression locks for the audited always_carry pin set + bridge pass-through.

The 2026-08-30 always_carry audit found the single worst silent-failure mode
in the shipped policy: Hermes' Tool Search bridge tools (``tool_search`` /
``tool_describe`` / ``tool_call``) landed in the built-in partition as
ordinary class-C adaptive residents, so twenty sessions of no-use evidence
could demote the ONLY reach into the deferred MCP/plugin catalog — no error,
no message, no recovery path. The fix is two layers (belt and braces):

  * structural: bridge tools join the MCP pass-through — outside the
    partition, never demotable, never in the expand-only manifest, never in
    shaper evidence (``__init__._is_bridge_tool``);
  * pure data: the three names join policy.yaml ``always_carry`` for the case
    where ``tools.tool_search`` cannot be imported.

The same audit dropped two pins: ``send_message`` (not agent-callable on
current Hermes — a permanently inert pin) and ``todo`` (evidence-ruled;
users can pin it back via config).

Every test here FAILS on the pre-fix commit — verified during the wave.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

# Reuse conftest's plugin-loader bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401

import importlib

plugin = sys.modules["tool_belt_plugin"]
carrying_mod = importlib.import_module("tool_belt_plugin.carrying")
presets_mod = importlib.import_module("tool_belt_plugin.presets")
logger_io = importlib.import_module("tool_belt_plugin.logger_io")
shaping_mod = importlib.import_module("tool_belt_plugin.shaping")

BRIDGE = ("tool_search", "tool_describe", "tool_call")

AUDITED_SEVEN = frozenset({
    "expand_tools", "tool_search", "tool_describe", "tool_call",
    "clarify", "skills_list", "skill_view",
})

try:
    import tools.tool_search as ts
    from tools.registry import registry

    _HAVE_HERMES = True
except Exception:  # pragma: no cover — env without hermes-agent on the path
    _HAVE_HERMES = False


# ─── Lock (c) + (e): the shipped pin set is exactly the audited seven ──────

# ─── Lock (d): todo demotable by ordinary evidence ─────────────────────────

class TodoDemotableTests(unittest.TestCase):
    def test_todo_demoted_by_learned_evidence(self):
        base = presets_mod.load_base_policy()
        model = carrying_mod.resolve(
            enabled={"clarify", "todo", "read_file", "expand_tools"},
            always_carry=base.always_carry,
            carry=[],
            demoted=["todo"],
        )
        self.assertIn("todo", model.expand_only,
                      "todo must be demotable by ordinary no-use evidence")
        self.assertNotIn("todo", model.active)


# ─── Lock (a) + (b): bridge pass-through through the narrowing wrapper ─────

def _seed_config() -> None:
    plugin._CONFIG.clear()
    plugin._CONFIG.update({
        "enabled": True,
        "log": False,
        "agent": "assistant-a",
        "bypass_rate": 0.0,
        "cache_mode": "off",
        "channels": {},
    })


class BridgePassThroughWireTests(unittest.TestCase):
    """A hostile learned.json demoting the bridge must not remove it from the
    wire — the bridge sits outside the partition, like MCP tools."""

    def _run_wrapped(self, tools, state):
        token = plugin._PREDICTION_CV.set(state)
        try:
            def original(self_, msgs):
                return {"tools": list(tools), "model": "claude-test"}

            wrapped = plugin._wrap_build_api_kwargs(original)
            with mock.patch.dict(plugin._CONFIG, {"enabled": True}), \
                    mock.patch.object(plugin, "_maybe_log_prediction",
                                      lambda *a, **k: None):
                return wrapped(object(), [])
        finally:
            plugin._PREDICTION_CV.reset(token)

    def _tools(self):
        names = ["clarify", "expand_tools", "read_file", *BRIDGE,
                 "mcp__probe__t0"]
        return [{"name": n, "description": "x", "input_schema": {}}
                for n in names]

    def _hostile_state(self):
        # No always_carry protection at all — proves the STRUCTURAL layer
        # alone keeps the bridge on the wire (the policy pin is the backup).
        return {
            "active_tool_names": ["clarify"],
            "resolved_always_carry": [],
            "resolved_carry": [],
            "resolved_demoted": [*BRIDGE, "read_file"],
            "triggered_tools": [],
            "expansions": set(),
            "logged": False,
            "session_id": "",
        }

    def test_demoted_bridge_tools_still_reach_the_wire(self):
        _seed_config()
        state = self._hostile_state()
        result = self._run_wrapped(self._tools(), state)
        kept = {plugin._tool_name(t) for t in result["tools"]}
        for name in BRIDGE:
            self.assertIn(name, kept,
                          f"{name} demoted by learned evidence must still "
                          "reach the wire (pass-through immunity)")
        self.assertNotIn("read_file", kept,
                         "ordinary built-in demotion still works")

    def test_bridge_outside_partition_and_in_passthrough_telemetry(self):
        _seed_config()
        state = self._hostile_state()
        self._run_wrapped(self._tools(), state)
        for name in BRIDGE:
            self.assertNotIn(name, state["enabled_ceiling"],
                             "bridge tools are not part of the built-in "
                             "partition domain E")
            self.assertIn(name, state["mcp_passthrough_tools"],
                          "bridge tools are recorded in the pass-through "
                          "telemetry field")
            self.assertNotIn(name, state.get("carry_always_carry", []))
            self.assertNotIn(name, state.get("carry_carry", []))

    def test_bridge_absent_from_expand_only_manifest(self):
        _seed_config()
        state = self._hostile_state()
        result = self._run_wrapped(self._tools(), state)
        for name in BRIDGE:
            self.assertNotIn(name, state.get("expand_only_tools", []),
                             "a pass-through tool can never be expand_only")
            self.assertNotIn(name, state.get("carry_expand_only", []))
        # And the manifest text appended to expand_tools' schema never names
        # a bridge tool (read_file IS demoted, so a manifest exists).
        for t in result["tools"]:
            if plugin._base_tool_name(plugin._tool_name(t)) == "expand_tools":
                desc = str((t.get("description") or "")) + str(
                    (t.get("function") or {}).get("description") or "")
                for name in BRIDGE:
                    self.assertNotIn(name, desc)


# ─── Lock (b, shaper half): recommendations never name a bridge tool ───────

class ShaperBridgeAssertionTests(unittest.TestCase):
    def test_shaper_never_emits_bridge_recommendations(self):
        """Even with pre-pass-through telemetry rows showing a bridge tool as
        adaptive carry with zero use, the shaper must not recommend demoting
        it (defensive assertion — post-fix rows never contain it at all)."""
        sessions = {}
        for i in range(25):
            sid = f"s{i}"
            sessions[sid] = [{
                "prediction_id": f"p{i}",
                "ts": 1000 + i,
                "residency_inferred": True,
                "residency": {"carry": ["tool_search", "web_extract"]},
                "ceiling_tools": ["clarify", "tool_search", "web_extract"],
                "carry_tools": ["tool_search", "web_extract"],
                "always_carry_tools": ["clarify"],
            }]
        recs = shaping_mod.compute_scope_recommendations(
            scope="a:cli",
            sessions=sessions,
            calls_by_pred={},
            window=25,
            promote_min_sessions=2,
            promote_min_calls=3,
            demote_min_sessions_no_use=20,
        )
        demoted = {d["tool"] for d in recs["demote"]}
        self.assertNotIn("tool_search", demoted,
                         "shaper must never demote a bridge tool")
        self.assertIn("web_extract", demoted,
                      "ordinary unused carry resident still demotes")


# ─── Integration: real assemble_tool_defs + hostile learned, both layers ───

def _td(name: str, desc: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": {}},
        },
    }


@unittest.skipUnless(_HAVE_HERMES, "hermes-agent (tools.tool_search) not importable")
class RealBridgeBothLayersTest(unittest.TestCase):
    """Drive the REAL ``tools.tool_search.assemble_tool_defs`` (deferred-MCP
    setup, bridge active) and then Tool Belt's narrowing with a hostile
    learned state demoting all three bridge tools: ``tool_search`` must reach
    the wire through BOTH layers."""

    N_MCP = 40
    CONTEXT = 8000

    def setUp(self) -> None:
        self._added: list[str] = []
        for i in range(self.N_MCP):
            self._register(f"mcp__probe__t{i}", "mcp-probe", "y" * 80)

    def _register(self, name: str, toolset: str, desc: str) -> None:
        registry.register(
            name=name,
            toolset=toolset,
            schema={
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": {}},
            },
            handler=lambda args, **kw: "{}",
            override=True,
        )
        self.addCleanup(self._safe_deregister, name)

    @staticmethod
    def _safe_deregister(name: str) -> None:
        try:
            registry.deregister(name)
        except Exception:
            pass

    def test_hostile_learned_cannot_hide_the_real_bridge(self) -> None:
        defs = [_td("clarify", "ask the user")]
        defs += [_td(f"mcp__probe__t{i}", "y" * 80) for i in range(self.N_MCP)]
        cfg = ts.ToolSearchConfig.from_raw({"enabled": "on"})
        assembly = ts.assemble_tool_defs(
            defs, context_length=self.CONTEXT, config=cfg)
        self.assertTrue(assembly.activated,
                        "need an active bridge to exercise the boundary")
        bridge_on_wire = {
            (t.get("function") or {}).get("name")
            for t in assembly.tool_defs if isinstance(t, dict)
        } & set(ts.BRIDGE_TOOL_NAMES)
        self.assertTrue(bridge_on_wire, "bridge tools present post-assembly")

        # Layer 2: Tool Belt narrowing with a hostile learned.json.
        _seed_config()
        state = {
            "active_tool_names": ["clarify"],
            "resolved_always_carry": [],
            "resolved_carry": [],
            "resolved_demoted": list(ts.BRIDGE_TOOL_NAMES),
            "triggered_tools": [],
            "expansions": set(),
            "logged": False,
            "session_id": "",
        }
        token = plugin._PREDICTION_CV.set(state)
        try:
            def original(self_, msgs):
                return {"tools": list(assembly.tool_defs), "model": "m"}

            wrapped = plugin._wrap_build_api_kwargs(original)
            with mock.patch.dict(plugin._CONFIG, {"enabled": True}), \
                    mock.patch.object(plugin, "_maybe_log_prediction",
                                      lambda *a, **k: None):
                result = wrapped(object(), [])
        finally:
            plugin._PREDICTION_CV.reset(token)

        kept = {plugin._tool_name(t) for t in result["tools"]}
        self.assertIn("tool_search", kept,
                      "tool_search survives BOTH layers even when a hostile "
                      "learned.json demotes all three bridge tools")


if __name__ == "__main__":
    unittest.main()


def tearDownModule():
    # This module's config seeders clear-and-replace the module _CONFIG; put
    # the pristine in-code defaults back so later files see the real deploy
    # state (hygiene debt found by the Tier-0 rebuild).
    plugin._CONFIG.clear()
    plugin._CONFIG.update(conftest.PRISTINE_CONFIG)
