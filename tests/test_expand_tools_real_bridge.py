"""Real-bridge regression LOCK for the expand_tools pin.

Why this exists as a separate test: the rest of the suite feeds tool-belt
a *synthetic* tool list that
already contains ``expand_tools`` -- so every test verified "given expand_tools
is present, tool-belt keeps it", which was always true. The production bug was
one layer up: Hermes' native tiered-disclosure bridge
(``model_tools._compute_tool_definitions`` -> ``tools.tool_search.assemble_tool_defs``)
deferred ``expand_tools`` behind ``tool_search``/``describe``/``call`` *before*
tool-belt's narrowing ever saw it. No unit test touched that boundary, so it
went unnoticed until live telemetry showed ``expand_tools`` missing from every
turn's ceiling.

This test drives the REAL ``tools.tool_search.assemble_tool_defs`` with
``expand_tools`` registered under the ``tool-belt`` toolset (as the plugin does
at load), and asserts BOTH directions so it is a genuine regression lock that
fails on pre-fix code:

  * WITHOUT the pin  -> the real bridge defers ``expand_tools`` (reproduces the bug),
  * WITH the pin     -> ``expand_tools`` survives on the wire, the MCP surface
                        still defers, and the bridge is still active.

It is guarded with ``skipUnless`` so the suite stays portable where hermes-agent
isn't importable; it gives real coverage in the install and in CI where Hermes
is present.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Reuse conftest's plugin-loader bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401
import unittest
import types

plugin = sys.modules["tool_belt_plugin"]
EXPAND_TOOLS_NAME = plugin.expand_tools_mod.SCHEMA["name"]

try:
    import tools.tool_search as ts
    from tools.registry import registry

    _HAVE_HERMES = True
except Exception:  # pragma: no cover — env without hermes-agent on the path
    _HAVE_HERMES = False


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
class ExpandToolsRealBridgeTest(unittest.TestCase):
    # Enough deferrable MCP surface (at a small context) to force the bridge to
    # activate — mirrors the manual venv reproduction.
    N_MCP = 40
    CONTEXT = 8000

    def setUp(self) -> None:
        # Establish a clean, fully-unwrapped classifier as the per-test baseline
        # and restore it afterward, so a pin installed by one test can never leak
        # into another test (or another test file).
        original = ts.is_deferrable_tool_name
        while getattr(original, "_tool_belt_expand_pin", False) and hasattr(
            original, "__wrapped__"
        ):
            original = original.__wrapped__
        self._clean_classifier = original
        ts.is_deferrable_tool_name = original
        self.addCleanup(setattr, ts, "is_deferrable_tool_name", original)

        # The bridge's classifier consults the registry, so expand_tools and the
        # MCP surface must actually be registered for the run to be realistic.
        # Track only what we add, and only remove what we added, so a real
        # registration (if any) is never disturbed.
        self._added: list[str] = []
        if registry.get_entry(EXPAND_TOOLS_NAME) is None:
            self._register(EXPAND_TOOLS_NAME, "tool-belt", "Load deferred toolsets on demand")
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
        self._added.append(name)
        self.addCleanup(self._safe_deregister, name)

    @staticmethod
    def _safe_deregister(name: str) -> None:
        try:
            registry.deregister(name)
        except Exception:
            pass

    def _assemble(self) -> "ts.AssemblyResult":
        defs = [_td(EXPAND_TOOLS_NAME, "Load deferred toolsets on demand")]
        defs += [_td(f"mcp__probe__t{i}", "y" * 80) for i in range(self.N_MCP)]
        cfg = ts.ToolSearchConfig.from_raw({"enabled": "on"})
        return ts.assemble_tool_defs(defs, context_length=self.CONTEXT, config=cfg)

    @staticmethod
    def _wire_names(assembly) -> set[str]:
        return {
            (t.get("function") or {}).get("name")
            for t in assembly.tool_defs
            if isinstance(t, dict)
        }

    def test_bug_reproduces_without_the_pin(self) -> None:
        """Pre-fix behaviour: the real bridge defers expand_tools.

        This is the half that makes the whole thing a regression lock — it must
        fail (expand_tools present) the day the pin regresses.
        """
        self.assertFalse(
            getattr(ts.is_deferrable_tool_name, "_tool_belt_expand_pin", False),
            "baseline classifier must be unpinned for this direction",
        )
        # The bridge only classifies expand_tools as deferrable because it is a
        # non-core, non-MCP registry tool — assert that precondition directly.
        self.assertTrue(ts.is_deferrable_tool_name(EXPAND_TOOLS_NAME))

        assembly = self._assemble()
        self.assertTrue(assembly.activated, "need an active bridge to exercise deferral")
        wire = self._wire_names(assembly)
        self.assertNotIn(EXPAND_TOOLS_NAME, wire)  # THE BUG
        self.assertTrue(ts.BRIDGE_TOOL_NAMES & wire)  # deferred behind the bridge tools

    def test_pin_keeps_expand_tools_on_the_wire(self) -> None:
        """Post-fix behaviour: the pin keeps expand_tools visible end to end."""
        plugin._pin_expand_tools_visible()
        self.assertTrue(
            getattr(ts.is_deferrable_tool_name, "_tool_belt_expand_pin", False),
            "pin did not attach to the real classifier",
        )

        assembly = self._assemble()
        self.assertTrue(assembly.activated, "bridge should still activate with the pin")
        wire = self._wire_names(assembly)
        self.assertIn(EXPAND_TOOLS_NAME, wire)  # survives — the fix
        self.assertNotIn("mcp__probe__t0", wire)  # MCP surface still deferred
        self.assertTrue(ts.BRIDGE_TOOL_NAMES & wire)  # bridge still active




class PinFailOpenTest(unittest.TestCase):
    """Fail-open lock (from the retired fake-bridge file): a broken/absent
    ``tools.tool_search`` must never make the pin raise — the one case the
    real-bridge tests above cannot produce in this environment."""

    def test_fail_open_when_tool_search_unimportable(self) -> None:
        # A non-package ``tools`` with no importable ``tool_search`` submodule
        # makes ``import tools.tool_search`` raise — the pin must swallow it.
        fake_tools = types.ModuleType("tools")  # no __path__ → not a package
        saved = {k: v for k, v in sys.modules.items()
                 if k == "tools" or k.startswith("tools.")}
        self.addCleanup(sys.modules.update, saved)
        for name in list(saved):
            sys.modules.pop(name, None)
        sys.modules["tools"] = fake_tools

        try:
            plugin._pin_expand_tools_visible()
        except Exception as exc:  # pragma: no cover — the point of the test
            self.fail(f"_pin_expand_tools_visible must fail open, raised: {exc!r}")
        finally:
            sys.modules.pop("tools", None)
            sys.modules.update(saved)

        # Failing open means doing NOTHING — "did not raise" alone would also
        # pass if the pin had attached itself while the import was broken.
        # Once the real module is reachable again its classifier must still be
        # unpinned.
        if _HAVE_HERMES:
            self.assertFalse(
                getattr(ts.is_deferrable_tool_name, "_tool_belt_expand_pin", False),
                "a failed pin attempt must leave the real classifier untouched",
            )
        else:
            self.assertNotIn("tools.tool_search", sys.modules,
                             "the failed import left no half-pinned module behind")

if __name__ == "__main__":
    unittest.main()
