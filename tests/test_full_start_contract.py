"""Regression locks for the zero-config full-start contract + always_carry pins.

Promise #2 (2026-08-30): on a scope with no learned state, Tool Belt carries
EVERYTHING Hermes enables — ``active == E`` — and only evidence-driven
demotion moves tools to ``expand_only`` over time. This flips the old
invariant "unknown enabled built-ins default to expand_only" and retires the
shipped curated warm-start ``carry`` list from runtime resolution. The
default ``learned_mode`` flips ``recommend`` → ``apply``.

Promise #3 (2026-08-30): per-agent config pinning via
``plugins.entries.tool-belt.settings.always_carry: [...]`` (global, plus additive
``channels.<scope>.always_carry``). Pins union with the shipped structural
baseline, are validated ∩ E at resolution, and are undemotable by
construction.

Every test here was proven to FAIL on the pre-change HEAD (old contract)
before the implementation landed — they are the regression locks for the
new contract.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401,E402

plugin = sys.modules["tool_belt_plugin"]
carrying_mod = importlib.import_module("tool_belt_plugin.carrying")
presets_mod = importlib.import_module("tool_belt_plugin.presets")
learned_mod = importlib.import_module("tool_belt_plugin.learned")
shaping_mod = importlib.import_module("tool_belt_plugin.shaping")

PLUGIN_DIR = Path(__file__).resolve().parent.parent

# Captured at import (discovery) time, before any test can mutate the live
# _CONFIG dict — this is the shipped default, not a test's leftover.
_DEFAULT_LEARNED_MODE = plugin._CONFIG.get("learned_mode")
_DEFAULT_ENABLED = plugin._CONFIG.get("enabled")

CEILING = {
    "clarify", "todo", "expand_tools",          # structural always_carry
    "web_search", "read_file", "terminal",      # formerly curated carry
    "vision_analyze", "image_generate",         # formerly expand_only unknowns
    "mystery_tool",                             # unknown to any policy list
}
POLICY_A = ["clarify", "todo", "expand_tools"]


class TestFullStartDefault(unittest.TestCase):
    """(a)+(b): fresh scope, zero learned state → active == full ceiling."""

    def test_fresh_scope_active_equals_enabled_ceiling(self):
        model = carrying_mod.resolve(
            enabled=CEILING, always_carry=POLICY_A, carry=[],
        )
        self.assertEqual(model.active, set(CEILING),
                         "with no learned state, everything enabled is active")
        self.assertEqual(model.expand_only, set(),
                         "nothing is expand_only until demoted by evidence")

    def test_unknown_enabled_tool_is_carried_not_expand_only(self):
        model = carrying_mod.resolve(
            enabled=CEILING, always_carry=POLICY_A, carry=[],
        )
        self.assertIn("mystery_tool", model.carry,
                      "an unknown enabled built-in is CARRIED until demoted")
        self.assertIn("mystery_tool", model.active)
        self.assertNotIn("mystery_tool", model.expand_only)

    def test_only_demoted_tools_leave_residency(self):
        model = carrying_mod.resolve(
            enabled=CEILING, always_carry=POLICY_A, carry=[],
            demoted=["image_generate", "vision_analyze"],
        )
        self.assertEqual(model.expand_only, {"image_generate", "vision_analyze"})
        self.assertEqual(model.active, set(CEILING) - model.expand_only)
        # Partition stays a genuine three-way split of E.
        self.assertEqual(model.always_carry | model.carry | model.expand_only,
                         set(CEILING))

    def test_learned_carry_wins_over_demotion(self):
        model = carrying_mod.resolve(
            enabled=CEILING, always_carry=POLICY_A,
            carry=["image_generate"], demoted=["image_generate", "terminal"],
        )
        self.assertIn("image_generate", model.carry)
        self.assertEqual(model.expand_only, {"terminal"})

    def test_resolve_preset_fresh_scope_has_no_demotions(self):
        """End-to-end: empty learned state resolves to a full-start preset."""
        with tempfile.TemporaryDirectory() as home:
            with mock.patch.dict(os.environ, {"HERMES_HOME": home}):
                learned_mod.load_state(force=True)
                preset = presets_mod.resolve_preset({"learned_mode": "apply"}, "fresh:cli")
                self.assertEqual(list(getattr(preset, "demoted", []) or []), [],
                                 "fresh scope carries everything — no demotions")
                model = carrying_mod.resolve(
                    enabled=CEILING,
                    always_carry=preset.always_carry,
                    carry=preset.carry,
                    demoted=getattr(preset, "demoted", []),
                )
                self.assertEqual(model.active, set(CEILING))

    def test_shipped_policy_carry_list_retired_from_runtime(self):
        """policy.yaml no longer ships a runtime ``carry`` list — the curated
        warm-start is retired; residency comes from E minus demotions."""
        base = presets_mod.load_base_policy()
        self.assertEqual(list(base.carry), [],
                         "shipped policy must not seed a curated carry list")


class TestConfigAlwaysCarry(unittest.TestCase):
    """(d)+(e)+(f): per-agent always_carry config pinning."""

    def _preset(self, plugin_config, scope="agent:cli", home=None):
        learned_mod.load_state(force=True)
        return presets_mod.resolve_preset(plugin_config, scope)

    def test_config_pin_joins_effective_always_carry(self):
        with tempfile.TemporaryDirectory() as home:
            with mock.patch.dict(os.environ, {"HERMES_HOME": home}):
                preset = self._preset({"always_carry": ["web_search"]})
                self.assertIn("web_search", preset.always_carry)
                # Shipped structural baseline still present.
                self.assertIn("clarify", preset.always_carry)

    def test_scope_pin_unions_never_removes(self):
        cfg = {
            "always_carry": ["web_search"],
            "channels": {"agent:cli": {"always_carry": ["terminal"]}},
        }
        with tempfile.TemporaryDirectory() as home:
            with mock.patch.dict(os.environ, {"HERMES_HOME": home}):
                preset = self._preset(cfg, scope="agent:cli")
                self.assertIn("web_search", preset.always_carry)
                self.assertIn("terminal", preset.always_carry)

    def test_per_profile_configs_resolve_differently(self):
        with tempfile.TemporaryDirectory() as home:
            with mock.patch.dict(os.environ, {"HERMES_HOME": home}):
                a = self._preset({"always_carry": ["web_search"]}, "bernard:cli")
                b = self._preset({"always_carry": ["vision_analyze"]}, "sue:cli")
                self.assertIn("web_search", a.always_carry)
                self.assertNotIn("web_search", b.always_carry)
                self.assertIn("vision_analyze", b.always_carry)

    def test_pin_survives_learned_demotion(self):
        """A learned demotion naming a config-pinned tool is ignored."""
        with tempfile.TemporaryDirectory() as home:
            state_dir = Path(home) / "state" / "tool-belt"
            state_dir.mkdir(parents=True)
            (state_dir / "learned.json").write_text(json.dumps({
                "version": 2,
                "scopes": {"agent:cli": {"carry": [],
                                         "expand_only": ["web_search", "image_generate"],
                                         "shaping": {}}},
            }))
            with mock.patch.dict(os.environ, {"HERMES_HOME": home}):
                preset = self._preset(
                    {"learned_mode": "apply", "always_carry": ["web_search"]},
                    scope="agent:cli",
                )
                self.assertIn("web_search", preset.always_carry)
                self.assertNotIn("web_search", getattr(preset, "demoted", []))
                self.assertIn("image_generate", getattr(preset, "demoted", []))
                model = carrying_mod.resolve(
                    enabled=CEILING,
                    always_carry=preset.always_carry,
                    carry=preset.carry,
                    demoted=getattr(preset, "demoted", []),
                )
                self.assertIn("web_search", model.active)
                self.assertNotIn("web_search", model.expand_only)

    def test_pin_survives_auto_shape_apply(self):
        """The auto-shape engine never writes a demotion for a pinned tool."""
        plugin_config = {"always_carry": ["web_search"]}
        recs = {
            "agent:cli": {
                "scope": "agent:cli",
                "computed_at": "2026-08-30T00:00:00Z",
                "sessions_considered": 25,
                "window_requested": 20,
                "promote": [],
                "demote": [
                    {"tool": "web_search", "sessions_without_use": 25,
                     "evidence": "carry_unused"},
                    {"tool": "image_generate", "sessions_without_use": 25,
                     "evidence": "carry_unused"},
                ],
                "enabled_tool_names": sorted(CEILING),
            }
        }
        filtered = shaping_mod.filter_protected_demotions(plugin_config, recs)
        demoted = [d["tool"] for d in filtered["agent:cli"]["demote"]]
        self.assertNotIn("web_search", demoted,
                         "config-pinned tool is undemotable by construction")
        self.assertIn("image_generate", demoted)

        state, changes = shaping_mod.apply_recommendations(
            {"version": 2, "scopes": {}}, filtered
        )
        entry = state["scopes"]["agent:cli"]
        self.assertNotIn("web_search", entry["expand_only"])
        self.assertIn("image_generate", entry["expand_only"])

    def test_disabled_pin_stays_absent_with_warning(self):
        """A pinned tool absent from E cannot enter through the list."""
        with tempfile.TemporaryDirectory() as home:
            with mock.patch.dict(os.environ, {"HERMES_HOME": home}):
                preset = self._preset({"always_carry": ["ghost_tool"]})
                model = carrying_mod.resolve(
                    enabled=CEILING,
                    always_carry=preset.always_carry,
                    carry=preset.carry,
                    demoted=getattr(preset, "demoted", []),
                )
                self.assertNotIn("ghost_tool", model.active)
                self.assertNotIn("ghost_tool", model.always_carry)
                # The runtime warns once, naming the inert pin.
                with self.assertLogs("tool_belt_plugin", level="WARNING") as cm:
                    plugin._warn_inert_pins(
                        "agent:cli", ["ghost_tool", "web_search"], CEILING
                    )
                self.assertTrue(any("ghost_tool" in line for line in cm.output))
                self.assertFalse(any("web_search'" in line for line in cm.output))


class TestEnabledByDefault(unittest.TestCase):
    """Zero-config activation: an unset ``enabled`` key must not silently
    disable every hook while expand_tools stays registered (2026-08-30
    live incident)."""

    def test_internal_enabled_default_is_true(self):
        self.assertIs(_DEFAULT_ENABLED, True,
                      "hooks must be live with no `enabled` key in config")

    def test_config_disabled_logs_one_shot_warning(self):
        class _Ctx:
            def register_tool(self, **kwargs):
                pass
            def register_hook(self, *a, **k):
                pass
            def register_cli_command(self, *a, **k):
                pass
            def __getattr__(self, name):
                return lambda *a, **k: None

        old = dict(plugin._CONFIG)
        try:
            with mock.patch.object(plugin, "_load_user_config", lambda: None):
                plugin._CONFIG["enabled"] = False
                with self.assertLogs("tool_belt_plugin", level="WARNING") as cm:
                    plugin.register(_Ctx())
                self.assertTrue(any("disabled by config" in line for line in cm.output))
        finally:
            plugin._CONFIG.clear()
            plugin._CONFIG.update(old)


if __name__ == "__main__":
    unittest.main()
