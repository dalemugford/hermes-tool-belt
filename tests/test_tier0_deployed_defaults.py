"""Tier 0: the shipped artifacts and code defaults ARE the deployed promise.

The 90-minute outage was not a logic bug — it was drift between an in-code
default and the promise the docs made, invisible to every test that set its
config explicitly. These two tests read the real shipped files and the real
in-code defaults with no test-local overrides.

Justifications:
  · deployed defaults — one dense sweep of every promise-bearing default
    (enabled, learned_mode, cache_mode, auto-shape, logging) across their
    SEPARATE homes (in-code _CONFIG, learned.py, shaping.py); the only test
    that fails when a refactor moves a default out from under the promise.
  · policy integrity — the shipped policy.yaml IS product behavior: the
    audited pin set exact (an eighth pin sneaking in is also a contract
    change), thresholds match the documented levers, no unknown keys, every
    trigger regex compiles; the only test that reads the file as a contract
    rather than as fixture input.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest

plugin = sys.modules["tool_belt_plugin"]
from tool_belt_plugin import learned, logger_io, presets, shaping  # noqa: E402

AUDITED_PINS = {"expand_tools", "tool_search", "tool_describe", "tool_call",
                "clarify", "skills_list", "skill_view"}


class DeployedDefaults(unittest.TestCase):
    def setUp(self):
        # Assert against the PRISTINE in-code defaults (snapshotted by
        # conftest before any test could mutate the module dict).
        self._live = dict(plugin._CONFIG)
        plugin._CONFIG.clear()
        plugin._CONFIG.update(conftest.PRISTINE_CONFIG)
        self.addCleanup(lambda: (plugin._CONFIG.clear(),
                                 plugin._CONFIG.update(self._live)))

    def test_promise_bearing_defaults_all_homes(self):
        # Promise: install → ON. (The incident: in-code False + config unset.)
        self.assertIs(plugin._CONFIG.get("enabled"), True)
        self.assertIs(plugin._CONFIG.get("log"), True,
                      "telemetry on by default — savings need evidence")
        self.assertEqual(plugin._CONFIG.get("cache_mode"), "auto")
        # Promise: shaping lands automatically (apply is the default mode).
        self.assertEqual(learned.DEFAULT_MODE, "apply")
        self.assertEqual(learned.learned_mode({}, "any:scope"), "apply")
        # Promise: auto-shape runs itself with a daily cadence per scope.
        self.assertEqual(shaping.AUTO_SHAPE_DEFAULT_INTERVAL_HOURS, 24.0)
        # Economic levers ship at their documented values.
        self.assertEqual(shaping.DEFAULTS["window_days"], 7)
        self.assertEqual(shaping.DEFAULTS["demote_min_sessions_no_use"], 2)
        self.assertEqual(shaping.DEFAULTS["demote_k"], 1.5)
        self.assertEqual(shaping.INVENTORY_GRACE_DAYS, 7)
        self.assertEqual(logger_io.DEFAULT_PER_TOOL_TOKENS, 388)


class PolicyIntegrity(unittest.TestCase):
    def test_shipped_policy_yaml_is_the_contract(self):
        policy_path = Path(plugin.__file__).parent / "policy.yaml"
        preset = presets.load_preset_file(policy_path)  # raises on bad shape

        self.assertEqual(set(preset.always_carry), AUDITED_PINS,
                         "the audited pin set, exactly — additions are "
                         "contract changes too")
        self.assertEqual(preset.carry, [],
                         "full-start: the policy ships NO warm-start carry "
                         "list; a fresh scope carries all of E")
        self.assertFalse(preset.no_narrowing)
        self.assertTrue(preset.triggers, "trigger groups shipped")
        for group in preset.triggers:
            for pat in list(group.keyword_patterns) + list(group.exclude_patterns):
                self.assertTrue(hasattr(pat, "search"),
                                f"trigger pattern in {group.name} must be a "
                                "compiled regex")
        # The two independent copies of the pins must agree (drift guard is
        # test_telemetry_schema.FallbackBaselineDriftTests; this asserts the
        # thresholds' single home the same way).
        self.assertEqual(shaping.load_shape_ceiling_defaults(), {
            "window_days": 7, "promote_min_sessions": 1,
            "promote_min_calls": 2, "demote_min_sessions_no_use": 2,
            "demote_k": 1.5,
        }, "policy.yaml thresholds match the documented levers")


if __name__ == "__main__":
    unittest.main()
