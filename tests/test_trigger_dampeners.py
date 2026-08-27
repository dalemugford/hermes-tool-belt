"""Regression tests for trigger dampeners and learned-policy merging."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

if "tool_belt_plugin" not in sys.modules:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent.parent))
    sys.path.insert(0, str(here.parent))
    importlib.import_module("tests.conftest")  # registers the plugin package

learned = importlib.import_module("tool_belt_plugin.learned")
presets = importlib.import_module("tool_belt_plugin.presets")

PLUGIN_DIR = Path(__file__).resolve().parent.parent


def _trigger(name: str):
    preset = presets.load_preset_file(PLUGIN_DIR / "policy.yaml")
    return next(group for group in preset.triggers if group.name == name)


class TriggerDampenerTests(unittest.TestCase):
    def assert_fires(self, group_name: str, message: str) -> None:
        self.assertTrue(_trigger(group_name).matches(message, []))

    def assert_suppressed(self, group_name: str, message: str) -> None:
        group = _trigger(group_name)
        self.assertFalse(group.matches(message, []))
        self.assertTrue(group.is_excluded(message))
        self.assertTrue(group.would_fire_positive(message, []))

    def test_file_write_direct_requests_fire_and_discussion_is_suppressed(self):
        self.assert_fires("file_write", "Save notes on this to a markdown file.")
        self.assert_fires("file_write", "Patch `presets.py` with that change.")
        self.assert_suppressed("file_write", "Should we save this as a file later?")
        self.assert_suppressed("file_write", "What do you think about the file-write trigger?")
        self.assert_suppressed("file_write", "Don't patch it yet.")

    def test_delegation_and_cron_dampeners_preserve_direct_actions(self):
        self.assert_fires("delegation", "Delegate this to a coding agent.")
        self.assert_suppressed("delegation", "Should we use subagents for this eventually?")
        self.assert_fires("cronjob", "Schedule a weekly report.")
        self.assert_suppressed("cronjob", "Should we schedule this reminder later?")

    def test_learned_merge_preserves_dampeners(self):
        preset = presets.load_preset_file(PLUGIN_DIR / "policy.yaml")
        original_home = os.environ.get("HERMES_HOME")

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["HERMES_HOME"] = tmpdir
            try:
                state_dir = Path(tmpdir) / "state" / "tool-belt"
                state_dir.mkdir(parents=True)
                payload = {
                    "version": 1,
                    "updated_at": "2026-01-01T00:00:00Z",
                    "scopes": {"assistant-a:telegram": {"always_on": ["browser_navigate"]}},
                    "global": {},
                }
                (state_dir / "learned.json").write_text(json.dumps(payload), encoding="utf-8")
                learned.load_state(force=True)

                result = learned.apply_to_preset(
                    preset,
                    {"learned_mode": "apply"},
                    "assistant-a:telegram",
                )
                self.assertEqual(result.policy_source, "learned")
                file_write = next(group for group in result.preset.triggers if group.name == "file_write")
                self.assertTrue(file_write.exclude_patterns)
                self.assertFalse(file_write.matches("Should we save this as a file later?", []))

                # recommend (default) must NOT merge learned.json.
                rec = learned.apply_to_preset(
                    preset,
                    {"learned_mode": "recommend"},
                    "assistant-a:telegram",
                )
                self.assertEqual(rec.mode, "recommend")
                self.assertEqual(rec.policy_source, "preset")
                self.assertNotIn("browser_navigate", rec.preset.always_on)

                # apply merges the learned always_on tool.
                self.assertIn("browser_navigate", result.preset.always_on)
            finally:
                if original_home is None:
                    os.environ.pop("HERMES_HOME", None)
                else:
                    os.environ["HERMES_HOME"] = original_home
                learned.load_state(force=True)


class LearnedModeNormalizationTests(unittest.TestCase):
    def test_legacy_values_normalize(self):
        # Clean migration: old config values map to the two-value model.
        self.assertEqual(learned.normalize_mode("off"), "recommend")
        self.assertEqual(learned.normalize_mode("auto"), "apply")
        self.assertEqual(learned.normalize_mode("audit"), "apply")
        # Canonical values pass through.
        self.assertEqual(learned.normalize_mode("recommend"), "recommend")
        self.assertEqual(learned.normalize_mode("apply"), "apply")
        # Blank/unknown fall back to the safe default.
        self.assertEqual(learned.normalize_mode(""), "recommend")
        self.assertEqual(learned.normalize_mode(None), "recommend")
        self.assertEqual(learned.normalize_mode("bogus"), "recommend")

    def test_learned_mode_resolution_uses_aliases(self):
        self.assertEqual(learned.learned_mode({"learned_mode": "auto"}, "telegram"), "apply")
        self.assertEqual(learned.learned_mode({"learned_mode": "off"}, "telegram"), "recommend")
        # Per-scope override with a legacy value normalizes too.
        cfg = {"learned_mode": "recommend", "channels": {"telegram": {"learned_mode": "audit"}}}
        self.assertEqual(learned.learned_mode(cfg, "telegram"), "apply")


if __name__ == "__main__":
    unittest.main()
