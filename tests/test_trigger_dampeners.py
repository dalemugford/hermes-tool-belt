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


class _TriggerAssertions:
    """Shared trigger assertions for the policy-keyword test cases."""

    def assert_fires(self, group_name: str, message: str) -> None:
        self.assertTrue(_trigger(group_name).matches(message, []))

    def assert_suppressed(self, group_name: str, message: str) -> None:
        group = _trigger(group_name)
        self.assertFalse(group.matches(message, []))
        self.assertTrue(group.is_excluded(message))
        self.assertTrue(group.would_fire_positive(message, []))

    def assert_quiet(self, group_name: str, message: str) -> None:
        """No keyword matches at all — distinct from a dampened match."""
        group = _trigger(group_name)
        self.assertFalse(
            group.would_fire_positive(message, []),
            f"{group_name} should not match {message!r}",
        )
        self.assertFalse(group.matches(message, []))


class TriggerDampenerTests(_TriggerAssertions, unittest.TestCase):
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

    def test_code_execution_bulk_counts_use_a_numeral(self):
        # The policy shipped a literal ``N pages`` placeholder that could only
        # ever match the letter N; the intent was a count.
        self.assert_fires("code_execution", "Summarise 40 pages of this PDF.")
        self.assert_fires("code_execution", "Rename 12 files in that folder.")
        self.assert_fires("code_execution", "Pull 500 rows out of the export.")

    def test_batch_belongs_to_code_execution_only(self):
        # `batch` used to sit in both groups, so one word loaded execute_code
        # *and* 36 mnemosyne schemas.
        self.assert_fires("code_execution", "Send these in one batch, please.")
        self.assert_quiet("mnemosyne_extended", "Send these in one batch, please.")

    def test_learned_merge_preserves_dampeners(self):
        preset = presets.load_preset_file(PLUGIN_DIR / "policy.yaml")
        original_home = os.environ.get("HERMES_HOME")

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["HERMES_HOME"] = tmpdir
            try:
                state_dir = Path(tmpdir) / "state" / "tool-belt"
                state_dir.mkdir(parents=True)
                # A v1 learned file (``always_on``) — exercised through the
                # v1→v2 read normalization: on load it maps to a ``carry``
                # promotion for the scope.
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
                self.assertNotIn("browser_navigate", rec.preset.carry)

                # apply promotes the learned tool into adaptive carry residency
                # (v1 always_on → v2 carry), never into the immutable baseline.
                self.assertIn("browser_navigate", result.preset.carry)
                self.assertNotIn("browser_navigate", result.preset.always_carry)
            finally:
                if original_home is None:
                    os.environ.pop("HERMES_HOME", None)
                else:
                    os.environ["HERMES_HOME"] = original_home
                learned.load_state(force=True)


class TriggerKeywordPrecisionTests(_TriggerAssertions, unittest.TestCase):
    """Everyday words must not load specialist tool groups.

    Each negative here is a false positive the shipped policy had: the
    interjection "ha", a bare `sync`/`batch`/`graph`/`diagnose`, and any
    mention of "history".
    """

    def test_home_assistant_requires_the_product_not_the_interjection(self):
        self.assert_fires("homeassistant", "Ask Home Assistant for the porch light state.")
        self.assert_fires("homeassistant", "homeassistant is offline again")
        self.assert_fires("homeassistant", "hass keeps dropping the zigbee stick")
        self.assert_fires("homeassistant", "Call the HA service for the hallway switch.")
        # …but the acronym alone, lower-cased or laughing, must stay quiet.
        self.assert_quiet("homeassistant", "ha")
        self.assert_quiet("homeassistant", "ha ha, that's a good one")
        self.assert_quiet("homeassistant", "haha nice")
        self.assert_quiet("homeassistant", "ha! the state machine bit us again")

    def test_mnemosyne_extended_needs_a_memory_co_word(self):
        self.assert_fires("mnemosyne_extended", "Run mnemosyne diagnose on Sue's bank.")
        self.assert_fires("mnemosyne_extended", "Push the memory sync to the other machine.")
        self.assert_fires("mnemosyne_extended", "Sync my memories across the fleet.")
        self.assert_fires("mnemosyne_extended", "Write that to the scratchpad.")
        self.assert_fires("mnemosyne_extended", "Add it to the knowledge graph.")
        self.assert_fires("mnemosyne_extended", "Promote that to a persona fact.")
        self.assert_fires("mnemosyne_extended", "Export memories to a file.")
        # The 36-schema group must not load on ordinary developer English.
        self.assert_quiet("mnemosyne_extended", "sync the repo before you start")
        self.assert_quiet("mnemosyne_extended", "diagnose the failing build")
        self.assert_quiet("mnemosyne_extended", "show me a graph of the savings")
        self.assert_quiet("mnemosyne_extended", "batch those API calls")
        self.assert_quiet("mnemosyne_extended", "triple check the numbers")
        self.assert_quiet("mnemosyne_extended", "use the canonical path here")

    def test_history_search_needs_a_conversational_qualifier(self):
        self.assert_fires("history_search", "Search my chat history for that link.")
        self.assert_fires("history_search", "Pull up our conversation history from Tuesday.")
        self.assert_fires("history_search", "Search my history for the pricing thread.")
        self.assert_fires("history_search", "What did we decide about pricing yesterday?")
        # Every other kind of history is somebody else's.
        self.assert_quiet("history_search", "check the git history for that line")
        self.assert_quiet("history_search", "clear my browser history")
        self.assert_quiet("history_search", "the history of this project is messy")
        self.assert_suppressed("history_search", "search the git history for that commit")


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
