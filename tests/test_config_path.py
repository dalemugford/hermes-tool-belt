"""Config lives at ``plugins.entries.tool-belt.settings`` — the plugin
settings subtree Hermes reserves for every plugin — with channels nested as
``channels.<agent>.<platform>`` (Hermes forbids ``:`` in a settings key
segment). Locks the on-disk ↔ internal mapping and the legacy-block warning.
"""

from __future__ import annotations

import logging
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401,E402

plugin = sys.modules["tool_belt_plugin"]
from tool_belt_plugin import learned  # noqa: E402


class FlattenChannelsTests(unittest.TestCase):
    def test_nested_agent_platform_becomes_scope_keys(self):
        flat = learned.flatten_channels({
            "bernard": {"slack": {"learned_mode": "apply"},
                        "telegram": {"bypass_rate": 1.0}},
            "sue": {"slack": {"always_carry": ["terminal"]}},
        })
        self.assertEqual(flat, {
            "bernard:slack": {"learned_mode": "apply"},
            "bernard:telegram": {"bypass_rate": 1.0},
            "sue:slack": {"always_carry": ["terminal"]},
        })

    def test_flat_scope_keys_and_bare_platform_pass_through(self):
        flat = learned.flatten_channels({
            "bernard:slack": {"learned_mode": "apply"},
            "telegram": {"learned_mode": "recommend"},  # bare platform
        })
        self.assertEqual(flat, {
            "bernard:slack": {"learned_mode": "apply"},
            "telegram": {"learned_mode": "recommend"},
        })

    def test_garbage_is_ignored(self):
        self.assertEqual(learned.flatten_channels(None), {})
        self.assertEqual(learned.flatten_channels("x"), {})


class _FakeHermesConfig(types.ModuleType):
    def __init__(self, doc):
        super().__init__("hermes_cli.config")
        self._doc = doc

    def load_config(self):
        return self._doc

    @staticmethod
    def cfg_get(cfg, *path, default=None):
        node = cfg
        for part in path:
            node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                return default
        return node


class LoaderTests(unittest.TestCase):
    """``_load_user_config`` reads only the canonical path and reports a
    stale legacy block instead of silently ignoring it."""

    def _load(self, doc):
        fake = _FakeHermesConfig(doc)
        pkg = types.ModuleType("hermes_cli")
        pkg.config = fake
        with mock.patch.dict(sys.modules, {"hermes_cli": pkg,
                                           "hermes_cli.config": fake}), \
                mock.patch.dict(plugin._CONFIG, {}, clear=False), \
                self.assertLogs(plugin.logger.name, level="DEBUG") as logs:
            plugin.logger.debug("probe")  # assertLogs needs ≥1 record
            plugin._load_user_config()
            snapshot = dict(plugin._CONFIG)
        return snapshot, "\n".join(logs.output)

    def test_canonical_path_is_read_and_channels_flattened(self):
        cfg, _ = self._load({"plugins": {"entries": {"tool-belt": {"settings": {
            "agent": "bernard",
            "channels": {"bernard": {"telegram": {"learned_mode": "recommend"}}},
        }}}}})
        self.assertEqual(cfg["agent"], "bernard")
        self.assertEqual(cfg["channels"],
                         {"bernard:telegram": {"learned_mode": "recommend"}})

    def test_legacy_block_is_ignored_with_a_warning(self):
        cfg, out = self._load({"plugins": {"tool-belt": {"agent": "old"}}})
        self.assertNotEqual(cfg.get("agent"), "old", "legacy block must not be read")
        self.assertIn("plugins.entries.tool-belt.settings", out)
        self.assertIn("WARNING", out)

    def test_no_legacy_block_means_no_warning(self):
        _, out = self._load({"plugins": {"entries": {"tool-belt": {"settings": {}}}}})
        self.assertNotIn("WARNING", out)


if __name__ == "__main__":
    unittest.main()
