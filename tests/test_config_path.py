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


class MixedAgentBlockTests(unittest.TestCase):
    """Review find C3: `channels.<agent>.always_carry` next to nested
    platform entries used to demote the whole agent to a bare-platform key,
    silently disabling its platform overrides. Agent-level scalars are now
    defaults for every platform; the platform entry wins on conflict."""

    def test_agent_level_scalars_default_into_each_platform(self):
        flat = learned.flatten_channels({"bernard": {
            "always_carry": ["terminal"],
            "learned_mode": "recommend",
            "telegram": {"learned_mode": "apply"},
            "slack": {},
        }})
        self.assertEqual(flat, {
            "bernard:telegram": {"always_carry": ["terminal"],
                                 "learned_mode": "apply"},
            "bernard:slack": {"always_carry": ["terminal"],
                              "learned_mode": "recommend"},
        })
        # And configure.py's degraded-mode mirror agrees.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "tb_configure_mirror",
            Path(__file__).resolve().parent.parent / "scripts" / "configure.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # dataclasses need the module registered
        spec.loader.exec_module(mod)
        with mock.patch.object(mod, "load_learned", lambda: None):
            self.assertEqual(mod._flatten_channels({"bernard": {
                "always_carry": ["terminal"],
                "telegram": {"learned_mode": "apply"}}}),
                {"bernard:telegram": {"always_carry": ["terminal"],
                                      "learned_mode": "apply"}})


class HarvestReplayConfigTests(unittest.TestCase):
    """Review find C2: harvest-replay loaded the raw settings block without
    flattening channels, so every per-channel override was ignored during
    replay back-tests."""

    def test_replay_config_flattens_nested_channels(self):
        import importlib.util, tempfile
        spec = importlib.util.spec_from_file_location(
            "tb_harvest_replay",
            Path(__file__).resolve().parent.parent / "scripts" / "harvest-replay.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # dataclasses need the module registered
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "config.yaml").write_text(
                "plugins:\n  entries:\n    tool-belt:\n      settings:\n"
                "        channels:\n          default:\n            telegram:\n"
                "              learned_mode: recommend\n", encoding="utf-8")
            cfg = mod._load_plugin_config(home)
        self.assertEqual(cfg["channels"],
                         {"default:telegram": {"learned_mode": "recommend"}})


class SchemaMirrorsDefaultsTests(unittest.TestCase):
    """plugin.yaml config_schema must declare every top-level _CONFIG key
    with a matching default (TESTING.md meta-rule: deployed defaults are
    pinned, not assumed)."""

    def test_every_config_key_is_declared_with_its_default(self):
        import yaml
        manifest = yaml.safe_load(
            (Path(__file__).resolve().parent.parent / "plugin.yaml").read_text())
        schema = manifest["config_schema"]
        missing = set(plugin._CONFIG) - set(schema)
        self.assertEqual(missing, set(), f"undeclared settings: {sorted(missing)}")
        from tool_belt_plugin import shaping
        # Keys resolved outside _CONFIG: compare against the live resolver.
        resolved = {"auto_shape_interval_hours":
                    shaping.auto_shape_interval_hours({}, "any:scope")}
        for key, spec in schema.items():
            if "default" in spec:
                actual = plugin._CONFIG.get(key, resolved.get(key))
                self.assertEqual(spec["default"], actual,
                                 f"schema default for {key} drifted from code")


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

    def test_learning_shape_ceiling_is_copied_into_config(self):
        """Regression (2026-09-03): the ``learning`` block must survive
        ``_load_user_config`` — it was missing from the copy allowlist, so a
        user-configured ``learning.shape_ceiling`` never reached
        ``_CONFIG`` and the auto-shaper silently ran policy defaults."""
        cfg, _ = self._load({"plugins": {"entries": {"tool-belt": {"settings": {
            "agent": "bernard",
            "learning": {"shape_ceiling": {"session_window": 30,
                                           "demote_min_sessions_no_use": 8}},
        }}}}})
        self.assertEqual(cfg.get("learning"),
                         {"shape_ceiling": {"session_window": 30,
                                            "demote_min_sessions_no_use": 8}})

    def test_learning_non_dict_is_dropped_fail_open(self):
        cfg, _ = self._load({"plugins": {"entries": {"tool-belt": {"settings": {
            "agent": "bernard",
            "learning": "tighten everything",
        }}}}})
        self.assertNotIn("learning", cfg,
                         "non-dict learning must not enter _CONFIG; the "
                         "shaper falls back to the policy layer")

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
