"""Direct unit coverage for the v2 learned overlay.

Exercises ``learned.py`` at the module level (no configure/shaper flow):

  * ``apply_to_preset`` immunity — ``expand_only`` naming an ``always_carry``
    tool leaves the tool resident and warns (it is immutable);
  * ``apply_to_preset`` overlap belt-and-braces — a hand-built scope dict with
    a ``carry`` ∩ ``expand_only`` overlap resolves toward carry and warns;
  * ``write_state`` round-trip — atomic replace, v1 → v2 normalization on write,
    and preservation of unrelated top-level / per-scope metadata.

``write_state`` is the single production writer of ``learned.json`` — the
shaper's ``merge_into_learned`` and configure's flows persist through it —
so its atomicity contract (unique temp name, fsync, cleanup on failure) is
pinned here too.
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

if "tool_belt_plugin" not in sys.modules:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent.parent))
    sys.path.insert(0, str(here.parent))
    importlib.import_module("tests.conftest")  # registers the plugin package

learned = importlib.import_module("tool_belt_plugin.learned")
presets = importlib.import_module("tool_belt_plugin.presets")

_LOGGER = "tool_belt_plugin.learned"


class LearnedV2TestCase(unittest.TestCase):
    def setUp(self) -> None:
        # learned.py caches learned.json by mtime; drop it so state written by
        # one test can't leak into another through the cache.
        learned._CACHE.clear()
        self.addCleanup(learned._CACHE.clear)


class ApplyToPresetImmunityTests(LearnedV2TestCase):
    def test_expand_only_naming_always_carry_tool_is_ignored_and_warns(self) -> None:
        """always_carry is immutable: a demotion signal naming it is dropped."""
        preset = presets.Preset(
            name="t",
            always_carry=["send_message", "expand_tools"],
            carry=["read_file"],
            triggers=[],
        )
        state = {"version": 2, "scopes": {"a:slack": {"expand_only": ["send_message"]}}}

        with mock.patch.object(learned, "load_state", return_value=state):
            with self.assertLogs(_LOGGER, level="WARNING") as cm:
                result = learned.apply_to_preset(
                    preset, {"learned_mode": "apply"}, "a:slack"
                )

        # The tool stays resident on the immutable surface, never demoted.
        self.assertIn("send_message", result.preset.always_carry)
        self.assertNotIn("send_message", result.preset.carry)
        warned = "\n".join(cm.output)
        self.assertIn("always_carry", warned)
        self.assertIn("send_message", warned)


class ApplyToPresetOverlapTests(LearnedV2TestCase):
    def test_carry_expand_only_overlap_resolves_toward_carry_and_warns(self) -> None:
        """A hand-built scope dict naming a tool in both lists: carry wins.

        ``normalize_state`` reconciles this at load time, so to reach the
        belt-and-braces guard inside ``apply_to_preset`` the (un-normalized)
        scope dict is injected directly via ``load_state``.
        """
        preset = presets.Preset(
            name="t",
            always_carry=["send_message"],
            carry=["read_file", "process"],
            triggers=[],
        )
        state = {
            "version": 2,
            "scopes": {
                "a:slack": {"carry": ["web_search"], "expand_only": ["web_search", "process"]}
            },
        }

        with mock.patch.object(learned, "load_state", return_value=state):
            with self.assertLogs(_LOGGER, level="WARNING") as cm:
                result = learned.apply_to_preset(
                    preset, {"learned_mode": "apply"}, "a:slack"
                )

        # web_search was in both lists → carry wins (resident).
        self.assertIn("web_search", result.preset.carry)
        # The genuine demote (process, expand_only only) still applied.
        self.assertNotIn("process", result.preset.carry)
        warned = "\n".join(cm.output)
        self.assertIn("both carry and expand_only", warned)
        self.assertIn("web_search", warned)


class WriteStateRoundTripTests(LearnedV2TestCase):
    def test_round_trip_normalizes_v1_and_preserves_unrelated_metadata(self) -> None:
        state = {
            "version": 1,
            "provenance": "seed-run-abc",  # unrelated top-level key
            "scopes": {
                "a:slack": {
                    "always_on": ["read_file"],              # v1 → carry
                    "always_off": ["process"],               # v1 → expand_only
                    "cache_aware": {"scope": "a:slack"},     # v1 → shaping
                    "notes": "hand-edited, keep me",         # unrelated per-scope key
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"HERMES_HOME": tmp}):
                learned.write_state(state)
                path = learned.learned_path()
                on_disk = json.loads(path.read_text(encoding="utf-8"))
                tmp_path = path.with_suffix(path.suffix + ".tmp")

                # Atomic replace: the final file is in place, no .tmp left behind.
                self.assertTrue(path.exists())
                self.assertFalse(tmp_path.exists())

                # Normalized to v2 on write.
                self.assertEqual(on_disk["version"], 2)
                self.assertTrue(on_disk["updated_at"])
                entry = on_disk["scopes"]["a:slack"]
                self.assertEqual(entry["carry"], ["read_file"])
                self.assertEqual(entry["expand_only"], ["process"])
                self.assertEqual(entry["shaping"], {"scope": "a:slack"})

                # v1 spellings are renamed, not duplicated.
                for stale in ("always_on", "always_off", "cache_aware"):
                    self.assertNotIn(stale, entry)

                # Unrelated metadata passes through untouched (top-level + per-scope).
                self.assertEqual(on_disk["provenance"], "seed-run-abc")
                self.assertEqual(entry["notes"], "hand-edited, keep me")

    def test_unique_temp_name_never_clobbers_a_concurrent_writers_temp(self) -> None:
        """Regression: the old fixed ``learned.json.tmp`` temp name meant two
        concurrent writers clobbered each other's temp file. The writer must
        use a uniquely named temp, so a pre-existing fixed-name temp (standing
        in for a concurrent writer's file) survives the write untouched."""
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"HERMES_HOME": tmp}):
                path = learned.learned_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                decoy = path.with_suffix(path.suffix + ".tmp")
                decoy.write_text("concurrent writer's in-flight temp", encoding="utf-8")

                learned.write_state({"version": 2, "scopes": {"a:slack": {"carry": ["x"]}}})

                self.assertTrue(path.exists())
                self.assertEqual(decoy.read_text(encoding="utf-8"),
                                 "concurrent writer's in-flight temp",
                                 "a concurrent writer's temp file must survive untouched")
                decoy.unlink()
                leftovers = [p for p in path.parent.iterdir() if ".tmp" in p.name]
                self.assertEqual(leftovers, [], "no temp debris after a successful write")

    def test_failed_write_cleans_up_temp_and_preserves_target(self) -> None:
        """Regression: a failure mid-write (fsync here) must propagate, remove
        the temp file, and leave the existing target byte-identical."""
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"HERMES_HOME": tmp}):
                learned.write_state({"version": 2, "scopes": {"a:slack": {"carry": ["x"]}}})
                path = learned.learned_path()
                before = path.read_bytes()

                with mock.patch.object(learned.os, "fsync",
                                       side_effect=OSError("disk full")):
                    with self.assertRaises(OSError):
                        learned.write_state(
                            {"version": 2, "scopes": {"a:slack": {"carry": ["y"]}}}
                        )

                self.assertEqual(path.read_bytes(), before,
                                 "a failed write must not touch the target")
                leftovers = [p for p in path.parent.iterdir() if ".tmp" in p.name]
                self.assertEqual(leftovers, [],
                                 "a failed write must clean up its temp file")


if __name__ == "__main__":
    unittest.main()
