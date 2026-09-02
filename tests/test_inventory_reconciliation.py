"""Promise #4 — automatic inventory reconciliation.

Locks the auto-management contract for tools that vanish from (or join) the
install's registry:

  · a tool absent from the REGISTRY for the 7-day grace is pruned from
    learned.json (carry/expand_only/shaping/overlay references) AND from the
    config always_carry pins, with the config write going through the same
    machinery `hermes config set` uses (atomic YAML replace of
    ``$HERMES_HOME/config.yaml``);
  · absence is tracked from first observation — inside the grace nothing is
    pruned;
  · a tool that reappears within the grace resets the clock;
  · a registry that can't be resolved fails OPEN — nothing is touched;
  · a config-write failure logs a warning, skips the pin, and never
    propagates;
  · a NEW tool joining an ALREADY-SHAPED scope comes up carried (full-start).

Everything runs against a throwaway ``HERMES_HOME``; no live state is
touched.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import conftest  # noqa: F401,E402 — registers the tool_belt_plugin package

shaping = importlib.import_module("tool_belt_plugin.shaping")
learned = importlib.import_module("tool_belt_plugin.learned")
presets = importlib.import_module("tool_belt_plugin.presets")
carrying = importlib.import_module("tool_belt_plugin.carrying")

SCOPE = "agent-a:telegram"
GHOST = "ghost_tool"
LIVE = "read_file"

GRACE_SECONDS = shaping.INVENTORY_GRACE_DAYS * 86400.0


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


class ReconcileBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.state_dir = self.home / "state" / "tool-belt"
        self.state_dir.mkdir(parents=True)
        env = mock.patch.dict(os.environ, {"HERMES_HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)
        self.now = time.time()

    def seed_learned(self, *, ghost_in_overlay: bool = True):
        doc = {
            "version": 2,
            "scopes": {
                SCOPE: {
                    "carry": [GHOST, LIVE],
                    "expand_only": ["grep_files"],
                    "shaping": {
                        "promote": [{"tool": GHOST, "sessions": 3, "calls": 5}],
                        "demote": [{"tool": "grep_files", "sessions_without_use": 20}],
                    },
                }
            },
        }
        if ghost_in_overlay:
            doc["scopes"][SCOPE]["triggers"] = [
                {"name": f"auto:{GHOST}", "tools": [GHOST],
                 "keywords": [r"\bghost\b"], "source": "name_tokens"},
                {"name": "auto:shared", "tools": [GHOST, "grep_files"],
                 "keywords": [r"\bshared\b"], "source": "mined"},
            ]
        (self.state_dir / "learned.json").write_text(
            json.dumps(doc), encoding="utf-8")

    def seed_config_yaml(self):
        (self.home / "config.yaml").write_text(
            "plugins:\n"
            "  entries:\n"
            "    tool-belt:\n"
            "      settings:\n"
            "        enabled: true\n"
            f"        always_carry: [{GHOST}, {LIVE}]\n"
            "        channels:\n"
            f"          {SCOPE.split(':')[0]}:\n"
            f"            {SCOPE.split(':')[1]}:\n"
            f"              always_carry: [{GHOST}]\n",
            encoding="utf-8",
        )

    def seed_inventory(self, age_seconds: float):
        (self.state_dir / "inventory.json").write_text(
            json.dumps({"missing_since": {GHOST: _iso(self.now - age_seconds)}}),
            encoding="utf-8",
        )

    def plugin_config(self):
        return {
            "enabled": True,
            "always_carry": [GHOST, LIVE],
            "channels": {SCOPE: {"always_carry": [GHOST], "learned_mode": "apply"}},
        }

    def learned_doc(self):
        return json.loads((self.state_dir / "learned.json").read_text(encoding="utf-8"))

    def inventory_doc(self):
        return json.loads((self.state_dir / "inventory.json").read_text(encoding="utf-8"))


class GracePruneTests(ReconcileBase):
    """Time-travel fixture: the ghost has been missing for 8 days."""

    def test_expired_ghost_is_pruned_from_learned_and_config(self):
        self.seed_learned()
        self.seed_config_yaml()
        self.seed_inventory(GRACE_SECONDS + 86400)  # 8 days missing
        cfg = self.plugin_config()

        summary = shaping.reconcile_inventory(
            cfg, self.state_dir, now=self.now, registry_names={LIVE, "grep_files"},
        )
        self.assertEqual(summary["pruned"], [GHOST])

        doc = self.learned_doc()
        entry = doc["scopes"][SCOPE]
        self.assertNotIn(GHOST, entry["carry"])
        self.assertIn(LIVE, entry["carry"])
        # Shaping evidence rows naming the ghost are gone; others survive.
        self.assertEqual(entry["shaping"]["promote"], [])
        self.assertEqual(len(entry["shaping"]["demote"]), 1)
        # Overlay: ghost-only entry dropped entirely; shared entry keeps the
        # surviving tool.
        overlay = entry.get("triggers") or []
        names = {g["name"] for g in overlay}
        self.assertNotIn(f"auto:{GHOST}", names)
        shared = [g for g in overlay if g["name"] == "auto:shared"]
        self.assertEqual(shared[0]["tools"], ["grep_files"])

        # Config pins removed on disk — global AND per-channel lists.
        text = (self.home / "config.yaml").read_text(encoding="utf-8")
        self.assertNotIn(GHOST, text)
        self.assertIn(LIVE, text)
        # In-memory plugin config kept consistent.
        self.assertNotIn(GHOST, cfg["always_carry"])
        # Absence tracking cleared — a later reappearance is a fresh journey.
        self.assertEqual(self.inventory_doc()["missing_since"], {})

    def test_prune_logs_one_info_line_per_removal(self):
        self.seed_learned()
        self.seed_config_yaml()
        self.seed_inventory(GRACE_SECONDS + 86400)
        with self.assertLogs("tool_belt_plugin.shaping", level="INFO") as captured:
            shaping.reconcile_inventory(
                self.plugin_config(), self.state_dir, now=self.now,
                registry_names={LIVE, "grep_files"},
            )
        text = "\n".join(captured.output)
        self.assertIn("pruned vanished tool", text)
        self.assertIn("removed stale always_carry config pin", text)

    def test_within_grace_nothing_is_pruned(self):
        self.seed_learned()
        self.seed_config_yaml()
        self.seed_inventory(86400)  # only 1 day missing
        summary = shaping.reconcile_inventory(
            self.plugin_config(), self.state_dir, now=self.now,
            registry_names={LIVE, "grep_files"},
        )
        self.assertEqual(summary["pruned"], [])
        self.assertIn(GHOST, self.learned_doc()["scopes"][SCOPE]["carry"])
        self.assertIn(GHOST, (self.home / "config.yaml").read_text(encoding="utf-8"))
        self.assertIn(GHOST, self.inventory_doc()["missing_since"])

    def test_first_observed_absence_starts_the_clock(self):
        self.seed_learned()
        summary = shaping.reconcile_inventory(
            self.plugin_config(), self.state_dir, now=self.now,
            registry_names={LIVE, "grep_files"},
        )
        self.assertEqual(summary["pruned"], [])
        self.assertIn(GHOST, self.inventory_doc()["missing_since"])

    def test_reappearance_within_grace_resets_the_clock(self):
        self.seed_learned()
        self.seed_inventory(GRACE_SECONDS - 3600)  # nearly expired
        summary = shaping.reconcile_inventory(
            self.plugin_config(), self.state_dir, now=self.now,
            registry_names={LIVE, GHOST, "grep_files"},  # ghost is back
        )
        self.assertEqual(summary["pruned"], [])
        self.assertEqual(self.inventory_doc()["missing_since"], {})
        # Learned state keeps its journey untouched.
        self.assertIn(GHOST, self.learned_doc()["scopes"][SCOPE]["carry"])

    def test_unresolvable_registry_fails_open(self):
        self.seed_learned()
        self.seed_inventory(GRACE_SECONDS + 86400)
        summary = shaping.reconcile_inventory(
            self.plugin_config(), self.state_dir, now=self.now, registry_names=None,
        )
        self.assertEqual(summary["status"], "registry_unavailable")
        self.assertIn(GHOST, self.learned_doc()["scopes"][SCOPE]["carry"])

    def test_config_write_failure_warns_and_skips(self):
        self.seed_learned()
        self.seed_config_yaml()
        self.seed_inventory(GRACE_SECONDS + 86400)

        def broken_remover(tool):
            raise RuntimeError("machinery down")

        with self.assertLogs("tool_belt_plugin.shaping", level="WARNING") as captured:
            summary = shaping.reconcile_inventory(
                self.plugin_config(), self.state_dir, now=self.now,
                registry_names={LIVE, "grep_files"},
                config_pin_remover=broken_remover,
            )
        self.assertIn("could not remove stale config pin", "\n".join(captured.output))
        # Pin survives on disk; the absence clock is kept so a later pass retries.
        self.assertIn(GHOST, (self.home / "config.yaml").read_text(encoding="utf-8"))
        self.assertIn(GHOST, self.inventory_doc()["missing_since"])
        self.assertNotIn(GHOST, summary["pruned"])


class AutoPassIntegrationTests(ReconcileBase):
    """Reconciliation rides the session-end auto pass, fail-open."""

    def test_auto_shape_run_reconciles_before_shaping(self):
        self.seed_learned()
        self.seed_inventory(GRACE_SECONDS + 86400)
        cfg = self.plugin_config()
        with mock.patch.object(
            shaping, "registry_tool_names",
            return_value={LIVE, "grep_files"},
        ):
            summary = shaping.auto_shape_run(cfg, self.state_dir, now=self.now)
        self.assertEqual(summary.get("inventory", {}).get("pruned"), [GHOST])
        self.assertNotIn(GHOST, self.learned_doc()["scopes"][SCOPE]["carry"])

    def test_reconcile_exception_never_blocks_the_pass(self):
        self.seed_learned()
        with mock.patch.object(
            shaping, "reconcile_inventory", side_effect=RuntimeError("boom"),
        ):
            summary = shaping.auto_shape_run(
                self.plugin_config(), self.state_dir, now=self.now)
        self.assertEqual(summary.get("inventory", {}).get("status"), "error")


class NewToolFullStartTests(unittest.TestCase):
    """Promise #4(a) regression lock: a NEW tool joining an ALREADY-SHAPED
    scope comes up carried — full-start needs no code for arrivals."""

    def test_new_tool_in_shaped_scope_is_carried(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = Path(tmp.name)
        state_dir = home / "state" / "tool-belt"
        state_dir.mkdir(parents=True)
        (state_dir / "learned.json").write_text(json.dumps({
            "version": 2,
            "scopes": {SCOPE: {
                "carry": [],
                "expand_only": ["grep_files"],  # the scope IS shaped
                "shaping": {"source": "auto"},
            }},
        }), encoding="utf-8")
        with mock.patch.dict(os.environ, {"HERMES_HOME": str(home)}):
            learned.load_state(force=True)
            preset = presets.resolve_preset(
                {"enabled": True, "learned_mode": "apply"}, SCOPE)
            model = carrying.resolve(
                enabled=["read_file", "grep_files", "brand_new_tool"],
                always_carry=preset.always_carry,
                carry=preset.carry,
                demoted=preset.demoted,
            )
        self.assertIn("grep_files", model.expand_only)   # shaping still holds
        self.assertIn("brand_new_tool", model.carry)     # the arrival is carried
        self.assertIn("brand_new_tool", model.active)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
