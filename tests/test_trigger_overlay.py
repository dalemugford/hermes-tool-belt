"""Promise #5 — learned trigger overlay (automatic anticipation).

Locks the additive, activation-only trigger overlay:

  · expansion-evidence keyword mining is AUTO-APPLIED above the strict bar
    (support ≥ OVERLAY_MINE_MIN_SUPPORT, precision ≥
    OVERLAY_MINE_MIN_PRECISION) and withheld below it;
  · demotion of a tool with no trigger coverage mints a conservative
    name-token trigger; the stoplist / minimum length filter can veto it
    entirely;
  · overlay triggers union with the shipped policy triggers at a REAL preset
    resolution and activate the tool end-to-end through carrying.resolve;
  · the overlay is structurally unable to reference a tool outside E or to
    touch always_carry;
  · a scope reset clears the overlay; configure --status reports the count.

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
TOOL = "ledger_export"
PHRASE = "quarterly ledger rollup"


def _sessions_with_evidence(evidence_count: int, noise_count: int = 6):
    """Build (sessions, calls_by_pred) — evidence previews all share PHRASE.

    All evidence lands in ONE session so the promote path (which needs 2
    distinct sessions) never fires and the tool stays expand_only.
    """
    now = time.time()
    sessions: dict[str, list[dict]] = {"sess-1": []}
    calls_by_pred: dict[str, list[dict]] = {}
    for i in range(evidence_count):
        pid = f"ev-{i}"
        sessions["sess-1"].append({
            "ts": now + i,
            "prediction_id": pid,
            "message_preview": f"please run the {PHRASE} for {i}",
        })
        calls_by_pred[pid] = [{
            "tool_name": TOOL, "activated_by_expansion": True,
            "prediction_id": pid,
        }]
    for i in range(noise_count):
        pid = f"noise-{i}"
        sessions["sess-1"].append({
            "ts": now + 100 + i,
            "prediction_id": pid,
            "message_preview": f"unrelated chatter about weather {i}",
        })
    return sessions, calls_by_pred


class MiningBarTests(unittest.TestCase):
    def _updates(self, evidence_count):
        sessions, calls = _sessions_with_evidence(evidence_count)
        return shaping.compute_overlay_updates(
            scope=SCOPE,
            scope_entry={"carry": [], "expand_only": [TOOL], "shaping": {}},
            sessions=sessions,
            calls_by_pred=calls,
            protected=set(),
            newly_demoted=[],
            policy_preset=presets.Preset(name="t"),
        )

    def test_mined_candidate_above_bar_is_auto_applied(self):
        updates = self._updates(shaping.OVERLAY_MINE_MIN_SUPPORT)
        mined = [u for u in updates if u["source"] == "mined"]
        self.assertEqual(len(mined), 1)
        self.assertEqual(mined[0]["tools"], [TOOL])
        self.assertTrue(mined[0]["keywords"])
        # The mined keyword is a word-boundary regex matching the phrase.
        import re
        self.assertTrue(any(
            re.search(k, f"do the {PHRASE} now", re.IGNORECASE)
            for k in mined[0]["keywords"]
        ))

    def test_candidate_below_support_bar_is_withheld(self):
        updates = self._updates(shaping.OVERLAY_MINE_MIN_SUPPORT - 1)
        self.assertEqual([u for u in updates if u["source"] == "mined"], [])

    def test_protected_tool_never_gets_an_overlay_entry(self):
        sessions, calls = _sessions_with_evidence(shaping.OVERLAY_MINE_MIN_SUPPORT)
        updates = shaping.compute_overlay_updates(
            scope=SCOPE,
            scope_entry={"carry": [], "expand_only": [TOOL], "shaping": {}},
            sessions=sessions,
            calls_by_pred=calls,
            protected={TOOL},  # always_carry ∪ config pins
            newly_demoted=[TOOL],
            policy_preset=presets.Preset(name="t"),
        )
        self.assertEqual(updates, [])


class NameTokenTests(unittest.TestCase):
    def test_distinctive_tokens_become_word_boundary_regexes(self):
        kws = shaping.name_token_keywords("mnemosyne_diagnose")
        self.assertIn(r"\bmnemosyne\b", kws)
        self.assertIn(r"\bdiagnose\b", kws)

    def test_stoplist_and_min_length_filter_generic_tokens(self):
        # Every token is generic (stoplist) or too short — skip entirely.
        self.assertEqual(shaping.name_token_keywords("get_list_run"), [])
        self.assertEqual(shaping.name_token_keywords("set_x_y"), [])
        # Mixed: only the distinctive token survives.
        kws = shaping.name_token_keywords("get_mnemosyne_list")
        self.assertEqual(kws, [r"\bmnemosyne\b"])

    def _mint(self, tool):
        return shaping.compute_overlay_updates(
            scope=SCOPE,
            scope_entry={"carry": [], "expand_only": [tool], "shaping": {}},
            sessions={}, calls_by_pred={},
            protected=set(),
            newly_demoted=[tool],
            policy_preset=presets.Preset(name="t"),
        )

    def test_uncovered_demotion_mints_a_name_token_trigger(self):
        updates = self._mint("mnemosyne_diagnose")
        minted = [u for u in updates if u["source"] == "name_tokens"]
        self.assertEqual(len(minted), 1)
        self.assertEqual(minted[0]["tools"], ["mnemosyne_diagnose"])
        self.assertIn(r"\bmnemosyne\b", minted[0]["keywords"])
        # A name with no distinctive token yields nothing to trigger on, so
        # derivation is skipped entirely rather than minting an empty group.
        self.assertEqual(self._mint("get_list"), [])

    def test_covered_demotion_mints_nothing(self):
        covered_preset = presets.Preset(
            name="t",
            triggers=[presets.TriggerGroup(
                name="memory", tools=["mnemosyne_diagnose"],
            )],
        )
        updates = shaping.compute_overlay_updates(
            scope=SCOPE,
            scope_entry={"carry": [], "expand_only": ["mnemosyne_diagnose"],
                         "shaping": {}},
            sessions={}, calls_by_pred={},
            protected=set(),
            newly_demoted=["mnemosyne_diagnose"],
            policy_preset=covered_preset,
        )
        self.assertEqual(updates, [])


class OverlayResolutionTests(unittest.TestCase):
    """Overlay unions with policy triggers on a REAL resolve, and activation
    flows end-to-end through carrying.resolve — where T = triggered ∩ X makes
    the overlay structurally unable to demote, disable, or escape E."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        state_dir = self.home / "state" / "tool-belt"
        state_dir.mkdir(parents=True)
        (state_dir / "learned.json").write_text(json.dumps({
            "version": 2,
            "scopes": {SCOPE: {
                "carry": [],
                "expand_only": [TOOL],
                "shaping": {},
                "triggers": [{
                    "name": f"auto:{TOOL}", "tools": [TOOL],
                    "keywords": [r"\bledger\b"],
                    "exclude_keywords": [r"\bignore the ledger\b"],
                    "source": "mined",
                }],
            }},
        }), encoding="utf-8")
        env = mock.patch.dict(os.environ, {"HERMES_HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)
        learned.load_state(force=True)
        self.preset = presets.resolve_preset(
            {"enabled": True, "learned_mode": "apply"}, SCOPE)

    def overlay_group(self):
        groups = [g for g in self.preset.triggers if g.name == f"auto:{TOOL}"]
        self.assertEqual(len(groups), 1, "overlay group must join the resolved preset")
        return groups[0]

    def test_overlay_group_unions_with_policy_triggers(self):
        policy_names = {g.name for g in presets.load_base_policy().triggers}
        resolved_names = {g.name for g in self.preset.triggers}
        self.assertTrue(policy_names <= resolved_names)  # policy untouched
        self.overlay_group()

    def test_overlay_trigger_activates_expand_only_tool_end_to_end(self):
        group = self.overlay_group()
        self.assertTrue(group.matches("please update the ledger today"))
        model = carrying.resolve(
            enabled=["read_file", TOOL],
            always_carry=self.preset.always_carry,
            carry=self.preset.carry,
            demoted=self.preset.demoted,
            triggered=[TOOL],
        )
        self.assertIn(TOOL, model.expand_only)  # residency unchanged
        self.assertIn(TOOL, model.active)       # but activated this turn

    def test_dampeners_apply_to_overlay_triggers(self):
        group = self.overlay_group()
        self.assertFalse(group.matches("ignore the ledger for now"))

    def test_overlay_cannot_reference_a_tool_outside_e(self):
        model = carrying.resolve(
            enabled=["read_file"],  # TOOL is not in this scope's ceiling
            always_carry=self.preset.always_carry,
            carry=self.preset.carry,
            demoted=self.preset.demoted,
            triggered=[TOOL],
        )
        self.assertNotIn(TOOL, model.active)

    def test_overlay_cannot_touch_always_carry(self):
        # An overlay entry naming an always_carry tool is filtered at compile.
        groups = learned._compile_overlay_groups(
            {"triggers": [{"name": "auto:x", "tools": ["clarify"],
                           "keywords": [r"\bclarify\b"]}]},
            always_carry_set={"clarify"},
            scope=SCOPE,
        )
        self.assertEqual(groups, [])


class OverlayLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.state_dir = self.home / "state" / "tool-belt"
        self.state_dir.mkdir(parents=True)
        env = mock.patch.dict(os.environ, {"HERMES_HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)

    def test_reset_scope_clears_the_overlay(self):
        state = {"version": 2, "scopes": {SCOPE: {
            "carry": [], "expand_only": [TOOL], "shaping": {},
            "triggers": [{"name": f"auto:{TOOL}", "tools": [TOOL],
                          "keywords": [r"\bledger\b"]}],
            "unrelated_meta": {"keep": True},
        }}}
        new_state, changed = learned.reset_scope(state, SCOPE)
        self.assertTrue(changed)
        entry = new_state["scopes"][SCOPE]
        self.assertNotIn("triggers", entry)
        self.assertEqual(entry["unrelated_meta"], {"keep": True})

    def test_status_reports_auto_learned_trigger_count(self):
        (self.state_dir / "learned.json").write_text(json.dumps({
            "version": 2,
            "scopes": {SCOPE: {
                "carry": ["read_file"], "expand_only": [TOOL],
                "shaping": {"source": "auto", "applied_at": "2026-08-30T00:00:00Z"},
                "triggers": [
                    {"name": f"auto:{TOOL}", "tools": [TOOL],
                     "keywords": [r"\bledger\b"]},
                    {"name": "auto:other", "tools": ["other_tool"],
                     "keywords": [r"\bother\b"]},
                ],
            }},
        }), encoding="utf-8")
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        configure = importlib.import_module("configure")
        info = configure.ScopeInfo(
            scope=SCOPE, agent="agent-a", platform="telegram",
            state_dir=self.state_dir, sessions=25,
        )
        row = configure.render_status_row(info, configure.STATE_SHAPED, {})
        self.assertIn("auto-learned triggers: 2", row)

    def test_auto_pass_persists_mined_overlay_end_to_end(self):
        """A full auto_shape_run writes the mined overlay into learned.json."""
        now = time.time()
        preds, calls = [], []
        for i in range(shaping.OVERLAY_MINE_MIN_SUPPORT):
            pid = f"ev-{i}"
            preds.append({
                "ts": now + i, "schema_version": 2, "scope": SCOPE,
                "session_id": "key", "hermes_session_id": "sess-1",
                "prediction_id": pid,
                "message_preview": f"please run the {PHRASE} for {i}",
                "ceiling_tools": ["read_file", TOOL],
                "always_carry_tools": [], "carry_tools": ["read_file"],
                "expand_only_tools": [TOOL], "active_tools": ["read_file"],
            })
            calls.append({
                "ts": now + i, "schema_version": 2, "prediction_id": pid,
                "tool_name": TOOL, "activated_by_expansion": True,
            })
        for i in range(6):
            preds.append({
                "ts": now + 100 + i, "schema_version": 2, "scope": SCOPE,
                "session_id": "key", "hermes_session_id": "sess-1",
                "prediction_id": f"noise-{i}",
                "message_preview": f"unrelated chatter about weather {i}",
                "ceiling_tools": ["read_file", TOOL],
                "always_carry_tools": [], "carry_tools": ["read_file"],
                "expand_only_tools": [TOOL], "active_tools": ["read_file"],
            })
        (self.state_dir / "predictions.jsonl").write_text(
            "".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8")
        (self.state_dir / "tool_calls.jsonl").write_text(
            "".join(json.dumps(c) + "\n" for c in calls), encoding="utf-8")
        (self.state_dir / "learned.json").write_text(json.dumps({
            "version": 2,
            "scopes": {SCOPE: {"carry": [], "expand_only": [TOOL], "shaping": {}}},
        }), encoding="utf-8")

        # The overlay can only ACTIVATE expand_only tools, so this fixture's
        # precondition is that TOOL stays demoted. Under the shipped promote
        # gates (1 session / 2 calls) the four mining events would themselves
        # promote it, leaving nothing to mine — so the promote arm is held off
        # explicitly here rather than by luck of the thresholds.
        cfg = {
            "enabled": True,
            "channels": {SCOPE: {"learned_mode": "apply"}},
            "learning": {"shape_ceiling": {"promote_min_calls": 99}},
        }
        summary = shaping.auto_shape_run(cfg, self.state_dir, now=now)
        self.assertTrue(summary["ran"])
        self.assertIn(SCOPE, summary.get("overlay", {}))

        doc = json.loads((self.state_dir / "learned.json").read_text(encoding="utf-8"))
        self.assertIn(TOOL, doc["scopes"][SCOPE].get("expand_only") or [],
                      "precondition: the mined tool is still expand_only, "
                      "which is the only class the overlay can activate")
        overlay = doc["scopes"][SCOPE].get("triggers") or []
        self.assertTrue(any(
            g["name"] == f"auto:{TOOL}" and g["tools"] == [TOOL]
            for g in overlay
        ), f"mined overlay entry must persist; got {overlay!r}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
