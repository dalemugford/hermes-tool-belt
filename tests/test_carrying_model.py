"""Carrying-model contract tests for Tool Belt 1.0.

The locked model (Hermes supplies the enabled built-in ceiling ``E``)::

    A = always_carry ∩ E                     # permanent residents, never shaped
    C = ((E − A) − demoted) ∪ ((carry ∩ E) − A)   # adaptive residents (full-start)
    X = E − (A ∪ C)                          # expand_only — the demoted remainder
    T = triggered ∩ X                        # trigger-activated expand_only tools
    R = expanded  ∩ X                        # explicitly expanded expand_only tools
    active = A ∪ C ∪ T ∪ R                   # what the model sees this message

Key invariants pinned here (full-start contract, Promise #2 2026-08-30):

  * A, C, X are a genuine three-way partition of E (precedence
    always_carry > carry > demoted).
  * Promotion (expand_only→carry) and demotion (carry→expand_only) move a tool
    across the C/X boundary via the adaptive ``carry`` loadout and never touch
    trigger definitions.
  * expand_only tools may be *activated* by a trigger or by expand_tools
    without changing residency.
  * Disabled/absent tools (not in E) can never re-enter through policy,
    learning, triggers, or explicit expansion.
  * Cache-off trigger activation is per-turn — it never carries forward.

The contract surface::

    tool_belt_plugin.carrying.resolve(
        enabled,          # E — enabled built-in ceiling (tool names)
        always_carry,     # immutable resident baseline (tool names)
        carry,            # explicit carry promotions (win over demoted)
        demoted=(),       # evidence-driven expand_only assignments
        triggered=(),     # names activated by trigger match this message
        expanded=(),      # names activated by explicit expand_tools
    ) -> model exposing .always_carry(A) .carry(C) .expand_only(X) .active

Learned demotion *reconciliation* (always_carry immunity, carry-wins overlap)
lives upstream in ``learned.apply_to_preset`` and is pinned by
``test_full_start_contract`` and ``test_shaper_merge``; the reconciled
``demoted`` loadout then reaches ``resolve`` as its own argument.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import re
import sys
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path

# Reuse conftest's plugin-loader bootstrap (hyphenated plugin dir).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401,E402

presets_mod = importlib.import_module("tool_belt_plugin.presets")
learned_mod = importlib.import_module("tool_belt_plugin.learned")

# The immutable always_carry baseline from the locked model.
ALWAYS_CARRY = frozenset(
    {"clarify", "skill_view", "skills_list", "expand_tools",
     "tool_search", "tool_describe", "tool_call"}
)

# ─── Strict access to the shipped surface ─────────────────────────────────
# Field aliases could mask a rename — a renamed field must FAIL these
# contracts, not silently fall back.

from tool_belt_plugin import carrying as _carrying_mod  # noqa: E402

_Model = namedtuple("_Model", "A C X active")


def _read_model(raw):
    return _Model(
        A=set(raw.always_carry), C=set(raw.carry), X=set(raw.expand_only),
        active=set(raw.active),
    )


def _trigger_fingerprint(triggers):
    """A stable, byte-comparable serialization of trigger *definitions*.

    Captures every load-bearing field so a promotion/demotion that rewrites a
    trigger group (e.g. stripping a demoted tool out of its group) shows up as
    a different fingerprint.
    """
    return json.dumps(
        [
            {
                "name": g.name,
                "tools": list(g.tools),
                "keywords": [p.pattern for p in g.keyword_patterns],
                "excludes": [p.pattern for p in g.exclude_patterns],
                "has_attachment": g.has_attachment,
            }
            for g in triggers
        ],
        sort_keys=True,
    )


@contextlib.contextmanager
def _temp_learned_state(doc):
    """Point HERMES_HOME at a throwaway home holding ``doc`` as learned.json.

    Mirrors the established pattern in test_trigger_dampeners: never touches
    the live ~/.hermes; restores HERMES_HOME and reloads on exit.
    """
    original_home = os.environ.get("HERMES_HOME")
    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir) / "state" / "tool-belt"
        state_dir.mkdir(parents=True)
        (state_dir / "learned.json").write_text(
            json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.environ["HERMES_HOME"] = tmpdir
        try:
            learned_mod.load_state(force=True)
            yield Path(state_dir / "learned.json")
        finally:
            if original_home is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = original_home
            learned_mod.load_state(force=True)


class _CarryingContract(unittest.TestCase):
    """Base class: resolve the partition API and normalize its result."""

    def _resolve_raw(self, **kwargs):
        return _carrying_mod.resolve(**kwargs)

    def resolve(self, *, enabled, always_carry, carry, demoted=(), triggered=(),
                expanded=()):
        return _read_model(self._resolve_raw(
            enabled=set(enabled),
            always_carry=set(always_carry),
            carry=set(carry),
            demoted=set(demoted),
            triggered=set(triggered),
            expanded=set(expanded),
        ))


# ─── 1. three-way partition over an explicit enabled ceiling ───────────────

class ThreeWayPartitionContract(_CarryingContract):
    def test_three_way_partition_over_explicit_ceiling(self):
        E = {"clarify", "send_message", "expand_tools", "read_file",
             "web_extract", "browser_exec"}
        # Full-start contract: only a demotion moves a tool into X; an explicit
        # carry entry (learned promotion) wins over a demotion naming it.
        carry = {"read_file"}
        m = self.resolve(enabled=E, always_carry=ALWAYS_CARRY, carry=carry,
                         demoted={"read_file", "web_extract", "browser_exec"})

        # Genuine partition: covers E exactly, pairwise disjoint.
        self.assertEqual(m.A | m.C | m.X, set(E), "A ∪ C ∪ X must equal E")
        self.assertEqual(m.A & m.C, set(), "A and C must be disjoint")
        self.assertEqual(m.A & m.X, set(), "A and X must be disjoint")
        self.assertEqual(m.C & m.X, set(), "C and X must be disjoint")

        # Class contents follow the locked definitions (precedence AC > C > X).
        self.assertEqual(m.A, set(ALWAYS_CARRY) & set(E))
        # send_message is not pinned; as an undemoted enabled tool it is an
        # ordinary class-C resident here.
        self.assertEqual(m.C, {"read_file", "send_message"})
        self.assertEqual(m.X, {"web_extract", "browser_exec"})

        # With no triggers/expansions, active == residents (A ∪ C).
        self.assertEqual(m.active, m.A | m.C)


# ─── 2. expand_only activation without residency change ────────────────────

class ExpandOnlyActivationContract(_CarryingContract):
    def test_expand_only_tool_activates_via_trigger_without_promotion(self):
        E = {"clarify", "send_message", "web_extract"}
        m = self.resolve(enabled=E, always_carry=ALWAYS_CARRY, carry=set(),
                         demoted={"web_extract"}, triggered={"web_extract"})

        self.assertIn("web_extract", m.X, "trigger activation does not change residency")
        self.assertNotIn("web_extract", m.C)
        self.assertIn("web_extract", m.active, "triggered expand_only tool is active")

    def test_expand_only_tool_activates_via_expand_tools_without_promotion(self):
        E = {"clarify", "send_message", "browser_exec"}
        m = self.resolve(enabled=E, always_carry=ALWAYS_CARRY, carry=set(),
                         demoted={"browser_exec"}, expanded={"browser_exec"})

        self.assertIn("browser_exec", m.X, "explicit expansion does not change residency")
        self.assertNotIn("browser_exec", m.C)
        self.assertIn("browser_exec", m.active, "expanded expand_only tool is active")


# ─── 3. disabled/absent tool can never re-enter ────────────────────────────

class DisabledToolContract(_CarryingContract):
    def test_disabled_or_absent_tool_never_reenters(self):
        E = {"clarify", "send_message", "read_file"}  # ghost_tool is NOT enabled
        # Every possible re-entry path references the ghost tool at once.
        m = self.resolve(
            enabled=E,
            always_carry=set(ALWAYS_CARRY) | {"ghost_tool"},  # policy reference
            carry={"ghost_tool"},                             # learned/adaptive reference
            triggered={"ghost_tool"},                         # trigger reference
            expanded={"ghost_tool"},                          # explicit expansion
        )
        for label, bucket in (("A", m.A), ("C", m.C), ("X", m.X), ("active", m.active)):
            self.assertNotIn("ghost_tool", bucket,
                             f"disabled/absent tool must not enter {label}")


# ─── 4. trigger definitions byte-equivalent across promotion/demotion ──────

class TriggerImmutabilityContract(unittest.TestCase):
    """Promotion/demotion change residency only — never trigger definitions.

    Exercises the REAL learned-merge path. A demotion that strips the demoted
    tool out of its trigger group (so it could no longer be trigger-activated
    as expand_only) is exactly the v2 violation this pins.
    """

    def _base_preset(self):
        # Phase 2 API alignment: the carrying model renamed the Preset resident
        # fields to always_carry/carry. ``clarify`` is an immutable resident.
        return presets_mod.Preset(
            name="carrying-contract",
            always_carry=["clarify"],
            triggers=[
                presets_mod.TriggerGroup(
                    name="web_extract",
                    tools=["web_extract"],
                    keyword_patterns=[re.compile(r"https?://", re.IGNORECASE)],
                    exclude_patterns=[re.compile(r"\blater\b", re.IGNORECASE)],
                    has_attachment=None,
                )
            ],
        )

    def test_trigger_definitions_unchanged_across_promotion_and_demotion(self):
        cfg = {"learned_mode": "apply", "channels": {}}
        scope = "assistant-a:telegram"

        base = self._base_preset()
        fp0 = _trigger_fingerprint(base.triggers)

        # Promotion: learned promotes web_extract into residency.
        promote_doc = {"scopes": {scope: {"carry": ["web_extract"]}}}
        with _temp_learned_state(promote_doc):
            promoted = learned_mod.apply_to_preset(self._base_preset(), cfg, scope)
        self.assertEqual(_trigger_fingerprint(promoted.preset.triggers), fp0,
                         "promotion must not alter trigger definitions")

        # Demotion: learned demotes web_extract. It must remain in its trigger
        # group (expand_only tools stay trigger-activatable).
        demote_doc = {"scopes": {scope: {"expand_only": ["web_extract"]}}}
        with _temp_learned_state(demote_doc):
            demoted = learned_mod.apply_to_preset(self._base_preset(), cfg, scope)
        self.assertEqual(_trigger_fingerprint(demoted.preset.triggers), fp0,
                         "demotion must not alter trigger definitions")


# ─── 5. cache-off trigger activation is per-turn (no carry-forward) ─────────

class CacheOffTriggerEphemeralContract(_CarryingContract):
    def test_cache_off_trigger_activation_disappears_on_next_turn(self):
        E = {"clarify", "send_message", "web_extract"}

        # Turn 1: web_extract triggers. Cache-off recomputes per turn.
        m1 = self.resolve(enabled=E, always_carry=ALWAYS_CARRY, carry=set(),
                          demoted={"web_extract"}, triggered={"web_extract"})
        self.assertIn("web_extract", m1.active, "trigger activates the tool this turn")

        # Turn 2: an unrelated message — nothing triggers, nothing carried.
        m2 = self.resolve(enabled=E, always_carry=ALWAYS_CARRY, carry=set(),
                          demoted={"web_extract"}, triggered=set())
        self.assertNotIn("web_extract", m2.active,
                         "cache-off trigger activation does not persist to the next turn")
        self.assertEqual(m2.active, m2.A | m2.C, "turn 2 active is residents-only")


if __name__ == "__main__":
    unittest.main()
