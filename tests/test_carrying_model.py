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
  * Unknown enabled built-ins default to CARRY — everything enabled is
    carried until evidence-driven demotion moves it to expand_only (this
    deliberately flips the pre-full-start "unknowns → expand_only" default).
  * always_carry is immune to learned demotion — proven against
    ``learned.apply_to_preset``, the single home of demotion precedence
    (the carrying partition receives an already-reconciled loadout).
  * Promotion (expand_only→carry) and demotion (carry→expand_only) move a tool
    across the C/X boundary via the adaptive ``carry`` loadout and never touch
    trigger definitions.
  * expand_only tools may be *activated* by a trigger or by expand_tools
    without changing residency.
  * Disabled/absent tools (not in E) can never re-enter through policy,
    learning, triggers, or explicit expansion.
  * Malformed overlap fails safe toward carrying and warns.
  * Internal failures fail open — the original Hermes ceiling is returned.
  * v1 telemetry normalizes into v2 in memory (``residency_inferred`` only
    when membership is complete).

The contract surface::

    tool_belt_plugin.carrying.resolve(
        enabled,          # E — enabled built-in ceiling (tool names)
        always_carry,     # immutable resident baseline (tool names)
        carry,            # explicit carry promotions (win over demoted)
        demoted=(),       # evidence-driven expand_only assignments
        triggered=(),     # names activated by trigger match this message
        expanded=(),      # names activated by explicit expand_tools
    ) -> model exposing .always_carry(A) .carry(C) .expand_only(X) .active

Learned demotion *reconciliation* (always_carry immunity, carry-wins
overlap) lives upstream in ``learned.apply_to_preset``; the reconciled
``demoted`` loadout then reaches ``resolve`` as its own argument.

Plus, for the persistence/telemetry cases, the real modules are exercised
directly (``learned``/``analyze``/``logger_io``) with a v2 normalization
entry point that is likewise resolved lazily.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

# Reuse conftest's plugin-loader bootstrap (hyphenated plugin dir).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401,E402

plugin = sys.modules["tool_belt_plugin"]
presets_mod = importlib.import_module("tool_belt_plugin.presets")
predictor_mod = importlib.import_module("tool_belt_plugin.predictor")
learned_mod = importlib.import_module("tool_belt_plugin.learned")
analyze_mod = importlib.import_module("tool_belt_plugin.analyze")
logger_io_mod = importlib.import_module("tool_belt_plugin.logger_io")
expand_tools_mod = importlib.import_module("tool_belt_plugin.expand_tools")

PLUGIN_DIR = Path(__file__).resolve().parent.parent


def _load_shaper():
    """Load ``scripts/shape-ceiling.py`` as an importable module for direct
    unit exercise of the between-session shaper's contract surface."""
    spec = importlib.util.spec_from_file_location(
        "tool_belt_shape_ceiling_contract", PLUGIN_DIR / "scripts" / "shape-ceiling.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shaper = _load_shaper()

# The immutable always_carry baseline from the locked model.
ALWAYS_CARRY = frozenset(
    {"clarify", "skill_view", "skills_list", "expand_tools",
     "tool_search", "tool_describe", "tool_call"}
)

# ─── Strict access to the shipped surface ─────────────────────────────────
# Field aliases could mask a rename — a renamed field must FAIL these
# contracts, not silently fall back.

from tool_belt_plugin import carrying as _carrying_mod
from tool_belt_plugin import logger_io as _logger_io_mod

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
        raw = self._resolve_raw(
            enabled=set(enabled),
            always_carry=set(always_carry),
            carry=set(carry),
            demoted=set(demoted),
            triggered=set(triggered),
            expanded=set(expanded),
        )
        m = _read_model(raw)
        self.assertIsNotNone(m.A, "carrying result exposes no always_carry (A) set")
        self.assertIsNotNone(m.C, "carrying result exposes no carry (C) set")
        self.assertIsNotNone(m.X, "carrying result exposes no expand_only (X) set")
        self.assertIsNotNone(m.active, "carrying result exposes no active set")
        return m


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


# ─── 2. unknown enabled built-ins default to CARRY (full-start) ────────────
# Deliberate contract flip (Promise #2, 2026-08-30): the pre-full-start model
# defaulted unknowns to expand_only; now they are carried until demoted.

# ─── 3. always_carry immunity from learned demotion ────────────────────────

class AlwaysCarryImmunityContract(_CarryingContract):
    """Demotion precedence lives in ``learned.apply_to_preset`` — its single
    home. The contract: a learned ``expand_only`` (demotion) signal naming an
    always_carry tool is ignored with a warning, and the partition computed
    from the reconciled preset keeps the tool resident in A."""

    def test_always_carry_immune_from_learned_demotion(self):
        E = {"clarify", "send_message", "web_extract"}
        scope = "assistant-a:telegram"
        preset = presets_mod.Preset(
            name="immunity-contract",
            always_carry=sorted(ALWAYS_CARRY),
            carry=[],
            triggers=[],
        )
        # The learned layer emits a demotion (expand_only) signal naming an
        # always_carry tool. It must be ignored, and warned about.
        doc = {"version": 2,
               "scopes": {scope: {"carry": ["web_extract"],
                                  "expand_only": ["clarify"]}}}
        with _temp_learned_state(doc):
            with self.assertLogs(_LOGGER_LEARNED, level="WARNING") as cm:
                merged = learned_mod.apply_to_preset(
                    preset, {"learned_mode": "apply", "channels": {}}, scope)

        self.assertIn("clarify", merged.preset.always_carry,
                      "always_carry tool stays on the immutable surface")
        self.assertNotIn("clarify", merged.preset.carry)
        warned = "\n".join(cm.output)
        self.assertIn("clarify", warned, "demoting an always_carry tool must warn")

        # The partition of the reconciled preset keeps the tool resident.
        m = self.resolve(enabled=E, always_carry=merged.preset.always_carry,
                         carry=merged.preset.carry)
        self.assertIn("clarify", m.A, "always_carry tool stays resident")
        self.assertIn("clarify", m.active)
        self.assertNotIn("clarify", m.X, "always_carry never falls to expand_only")


# ─── 4. promotion expand_only -> carry ─────────────────────────────────────

# ─── 5. demotion carry -> expand_only ──────────────────────────────────────

# ─── 6/7. expand_only activation without residency change ──────────────────

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


# ─── 8. disabled/absent tool can never re-enter ────────────────────────────

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


# ─── 9. malformed overlap resolves safely and warns ────────────────────────

class MalformedOverlapContract(_CarryingContract):
    """The carried∧demoted conflict rule lives in ``learned.py`` — its single
    home (``_reconcile_overlap``, routed by ``apply_to_preset``). The contract:
    a tool named both carried and demoted fails safe *toward carrying* with a
    warning, and the partition keeps it resident."""

    def test_malformed_overlap_fails_safe_toward_carrying_and_warns(self):
        E = {"clarify", "send_message", "web_extract"}
        scope = "assistant-a:telegram"
        preset = presets_mod.Preset(
            name="overlap-contract",
            always_carry=sorted(ALWAYS_CARRY),
            carry=[],
            triggers=[],
        )
        # web_extract is named carried AND demoted — a contradictory overlap.
        doc = {"version": 2,
               "scopes": {scope: {"carry": ["web_extract"],
                                  "expand_only": ["web_extract"]}}}
        with _temp_learned_state(doc):
            with self.assertLogs(_LOGGER_LEARNED, level="WARNING") as cm:
                # Force a fresh read so the load-time normalization (where the
                # overlap warning is emitted) runs inside the log capture.
                learned_mod.load_state(force=True)
                merged = learned_mod.apply_to_preset(
                    preset, {"learned_mode": "apply", "channels": {}}, scope)

        warned = "\n".join(cm.output)
        self.assertIn("web_extract", warned, "malformed overlap must warn")
        self.assertIn("web_extract", merged.preset.carry,
                      "fail safe toward carrying keeps it resident")

        m = self.resolve(enabled=E, always_carry=merged.preset.always_carry,
                         carry=merged.preset.carry)
        self.assertIn("web_extract", m.C, "fail safe toward carrying keeps it resident")
        self.assertNotIn("web_extract", m.X, "must not be silently narrowed away")
        self.assertIn("web_extract", m.active)


# ─── 10. fail-open no-narrowing behavior ───────────────────────────────────

class FailOpenContract(_CarryingContract):
    def test_internal_failure_fails_open_to_original_ceiling(self):
        E = {"clarify", "send_message", "read_file", "web_extract"}

        class _Boom:
            def __iter__(self):
                raise RuntimeError("boom")

        # An internal failure while computing residency must fail OPEN — return
        # the original Hermes ceiling (no narrowing), never a narrowed set.
        # _Boom is passed straight through (not coerced) to trip the resolver.
        raw = self._resolve_raw(
            enabled=set(E), always_carry=_Boom(), carry=set(),
            triggered=set(), expanded=set(),
        )
        m = _read_model(raw)
        self.assertIsNotNone(m.active, "fail-open must still return an active set")
        self.assertTrue(set(E) <= m.active,
                        "fail-open returns the whole enabled ceiling (no narrowing)")

    def test_unreadable_ceiling_entry_never_narrows_the_readable_names(self):
        # A non-string name in the ceiling is skipped, never fatal: every
        # readable ceiling name stays active (never active == ∅).
        class _BadName:
            def __str__(self):
                raise RuntimeError("boom")

        good = {"clarify", "send_message", "read_file", "web_extract"}

        raw = self._resolve_raw(
            enabled=list(good) + [_BadName()], always_carry=set(ALWAYS_CARRY),
            carry=set(), triggered=set(), expanded=set(),
        )
        m = _read_model(raw)
        self.assertTrue(good <= m.active,
                        "the readable ceiling names stay active")


# ─── 14. trigger definitions byte-equivalent across promotion/demotion ─────

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


# ─── 15. cache-off trigger activation is per-turn (no carry-forward) ────────

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


# ─── 17. sticky/category expansions never escape the enabled ceiling ────────

# ─── 18. discoverability manifest is derived from X (Phase 4 contract) ──────

# ─── 19. between-session shaper contracts (Phase 6) ────────────────────────

def _pred_row(scope, sid, pid, *, ceiling, always_carry, carry, active, ts=0.0):
    """A complete canonical v2 prediction row (residency reconstructible)."""
    expand_only = [t for t in ceiling if t not in set(always_carry) | set(carry)]
    return {
        "schema_version": 2,
        "scope": scope,
        "hermes_session_id": sid,
        "prediction_id": pid,
        "ts": ts,
        "ceiling_tools": list(ceiling),
        "always_carry_tools": list(always_carry),
        "carry_tools": list(carry),
        "active_tools": list(active),
        "expand_only_tools": expand_only,
    }


def _sparse_v1_row(scope, sid, pid, *, always_on, ts=0.0):
    """A sparse v1 prediction row: residents only, no ceiling/active. The
    normalizer cannot reconstruct residency → ``residency_inferred`` is False."""
    return {
        "scope": scope,
        "hermes_session_id": sid,
        "prediction_id": pid,
        "ts": ts,
        "always_on_tools": list(always_on),
    }


def _complete_v1_row(scope, sid, pid, *, ceiling, residents, active, ts=0.0):
    """A *complete* v1 prediction row (no schema_version, no always_carry_tools).

    Ceiling + residents + active are all present, so the normalizer can
    reconstruct residency (``residency_inferred`` True). v1 has no immutable
    split, so the normalizer collapses every resident into the ``carry``
    residency class — including any always_carry baseline tool that is resident.
    This is the exact transitional shape that used to abort the shaper.
    """
    return {
        "scope": scope,
        "hermes_session_id": sid,
        "prediction_id": pid,
        "ts": ts,
        "ceiling_tools": list(ceiling),
        "always_on_tools": list(residents),
        "allowed_tools": list(active),
    }


def _expansion_call(pid, tool):
    """A tool-call row that is direct expansion evidence (expand_only → carry)."""
    return {
        "schema_version": 2,
        "prediction_id": pid,
        "tool_name": tool,
        "was_initially_active": False,
        "was_expand_only": True,
        "activated_by_expansion": True,
        "expansion_provided_access": True,
        "activation_source": "expansion",
    }


def _trigger_call(pid, tool):
    """A tool-call row activated by a trigger — NOT expansion evidence."""
    return {
        "schema_version": 2,
        "prediction_id": pid,
        "tool_name": tool,
        "was_initially_active": True,
        "was_expand_only": True,
        "activation_source": "trigger",
    }


def _compute(scope, pred_rows, call_rows, **overrides):
    grouped = shaper.group_predictions_by_scope_session(pred_rows)
    calls = shaper.index_tool_calls_by_prediction(call_rows)
    kwargs = dict(
        window_days=7, promote_min_sessions=2, promote_min_calls=3,
        demote_min_sessions_no_use=20,
    )
    kwargs.update(overrides)
    return shaper.compute_scope_recommendations(
        scope=scope, sessions=grouped.get(scope, {}), calls_by_pred=calls, **kwargs
    )


class ShaperPromotionContract(unittest.TestCase):
    SCOPE = "assistant-a:telegram"

    def test_expand_only_tool_promotes_after_qualifying_expansions(self):
        E = ["clarify", "send_message", "web_extract"]
        preds, calls = [], []
        for i in range(3):  # 3 sessions ≥ promote_min_sessions
            pid = f"p{i}"
            preds.append(_pred_row(
                self.SCOPE, f"s{i}", pid,
                ceiling=E, always_carry=["clarify", "send_message"], carry=[],
                active=["clarify", "send_message"], ts=i,
            ))
            calls.append(_expansion_call(pid, "web_extract"))  # 3 calls ≥ min_calls
        recs = _compute(self.SCOPE, preds, calls)
        promoted = {p["tool"] for p in recs["promote"]}
        self.assertIn("web_extract", promoted,
                      "an expand_only tool reached via expand_tools promotes to carry")
        self.assertEqual(recs["demote"], [], "no demotion in the promote arm")

    def test_validation_domain_is_scope_local_not_global(self):
        # Regression: enabled_tool_names (the candidate-validation domain
        # merge_into_learned's _accept checks against) was built from the
        # GLOBAL tool-call index, so another agent's tool names could validate
        # into this scope's carrying lists. It must be built only from this
        # scope's own predictions and their tool calls.
        E = ["clarify", "send_message", "web_extract"]
        preds, calls = [], []
        for i in range(3):
            pid = f"p{i}"
            preds.append(_pred_row(
                self.SCOPE, f"s{i}", pid,
                ceiling=E, always_carry=["clarify", "send_message"], carry=[],
                active=["clarify", "send_message"], ts=i,
            ))
            calls.append(_expansion_call(pid, "web_extract"))
        # A different agent's telemetry shares the tool_calls file: its calls
        # are indexed under ITS prediction ids, never this scope's.
        calls.append(_expansion_call("foreign-p0", "foreign_agent_tool"))
        recs = _compute(self.SCOPE, preds, calls)
        self.assertNotIn(
            "foreign_agent_tool", recs["enabled_tool_names"],
            "another agent's tool must not enter this scope's validation domain",
        )
        self.assertIn("web_extract", recs["enabled_tool_names"],
                      "this scope's own observed tools still validate")

    def test_trigger_only_use_does_not_promote(self):
        E = ["clarify", "send_message", "web_extract"]
        preds, calls = [], []
        for i in range(5):
            pid = f"p{i}"
            preds.append(_pred_row(
                self.SCOPE, f"s{i}", pid,
                ceiling=E, always_carry=["clarify", "send_message"], carry=[],
                active=["clarify", "send_message", "web_extract"], ts=i,
            ))
            calls.append(_trigger_call(pid, "web_extract"))
        recs = _compute(self.SCOPE, preds, calls)
        self.assertEqual(recs["promote"], [],
                         "trigger activation is never promotion evidence")


class ShaperDemotionContract(unittest.TestCase):
    SCOPE = "assistant-a:telegram"

    def test_adaptive_carry_demotes_after_no_use_window(self):
        E = ["clarify", "send_message", "read_file"]
        preds = [
            _pred_row(self.SCOPE, f"s{i}", f"p{i}",
                      ceiling=E, always_carry=["clarify", "send_message"],
                      carry=["read_file"], active=["clarify", "send_message", "read_file"], ts=i)
            for i in range(20)  # ≥ demote_min_sessions_no_use
        ]
        recs = _compute(self.SCOPE, preds, [])
        demoted = {d["tool"] for d in recs["demote"]}
        self.assertIn("read_file", demoted,
                      "an unused adaptive carry resident demotes to expand_only")

    def test_always_carry_never_a_demotion_candidate(self):
        # clarify/send_message are always_carry, resident every session, never
        # called — they must never be demoted (excluded by construction).
        E = ["clarify", "send_message", "read_file"]
        preds = [
            _pred_row(self.SCOPE, f"s{i}", f"p{i}",
                      ceiling=E, always_carry=["clarify", "send_message"],
                      carry=["read_file"], active=["clarify", "send_message", "read_file"], ts=i)
            for i in range(20)
        ]
        recs = _compute(self.SCOPE, preds, [])
        demoted = {d["tool"] for d in recs["demote"]}
        self.assertNotIn("clarify", demoted)
        self.assertNotIn("send_message", demoted)

    def test_sparse_v1_row_cannot_drive_demotion(self):
        # 20 sessions of sparse v1 rows: residents present but no ceiling/active,
        # so residency is not inferable and the tools cannot demote.
        preds = [
            _sparse_v1_row(self.SCOPE, f"s{i}", f"p{i}",
                           always_on=["read_file", "web_search"], ts=i)
            for i in range(20)
        ]
        recs = _compute(self.SCOPE, preds, [])
        self.assertEqual(recs["demote"], [],
                         "a sparse (residency_inferred=False) row cannot demote")

    def test_complete_v1_unused_always_carry_never_demotes_and_no_abort(self):
        # Regression for the transitional-v1 abort: complete v1 rows normalize
        # every resident (including the immutable always_carry baseline tool
        # ``clarify``) into the ``carry`` residency class. With a full window of
        # such rows where ``clarify`` is resident-but-unused, the shaper must
        # neither raise nor recommend demoting ``clarify``. The adaptive
        # (non-baseline) resident ``read_file`` still demotes normally.
        ceiling = ["clarify", "read_file", "web_extract"]
        residents = ["clarify", "read_file"]  # clarify is always_carry baseline
        preds = [
            _complete_v1_row(self.SCOPE, f"s{i}", f"p{i}",
                             ceiling=ceiling, residents=residents,
                             active=residents, ts=i)
            for i in range(20)  # ≥ demote_min_sessions_no_use
        ]
        # Must not raise (the old assert aborted the whole run here).
        recs = _compute(self.SCOPE, preds, [])
        demoted = {d["tool"] for d in recs["demote"]}
        self.assertNotIn("clarify", demoted,
                         "an unused always_carry resident from a v1 row never demotes")
        self.assertIn("read_file", demoted,
                      "a non-baseline carry resident still demotes on a v1 window")


class ShaperMergeContract(unittest.TestCase):
    SCOPE = "assistant-a:telegram"

    def _recs(self, promote=(), demote=(), enabled=None):
        return {
            "scope": self.SCOPE,
            "computed_at": "2026-01-01T00:00:00Z",
            "sessions_considered": 20,
            "window_days": 7,
            "promote": [{"tool": t, "sessions": 3, "calls": 5, "evidence": "expansion"}
                        for t in promote],
            "demote": [{"tool": t, "sessions_without_use": 20, "evidence": "carry_unused"}
                       for t in demote],
            "enabled_tool_names": sorted(enabled if enabled is not None
                                         else set(promote) | set(demote)),
        }

    def _write(self, tmp, doc):
        (tmp / "learned.json").write_text(json.dumps(doc), encoding="utf-8")

    def _read(self, tmp):
        return json.loads((tmp / "learned.json").read_text(encoding="utf-8"))

    def test_promotion_moves_tool_from_expand_only_to_carry(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self._write(tmp, {"version": 2, "scopes": {
                self.SCOPE: {"carry": [], "expand_only": ["web_extract"]}}})
            recs = self._recs(promote=["web_extract"], enabled=["web_extract"])
            _state, changed = shaper.merge_into_learned(tmp, {self.SCOPE: recs}, False)
            self.assertTrue(changed)
            out = self._read(tmp)
            self.assertEqual(out["version"], 2)  # written as learned schema v2
            entry = out["scopes"][self.SCOPE]
            self.assertEqual(entry["carry"], ["web_extract"])
            self.assertEqual(entry["expand_only"], [])

    def test_demotion_moves_tool_from_carry_to_expand_only(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self._write(tmp, {"version": 2, "scopes": {
                self.SCOPE: {"carry": ["web_extract"], "expand_only": []}}})
            recs = self._recs(demote=["web_extract"], enabled=["web_extract"])
            _state, changed = shaper.merge_into_learned(tmp, {self.SCOPE: recs}, False)
            self.assertTrue(changed)
            out = self._read(tmp)["scopes"][self.SCOPE]
            self.assertEqual(out["expand_only"], ["web_extract"])
            self.assertEqual(out["carry"], [])

    def test_writes_v2_fields_and_preserves_unrelated_metadata(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self._write(tmp, {
                "version": 2,
                "provenance": "keep-me",              # unrelated top-level key
                "scopes": {
                    self.SCOPE: {"notes": "hand-edited"},   # unrelated per-scope key
                    "other:cli": {"carry": ["keepme"]},     # unrelated scope
                },
            })
            recs = self._recs(promote=["web_extract"], enabled=["web_extract"])
            shaper.merge_into_learned(tmp, {self.SCOPE: recs}, False)
            out = self._read(tmp)
            self.assertEqual(out["version"], 2)
            entry = out["scopes"][self.SCOPE]
            self.assertEqual(entry["carry"], ["web_extract"])
            self.assertIn("expand_only", entry)
            self.assertEqual(entry["shaping"]["scope"], self.SCOPE)
            # … unrelated metadata (top-level, per-scope) preserved, and the
            # untouched scope's assignment survives.
            self.assertEqual(out["provenance"], "keep-me")
            self.assertEqual(entry["notes"], "hand-edited")
            self.assertEqual(out["scopes"]["other:cli"]["carry"], ["keepme"],
                             "an untouched scope's assignment survives")

    def test_category_candidate_is_rejected_not_stored(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self._write(tmp, {"version": 2, "scopes": {}})
            # "web" is a toolset/category, not a concrete enabled tool name.
            recs = self._recs(promote=["web", "web_extract"],
                              enabled=["web_extract", "clarify"])
            with self.assertLogs("tool_belt_plugin.shaping", level="WARNING") as cm:
                shaper.merge_into_learned(tmp, {self.SCOPE: recs}, False)
            out = self._read(tmp)["scopes"][self.SCOPE]
            self.assertIn("web_extract", out["carry"])
            self.assertNotIn("web", out["carry"], "a category name is never stored")
            self.assertIn("web", "\n".join(cm.output))

    def test_empty_enabled_ceiling_refuses_all_candidates(self):
        # Defense-in-depth (O4): an empty enabled ceiling means no candidate can
        # be proven eligible, so hand-built recs that bypass ``compute`` must not
        # write anything into a carrying list.
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self._write(tmp, {"version": 2, "scopes": {
                self.SCOPE: {"carry": [], "expand_only": []}}})
            recs = self._recs(promote=["web_extract"], demote=["read_file"],
                              enabled=[])  # empty enabled ceiling
            with self.assertLogs("tool_belt_plugin.shaping", level="WARNING"):
                shaper.merge_into_learned(tmp, {self.SCOPE: recs}, False)
            entry = self._read(tmp)["scopes"][self.SCOPE]
            # No candidate may enter a carrying list when nothing can be validated.
            self.assertEqual(entry["carry"], [],
                             "no promote candidate is carried with an empty enabled ceiling")
            self.assertEqual(entry["expand_only"], [],
                             "no demote candidate is written with an empty enabled ceiling")

    def test_dry_run_performs_no_writes(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            recs = self._recs(promote=["web_extract"], enabled=["web_extract"])
            _state, changed = shaper.merge_into_learned(tmp, {self.SCOPE: recs}, True)
            self.assertTrue(changed, "dry-run still reports the intended change")
            self.assertFalse((tmp / "learned.json").exists(),
                             "dry-run writes nothing to disk")

    def test_demoted_tool_stays_in_its_trigger_group_after_apply(self):
        # End-to-end: the shaper demotes web_extract, and applying the learned
        # overlay to a preset with web_extract in a trigger group leaves that
        # group byte-identical (an expand_only tool stays trigger-activatable).
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self._write(tmp, {"version": 2, "scopes": {
                self.SCOPE: {"carry": ["web_extract"], "expand_only": []}}})
            recs = self._recs(demote=["web_extract"], enabled=["web_extract"])
            shaper.merge_into_learned(tmp, {self.SCOPE: recs}, False)
            demote_doc = self._read(tmp)

        preset = presets_mod.Preset(
            name="shaper-trigger",
            always_carry=["clarify"],
            triggers=[presets_mod.TriggerGroup(
                name="web_extract", tools=["web_extract"],
                keyword_patterns=[re.compile(r"https?://", re.IGNORECASE)],
                exclude_patterns=[], has_attachment=None)],
        )
        fp0 = _trigger_fingerprint(preset.triggers)
        with _temp_learned_state(demote_doc):
            applied = learned_mod.apply_to_preset(
                preset, {"learned_mode": "apply", "channels": {}}, self.SCOPE)
        self.assertEqual(_trigger_fingerprint(applied.preset.triggers), fp0,
                         "demotion must not alter the trigger group")
        self.assertIn("web_extract", applied.preset.triggers[0].tools)


class ShaperResetAndOverlapContract(unittest.TestCase):
    SCOPE = "assistant-a:telegram"

    def test_reset_scope_isolation_preserves_unrelated_metadata(self):
        state = {
            "version": 2,
            "provenance": "keep-top",
            "scopes": {
                self.SCOPE: {
                    "carry": ["web_extract"],
                    "expand_only": ["read_file"],
                    "shaping": {"scope": self.SCOPE},
                    "notes": "unrelated, keep me",
                },
                "other:cli": {"carry": ["terminal"]},
            },
        }
        new_state, changed = learned_mod.reset_scope(state, self.SCOPE)
        self.assertTrue(changed)
        entry = new_state["scopes"][self.SCOPE]
        # Adaptive assignments/evidence gone …
        for key in ("carry", "expand_only", "shaping"):
            self.assertNotIn(key, entry)
        # … unrelated per-scope metadata, other scopes, top-level metadata kept.
        self.assertEqual(entry["notes"], "unrelated, keep me")
        self.assertEqual(new_state["scopes"]["other:cli"], {"carry": ["terminal"]})
        self.assertEqual(new_state["provenance"], "keep-top")
        # The original state object is not mutated.
        self.assertIn("carry", state["scopes"][self.SCOPE])

    def test_reset_scope_drops_entry_when_only_adaptive_keys(self):
        state = {"version": 2, "scopes": {
            self.SCOPE: {"carry": ["web_extract"], "expand_only": []},
            "other:cli": {"carry": ["terminal"]}}}
        new_state, changed = learned_mod.reset_scope(state, self.SCOPE)
        self.assertTrue(changed)
        self.assertNotIn(self.SCOPE, new_state["scopes"])
        self.assertIn("other:cli", new_state["scopes"])

    def test_malformed_learned_overlap_fails_safe_toward_carry_and_warns(self):
        # A hand-built scope naming a tool in both carry and expand_only.
        doc = {"version": 2, "scopes": {
            self.SCOPE: {"carry": ["web_extract"], "expand_only": ["web_extract", "read_file"]}}}
        with self.assertLogs(_LOGGER_LEARNED, level="WARNING") as cm:
            v2 = learned_mod.normalize_state(doc)
        entry = v2["scopes"][self.SCOPE]
        self.assertIn("web_extract", entry["carry"], "carry wins the overlap")
        self.assertNotIn("web_extract", entry["expand_only"])
        self.assertIn("read_file", entry["expand_only"], "the genuine demote survives")
        self.assertIn("web_extract", "\n".join(cm.output))


_LOGGER_LEARNED = "tool_belt_plugin.learned"


if __name__ == "__main__":
    unittest.main()
