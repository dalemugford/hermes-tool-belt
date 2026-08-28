"""Carrying-model contract tests for Tool Belt 1.0 (Phase 1 — the *red* layer).

These tests pin the locked carrying model **before** the production v2 code
lands. They are expected to FAIL today: the v2 carrying partition does not
exist yet. The point of this file is to state the contract precisely so the
implementation has an unambiguous target and so any later regression against
the model is caught.

The locked model (Hermes supplies the enabled built-in ceiling ``E``)::

    A = always_carry ∩ E                     # permanent residents, never shaped
    C = (carry ∩ E) − A                      # adaptive residents
    X = E − (A ∪ C)                          # expand_only — the enabled remainder
    T = triggered ∩ X                        # trigger-activated expand_only tools
    R = expanded  ∩ X                        # explicitly expanded expand_only tools
    active = A ∪ C ∪ T ∪ R (∪ passthrough)   # what the model sees this message

Key invariants pinned here:

  * A, C, X are a genuine three-way partition of E (precedence
    always_carry > carry > expand_only).
  * Unknown enabled built-ins default to expand_only.
  * always_carry is immune to learned demotion.
  * Promotion (expand_only→carry) and demotion (carry→expand_only) move a tool
    across the C/X boundary via the adaptive ``carry`` loadout and never touch
    trigger definitions.
  * expand_only tools may be *activated* by a trigger or by expand_tools
    without changing residency.
  * Disabled/absent tools (not in E) can never re-enter through policy,
    learning, triggers, sticky/prior-active carry-forward, or explicit
    expansion.
  * Malformed overlap fails safe toward carrying and warns.
  * Internal failures fail open — the original Hermes ceiling is returned.
  * MCP/plugin tools pass through, outside the built-in partition.
  * v1 learned state / v1 telemetry normalize into v2 in memory (no
    read-time write; ``residency_inferred`` only when membership is complete).
  * Under cache-on the carrying assignment is frozen per session while trigger
    matching still runs each message, monotonically unioning newly triggered
    tools into the frozen active set.

Expected v2 contract surface (referenced lazily so a missing implementation
produces a clean test *failure*, never an import-time collection error)::

    tool_belt_plugin.carrying.resolve(
        enabled,          # E — enabled built-in ceiling (tool names)
        always_carry,     # immutable resident baseline (tool names)
        carry,            # adaptive resident loadout (promotion/demotion mutate this)
        triggered=(),     # names activated by trigger match this message
        expanded=(),      # names activated by explicit expand_tools
        passthrough=(),   # MCP/plugin names — outside the built-in partition
        demoted=(),       # learned/adaptive "always_off" signal (fail-safe/immune)
        prior_active=(),  # active names carried from earlier messages (cache-on)
    ) -> model exposing .always_carry(A) .carry(C) .expand_only(X)
                        .active .passthrough .warnings

Plus, for the persistence/telemetry cases, the real modules are exercised
directly (``learned``/``analyze``/``logger_io``) with a v2 normalization
entry point that is likewise resolved lazily.
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

PLUGIN_DIR = Path(__file__).resolve().parent.parent

# The immutable always_carry baseline from the locked model.
ALWAYS_CARRY = frozenset(
    {"clarify", "skill_view", "skills_list", "todo", "send_message", "expand_tools"}
)

_EXPECTED_SIGNATURE = (
    "carrying.resolve(enabled, always_carry, carry, triggered=(), expanded=(), "
    "passthrough=(), demoted=(), prior_active=()) -> "
    ".always_carry/.carry/.expand_only/.active/.passthrough/.warnings"
)
_MISSING_API_MSG = (
    "v2 carrying-model API not implemented. Expected " + _EXPECTED_SIGNATURE + ". "
    "This contract test is red until the Tool Belt 1.0 carrying partition ships."
)


# ─── Lazy resolution of the expected v2 surface ────────────────────────────
# We look in the documented home first, then a couple of plausible fallbacks,
# so the contract pins BEHAVIOR rather than module layout. When nothing is
# found the caller fails the test with a descriptive message.

def _first_callable(candidates):
    for modname, attrs in candidates:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for attr in attrs:
            fn = getattr(mod, attr, None)
            if callable(fn):
                return fn
    return None


def _resolve_carrying_api():
    return _first_callable([
        ("tool_belt_plugin.carrying", ("resolve", "resolve_carrying", "carrying_model")),
        ("tool_belt_plugin", ("resolve_carrying", "carrying_model", "carrying_resolve")),
        ("tool_belt_plugin.predictor", ("resolve_carrying", "carrying_model")),
        ("tool_belt_plugin.presets", ("resolve_carrying", "carrying_model")),
    ])


def _resolve_learned_v2_normalizer():
    return _first_callable([
        ("tool_belt_plugin.learned", ("normalize_state", "normalize_to_v2", "to_v2")),
        ("tool_belt_plugin.carrying", ("normalize_learned", "normalize_learned_state")),
    ])


def _resolve_telemetry_normalizer():
    return _first_callable([
        ("tool_belt_plugin.logger_io",
         ("normalize_prediction_row", "normalize_row", "upgrade_row", "to_v2_row")),
        ("tool_belt_plugin.analyze",
         ("normalize_prediction_row", "normalize_row", "normalize_telemetry_row", "to_v2_row")),
        ("tool_belt_plugin.carrying", ("normalize_telemetry_row", "normalize_row")),
    ])


# ─── Tolerant accessors over the (not-yet-defined) result shape ────────────

def _get(result, *names):
    """Read a field from a dict / namedtuple / dataclass-ish result."""
    for name in names:
        if isinstance(result, dict):
            if name in result:
                return result[name]
        elif hasattr(result, name):
            return getattr(result, name)
    return None


def _as_set(value):
    """Coerce a name-collection to a set; leave a lone string as one name."""
    if value is None:
        return None
    if isinstance(value, (set, frozenset)):
        return set(value)
    if isinstance(value, str):
        return {value}
    try:
        return set(value)
    except TypeError:
        return None


_Model = namedtuple("_Model", "A C X active passthrough warnings")


def _read_model(raw):
    return _Model(
        A=_as_set(_get(raw, "always_carry", "A")),
        C=_as_set(_get(raw, "carry", "C")),
        X=_as_set(_get(raw, "expand_only", "X", "expand")),
        active=_as_set(_get(raw, "active", "active_tools", "allowed", "allowed_tool_names")),
        passthrough=_as_set(_get(raw, "passthrough", "mcp_passthrough", "pass_through")),
        warnings=_get(raw, "warnings", "warns", "warning"),
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
    """Base class: resolve the v2 partition API and normalize its result."""

    def _resolve_raw(self, **kwargs):
        fn = _resolve_carrying_api()
        if fn is None:
            self.fail(_MISSING_API_MSG)
        try:
            return fn(**kwargs)
        except TypeError as exc:
            self.fail(
                "v2 carrying API does not match the contract signature: "
                f"{exc}\nExpected: {_EXPECTED_SIGNATURE}"
            )

    def resolve(self, *, enabled, always_carry, carry, triggered=(), expanded=(),
                passthrough=(), demoted=(), prior_active=()):
        raw = self._resolve_raw(
            enabled=set(enabled),
            always_carry=set(always_carry),
            carry=set(carry),
            triggered=set(triggered),
            expanded=set(expanded),
            passthrough=set(passthrough),
            demoted=set(demoted),
            prior_active=set(prior_active),
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
        carry = {"read_file"}
        m = self.resolve(enabled=E, always_carry=ALWAYS_CARRY, carry=carry)

        # Genuine partition: covers E exactly, pairwise disjoint.
        self.assertEqual(m.A | m.C | m.X, set(E), "A ∪ C ∪ X must equal E")
        self.assertEqual(m.A & m.C, set(), "A and C must be disjoint")
        self.assertEqual(m.A & m.X, set(), "A and X must be disjoint")
        self.assertEqual(m.C & m.X, set(), "C and X must be disjoint")

        # Class contents follow the locked definitions (precedence AC > C > X).
        self.assertEqual(m.A, set(ALWAYS_CARRY) & set(E))
        self.assertEqual(m.C, {"read_file"})
        self.assertEqual(m.X, {"web_extract", "browser_exec"})

        # With no triggers/expansions, active == residents (A ∪ C).
        self.assertEqual(m.active, m.A | m.C)


# ─── 2. unknown enabled built-ins default to expand_only ───────────────────

class UnknownDefaultsContract(_CarryingContract):
    def test_unknown_enabled_builtin_defaults_to_expand_only(self):
        E = {"clarify", "send_message", "brand_new_builtin"}
        m = self.resolve(enabled=E, always_carry=ALWAYS_CARRY, carry=set())

        self.assertIn("brand_new_builtin", m.X,
                      "an unknown enabled built-in defaults to expand_only")
        self.assertNotIn("brand_new_builtin", m.A)
        self.assertNotIn("brand_new_builtin", m.C)
        self.assertNotIn("brand_new_builtin", m.active,
                         "unknown tool is not active until triggered/expanded")


# ─── 3. always_carry immunity from learned demotion ────────────────────────

class AlwaysCarryImmunityContract(_CarryingContract):
    def test_always_carry_immune_from_learned_demotion(self):
        E = {"clarify", "send_message", "web_extract"}
        # The learned/adaptive layer emits a demotion (always_off) signal
        # naming an always_carry tool. It must be ignored, and warned about.
        m = self.resolve(enabled=E, always_carry=ALWAYS_CARRY, carry=set(),
                         demoted={"clarify"})

        self.assertIn("clarify", m.A, "always_carry tool stays resident")
        self.assertIn("clarify", m.active)
        self.assertNotIn("clarify", m.X, "always_carry never falls to expand_only")
        self.assertTrue(m.warnings, "demoting an always_carry tool must warn")


# ─── 4. promotion expand_only -> carry ─────────────────────────────────────

class PromotionContract(_CarryingContract):
    def test_promotion_expand_only_to_carry(self):
        E = {"clarify", "send_message", "web_extract"}

        before = self.resolve(enabled=E, always_carry=ALWAYS_CARRY, carry=set())
        self.assertIn("web_extract", before.X, "starts as expand_only")
        self.assertNotIn("web_extract", before.C)

        # Promotion == the adaptive carry loadout gains the tool.
        after = self.resolve(enabled=E, always_carry=ALWAYS_CARRY, carry={"web_extract"})
        self.assertIn("web_extract", after.C, "promotion moves expand_only -> carry")
        self.assertNotIn("web_extract", after.X)
        self.assertEqual(before.A, after.A, "promotion does not touch always_carry")


# ─── 5. demotion carry -> expand_only ──────────────────────────────────────

class DemotionContract(_CarryingContract):
    def test_demotion_carry_to_expand_only(self):
        E = {"clarify", "send_message", "web_extract"}

        resident = self.resolve(enabled=E, always_carry=ALWAYS_CARRY, carry={"web_extract"})
        self.assertIn("web_extract", resident.C, "starts as a carry resident")
        self.assertNotIn("web_extract", resident.X)

        # Demotion == the adaptive carry loadout drops the tool.
        demoted = self.resolve(enabled=E, always_carry=ALWAYS_CARRY, carry=set())
        self.assertIn("web_extract", demoted.X, "demotion moves carry -> expand_only")
        self.assertNotIn("web_extract", demoted.C)


# ─── 6/7. expand_only activation without residency change ──────────────────

class ExpandOnlyActivationContract(_CarryingContract):
    def test_expand_only_tool_activates_via_trigger_without_promotion(self):
        E = {"clarify", "send_message", "web_extract"}
        m = self.resolve(enabled=E, always_carry=ALWAYS_CARRY, carry=set(),
                         triggered={"web_extract"})

        self.assertIn("web_extract", m.X, "trigger activation does not change residency")
        self.assertNotIn("web_extract", m.C)
        self.assertIn("web_extract", m.active, "triggered expand_only tool is active")

    def test_expand_only_tool_activates_via_expand_tools_without_promotion(self):
        E = {"clarify", "send_message", "browser_exec"}
        m = self.resolve(enabled=E, always_carry=ALWAYS_CARRY, carry=set(),
                         expanded={"browser_exec"})

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
            prior_active={"ghost_tool"},                      # stale frozen active
        )
        for label, bucket in (("A", m.A), ("C", m.C), ("X", m.X), ("active", m.active)):
            self.assertNotIn("ghost_tool", bucket,
                             f"disabled/absent tool must not enter {label}")


# ─── 9. malformed overlap resolves safely and warns ────────────────────────

class MalformedOverlapContract(_CarryingContract):
    def test_malformed_overlap_fails_safe_toward_carrying_and_warns(self):
        E = {"clarify", "send_message", "web_extract"}
        # web_extract is a carry resident AND named by a demotion signal — a
        # contradictory overlap. Fail safe toward carrying: keep it resident.
        m = self.resolve(enabled=E, always_carry=ALWAYS_CARRY, carry={"web_extract"},
                         demoted={"web_extract"})

        self.assertTrue(m.warnings, "malformed overlap must warn")
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
            triggered=set(), expanded=set(), passthrough=set(),
            demoted=set(), prior_active=set(),
        )
        m = _read_model(raw)
        self.assertIsNotNone(m.active, "fail-open must still return an active set")
        self.assertTrue(set(E) <= m.active,
                        "fail-open returns the whole enabled ceiling (no narrowing)")


# ─── 11. MCP/plugin pass-through outside the built-in partition ────────────

class PassthroughContract(_CarryingContract):
    def test_mcp_plugin_passthrough_outside_builtin_partition(self):
        mcp_name = "mcp__github__create_issue"
        # Grounded in the runtime's own notion of an MCP tool.
        self.assertTrue(plugin._is_mcp_tool(mcp_name),
                        "sanity: runtime recognizes the MCP tool name")

        E = {"clarify", "send_message", "read_file"}
        m = self.resolve(enabled=E, always_carry=ALWAYS_CARRY, carry=set(),
                         passthrough={mcp_name})

        self.assertNotIn(mcp_name, m.A)
        self.assertNotIn(mcp_name, m.C)
        self.assertNotIn(mcp_name, m.X,
                         "MCP/plugin tools live outside the built-in partition")
        self.assertEqual(m.A | m.C | m.X, set(E),
                         "the partition covers only built-ins, not passthrough")
        self.assertIn(mcp_name, m.active,
                      "MCP/plugin tools pass through and are never narrowed")


# ─── 12. v1 learned-state normalization into v2 (no read-time write) ───────

class LearnedV1NormalizationContract(unittest.TestCase):
    def test_v1_learned_state_normalizes_to_v2_without_read_time_write(self):
        v1_doc = {
            "version": 1,
            "updated_at": "2026-01-01T00:00:00Z",
            "scopes": {"assistant-a:telegram": {"always_on": ["web_extract"],
                                                "always_off": ["memory"]}},
            "global": {},
        }
        with _temp_learned_state(v1_doc) as path:
            before_bytes = path.read_bytes()
            before_mtime = path.stat().st_mtime_ns

            # Reading a v1 file must NOT rewrite it (no read-time write). This
            # invariant is pinned first (it already holds and must keep holding).
            learned_mod.load_state(force=True)
            self.assertEqual(path.read_bytes(), before_bytes,
                             "loading learned state must not rewrite the file")
            self.assertEqual(path.stat().st_mtime_ns, before_mtime,
                             "loading learned state must not touch the file mtime")

            # Missing v2 behavior: a v1 document normalizes into the v2 shape
            # in memory (not discarded), mapping old always_on -> carry residency.
            normalize = _resolve_learned_v2_normalizer()
            if normalize is None:
                self.fail(
                    "v2 learned-state normalizer not implemented (expected e.g. "
                    "tool_belt_plugin.learned.normalize_state(doc) -> v2 dict)"
                )
            v2 = normalize(v1_doc)
            self.assertEqual(_get(v2, "version"), 2,
                             "v1 learned state normalizes to v2 in memory")
            scope = (_get(v2, "scopes") or {}).get("assistant-a:telegram") or {}
            self.assertIn("web_extract", _as_set(_get(scope, "carry")) or set(),
                          "a v1 always_on entry becomes a v2 'carry' resident")


# ─── 13. v1 telemetry normalization incl. conditional residency_inferred ───

class TelemetryV1NormalizationContract(unittest.TestCase):
    @staticmethod
    def _truthy(v):
        return bool(v)

    def test_v1_telemetry_normalizes_with_conditional_residency_inference(self):
        normalize = _resolve_telemetry_normalizer()
        if normalize is None:
            self.fail(
                "v2 telemetry-row normalizer not implemented (expected e.g. "
                "tool_belt_plugin.analyze.normalize_prediction_row(row) -> v2 row)"
            )

        # Complete membership: ceiling + residents + active are all present, so
        # residency CAN be reconstructed -> residency_inferred is True.
        complete = {
            "prediction_id": "p1",
            "ceiling_tools": ["clarify", "read_file", "web_extract"],
            "always_on_tools": ["clarify", "read_file"],
            "allowed_tools": ["clarify", "read_file"],
            "cut_tools": ["web_extract"],
        }
        row = normalize(complete)
        self.assertTrue(self._truthy(_get(row, "residency_inferred")),
                        "complete membership permits residency inference")
        residency = _get(row, "residency")
        self.assertIsNotNone(residency,
                             "a complete row carries a residency mapping")

        # Incomplete membership: no ceiling / no residents -> the partition is
        # unknowable, so residency must NOT be inferred.
        incomplete = {"prediction_id": "p2", "allowed_tools": ["clarify"]}
        row2 = normalize(incomplete)
        self.assertFalse(self._truthy(_get(row2, "residency_inferred")),
                         "incomplete membership must not infer residency")


# ─── 14. trigger definitions byte-equivalent across promotion/demotion ─────

class TriggerImmutabilityContract(unittest.TestCase):
    """Promotion/demotion change residency only — never trigger definitions.

    Exercises the REAL learned-merge path. A demotion that strips the demoted
    tool out of its trigger group (so it could no longer be trigger-activated
    as expand_only) is exactly the v2 violation this pins.
    """

    def _base_preset(self):
        return presets_mod.Preset(
            name="carrying-contract",
            always_on=["clarify"],
            triggers=[
                presets_mod.TriggerGroup(
                    name="web_extract",
                    tools=["web_extract"],
                    keyword_patterns=[re.compile(r"https?://", re.IGNORECASE)],
                    exclude_patterns=[re.compile(r"\blater\b", re.IGNORECASE)],
                    has_attachment=None,
                )
            ],
            always_off=[],
        )

    def test_trigger_definitions_unchanged_across_promotion_and_demotion(self):
        cfg = {"learned_mode": "apply", "channels": {}}
        scope = "assistant-a:telegram"

        base = self._base_preset()
        fp0 = _trigger_fingerprint(base.triggers)

        # Promotion: learned promotes web_extract into residency.
        promote_doc = {"scopes": {scope: {"always_on": ["web_extract"]}}, "global": {}}
        with _temp_learned_state(promote_doc):
            promoted = learned_mod.apply_to_preset(self._base_preset(), cfg, scope)
        self.assertEqual(_trigger_fingerprint(promoted.preset.triggers), fp0,
                         "promotion must not alter trigger definitions")

        # Demotion: learned demotes web_extract. It must remain in its trigger
        # group (expand_only tools stay trigger-activatable).
        demote_doc = {"scopes": {scope: {"always_off": ["web_extract"]}}, "global": {}}
        with _temp_learned_state(demote_doc):
            demoted = learned_mod.apply_to_preset(self._base_preset(), cfg, scope)
        self.assertEqual(_trigger_fingerprint(demoted.preset.triggers), fp0,
                         "demotion must not alter trigger definitions")


# ─── 15. cache-on trigger activation grows the frozen active set once ──────

class CacheOnFrozenActiveSetContract(_CarryingContract):
    def test_cache_on_trigger_activation_grows_frozen_active_set_once(self):
        E = {"clarify", "send_message", "read_file", "web_extract", "browser_exec"}
        carry = {"read_file"}

        # message 0 — freeze: residency fixed, no triggers fired yet.
        m0 = self.resolve(enabled=E, always_carry=ALWAYS_CARRY, carry=carry)
        frozen_residency = (m0.A, m0.C, m0.X)

        # message 1 — web_extract (an expand_only tool) newly triggers. Prior
        # active carries forward (the session's frozen active set).
        m1 = self.resolve(enabled=E, always_carry=ALWAYS_CARRY, carry=carry,
                          triggered={"web_extract"}, prior_active=m0.active)
        self.assertEqual((m1.A, m1.C, m1.X), frozen_residency,
                         "carrying assignment is fixed per session")
        self.assertTrue(m0.active < m1.active,
                        "a newly triggered tool grows the active set")
        self.assertIn("web_extract", m1.active)

        # message 2 — no new trigger: the previously triggered tool STAYS active.
        m2 = self.resolve(enabled=E, always_carry=ALWAYS_CARRY, carry=carry,
                          triggered=set(), prior_active=m1.active)
        self.assertEqual(m2.active, m1.active,
                         "active is monotonic — grew once, now stable")

        # message 3 — re-trigger the same tool: the union is idempotent.
        m3 = self.resolve(enabled=E, always_carry=ALWAYS_CARRY, carry=carry,
                          triggered={"web_extract"}, prior_active=m2.active)
        self.assertEqual(m3.active, m2.active,
                         "re-trigger does not regrow the frozen active set")


if __name__ == "__main__":
    unittest.main()
