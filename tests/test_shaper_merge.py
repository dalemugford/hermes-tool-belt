"""Between-session shaper contracts: evidence → recommendations → learned.json.

The shaper is the only writer that turns telemetry into carrying assignments,
so these pin the two halves of that path:

  * ``shaping.compute_scope_recommendations`` — what counts as promotion
    evidence (expansion/recovery only, never a trigger activation), what
    counts as demotion evidence (an unused adaptive ``carry`` resident on a
    residency-inferred row), and the candidate-validation domain (scope-local
    concrete tool names — never a category, never another agent's tool).
  * ``shaping.merge_into_learned`` — the learned.json write: schema v2 fields,
    class moves in both directions, unrelated metadata preserved, dry-run
    writing nothing.

Plus the ``learned`` reconciliation the shaper's output is read back through
(``reset_scope`` scope isolation, carry-wins overlap normalization).
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

# Reuse conftest's plugin-loader bootstrap (hyphenated plugin dir).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401,E402

learned_mod = importlib.import_module("tool_belt_plugin.learned")
from tests.shaping_fixtures import (  # noqa: E402
    shaper, _pred_row, _expansion_call, _trigger_call, _compute,
)

_LOGGER_LEARNED = "tool_belt_plugin.learned"


def _sparse_pred_row(scope, sid, pid, *, carry, ts=0.0):
    """A sparse prediction row: residents only, no ceiling/active. The
    normalizer cannot reconstruct residency → ``residency_inferred`` is False."""
    return {
        "schema_version": 2,
        "scope": scope,
        "hermes_session_id": sid,
        "prediction_id": pid,
        "ts": ts,
        "carry_tools": list(carry),
    }



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
        # ``compute_scope_recommendations`` checks candidates against) was built
        # from the GLOBAL tool-call index, so another agent's names could validate
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

    def test_sparse_row_cannot_drive_demotion(self):
        # 20 sessions of sparse rows: residents present but no ceiling/active,
        # so residency is not inferable and the tools cannot demote.
        preds = [
            _sparse_pred_row(self.SCOPE, f"s{i}", f"p{i}",
                             carry=["read_file", "web_search"], ts=i)
            for i in range(20)
        ]
        recs = _compute(self.SCOPE, preds, [])
        self.assertEqual(recs["demote"], [],
                         "a sparse (residency_inferred=False) row cannot demote")

    def test_candidate_outside_the_enabled_domain_is_rejected(self):
        """``_valid``'s enabled-domain gate, not just the report field.

        Demote candidates come from the reconstructed ``residency.carry`` list,
        which is a *different* source from the tool lists that build the
        validation domain. A name reaching the demote arm without ever having
        been observed as a concrete enabled tool for this scope (a stale or
        hand-edited residency entry, a category string) must be dropped with a
        warning — otherwise it lands in learned.json's per-tool ``expand_only``
        list and can never be recovered by expand_tools.
        """
        sessions = {}
        for i in range(25):
            sessions[f"s{i}"] = [{
                "prediction_id": f"p{i}",
                "ts": 1000 + i,
                "residency_inferred": True,
                # ``phantom_tool`` is carried per the residency reconstruction
                # but appears in NO tool list, so it is not a concrete enabled
                # name for this scope.
                "residency": {"carry": ["phantom_tool", "web_extract"]},
                "ceiling_tools": ["clarify", "web_extract"],
                "carry_tools": ["web_extract"],
                "always_carry_tools": ["clarify"],
            }]
        with self.assertLogs("tool_belt_plugin.shaping", level="WARNING") as cm:
            recs = shaper.compute_scope_recommendations(
                scope="assistant-a:telegram",
                sessions=sessions,
                calls_by_pred={},
                window_days=7,
                promote_min_sessions=2,
                promote_min_calls=3,
                demote_min_sessions_no_use=20,
            )
        demoted = {d["tool"] for d in recs["demote"]}
        self.assertNotIn("phantom_tool", demoted,
                         "a candidate outside the enabled domain must be rejected")
        self.assertNotIn("phantom_tool", recs["enabled_tool_names"],
                         "PRECONDITION: the name really is outside the domain")
        self.assertIn("web_extract", demoted,
                      "PRECONDITION: an in-domain unused resident still demotes")
        self.assertTrue(any("phantom_tool" in line for line in cm.output),
                        "the rejection is warned, not silent")


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

    def test_category_never_becomes_a_candidate(self):
        # A toolset/category ("web") is a grouping key the model names in
        # expand_tools — it is never a concrete tool. Candidates are drawn
        # only from concrete observed tool names, so the category can reach
        # neither the validation domain nor a carrying list.
        E = ["clarify", "send_message", "web_extract"]
        preds, calls = [], []
        for i in range(3):
            pid = f"p{i}"
            preds.append(_pred_row(
                self.SCOPE, f"s{i}", pid,
                ceiling=E, always_carry=["clarify", "send_message"], carry=[],
                active=["clarify", "send_message"], ts=i,
            ))
            # The model expanded the *category*; only the resolved tool is a
            # dispatch with promotion evidence behind it.
            calls.append(_expansion_call(pid, "expand_tools"))
            calls.append(_expansion_call(pid, "web_extract"))
        recs = _compute(self.SCOPE, preds, calls)
        promoted = {p["tool"] for p in recs["promote"]}
        self.assertIn("web_extract", promoted)
        self.assertNotIn("web", promoted, "a category name is never a candidate")
        self.assertNotIn("web", recs["enabled_tool_names"],
                         "a category name never enters the validation domain")
        self.assertNotIn("expand_tools", promoted,
                         "the expansion meta-call is never a candidate")

    def test_dry_run_performs_no_writes(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            recs = self._recs(promote=["web_extract"], enabled=["web_extract"])
            _state, changed = shaper.merge_into_learned(tmp, {self.SCOPE: recs}, True)
            self.assertTrue(changed, "dry-run still reports the intended change")
            self.assertFalse((tmp / "learned.json").exists(),
                             "dry-run writes nothing to disk")


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


if __name__ == "__main__":
    unittest.main()
