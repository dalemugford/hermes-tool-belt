"""The economic demotion test — session-priced (Phase 2, math audit fix).

Dale's promise: "When Tool Belt determines tokens would be saved by not
carrying a tool, it demotes it; conversely, if carrying a tool would be more
efficient, it's carried." Token-denominated by design — no price table.

``expand_tools`` is sticky (expand once, carried for the rest of the
session), so a session WITH use costs about the same either way plus one
round-trip; demotion only saves in sessions WITHOUT use:

    saving  = schema_size × billable exposures in sessions WITHOUT use
    penalty = EXPAND_ROUND_TRIP_TOKENS × sessions WITH use
    demote when saving > k × penalty; promote when penalty > saving

Billable exposures: cache off = api calls per prediction (min 1); cache
on/unknown = 1 per session. The burst-use and api-exposure locks fail on the
call-priced math; the class locks fail on pre-economic code entirely.
"""

import unittest

from tests.test_carrying_model import (
    _pred_row, _expansion_call, _trigger_call, _compute, shaper,  # noqa: F401
)
from tool_belt_plugin.shaping import EXPAND_ROUND_TRIP_TOKENS


def _use_call(pid, tool):
    """A plain carried-tool use (non-trigger, non-expansion)."""
    return {
        "schema_version": 2, "prediction_id": pid, "tool_name": tool,
        "was_initially_active": True, "activation_source": "carry",
    }


class EconomicDemotion(unittest.TestCase):
    SCOPE = "assistant-a:telegram"
    E = ["clarify", "browser_exec", "terminal"]

    def _sessions(self, n, browser_use_sessions=(), calls_per_use_session=1,
                  calls_factory=_use_call):
        """n sessions, one prediction each; ``browser_exec`` carried
        everywhere and called in the named session indices."""
        preds, calls = [], []
        for i in range(n):
            pid = f"p{i}"
            preds.append(_pred_row(
                self.SCOPE, f"s{i}", pid,
                ceiling=self.E, always_carry=["clarify"],
                carry=["browser_exec", "terminal"],
                active=self.E, ts=float(i),
            ))
            calls.append(_use_call(pid, "terminal"))
            if i in browser_use_sessions:
                for _ in range(calls_per_use_session):
                    calls.append(calls_factory(pid, "browser_exec"))
        return preds, calls

    def test_rarely_used_fat_tool_demotes_as_uneconomic(self):
        # 30 sessions cache-off (1 pred each → 1 exposure each), 2000-token
        # schema, used in one session: saving = 2000×29 = 58000 >
        # k2 × penalty (1×1500 = 1500) → demote. Pre-economic code never
        # demoted a tool with ANY use.
        preds, calls = self._sessions(30, browser_use_sessions={7})
        recs = _compute(self.SCOPE, preds, calls, window=100,
                        schema_sizes={"browser_exec": 2000}, cache_mode="off")
        entry = {d["tool"]: d for d in recs["demote"]}
        self.assertIn("browser_exec", entry)
        d = entry["browser_exec"]
        self.assertEqual(d["evidence"], "carry_uneconomic")
        self.assertEqual(d["sessions_with_use"], 1)
        self.assertEqual(d["sessions_without_use"], 29)
        self.assertEqual(d["carry_tokens"], 2000 * 29)
        self.assertEqual(d["demote_tokens"], EXPAND_ROUND_TRIP_TOKENS)

    def test_burst_use_in_one_session_does_not_defend_carry(self):
        # 25 calls, all inside ONE session. Sticky expansion means that
        # session costs one round-trip, not 25 — the tool still demotes.
        # Fails on call-priced math (25×1500×k = 75000 > 2000×30) and on
        # pre-economic code.
        preds, calls = self._sessions(30, browser_use_sessions={3},
                                      calls_per_use_session=25)
        recs = _compute(self.SCOPE, preds, calls, window=100,
                        schema_sizes={"browser_exec": 2000}, cache_mode="off")
        entry = {d["tool"]: d for d in recs["demote"]}
        self.assertIn("browser_exec", entry)
        self.assertEqual(entry["browser_exec"]["uses_in_window"], 25)
        self.assertEqual(entry["browser_exec"]["sessions_with_use"], 1)
        self.assertEqual(entry["browser_exec"]["demote_tokens"],
                         EXPAND_ROUND_TRIP_TOKENS)

    def test_regularly_used_fat_tool_holds(self):
        # Used in 25 of 30 sessions: saving = 2000×5 = 10000 ≤
        # k2 × penalty (25×1500 = 37500) → carrying is cheaper, hold.
        preds, calls = self._sessions(30, browser_use_sessions=set(range(25)))
        recs = _compute(self.SCOPE, preds, calls, window=100,
                        schema_sizes={"browser_exec": 2000}, cache_mode="off")
        self.assertNotIn("browser_exec", {d["tool"] for d in recs["demote"]})

    def test_trigger_covered_uses_are_free(self):
        # Trigger activations stay free for a demoted tool, so they don't
        # defend a carry slot: 5 trigger-only sessions count as no-use.
        preds, calls = self._sessions(30, browser_use_sessions=set(range(5)),
                                      calls_factory=_trigger_call)
        recs = _compute(self.SCOPE, preds, calls, window=100,
                        schema_sizes={"browser_exec": 2000}, cache_mode="off")
        entry = {d["tool"]: d for d in recs["demote"]}
        self.assertIn("browser_exec", entry)
        self.assertEqual(entry["browser_exec"]["sessions_with_use"], 0)
        self.assertEqual(entry["browser_exec"]["evidence"], "carry_unused")

    def test_cache_on_lean_tool_holds(self):
        # Cache on (1 exposure/session), 150-token schema, used in 2
        # sessions: saving = 150×28 = 4200 ≤ k2×(2×1500) = 6000 → hold.
        preds, calls = self._sessions(30, browser_use_sessions={0, 1})
        recs = _compute(self.SCOPE, preds, calls, window=100,
                        schema_sizes={"browser_exec": 150}, cache_mode="on")
        self.assertNotIn("browser_exec", {d["tool"] for d in recs["demote"]})

    def test_zero_use_limit_case_still_demotes_as_unused(self):
        preds, calls = self._sessions(30)
        recs = _compute(self.SCOPE, preds, calls, window=100)
        entry = {d["tool"]: d for d in recs["demote"]}
        self.assertIn("browser_exec", entry)
        self.assertEqual(entry["browser_exec"]["evidence"], "carry_unused")
        self.assertEqual(entry["browser_exec"]["sessions_without_use"], 30)

    def test_api_call_counts_scale_cache_off_exposures(self):
        # A cache-off agentic turn pays the manifest on every API call. With
        # 10 api calls per prediction the zero-use saving is 10× the
        # prediction count. Fails before api_call_counts existed.
        preds, calls = self._sessions(30)
        api_counts = {f"p{i}": 10 for i in range(30)}
        recs = _compute(self.SCOPE, preds, calls, window=100,
                        schema_sizes={"browser_exec": 2000}, cache_mode="off",
                        api_call_counts=api_counts)
        entry = {d["tool"]: d for d in recs["demote"]}
        self.assertEqual(entry["browser_exec"]["carry_tokens"], 2000 * 300)


class EconomicPromotion(unittest.TestCase):
    SCOPE = "assistant-a:telegram"
    E = ["clarify", "web_extract"]

    def _expansion_history(self, n_sessions, use_sessions, preds_per_session=1,
                           calls_in_use_session=1):
        preds, calls = [], []
        for i in range(n_sessions):
            for j in range(preds_per_session):
                pid = f"p{i}-{j}"
                preds.append(_pred_row(
                    self.SCOPE, f"s{i}", pid,
                    ceiling=self.E, always_carry=["clarify"], carry=[],
                    active=["clarify"], ts=float(i * 1000 + j),
                ))
                if i in use_sessions and j == 0:
                    for _ in range(calls_in_use_session):
                        calls.append(_expansion_call(pid, "web_extract"))
        return preds, calls

    def test_promotion_vetoed_when_carrying_would_cost_more(self):
        # Expanded in 2 of 30 cache-off sessions (5 predictions each, 3
        # calls total → meets the anti-flap gates): penalty = 2×1500 = 3000
        # vs marginal carry 388×(150−10) = 54320 → expanding stays cheaper.
        preds, calls = self._expansion_history(
            30, use_sessions={0, 1}, preds_per_session=5,
            calls_in_use_session=2)
        recs = _compute(self.SCOPE, preds, calls, window=100, cache_mode="off")
        self.assertNotIn("web_extract", {p["tool"] for p in recs["promote"]})

    def test_promotion_granted_when_expansion_spend_exceeds_carry(self):
        # Expanded in every one of 3 cache-on sessions: no unused sessions,
        # marginal carry cost 0 vs penalty 3×1500 → promote, economics
        # stamped on the entry.
        preds, calls = self._expansion_history(3, use_sessions={0, 1, 2})
        recs = _compute(self.SCOPE, preds, calls, cache_mode="on")
        entry = {p["tool"]: p for p in recs["promote"]}
        self.assertIn("web_extract", entry)
        self.assertEqual(entry["web_extract"]["expansion_tokens"],
                         3 * EXPAND_ROUND_TRIP_TOKENS)
        self.assertEqual(entry["web_extract"]["carry_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
