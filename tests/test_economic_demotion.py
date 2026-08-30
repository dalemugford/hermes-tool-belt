"""The economic demotion test (Phase 2).

Dale's promise: "When Tool Belt determines tokens would be saved by not
carrying a tool, it demotes it; conversely, if carrying a tool would be more
efficient, it's carried." Token-denominated by design — no price table.

    carry_tokens  = schema_size(tool) × billable manifest exposures
    demote_tokens = non-trigger uses × EXPAND_ROUND_TRIP_TOKENS
    demote when carry_tokens > k × demote_tokens; promote on the reversed
    inequality (hysteresis band between them holds the current class).

The key new locks fail on pre-fix code, where demotion was binary
(zero-use only) and promotion never weighed carry cost.
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

    def _sessions(self, n, uses_of_browser=0, calls_factory=_use_call):
        preds, calls = [], []
        used = 0
        for i in range(n):
            pid = f"p{i}"
            preds.append(_pred_row(
                self.SCOPE, f"s{i}", pid,
                ceiling=self.E, always_carry=["clarify"],
                carry=["browser_exec", "terminal"],
                active=self.E, ts=float(i),
            ))
            calls.append(_use_call(pid, "terminal"))
            if used < uses_of_browser:
                calls.append(calls_factory(pid, "browser_exec"))
                used += 1
        return preds, calls

    def test_rarely_used_fat_tool_demotes_as_uneconomic(self):
        # 30 sessions, cache off (every prediction billable), 2000-token schema,
        # used once: carry = 2000×30 = 60000 > k2 × 1500 = 3000 → demote.
        # Pre-fix code never demoted a tool with ANY use — this fails there.
        preds, calls = self._sessions(30, uses_of_browser=1)
        recs = _compute(self.SCOPE, preds, calls, window=100,
                        schema_sizes={"browser_exec": 2000}, cache_mode="off")
        entry = {d["tool"]: d for d in recs["demote"]}
        self.assertIn("browser_exec", entry)
        d = entry["browser_exec"]
        self.assertEqual(d["evidence"], "carry_uneconomic")
        self.assertEqual(d["uses_in_window"], 1)
        self.assertEqual(d["carry_tokens"], 2000 * 30)
        self.assertEqual(d["demote_tokens"], EXPAND_ROUND_TRIP_TOKENS)

    def test_frequently_used_fat_tool_holds(self):
        # Same tool used 25 times: demote side = 25×1500 = 37500, k× = 75000
        # > carry 60000 → carrying is the cheaper side, hold.
        preds, calls = self._sessions(30, uses_of_browser=25)
        recs = _compute(self.SCOPE, preds, calls, window=100,
                        schema_sizes={"browser_exec": 2000}, cache_mode="off")
        self.assertNotIn("browser_exec",
                         {d["tool"] for d in recs["demote"]})

    def test_trigger_covered_uses_are_free(self):
        # Uses that arrived via triggers stay free after demotion, so they
        # don't defend a carry slot. Pre-fix, any call protected the tool.
        preds, calls = self._sessions(30, uses_of_browser=5,
                                      calls_factory=_trigger_call)
        recs = _compute(self.SCOPE, preds, calls, window=100,
                        schema_sizes={"browser_exec": 2000}, cache_mode="off")
        entry = {d["tool"]: d for d in recs["demote"]}
        self.assertIn("browser_exec", entry)
        self.assertEqual(entry["browser_exec"]["uses_in_window"], 0)
        self.assertEqual(entry["browser_exec"]["evidence"], "carry_unused")

    def test_cache_on_scope_is_harder_to_demote(self):
        # Cache on: exposures ≈ sessions (30), carry = 2000×30 with one use
        # still demotes; but a lean 150-token schema does not:
        # 150×30 = 4500 ≤ k2×1500 = 3000? No — 4500 > 3000. Use 2 uses:
        # 150×30 = 4500 ≤ 2×(2×1500) = 6000 → hold.
        preds, calls = self._sessions(30, uses_of_browser=2)
        recs = _compute(self.SCOPE, preds, calls, window=100,
                        schema_sizes={"browser_exec": 150}, cache_mode="on")
        self.assertNotIn("browser_exec",
                         {d["tool"] for d in recs["demote"]})

    def test_zero_use_limit_case_still_demotes_as_unused(self):
        preds, calls = self._sessions(30, uses_of_browser=0)
        recs = _compute(self.SCOPE, preds, calls, window=100)
        entry = {d["tool"]: d for d in recs["demote"]}
        self.assertIn("browser_exec", entry)
        self.assertEqual(entry["browser_exec"]["evidence"], "carry_unused")
        self.assertEqual(entry["browser_exec"]["sessions_without_use"], 30)


class EconomicPromotionVeto(unittest.TestCase):
    SCOPE = "assistant-a:telegram"

    def test_promotion_vetoed_when_carrying_would_cost_more(self):
        # Meets the anti-flap gates (3 sessions, 3 expansion calls) but the
        # scope is cache-off with 20 predictions per session: carrying would
        # cost 388×60 = 23280 tokens vs 3×1500 = 4500 observed expansion
        # spend → expanding on demand is cheaper, hold. Pre-fix promoted.
        E = ["clarify", "web_extract"]
        preds, calls = [], []
        for i in range(3):
            for j in range(20):
                pid = f"p{i}-{j}"
                preds.append(_pred_row(
                    self.SCOPE, f"s{i}", pid,
                    ceiling=E, always_carry=["clarify"], carry=[],
                    active=["clarify"], ts=float(i * 100 + j),
                ))
                if j == 0:
                    calls.append(_expansion_call(pid, "web_extract"))
        recs = _compute(self.SCOPE, preds, calls, cache_mode="off")
        self.assertNotIn("web_extract", {p["tool"] for p in recs["promote"]})

    def test_promotion_granted_when_expansion_spend_exceeds_carry(self):
        # Cache on, 3 sessions → exposures 3: carrying costs 388×3 = 1164
        # vs 4500 expansion spend → promote, with the economics stamped.
        E = ["clarify", "web_extract"]
        preds, calls = [], []
        for i in range(3):
            pid = f"p{i}"
            preds.append(_pred_row(
                self.SCOPE, f"s{i}", pid,
                ceiling=E, always_carry=["clarify"], carry=[],
                active=["clarify"], ts=float(i),
            ))
            calls.append(_expansion_call(pid, "web_extract"))
        recs = _compute(self.SCOPE, preds, calls, cache_mode="on")
        entry = {p["tool"]: p for p in recs["promote"]}
        self.assertIn("web_extract", entry)
        self.assertEqual(entry["web_extract"]["expansion_tokens"],
                         3 * EXPAND_ROUND_TRIP_TOKENS)
        self.assertGreater(entry["web_extract"]["expansion_tokens"],
                           entry["web_extract"]["carry_tokens"])


if __name__ == "__main__":
    unittest.main()
