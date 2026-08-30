"""Session-window contract: use ALL available history up to a ceiling.

Dale's directive (2026-08-30): "If we have more than 20 sessions, it should use
all available, not be limited to 20. 20 is the floor, 100 should be the ceiling."

Locks:
  · DEFAULTS["session_window"] == 100 (the ceiling) while
    demote_min_sessions_no_use == 20 stays the floor.
  · Behavioral: with >20 sessions of history, a carry tool used ONLY in the
    older part of the history (outside the most-recent 20) is protected from
    demotion — the shaper reads the whole window, not just the last 20.
Both fail on pre-fix code (window default 20).
"""

import unittest

from tests.test_carrying_model import _pred_row, _compute, shaper


class SessionWindowDefaults(unittest.TestCase):
    def test_window_ceiling_is_100_and_floor_is_20(self):
        self.assertEqual(shaper.DEFAULTS["session_window"], 100,
                         "session_window is a 100-session ceiling")
        self.assertEqual(shaper.DEFAULTS["demote_min_sessions_no_use"], 20,
                         "20 sessions stays the demotion floor")


class SessionWindowUsesAllHistory(unittest.TestCase):
    SCOPE = "assistant-a:telegram"

    def _history(self, n_sessions, used_in_sessions):
        """n sessions, newest ts = highest index; ``web_extract`` is adaptive
        carry everywhere and called only in ``used_in_sessions`` (by index)."""
        E = ["clarify", "web_extract", "terminal"]
        preds, calls = [], []
        for i in range(n_sessions):
            pid = f"p{i}"
            preds.append(_pred_row(
                self.SCOPE, f"s{i}", pid,
                ceiling=E, always_carry=["clarify"],
                carry=["web_extract", "terminal"],
                active=["clarify", "web_extract", "terminal"], ts=float(i),
            ))
            calls.append({
                "schema_version": 2, "prediction_id": pid,
                "tool_name": "terminal", "was_initially_active": True,
            })
            if i in used_in_sessions:
                calls.append({
                    "schema_version": 2, "prediction_id": pid,
                    "tool_name": "web_extract", "was_initially_active": True,
                })
        return preds, calls

    def test_old_use_beyond_last_20_sessions_protects_from_demotion(self):
        # 30 sessions; web_extract used only in the 5 OLDEST (indices 0-4),
        # i.e. outside the most-recent-20 slice a 20-window shaper would read.
        preds, calls = self._history(30, used_in_sessions={0, 1, 2, 3, 4})
        recs = _compute(self.SCOPE, preds, calls,
                        window=shaper.DEFAULTS["session_window"])
        demoted = {d["tool"] for d in recs["demote"]}
        self.assertNotIn(
            "web_extract", demoted,
            "a use anywhere in the available history (≤100 sessions) protects "
            "a carry tool; the shaper must not truncate to the last 20")
        self.assertEqual(recs["sessions_considered"], 30,
                         "all 30 available sessions are considered")

    def test_ceiling_still_truncates_beyond_100(self):
        preds, calls = self._history(105, used_in_sessions=set())
        recs = _compute(self.SCOPE, preds, calls,
                        window=shaper.DEFAULTS["session_window"])
        self.assertEqual(recs["sessions_considered"], 100,
                         "history beyond the 100-session ceiling is truncated")

    def test_below_floor_never_demotes(self):
        preds, calls = self._history(19, used_in_sessions=set())
        recs = _compute(self.SCOPE, preds, calls,
                        window=shaper.DEFAULTS["session_window"])
        self.assertEqual(recs["demote"], [],
                         "fewer than 20 sessions: demotion never fires")


if __name__ == "__main__":
    unittest.main()
