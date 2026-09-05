"""Recency window contract: the shaper's window is DAYS, anchored in the data.

A session-count window would make the shaper's verdict depend on how chatty
a scope had been rather than on how recent its evidence was: one call from
months ago could protect a tool forever, and a scope that ran fifty short
sessions in a day would age out yesterday's evidence. ``window_days`` avoids
both: a session is evidence only when its LAST activity falls inside the
trailing window, and "now" is the newest activity in the data handed to the
shaper — never ``time.time()``.

Locks here (each fails on a session-count shaper):

  · Day filtering — sessions at 0/3/6/10/20 days back: a 7-day window
    considers 3 of them, a 30-day window all 5. A session-count window
    considers all 5 either way.
  · Data-relative "now" — the same history timestamped in 2001 gives the
    same answer. Against a wall-clock cutoff every session would fall
    outside the window and the shaper would consider nothing.
  · The floor is still counted in SESSIONS, but only sessions *inside the
    day window* count toward it: with 1 session in-window and a floor of 2,
    demotion never fires and the porcelain says so
    (``demote_skipped_insufficient_sessions``); with 2, it can.
  · The result dict reports ``window_days``.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from tests.shaping_fixtures import _pred_row, _compute

DAY = 86400
#: A fixed anchor deep in the past (2001-09-09). Every fixture below is dated
#: relative to it, so a shaper that measured recency against the wall clock
#: would place the entire history outside any sane window.
ANCHOR = 1_000_000_000.0

_PLUGIN_DIR = Path(__file__).resolve().parent.parent


def _load_shape_ceiling():
    """The shaper CLI module (for its porcelain builder)."""
    spec = importlib.util.spec_from_file_location(
        "tool_belt_shape_ceiling_day_window",
        _PLUGIN_DIR / "scripts" / "shape-ceiling.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DayWindowFiltering(unittest.TestCase):
    SCOPE = "assistant-a:telegram"
    #: Age of each session, in days before the newest one.
    AGES_DAYS = (0, 3, 6, 10, 20)

    def _history(self, ages_days=AGES_DAYS):
        """One session per age, all carrying the same loadout. ``web_extract``
        is an adaptive carry resident nobody ever calls."""
        E = ["clarify", "web_extract", "terminal"]
        preds, calls = [], []
        for i, age in enumerate(ages_days):
            pid = f"p{i}"
            preds.append(_pred_row(
                self.SCOPE, f"s{i}", pid,
                ceiling=E, always_carry=["clarify"],
                carry=["web_extract", "terminal"],
                active=["clarify", "web_extract", "terminal"],
                ts=ANCHOR - age * DAY,
            ))
            calls.append({
                "schema_version": 2, "prediction_id": pid,
                "tool_name": "terminal", "was_initially_active": True,
            })
        return preds, calls

    def test_a_wider_window_reaches_the_older_sessions(self):
        preds, calls = self._history()
        recs = _compute(self.SCOPE, preds, calls, window_days=30,
                        demote_min_sessions_no_use=2)
        self.assertEqual(recs["sessions_considered"], len(self.AGES_DAYS),
                         "30 days covers the whole history")

    def test_now_is_the_newest_session_not_the_wall_clock(self):
        # The fixture is dated 2001. Under a wall-clock cutoff a 7-day window
        # would contain nothing at all; under the data-relative cutoff it
        # contains exactly the same three sessions as a fresh history would.
        preds, calls = self._history()
        recs = _compute(self.SCOPE, preds, calls, window_days=7,
                        demote_min_sessions_no_use=2)
        self.assertEqual(recs["sessions_considered"], 3,
                         "'now' is the newest activity in the data; telemetry "
                         "from 2001 replays exactly as it did when it was new")
        # The porcelain/merge consumers read the window off the result dict.
        self.assertEqual(recs["window_days"], 7)

    def test_a_use_outside_the_window_no_longer_protects_a_carry_tool(self):
        # web_extract used ONLY in the oldest session (20 days back). Inside
        # the 7-day window it is an unused carry resident and demotes — the
        # whole point of a recency window: evidence expires.
        preds, calls = self._history()
        oldest_pid = f"p{len(self.AGES_DAYS) - 1}"
        calls.append({
            "schema_version": 2, "prediction_id": oldest_pid,
            "tool_name": "web_extract", "was_initially_active": True,
        })
        stale = _compute(self.SCOPE, preds, calls, window_days=7,
                         demote_min_sessions_no_use=2)
        self.assertIn("web_extract", {d["tool"] for d in stale["demote"]},
                      "a use 20 days ago is outside a 7-day window and does "
                      "not protect the tool")
        # Widen the window past that use and the same evidence protects again.
        fresh = _compute(self.SCOPE, preds, calls, window_days=30,
                         demote_min_sessions_no_use=2)
        self.assertNotIn("web_extract", {d["tool"] for d in fresh["demote"]},
                         "inside a 30-day window the use is evidence again")


class FloorIsCountedInSessionsInsideTheWindow(unittest.TestCase):
    """``demote_min_sessions_no_use`` is a floor in SESSIONS applied to the
    sessions that survive the DAY filter — the two thresholds answer different
    questions ("is there enough evidence?" vs "is it still current?").

    Fails if the floor is ever re-expressed in days, or if it is applied to
    the full history instead of the in-window subset (which would let a scope
    with one recent session demote on evidence from a month ago)."""

    SCOPE = "assistant-a:telegram"

    def _history(self, ages_days):
        E = ["clarify", "web_extract"]
        preds = [
            _pred_row(self.SCOPE, f"s{i}", f"p{i}",
                      ceiling=E, always_carry=["clarify"], carry=["web_extract"],
                      active=E, ts=ANCHOR - age * DAY)
            for i, age in enumerate(ages_days)
        ]
        return preds, []

    def _porcelain_flag(self, recs, floor):
        shape_ceiling = _load_shape_ceiling()
        doc = shape_ceiling.build_porcelain(
            {self.SCOPE: recs},
            Path("/nonexistent-state-dir"),
            {"window_days": 7, "promote_min_sessions": 1,
             "promote_min_calls": 2, "demote_min_sessions_no_use": floor},
            dry_run=True, changed=False,
        )
        return doc["scopes"][0]["demote_skipped_insufficient_sessions"]

    def test_one_session_in_window_below_the_floor_never_demotes(self):
        # Two sessions, but only the newest is inside the 7-day window.
        preds, calls = self._history(ages_days=(0, 10))
        recs = _compute(self.SCOPE, preds, calls, window_days=7,
                        demote_min_sessions_no_use=2)
        self.assertEqual(recs["sessions_considered"], 1)
        self.assertEqual(recs["demote"], [],
                         "1 session inside the window is below a floor of 2")
        self.assertIs(self._porcelain_flag(recs, floor=2), True,
                      "the porcelain says the demote arm was skipped for want "
                      "of sessions, not that it ran and found nothing")

    def test_two_sessions_in_window_meet_the_floor_and_demotion_fires(self):
        preds, calls = self._history(ages_days=(0, 3))
        recs = _compute(self.SCOPE, preds, calls, window_days=7,
                        demote_min_sessions_no_use=2)
        self.assertEqual(recs["sessions_considered"], 2)
        self.assertIn("web_extract", {d["tool"] for d in recs["demote"]},
                      "at the floor, an unused carry resident demotes")
        self.assertIs(self._porcelain_flag(recs, floor=2), False)


if __name__ == "__main__":
    unittest.main()
