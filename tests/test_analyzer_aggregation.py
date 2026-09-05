"""Regression tests for analyzer aggregation + replay crash bugs.

Each test here locks one output-correctness fix and fails on the pre-fix
code:

  1. ``matched_counterfactual`` crashed with ``TypeError`` on an explicit
     JSON ``null`` in ``cache_read_tokens`` / ``input_tokens`` /
     ``api_call_idx``.
  2. ``stability_simulation`` raised ``KeyError`` on prediction rows lacking
     ``prediction_id`` — rows the module documents as tolerated.
  3. ``summary_payload`` double-counted sessions observed in more than one
     scope.
  4. ``harvest_recommendation_rows`` emitted a scope-naming learned patch
     that proposed nothing for ``keep_expand_only`` rows.
  5. ``expand_only_rate`` divided a call count by a prediction count and
     could exceed 100%.
  6. The summary payload and the recommendations JSON stamped different
     schema versions for one run.
  7. The trigger-keyword report section never rendered the degraded-mode
     callout its ``preset_triggers_status`` field carries.
  8. The keyword miner used raw-substring containment, so an unrelated
     n-gram could suppress a real candidate.
  9. ``--format json`` omitted the cache-cost data the markdown report shows,
     and unrecognized ``source`` values were silently reclassified instead of
     bucketed as "other".
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

if "tool_belt_plugin" not in sys.modules:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent.parent))
    sys.path.insert(0, str(here.parent))
    from tests import conftest  # noqa: F401 — side-effect: register package

analyze = importlib.import_module("tool_belt_plugin.analyze")

PLUGIN_DIR = Path(__file__).resolve().parent.parent


def _args(*extra: str):
    """Real analyzer defaults, so the tests track the shipped thresholds."""
    with mock.patch.object(sys, "argv", ["analyze.py", *extra]):
        return analyze.parse_args()


class ReplayNullToleranceTests(unittest.TestCase):
    """Fix 1 + 2 — crash bugs in cache_replay.py."""

    def setUp(self):
        self.replay = importlib.import_module("tool_belt_plugin.cache_replay")

    def test_matched_counterfactual_tolerates_explicit_nulls(self):
        # A provider that reports no cache usage writes an explicit null;
        # ``int(row.get(k, 0))`` raised TypeError on exactly these rows.
        calls = [
            {"scope": "a:telegram", "hermes_session_id": "s1", "ts": 1,
             "api_call_idx": 0, "tool_list_hash": "h1", "model": "m",
             "cache_read_tokens": 100, "input_tokens": 500},
            {"scope": "a:telegram", "hermes_session_id": "s1", "ts": 2,
             "api_call_idx": 1, "tool_list_hash": "h1", "model": "m",
             "cache_read_tokens": None, "input_tokens": None},
            {"scope": "a:telegram", "hermes_session_id": "s1", "ts": 3,
             "api_call_idx": None, "tool_list_hash": "h2", "model": "m",
             "cache_read_tokens": None, "input_tokens": 400},
        ]
        result = self.replay.matched_counterfactual(calls, scope_filter="a:telegram")
        self.assertEqual(result["total_calls"], 3)

    def test_stability_simulation_tolerates_rows_without_prediction_id(self):
        preds = [
            {"scope": "a:telegram", "hermes_session_id": "s1", "ts": 1,
             "prediction_id": "p1"},
            # Older/malformed row: no prediction_id at all.
            {"scope": "a:telegram", "hermes_session_id": "s1", "ts": 2},
            {"scope": "a:telegram", "hermes_session_id": "s1", "ts": 3,
             "prediction_id": ""},
        ]
        calls = [
            {"scope": "a:telegram", "hermes_session_id": "s1", "ts": 1,
             "api_call_idx": 0, "tool_list_hash": "h1", "prediction_id": "p1"},
            {"scope": "a:telegram", "hermes_session_id": "s1", "ts": 2,
             "api_call_idx": 1, "tool_list_hash": "h1", "prediction_id": "p1"},
        ]
        result = self.replay.stability_simulation(preds, calls, tool_calls=[],
                                               scope_filter="a:telegram")
        self.assertEqual(result["matches_stable"], 1)


class SessionUnionTests(unittest.TestCase):
    """Fix 3 — one session seen in two scopes must count once."""

    def test_sessions_are_unioned_across_scopes(self):
        stats = {}
        for scope in ("a:telegram", "a:cron"):
            stat = analyze.ScopeStats(scope=scope)
            stat.predictions = 2
            stat.sessions.add("shared-session")
            stat.sessions_with_mutation.add("shared-session")
            stats[scope] = stat
        totals = analyze.summary_payload(stats, [], _args())["totals"]
        self.assertEqual(totals["sessions_observed"], 1)
        self.assertEqual(totals["sessions_with_mutation"], 1)


class HarvestRowTests(unittest.TestCase):
    """Fixes 4 + 5 — learned patch shape and a rate that is a real rate."""

    def _stats(self, predictions: int, expand_only_predictions: int,
               calls_each: int):
        """Build stats the real way — through collect_stats on harvest rows."""
        pred_rows = [
            {"scope": "a:telegram", "prediction_id": f"p{i}", "ts": i,
             "hermes_session_id": "s1", "policy_source": "harvest",
             "message_preview": "do the thing"}
            for i in range(predictions)
        ]
        call_rows = [
            {"scope": "a:telegram", "prediction_id": f"p{i}",
             "tool_name": "shell", "was_expand_only": True}
            for i in range(expand_only_predictions)
            for _ in range(calls_each)
        ]
        return analyze.collect_stats(pred_rows, call_rows)

    def _stats_with_calls(self, calls_per_prediction: int, predictions: int):
        return self._stats(predictions, predictions, calls_per_prediction)

    def test_expand_only_rate_never_exceeds_one(self):
        # 3 was_expand_only calls in each of 4 harvest predictions: the old
        # call/prediction quotient reported 300%.
        stats = self._stats_with_calls(calls_per_prediction=3, predictions=4)
        rows = analyze.harvest_recommendation_rows(stats, _args(), set())
        self.assertTrue(rows)
        metrics = rows[0]["metrics"]
        self.assertLessEqual(metrics["expand_only_rate"], 1.0)
        self.assertEqual(metrics["expand_only_rate"], 1.0)
        self.assertEqual(metrics["expand_only_calls_per_prediction"], 3.0)

    def test_keep_expand_only_emits_an_empty_patch(self):
        # High per-tool carry cost, low call volume → keep_expand_only, which
        # proposes no state change and so must not name a scope.
        stats = self._stats(predictions=400, expand_only_predictions=3,
                            calls_each=1)
        rows = analyze.harvest_recommendation_rows(stats, _args(), set())
        self.assertTrue(rows)
        row = rows[0]
        self.assertEqual(row["action"], "keep_expand_only")
        self.assertEqual(row["proposed_learned_patch"], {"scopes": {}})

    def test_promotion_still_emits_a_carry_patch(self):
        stats = self._stats_with_calls(calls_per_prediction=3, predictions=4)
        rows = analyze.harvest_recommendation_rows(stats, _args(), set())
        row = rows[0]
        self.assertEqual(row["action"], "promote_to_carry")
        self.assertEqual(
            row["proposed_learned_patch"],
            {"scopes": {"a:telegram": {"carry": ["shell"]}}},
        )


class KeywordMinerContainmentTests(unittest.TestCase):
    """Fix 8 — containment must be word-wise, not raw substring."""

    def test_unrelated_ngram_is_not_treated_as_contained(self):
        # "read the notes" is a raw substring of "sp[read the notes]" but is
        # not contained in it word-wise. Pre-fix, the higher-precision
        # "spread the notes" suppressed "read the notes" outright.
        previews = ["read the notes and spread the notes"] * 4
        noise = ["please read the notes later"] * 2
        out = analyze._suggest_keywords_for_expand_only_tool(
            expand_only_previews=previews,
            noise_previews=noise,
            existing_patterns=[],
            min_n=3, max_n=3, min_support=3, min_precision=0.5,
            max_candidates=10,
        )
        patterns = {row["pattern"] for row in out}
        self.assertIn("spread the notes", patterns)
        self.assertIn("read the notes", patterns)

    def test_word_wise_containment_still_dedupes(self):
        # The legitimate case the dedupe exists for: the longer form covers
        # the shorter one exactly, so only the longer survives.
        previews = ["open the page"] * 4
        out = analyze._suggest_keywords_for_expand_only_tool(
            expand_only_previews=previews,
            noise_previews=[],
            existing_patterns=[],
            min_n=2, max_n=3, min_support=3, min_precision=0.5,
            max_candidates=10,
        )
        patterns = {row["pattern"] for row in out}
        self.assertIn("open the page", patterns)
        self.assertNotIn("open the", patterns)

    def test_emitted_candidates_are_never_ones_the_dedupe_dropped(self):
        # Parity with _suggest_dampeners_for_trigger's final re-filter.
        previews = ["open the page now"] * 4
        out = analyze._suggest_keywords_for_expand_only_tool(
            expand_only_previews=previews,
            noise_previews=[],
            existing_patterns=[],
            min_n=2, max_n=4, min_support=3, min_precision=0.5,
            max_candidates=20,
        )
        patterns = [row["pattern"] for row in out]
        for shorter in patterns:
            for longer in patterns:
                if analyze._ngram_contains(longer, shorter):
                    self.fail(f"emitted both {shorter!r} and {longer!r}")


class CliOutputParityTests(unittest.TestCase):
    """Fix 9 — JSON output carries cache cost and an honest source bucket."""

    def _write(self, path: Path, rows: list[dict]) -> None:
        path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def test_json_format_includes_cache_cost_and_other_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state" / "tool-belt"
            state.mkdir(parents=True)
            self._write(state / "predictions.jsonl", [
                {"scope": "a:telegram", "prediction_id": "p1", "ts": 1,
                 "hermes_session_id": "s1", "tool_list_hash": "h1"},
                {"scope": "a:telegram", "prediction_id": "p2", "ts": 2,
                 "hermes_session_id": "s1", "tool_list_hash": "h2"},
            ])
            self._write(state / "tool_calls.jsonl", [
                {"scope": "a:telegram", "prediction_id": "p1",
                 "tool_name": "shell", "source": "gateway"},
                # A source value no current writer emits.
                {"scope": "a:telegram", "prediction_id": "p2",
                 "tool_name": "shell", "source": "workflow"},
            ])
            self._write(state / "api_calls.jsonl", [
                {"scope": "a:telegram", "hermes_session_id": "s1", "ts": 1,
                 "api_call_idx": 0, "tool_list_hash": "h1", "model": "m",
                 "prediction_id": "p1", "cache_read_tokens": 1000,
                 "input_tokens": 2000},
                {"scope": "a:telegram", "hermes_session_id": "s1", "ts": 2,
                 "api_call_idx": 1, "tool_list_hash": "h2", "model": "m",
                 "prediction_id": "p2", "cache_read_tokens": 0,
                 "input_tokens": 2000},
            ])
            proc = subprocess.run(
                [sys.executable, str(PLUGIN_DIR / "analyze.py"),
                 "--state-dir", str(state), "--format", "json", "--no-report"],
                capture_output=True, text=True, check=True,
            )
        payload = json.loads(proc.stdout)
        self.assertIn("cache_cost", payload)
        self.assertIn("a:telegram", payload["cache_cost"]["scopes"])
        self.assertIn("cache_read_lost_upper_bound",
                      payload["cache_cost"]["totals"])
        counts = payload["totals"]["tool_call_source_counts"]
        self.assertEqual(counts["gateway"], 1)
        self.assertEqual(counts["other"], 1)
        self.assertEqual(counts["subagent"], 0)


if __name__ == "__main__":
    unittest.main()
