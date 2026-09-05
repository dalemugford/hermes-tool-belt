"""Every shaping entrypoint must price an expansion at the SAME measured cost.

The demote/promote economics turn on ``expand_round_trip_tokens`` — the per-
event cost of one on-demand ``expand_tools`` round-trip. Production
``auto_shape_run`` measures it per scope (``savings.measure_expand_overhead``);
if the manual ``shape-ceiling`` CLI or the interactive ``configure`` review
instead used the flat :data:`EXPAND_ROUND_TRIP_TOKENS` fallback (26× cheaper on
a long uncached session), they would recommend demotions the production engine
itself rejects — a review that lies about how aggressive shaping really is.

These tests seed one non-caching scope whose measured per-event is a fixed,
non-fallback value and assert all three entrypoints agree on it. They FAIL on a
tree where either script omits ``expand_round_trip_tokens`` from its
``compute_scope_recommendations`` call (the pre-fix state).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import conftest  # noqa: F401,E402 — registers the tool_belt_plugin package

shaping = importlib.import_module("tool_belt_plugin.shaping")
from tool_belt_plugin.shaping import EXPAND_ROUND_TRIP_TOKENS  # noqa: E402

SCOPE = "agent-x:telegram"
#: The fixed measured per-event: median of the expand meta-call input_tokens
#: across the 6 non-caching expand predictions below. Deliberately far from the
#: 1,500 fallback so a fallback regression is unmistakable.
MEASURED = 20_000
N_SESSIONS = 22          # comfortably ≥ demote_min_sessions_no_use so demotion runs
N_EXPAND = 6             # ≥ measure_expand_overhead's min (5) → a MEASURED cost


def _seed(state_dir: Path) -> None:
    """One non-caching (ollama) scope: an always-unused carried tool that must
    demote, plus 6 expand events fixing the measured per-event at MEASURED."""
    state_dir.mkdir(parents=True, exist_ok=True)
    preds, apis, tcs = [], [], []
    for i in range(N_SESSIONS):
        pid = f"p{i}"
        preds.append({
            "schema_version": 2, "ts": 1000 + i, "scope": SCOPE,
            "session_id": f"{SCOPE}:s{i}", "hermes_session_id": f"s{i}",
            "prediction_id": pid,
            # fat_tool is an adaptive carry resident, never called → a
            # carry_unused demote under any per-event cost.
            "residency_inferred": True,
            "carry_tools": ["fat_tool"],
            "ceiling_tools": ["fat_tool"],
            "active_tools": ["fat_tool"],
        })
        expanding = i < N_EXPAND
        apis.append({
            "ts": 1000 + i, "prediction_id": pid, "scope": SCOPE,
            "provider": "ollama-cloud",   # never caches → non-caching cohort
            "cache_mode": "off", "cache_read_tokens": 0,
            "input_tokens": MEASURED if expanding else 5000,
            "prompt_tokens": MEASURED if expanding else 5000,
            "output_tokens": 10 if expanding else 200,
        })
        if expanding:
            tcs.append({
                "schema_version": 2, "ts": 1000 + i + 0.5, "prediction_id": pid,
                "scope": SCOPE, "tool_name": "expand_tools", "source": "gateway",
            })
    (state_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in preds), encoding="utf-8")
    (state_dir / "api_calls.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in apis), encoding="utf-8")
    (state_dir / "tool_calls.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in tcs), encoding="utf-8")


def _load_shape_ceiling():
    path = _TESTS_DIR.parent / "scripts" / "shape-ceiling.py"
    spec = importlib.util.spec_from_file_location("tool_belt_shape_ceiling", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tool_belt_shape_ceiling"] = module
    spec.loader.exec_module(module)
    return module


def _load_configure():
    path = _TESTS_DIR.parent / "scripts" / "configure.py"
    spec = importlib.util.spec_from_file_location("tool_belt_configure", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tool_belt_configure"] = module
    spec.loader.exec_module(module)
    return module


class ExpandCostConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name) / "state" / "tool-belt"
        _seed(self.state)

    def test_measured_penalty_is_the_measured_value_not_the_fallback(self):
        preds = shaping.load_jsonl(self.state / "predictions.jsonl")
        apis = shaping.load_jsonl(self.state / "api_calls.jsonl")
        tcs = shaping.load_jsonl(self.state / "tool_calls.jsonl")
        penalty = shaping.measured_expand_penalty(preds, apis, tcs, SCOPE)
        self.assertEqual(penalty, MEASURED)
        self.assertNotEqual(penalty, EXPAND_ROUND_TRIP_TOKENS)

    def test_configure_review_prices_at_the_measured_cost(self):
        configure = _load_configure()
        info = configure.ScopeInfo(
            scope=SCOPE, agent="agent-x", platform="telegram",
            state_dir=self.state, sessions=N_SESSIONS)
        recs = configure.compute_recommendations(info, configure.shape_thresholds())
        self.assertIsNotNone(recs)
        self.assertEqual(recs["expand_round_trip_tokens"], MEASURED)
        # And the demote it would show is priced on that cost, not the fallback.
        self.assertIn("fat_tool", [d["tool"] for d in recs["demote"]])

    def test_shape_ceiling_cli_prices_at_the_measured_cost(self):
        module = _load_shape_ceiling()
        out = self.state.parent / "porcelain.json"
        argv = ["shape-ceiling", "--state-dir", str(self.state),
                "--dry-run", "--json-file", str(out)]
        with mock.patch.object(sys, "argv", argv):
            module.main()
        doc = json.loads(out.read_text(encoding="utf-8"))
        scope_row = next(s for s in doc["scopes"] if s["scope"] == SCOPE)
        self.assertEqual(scope_row["expand_round_trip_tokens"], MEASURED)

    def test_auto_shape_applies_the_same_measured_cost(self):
        config = {
            "enabled": True, "auto_shape": True, "learned_mode": "recommend",
            "channels": {SCOPE: {"learned_mode": "apply"}},
        }
        summary = shaping.auto_shape_run(config, state_dir=self.state)
        self.assertTrue(summary.get("ran"))
        doc = json.loads((self.state / "learned.json").read_text(encoding="utf-8"))
        meta = doc["scopes"][SCOPE]["shaping"]
        self.assertEqual(meta["expand_round_trip_tokens"], MEASURED)
        # The three entrypoints therefore agree, and none used the fallback.
        self.assertNotEqual(meta["expand_round_trip_tokens"], EXPAND_ROUND_TRIP_TOKENS)


if __name__ == "__main__":
    unittest.main()
