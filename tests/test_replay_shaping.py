"""``scripts/replay-shaping.py`` — the operator replay harness.

The replay is the tool that priced the aggressive-shaping defaults, so it has
to keep two properties nothing else in the suite protects:

  · it is READ-ONLY. It drives ``apply_recommendations`` (the real merge) over
    an in-memory learned state; if it ever grew a ``write_state`` call — or
    reached the merge helper that writes one — it would rewrite a live
    operator's loadout from a what-if run. This asserts the state dir is
    byte-for-byte untouched and that no ``learned.json`` appears.
  · it reads schema sizes through ``logger_io.load_schema_sizes``. The
    on-disk document nests the per-tool map under a ``tools`` key; a raw
    ``json.load`` of the file (a mistake made once already) yields an empty
    map and every tool silently prices at the fallback size.

Plus the two behavioural locks the report's numbers rest on: one result row
per (window_days × floor) combo, and "implied expand events" counting only
PRIMARY dispatches (tool-call rows carrying a ``tool_call_id``) — nested
sandbox/MCP/memory fan-out rows never faced narrowing and would inflate the
count many times over.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import conftest  # noqa: F401,E402 — registers the tool_belt_plugin package

_PLUGIN_DIR = _TESTS_DIR.parent
DAY = 86400.0
#: A fixed anchor in the past — the shaper's window is data-relative, so the
#: replay must reach the same verdict on archived telemetry as on fresh.
ANCHOR = 1_000_000_000.0
SCOPE = "agent-a:telegram"


def _load_replay():
    spec = importlib.util.spec_from_file_location(
        "tool_belt_replay_shaping_test", _PLUGIN_DIR / "scripts" / "replay-shaping.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


replay = _load_replay()


def _seed(state_dir: Path, *, reach_row: str | None = None, sessions: int = 4) -> None:
    """A tiny scope: ``terminal`` is used every session, ``web_extract`` is a
    carried resident nobody calls (so it demotes), one session per day.

    ``reach_row`` optionally adds a ``web_extract`` call in the LAST session:
    ``"primary"`` writes a real model dispatch (with ``tool_call_id``),
    ``"nested"`` writes a secondary row without one.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    preds, calls, api_calls = [], [], []
    for i in range(sessions):
        pid = f"pred-{i}"
        ts = ANCHOR - (sessions - 1 - i) * DAY
        preds.append({
            "ts": ts,
            "schema_version": 2,
            "scope": SCOPE,
            "session_id": "key",
            "hermes_session_id": f"sess-{i}",
            "prediction_id": pid,
            "ceiling_tools": ["clarify", "terminal", "web_extract"],
            "always_carry_tools": ["clarify"],
            "carry_tools": ["terminal", "web_extract"],
            "active_tools": ["clarify", "terminal", "web_extract"],
            "expand_only_tools": [],
            "ceiling_tokens": 3000,
        })
        calls.append({
            "ts": ts, "schema_version": 2, "prediction_id": pid,
            "tool_name": "terminal", "tool_call_id": f"call-{i}",
            "was_initially_active": True,
        })
        api_calls.append({"ts": ts, "schema_version": 2, "prediction_id": pid,
                          "scope": SCOPE})
    if reach_row is not None:
        last = f"pred-{sessions - 1}"
        row = {
            "ts": ANCHOR, "schema_version": 2, "prediction_id": last,
            "tool_name": "web_extract", "was_initially_active": True,
        }
        if reach_row == "primary":
            row["tool_call_id"] = "call-reach"
        calls.append(row)
    (state_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in preds), encoding="utf-8")
    (state_dir / "tool_calls.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in calls), encoding="utf-8")
    (state_dir / "api_calls.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in api_calls), encoding="utf-8")
    # The per-tool sizes live under a "tools" key — the shape load_schema_sizes
    # understands and a raw json.load does not.
    (state_dir / "schema_sizes.json").write_text(json.dumps({
        "updated_at": "2001-09-09T01:46:40Z",
        "tools": {"terminal": 900, "web_extract": 700, "clarify": 120},
    }), encoding="utf-8")


class ReplayHarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state_dir = Path(self.tmp.name) / "tool-belt"

    def _snapshot(self) -> dict[str, bytes]:
        return {str(p.relative_to(self.state_dir)): p.read_bytes()
                for p in sorted(self.state_dir.rglob("*")) if p.is_file()}

    def _replay(self, **overrides):
        kwargs = dict(
            scope_filter=SCOPE, window_days=[7], floors=[2],
            promote_min_sessions=1, promote_min_calls=2, demote_k=1.5,
            plugin_config={},
        )
        kwargs.update(overrides)
        return replay.replay_state_dir(self.state_dir, **kwargs)

    def test_replay_produces_a_result_row_per_combo(self):
        _seed(self.state_dir)
        doc = self._replay(window_days=[7, 30], floors=[2, 3])
        self.assertEqual(len(doc["scopes"]), 1)
        entry = doc["scopes"][0]
        self.assertEqual(entry["scope"], SCOPE)
        self.assertEqual(entry["sessions"], 4)
        self.assertEqual(
            [(c["window_days"], c["floor"]) for c in entry["combos"]],
            [(7, 2), (7, 3), (30, 2), (30, 3)],
            "one run per window_days x floor pair, windows outermost")
        for combo in entry["combos"]:
            for key in ("sessions", "first_demotion_session", "converged_session",
                        "final_demoted", "steady_tokens_per_turn", "carried_tokens",
                        "implied_expand_events", "promotes", "flap", "curve",
                        "final_expand_only", "reached_while_demoted"):
                self.assertIn(key, combo)
            self.assertEqual(combo["sessions"], 4)
            self.assertEqual(len(combo["curve"]), 4,
                             "the convergence curve has one point per session")

    def test_replay_actually_shapes_the_fixture(self):
        # Precondition for every number above: the replay must reach a
        # demotion at all, otherwise the harness could be reporting zeros
        # from a fixture it never shaped.
        _seed(self.state_dir)
        combo = self._replay()["scopes"][0]["combos"][0]
        self.assertIn("web_extract", combo["final_expand_only"],
                      "an unused carried resident demotes over the replay")
        self.assertNotIn("terminal", combo["final_expand_only"],
                         "a tool used every session stays carried")
        self.assertIsNotNone(combo["first_demotion_session"])

    def test_replay_never_writes_learned_state(self):
        _seed(self.state_dir)
        before = self._snapshot()
        self._replay(window_days=[7, 30], floors=[2])
        self.assertEqual(self._snapshot(), before,
                         "a replay is read-only: not one byte of the state "
                         "dir may change")
        self.assertFalse((self.state_dir / "learned.json").exists(),
                         "the replay's learned state lives in memory only")

    def test_schema_sizes_come_through_logger_io(self):
        _seed(self.state_dir)
        sizes = replay.load_scope_inputs(self.state_dir)["schema_sizes"]
        self.assertEqual(sizes, {"terminal": 900, "web_extract": 700, "clarify": 120},
                         "sizes are nested under a 'tools' key; reading the "
                         "file raw yields {} and prices every tool at the "
                         "fallback size")

    def test_only_primary_dispatches_count_as_implied_expands(self):
        # Same call, twice: once as a real model dispatch, once as a nested
        # secondary row. Only the first is an expansion the operator pays for.
        with tempfile.TemporaryDirectory() as tmp:
            nested_dir = Path(tmp) / "tool-belt"
            _seed(nested_dir, reach_row="nested")
            nested = replay.replay_state_dir(
                nested_dir, scope_filter=SCOPE, window_days=[7], floors=[2],
                promote_min_sessions=1, promote_min_calls=2, demote_k=1.5,
                plugin_config={})
        self.assertEqual(nested["scopes"][0]["combos"][0]["implied_expand_events"], 0,
                         "a nested/secondary tool-call row never faced "
                         "narrowing and is not an implied expansion")

        _seed(self.state_dir, reach_row="primary")
        primary = self._replay()["scopes"][0]["combos"][0]
        self.assertEqual(primary["implied_expand_events"], 1)
        self.assertEqual(primary["reached_while_demoted"], {"web_extract": 1})

    def test_no_matching_scope_yields_no_rows(self):
        _seed(self.state_dir)
        doc = self._replay(scope_filter="nobody:nowhere")
        self.assertEqual(doc["scopes"], [],
                         "an unmatched filter reports nothing rather than "
                         "silently replaying every scope")


if __name__ == "__main__":
    unittest.main()
