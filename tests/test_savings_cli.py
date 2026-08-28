"""Canonical savings engine + `tool-belt savings` CLI coverage.

Maps 1:1 to the Phase 7B required-tests list. Every fixture writes into a
throwaway HERMES_HOME; nothing here touches live Hermes state.
"""
from __future__ import annotations

import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

PLUGIN_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = PLUGIN_DIR / "tests"
sys.path.insert(0, str(TESTS_DIR))
import conftest  # noqa: F401,E402 — registers tool_belt_plugin

savings = importlib.import_module("tool_belt_plugin.savings")
savings_cli = importlib.import_module("tool_belt_plugin.savings_cli")
learned = importlib.import_module("tool_belt_plugin.learned")


# ─── fixtures ─────────────────────────────────────────────────────────────────


def _full_def(name: str) -> dict:
    """A COMPLETE OpenAI-shape tool definition with description + parameters."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                f"The {name} tool. It performs the {name} operation against the "
                f"user's environment and returns a structured result describing "
                f"what changed, including any diagnostics the caller may need."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": f"What {name} acts on."},
                    "options": {
                        "type": "object",
                        "description": "Free-form options bag.",
                        "properties": {
                            "verbose": {"type": "boolean", "description": "Chatty output."},
                            "dry_run": {"type": "boolean", "description": "Preview only."},
                        },
                    },
                },
                "required": ["target"],
            },
        },
    }


def _names_only(name: str) -> dict:
    return {"name": name}


def _write_session(sessions_dir: Path, stem: str, *, platform: str, model: str,
                   ceiling: list[str], turns: list[dict], api_mode: str = "",
                   full_defs: bool = True) -> Path:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    tool_entries = [_full_def(n) if full_defs else _names_only(n) for n in ceiling]
    meta = {"role": "session_meta", "platform": platform, "model": model, "tools": tool_entries}
    if api_mode:
        meta["api_mode"] = api_mode
    rows = [meta]
    for turn in turns:
        rows.append({"role": "user", "content": turn["user"]})
        calls = turn.get("calls") or []
        if calls:
            rows.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": c}} for c in calls],
            })
    path = sessions_dir / f"{stem}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _write_observed(state_dir: Path, scope: str, *, expand_events: int = 0) -> None:
    """Write a minimal observed-telemetry triple for a scope."""
    state_dir.mkdir(parents=True, exist_ok=True)
    preds = []
    apis = []
    # cache-on narrowed row
    preds.append({
        "schema_version": 2, "ts": 1780000000.0, "prediction_id": "on1",
        "session_id": f"{scope}:S1", "scope": scope, "policy_source": "preset",
        "ceiling_count": 40, "narrowed_count": 20,
        "ceiling_tokens": 10000, "narrowed_tokens": 4000, "frozen_reuse": False,
    })
    apis.append({"ts": 1780000000.0, "prediction_id": "on1", "scope": scope,
                 "api_call_idx": 0, "cache_mode": "on", "input_tokens": 4000,
                 "cache_read_tokens": 6000, "cache_write_tokens": 0})
    # cache-off narrowed row
    preds.append({
        "schema_version": 2, "ts": 1780000001.0, "prediction_id": "off1",
        "session_id": f"{scope}:S2", "scope": scope, "policy_source": "preset",
        "ceiling_count": 40, "narrowed_count": 22,
        "ceiling_tokens": 10000, "narrowed_tokens": 5000, "frozen_reuse": False,
    })
    apis.append({"ts": 1780000001.0, "prediction_id": "off1", "scope": scope,
                 "api_call_idx": 0, "cache_mode": "off", "input_tokens": 5000,
                 "cache_read_tokens": 0, "cache_write_tokens": 0})
    tcs = []
    for i in range(expand_events):
        tcs.append({"schema_version": 2, "ts": 1780000002.0, "prediction_id": "off1",
                    "session_id": f"{scope}:S2", "scope": scope,
                    "tool_name": "expand_tools", "source": "gateway"})
    (state_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in preds), encoding="utf-8")
    (state_dir / "api_calls.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in apis), encoding="utf-8")
    (state_dir / "tool_calls.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in tcs), encoding="utf-8")


def _write_route_evidence(state_dir: Path, session_stem: str, calls: list[dict],
                          *, scope: str = "default:telegram",
                          base_ts: float = 1780000000.0) -> None:
    """Seed the predictions→api_calls bridge for one session file.

    Mirrors what the plugin actually writes: ``predictions.jsonl`` rows carry
    ``hermes_session_id`` (the session file stem) and ``api_calls.jsonl`` rows
    carry ``provider``/``api_mode``/``input_tokens`` — the fields live
    ``session_meta`` records never contain.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    preds, apis = [], []
    for i, call in enumerate(calls):
        pid = f"{session_stem}-p{i}"
        preds.append({
            "schema_version": 2, "ts": base_ts + i, "prediction_id": pid,
            "hermes_session_id": session_stem, "session_id": f"chat-{session_stem}",
            "scope": scope, "policy_source": "preset",
            "ceiling_count": 40, "narrowed_count": 20,
            "ceiling_tokens": 10000, "narrowed_tokens": 4000, "frozen_reuse": False,
        })
        apis.append({
            "ts": base_ts + i, "prediction_id": pid, "session_id": f"chat-{session_stem}",
            "scope": scope, "api_call_idx": 0, "cache_mode": "on",
            "model": call.get("model", "claude-sonnet-4-6"),
            "provider": call.get("provider", ""),
            "api_mode": call.get("api_mode", ""),
            "input_tokens": int(call.get("input_tokens") or 0),
            "cache_read_tokens": 0, "cache_write_tokens": 0,
        })
    (state_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in preds), encoding="utf-8")
    (state_dir / "api_calls.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in apis), encoding="utf-8")


# A ceiling mixing residents, trigger-activatable, and pure expand_only tools.
CEILING = [
    "clarify", "todo", "send_message", "expand_tools",       # always_carry
    "read_file", "web_search", "terminal", "write_file",     # carry residents
    "delegate_task",       # trigger: delegation
    "execute_code",        # trigger: code_execution (also reachable bare)
    "vision_analyze",      # trigger: vision
    "image_generate",      # trigger: image_gen
    "session_search",      # trigger: history_search
    "cronjob",             # trigger: cronjob
]


class _HomeCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / "hermes home"
        self.home.mkdir()
        self._prev_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(self.home)
        learned._CACHE.update({"path": None, "mtime_ns": None, "state": None, "hash": ""})
        self.addCleanup(self._restore_home)

    def _restore_home(self):
        if self._prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self._prev_home
        learned._CACHE.update({"path": None, "mtime_ns": None, "state": None, "hash": ""})


# ══════════════════════════════════════════════════════════════════════════════


class DefaultAllAgentsTests(_HomeCase):
    """(1) `tool-belt savings` defaults to all enabled/discovered agents and
    emits aggregate + per-agent results."""

    def test_all_agents_and_aggregate(self):
        _write_observed(self.home / "state" / "tool-belt", "default:telegram")
        _write_session(self.home / "sessions", "s1", platform="telegram",
                       model="claude-sonnet-4-6", ceiling=CEILING,
                       turns=[{"user": "hello there"}])
        named = self.home / "profiles" / "assistant-a"
        _write_observed(named / "state" / "tool-belt", "assistant-a:slack")
        _write_session(named / "sessions", "s2", platform="slack",
                       model="claude-sonnet-4-6", ceiling=CEILING,
                       turns=[{"user": "hi"}])

        report = savings.compute(hermes_home=self.home)
        agents = {a.agent for a in report.agents}
        self.assertEqual(agents, {"default", "assistant-a"})
        payload = report.to_json()
        self.assertEqual(payload["generated_for"], "all")
        self.assertIn("aggregate", payload)
        self.assertEqual(len(payload["agents"]), 2)


class AgentSelectorTests(_HomeCase):
    """(2) --agent=default includes all default platforms and excludes other
    agents; unknown/disabled agent errors non-zero."""

    def _seed_two_agents(self):
        _write_observed(self.home / "state" / "tool-belt", "default:telegram")
        _write_session(self.home / "sessions", "t", platform="telegram",
                       model="m", ceiling=CEILING, turns=[{"user": "hey"}])
        _write_session(self.home / "sessions", "c", platform="cli",
                       model="m", ceiling=CEILING, turns=[{"user": "hey"}])
        named = self.home / "profiles" / "assistant-a"
        _write_observed(named / "state" / "tool-belt", "assistant-a:slack")

    def test_agent_scopes_all_platforms_excludes_others(self):
        self._seed_two_agents()
        report = savings.compute(agent="default", hermes_home=self.home)
        self.assertEqual([a.agent for a in report.agents], ["default"])
        # default has two platforms (telegram from observed, cli from sessions).
        plats = set(report.agents[0].platforms)
        self.assertIn("default:telegram", plats)
        self.assertIn("default:cli", plats)
        # No assistant-a scope leaks into the default report.
        self.assertFalse(any(p.startswith("assistant-a") for p in plats))

    def test_unknown_agent_errors_nonzero(self):
        self._seed_two_agents()
        with self.assertRaises(savings.UnknownAgentError):
            savings.compute(agent="nope", hermes_home=self.home)
        # CLI surfaces it as a non-zero exit.
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = savings_cli.run(["--agent", "nope", "--hermes-home", str(self.home)])
        self.assertNotEqual(rc, 0)

    def test_stale_explicitly_disabled_profile_is_excluded(self):
        disabled = self.home / "profiles" / "retired"
        _write_session(
            disabled / "sessions", "old", platform="slack", model="m",
            ceiling=CEILING, turns=[{"user": "old telemetry"}],
        )
        disabled.joinpath("config.yaml").write_text(
            "plugins:\n  enabled: []\n  tool-belt:\n    enabled: false\n",
            encoding="utf-8",
        )
        self.assertEqual(savings.discover_agents(self.home), [])
        with self.assertRaises(savings.UnknownAgentError):
            savings.compute(agent="retired", hermes_home=self.home)

    def test_present_profile_without_config_remains_discoverable(self):
        profile = self.home / "profiles" / "portable"
        _write_session(
            profile / "sessions", "s", platform="cli", model="m",
            ceiling=CEILING, turns=[{"user": "hello"}],
        )
        self.assertEqual(
            [loc.agent for loc in savings.discover_agents(self.home)], ["portable"]
        )


class JsonDeterminismTests(_HomeCase):
    """(3) JSON output is deterministic and prose-free."""

    def test_json_is_stable_and_prose_free(self):
        _write_observed(self.home / "state" / "tool-belt", "default:telegram")
        _write_session(self.home / "sessions", "s", platform="telegram",
                       model="claude-sonnet-4-6", ceiling=CEILING,
                       turns=[{"user": "please delegate this task"}])

        def _json_run():
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = savings_cli.run(["--json", "--hermes-home", str(self.home)])
            self.assertEqual(rc, 0)
            return buf.getvalue()

        a = _json_run()
        b = _json_run()
        self.assertEqual(a, b)  # deterministic
        parsed = json.loads(a)  # valid JSON
        self.assertEqual(a.strip()[0], "{")  # no prose preamble
        self.assertEqual(parsed["schema"], "tool-belt/savings/v1")


class FullDefinitionTokenTests(_HomeCase):
    """(4) Complete tool definitions (parameters/descriptions) materially affect
    token counts; names-only placeholders cannot pass."""

    def test_full_defs_dominate_names_only(self):
        full = [_full_def(n) for n in CEILING]
        names = [_names_only(n) for n in CEILING]
        full_tok = savings.schema_tokens(full)
        names_tok = savings.schema_tokens(names)
        # Descriptions + JSON-schema params dominate: full defs are many times
        # larger than names-only. A names-only count could never stand in.
        self.assertGreater(full_tok, names_tok * 5)

    def test_projection_uses_full_defs(self):
        # Same ceiling, one session with full defs and one names-only; the
        # projected gross reduction must be far larger for the full-def session.
        _write_session(self.home / "sessions", "full", platform="telegram",
                       model="m", ceiling=CEILING, turns=[{"user": "hello"}],
                       full_defs=True)
        loc_full = savings.discover_agents(self.home)[0]
        proj_full = savings.compute_projected(loc_full, {"enabled": True})

        # Rewrite the session names-only.
        (self.home / "sessions" / "full.jsonl").unlink()
        _write_session(self.home / "sessions", "names", platform="telegram",
                       model="m", ceiling=CEILING, turns=[{"user": "hello"}],
                       full_defs=False)
        loc_names = savings.discover_agents(self.home)[0]
        proj_names = savings.compute_projected(loc_names, {"enabled": True})

        self.assertGreater(proj_full.gross_schema_token_reduction, 0)
        self.assertEqual(proj_names.sessions_analyzed, 0)
        self.assertEqual(proj_names.confidence, "insufficient")
        self.assertIsNone(proj_names.net_input_reduction_pct)
        self.assertIsNone(proj_names.estimated_usd_savings)

    def test_partial_incomplete_evidence_suppresses_percentage_and_usd(self):
        _write_session(
            self.home / "sessions", "complete", platform="telegram",
            model="claude-sonnet-4-6", api_mode="api_key", ceiling=CEILING,
            turns=[{"user": "hello"}], full_defs=True,
        )
        _write_session(
            self.home / "sessions", "incomplete", platform="telegram",
            model="claude-sonnet-4-6", api_mode="api_key", ceiling=CEILING,
            turns=[{"user": "hello"}], full_defs=False,
        )
        projection = savings.compute_projected(
            savings.discover_agents(self.home)[0], {"enabled": True}
        )
        self.assertEqual(projection.sessions_analyzed, 1)
        self.assertEqual(projection.confidence, "low")
        self.assertIsNone(projection.net_input_reduction_pct)
        self.assertIsNone(projection.estimated_usd_savings)
        self.assertEqual(projection.usd_coverage, "none")


class RenderLabelTests(_HomeCase):
    """Text render shows the provider-only session-input % OR the labeled
    schema-only figure — never a reconstructed-denominator percentage."""

    def _render(self, proj):
        report = savings.SavingsReport(
            generated_for="default", cache_mode="on",
            agents=[savings.AgentSavings(agent="default", platforms=["telegram"],
                                         observed=savings.ObservedCohort(),
                                         projected=proj)],
            hermes_home=str(self.home), token_estimator="chars-div-4")
        return savings_cli.render_text(report)

    def test_provider_basis_shows_session_input_pct(self):
        _write_session(self.home / "sessions", "prov", platform="telegram",
                       model="claude-sonnet-4-6", ceiling=CEILING,
                       turns=[{"user": "hello"}])
        state = self.home / "state" / "tool-belt"
        state.mkdir(parents=True, exist_ok=True)
        (state / "api_calls.jsonl").write_text(
            json.dumps({"ts": 1780000000.0, "session_file": "prov",
                        "input_tokens": 99999}) + "\n", encoding="utf-8")
        proj = savings.compute_projected(
            savings.discover_agents(self.home)[0], {"enabled": True})
        self.assertIsNotNone(proj.net_input_reduction_pct)
        text = self._render(proj)
        self.assertIn("net input reduction:", text)
        self.assertIn("(denominator: provider_reported)", text)
        self.assertNotIn("not the session-input %", text)

    def test_reconstructed_basis_shows_schema_only_label(self):
        _write_session(self.home / "sessions", "recon", platform="telegram",
                       model="m", ceiling=CEILING, turns=[{"user": "hello"}])
        proj = savings.compute_projected(
            savings.discover_agents(self.home)[0], {"enabled": True})
        self.assertIsNone(proj.net_input_reduction_pct)
        text = self._render(proj)
        self.assertIn("schema-only reduction:", text)
        self.assertIn("(not the session-input %)", text)
        self.assertNotIn("net input reduction:", text)


class CohortSeparationTests(_HomeCase):
    """(5) Observed and projected cohorts are labeled separately and never
    summed."""

    def test_cohorts_labeled_and_not_summed(self):
        _write_observed(self.home / "state" / "tool-belt", "default:telegram")
        _write_session(self.home / "sessions", "s", platform="telegram",
                       model="m", ceiling=CEILING, turns=[{"user": "hello"}])
        payload = savings.compute(hermes_home=self.home).to_json()
        agg = payload["aggregate"]
        self.assertEqual(agg["observed"]["label"], "observed")
        self.assertEqual(agg["projected"]["label"], "projected")
        self.assertTrue(agg["projected"]["counterfactual"])
        # No key merges the two cohorts into a single total.
        self.assertNotIn("total_token_reduction", agg)
        self.assertNotIn("combined", agg)
        a0 = payload["agents"][0]
        self.assertEqual(a0["observed"]["label"], "observed")
        self.assertEqual(a0["projected"]["label"], "projected")


class CacheModeReplayTests(_HomeCase):
    """(6) Cache-on frozen/monotonic replay and cache-off per-turn replay are
    distinct."""

    def test_cache_on_and_off_differ(self):
        turns = [
            {"user": "hello there, how are you"},          # residents only
            {"user": "please delegate this task to a subagent"},  # trigger delegation
            {"user": "now search my history for it"},      # trigger history_search
        ]
        _write_session(self.home / "sessions", "s", platform="telegram",
                       model="m", ceiling=CEILING, turns=turns)
        loc = savings.discover_agents(self.home)[0]
        on = savings.compute_projected(loc, {"enabled": True}, cache_mode="on")
        off = savings.compute_projected(loc, {"enabled": True}, cache_mode="off")
        self.assertEqual(on.cache_mode, "on")
        self.assertEqual(off.cache_mode, "off")
        # The frozen (monotonic) active set diverges from per-turn resolution,
        # so gross reductions are not identical.
        self.assertNotEqual(on.gross_schema_token_reduction,
                            off.gross_schema_token_reduction)


class ExpansionVsTriggerTests(_HomeCase):
    """(7) Trigger activation adds no expansion charge; explicit expansion
    does."""

    def test_trigger_free_expansion_charged(self):
        # Trigger path: 'delegate this task' fires the delegation trigger, which
        # activates delegate_task — no expand_tools round trip.
        _write_session(self.home / "sessions", "trig", platform="telegram",
                       model="m", ceiling=CEILING,
                       turns=[{"user": "please delegate this task",
                               "calls": ["delegate_task"]}])
        loc = savings.discover_agents(self.home)[0]
        trig = savings.compute_projected(loc, {"enabled": True}, cache_mode="off")
        self.assertEqual(trig.expansion_events, 0)
        self.assertEqual(trig.estimated_expansion_overhead, 0)

        # Expansion path: a bare conversational message that calls an
        # untriggered expand_only tool — that IS an explicit expansion.
        (self.home / "sessions" / "trig.jsonl").unlink()
        _write_session(self.home / "sessions", "exp", platform="telegram",
                       model="m", ceiling=CEILING,
                       turns=[{"user": "hello there",
                               "calls": ["execute_code"]}])
        loc = savings.discover_agents(self.home)[0]
        exp = savings.compute_projected(loc, {"enabled": True}, cache_mode="off")
        self.assertEqual(exp.expansion_events, 1)
        self.assertEqual(exp.estimated_expansion_overhead,
                         savings.EXPAND_ROUND_TRIP_TOKENS)


class DenominatorTests(_HomeCase):
    """(8) Provider-reported denominator wins; reconstructed cumulative
    denominator is labeled and confidence-downgraded; incomplete denominator
    suppresses percentage."""

    def test_reconstructed_labeled_low_confidence_pct_suppressed(self):
        # Reconstruction omits tool results, system prompt, and per-call
        # accumulation, so its session-input percentage is never shown.
        _write_session(self.home / "sessions", "recon", platform="telegram",
                       model="m", ceiling=CEILING,
                       turns=[{"user": "hello"}, {"user": "and now write a file"}])
        loc = savings.discover_agents(self.home)[0]
        proj = savings.compute_projected(loc, {"enabled": True})
        self.assertEqual(proj.denominator_source, "reconstructed")
        self.assertEqual(proj.confidence, "low")
        self.assertIsNone(proj.net_input_reduction_pct)
        # The schema-only percentage is shown instead, labeled separately.
        self.assertIsNotNone(proj.schema_reduction_pct)

    def test_partial_provider_coverage_suppresses_percentage(self):
        _write_session(self.home / "sessions", "bridged", platform="telegram",
                       model="m", ceiling=CEILING, turns=[{"user": "hello"}])
        _write_session(self.home / "sessions", "unbridged", platform="cli",
                       model="m", ceiling=CEILING, turns=[{"user": "hello"}])
        state = self.home / "state" / "tool-belt"
        state.mkdir(parents=True, exist_ok=True)
        (state / "predictions.jsonl").write_text(
            json.dumps({"prediction_id": "p1",
                        "hermes_session_id": "bridged"}) + "\n",
            encoding="utf-8")
        (state / "api_calls.jsonl").write_text(
            json.dumps({"ts": 1780000000.0, "prediction_id": "p1",
                        "session_id": "agent:main:telegram:dm:1",
                        "input_tokens": 50000}) + "\n",
            encoding="utf-8")
        proj = savings.compute_projected(
            savings.discover_agents(self.home)[0], {"enabled": True})
        self.assertEqual(proj.denominator_source, "partial")
        self.assertEqual(proj.confidence, "low")
        self.assertIsNone(proj.net_input_reduction_pct)

    def test_provider_reported_wins_high_confidence(self):
        path = _write_session(self.home / "sessions", "prov", platform="telegram",
                              model="claude-sonnet-4-6", ceiling=CEILING,
                              turns=[{"user": "hello"}])
        state = self.home / "state" / "tool-belt"
        state.mkdir(parents=True, exist_ok=True)
        (state / "api_calls.jsonl").write_text(
            json.dumps({"ts": 1780000000.0, "session_file": path.stem,
                        "input_tokens": 99999, "cache_read_tokens": 0}) + "\n",
            encoding="utf-8")
        loc = savings.discover_agents(self.home)[0]
        loc = savings.discover_agents(self.home)[0]
        proj = savings.compute_projected(loc, {"enabled": True})
        self.assertEqual(proj.denominator_source, "provider_reported")
        self.assertEqual(proj.input_token_denominator, 99999)
        self.assertEqual(proj.confidence, "high")
        # The provider-basis percentage is the single source of
        # net_input_reduction_pct; assert its exact value.
        self.assertEqual(
            proj.net_input_reduction_pct,
            round(proj.net_token_reduction / 99999 * 100, 2),
        )
        self.assertGreaterEqual(proj.net_input_reduction_pct, 0.0)

    def test_prediction_bridge_joins_chat_key_to_hermes_session(self):
        # Production reality: api_calls rows key on the chat session_id, while
        # historical files are named by Hermes session UUID. The join must go
        # through predictions.jsonl (prediction_id -> hermes_session_id).
        path = _write_session(self.home / "sessions", "20260828_090000_ab12",
                              platform="telegram", model="claude-sonnet-4-6",
                              ceiling=CEILING, turns=[{"user": "hello"}])
        state = self.home / "state" / "tool-belt"
        state.mkdir(parents=True, exist_ok=True)
        (state / "predictions.jsonl").write_text(
            json.dumps({"prediction_id": "p1", "hermes_session_id": path.stem,
                        "session_id": "agent:main:telegram:dm:8499413300"}) + "\n",
            encoding="utf-8")
        (state / "api_calls.jsonl").write_text(
            json.dumps({"ts": 1780000000.0, "prediction_id": "p1",
                        "session_id": "agent:main:telegram:dm:8499413300",
                        "input_tokens": 4242}) + "\n" +
            json.dumps({"ts": 1780000001.0, "prediction_id": "p2",
                        "session_id": "agent:main:telegram:dm:8499413300",
                        "input_tokens": 111}) + "\n",
            encoding="utf-8")
        loc = savings.discover_agents(self.home)[0]
        proj = savings.compute_projected(loc, {"enabled": True})
        # p1 joins via the bridge and lands on the session file stem; p2 has no
        # bridge row and falls back to the chat key (no double counting since
        # the session lookup keys on the stem).
        self.assertEqual(proj.denominator_source, "provider_reported")
        self.assertEqual(proj.input_token_denominator, 4242)
        self.assertEqual(proj.confidence, "high")

    def test_incomplete_schema_suppresses_percentage(self):
        # A session whose meta lists no tools -> incomplete; skipped; percentage
        # suppressed and confidence insufficient.
        sessions = self.home / "sessions"
        sessions.mkdir(parents=True, exist_ok=True)
        meta = {"role": "session_meta", "platform": "telegram", "model": "m", "tools": []}
        (sessions / "empty.jsonl").write_text(
            json.dumps(meta) + "\n" + json.dumps({"role": "user", "content": "hi"}) + "\n",
            encoding="utf-8")
        loc = savings.discover_agents(self.home)[0]
        proj = savings.compute_projected(loc, {"enabled": True})
        self.assertEqual(proj.sessions_analyzed, 0)
        self.assertIsNone(proj.net_input_reduction_pct)
        self.assertEqual(proj.confidence, "insufficient")


class CostClassificationTests(_HomeCase):
    """(9) Known variable cost shows estimated USD/rate basis; subscription and
    unknown routes show no dollars and use net input percentage when
    defensible."""

    def test_known_metered_shows_usd(self):
        _write_session(self.home / "sessions", "k", platform="telegram",
                       model="claude-sonnet-4-6", api_mode="api_key",
                       ceiling=CEILING, turns=[{"user": "hello"}, {"user": "write a file"}])
        loc = savings.discover_agents(self.home)[0]
        proj = savings.compute_projected(loc, {"enabled": True})
        self.assertEqual(proj.models[0].cost_class, "known")
        self.assertIsNotNone(proj.estimated_usd_savings)
        self.assertTrue(proj.models[0].rate_basis)

    def test_subscription_route_shows_no_dollars(self):
        _write_session(self.home / "sessions", "s", platform="telegram",
                       model="claude-sonnet-4-6", api_mode="subscription",
                       ceiling=CEILING, turns=[{"user": "hello"}, {"user": "write a file"}])
        loc = savings.discover_agents(self.home)[0]
        proj = savings.compute_projected(loc, {"enabled": True})
        self.assertEqual(proj.models[0].cost_class, "subscription")
        self.assertIsNone(proj.estimated_usd_savings)
        # No provider usage in the fixture -> session-input % suppressed; the
        # schema-only percentage is what remains defensible.
        self.assertIsNone(proj.net_input_reduction_pct)
        self.assertIsNotNone(proj.schema_reduction_pct)

    def test_unknown_route_shows_no_dollars(self):
        _write_session(self.home / "sessions", "u", platform="telegram",
                       model="claude-sonnet-4-6", ceiling=CEILING,  # no api_mode
                       turns=[{"user": "hello"}, {"user": "write a file"}])
        loc = savings.discover_agents(self.home)[0]
        proj = savings.compute_projected(loc, {"enabled": True})
        self.assertEqual(proj.models[0].cost_class, "unknown")
        self.assertIsNone(proj.estimated_usd_savings)

    def test_classify_cost_list_price_not_enough(self):
        # A model with a list price but a subscription route is NOT known-cost.
        cc = savings.classify_cost("claude-sonnet-4-6", api_mode="oauth")
        self.assertEqual(cc.cost_class, "subscription")
        self.assertFalse(cc.dollars_allowed)


class ApiCallRouteCostingTests(_HomeCase):
    """Costing evidence comes from `api_calls.jsonl` — the only place the
    billing route is actually recorded. Every evidence gate is preserved: no
    route evidence means `unknown` means no dollars."""

    TURNS = [{"user": "hello there"}, {"user": "write a file for me"}]

    def _project(self):
        return savings.compute_projected(
            savings.discover_agents(self.home)[0], {"enabled": True})

    def test_metered_route_from_api_calls_produces_usd(self):
        """session_meta carries no route; api_calls prove a metered one."""
        _write_session(self.home / "sessions", "m", platform="telegram",
                       model="claude-sonnet-4-6", ceiling=CEILING, turns=self.TURNS)
        _write_route_evidence(
            self.home / "state" / "tool-belt", "m",
            [{"provider": "anthropic", "api_mode": "api_key", "input_tokens": 50000}])
        proj = self._project()
        self.assertEqual(len(proj.models), 1)
        row = proj.models[0]
        self.assertEqual(row.cost_class, "known")
        self.assertEqual(row.provider, "anthropic")
        self.assertEqual(row.rate_basis, savings.PRICE_TABLE_RATE_BASIS)
        self.assertIsNotNone(row.estimated_usd_savings)
        self.assertGreater(proj.estimated_usd_savings, 0.0)
        self.assertEqual(proj.usd_coverage, "full")

    def test_no_api_call_rows_still_suppress_usd(self):
        """A session with no matching api_calls row has no route evidence."""
        _write_session(self.home / "sessions", "orphan", platform="telegram",
                       model="claude-sonnet-4-6", ceiling=CEILING, turns=self.TURNS)
        # Evidence exists, but for a different session file entirely.
        _write_route_evidence(
            self.home / "state" / "tool-belt", "elsewhere",
            [{"provider": "anthropic", "api_mode": "api_key", "input_tokens": 50000}])
        proj = self._project()
        self.assertEqual(proj.models[0].cost_class, "unknown")
        self.assertEqual(proj.models[0].provider, "")
        self.assertIsNone(proj.estimated_usd_savings)
        self.assertEqual(proj.usd_coverage, "none")

    def test_subscription_route_from_api_calls_shows_tokens_only(self):
        _write_session(self.home / "sessions", "sub", platform="telegram",
                       model="claude-sonnet-4-6", ceiling=CEILING, turns=self.TURNS)
        _write_route_evidence(
            self.home / "state" / "tool-belt", "sub",
            [{"provider": "anthropic", "api_mode": "oauth", "input_tokens": 50000}])
        proj = self._project()
        self.assertEqual(proj.models[0].cost_class, "subscription")
        self.assertIsNone(proj.estimated_usd_savings)
        self.assertEqual(proj.usd_coverage, "none")
        self.assertGreater(proj.gross_schema_token_reduction, 0)

    def test_conflicting_route_within_session_is_resolved_conservatively(self):
        """Calls that disagree on the route prove nothing; dollars stay off."""
        _write_session(self.home / "sessions", "mixed", platform="telegram",
                       model="claude-sonnet-4-6", ceiling=CEILING, turns=self.TURNS)
        _write_route_evidence(
            self.home / "state" / "tool-belt", "mixed",
            [{"provider": "anthropic", "api_mode": "api_key", "input_tokens": 20000},
             {"provider": "anthropic", "api_mode": "oauth", "input_tokens": 20000}])
        proj = self._project()
        self.assertEqual(proj.models[0].cost_class, "unknown")
        self.assertIsNone(proj.estimated_usd_savings)
        self.assertTrue(any("disagreed" in r for r in proj.reasons), proj.reasons)

    def test_session_meta_route_overrides_api_calls(self):
        """session_meta remains an explicit override when it carries a route."""
        _write_session(self.home / "sessions", "ovr", platform="telegram",
                       model="claude-sonnet-4-6", api_mode="subscription",
                       ceiling=CEILING, turns=self.TURNS)
        _write_route_evidence(
            self.home / "state" / "tool-belt", "ovr",
            [{"provider": "anthropic", "api_mode": "api_key", "input_tokens": 50000}])
        proj = self._project()
        self.assertEqual(proj.models[0].cost_class, "subscription")
        self.assertIsNone(proj.estimated_usd_savings)


class ComparableBaselineTests(_HomeCase):
    """`current` and `proposed` projections must be computed against the same
    non-learned configuration, or the before/after onboarding shows is
    apples-to-oranges."""

    SCOPE = "default:telegram"
    CONFIG = {"enabled": True, "channels": {SCOPE: {"learned_mode": "apply"}}}
    OVERLAY = {"version": 2, "scopes": {
        SCOPE: {"expand_only": ["web_search"], "carry": ["session_search"]}}}

    def test_proposed_branch_keeps_learned_overlay_and_channel_config(self):
        with mock.patch.object(learned, "load_state", return_value=self.OVERLAY):
            current = savings._resolve_effective_preset(self.CONFIG, self.SCOPE, None)
            proposed = savings._resolve_effective_preset(
                self.CONFIG, self.SCOPE, {"carry": [], "expand_only": []})
        # An empty proposed delta must reproduce the current effective preset.
        self.assertEqual(list(current.always_carry), list(proposed.always_carry))
        self.assertEqual(list(current.carry), list(proposed.carry))
        # ...and that shared baseline is the learned one, not raw policy.yaml.
        self.assertNotIn("web_search", proposed.carry)   # learned demotion honored
        self.assertIn("session_search", proposed.carry)  # learned promotion honored

    def test_proposed_delta_still_applies_on_top_of_the_overlay(self):
        with mock.patch.object(learned, "load_state", return_value=self.OVERLAY):
            proposed = savings._resolve_effective_preset(
                self.CONFIG, self.SCOPE,
                {"carry": ["cronjob"], "expand_only": ["read_file"]})
        self.assertIn("cronjob", proposed.carry)         # proposed promotion
        self.assertNotIn("read_file", proposed.carry)    # proposed demotion
        self.assertNotIn("web_search", proposed.carry)   # learned demotion survives
        self.assertIn("session_search", proposed.carry)  # learned promotion survives


class SinceParsingTests(_HomeCase):
    """(11) A malformed `--since` must fail loudly, never degrade to
    'report the entire history'."""

    def test_absent_since_means_no_cutoff(self):
        self.assertEqual(savings.parse_since(None), 0.0)
        self.assertEqual(savings.parse_since(""), 0.0)

    def test_valid_since_parses(self):
        self.assertGreater(savings.parse_since("2026-05-15"), 0.0)
        self.assertGreater(savings.parse_since("2026-05-15T10:30:00"), 0.0)

    def test_malformed_since_raises(self):
        for bad in ("2026-13-45", "lastweek", "05/15/2026"):
            with self.subTest(bad=bad):
                with self.assertRaises(savings.InvalidSinceError):
                    savings.parse_since(bad)

    def test_cli_exits_non_zero_on_malformed_since(self):
        _write_session(self.home / "sessions", "s", platform="telegram",
                       model="m", ceiling=CEILING, turns=[{"user": "hello"}])
        err = io.StringIO()
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = savings_cli.run(
                ["--since", "lastweek", "--hermes-home", str(self.home)])
        self.assertEqual(code, 2)
        self.assertIn("lastweek", err.getvalue())
        self.assertNotIn("OBSERVED", out.getvalue())


class ProposedAssignmentTests(_HomeCase):
    """(10) Onboarding-style proposed assignments can call the engine without
    writing state."""

    def test_proposed_projection_no_writes(self):
        _write_session(self.home / "sessions", "s", platform="telegram",
                       model="m", ceiling=CEILING,
                       turns=[{"user": "hello"}, {"user": "run some code"}])
        proposed = {"default:telegram": {"carry": ["execute_code"], "expand_only": ["terminal"]}}
        before = _snapshot(self.home)
        report = savings.compute(hermes_home=self.home, proposed_by_scope=proposed)
        after = _snapshot(self.home)
        self.assertEqual(before, after)  # no writes
        self.assertEqual(report.agents[0].projected.assignment_source, "proposed")


class NoWriteTests(_HomeCase):
    """(11)+(12) Engine and CLI perform no writes; live config/learned hashes
    and mtimes remain unchanged."""

    def _seed(self):
        _write_observed(self.home / "state" / "tool-belt", "default:telegram")
        _write_session(self.home / "sessions", "s", platform="telegram",
                       model="claude-sonnet-4-6", api_mode="api_key",
                       ceiling=CEILING, turns=[{"user": "hello"}, {"user": "write a file"}])
        # A live-looking learned.json + config we must not touch.
        state = self.home / "state" / "tool-belt"
        (state / "learned.json").write_text(
            json.dumps({"version": 2, "scopes": {}}), encoding="utf-8")
        (self.home / "config.yaml").write_text("plugins:\n  tool-belt:\n    enabled: true\n",
                                               encoding="utf-8")

    def test_engine_and_cli_never_write(self):
        self._seed()
        before = _snapshot(self.home)
        savings.compute(hermes_home=self.home)
        buf = io.StringIO()
        with redirect_stdout(buf):
            savings_cli.run(["--hermes-home", str(self.home)])
            savings_cli.run(["--json", "--hermes-home", str(self.home)])
            savings_cli.run(["--agent", "default", "--hermes-home", str(self.home)])
        after = _snapshot(self.home)
        self.assertEqual(before, after)


class LauncherHelperTests(_HomeCase):
    """The Phase 8 launcher helper writes only on explicit confirmation and
    never during a report."""

    def test_launcher_requires_confirmation(self):
        msgs = []
        user_home = Path(self.tmp.name) / "barehome"  # no ~/.local/bin → fallback
        created = savings_cli.ensure_launcher(
            self.home, PLUGIN_DIR / "tool-belt",
            confirm=lambda _p: False, out=msgs.append, user_home=user_home)
        self.assertFalse(created)
        self.assertFalse(savings_cli.launcher_path(self.home, user_home=user_home).exists())

    def test_launcher_created_on_confirmation(self):
        msgs = []
        user_home = Path(self.tmp.name) / "barehome"  # no ~/.local/bin → fallback
        created = savings_cli.ensure_launcher(
            self.home, PLUGIN_DIR / "tool-belt",
            confirm=lambda _p: True, out=msgs.append, user_home=user_home)
        self.assertTrue(created)
        launcher = savings_cli.launcher_path(self.home, user_home=user_home)
        self.assertTrue(launcher.exists())
        self.assertTrue(os.access(launcher, os.X_OK))

    def test_repo_executable_honors_hermes_python(self):
        env = dict(os.environ)
        env.update(HERMES_HOME=str(self.home), HERMES_PYTHON=sys.executable)
        result = subprocess.run(
            [str(PLUGIN_DIR / "tool-belt"), "--help"],
            env=env, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("tool-belt <command>", result.stdout)
        self.assertNotIn("cannot import run_agent", result.stderr)

    def test_launcher_quotes_interpreter_and_repo_paths(self):
        repo = self.home / "plugin repo" / "tool-belt"
        savings_cli.ensure_launcher(
            self.home, repo, confirm=lambda _p: True,
            python="/python env/bin/python3", out=lambda _m: None,
            user_home=self.home,
        )
        content = savings_cli.launcher_path(self.home, user_home=self.home).read_text(
            encoding="utf-8")
        self.assertIn("HERMES_PYTHON='/python env/bin/python3'", content)
        self.assertIn("exec '" + str(repo) + "' \"$@\"", content)

    def test_launcher_prefers_user_local_bin(self):
        """A standard install gets ~/.local/bin — the dir the Hermes installer
        guarantees is on PATH. $HERMES_HOME/bin is never on PATH by design."""
        user_home = Path(self.tmp.name) / "userhome"
        (user_home / ".local" / "bin").mkdir(parents=True)
        target = savings_cli.launcher_path(self.home, user_home=user_home)
        self.assertEqual(target.parent, user_home / ".local" / "bin")
        created = savings_cli.ensure_launcher(
            self.home, PLUGIN_DIR / "tool-belt",
            confirm=lambda _p: True, out=lambda _m: None,
            user_home=user_home,
        )
        self.assertTrue(created)
        self.assertTrue(target.exists())
        self.assertTrue(os.access(target, os.X_OK))

    def test_launcher_falls_back_when_no_user_local_bin(self):
        """Homes without ~/.local/bin (headless/CI) fall back to
        $HERMES_HOME/bin and receive PATH guidance for it."""
        user_home = Path(self.tmp.name) / "barehome"  # exists, no .local/bin
        msgs = []
        created = savings_cli.ensure_launcher(
            self.home, PLUGIN_DIR / "tool-belt",
            confirm=lambda _p: True, out=msgs.append,
            user_home=user_home,
        )
        self.assertTrue(created)
        target = self.home / "bin" / "tool-belt"
        self.assertTrue(target.exists())
        self.assertTrue(
            any("$HOME/.hermes" not in m and (str(self.home / "bin") in m
                or "not on your PATH" in m) for m in msgs),
            f"expected PATH guidance mentioning the launcher dir, got {msgs}")


class LauncherStalenessTests(_HomeCase):
    """(13a) An existing file at the launcher target is never taken on faith:
    the baked-in absolute exec path must still exist and still be ours."""

    def setUp(self):
        super().setUp()
        self.user_home = Path(self.tmp.name) / "barehome"  # no ~/.local/bin
        self.target = savings_cli.launcher_path(self.home, user_home=self.user_home)

    def _ensure(self, repo, *, confirm=True, msgs=None):
        return savings_cli.ensure_launcher(
            self.home, repo, confirm=lambda _p: confirm,
            out=(msgs.append if msgs is not None else (lambda _m: None)),
            user_home=self.user_home)

    def test_correct_launcher_is_idempotent_without_reconfirming(self):
        real = PLUGIN_DIR / "tool-belt"
        self._ensure(real)
        first = self.target.read_text(encoding="utf-8")
        msgs = []

        def _never(_prompt):
            raise AssertionError("a current launcher must not re-prompt")

        created = savings_cli.ensure_launcher(
            self.home, real, confirm=_never, out=msgs.append,
            user_home=self.user_home)
        self.assertTrue(created)
        self.assertEqual(self.target.read_text(encoding="utf-8"), first)
        self.assertTrue(any("already present" in m for m in msgs), msgs)

    def test_stale_exec_target_is_refreshed(self):
        gone = Path(self.tmp.name) / "moved-plugin" / "tool-belt"
        self._ensure(gone)
        self.assertIn(str(gone), self.target.read_text(encoding="utf-8"))
        real = PLUGIN_DIR / "tool-belt"
        msgs = []
        created = self._ensure(real, msgs=msgs)
        self.assertTrue(created)
        content = self.target.read_text(encoding="utf-8")
        self.assertIn(f"exec {real} ", content)
        self.assertNotIn(str(gone), content)
        self.assertTrue(any("stale" in m for m in msgs), msgs)

    def test_stale_launcher_refresh_still_requires_confirmation(self):
        gone = Path(self.tmp.name) / "moved-plugin" / "tool-belt"
        self._ensure(gone)
        before = self.target.read_text(encoding="utf-8")
        msgs = []
        created = self._ensure(PLUGIN_DIR / "tool-belt", confirm=False, msgs=msgs)
        self.assertFalse(created)
        self.assertEqual(self.target.read_text(encoding="utf-8"), before)
        self.assertTrue(any("Skipped" in m for m in msgs), msgs)

    def test_foreign_file_is_never_adopted_or_overwritten(self):
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.target.write_text("#!/bin/sh\necho some other program\n", encoding="utf-8")
        msgs = []
        created = self._ensure(PLUGIN_DIR / "tool-belt", msgs=msgs)
        self.assertFalse(created)
        self.assertIn("some other program", self.target.read_text(encoding="utf-8"))
        self.assertTrue(any("not created by Tool Belt" in m for m in msgs), msgs)


class ShLauncherTests(unittest.TestCase):
    """(13b) The repo-root `tool-belt` sh launcher stays POSIX-clean and works
    through a hand-made symlink (`ln -s ... ~/.local/bin/tool-belt`)."""

    def test_launcher_is_posix_sh_clean(self):
        result = subprocess.run(["sh", "-n", str(PLUGIN_DIR / "tool-belt")],
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_launcher_works_through_a_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp) / "bin" / "tool-belt"
            link.parent.mkdir(parents=True)
            link.symlink_to(PLUGIN_DIR / "tool-belt")
            env = dict(os.environ)
            env.update(HERMES_PYTHON=sys.executable)
            env.pop("HERMES_HOME", None)
            result = subprocess.run([str(link), "--help"], env=env, text=True,
                                    capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("tool-belt <command>", result.stdout)


def _snapshot(root: Path) -> dict:
    """Map every file under ``root`` to (size, mtime_ns, sha1)."""
    import hashlib
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            data = p.read_bytes()
            st = p.stat()
            out[str(p.relative_to(root))] = (
                st.st_size, st.st_mtime_ns, hashlib.sha1(data).hexdigest())
    return out


if __name__ == "__main__":
    unittest.main()
