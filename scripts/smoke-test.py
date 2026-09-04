#!/usr/bin/env python3
"""End-to-end smoke test for the tool-belt plugin (both postures).

Runs hand-curated session scenarios through the plugin in an isolated
tempdir, then asserts on the resulting telemetry that the plugin's
invariants hold in both cache-off (per-turn narrowing) and cache-on
(carry-all) postures.

What this validates (mechanical, not behavioral):
  Cache-off block (narrowing — the value engine):
    · session_id populated on every prediction row
    · bypass cohort distribution matches configured bypass_rate
    · expansion_provided_access NEVER credited when was_initially_active
    · Sticky residency carries within session, survives on_session_end
      (per-turn hook), evicts on on_session_reset
    · Cross-session isolation: expansion doesn't leak across sessions
  Cache-on block (carry-all):
    · every prediction row is policy_source == "cache_on_carry_all" with
      ceiling_count == narrowed_count (nothing narrowed, reduction 0)
    · expand_tools is ABSENT from the wire (kwargs["tools"]) and from the
      row's active_tools — nothing to expand, so it is not shipped
    · EXACTLY ONE tool_list_hash per session — the list is never mutated
    · no expand_tools rows in tool_calls.jsonl for cache-on sessions
    · a tool the narrowing posture would have gated (browser_navigate) is
      already on the wire and its call is was_initially_active, never
      credited to expansion
    · api_calls.jsonl rows carry the provider_caches key
    · on_session_end does NOT evict the posture pin (per-turn hook);
      on_session_reset DOES (true reset)

What this does NOT validate:
  · Whether the predictor's regex triggers classify real user intent
    correctly (behavioral — needs organic data, not synthetic input)
  · Token-savings numbers (depends on real Hermes toolset sizes)
  · The compaction patch (would need a mock _compress_context call)

Usage:
  python3 scripts/smoke-test.py

Exits 0 on all-green, 1 on any assertion failure. No side effects
outside the temp directory it creates.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
PLUGIN_DIR = HERE.parent
sys.path.insert(0, str(PLUGIN_DIR.parent))
sys.path.insert(0, str(PLUGIN_DIR))

# The shared loader registers the package alias the plugin uses internally,
# so this script, the other scripts, and the unit tests all load the plugin
# the same way.
sys.path.insert(0, str(HERE))
from _plugin_loader import load_plugin_package  # noqa: E402

plugin = load_plugin_package()
logger_io = importlib.import_module("tool_belt_plugin.logger_io")


# Minimal but realistic-shaped Hermes tools payload. Each entry is the
# Anthropic-format dict the plugin sees in kwargs["tools"]. Names cover
# the trigger categories the scenarios exercise (file, shell, browser,
# delegation, code execution) plus a few unlisted enabled tools to exercise
# their derived expand-only path.
SYNTHETIC_TOOLS = [
    # always_carry / carry per policy
    {"name": "session_search", "description": "x", "input_schema": {}},
    {"name": "mnemosyne_remember", "description": "x", "input_schema": {}},
    {"name": "mnemosyne_recall", "description": "x", "input_schema": {}},
    {"name": "clarify", "description": "x", "input_schema": {}},
    {"name": "skill_view", "description": "x", "input_schema": {}},
    {"name": "skills_list", "description": "x", "input_schema": {}},
    {"name": "web_search", "description": "x", "input_schema": {}},
    {"name": "read_file", "description": "x", "input_schema": {}},
    {"name": "search_files", "description": "x", "input_schema": {}},
    {"name": "send_message", "description": "x", "input_schema": {}},
    {"name": "expand_tools", "description": "x", "input_schema": {}},
    # Trigger-gated
    {"name": "write_file", "description": "x", "input_schema": {}},
    {"name": "patch", "description": "x", "input_schema": {}},
    {"name": "terminal", "description": "x", "input_schema": {}},
    {"name": "process", "description": "x", "input_schema": {}},
    {"name": "browser_navigate", "description": "x", "input_schema": {}},
    {"name": "browser_click", "description": "x", "input_schema": {}},
    {"name": "delegate_task", "description": "x", "input_schema": {}},
    {"name": "execute_code", "description": "x", "input_schema": {}},
    # Unknown to the policy — exercises the derived expand-only path
    {"name": "custom_unknown_tool", "description": "x", "input_schema": {}},
]


@dataclass
class Scenario:
    """A single session's worth of synthetic traffic.

    Each turn is (message, [tool_calls_during_this_turn]). The driver
    fires pre_gateway_dispatch + the wrapped _build_api_kwargs once per
    turn, then synthesizes a post_tool_call for each named tool.
    """
    chat_id: str
    platform: str = "telegram"
    turns: list[tuple[str, list[str]]] = field(default_factory=list)
    # If a turn has an "expand_tools(<cat>)" entry, the driver fakes a
    # pending_expansion in state so subsequent calls in the same turn
    # see the right post-expansion world.


# Targeted scenarios that pin specific audit behaviors. Followed by a
# bulk batch of throwaway sessions for bypass-cohort distribution.
TARGETED_SCENARIOS: list[Scenario] = [
    # 1. Conversational. No triggers. Validates session_id population.
    Scenario("conv-001", turns=[("hi how are you", [])]),
    Scenario("conv-002", turns=[("ok thanks", [])]),
    Scenario("conv-003", turns=[("what do you think about that", [])]),

    # 2. Trigger fires + tool called same turn. Validates TP classification.
    Scenario("write-001", turns=[
        ("please write that to a file", ["write_file"]),
    ]),
    Scenario("shell-001", turns=[
        ("run ls in the workspace", ["terminal"]),
    ]),

    # 3. Already-available tool + sticky context — must NOT be credited
    #    as expansion-driven. This guards the expansion-attribution invariant.
    Scenario("already-available-001", turns=[
        # terminal is not carried by policy — it activates via the shell trigger.
        # Simulate a case where it is already in the initial active set.
        # Use write_file: it's trigger-gated, fires on "save". Then we
        # call expand_tools(file) which re-adds the same tool. The
        # post_tool_call for write_file must NOT show expansion_provided_access.
        ("save these notes please", [
            "write_file",                # initially available via trigger
            "expand_tools(file)",        # redundant expansion
            "write_file",                # second call — must NOT be credited
        ]),
    ]),

    # 4. Genuine expansion: tool NOT initially available, expand_tools
    #    loads it, model calls it. Must credit.
    Scenario("genuine-expansion-001", turns=[
        # "open google" doesn't match any default trigger; browser is gated.
        ("could you check out google.com", [
            "expand_tools(browser)",
            "browser_navigate",          # must credit as expansion-driven
        ]),
    ]),

    # 5. Cross-session isolation. Two sessions, same scope. First
    #    expands browser; second must NOT inherit that sticky state.
    Scenario("isolation-1-leaker", turns=[
        ("check google.com", [
            "expand_tools(browser)",
            "browser_navigate",
        ]),
    ]),
    Scenario("isolation-2-victim", turns=[
        # Plain conversational; no triggers; should have ZERO sticky tools.
        ("nice weather today", []),
    ]),
]

# Bulk filler for bypass-cohort distribution. With bypass_rate=0.05 on
# a 100-session sample, expect ~5; the assertion accepts 1-15.
FILLER_SCENARIOS: list[Scenario] = [
    Scenario(f"filler-{i:03d}", turns=[(f"message number {i}", [])])
    for i in range(100)
]


# ────────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────────

def _build_event(chat_id: str, platform: str, text: str) -> SimpleNamespace:
    """Mirror Hermes' MessageEvent shape enough for the plugin to read it."""
    source = SimpleNamespace(
        platform=SimpleNamespace(value=platform),
        chat_id=chat_id,
    )
    return SimpleNamespace(source=source, text=text, attachments=[])


def _build_session_store(canonical_key: str):
    store = mock.MagicMock()
    store._generate_session_key.return_value = canonical_key
    return store


# Wire-level record of what each dispatch actually shipped:
# (canonical_session_key, turn_idx, [tool names in kwargs["tools"]]).
# Cleared between blocks. Lets the cache-on assertions check the wire
# itself rather than only the telemetry row that describes it.
WIRE_LOG: list[tuple[str, int, list[str]]] = []


def _run_wrapped_build_api_kwargs(tools: list[dict]) -> dict:
    """Invoke the plugin's per-call path with a synthetic tools list.

    Constructs a temporary wrapped function over a fake "original" that
    returns kwargs unchanged. This exercises the same code path the
    AIAgent monkey-patch runs in production, including writing the
    prediction row. Returns the kwargs the plugin would hand to the API.
    """
    def fake_original(self, api_messages):
        return {"tools": list(tools)}

    wrapped = plugin._wrap_build_api_kwargs(fake_original)
    return wrapped(self=SimpleNamespace(), api_messages=[]) or {}


def _canonical_key(chat_id: str, platform: str) -> str:
    # Mirror gateway.session.build_session_key shape.
    return f"agent:main:{platform}:dm:{chat_id}"


def run_scenario(scenario: Scenario, profile_home: Path,
                 api_usage: dict | None = None) -> None:
    """Drive one scenario. ``api_usage``, when given, is fed to the
    post_api_request hook after every dispatch (as Hermes would with the
    provider's usage block) so api_calls.jsonl rows get written; the
    cache-off block leaves it None — that path is asserted on
    predictions/tool_calls only and is unchanged."""
    canonical = _canonical_key(scenario.chat_id, scenario.platform)
    sticky_key_expected = plugin._sticky_key_for_session(canonical)

    for turn_idx, (message, tool_calls) in enumerate(scenario.turns):
        event = _build_event(scenario.chat_id, scenario.platform, message)
        store = _build_session_store(canonical)

        with mock.patch.dict(os.environ,
                             {"HERMES_SESSION_KEY": canonical,
                              "HERMES_HOME": str(profile_home)},
                             clear=False):
            plugin._on_pre_gateway_dispatch(event=event, gateway=None, session_store=store)
            kwargs = _run_wrapped_build_api_kwargs(SYNTHETIC_TOOLS)
            WIRE_LOG.append((canonical, turn_idx, [
                str(t.get("name") or "") for t in (kwargs.get("tools") or [])
                if isinstance(t, dict)
            ]))
            if api_usage is not None:
                plugin._on_post_api_request(
                    usage=dict(api_usage),
                    model=str(api_usage.get("_model") or "smoke-model"),
                    provider=str(api_usage.get("_provider") or "smoke-provider"),
                    api_call_count=turn_idx + 1,
                )

            for call in tool_calls:
                if call.startswith("expand_tools("):
                    # Synthesize a pending_expansion + sticky refresh.
                    category = call[len("expand_tools("):-1]
                    # Faux resolved tools for the category — small set
                    # is fine; the driver doesn't validate the category
                    # math, just the attribution logic.
                    resolved = _fake_resolved_for(category)
                    state = plugin._PREDICTION_CV.get() or {}
                    state["pending_expansion"] = {
                        "category": category,
                        "resolved_tools": resolved,
                        "tools_added": resolved,
                    }
                    state["expansions"] = (state.get("expansions") or set()) | set(resolved)
                    plugin._PREDICTION_CV.set(state)
                    plugin._refresh_sticky(
                        sticky_key=sticky_key_expected,
                        category=category,
                        tools=resolved,
                        policy_scope=state.get("scope", ""),
                    )
                    # Log the expand_tools call itself. A primary model
                    # dispatch carries the provider tool_call_id — synthesize
                    # one so the plugin's primary-dispatch gate logs the row.
                    plugin._on_post_tool_call(
                        tool_name="expand_tools",
                        args={"category": category},
                        result={"ok": True, "tools": resolved},
                        task_id=f"task-{scenario.chat_id}-{turn_idx}",
                        session_id=canonical,
                        tool_call_id=f"tc-{scenario.chat_id}-{turn_idx}-expand",
                    )
                else:
                    plugin._on_post_tool_call(
                        tool_name=call,
                        args={},
                        result="ok",
                        task_id=f"task-{scenario.chat_id}-{turn_idx}",
                        session_id=canonical,
                        tool_call_id=f"tc-{scenario.chat_id}-{turn_idx}-{call}",
                    )

            plugin._PREDICTION_CV.set(None)

    plugin._on_session_end(session_id=canonical)


def _fake_resolved_for(category: str) -> list[str]:
    mapping = {
        "browser": ["browser_navigate", "browser_click"],
        "file": ["write_file", "patch", "read_file", "search_files"],
        "terminal": ["terminal", "process"],
        "code_execution": ["execute_code"],
        "delegation": ["delegate_task"],
    }
    return mapping.get(category, [])


# ────────────────────────────────────────────────────────────────────────
# Assertions
# ────────────────────────────────────────────────────────────────────────

class Check:
    def __init__(self):
        self.results: list[tuple[bool, str]] = []

    def ok(self, msg: str) -> None:
        self.results.append((True, msg))

    def fail(self, msg: str) -> None:
        self.results.append((False, msg))

    def assert_(self, cond: bool, msg: str) -> None:
        (self.ok if cond else self.fail)(msg)

    def report(self) -> int:
        passed = sum(1 for ok, _ in self.results if ok)
        total = len(self.results)
        print(f"\n{'='*70}\n  {passed}/{total} checks passed\n{'='*70}")
        for ok, msg in self.results:
            print(f"  {'PASS' if ok else 'FAIL'}  {msg}")
        return 0 if passed == total else 1


def run_cache_off_assertions(state_dir: Path, check: Check) -> None:
    """Cache-off path assertions: sticky carries, expansion attribution, etc."""
    preds = [json.loads(l) for l in (state_dir / "predictions.jsonl").read_text().splitlines() if l]
    calls = [json.loads(l) for l in (state_dir / "tool_calls.jsonl").read_text().splitlines() if l]

    # ─── session_id populated on every prediction row ───
    blank_sids = [p for p in preds if not p.get("session_id")]
    check.assert_(not blank_sids,
        f"session_id populated on all prediction rows (blank: {len(blank_sids)}/{len(preds)})")

    # Same for tool_calls
    blank_call_sids = [c for c in calls if not c.get("session_id")]
    check.assert_(not blank_call_sids,
        f"session_id populated on all tool_call rows (blank: {len(blank_call_sids)}/{len(calls)})")

    # ─── bypass cohort distribution ───
    # We ran 100+ sessions with bypass_rate=0.05. Each session is in or out
    # deterministically by hash; expect ~5 bypass sessions, accepted range 1-15.
    sessions_in_bypass = {p["session_id"] for p in preds if p.get("policy_source") == "bypass"}
    total_sessions = {p["session_id"] for p in preds if p.get("session_id")}
    bypass_rate_observed = len(sessions_in_bypass) / max(1, len(total_sessions))
    check.assert_(1 <= len(sessions_in_bypass) <= 15,
        f"bypass cohort within expected range "
        f"(observed: {len(sessions_in_bypass)} sessions / {len(total_sessions)} "
        f"= {bypass_rate_observed:.1%}, target ~5%)")

    # ─── expansion_provided_access NEVER True when was_initially_active ───
    spurious = [c for c in calls
                if c.get("expansion_provided_access") is True
                and c.get("was_initially_active") is True]
    check.assert_(not spurious,
        f"expansion_provided_access never credited when was_initially_active "
        f"(spurious: {len(spurious)})")

    # ─── attribution sanity: legitimate expansion IS credited ───
    legit = [c for c in calls
             if c.get("expansion_provided_access") is True
             and c.get("was_initially_active") is False
             and c.get("tool_name") == "browser_navigate"]
    check.assert_(legit,
        f"legitimate post-expand calls ARE credited "
        f"(found {len(legit)} browser_navigate expansion-driven calls)")

    # ─── Cross-session isolation ───
    # The "victim" session must have zero expansion_provided_access flags despite
    # the prior session having expanded browser on the same scope.
    victim_calls = [c for c in calls if c.get("session_id", "").endswith("isolation-2-victim")]
    victim_expanded = [c for c in victim_calls if c.get("expansion_provided_access") is True]
    check.assert_(not victim_expanded,
        f"sticky residency does not leak across sessions "
        f"(victim session expansion credits: {len(victim_expanded)})")

    # ─── Session lifecycle (M2 contract) ───
    # on_session_end fires PER TURN, so session-scoped sticky residency and
    # lookback history must SURVIVE it (evicting them per turn defeated both
    # features); only on_session_reset clears them. Sticky additionally
    # self-decays on its TTL. Prove both halves on a fresh probe session.
    probe = _canonical_key("reset-probe", "telegram")
    probe_sticky = plugin._sticky_key_for_session(probe)
    plugin._STICKY_BY_KEY[probe_sticky] = {
        "browser": {"tools": {"browser_navigate"}, "remaining_turns": 3}}
    plugin._PRIOR_MESSAGES_BY_SESSION[probe] = ["remember me"]
    with mock.patch.dict(os.environ, {"HERMES_SESSION_KEY": probe}, clear=False):
        plugin._on_session_end(session_id=probe)
    check.assert_(
        probe_sticky in plugin._STICKY_BY_KEY
        and probe in plugin._PRIOR_MESSAGES_BY_SESSION,
        "on_session_end LEAVES session-scoped sticky/lookback (per-turn hook)")
    plugin._on_session_reset(session_id="reset-probe-new-uuid",
                             platform="telegram", session_key=probe)
    check.assert_(probe_sticky not in plugin._STICKY_BY_KEY,
        "on_session_reset evicts sticky residency")
    check.assert_(probe not in plugin._PRIOR_MESSAGES_BY_SESSION,
        "on_session_reset evicts lookback prior-message buffer")


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

CACHE_ON_SCENARIOS: list[Scenario] = [
    # Multi-turn chit-chat: carry-all on every turn, one hash for the session.
    Scenario("cache-on-multi-001", turns=[
        ("hi how's it going", []),
        ("any updates today", []),
        ("thanks", []),
    ]),
    # A tool the narrowing posture gates behind expand_tools (browser_* is
    # demoted by _seed_demoted). Under carry-all it is on the wire from the
    # first call, so the model calls it directly — there is no expand_tools
    # to call (it isn't shipped) and the list never changes. The scenario is
    # deliberately mixed-intent across turns to give the (non-running)
    # predictor every chance to change its mind; the hash must not move.
    Scenario("cache-on-direct-001", turns=[
        ("morning", []),
        ("open google.com", ["browser_navigate"]),
        ("now write that to a file", ["write_file"]),
        ("and run ls", ["terminal"]),
    ]),
]

# Provider usage block fed to post_api_request for the cache-on block —
# a warm caching provider (cache_read > 0). The _model/_provider keys are
# driver metadata, not part of the usage dict the hook sees.
CACHE_ON_USAGE = {
    "input_tokens": 1200, "output_tokens": 80,
    "cache_read_tokens": 9000, "cache_write_tokens": 0,
    "_model": "smoke-model", "_provider": "openrouter",
}


def run_cache_on_assertions(state_dir: Path, check: Check) -> None:
    """Cache-on (carry-all) contract: everything ships, nothing mutates."""
    preds = [json.loads(l) for l in (state_dir / "predictions.jsonl").read_text().splitlines() if l]
    calls = [json.loads(l) for l in (state_dir / "tool_calls.jsonl").read_text().splitlines() if l]
    api_path = state_dir / "api_calls.jsonl"
    api_rows = ([json.loads(l) for l in api_path.read_text().splitlines() if l]
                if api_path.exists() else [])
    expected_turns = sum(len(sc.turns) for sc in CACHE_ON_SCENARIOS)

    check.assert_(len(preds) == expected_turns,
        f"one prediction row per dispatch under cache-on "
        f"(found {len(preds)}, expected {expected_turns})")

    # ─── Every row is the carry-all cohort with nothing narrowed ───
    not_carry_all = [p for p in preds if p.get("policy_source") != "cache_on_carry_all"]
    check.assert_(not not_carry_all,
        f"every cache-on prediction row has policy_source == cache_on_carry_all "
        f"(other: {sorted({str(p.get('policy_source')) for p in not_carry_all})})")
    narrowed = [p for p in preds if p.get("ceiling_count") != p.get("narrowed_count")]
    check.assert_(not narrowed and all(int(p.get("tokens_saved") or 0) == 0 for p in preds),
        f"ceiling_count == narrowed_count and tokens_saved == 0 on every row "
        f"(narrowed rows: {len(narrowed)})")

    # ─── expand_tools is NOT shipped: on the wire, and on the row ───
    wire_with_expand = [(sid, t) for sid, t, names in WIRE_LOG if "expand_tools" in names]
    check.assert_(WIRE_LOG and not wire_with_expand,
        f"expand_tools absent from kwargs['tools'] on every cache-on dispatch "
        f"({len(WIRE_LOG)} dispatches captured, {len(wire_with_expand)} shipped it)")
    row_with_expand = [p for p in preds if "expand_tools" in (p.get("active_tools") or [])]
    check.assert_(not row_with_expand,
        f"expand_tools absent from active_tools on every prediction row "
        f"(rows carrying it: {len(row_with_expand)})")
    # Everything ELSE in the synthetic ceiling is on the wire — including the
    # tools _seed_demoted marks expand_only, which carry-all ignores.
    expected_wire = sorted(t["name"] for t in SYNTHETIC_TOOLS if t["name"] != "expand_tools")
    short_wire = [(sid, t) for sid, t, names in WIRE_LOG if sorted(names) != expected_wire]
    check.assert_(not short_wire,
        f"full ceiling (minus expand_tools) on the wire for every dispatch, "
        f"demotions ignored (short dispatches: {len(short_wire)})")

    # ─── EXACTLY ONE tool_list_hash per session ───
    preds_by_session: dict[str, list[dict]] = defaultdict(list)
    for p in preds:
        preds_by_session[p["session_id"]].append(p)
    unstable = {
        sid[-40:]: len({pp.get("tool_list_hash", "") for pp in plist})
        for sid, plist in preds_by_session.items()
        if len({pp.get("tool_list_hash", "") for pp in plist}) != 1
    }
    check.assert_(len(preds_by_session) == len(CACHE_ON_SCENARIOS) and not unstable,
        f"exactly one tool_list_hash per cache-on session, including the mixed-intent one "
        f"(sessions: {len(preds_by_session)}, unstable: {unstable})")
    wire_hashes_by_session: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for sid, _t, names in WIRE_LOG:
        wire_hashes_by_session[sid].add(tuple(names))
    check.assert_(all(len(v) == 1 for v in wire_hashes_by_session.values()),
        "the wire-level tool list is byte-identical across every turn of a session")

    # ─── No expand_tools calls, and gated tools were already available ───
    expand_calls = [c for c in calls if c.get("tool_name") == "expand_tools"]
    check.assert_(not expand_calls,
        f"no expand_tools rows in tool_calls.jsonl for cache-on sessions "
        f"(found {len(expand_calls)})")
    direct = [c for c in calls if c.get("tool_name") in ("browser_navigate", "write_file", "terminal")]
    check.assert_(
        len(direct) == 3
        and all(c.get("was_initially_active") is True for c in direct)
        and not any(c.get("expansion_provided_access") is True for c in direct),
        f"gated-under-narrowing tools were initially active and never credited to "
        f"expansion (calls: {len(direct)})")

    # ─── api_calls.jsonl rows carry the provider verdict ───
    check.assert_(len(api_rows) == expected_turns,
        f"one api_calls.jsonl row per dispatch (found {len(api_rows)}, expected {expected_turns})")
    missing_pc = [r for r in api_rows if "provider_caches" not in r]
    check.assert_(api_rows and not missing_pc,
        f"every api_calls.jsonl row carries the provider_caches key "
        f"(missing: {len(missing_pc)})")
    bad_pc = [r for r in api_rows if r.get("provider_caches") is not True]
    check.assert_(not bad_pc,
        f"provider_caches is True under forced cache_mode=on "
        f"(other values: {sorted({repr(r.get('provider_caches')) for r in bad_pc})})")
    api_by_session: dict[str, set[str]] = defaultdict(set)
    for r in api_rows:
        api_by_session[r.get("session_id", "")].add(str(r.get("tool_list_hash") or ""))
    check.assert_(api_by_session and all(len(v) == 1 for v in api_by_session.values()),
        "api_calls.jsonl sees one tool_list_hash per session too")

    # ─── Posture pin lifecycle ───
    # on_session_end fires PER TURN and must leave the session's posture pin
    # in place; on_session_reset is the true reset and must evict it.
    probe = _canonical_key("cache-on-reset-probe", "telegram")
    plugin._CACHE_DECISION_BY_SESSION[probe] = {"mode": "on", "provider": "openrouter"}
    with mock.patch.dict(os.environ, {"HERMES_SESSION_KEY": probe}, clear=False):
        plugin._on_session_end(session_id=probe)
    check.assert_(probe in plugin._CACHE_DECISION_BY_SESSION,
        "on_session_end LEAVES the session's posture pin (per-turn hook)")
    plugin._on_session_reset(session_id="cache-on-reset-probe-new-uuid",
                             platform="telegram", session_key=probe)
    check.assert_(probe not in plugin._CACHE_DECISION_BY_SESSION,
        "on_session_reset evicts the session's posture pin")


def _reset_plugin_state() -> None:
    """Clear every per-session / cross-session global so blocks don't bleed.
    Mirrors tests/test_cache_aware.py's helper of the same name."""
    plugin._STICKY_BY_KEY.clear()
    plugin._POLICY_TURN_BY_SCOPE.clear()
    plugin._PRIOR_MESSAGES_BY_SESSION.clear()
    plugin._CACHE_MODE_BY_SESSION.clear()       # per-session detection buckets
    plugin._CACHE_DECISION_BY_SESSION.clear()   # per-session posture pin
    plugin._LAST_CANONICAL_BY_PLATFORM.clear()
    plugin._DETECTION_CACHE.clear()             # cross-session scope|provider locks
    plugin._DETECTION_CACHE_LOADED = False
    WIRE_LOG.clear()


# Tools the smoke scenarios treat as expand_only. Under the full-start
# contract a scope with no learned state carries EVERYTHING enabled, so the
# expand_tools recovery path would never fire and the expansion-credit
# assertions would have nothing to bite on. Seeding these demotions makes the
# temp homes look like an evidence-shaped install — which is exactly the state
# in which expansion crediting matters.
SMOKE_DEMOTED = ["browser_navigate", "browser_click", "custom_unknown_tool"]


def _seed_demoted(state_dir: Path) -> None:
    """Write a learned.json demoting SMOKE_DEMOTED for the telegram platform
    scope (matches assistant-a:telegram via the platform fallback)."""
    (state_dir / "learned.json").write_text(json.dumps({
        "version": 2,
        "scopes": {"telegram": {
            "carry": [], "expand_only": list(SMOKE_DEMOTED), "shaping": {},
        }},
    }, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        original_config = dict(plugin._CONFIG)
        off_home = Path(tmp) / "off" / "profiles" / "assistant-a"
        off_state = off_home / "state" / "tool-belt"
        off_state.mkdir(parents=True, exist_ok=True)
        _seed_demoted(off_state)
        on_home = Path(tmp) / "on" / "profiles" / "assistant-a"
        on_state = on_home / "state" / "tool-belt"
        on_state.mkdir(parents=True, exist_ok=True)
        _seed_demoted(on_state)

        try:
            # ──────── Block 1: cache-off (per-turn pipeline) ────────
            _reset_plugin_state()
            plugin._CONFIG.clear()
            plugin._CONFIG.update({
                "enabled": True,
                "log": True,
                "agent": "assistant-a",
                "bypass_rate": 0.05,
                "cache_mode": "off",  # exercise the cache-off per-turn path
                "channels": {},
                "cache_off": {
                    "sticky": {"enabled": True, "ttl_turns": 3, "categories": ["*"]},
                    "predictor": {"lookback_turns": 1},
                },
            })

            scenarios = TARGETED_SCENARIOS + FILLER_SCENARIOS
            print(f"[cache-off] Running {len(scenarios)} scenarios in {off_state}...")
            for sc in scenarios:
                run_scenario(sc, off_home)
            print(f"  → {sum(1 for f in off_state.iterdir())} state files written")

            # Primary-dispatch gate: a nested/secondary dispatch arrives with
            # an EMPTY tool_call_id (code-execution sandbox, MCP tools
            # server, memory/mnemosyne batch fan-out) and must NOT create a
            # tool_calls.jsonl row. Emit one right into the pipeline's state
            # dir so the drop contract is smoke-tested end-to-end, then
            # assert it in run_cache_off_assertions.
            pre_row_count = len(
                (off_state / "tool_calls.jsonl").read_text().splitlines()
            ) if (off_state / "tool_calls.jsonl").exists() else 0
            plugin._PREDICTION_CV.set({
                "prediction_id": "smoke-nested-gate",
                "agent": "assistant-a",
                "platform": "telegram",
                "scope": "assistant-a:telegram",
                "session_id": "smoke-nested-gate-session",
                "initial_active_tools": ["terminal"],
                "baseline_active_tools": ["terminal"],
                "expand_only_tools": ["memory"],
                "expansions": {"coding", "memory"},
                "pending_expansion": {
                    "category": "coding",
                    "resolved_tools": ["memory", "read_file", "write_file"],
                    "tools_added": ["memory", "read_file", "write_file"],
                },
                "ceiling_tools": ["terminal", "memory"],
            })
            plugin._on_post_tool_call(
                tool_name="memory", args={}, result="ok",
                task_id="smoke-nested-gate-session", session_id="",
                tool_call_id="",
            )
            plugin._PREDICTION_CV.set(None)
            post_row_count = len(
                (off_state / "tool_calls.jsonl").read_text().splitlines()
            ) if (off_state / "tool_calls.jsonl").exists() else 0
            check = Check()
            check.assert_(pre_row_count == post_row_count,
                f"id-less nested dispatch logged no row "
                f"({pre_row_count} -> {post_row_count})")
            check.report()

            check = Check()
            run_cache_off_assertions(off_state, check)
            off_rc = check.report()

            # ──────── Block 2: cache-on (carry-all) ────────
            _reset_plugin_state()
            plugin._CONFIG["cache_mode"] = "on"
            # bypass is the OTHER unnarrowed cohort (policy_source "bypass");
            # keep it out so every row here is cache_on_carry_all.
            plugin._CONFIG["bypass_rate"] = 0.0

            print(f"\n[cache-on] Running {len(CACHE_ON_SCENARIOS)} multi-turn scenarios in {on_state}...")
            for sc in CACHE_ON_SCENARIOS:
                run_scenario(sc, on_home, api_usage=CACHE_ON_USAGE)
            print(f"  → {sum(1 for f in on_state.iterdir())} state files written")

            on_check = Check()
            run_cache_on_assertions(on_state, on_check)
            on_rc = on_check.report()

            return off_rc or on_rc
        finally:
            plugin._CONFIG.clear()
            plugin._CONFIG.update(original_config)
            _reset_plugin_state()


if __name__ == "__main__":
    sys.exit(main())
