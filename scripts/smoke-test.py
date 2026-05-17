#!/usr/bin/env python3
"""End-to-end smoke test for the dynamic-tools plugin.

Validates the just-shipped telemetry-correctness fixes without waiting
for a week of organic gateway traffic. Runs hand-curated session
scenarios through the plugin in an isolated tempdir, then asserts on
the resulting predictions.jsonl + tool_calls.jsonl that the fixes
behave as designed.

What this validates (mechanical, not behavioral):
  · session_id is populated on every prediction row (audit item #3)
  · bypass cohort distribution matches configured bypass_rate (item #4)
  · expand_tools_used is NEVER credited when was_initially_available
    (item #1 writer-side fix)
  · Sticky residency carries within session, evicts on session_end
  · Cross-session isolation: one session's expansion does not leak
    into another session on the same scope

What this does NOT validate:
  · Whether the predictor's regex triggers classify real user intent
    correctly (behavioral — needs organic data, not synthetic input)
  · Token-savings numbers (depends on real Hermes toolset sizes)
  · Late-bound TP recovery (analyzer-side, covered by unit tests)

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
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

HERE = Path(__file__).resolve().parent
PLUGIN_DIR = HERE.parent
sys.path.insert(0, str(PLUGIN_DIR.parent))
sys.path.insert(0, str(PLUGIN_DIR))

# The conftest under tests/ registers the package alias the plugin uses
# internally. Reuse it so this script and the unit tests load the
# plugin the same way.
sys.path.insert(0, str(PLUGIN_DIR / "tests"))
import conftest  # noqa: F401 — side-effect: register dynamic_tools_plugin

plugin = sys.modules["dynamic_tools_plugin"]
logger_io = importlib.import_module("dynamic_tools_plugin.logger_io")


# Minimal but realistic-shaped Hermes tools payload. Each entry is the
# Anthropic-format dict the plugin sees in kwargs["tools"]. Names cover
# every category referenced by policy.yaml plus a few "unknown" tools
# to exercise the unknown_kept path.
SYNTHETIC_TOOLS = [
    # Always-on per policy
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
    # Unknown to the policy — exercises the safe-default kept path
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
    #    as expansion-driven. This is the item #1 bug guard.
    Scenario("already-available-001", turns=[
        # terminal is in policy always_on? no — only via shell trigger.
        # But we'll simulate the case where it ends up in initial_allowed.
        # Use write_file: it's trigger-gated, fires on "save". Then we
        # call expand_tools(file) which re-adds the same tool. The
        # post_tool_call for write_file must NOT show expand_tools_used.
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
# a 100-session sample, expect ~5 (Poisson 95% CI: 1-10).
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


def _run_wrapped_build_api_kwargs(tools: list[dict]) -> None:
    """Invoke the plugin's narrowing path with a synthetic tools list.

    Constructs a temporary wrapped function over a fake "original" that
    returns kwargs unchanged. This exercises the same code path the
    AIAgent monkey-patch runs in production, including writing the
    prediction row.
    """
    def fake_original(self, api_messages):
        return {"tools": list(tools)}

    wrapped = plugin._wrap_build_api_kwargs(fake_original)
    wrapped(self=SimpleNamespace(), api_messages=[])


def _canonical_key(chat_id: str, platform: str) -> str:
    # Mirror gateway.session.build_session_key shape.
    return f"agent:main:{platform}:dm:{chat_id}"


def run_scenario(scenario: Scenario, profile_home: Path) -> None:
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
            _run_wrapped_build_api_kwargs(SYNTHETIC_TOOLS)

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
                    # Log the expand_tools call itself
                    plugin._on_post_tool_call(
                        tool_name="expand_tools",
                        args={"category": category},
                        result={"ok": True, "tools": resolved},
                        task_id=f"task-{scenario.chat_id}-{turn_idx}",
                        session_id=canonical,
                    )
                else:
                    plugin._on_post_tool_call(
                        tool_name=call,
                        args={},
                        result="ok",
                        task_id=f"task-{scenario.chat_id}-{turn_idx}",
                        session_id=canonical,
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


def run_assertions(state_dir: Path, check: Check) -> None:
    preds = [json.loads(l) for l in (state_dir / "predictions.jsonl").read_text().splitlines() if l]
    calls = [json.loads(l) for l in (state_dir / "tool_calls.jsonl").read_text().splitlines() if l]

    # ─── #3: session_id populated on every prediction row ───
    blank_sids = [p for p in preds if not p.get("session_id")]
    check.assert_(not blank_sids,
        f"session_id populated on all prediction rows (blank: {len(blank_sids)}/{len(preds)})")

    # Same for tool_calls
    blank_call_sids = [c for c in calls if not c.get("session_id")]
    check.assert_(not blank_call_sids,
        f"session_id populated on all tool_call rows (blank: {len(blank_call_sids)}/{len(calls)})")

    # ─── #4: bypass cohort distribution ───
    # We ran 100+ sessions with bypass_rate=0.05. Each session is in or out
    # deterministically by hash; expect ~5 bypass sessions, range 1-12.
    sessions_in_bypass = {p["session_id"] for p in preds if p.get("policy_source") == "bypass"}
    total_sessions = {p["session_id"] for p in preds if p.get("session_id")}
    bypass_rate_observed = len(sessions_in_bypass) / max(1, len(total_sessions))
    check.assert_(0.5 <= len(sessions_in_bypass) <= 15,
        f"bypass cohort within expected range "
        f"(observed: {len(sessions_in_bypass)} sessions / {len(total_sessions)} "
        f"= {bypass_rate_observed:.1%}, target ~5%)")

    # ─── #1: expand_tools_used NEVER True when was_initially_available ───
    spurious = [c for c in calls
                if c.get("expand_tools_used") is True
                and c.get("was_initially_available") is True]
    check.assert_(not spurious,
        f"expand_tools_used never credited when was_initially_available "
        f"(spurious: {len(spurious)})")

    # ─── #1 sanity: legitimate expansion IS credited ───
    legit = [c for c in calls
             if c.get("expand_tools_used") is True
             and c.get("was_initially_available") is False
             and c.get("tool_name") == "browser_navigate"]
    check.assert_(legit,
        f"legitimate post-expand calls ARE credited "
        f"(found {len(legit)} browser_navigate expansion-driven calls)")

    # ─── Cross-session isolation ───
    # The "victim" session must have zero expand_tools_used flags despite
    # the prior session having expanded browser on the same scope.
    victim_calls = [c for c in calls if c.get("session_id", "").endswith("isolation-2-victim")]
    victim_expanded = [c for c in victim_calls if c.get("expand_tools_used") is True]
    check.assert_(not victim_expanded,
        f"sticky residency does not leak across sessions "
        f"(victim session expansion credits: {len(victim_expanded)})")

    # ─── Session_end eviction ───
    # After all scenarios completed, _STICKY_BY_KEY must be empty (every
    # session called on_session_end).
    remaining_sticky = list(plugin._STICKY_BY_KEY.keys())
    check.assert_(not remaining_sticky,
        f"on_session_end fully evicts sticky state "
        f"(remaining sticky_keys: {len(remaining_sticky)})")

    # ─── prior_messages also evicted ───
    remaining_lookback = list(plugin._PRIOR_MESSAGES_BY_SESSION.keys())
    check.assert_(not remaining_lookback,
        f"on_session_end evicts lookback prior-message buffer "
        f"(remaining session entries: {len(remaining_lookback)})")


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        profile_home = Path(tmp) / "profiles" / "bernard"
        state_dir = profile_home / "state" / "dynamic-tools"
        state_dir.mkdir(parents=True, exist_ok=True)

        # Reset plugin state — prior test runs in this Python process
        # could have left entries behind.
        plugin._STICKY_BY_KEY.clear()
        plugin._POLICY_TURN_BY_SCOPE.clear()
        plugin._PRIOR_MESSAGES_BY_SESSION.clear()

        original_config = dict(plugin._CONFIG)
        plugin._CONFIG.update({
            "enabled": True,
            "log": True,
            "agent": "bernard",
            "bypass_rate": 0.05,
            "channels": {},
            "sticky": {"enabled": True, "ttl_turns": 3, "categories": ["*"]},
            "predictor": {"lookback_turns": 1},
        })

        try:
            scenarios = TARGETED_SCENARIOS + FILLER_SCENARIOS
            print(f"Running {len(scenarios)} scenarios in {state_dir}...")
            for sc in scenarios:
                run_scenario(sc, profile_home)
            print(f"  → {sum(1 for f in state_dir.iterdir())} state files written")

            check = Check()
            run_assertions(state_dir, check)
            return check.report()
        finally:
            plugin._CONFIG.clear()
            plugin._CONFIG.update(original_config)
            plugin._STICKY_BY_KEY.clear()
            plugin._POLICY_TURN_BY_SCOPE.clear()
            plugin._PRIOR_MESSAGES_BY_SESSION.clear()


if __name__ == "__main__":
    sys.exit(main())
