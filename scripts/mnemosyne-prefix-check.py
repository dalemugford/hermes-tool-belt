#!/usr/bin/env python3
"""Mnemosyne A/B verification — does memory injection break the cached prefix?

The freeze in cache-on mode treats ``tool_list_hash`` as a proxy for
cache-prefix stability. That only holds if NOTHING else mutates the prefix
between turns. Mnemosyne is the obvious threat — it fires ``pre_llm_call``
and can inject recalled memory content into the message array.

This script answers: within a session, does the SYSTEM message bytes
stay stable across turns?

  · system_hash stable AND tool_list_hash stable → the cached prefix is
    genuinely stable. Mnemosyne (if injecting) injects below the
    breakpoint — into the most recent user message — and stays cache-
    friendly.
  · system_hash varies → Mnemosyne (or something) is mutating the
    system prompt itself. The freeze headline is partly attributable
    to whatever else is constant, not to dynamic-tools alone. The
    savings story changes shape.

This is a verification gate, not an A/B test. We don't need to disable
Mnemosyne to answer the question; we just need to look at where the
mutation actually lives.

Usage:
  python3 scripts/mnemosyne-prefix-check.py
  python3 scripts/mnemosyne-prefix-check.py --scope bernard:telegram
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def default_state_dir() -> Path:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(home) / "state" / "dynamic-tools"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def analyze(calls: list[dict[str, Any]], scope_filter: str = "") -> dict[str, Any]:
    if scope_filter:
        calls = [c for c in calls if c.get("scope") == scope_filter]

    # Drop pre-system_hash rows. The field landed after Phase 6, so
    # historical rows have no value to compare against.
    calls = [c for c in calls if c.get("system_hash")]
    if not calls:
        return {"calls": 0, "skipped": True}

    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in calls:
        by_session[c.get("session_id", "")].append(c)
    for sid in by_session:
        by_session[sid].sort(key=lambda c: (c.get("ts", 0), c.get("api_call_idx", 0)))

    # Per-session: count distinct system_hashes vs tool_list_hashes.
    # Stability is "1 distinct value seen across all calls".
    sys_stable_sessions = 0
    sys_mutated_sessions = 0
    tools_stable_sessions = 0
    tools_mutated_sessions = 0
    both_stable_sessions = 0
    sys_only_mutated_sessions = 0
    tools_only_mutated_sessions = 0
    both_mutated_sessions = 0
    sys_only_examples: list[tuple[str, set[str], set[str]]] = []  # (sid, sys_hashes, tool_hashes)

    for sid, cs in by_session.items():
        if len(cs) < 2:
            continue
        sys_set = {c.get("system_hash", "") for c in cs if c.get("system_hash")}
        tool_set = {c.get("tool_list_hash", "") for c in cs if c.get("tool_list_hash")}
        sys_stable = len(sys_set) <= 1
        tools_stable = len(tool_set) <= 1
        if sys_stable:
            sys_stable_sessions += 1
        else:
            sys_mutated_sessions += 1
        if tools_stable:
            tools_stable_sessions += 1
        else:
            tools_mutated_sessions += 1
        if sys_stable and tools_stable:
            both_stable_sessions += 1
        elif not sys_stable and tools_stable:
            sys_only_mutated_sessions += 1
            if len(sys_only_examples) < 3:
                sys_only_examples.append((sid, sys_set, tool_set))
        elif sys_stable and not tools_stable:
            tools_only_mutated_sessions += 1
        else:
            both_mutated_sessions += 1

    return {
        "calls": len(calls),
        "sessions_with_2plus_calls": sys_stable_sessions + sys_mutated_sessions,
        "system_stable_sessions": sys_stable_sessions,
        "system_mutated_sessions": sys_mutated_sessions,
        "tools_stable_sessions": tools_stable_sessions,
        "tools_mutated_sessions": tools_mutated_sessions,
        "both_stable_sessions": both_stable_sessions,
        "sys_only_mutated_sessions": sys_only_mutated_sessions,
        "tools_only_mutated_sessions": tools_only_mutated_sessions,
        "both_mutated_sessions": both_mutated_sessions,
        "sys_only_examples": [(sid, sorted(s), sorted(t)) for sid, s, t in sys_only_examples],
    }


def verdict(r: dict[str, Any]) -> str:
    if r.get("skipped"):
        return "INSUFFICIENT_DATA"
    sessions = r["sessions_with_2plus_calls"]
    if sessions == 0:
        return "INSUFFICIENT_DATA"
    sys_mut_rate = r["system_mutated_sessions"] / sessions
    if sys_mut_rate >= 0.10:
        return "MNEMOSYNE_MUTATES_PREFIX"
    return "PREFIX_STABLE"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-dir", default=str(default_state_dir()))
    ap.add_argument("--scope", default="")
    args = ap.parse_args()

    calls = load_jsonl(Path(args.state_dir) / "api_calls.jsonl")
    if not calls:
        print(f"No api_calls.jsonl under {args.state_dir}.", file=sys.stderr)
        return 1

    r = analyze(calls, scope_filter=args.scope)
    if r.get("skipped"):
        print("No api_calls rows have a system_hash yet — feature landed in the Mnemosyne A/B telemetry pass.")
        print("Send 3+ Bernard/Sue messages and re-run.", file=sys.stderr)
        return 1

    print(f"Mnemosyne prefix-stability check  (scope: {args.scope or '(all)'})\n")
    print(f"  api_calls considered: {r['calls']}")
    print(f"  sessions with ≥2 calls: {r['sessions_with_2plus_calls']}\n")
    print(f"  system_hash stable across session: {r['system_stable_sessions']:>4}")
    print(f"  system_hash mutated within session: {r['system_mutated_sessions']:>4}")
    print(f"  tool_list_hash stable across session: {r['tools_stable_sessions']:>4}")
    print(f"  tool_list_hash mutated within session: {r['tools_mutated_sessions']:>4}\n")
    print(f"  both stable:                  {r['both_stable_sessions']:>4}")
    print(f"  system-only mutated:          {r['sys_only_mutated_sessions']:>4}  ← Mnemosyne / system-prompt drift signal")
    print(f"  tools-only mutated:           {r['tools_only_mutated_sessions']:>4}  ← classic narrowing mutation (Phase 1's target)")
    print(f"  both mutated:                 {r['both_mutated_sessions']:>4}\n")

    v = verdict(r)
    if v == "PREFIX_STABLE":
        print("VERDICT: PREFIX_STABLE")
        print("  System message is stable across turns within sessions. If Mnemosyne is injecting")
        print("  memory, it lands in a cache-friendly position (most-recent message). The freeze")
        print("  headline is real — dynamic-tools' freeze is the cache-stability mechanism.")
    elif v == "MNEMOSYNE_MUTATES_PREFIX":
        print("VERDICT: MNEMOSYNE_MUTATES_PREFIX")
        print(f"  {r['sys_only_mutated_sessions']} session(s) show system_hash variation with stable")
        print("  tool_list_hash. Something upstream is mutating the system prompt between turns —")
        print("  most likely Mnemosyne memory injection into the system block. The freeze numbers")
        print("  are PARTIALLY attributable to whatever else stays stable; restate the savings")
        print("  story before relying on the headline.")
        if r["sys_only_examples"]:
            print("\n  Examples:")
            for sid, sys_h, tool_h in r["sys_only_examples"]:
                print(f"    session={sid}")
                print(f"      system_hashes seen ({len(sys_h)}): {sys_h}")
                print(f"      tool_hashes seen   ({len(tool_h)}): {tool_h}")
    else:
        print("VERDICT: INSUFFICIENT_DATA")
        print("  Need ≥3 sessions with multi-call telemetry to draw a conclusion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
