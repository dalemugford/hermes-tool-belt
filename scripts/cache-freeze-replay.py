#!/usr/bin/env python3
"""Replay api_calls.jsonl + predictions.jsonl through the Phase-1 freeze
policy and report what would have happened.

The Phase 0 baseline answers two questions empirically rather than by
hand-wave:

  1. If we had frozen the tool set at session start, how much of the
     currently-observed mutation would have been eliminated?
  2. Of the mutations that would have survived, how many are
     expand_tools-driven (accepted by the pivot — model-paid, single
     break) vs not (cost the freeze policy isn't catching)?

Run this against the live data after Phase 0 ships to lock in a
baseline, and again after Phase 1 ships to verify the predicted drop
actually materialized. Phase 5's savings ledger correction will share
this script's per-call counterfactual machinery.

Usage:
  python3 scripts/cache-freeze-replay.py
  python3 scripts/cache-freeze-replay.py --state-dir /path/to/state/dynamic-tools
  python3 scripts/cache-freeze-replay.py --scope bernard:telegram
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
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


def freeze_simulation(
    preds: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]] | None = None,
    scope_filter: str = "",
) -> dict[str, Any]:
    """For each session, fix the frozen tool-list hash as the first call's
    hash and replay every subsequent call: would it have mutated against
    that frozen baseline?

    Distinguishes three call types:
      · matches_freeze    — would have hit the cached prefix
      · expand_driven     — hash differs but the prior tool_calls.jsonl
                            shows an ``expand_tools`` call in this turn,
                            so the mutation is model-driven (accepted)
      · would_break       — hash differs and no expand_tools precedes —
                            the freeze policy would have prevented this
                            break by holding the tool set steady

    NB: this is a conservative upper bound on accepted-cost. Phase 1's
    sticky-expansion behavior (expanded tools persist for the session)
    means many mutations counted here as "expand_driven" today would
    collapse into a single persistent hash change tomorrow. The Phase 5
    savings ledger refines this with matched counterfactuals.
    """
    if scope_filter:
        preds = [p for p in preds if p.get("scope") == scope_filter]
        pred_ids = {p["prediction_id"] for p in preds}
        calls = [c for c in calls if c.get("prediction_id") in pred_ids]
        if tool_calls:
            tool_calls = [t for t in tool_calls if t.get("prediction_id") in pred_ids]

    # Predictions that observed an expand_tools call. The expansion fires
    # mid-turn, after the prediction row is already snapshotted, so
    # predictions.jsonl's `expanded_tools` field can't be trusted here —
    # tool_calls.jsonl is the authoritative signal.
    preds_with_expand: set[str] = set()
    for t in (tool_calls or []):
        if t.get("tool_name") == "expand_tools":
            pid = t.get("prediction_id", "")
            if pid:
                preds_with_expand.add(pid)

    # Map prediction_id -> {turn_idx, session_id, scope, expand_called}.
    # turn_idx is the prediction's position within its session, ordered by ts.
    pred_meta: dict[str, dict[str, Any]] = {}
    sess_preds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in preds:
        sess_preds[p.get("session_id", "")].append(p)
    for sid, plist in sess_preds.items():
        plist.sort(key=lambda p: p.get("ts", 0))
        for idx, p in enumerate(plist):
            pid = p["prediction_id"]
            pred_meta[pid] = {
                "turn_idx": idx,
                "session_id": sid,
                "scope": p.get("scope", ""),
                "expand_called_this_turn": pid in preds_with_expand,
            }

    # Group calls by session_id, sort by (ts, api_call_idx).
    sess_calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in calls:
        sess_calls[c.get("session_id", "")].append(c)
    for sid in sess_calls:
        sess_calls[sid].sort(key=lambda c: (c.get("ts", 0), c.get("api_call_idx", 0)))

    matches = 0
    expand_driven = 0
    would_break = 0
    first_calls = 0
    cached_on_match: list[int] = []
    cached_on_break: list[int] = []
    expand_break_breakdown: Counter[str] = Counter()

    for sid, cs in sess_calls.items():
        if not cs:
            continue
        frozen_hash = (cs[0].get("tool_list_hash") or "")
        for i, c in enumerate(cs):
            h = c.get("tool_list_hash") or ""
            cached = int(c.get("cache_read_tokens") or 0)
            if i == 0:
                first_calls += 1
                continue
            if h == frozen_hash:
                matches += 1
                cached_on_match.append(cached)
                continue
            # Hash differs. Was this an expand_tools turn?
            pid = c.get("prediction_id", "")
            meta = pred_meta.get(pid, {})
            if meta.get("expand_called_this_turn"):
                expand_driven += 1
                expand_break_breakdown["expand_driven"] += 1
            else:
                would_break += 1
                cached_on_break.append(cached)
                expand_break_breakdown["would_break"] += 1

    def avg(xs: list[int]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    return {
        "scope_filter": scope_filter or "(all)",
        "sessions": len(sess_calls),
        "predictions": len(preds),
        "api_calls": len(calls),
        "first_calls_per_session": first_calls,
        "comparable_calls": matches + expand_driven + would_break,
        "matches_freeze": matches,
        "expand_driven_mutations": expand_driven,
        "would_break_mutations": would_break,
        "mutation_rate_today": (
            (expand_driven + would_break) / (matches + expand_driven + would_break)
            if (matches + expand_driven + would_break) else 0.0
        ),
        "freeze_eliminates_pct_of_mutations": (
            would_break / (expand_driven + would_break)
            if (expand_driven + would_break) else 0.0
        ),
        "avg_cache_read_when_matches": round(avg(cached_on_match), 1),
        "avg_cache_read_when_would_break": round(avg(cached_on_break), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-dir", default=str(default_state_dir()))
    ap.add_argument("--scope", default="", help="filter to a specific scope, e.g. bernard:telegram")
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    preds = load_jsonl(state_dir / "predictions.jsonl")
    calls = load_jsonl(state_dir / "api_calls.jsonl")
    tool_calls = load_jsonl(state_dir / "tool_calls.jsonl")
    if not calls:
        print(f"No api_calls.jsonl rows under {state_dir}. Run more sessions first.", file=sys.stderr)
        return 1

    result = freeze_simulation(preds, calls, tool_calls=tool_calls, scope_filter=args.scope)

    print(f"cache-freeze replay  (scope: {result['scope_filter']})")
    print(f"  sessions={result['sessions']}  predictions={result['predictions']}  api_calls={result['api_calls']}")
    print(f"  first-call-per-session (excluded): {result['first_calls_per_session']}")
    print(f"  comparable calls: {result['comparable_calls']}")
    print()
    print(f"  Today's behavior:")
    print(f"    mutation rate (any cause): {result['mutation_rate_today'] * 100:.1f}%")
    print()
    print(f"  Under session-start freeze (Phase 1):")
    print(f"    matches frozen hash:        {result['matches_freeze']:>4} calls  (cached avg: {result['avg_cache_read_when_matches']:,.0f})")
    print(f"    expand-driven mutations:    {result['expand_driven_mutations']:>4} calls  (accepted — model paid)")
    print(f"    would_break mutations:      {result['would_break_mutations']:>4} calls  (cached avg: {result['avg_cache_read_when_would_break']:,.0f})")
    print()
    print(f"  Freeze eliminates {result['freeze_eliminates_pct_of_mutations'] * 100:.1f}% of currently-observed mutations.")
    print(f"  The remainder is expand_tools-driven and is the accepted cost of the safety valve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
