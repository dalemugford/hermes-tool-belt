#!/usr/bin/env python3
"""Replay api_calls.jsonl + predictions.jsonl through the freeze policy
and report what happened — baseline (Phase 0), pre/post comparison
(Phase 1+), and savings-ledger correction (Phase 5).

What this gets right that the legacy savings number didn't
==========================================================

The original ``tokens_saved_via_narrowing`` ledger reports schema-token
savings without netting out the cache-miss penalty that mutation-driven
narrowing imposes on the conversation history prefix. Under cache-on
providers (Anthropic + OpenAI auto-cache, ~80% of our traffic) that
penalty dominates the savings on any session past a handful of turns.

The corrected counterfactual:

  · For each *mutated* call, the "what cache would I have gotten" baseline
    isn't the global stable-call average — that's biased high because
    stable calls cluster at deep-loop positions where the cache is warm.
    Instead, we match on ``api_call_idx`` within session: a mutated call
    at idx=K is compared against the stable cohort at idx=K. Removes
    most of the position bias.

  · We report ``cache_read_tokens_lost_upper_bound`` rather than a
    point estimate — the floor-at-zero in earlier drafts is a one-sided
    estimator (would never report a gain even when a mutation happens
    to coincide with a legitimate cache refresh). Naming the bound
    keeps the methodology honest.

Usage:
  python3 scripts/cache-freeze-replay.py                   # baseline + corrected savings
  python3 scripts/cache-freeze-replay.py --scope bernard:telegram
  python3 scripts/cache-freeze-replay.py --markdown        # markdown for the report dir
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
    return Path(home) / "state" / "tool-belt"


# ─── Phase 5: price table + counterfactual ────────────────────────────────
#
# Single source of truth for per-model token economics. Tokens-per-million
# costs in USD; ``miss_premium`` is the input/cache_read ratio that drives
# the savings correction. Unknown models fall back to "generic" — the
# Anthropic Sonnet ratio (~10×) is the right default because (a) it's the
# providers' published ratio for most modern caches and (b) erring high
# keeps the corrected savings honest.
PRICE_TABLE: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-sonnet-4-6": {"input": 3.00, "cache_read": 0.30, "cache_write": 3.75, "output": 15.00, "miss_premium": 10.0},
    "claude-haiku-4-5-20251001": {"input": 1.00, "cache_read": 0.10, "cache_write": 1.25, "output": 5.00, "miss_premium": 10.0},
    # OpenAI Codex (via subscription — list prices for ratio computation)
    "gpt-5.4": {"input": 1.25, "cache_read": 0.125, "cache_write": 1.25, "output": 10.00, "miss_premium": 10.0},
    "gpt-5.4-mini": {"input": 0.15, "cache_read": 0.075, "cache_write": 0.15, "output": 0.60, "miss_premium": 2.0},
    "gpt-5.5": {"input": 2.50, "cache_read": 0.25, "cache_write": 2.50, "output": 10.00, "miss_premium": 10.0},
    # Kimi (Ollama Cloud) — no provider-side prefix caching to break
    "kimi-k2.6:cloud": {"input": 0.0, "cache_read": 0.0, "cache_write": 0.0, "output": 0.0, "miss_premium": 1.0},
    "generic": {"input": 1.0, "cache_read": 0.1, "cache_write": 1.0, "output": 5.0, "miss_premium": 10.0},
}


def price_for(model: str) -> dict[str, float]:
    return PRICE_TABLE.get(model, PRICE_TABLE["generic"])


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


def matched_counterfactual(
    calls: list[dict[str, Any]],
    scope_filter: str = "",
) -> dict[str, Any]:
    """Phase 5: position-matched cache savings correction.

    For each *mutated* call (hash differs from the prior call in the
    same session), compute the counterfactual cache_read_tokens using
    the stable cohort's *position-matched* average for that
    ``api_call_idx`` bucket and model. Difference = lost cache reads,
    reported as upper bound (signed differences allowed — negative
    losses indicate gains, which the legacy floor-at-zero estimator
    couldn't surface).

    Per-model dollar estimates use the price table; report token-level
    numbers alongside since the dollar conversion depends on list
    prices that drift.
    """
    if scope_filter:
        calls = [c for c in calls if c.get("scope") == scope_filter]
    if not calls:
        return {"scope_filter": scope_filter or "(all)", "comparable": 0}

    # Group by session, sort
    sess_calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in calls:
        sess_calls[c.get("session_id", "")].append(c)
    for sid in sess_calls:
        sess_calls[sid].sort(key=lambda c: (c.get("ts", 0), c.get("api_call_idx", 0)))

    # Build idx-matched cohorts: (model, api_call_idx) → list of cache_read on stable calls
    stable_cohort: dict[tuple[str, int], list[int]] = defaultdict(list)
    classified: list[dict[str, Any]] = []  # rows we'll later compare against the cohort
    for sid, cs in sess_calls.items():
        prev_hash: str | None = None
        for c in cs:
            h = c.get("tool_list_hash") or ""
            model = c.get("model", "")
            idx = int(c.get("api_call_idx", 0))
            cache_read = int(c.get("cache_read_tokens", 0))
            if prev_hash is None:
                kind = "first_call"
            elif h == prev_hash:
                kind = "stable"
                stable_cohort[(model, idx)].append(cache_read)
            else:
                kind = "mutated"
            classified.append({"call": c, "kind": kind})
            prev_hash = h

    # Aggregate per-model corrected savings
    per_model: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "calls": 0,
        "mutated": 0,
        "stable": 0,
        "first": 0,
        "cache_read_lost_upper_bound": 0,
        "cache_read_actual": 0,
        "input_actual": 0,
        "freeze_eligible": 0,
        "expand_caused_mutations": 0,
    })

    for entry in classified:
        c = entry["call"]
        kind = entry["kind"]
        model = c.get("model", "generic")
        m = per_model[model]
        m["calls"] += 1
        m["cache_read_actual"] += int(c.get("cache_read_tokens", 0))
        m["input_actual"] += int(c.get("input_tokens", 0))
        if kind == "first_call":
            m["first"] += 1
        elif kind == "stable":
            m["stable"] += 1
        else:
            m["mutated"] += 1
            idx = int(c.get("api_call_idx", 0))
            cohort = stable_cohort.get((model, idx)) or []
            if cohort:
                cohort_mean = sum(cohort) / len(cohort)
                # Upper bound — signed difference, no floor at zero.
                # Negative = mutation luck (cache happened to refresh).
                lost = int(cohort_mean - int(c.get("cache_read_tokens", 0)))
                m["cache_read_lost_upper_bound"] += lost

    # Dollar-equivalent (best-effort) — per the price table
    for model, m in per_model.items():
        prices = price_for(model)
        # The cost of a lost cache token = (input_price - cache_read_price) / 1M
        lost = m["cache_read_lost_upper_bound"]
        m["est_usd_lost_upper_bound"] = round(
            lost * (prices["input"] - prices["cache_read"]) / 1_000_000.0, 4
        )
        m["miss_premium"] = prices["miss_premium"]

    return {
        "scope_filter": scope_filter or "(all)",
        "total_calls": sum(m["calls"] for m in per_model.values()),
        "per_model": dict(per_model),
    }


def render_markdown(result: dict[str, Any], cf: dict[str, Any]) -> str:
    out: list[str] = []
    out.append(f"# Cache-Aware Replay Report\n")
    out.append(f"_scope: {result['scope_filter']}_\n")
    out.append("## Freeze coverage\n")
    out.append(f"- sessions: {result['sessions']}, predictions: {result['predictions']}, api_calls: {result['api_calls']}")
    out.append(f"- first-call-per-session excluded: {result['first_calls_per_session']}")
    out.append(f"- comparable calls: {result['comparable_calls']}")
    out.append(f"- **matches frozen hash: {result['matches_freeze']}** (avg cache_read: {result['avg_cache_read_when_matches']:,.0f})")
    out.append(f"- expand-driven mutations: {result['expand_driven_mutations']} (accepted — model-paid)")
    out.append(f"- **would_break mutations: {result['would_break_mutations']}** (avg cache_read: {result['avg_cache_read_when_would_break']:,.0f})")
    out.append(f"- freeze eliminates **{result['freeze_eliminates_pct_of_mutations'] * 100:.1f}%** of currently-observed mutations\n")
    out.append("## Cache-adjusted savings (Phase 5 — matched counterfactual)\n")
    out.append("Per-model cache_read_tokens lost to mutation, computed against the stable cohort at the same api_call_idx position within session.\n")
    out.append("| model | calls | mut | stable | cache_read_lost_upper_bound | est_usd_lost |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for model, m in sorted(cf.get("per_model", {}).items(), key=lambda kv: -int(kv[1]["calls"])):
        out.append(f"| `{model}` | {m['calls']} | {m['mutated']} | {m['stable']} | {m['cache_read_lost_upper_bound']:,} | ${m['est_usd_lost_upper_bound']:.4f} |")
    out.append("")
    out.append("Numbers are upper bounds (signed). Negative values mean the mutation happened to coincide with a legitimate cache refresh — not common.\n")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-dir", default=str(default_state_dir()))
    ap.add_argument("--scope", default="", help="filter to a specific scope, e.g. bernard:telegram")
    ap.add_argument("--markdown", action="store_true", help="emit markdown report")
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    preds = load_jsonl(state_dir / "predictions.jsonl")
    calls = load_jsonl(state_dir / "api_calls.jsonl")
    tool_calls = load_jsonl(state_dir / "tool_calls.jsonl")
    if not calls:
        print(f"No api_calls.jsonl rows under {state_dir}. Run more sessions first.", file=sys.stderr)
        return 1

    result = freeze_simulation(preds, calls, tool_calls=tool_calls, scope_filter=args.scope)
    cf = matched_counterfactual(calls, scope_filter=args.scope)

    if args.markdown:
        print(render_markdown(result, cf))
        return 0

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

    if cf.get("per_model"):
        print()
        print(f"  Cache-adjusted savings (matched counterfactual, position-bucket by api_call_idx):")
        for model, m in sorted(cf["per_model"].items(), key=lambda kv: -int(kv[1]["calls"])):
            print(
                f"    {model:30s}  calls={m['calls']:4d}  mut={m['mutated']:3d}  stable={m['stable']:3d}  "
                f"lost_upper_bound={m['cache_read_lost_upper_bound']:>10,} tok  ≈ ${m['est_usd_lost_upper_bound']:.4f}"
            )
        print(f"  Methodology: per-call counterfactual = stable-cohort mean at the same api_call_idx for the same model.")
        print(f"  Numbers are upper bounds (signed). Negative = mutation coincided with a cache refresh.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
