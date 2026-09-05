#!/usr/bin/env python3
"""Measure tool-list stability per session from API-call telemetry and
estimate the cache cost of every tool-list mutation.

What it measures
================

For each session, the first call's ``tool_list_hash`` is the baseline; every
later call is classified as matching it or as a mutation (further split into
expand-driven / trigger-driven / avoidable). This is a pure read of what
went over the wire.

How to read it per posture:

  · Caching provider → carry-all. The baseline IS the enabled ceiling
    (minus ``expand_tools``, which is not shipped) and the list is never
    touched again for the session's life. Every mutation counter should be
    zero by construction, and ``matches_stable`` should equal
    ``comparable_calls``. A non-zero mutation on a caching scope is a
    REGRESSION of the carry-all contract — this script is the cheapest way
    to catch one (``--scope`` a caching scope and expect zeros).

  · Non-caching provider → per-turn narrowing. Mutations are expected and
    cheap: there is no prefix cache to bust, so the counters describe the
    narrowing activity (explicit expansions, trigger activations) rather
    than a cost. The cache-cost figures below are meaningless there and
    will read ~0 because ``cache_read_tokens`` is 0 on those calls.

"First-call hash" is the tool-list hash of a session's first call; a
session is *stable* when every later call carries that same hash. (This
diagnostic and its keys were once named for a per-session *freeze* policy
that has since been replaced by the carry-all posture; the mechanism it
measures — tool-list stability against the first-call hash — is unchanged.)

Why the matched counterfactual matters
======================================

The ``tokens_saved`` ledger (predictions.jsonl) reports schema-token savings
without netting out the cache-miss penalty that a tool-list mutation imposes
on the conversation history prefix. On caching providers that penalty can
dominate the savings on any session past a handful of turns — which is why
those scopes carry everything and never mutate.

The corrected counterfactual:

  · For each *mutated* call, the "what cache would I have gotten" baseline
    isn't the global stable-call average — that's biased high because
    stable calls cluster at deep-loop positions where the cache is warm.
    Instead, we match on ``api_call_idx`` within session: a mutated call
    at idx=K is compared against the stable cohort at idx=K. Removes
    most of the position bias.

  · We report ``cache_read_tokens_lost_upper_bound`` rather than a point
    estimate. Signed differences preserve legitimate cache refreshes instead
    of forcing every mutation to look costly.

``analyze.py`` imports this module for its cache-aware savings section; the
CLI below is an optional focused diagnostic.

Usage:
  python3 cache_replay.py
  python3 cache_replay.py --scope assistant-a:telegram
  python3 cache_replay.py --markdown        # markdown for the report dir
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Every telemetry row is read through the centralized normalizer so the replay
# sees one canonical shape (``hermes_session_id`` grouping,
# ``trigger_activated_tools``, ``activation_source``). Pricing is single-sourced in the canonical savings engine;
# it is imported rather than copied (see docs/SAVINGS.md — no duplicate price
# tables). Works both as the ``tool_belt_plugin.cache_replay`` module and as a
# directly-executed script (sibling on sys.path).
try:  # package / test import
    from .logger_io import normalize_prediction_row, normalize_tool_call_row  # type: ignore[import-not-found]
    from .savings import price_for  # type: ignore[import-not-found]
except ImportError:  # script mode
    from logger_io import normalize_prediction_row, normalize_tool_call_row  # type: ignore[no-redef]
    from savings import price_for  # type: ignore[no-redef]


def _session_key(row: dict[str, Any]) -> str:
    """Distinct-session key matching the shaper/analyzer semantics.

    ``hermes_session_id`` rotates on ``/new``, so pre- and post-reset turns
    land in *different* cohorts and a stale baseline hash can never pollute a
    fresh session.
    """
    return str(row.get("hermes_session_id") or "")


def default_state_dir() -> Path:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(home) / "state" / "tool-belt"


# ─── Matched counterfactual ───────────────────────────────────────────────
#
# ``price_for`` is imported from ``savings`` above (the single source of truth
# for per-model token economics). Unknown models fall back to "generic" there.


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


def stability_simulation(
    preds: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]] | None = None,
    scope_filter: str = "",
) -> dict[str, Any]:
    """For each session, take the first call's ``tool_list_hash`` as the
    baseline and classify every subsequent call: did the wire-level tool
    list stay identical to that baseline, or mutate?

    Sessions are grouped by :func:`_session_key` (``hermes_session_id``) so a
    ``/new`` reset starts a fresh cohort instead of polluting the baseline
    hash across the reset boundary.

    Distinguishes four call types:
      · matches_stable    — identical to the baseline: hit the cached prefix
      · expand_driven     — hash differs and the turn shows an ``expand_tools``
                            call, so the mutation is an explicit expansion
                            (model-driven; only possible under narrowing,
                            where ``expand_tools`` is shipped)
      · trigger_driven    — hash differs with no expansion, but the turn newly
                            fired a trigger that activated a previously
                            expand_only tool (policy-driven; narrowing only)
      · would_break       — hash differs with neither an expansion nor a new
                            trigger activation — an unexplained mutation

    On a caching (carry-all) scope all three mutation counters must be zero:
    the baseline is the full ceiling, ``expand_tools`` is not shipped, and
    the plugin never touches the list mid-session. Any non-zero count there
    is a regression, not an accepted cost. On a narrowing scope the counters
    are descriptive: a "mutation" costs nothing where there is no prefix
    cache, and expanded/triggered tools persist for the session, so repeated
    use after the initial mutation does not require repeated mutations.
    """
    # Canonicalize rows through the central adapter.
    preds = [normalize_prediction_row(p) for p in preds]
    calls = [normalize_tool_call_row(c) for c in calls]
    if tool_calls is not None:
        tool_calls = [normalize_tool_call_row(t) for t in tool_calls]

    if scope_filter:
        preds = [p for p in preds if p.get("scope") == scope_filter]
        # Rows missing ``prediction_id`` are tolerated (malformed telemetry);
        # they simply can't join against calls.
        pred_ids = {str(p.get("prediction_id") or "") for p in preds}
        pred_ids.discard("")
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

    # Map prediction_id -> {turn_idx, session_key, scope, expand_called,
    # trigger_driven_mutation}. turn_idx is the prediction's position within its
    # session, ordered by ts. A turn is a *trigger-driven mutation* only the
    # first time it activates a trigger tool not already accumulated for that
    # session (re-firing the same trigger reuses the enlarged set, no new
    # mutation) — mirroring the runtime's ``trigger_driven_mutation`` semantics.
    pred_meta: dict[str, dict[str, Any]] = {}
    sess_preds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in preds:
        sess_preds[_session_key(p)].append(p)
    for sid, plist in sess_preds.items():
        plist.sort(key=lambda p: p.get("ts") or 0)
        seen_trigger_tools: set[str] = set()
        for idx, p in enumerate(plist):
            pid = str(p.get("prediction_id") or "")
            activated = {str(t) for t in (p.get("trigger_activated_tools") or [])}
            new_trigger_tools = activated - seen_trigger_tools
            seen_trigger_tools |= activated
            if not pid:
                # Tolerated: the turn still advances the session's trigger
                # accumulator, but nothing can key off it.
                continue
            pred_meta[pid] = {
                "turn_idx": idx,
                "session_key": sid,
                "scope": p.get("scope", ""),
                "expand_called_this_turn": pid in preds_with_expand,
                "trigger_driven_mutation": bool(new_trigger_tools),
            }

    # Group calls by session key, sort by (ts, api_call_idx).
    sess_calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in calls:
        sess_calls[_session_key(c)].append(c)
    for sid in sess_calls:
        sess_calls[sid].sort(key=lambda c: (c.get("ts") or 0, c.get("api_call_idx") or 0))

    matches = 0
    expand_driven = 0
    trigger_driven = 0
    would_break = 0
    first_calls = 0
    cached_on_match: list[int] = []
    cached_on_break: list[int] = []

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
            # Hash differs. Classify the mutation: explicit expansion first,
            # then trigger activation, else an avoidable cache break.
            pid = c.get("prediction_id", "")
            meta = pred_meta.get(pid, {})
            if meta.get("expand_called_this_turn"):
                expand_driven += 1
            elif meta.get("trigger_driven_mutation"):
                trigger_driven += 1
            else:
                would_break += 1
                cached_on_break.append(cached)

    def avg(xs: list[int]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    total_mutations = expand_driven + trigger_driven + would_break
    comparable = matches + total_mutations
    return {
        "scope_filter": scope_filter or "(all)",
        "sessions": len(sess_calls),
        "predictions": len(preds),
        "api_calls": len(calls),
        "first_calls_per_session": first_calls,
        "comparable_calls": comparable,
        "matches_stable": matches,
        "expand_driven_mutations": expand_driven,
        "trigger_driven_mutations": trigger_driven,
        "would_break_mutations": would_break,
        "mutation_rate_today": (
            total_mutations / comparable if comparable else 0.0
        ),
        # Share of mutations that are unexplained (neither an explicit
        # expansion nor a trigger activation). Historical key name — kept
        # because analyze.py reads it — but on a carry-all scope the whole
        # numerator AND denominator should be zero.
        "unexplained_mutation_share": (
            would_break / total_mutations if total_mutations else 0.0
        ),
        "avg_cache_read_when_matches": round(avg(cached_on_match), 1),
        "avg_cache_read_when_would_break": round(avg(cached_on_break), 1),
    }


def matched_counterfactual(
    calls: list[dict[str, Any]],
    scope_filter: str = "",
) -> dict[str, Any]:
    """Compute a position-matched cache-savings correction.

    For each *mutated* call (hash differs from the prior call in the
    same session), compute the counterfactual cache_read_tokens using
    the stable cohort's *position-matched* average for that
    ``api_call_idx`` bucket and model. Difference = lost cache reads,
    reported as an upper bound. Signed differences allow a mutation to
    coincide with a legitimate cache refresh.

    Per-model dollar estimates use the price table; report token-level
    numbers alongside since the dollar conversion depends on list
    prices that drift.
    """
    calls = [normalize_tool_call_row(c) for c in calls]
    if scope_filter:
        calls = [c for c in calls if c.get("scope") == scope_filter]
    if not calls:
        return {"scope_filter": scope_filter or "(all)", "comparable": 0}

    # Group by session key (hermes_session_id), sort
    sess_calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in calls:
        sess_calls[_session_key(c)].append(c)
    for sid in sess_calls:
        sess_calls[sid].sort(key=lambda c: (c.get("ts") or 0, c.get("api_call_idx") or 0))

    # Build idx-matched cohorts: (model, api_call_idx) → list of cache_read on stable calls
    stable_cohort: dict[tuple[str, int], list[int]] = defaultdict(list)
    classified: list[dict[str, Any]] = []  # rows we'll later compare against the cohort
    for sid, cs in sess_calls.items():
        prev_hash: str | None = None
        for c in cs:
            h = c.get("tool_list_hash") or ""
            model = c.get("model", "")
            # ``or 0`` (not a ``get`` default): JSONL telemetry can carry an
            # explicit null for any of these numeric fields.
            idx = int(c.get("api_call_idx") or 0)
            cache_read = int(c.get("cache_read_tokens") or 0)
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
        "stable_eligible": 0,
        "expand_caused_mutations": 0,
    })

    for entry in classified:
        c = entry["call"]
        kind = entry["kind"]
        model = c.get("model", "generic")
        m = per_model[model]
        m["calls"] += 1
        m["cache_read_actual"] += int(c.get("cache_read_tokens") or 0)
        m["input_actual"] += int(c.get("input_tokens") or 0)
        if kind == "first_call":
            m["first"] += 1
        elif kind == "stable":
            m["stable"] += 1
        else:
            m["mutated"] += 1
            idx = int(c.get("api_call_idx") or 0)
            cohort = stable_cohort.get((model, idx)) or []
            if cohort:
                cohort_mean = sum(cohort) / len(cohort)
                # Upper bound — signed difference, no floor at zero.
                # Negative = mutation luck (cache happened to refresh).
                lost = int(cohort_mean - int(c.get("cache_read_tokens") or 0))
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
    out.append("# Cache-Aware Replay Report\n")
    out.append(f"_scope: {result['scope_filter']}_\n")
    out.append("## Tool-list stability (vs. each session's first-call hash)\n")
    out.append(f"- sessions: {result['sessions']}, predictions: {result['predictions']}, api_calls: {result['api_calls']}")
    out.append(f"- first-call-per-session excluded: {result['first_calls_per_session']}")
    out.append(f"- comparable calls: {result['comparable_calls']}")
    out.append(f"- **matches first-call hash: {result['matches_stable']}** (avg cache_read: {result['avg_cache_read_when_matches']:,.0f})")
    out.append(f"- expand-driven mutations: {result['expand_driven_mutations']} (explicit expansion — narrowing scopes only)")
    out.append(f"- trigger-driven mutations: {result['trigger_driven_mutations']} (trigger activation — narrowing scopes only)")
    out.append(f"- **would_break mutations: {result['would_break_mutations']}** (unexplained; avg cache_read: {result['avg_cache_read_when_would_break']:,.0f})")
    out.append(f"- unexplained share of mutations: **{result['unexplained_mutation_share'] * 100:.1f}%**")
    out.append("- on a caching (carry-all) scope every mutation counter must be 0 — anything else is a regression\n")
    out.append("## Cache-adjusted savings (matched counterfactual)\n")
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
    ap.add_argument("--scope", default="", help="filter to a specific scope, e.g. assistant-a:telegram")
    ap.add_argument("--markdown", action="store_true", help="emit markdown report")
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    preds = load_jsonl(state_dir / "predictions.jsonl")
    calls = load_jsonl(state_dir / "api_calls.jsonl")
    tool_calls = load_jsonl(state_dir / "tool_calls.jsonl")
    if not calls:
        print(f"No api_calls.jsonl rows under {state_dir}. Run more sessions first.", file=sys.stderr)
        return 1

    result = stability_simulation(preds, calls, tool_calls=tool_calls, scope_filter=args.scope)
    cf = matched_counterfactual(calls, scope_filter=args.scope)

    if args.markdown:
        print(render_markdown(result, cf))
        return 0

    print(f"cache-stability replay — tool-list stability  (scope: {result['scope_filter']})")
    print(f"  sessions={result['sessions']}  predictions={result['predictions']}  api_calls={result['api_calls']}")
    print(f"  first-call-per-session (excluded): {result['first_calls_per_session']}")
    print(f"  comparable calls: {result['comparable_calls']}")
    print()
    print("  Observed behavior:")
    print(f"    mutation rate (any cause): {result['mutation_rate_today'] * 100:.1f}%")
    print()
    print("  Against each session's first-call hash:")
    print(f"    matches first-call hash:    {result['matches_stable']:>4} calls  (cached avg: {result['avg_cache_read_when_matches']:,.0f})")
    print(f"    expand-driven mutations:    {result['expand_driven_mutations']:>4} calls  (explicit expansion — narrowing only)")
    print(f"    trigger-driven mutations:   {result['trigger_driven_mutations']:>4} calls  (trigger activation — narrowing only)")
    print(f"    would_break mutations:      {result['would_break_mutations']:>4} calls  (unexplained; cached avg: {result['avg_cache_read_when_would_break']:,.0f})")
    print()
    print(f"  Unexplained share of mutations: {result['unexplained_mutation_share'] * 100:.1f}%.")
    print("  Caching (carry-all) scope: every mutation counter above must be 0 — a non-zero count is a regression.")
    print("  Narrowing scope: mutations are expected and cost nothing (no prefix cache to bust).")

    if cf.get("per_model"):
        print()
        print("  Cache-adjusted savings (matched counterfactual, position-bucket by api_call_idx):")
        for model, m in sorted(cf["per_model"].items(), key=lambda kv: -int(kv[1]["calls"])):
            print(
                f"    {model:30s}  calls={m['calls']:4d}  mut={m['mutated']:3d}  stable={m['stable']:3d}  "
                f"lost_upper_bound={m['cache_read_lost_upper_bound']:>10,} tok  ≈ ${m['est_usd_lost_upper_bound']:.4f}"
            )
        print("  Methodology: per-call counterfactual = stable-cohort mean at the same api_call_idx for the same model.")
        print("  Numbers are upper bounds (signed). Negative = mutation coincided with a cache refresh.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
