#!/usr/bin/env python3
"""Hermes Tool Belt — headline savings report.

One report, two honest numbers: cache-on and cache-off. Baseline is
the full platform_toolsets ceiling from config — the tools the agent
COULD have shipped — vs what Tool Belt actually shipped.

Run:
  python3 scripts/savings-report.py                       # all scopes
  python3 scripts/savings-report.py --scope bernard:telegram
  python3 scripts/savings-report.py --json                # machine-readable
  python3 scripts/savings-report.py --since 2026-05-15    # time-bounded

Reads:
  $HERMES_HOME/state/tool-belt/predictions.jsonl
  $HERMES_HOME/state/tool-belt/api_calls.jsonl  (for cache mode + cache_read tokens)
  $HERMES_HOME/state/tool-belt/tool_calls.jsonl (for cron/subagent exclusion counts)

Methodology — see docs/SAVINGS.md for the long version.

  TOKENS SAVED: ceiling_tokens - narrowed_tokens, summed across every
                logged prediction row. This is the count of tool-schema
                tokens Tool Belt KEPT OUT of the request — the model
                never saw them, the provider never billed them.

  CACHE AMORTIZATION (cache-on only): cache_read_tokens reported by
                the provider. These tokens ARE shipped but billed at
                the cache-hit rate (~10% on Anthropic, varies on OpenAI).
                Tool Belt enables this by holding the tool prefix stable
                so the cache doesn't get busted.

  EXCLUDED: cron and subagent calls never go through the narrowing
                pipeline. They're reported as "excluded" so the savings
                figures stay honest.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def default_state_dir() -> Path:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(home) / "state" / "tool-belt"


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


def parse_since(s: str | None) -> float:
    if not s:
        return 0.0
    try:
        return dt.datetime.fromisoformat(s).timestamp()
    except Exception:
        try:
            return dt.datetime.strptime(s, "%Y-%m-%d").timestamp()
        except Exception:
            return 0.0


def last_api_call_by_prediction(api_calls: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """For each prediction_id, return the last (highest api_call_idx) API call.

    The last call's cache_mode reflects the detection state machine's
    most-evolved view for this turn — by then "pending" has usually
    resolved into "on" or "off".
    """
    by_pred: dict[str, dict[str, Any]] = {}
    for row in api_calls:
        pid = str(row.get("prediction_id") or "")
        if not pid:
            continue
        idx = int(row.get("api_call_idx") or 0)
        cur = by_pred.get(pid)
        if cur is None or idx > int(cur.get("api_call_idx") or 0):
            by_pred[pid] = row
    return by_pred


def load_detection_cache(state_dir: Path) -> dict[str, Any]:
    """Read the persisted cache-mode detection cache, if any."""
    path = state_dir / "cache_mode_detection.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def aggregate_api_call_totals(api_calls: list[dict[str, Any]],
                              pred_ids: set[str]) -> dict[str, int]:
    """Sum cache/input tokens across API calls linked to the given prediction set."""
    totals = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0, "n_calls": 0}
    for row in api_calls:
        pid = str(row.get("prediction_id") or "")
        if pid not in pred_ids:
            continue
        totals["input"] += int(row.get("input_tokens") or 0)
        totals["cache_read"] += int(row.get("cache_read_tokens") or 0)
        totals["cache_write"] += int(row.get("cache_write_tokens") or 0)
        totals["output"] += int(row.get("output_tokens") or 0)
        totals["n_calls"] += 1
    return totals


def count_tool_call_sources(tool_calls: list[dict[str, Any]]) -> dict[str, int]:
    """Count tool_call rows by source. Older rows may not have the field —
    fall back to session_id heuristic to keep historical counts honest.
    """
    counts: dict[str, int] = defaultdict(int)
    for row in tool_calls:
        src = row.get("source")
        if src in ("gateway", "cron", "subagent"):
            counts[src] += 1
            continue
        # Fallback for rows written before the source field existed.
        sid = str(row.get("session_id") or "")
        pid = str(row.get("prediction_id") or "")
        if sid.startswith("cron_"):
            counts["cron"] += 1
        elif pid:
            counts["gateway"] += 1
        else:
            counts["subagent"] += 1
    return dict(counts)


def classify_prediction_mode(p: dict[str, Any],
                             api_last: dict[str, dict[str, Any]]) -> str:
    """Classify one prediction as cache-on, cache-off, or pending.

    Precedence:
      1. The last api_call's cache_mode (most-evolved detection state).
      2. If api_calls didn't record a cache_mode AND frozen_reuse is
         true, treat as "on" (a reuse implies the freeze exists, which
         only happens under cache-on).
    """
    pid = str(p.get("prediction_id") or "")
    last = api_last.get(pid, {})
    mode = str(last.get("cache_mode") or "")
    if mode in ("on", "off", "pending"):
        return mode
    return "on" if p.get("frozen_reuse") else "pending"


def cohort_stats(predictions: list[dict[str, Any]],
                 api_last: dict[str, dict[str, Any]],
                 api_calls: list[dict[str, Any]],
                 mode_filter: str) -> dict[str, Any]:
    """Compute headline savings for one cache-mode cohort within a scope.

    mode_filter is "on", "off", or "pending".
    """
    rows = [p for p in predictions
            if classify_prediction_mode(p, api_last) == mode_filter]

    if not rows:
        return {"n_predictions": 0, "n_sessions": 0}

    sessions = {str(p.get("session_id") or "") for p in rows if p.get("session_id")}
    n_predictions = len(rows)
    n_sessions = len(sessions)
    ceiling_total = sum(int(p.get("ceiling_tokens") or 0) for p in rows)
    narrowed_total = sum(int(p.get("narrowed_tokens") or 0) for p in rows)
    saved_total = ceiling_total - narrowed_total
    ceiling_count_avg = (sum(int(p.get("ceiling_count") or 0) for p in rows) / n_predictions) if n_predictions else 0.0
    narrowed_count_avg = (sum(int(p.get("narrowed_count") or 0) for p in rows) / n_predictions) if n_predictions else 0.0
    reduction_pct = (saved_total / ceiling_total * 100) if ceiling_total else 0.0

    pred_ids = {str(p.get("prediction_id") or "") for p in rows if p.get("prediction_id")}
    api_totals = aggregate_api_call_totals(api_calls, pred_ids)

    out = {
        "n_predictions": n_predictions,
        "n_sessions": n_sessions,
        "ceiling_count_avg": ceiling_count_avg,
        "narrowed_count_avg": narrowed_count_avg,
        "ceiling_tokens_total": ceiling_total,
        "narrowed_tokens_total": narrowed_total,
        "saved_tokens_total": saved_total,
        "saved_tokens_per_turn_avg": (saved_total / n_predictions) if n_predictions else 0,
        "reduction_pct": reduction_pct,
        "api_input_tokens": api_totals["input"],
        "api_cache_read_tokens": api_totals["cache_read"],
        "api_cache_write_tokens": api_totals["cache_write"],
        "api_n_calls": api_totals["n_calls"],
    }
    # Provider-reported cache hit rate (only meaningful for cache-on).
    denom = api_totals["input"] + api_totals["cache_read"] + api_totals["cache_write"]
    out["cache_hit_rate"] = (api_totals["cache_read"] / denom * 100) if denom else 0.0
    return out


def print_text_report(scope: str, on_stats: dict[str, Any], off_stats: dict[str, Any],
                      pending_stats: dict[str, Any],
                      tool_source_counts: dict[str, int], n_expand_events: int,
                      locked_mode: str, locked_reason: str,
                      estimator_breakdown: dict[str, int]) -> None:
    width = 70
    line = "─" * width

    print()
    print("═" * width)
    print(f"  Hermes Tool Belt — Savings Report".ljust(width))
    print("═" * width)
    print(f"  Scope: {scope}")
    if locked_mode:
        reason = f" ({locked_reason})" if locked_reason else ""
        print(f"  Cache-mode lock: {locked_mode.upper()}{reason}")
    if estimator_breakdown:
        total = sum(estimator_breakdown.values())
        primary = max(estimator_breakdown, key=estimator_breakdown.get)
        primary_pct = (estimator_breakdown[primary] / total * 100) if total else 0
        est_label = {
            "tiktoken-cl100k": "tiktoken-cl100k (BPE — exact for GPT-family, ~5% off Claude)",
            "chars-div-4": "chars/4 (heuristic — install tiktoken for exact counts)",
        }.get(primary, primary)
        print(f"  Token estimator: {est_label}  ({primary_pct:.0f}% of rows)")
        print(f"  (For provider-billed truth see api_calls.jsonl:input_tokens)")
    print()

    # CACHE-ON cohort
    print(f"  ┌{line}┐")
    print(f"  │  CACHE-ON{' ' * (width - 12)}│")
    if on_stats.get("n_predictions"):
        print(f"  │     ({on_stats['n_predictions']} predictions across {on_stats['n_sessions']} session(s))".ljust(width + 3) + "│")
        print(f"  │{' ' * width}│")
        print(f"  │  Tools shipped:   {on_stats['narrowed_count_avg']:>5.1f}  / {on_stats['ceiling_count_avg']:>5.1f} ceiling  ({on_stats['reduction_pct']:>5.1f}% reduction)".ljust(width + 3) + "│")
        print(f"  │  Tokens saved:    {on_stats['saved_tokens_per_turn_avg']:>5,.0f}  per turn (avg)".ljust(width + 3) + "│")
        print(f"  │  Tokens saved:    {on_stats['saved_tokens_total']:>5,}  total  (vs ceiling)".ljust(width + 3) + "│")
        print(f"  │{' ' * width}│")
        print(f"  │  Cache amortization (provider-reported):".ljust(width + 3) + "│")
        print(f"  │    Cache hit rate:      {on_stats['cache_hit_rate']:>5.1f}%".ljust(width + 3) + "│")
        print(f"  │    Cache-read tokens:  {on_stats['api_cache_read_tokens']:>10,}  (billed at cache rate)".ljust(width + 3) + "│")
        print(f"  │    Fresh input tokens: {on_stats['api_input_tokens']:>10,}  (billed at full rate)".ljust(width + 3) + "│")
    else:
        print(f"  │     (no cache-on predictions in window)".ljust(width + 3) + "│")
    print(f"  └{line}┘")
    print()

    # CACHE-OFF cohort
    print(f"  ┌{line}┐")
    print(f"  │  CACHE-OFF{' ' * (width - 13)}│")
    if off_stats.get("n_predictions"):
        print(f"  │     ({off_stats['n_predictions']} predictions across {off_stats['n_sessions']} session(s))".ljust(width + 3) + "│")
        print(f"  │{' ' * width}│")
        print(f"  │  Tools shipped:   {off_stats['narrowed_count_avg']:>5.1f}  / {off_stats['ceiling_count_avg']:>5.1f} ceiling  ({off_stats['reduction_pct']:>5.1f}% reduction)".ljust(width + 3) + "│")
        print(f"  │  Tokens saved:    {off_stats['saved_tokens_per_turn_avg']:>5,.0f}  per turn (avg)".ljust(width + 3) + "│")
        print(f"  │  Tokens saved:    {off_stats['saved_tokens_total']:>5,}  total  (vs ceiling)".ljust(width + 3) + "│")
        if n_expand_events:
            overhead = n_expand_events * 1500
            net = off_stats["saved_tokens_total"] - overhead
            print(f"  │  expand_tools overhead: −{overhead:,}  ({n_expand_events} events × 1500)".ljust(width + 3) + "│")
            print(f"  │  Net savings:           {net:,}".ljust(width + 3) + "│")
    else:
        print(f"  │     (no cache-off predictions in window)".ljust(width + 3) + "│")
    print(f"  └{line}┘")
    print()

    # Pending cohort (detection still resolving)
    if pending_stats.get("n_predictions"):
        print(f"  ┌{line}┐")
        print(f"  │  PENDING (cache-mode detection in progress)".ljust(width + 3) + "│")
        print(f"  │     ({pending_stats['n_predictions']} predictions across {pending_stats['n_sessions']} session(s))".ljust(width + 3) + "│")
        print(f"  │{' ' * width}│")
        print(f"  │  Tokens saved:    {pending_stats['saved_tokens_total']:>5,}  total  (will be re-classified once locked)".ljust(width + 3) + "│")
        print(f"  └{line}┘")
        print()

    # Excluded
    cron_n = tool_source_counts.get("cron", 0)
    sub_n = tool_source_counts.get("subagent", 0)
    gate_n = tool_source_counts.get("gateway", 0)
    if cron_n or sub_n:
        print(f"  ┌{line}┐")
        print(f"  │  Excluded from savings (not subject to narrowing)".ljust(width + 3) + "│")
        print(f"  │{' ' * width}│")
        if cron_n:
            print(f"  │    Cron tool calls:     {cron_n:>5}  (bypass pre_gateway_dispatch)".ljust(width + 3) + "│")
        if sub_n:
            print(f"  │    Subagent calls:      {sub_n:>5}  (inherit parent ceiling)".ljust(width + 3) + "│")
        print(f"  │    Gateway tool calls:  {gate_n:>5}  (subject to narrowing)".ljust(width + 3) + "│")
        print(f"  └{line}┘")
        print()

    print("  Methodology: docs/SAVINGS.md")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-dir", type=Path, default=default_state_dir())
    parser.add_argument("--scope", default="", help="filter to a single scope (default: all)")
    parser.add_argument("--since", default="", help="only count rows since this date (YYYY-MM-DD)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    since_ts = parse_since(args.since)
    predictions = [p for p in load_jsonl(args.state_dir / "predictions.jsonl")
                   if float(p.get("ts") or 0) >= since_ts]
    api_calls = [a for a in load_jsonl(args.state_dir / "api_calls.jsonl")
                 if float(a.get("ts") or 0) >= since_ts]
    tool_calls = [t for t in load_jsonl(args.state_dir / "tool_calls.jsonl")
                  if float(t.get("ts") or 0) >= since_ts]

    if not predictions:
        print(f"No predictions.jsonl rows under {args.state_dir}. Nothing to report.",
              file=sys.stderr)
        return 1

    # Group predictions by scope.
    preds_by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in predictions:
        scope = str(p.get("scope") or "unknown")
        if args.scope and scope != args.scope:
            continue
        preds_by_scope[scope].append(p)

    if not preds_by_scope:
        print(f"No predictions match scope filter {args.scope!r}.", file=sys.stderr)
        return 1

    api_last = last_api_call_by_prediction(api_calls)
    detection_cache = load_detection_cache(args.state_dir)

    # Tool-call source counts (per scope).
    tc_by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in tool_calls:
        scope = str(t.get("scope") or "unknown")
        tc_by_scope[scope].append(t)

    # expand_tools count per scope (for cache-off overhead).
    expand_by_scope: dict[str, int] = defaultdict(int)
    for t in tool_calls:
        if t.get("tool_name") == "expand_tools":
            expand_by_scope[str(t.get("scope") or "unknown")] += 1

    json_report = {"scopes": {}}
    for scope, preds in sorted(preds_by_scope.items()):
        on_stats = cohort_stats(preds, api_last, api_calls, mode_filter="on")
        off_stats = cohort_stats(preds, api_last, api_calls, mode_filter="off")
        pending_stats = cohort_stats(preds, api_last, api_calls, mode_filter="pending")
        tc_counts = count_tool_call_sources(tc_by_scope.get(scope, []))
        n_expand = expand_by_scope.get(scope, 0)
        locked = detection_cache.get(scope, {}) if isinstance(detection_cache, dict) else {}
        locked_mode = str(locked.get("mode") or "")
        locked_reason = str(locked.get("lock_reason") or "")

        # Per-row token estimator provenance. Rows written before this
        # field existed default to "chars-div-4" so historical data stays
        # interpretable.
        est_counts: dict[str, int] = defaultdict(int)
        for p in preds:
            est_counts[str(p.get("tokens_estimator") or "chars-div-4")] += 1

        if args.json:
            json_report["scopes"][scope] = {
                "cache_on": on_stats,
                "cache_off": off_stats,
                "pending": pending_stats,
                "tool_call_source_counts": tc_counts,
                "expand_tools_events": n_expand,
                "locked_mode": locked_mode,
                "locked_reason": locked_reason,
                "token_estimators": dict(est_counts),
            }
        else:
            print_text_report(scope, on_stats, off_stats, pending_stats, tc_counts,
                              n_expand, locked_mode, locked_reason, dict(est_counts))

    if args.json:
        print(json.dumps(json_report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
