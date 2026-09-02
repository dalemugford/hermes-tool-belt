#!/usr/bin/env python3
"""Hermes Tool Belt — headline savings report (compatibility wrapper).

DEPRECATED SURFACE. The canonical, supported command is now::

    tool-belt savings            # human report
    tool-belt savings --json     # machine-readable

This script is retained as a thin backward-compatibility wrapper over the
canonical engine in ``savings.py``. All of the observed-cohort math
(``classify_prediction_mode`` / ``cohort_stats`` / cache-mode bucketing) lives
in the engine now and is imported here — there is no second implementation and
no duplicate price table, token math, or expansion-overhead constant. The
per-scope cache-on/off/pending/bypass text layout below is preserved so existing
operators and tests keep working.

Two things to know about the cohorts it prints:

  · CACHE-ON is the carry-all posture: on a caching provider the plugin ships
    the full ceiling (minus ``expand_tools``) and never narrows, so its
    "reduction" is 0 by construction and its value is the provider's cache
    reads, not schema tokens withheld. Nothing is expanded on that cohort.
  · The locked mode shown per scope is the scope's PRIMARY provider's bucket.
    ``cache_mode_detection.json`` is keyed ``"<scope>|<provider>"`` (legacy
    files may still carry a bare ``"<scope>"`` key); the entry is resolved
    through ``shaping.resolve_primary_detection_entry``.

Run:
  python3 scripts/savings-report.py                       # all scopes
  python3 scripts/savings-report.py --scope default:telegram
  python3 scripts/savings-report.py --json                # machine-readable
  python3 scripts/savings-report.py --since 2026-05-15    # time-bounded

Reads (read-only):
  $HERMES_HOME/state/tool-belt/predictions.jsonl
  $HERMES_HOME/state/tool-belt/api_calls.jsonl
  $HERMES_HOME/state/tool-belt/tool_calls.jsonl

Methodology — see docs/SAVINGS.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Import the canonical engine. Inserting the plugin dir on sys.path lets this
# hyphen-named script import both ``savings`` and ``logger_io`` standalone.
_PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PLUGIN_DIR))
import savings as _engine  # noqa: E402
from logger_io import normalize_prediction_row, normalize_tool_call_row  # noqa: E402


def _load_shaping():
    """The shared shaping core — ``tool_belt_plugin.shaping``.

    ``shaping.py`` uses package-relative imports (``from . import learned``),
    so it can't be imported as a bare top-level module the way ``savings`` can
    (whose relative imports are ``TYPE_CHECKING``-guarded). Register the
    hyphenated plugin directory as the ``tool_belt_plugin`` package first —
    mirroring ``scripts/shape-ceiling.py`` / ``tests/conftest.py`` so an
    already-registered module object is reused.
    """
    import importlib
    import importlib.util
    name = "tool_belt_plugin"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, _PLUGIN_DIR / "__init__.py",
            submodule_search_locations=[str(_PLUGIN_DIR)],
        )
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise ImportError("cannot load tool-belt package")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{name}.shaping")


_shaping = _load_shaping()

# ── Backward-compatible re-exports (the engine is the single implementation) ──
default_state_dir = _engine.default_state_dir
load_jsonl = _engine.load_jsonl
parse_since = _engine.parse_since
last_api_call_by_prediction = _engine.last_api_call_by_prediction
classify_prediction_mode = _engine.classify_prediction_mode
cohort_stats = _engine.cohort_stats
# Thin-data FALLBACK for the per-event expansion cost. The engine now
# measures the cost per cohort (``measure_expand_overhead``); this constant
# is what it charges when a cohort is too thin to measure. Re-exported for
# callers that imported it from here.
EXPAND_ROUND_TRIP_TOKENS = _engine.EXPAND_ROUND_TRIP_TOKENS


def load_detection_cache(state_dir: Path) -> dict[str, Any]:
    """Read the persisted cache-mode detection cache, if any.

    Returned as written: keys are ``"<scope>|<provider>"`` (or a legacy bare
    ``"<scope>"``). Resolve a scope's effective entry with
    :func:`resolve_scope_detection`, never with ``.get(scope)``.
    """
    path = state_dir / "cache_mode_detection.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_scope_detection(detection_cache: dict[str, Any], scope: str) -> dict[str, Any]:
    """The detection entry that defines ``scope``'s posture — its PRIMARY
    provider's bucket (most-locked ``scope|provider`` bucket, else the legacy
    scope-only key). ``{}`` when nothing resolves.

    The entry is returned as a copy with a ``provider`` field derived from
    the key it was found under (entries don't store their own provider);
    empty for a legacy scope-only entry."""
    if not isinstance(detection_cache, dict):
        return {}
    entry = _shaping.resolve_primary_detection_entry(detection_cache, scope)
    if not entry:
        return {}
    resolved = dict(entry)
    if not resolved.get("provider"):
        for key, candidate in detection_cache.items():
            if candidate is entry and isinstance(key, str) and "|" in key:
                resolved["provider"] = key.split("|", 1)[1]
                break
    return resolved


def count_tool_call_sources(tool_calls: list[dict[str, Any]]) -> dict[str, int]:
    """Count tool_call rows by source, with a session_id heuristic fallback."""
    counts: dict[str, int] = defaultdict(int)
    for row in tool_calls:
        src = row.get("source")
        if src in ("gateway", "cron", "subagent"):
            counts[src] += 1
            continue
        sid = str(row.get("session_id") or "")
        pid = str(row.get("prediction_id") or "")
        if sid.startswith("cron_"):
            counts["cron"] += 1
        elif pid:
            counts["gateway"] += 1
        else:
            counts["subagent"] += 1
    return dict(counts)


def print_text_report(scope: str, on_stats: dict[str, Any], off_stats: dict[str, Any],
                      pending_stats: dict[str, Any], bypass_stats: dict[str, Any],
                      tool_source_counts: dict[str, int], n_expand_events: int,
                      locked_mode: str, locked_reason: str,
                      estimator_breakdown: dict[str, int],
                      locked_entry: dict[str, Any] | None = None,
                      expand_per_event: int = EXPAND_ROUND_TRIP_TOKENS,
                      expand_basis: str = "fallback") -> None:
    width = 70
    line = "─" * width
    locked_entry = locked_entry or {}

    print()
    print("═" * width)
    print("  Hermes Tool Belt — Savings Report".ljust(width))
    print("═" * width)
    print(f"  Scope: {scope}")
    print("  (deprecated wrapper — prefer `tool-belt savings`)")
    if locked_mode:
        reason = f" ({locked_reason})" if locked_reason else ""
        posture = "carry-all" if locked_mode == "on" else "narrowing"
        print(f"  Cache-mode lock: {locked_mode.upper()}{reason}  → posture: {posture}")
        # Provenance of the lock: which provider bucket defines this scope's
        # posture, and what it saw when it locked.
        details = []
        if locked_entry.get("provider"):
            details.append(f"provider={locked_entry['provider']}")
        if locked_entry.get("hit_rate_at_lock") is not None:
            try:
                details.append(f"hit_rate_at_lock={float(locked_entry['hit_rate_at_lock']) * 100:.1f}%")
            except (TypeError, ValueError):
                pass
        if locked_entry.get("sessions_locked") is not None:
            details.append(f"sessions_locked={locked_entry['sessions_locked']}")
        if locked_entry.get("last_model"):
            details.append(f"last_model={locked_entry['last_model']}")
        if details:
            print(f"    (primary-provider bucket: {', '.join(details)})")
    if estimator_breakdown:
        total = sum(estimator_breakdown.values())
        primary = max(estimator_breakdown, key=estimator_breakdown.get)
        primary_pct = (estimator_breakdown[primary] / total * 100) if total else 0
        est_label = {
            "tiktoken-cl100k": "tiktoken-cl100k (BPE — exact for GPT-family, ~5% off Claude)",
            "chars-div-4": "chars/4 (heuristic — install tiktoken for exact counts)",
        }.get(primary, primary)
        print(f"  Token estimator: {est_label}  ({primary_pct:.0f}% of rows)")
        print("  (For provider-billed truth see api_calls.jsonl:input_tokens)")
    print()

    # CACHE-ON cohort
    print(f"  ┌{line}┐")
    print(f"  │  CACHE-ON (carry-all){' ' * (width - 24)}│")
    if on_stats.get("n_predictions"):
        print(f"  │     ({on_stats['n_predictions']} predictions across {on_stats['n_sessions']} session(s))".ljust(width + 3) + "│")
        print("  │     full ceiling shipped, no expand_tools — reduction is 0 by design;".ljust(width + 3) + "│")
        print("  │     the value here is the provider's cache reads, not narrowing".ljust(width + 3) + "│")
        print(f"  │{' ' * width}│")
        print(f"  │  Tools shipped:   {on_stats['narrowed_count_avg']:>5.1f}  / {on_stats['ceiling_count_avg']:>5.1f} ceiling  ({on_stats['reduction_pct']:>5.1f}% reduction)".ljust(width + 3) + "│")
        print(f"  │  Tokens saved:    {on_stats['saved_tokens_per_turn_avg']:>5,.0f}  per turn (avg)".ljust(width + 3) + "│")
        print(f"  │  Tokens saved:    {on_stats['saved_tokens_total']:>5,}  total  (vs ceiling)".ljust(width + 3) + "│")
        print(f"  │{' ' * width}│")
        print("  │  Cache amortization (provider-reported):".ljust(width + 3) + "│")
        print(f"  │    Cache hit rate:      {on_stats['cache_hit_rate']:>5.1f}%".ljust(width + 3) + "│")
        print(f"  │    Cache-read tokens:  {on_stats['api_cache_read_tokens']:>10,}  (billed at cache rate)".ljust(width + 3) + "│")
        print(f"  │    Fresh input tokens: {on_stats['api_input_tokens']:>10,}  (billed at full rate)".ljust(width + 3) + "│")
    else:
        print("  │     (no cache-on predictions in window)".ljust(width + 3) + "│")
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
            overhead = n_expand_events * expand_per_event
            net = off_stats.get("saved_tokens_total", 0) - overhead
            basis = "measured" if expand_basis == "measured" else "fallback constant"
            print(f"  │  expand_tools overhead: −{overhead:,}  ({n_expand_events} events × {expand_per_event}, {basis})".ljust(width + 3) + "│")
            print(f"  │  Net savings:           {net:,}".ljust(width + 3) + "│")
    else:
        print("  │     (no cache-off predictions in window)".ljust(width + 3) + "│")
    print(f"  └{line}┘")
    print()

    # Pending cohort
    if pending_stats.get("n_predictions"):
        print(f"  ┌{line}┐")
        print("  │  PENDING (cache-mode detection in progress)".ljust(width + 3) + "│")
        print(f"  │     ({pending_stats['n_predictions']} predictions across {pending_stats['n_sessions']} session(s))".ljust(width + 3) + "│")
        print(f"  │{' ' * width}│")
        print(f"  │  Tokens saved:    {pending_stats['saved_tokens_total']:>5,}  total  (will be re-classified once locked)".ljust(width + 3) + "│")
        print(f"  └{line}┘")
        print()

    # Bypass cohort
    if bypass_stats.get("n_predictions"):
        n_pred = bypass_stats["n_predictions"]
        n_sess = bypass_stats["n_sessions"]
        ceiling_avg = bypass_stats["ceiling_count_avg"]
        ceiling_total = bypass_stats["ceiling_tokens_total"]
        print(f"  ┌{line}┐")
        print("  │  BYPASS COHORT (A/B baseline — narrowing intentionally off)".ljust(width + 3) + "│")
        print(f"  │     ({n_pred} prediction(s) across {n_sess} session(s))".ljust(width + 3) + "│")
        print(f"  │{' ' * width}│")
        print(f"  │  Full toolset shipped:  {ceiling_avg:>5.1f} tools per turn".ljust(width + 3) + "│")
        print(f"  │  Tokens shipped:        {ceiling_total:>6,}  total".ljust(width + 3) + "│")
        print(f"  │{' ' * width}│")
        print("  │  Excluded from savings figures above — these sessions are".ljust(width + 3) + "│")
        print("  │  the deterministic A/B control (see bypass_rate in config).".ljust(width + 3) + "│")
        print(f"  └{line}┘")
        print()

    # Excluded
    cron_n = tool_source_counts.get("cron", 0)
    sub_n = tool_source_counts.get("subagent", 0)
    gate_n = tool_source_counts.get("gateway", 0)
    if cron_n or sub_n:
        print(f"  ┌{line}┐")
        print("  │  Excluded from savings (not subject to narrowing)".ljust(width + 3) + "│")
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

    print("note: `savings-report.py` is deprecated; prefer `tool-belt savings`.",
          file=sys.stderr)

    try:
        since_ts = parse_since(args.since)
    except _engine.InvalidSinceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    predictions = [normalize_prediction_row(p)
                   for p in load_jsonl(args.state_dir / "predictions.jsonl")
                   if float(p.get("ts") or 0) >= since_ts]
    api_calls = [a for a in load_jsonl(args.state_dir / "api_calls.jsonl")
                 if float(a.get("ts") or 0) >= since_ts]
    tool_calls = [normalize_tool_call_row(t)
                  for t in load_jsonl(args.state_dir / "tool_calls.jsonl")
                  if float(t.get("ts") or 0) >= since_ts]

    if not predictions:
        print(f"No predictions.jsonl rows under {args.state_dir}. Nothing to report.",
              file=sys.stderr)
        return 1

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

    tc_by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in tool_calls:
        tc_by_scope[str(t.get("scope") or "unknown")].append(t)

    expand_by_scope: dict[str, int] = defaultdict(int)
    for t in tool_calls:
        if t.get("tool_name") == "expand_tools":
            expand_by_scope[str(t.get("scope") or "unknown")] += 1

    json_report = {"scopes": {}}
    for scope, preds in sorted(preds_by_scope.items()):
        on_stats = cohort_stats(preds, api_last, api_calls, mode_filter="on")
        off_stats = cohort_stats(preds, api_last, api_calls, mode_filter="off")
        pending_stats = cohort_stats(preds, api_last, api_calls, mode_filter="pending")
        bypass_stats = cohort_stats(preds, api_last, api_calls, mode_filter="bypass")
        tc_counts = count_tool_call_sources(tc_by_scope.get(scope, []))
        n_expand = expand_by_scope.get(scope, 0)
        # Posture = the PRIMARY provider's bucket, not a bare scope lookup
        # (keys are ``scope|provider`` now; a bare ``.get(scope)`` misses).
        locked = resolve_scope_detection(detection_cache, scope)
        locked_mode = str(locked.get("mode") or "")
        locked_reason = str(locked.get("lock_reason") or "")

        # Expansion only happens on the narrowing (cache-off) cohort, so
        # price it at that cohort's measured per-event cost; the engine
        # falls back to EXPAND_ROUND_TRIP_TOKENS on thin data and says so.
        overhead = _engine.measure_expand_overhead(preds, api_calls, tool_calls,
                                                   scope=scope, api_last=api_last)
        expand_per_event = int(overhead.noncaching_per_event)
        expand_basis = str(overhead.noncaching_basis)

        est_counts: dict[str, int] = defaultdict(int)
        for p in preds:
            est_counts[str(p.get("tokens_estimator") or "chars-div-4")] += 1

        if args.json:
            json_report["scopes"][scope] = {
                "cache_on": on_stats,
                "cache_off": off_stats,
                "pending": pending_stats,
                "bypass": bypass_stats,
                "tool_call_source_counts": tc_counts,
                "expand_tools_events": n_expand,
                "expand_overhead_per_event": expand_per_event,
                "expand_overhead_basis": expand_basis,
                "locked_mode": locked_mode,
                "locked_reason": locked_reason,
                "locked_posture": ("carry_all" if locked_mode == "on"
                                   else "narrowing" if locked_mode == "off" else ""),
                "locked_hit_rate_at_lock": locked.get("hit_rate_at_lock"),
                "locked_sessions_locked": locked.get("sessions_locked"),
                "locked_last_model": locked.get("last_model"),
                "token_estimators": dict(est_counts),
            }
        else:
            print_text_report(scope, on_stats, off_stats, pending_stats, bypass_stats,
                              tc_counts, n_expand, locked_mode, locked_reason,
                              dict(est_counts), locked_entry=locked,
                              expand_per_event=expand_per_event,
                              expand_basis=expand_basis)

    if args.json:
        print(json.dumps(json_report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
