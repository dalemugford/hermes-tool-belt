#!/usr/bin/env python3
"""Chronological replay of the shaper over a scope's recorded telemetry.

What this is
============

Take a scope's ``predictions.jsonl`` / ``tool_calls.jsonl`` / ``api_calls.jsonl``,
throw away its learned state, and re-run history: session by session, in
order, feeding the shaper everything recorded up to that point and applying
its recommendations before moving to the next session. The result is the
counterfactual the live system can never show you — *what the loadout would
have looked like the whole way in*, under a set of thresholds you have not
deployed.

It answers the questions a single ``shape-ceiling.py --dry-run`` cannot:

  · when the first demotion would have landed, and when the loadout
    converges (90% of its final demoted set);
  · what carrying the untightened loadout cost over the whole replay (the
    "ramp cost" — tokens carried before the shaper caught up);
  · how many ``expand_tools`` round-trips the tightened loadout implies
    (a demoted tool the model then reached for), and which tools those are;
  · promotes and flap (tools demoted more than once — the resonance risk of
    a short recency window).

Each ``--window-days`` × ``--floor`` pair is one run, so the flags sweep the
thresholds side by side on identical telemetry.

Fidelity
========

High, deliberately: the replay drives the SAME code the gateway does —
``shaping.compute_scope_recommendations`` for the decision and
``shaping.apply_recommendations`` for the carry/expand_only merge — with the
scope's MEASURED expand penalty (``shaping.measured_expand_penalty``), its
real measured schema sizes, and its real ``always_carry`` pins. The recency
window is the shaper's own: every step hands it every session seen so far and
lets it apply ``window_days`` itself, anchored (as in production) on the
newest activity in the data rather than on the wall clock. Nothing here
re-implements a shaping rule.

Two deliberate deviations, both stated so nobody reads more into a number
than it can carry:

  · Cache posture is pinned to ``off``. On a caching provider the plugin
    carries everything and ships no ``expand_tools`` at all (decision D1), so
    a replay there has nothing to decide. This tool asks the non-caching
    question: how tight would the carried prefix have become?
  · Ramp cost is this script's own bookkeeping (carried ceiling tokens ×
    turns per session), not a shaper output.

Known blind spot
================

Presence is an invitation. The replay can only see tools the model actually
reached for in the recorded history — and the model only reaches for what it
was offered. A tool this replay demotes on session 3 might, in a live run,
never have been reached for again *because it stopped being visible*; the
replay would score that as "no implied expand events" when the truth is
"unmeasurable". Read the implied-expand column as a lower bound on the cost
of demotion, never as proof that demotion was free.

Read-only: this script never writes ``learned.json`` (or anything else) —
the replay's learned state lives entirely in memory.

Usage
=====

  python3 scripts/replay-shaping.py
  python3 scripts/replay-shaping.py --scope assistant-a:telegram
  python3 scripts/replay-shaping.py --window-days 7,14,30 --floor 2,8
  python3 scripts/replay-shaping.py --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# The plugin directory is hyphenated; ``scripts/_plugin_loader.py`` owns the
# import dance and is the same loader every other shipped script and the test
# suite go through, so this file never grows a second copy of it.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _plugin_loader import load_plugin_package  # noqa: E402

load_plugin_package(eager_submodules=("shaping", "learned", "logger_io", "savings"))
shaping = sys.modules["tool_belt_plugin.shaping"]
learned = sys.modules["tool_belt_plugin.learned"]
logger_io = sys.modules["tool_belt_plugin.logger_io"]
savings = sys.modules["tool_belt_plugin.savings"]

#: Machine-readable output identity, versioned like the shaper's porcelain.
PORCELAIN_SCHEMA = "tool-belt/replay-shaping"
PORCELAIN_VERSION = 1

#: Default threshold sweep: the shipped window against two wider ones, at the
#: shipped demote floor.
DEFAULT_WINDOW_DAYS = "7,14,30"
DEFAULT_FLOOR = "2"


def _int_list(raw: str, flag: str) -> list[int]:
    """Parse a ``7,14,30``-style flag value into positive ints, in order."""
    values: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            parsed = int(part)
        except ValueError:
            raise SystemExit(f"{flag}: {part!r} is not an integer")
        if parsed <= 0:
            raise SystemExit(f"{flag}: {parsed} is not a positive integer")
        values.append(parsed)
    if not values:
        raise SystemExit(f"{flag}: expected at least one positive integer")
    return values


def tools_used_by_session(
    sessions: dict[str, list[dict[str, Any]]],
    calls_by_pred: dict[str, list[dict[str, Any]]],
) -> dict[str, set[str]]:
    """Per session, the tools the model actually dispatched.

    Primary model dispatches only (``tool_call_id`` present). Historical
    nested/secondary rows (sandbox, MCP, memory fan-out) never faced narrowing
    and would inflate "reached for" 30x+ — the 2026.9.4 gate excludes them
    going forward; mirror that here.
    """
    used: dict[str, set[str]] = {}
    for sid, rows in sessions.items():
        names: set[str] = set()
        for pred in rows:
            for call in calls_by_pred.get(str(pred.get("prediction_id")), []):
                if (call.get("source", "gateway") == "gateway"
                        and call.get("tool_name")
                        and call.get("tool_call_id")):
                    names.add(str(call["tool_name"]))
        used[sid] = names
    return used


def replay_scope(
    *,
    scope: str,
    sessions: dict[str, list[dict[str, Any]]],
    calls_by_pred: dict[str, list[dict[str, Any]]],
    api_call_counts: dict[str, int],
    schema_sizes: dict[str, int],
    expand_round_trip_tokens: int | None,
    protected: set[str],
    window_days: int,
    demote_min_sessions_no_use: int,
    promote_min_sessions: int,
    promote_min_calls: int,
    demote_k: float,
) -> dict[str, Any]:
    """Replay one scope under one threshold combination.

    Starts from an EMPTY learned state and walks the scope's sessions oldest
    first. At each step the cost of the loadout going *in* is charged (the
    demotions from prior steps are already paying off), then the shaper is
    given every session seen so far — it applies ``window_days`` itself — and
    its recommendations are merged exactly as the live engine merges them.

    Returns the run's summary row; no state is written anywhere.
    """
    # Place each session in time by its first VALID timestamp. Rows with a
    # missing/zero ``ts`` cannot be placed on the replay clock: a session that
    # carries even one such row would sort to the very front (min ts = 0) while
    # its real rows lie months later, dragging the shaper's data-relative
    # ``now`` to the dataset's end from step 1 and starving the day window of
    # every genuinely-early session (observed: zero demotions for 107 steps).
    # The live shaper is unaffected (a zero can't raise a max); this is replay
    # hygiene only. Sessions with no valid timestamp at all are left out.
    def _valid_ts(rows):
        return [float(p.get("ts") or 0) for p in rows if float(p.get("ts") or 0) > 0]
    order = sorted(
        (sid for sid in sessions if _valid_ts(sessions[sid])),
        key=lambda sid: min(_valid_ts(sessions[sid])),
    )
    total = len(order)
    turns = {sid: len(sessions[sid]) for sid in order}
    ceiling = {
        sid: statistics.median([float(p.get("ceiling_tokens") or 0) for p in sessions[sid]])
        for sid in order
    }
    used = tools_used_by_session(sessions, calls_by_pred)

    def size(tool: str) -> int:
        return int(schema_sizes.get(tool, logger_io.DEFAULT_PER_TOOL_TOKENS))

    state = learned.normalize_state({})
    expand_only: set[str] = set()
    demote_count: Counter[str] = Counter()
    first_demoted_at: dict[str, int] = {}
    curve: list[int] = []
    carried_tokens = 0.0
    first_demotion: int | None = None
    reach_events: list[tuple[int, str]] = []
    promotes = 0

    for step in range(1, total + 1):
        sid = order[step - 1]
        # 1. Pay for this session under the loadout it starts with.
        carried = max(ceiling[sid] - sum(size(t) for t in expand_only), 0.0)
        carried_tokens += turns[sid] * carried
        # 2. A demoted tool reached for here is an implied expand_tools event.
        for tool in sorted(used[sid] & expand_only):
            reach_events.append((step, tool))
        # 3. Re-shape on everything seen so far; the shaper applies the day
        #    window (anchored on this session, the newest data it can see).
        # Simulated "now" is this session's last activity. Truncate EVERY
        # included session to rows at or before it: a long-lived session that
        # began early but ran for weeks must not leak its future rows into an
        # early step — that would drag the shaper's data-relative ``now`` to
        # the dataset's end and starve the day window of early sessions
        # (observed: zero demotions for 107 steps, then a cliff). Sessions
        # left empty by the cut haven't started yet and are dropped.
        now_sim = max((float(p.get("ts") or 0) for p in sessions[sid]), default=0.0)
        subset = {
            s: [p for p in sessions[s] if 0 < float(p.get("ts") or 0) <= now_sim]
            for s in order[:step]
        }
        subset = {s: rows for s, rows in subset.items() if rows}
        recs = shaping.compute_scope_recommendations(
            scope=scope,
            sessions=subset,
            calls_by_pred=calls_by_pred,
            window_days=window_days,
            promote_min_sessions=promote_min_sessions,
            promote_min_calls=promote_min_calls,
            demote_min_sessions_no_use=demote_min_sessions_no_use,
            demote_k=demote_k,
            schema_sizes=schema_sizes,
            cache_mode="off",
            api_call_counts=api_call_counts,
            expand_round_trip_tokens=expand_round_trip_tokens,
        )
        state, _changes = shaping.apply_recommendations(state, {scope: recs})
        scope_state = (state.get("scopes") or {}).get(scope) or {}
        now_expand_only = {str(t) for t in (scope_state.get("expand_only") or [])} - protected
        for tool in now_expand_only - expand_only:
            demote_count[tool] += 1
            first_demoted_at.setdefault(tool, step)
            if first_demotion is None:
                first_demotion = step
        promotes += len(expand_only - now_expand_only)
        expand_only = now_expand_only
        curve.append(len(expand_only))

    final = len(expand_only)
    converged_at = next(
        (i + 1 for i, count in enumerate(curve) if final and count >= 0.9 * final),
        None,
    )
    steady_tokens = (
        max(ceiling[order[-1]] - sum(size(t) for t in expand_only), 0.0) if order else 0.0
    )
    reached = Counter(tool for _step, tool in reach_events)
    return {
        "scope": scope,
        "window_days": window_days,
        "floor": demote_min_sessions_no_use,
        "promote_min_sessions": promote_min_sessions,
        "promote_min_calls": promote_min_calls,
        "demote_k": demote_k,
        "sessions": total,
        "first_demotion_session": first_demotion,
        "converged_session": converged_at,
        "final_demoted": final,
        "steady_tokens_per_turn": round(steady_tokens),
        "carried_tokens": round(carried_tokens),
        "implied_expand_events": len(reach_events),
        "promotes": promotes,
        "flap": sum(1 for n in demote_count.values() if n >= 2),
        "curve": curve,
        "final_expand_only": sorted(expand_only),
        "reached_while_demoted": dict(reached.most_common()),
    }


def load_scope_inputs(state_dir: Path, scope_filter: str = "") -> dict[str, Any]:
    """Read the telemetry a replay needs, once, for every matching scope."""
    preds = shaping.load_jsonl(state_dir / "predictions.jsonl")
    tool_calls = shaping.load_jsonl(state_dir / "tool_calls.jsonl")
    api_calls = shaping.load_jsonl(state_dir / "api_calls.jsonl")
    grouped = shaping.group_predictions_by_scope_session(preds)
    if scope_filter:
        grouped = {s: v for s, v in grouped.items() if s == scope_filter}
    return {
        "grouped": grouped,
        # Schema sizes come through logger_io: the on-disk document nests the
        # per-tool map under a "tools" key, so a raw json.load of the file
        # yields sizes that are silently all-default.
        "schema_sizes": logger_io.load_schema_sizes(state_dir),
        "calls_by_pred": shaping.index_tool_calls_by_prediction(tool_calls),
        "api_call_counts": shaping.index_api_call_counts(api_calls),
        # The penalty VALUE comes from the shaper's own helper, so the replay
        # prices expansions exactly as production does. Its BASIS (measured vs
        # the flat fallback on thin data) is not exposed there, so it is read
        # off the measurement itself — for the report only, never for pricing.
        "penalty_for": lambda scope: shaping.measured_expand_penalty(
            preds, api_calls, tool_calls, scope),
        "penalty_basis_for": lambda scope: _penalty_basis(
            preds, api_calls, tool_calls, scope),
    }


def _penalty_basis(
    preds: list[dict[str, Any]],
    api_calls: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    scope: str,
) -> str:
    """``"measured"`` or ``"fallback"`` for the scope's expansion penalty."""
    try:
        return str(savings.measure_expand_overhead(
            preds, api_calls, tool_calls, scope=scope).noncaching_basis)
    except Exception:
        return "unknown"


def replay_state_dir(
    state_dir: Path,
    *,
    scope_filter: str = "",
    window_days: list[int],
    floors: list[int],
    promote_min_sessions: int,
    promote_min_calls: int,
    demote_k: float,
    plugin_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay every matching scope under every (window_days × floor) combo."""
    inputs = load_scope_inputs(state_dir, scope_filter)
    plugin_config = plugin_config if plugin_config is not None else {}
    scopes: list[dict[str, Any]] = []
    for scope in sorted(inputs["grouped"]):
        sessions = inputs["grouped"][scope]
        if not sessions:
            continue
        protected = shaping.effective_always_carry(plugin_config, scope)
        penalty = inputs["penalty_for"](scope)
        combos = [
            replay_scope(
                scope=scope,
                sessions=sessions,
                calls_by_pred=inputs["calls_by_pred"],
                api_call_counts=inputs["api_call_counts"],
                schema_sizes=inputs["schema_sizes"],
                expand_round_trip_tokens=penalty,
                protected=protected,
                window_days=days,
                demote_min_sessions_no_use=floor,
                promote_min_sessions=promote_min_sessions,
                promote_min_calls=promote_min_calls,
                demote_k=demote_k,
            )
            for days in window_days
            for floor in floors
        ]
        scopes.append({
            "scope": scope,
            "sessions": len(sessions),
            "expand_round_trip_tokens": penalty,
            "expand_penalty_basis": inputs["penalty_basis_for"](scope),
            "protected_pins": sorted(protected),
            "combos": combos,
        })
    return {
        "schema": PORCELAIN_SCHEMA,
        "version": PORCELAIN_VERSION,
        "state_dir": str(state_dir),
        "scope_filter": scope_filter,
        "scopes": scopes,
    }


def _combo_label(combo: dict[str, Any]) -> str:
    return f"{combo['window_days']}d/floor{combo['floor']}"


def print_report(doc: dict[str, Any], out=sys.stdout) -> None:
    """The human report. Prose only — no caller parses this; use --json."""
    for entry in doc["scopes"]:
        penalty = entry["expand_round_trip_tokens"]
        print(f"\n=== {entry['scope']}  (sessions={entry['sessions']}, "
              f"expand penalty={penalty} tok [{entry['expand_penalty_basis']}], "
              f"protected pins={len(entry['protected_pins'])}) ===", file=out)
        header = (f"{'combo':>14} {'1st demote':>10} {'converged':>9} {'demoted':>7} "
                  f"{'steady tok/turn':>15} {'carried tok':>14} {'implied expands':>15} "
                  f"{'promotes':>8} {'flap':>5}")
        print(header, file=out)
        print("-" * len(header), file=out)
        for combo in entry["combos"]:
            print(f"{_combo_label(combo):>14} "
                  f"{str(combo['first_demotion_session']):>10} "
                  f"{str(combo['converged_session']):>9} "
                  f"{combo['final_demoted']:>7} "
                  f"{combo['steady_tokens_per_turn']:>15,} "
                  f"{combo['carried_tokens']:>14,} "
                  f"{combo['implied_expand_events']:>15} "
                  f"{combo['promotes']:>8} "
                  f"{combo['flap']:>5}", file=out)
        # Carried tokens are only meaningful against another combo on the same
        # telemetry; the widest/most patient combo is the reference.
        base = max(entry["combos"], key=lambda c: (c["window_days"], c["floor"]))
        print(f"\n  carried tokens over the replay, vs {_combo_label(base)}:", file=out)
        for combo in entry["combos"]:
            delta = base["carried_tokens"] - combo["carried_tokens"]
            verb = "saves" if delta > 0 else "costs"
            print(f"    {_combo_label(combo):>14} {combo['carried_tokens']:>14,}   "
                  f"{verb} {abs(delta):>12,}", file=out)
        print("\n  tools reached for while demoted (implied expand_tools "
              "round-trips):", file=out)
        for combo in entry["combos"]:
            reached = combo["reached_while_demoted"]
            top = ", ".join(f"{tool}({n})" for tool, n in list(reached.items())[:5])
            print(f"    {_combo_label(combo):>14} "
                  f"{combo['implied_expand_events']} events / {len(reached)} tools"
                  + (f"; top: {top}" if top else ""), file=out)
        print("\n  Blind spot: a tool demoted early may simply never have been "
              "offered again — implied expands are a lower bound.", file=out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--state-dir", default=str(shaping.default_state_dir()))
    ap.add_argument("--scope", default="", help="filter to a specific scope (default: all)")
    ap.add_argument("--window-days", default=DEFAULT_WINDOW_DAYS,
                    help="recency window(s) in days, comma-separated "
                         f"(default: {DEFAULT_WINDOW_DAYS})")
    ap.add_argument("--floor", default=DEFAULT_FLOOR,
                    help="demote_min_sessions_no_use value(s), comma-separated "
                         f"(default: {DEFAULT_FLOOR}); one run per window x floor pair")
    ap.add_argument("--promote-min-sessions", type=int, default=1)
    ap.add_argument("--promote-min-calls", type=int, default=2)
    ap.add_argument("--demote-k", type=float, default=1.5,
                    help="economic safety factor: demote only when carrying costs "
                         "more than k x the on-demand expansion cost (tokens)")
    ap.add_argument("--json", action="store_true",
                    help="emit the structured document on stdout instead of prose")
    args = ap.parse_args()

    state_dir = Path(args.state_dir).expanduser()
    windows = _int_list(args.window_days, "--window-days")
    floors = _int_list(args.floor, "--floor")

    doc = replay_state_dir(
        state_dir,
        scope_filter=args.scope,
        window_days=windows,
        floors=floors,
        promote_min_sessions=args.promote_min_sessions,
        promote_min_calls=args.promote_min_calls,
        demote_k=args.demote_k,
        plugin_config=shaping.load_cli_plugin_config(),
    )

    if not doc["scopes"]:
        where = f" matching {args.scope!r}" if args.scope else ""
        print(f"No scopes{where} with predictions under {state_dir}. "
              "Nothing to replay.", file=sys.stderr)
        if args.json:
            print(json.dumps(doc, indent=2))
        return 1

    if args.json:
        print(json.dumps(doc, indent=2))
    else:
        print_report(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
