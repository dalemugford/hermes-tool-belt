#!/usr/bin/env python3
"""Between-session tool-loadout shaper.

The principle in one line: usage-aware tool loading at per-tool granularity.
Adjustments happen at session boundaries (free cache moments), not per turn.

This script reads ``predictions.jsonl`` and ``tool_calls.jsonl`` for the
last N sessions per scope, then writes per-scope promote/demote
recommendations into ``learned.json``. The plugin's existing
``apply_to_preset`` machinery picks them up automatically when
``learned_mode`` is ``auto`` or ``audit``.

What it computes
================

For each scope:

  Promote candidates — tools that the model reached for via
    ``expand_tools`` across recent sessions. Direct evidence the tool
    was wanted and wasn't in the baseline. Folded into the next
    session's frozen ceiling so the model can call them immediately
    without a round-trip.

  Demote candidates — tools currently in always_on that went unused
    across enough recent sessions. Pulling them out of the ceiling
    keeps the prefix tighter without losing real capability.
    Conservative thresholds — easy to revert via a single use.

What it does NOT do
===================

  · No bundle-level decisions. The unit is the tool, not the toolset.
  · No per-turn adjustments. Run-frequency is on the order of daily
    or session-boundary, not per-message.
  · No clobbering of user-facing config. Hand-tuning happens via
    ``config.yaml``'s ``always_on_extra`` / ``always_off`` (existing
    mechanism); this script owns ``scopes[].always_on`` /
    ``scopes[].always_off`` *only* under its own ``cache_aware``
    sub-key, then mirrors the consumed values to the top-level
    ``always_on`` / ``always_off`` for the existing
    ``apply_to_preset`` reader.

Usage
=====

  python3 scripts/shape-ceiling.py
  python3 scripts/shape-ceiling.py --dry-run            # report only
  python3 scripts/shape-ceiling.py --scope assistant-a:telegram
  python3 scripts/shape-ceiling.py --window 50          # consider last 50 sessions per scope

Threshold defaults come from ``policy.yaml`` under
``learning.shape_ceiling``. CLI flags still win when passed explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Import the protected-always-on set from analyze.py so the shaper respects
# the same do-not-demote list the analyzer uses.  This prevents demoting
# core meta tools (expand_tools, send_message, etc.) that either never
# appear in tool_calls.jsonl or are structurally required by the plugin.
_PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PLUGIN_DIR))
try:
    from analyze import effective_protected_always_on  # noqa: E402
except Exception:
    # Fallback: hard-coded minimum set if analyze.py is unavailable.
    BASE_PROTECTED_ALWAYS_ON = {
        "memory",
        "session_search",
        "clarify",
        "skill_view",
        "skills_list",
        "todo",
        "send_message",
        "expand_tools",
        "tool_search",
        "tool_describe",
        "tool_call",
    }

    def effective_protected_always_on(plugin_dir: Path | None = None) -> set[str]:  # type: ignore[no-redef]
        return set(BASE_PROTECTED_ALWAYS_ON)


# Thresholds — conservative defaults that won't fire on noise.
DEFAULTS = {
    "session_window": 20,
    "promote_min_sessions": 2,
    "promote_min_calls": 3,
    "demote_min_sessions_no_use": 20,
    "version": 1,
}

_POLICY_PATH = Path(__file__).resolve().parent.parent / "policy.yaml"


def default_state_dir() -> Path:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(home) / "state" / "tool-belt"


def load_shape_ceiling_defaults(policy_path: Path = _POLICY_PATH) -> dict[str, int]:
    """Load shaper defaults from policy.yaml, falling back silently on errors.

    Prefer PyYAML when available so the parser follows the real policy shape.
    If this runtime lacks PyYAML, fall back to a tiny indentation-based reader
    for the `learning.shape_ceiling` block rather than disabling inheritance.
    """
    try:
        raw = policy_path.read_text(encoding="utf-8")
    except Exception:
        return dict(DEFAULTS)

    data: dict[str, Any] | None = None
    try:
        import yaml  # type: ignore[import-untyped]
        loaded = yaml.safe_load(raw) or {}
        if isinstance(loaded, dict):
            data = loaded
    except Exception:
        data = None

    if isinstance(data, dict):
        learning = data.get("learning")
        if isinstance(learning, dict):
            shape = learning.get("shape_ceiling")
            if isinstance(shape, dict):
                return _merge_shape_defaults(shape)

    shape: dict[str, int] = {}
    in_learning = False
    in_shape = False
    learning_indent = None
    shape_indent = None
    for raw_line in raw.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if stripped == "learning:":
            in_learning = True
            in_shape = False
            learning_indent = indent
            shape_indent = None
            continue
        if in_learning and learning_indent is not None and indent <= learning_indent and stripped != "learning:":
            in_learning = False
            in_shape = False
            learning_indent = None
            shape_indent = None
        if not in_learning:
            continue

        if stripped == "shape_ceiling:":
            in_shape = True
            shape_indent = indent
            continue
        if in_shape and shape_indent is not None and indent <= shape_indent and stripped != "shape_ceiling:":
            in_shape = False
            shape_indent = None
        if not in_shape or ":" not in stripped:
            continue

        key, value = [part.strip() for part in stripped.split(":", 1)]
        if key not in {"session_window", "promote_min_sessions", "promote_min_calls", "demote_min_sessions_no_use"}:
            continue
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed > 0:
            shape[key] = parsed

    return _merge_shape_defaults(shape)


def _merge_shape_defaults(overrides: dict[str, Any]) -> dict[str, int]:
    merged = dict(DEFAULTS)
    for key in ("session_window", "promote_min_sessions", "promote_min_calls", "demote_min_sessions_no_use"):
        value = overrides.get(key)
        if value is None:
            continue
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed > 0:
            merged[key] = parsed
    return merged


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


def group_predictions_by_scope_session(
    preds: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Returns {scope: {session_id: [preds-in-order]}}."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for p in preds:
        scope = str(p.get("scope") or "")
        # Group by the Hermes internal session UUID, which rotates on /new,
        # so a single long-lived chat yields the distinct-session count the
        # demote threshold needs. Fall back to session_id (the session key)
        # for older rows written before hermes_session_id existed.
        sid = str(p.get("hermes_session_id") or p.get("session_id") or "")
        if scope and sid:
            out[scope][sid].append(p)
    for scope, sessions in out.items():
        for sid in sessions:
            sessions[sid].sort(key=lambda p: p.get("ts", 0))
    return out


def index_tool_calls_by_prediction(tool_calls: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in tool_calls:
        pid = str(t.get("prediction_id") or "")
        if pid:
            out[pid].append(t)
    return out


def compute_scope_recommendations(
    scope: str,
    sessions: dict[str, list[dict[str, Any]]],
    calls_by_pred: dict[str, list[dict[str, Any]]],
    window: int,
    promote_min_sessions: int,
    promote_min_calls: int,
    demote_min_sessions_no_use: int,
) -> dict[str, Any]:
    """Per-scope promote/demote analysis.

    Promote evidence: a tool call row tagged ``expand_tools_used: true``
    (or ``was_expanded: true``) is direct evidence the model reached for
    a tool that wasn't initially available. Count distinct sessions per
    tool; promote when ≥ ``promote_min_sessions`` distinct sessions AND
    ≥ ``promote_min_calls`` total calls.

    Demote evidence: a tool that consistently appears in ``always_on_tools``
    in predictions but never in tool_calls.jsonl across the window. Only
    fires when the window has ≥ ``demote_min_sessions_no_use`` sessions —
    avoid removing capability on thin evidence.
    """
    session_ids_ordered = sorted(sessions.keys(), key=lambda sid: -max(
        (p.get("ts", 0) for p in sessions[sid]), default=0
    ))
    recent_session_ids = session_ids_ordered[:window]
    recent_sessions = {sid: sessions[sid] for sid in recent_session_ids}

    # Promote signals
    sessions_with_tool: dict[str, set[str]] = defaultdict(set)
    calls_for_tool: Counter[str] = Counter()
    for sid, plist in recent_sessions.items():
        for p in plist:
            pid = p.get("prediction_id", "")
            for tc in calls_by_pred.get(pid, []):
                # Treat both was_expanded and expand_tools_used as positive
                # evidence — was_expanded captures the in-turn expansion,
                # expand_tools_used captures sticky-carried use across turns.
                if tc.get("tool_name") == "expand_tools":
                    continue
                evidence = bool(tc.get("was_expanded")) or bool(tc.get("expand_tools_used"))
                if not evidence:
                    continue
                tool_name = str(tc.get("tool_name") or "")
                if not tool_name:
                    continue
                sessions_with_tool[tool_name].add(sid)
                calls_for_tool[tool_name] += 1

    promote: list[dict[str, Any]] = []
    for tool_name, sids in sessions_with_tool.items():
        if len(sids) >= promote_min_sessions and calls_for_tool[tool_name] >= promote_min_calls:
            promote.append({
                "tool": tool_name,
                "sessions": len(sids),
                "calls": calls_for_tool[tool_name],
                "evidence": "expand_tools",
            })
    promote.sort(key=lambda x: (-int(x["sessions"]), -int(x["calls"]), str(x["tool"])))

    # Demote signals — what was always-on in baseline but unused?
    always_on_observed: set[str] = set()
    tools_called: set[str] = set()
    for sid, plist in recent_sessions.items():
        for p in plist:
            for t in (p.get("always_on_tools") or []):
                always_on_observed.add(str(t))
            # Also track tools kept on via the unknown-tools safe-default.
            # Without this, new tools added by upstream (Hermes updates, plugin
            # changes) are invisible to the shaper and never auto-demoted.
            for t in (p.get("unknown_kept_tools") or []):
                # Skip MCP tools: Tool Belt passes them through without
                # narrowing, and Tool Search manages their activation layer.
                if t.startswith("mcp__") or t.startswith("mcp_"):
                    continue
                always_on_observed.add(str(t))
            pid = p.get("prediction_id", "")
            for tc in calls_by_pred.get(pid, []):
                name = str(tc.get("tool_name") or "")
                if name:
                    tools_called.add(name)

    demote: list[dict[str, Any]] = []
    if len(recent_sessions) >= demote_min_sessions_no_use:
        # Build the protected set once per scope evaluation.  Tools in this
        # set are structurally required (expand_tools, send_message, etc.)
        # and must never be demoted even if they don't appear in
        # tool_calls.jsonl.
        protected = effective_protected_always_on(_PLUGIN_DIR)
        for tool_name in sorted(always_on_observed - tools_called):
            if tool_name in protected:
                continue
            demote.append({
                "tool": tool_name,
                "sessions_without_use": len(recent_sessions),
                "evidence": "always_on_unused",
            })

    return {
        "scope": scope,
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sessions_considered": len(recent_sessions),
        "window_requested": window,
        "promote": promote,
        "demote": demote,
    }


def merge_into_learned(
    state_dir: Path,
    per_scope: dict[str, dict[str, Any]],
    dry_run: bool,
) -> tuple[dict[str, Any], bool]:
    """Merge cache-aware recommendations into ``learned.json``.

    The shaper owns ``scopes[].cache_aware``, ``scopes[].always_on``, and
    ``scopes[].always_off`` for the scopes it knows about. Other scopes
    are left untouched. Global config is preserved as-is. Returns the
    merged state and whether anything actually changed.
    """
    learned_path = state_dir / "learned.json"
    if learned_path.exists():
        try:
            existing = json.loads(learned_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except Exception:
            existing = {}
    else:
        existing = {}

    state = dict(existing)
    state.setdefault("version", DEFAULTS["version"])
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    scopes = dict(state.get("scopes") or {})

    changed = False
    for scope, recs in per_scope.items():
        entry = dict(scopes.get(scope) or {})
        prev_cache_aware = entry.get("cache_aware") or {}
        # Compare on the structurally significant fields, not the timestamp
        prev_sig = (
            [(p["tool"], p["sessions"], p["calls"]) for p in (prev_cache_aware.get("promote") or [])],
            [(d["tool"], d["sessions_without_use"]) for d in (prev_cache_aware.get("demote") or [])],
        )
        new_sig = (
            [(p["tool"], p["sessions"], p["calls"]) for p in recs["promote"]],
            [(d["tool"], d["sessions_without_use"]) for d in recs["demote"]],
        )
        if prev_sig == new_sig:
            continue  # no change for this scope
        changed = True
        entry["cache_aware"] = recs
        # Mirror promote → always_on and demote → always_off for the
        # existing apply_to_preset consumer. Owned by the shaper.
        entry["always_on"] = sorted(p["tool"] for p in recs["promote"])
        entry["always_off"] = sorted(d["tool"] for d in recs["demote"])
        scopes[scope] = entry
    state["scopes"] = scopes

    if changed and not dry_run:
        learned_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = learned_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(learned_path)

    return state, changed


def main() -> int:
    defaults = load_shape_ceiling_defaults()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-dir", default=str(default_state_dir()))
    ap.add_argument("--scope", default="", help="filter to a specific scope (default: all)")
    ap.add_argument("--window", type=int, default=defaults["session_window"])
    ap.add_argument("--promote-min-sessions", type=int, default=defaults["promote_min_sessions"])
    ap.add_argument("--promote-min-calls", type=int, default=defaults["promote_min_calls"])
    ap.add_argument("--demote-min-sessions", type=int, default=defaults["demote_min_sessions_no_use"])
    ap.add_argument("--dry-run", action="store_true", help="report only, don't write learned.json")
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    preds = load_jsonl(state_dir / "predictions.jsonl")
    tool_calls = load_jsonl(state_dir / "tool_calls.jsonl")
    if not preds:
        print(f"No predictions.jsonl rows under {state_dir}. Nothing to shape.", file=sys.stderr)
        return 1

    grouped = group_predictions_by_scope_session(preds)
    calls_by_pred = index_tool_calls_by_prediction(tool_calls)

    per_scope: dict[str, dict[str, Any]] = {}
    for scope, sessions in grouped.items():
        if args.scope and scope != args.scope:
            continue
        per_scope[scope] = compute_scope_recommendations(
            scope=scope,
            sessions=sessions,
            calls_by_pred=calls_by_pred,
            window=args.window,
            promote_min_sessions=args.promote_min_sessions,
            promote_min_calls=args.promote_min_calls,
            demote_min_sessions_no_use=args.demote_min_sessions,
        )

    if not per_scope:
        print(f"No scopes matched filter {args.scope!r}.", file=sys.stderr)
        return 1

    state, changed = merge_into_learned(state_dir, per_scope, args.dry_run)

    for scope, recs in per_scope.items():
        print(f"\n=== {scope}  (sessions_considered={recs['sessions_considered']}) ===")
        if recs["promote"]:
            print("  Promote:")
            for p in recs["promote"]:
                print(f"    + {p['tool']:<30} sessions={p['sessions']:>2}  calls={p['calls']:>3}  evidence={p['evidence']}")
        else:
            print("  Promote: (none — no tools met the threshold)")
        if recs["demote"]:
            print("  Demote:")
            for d in recs["demote"]:
                print(f"    - {d['tool']:<30} sessions_without_use={d['sessions_without_use']}  evidence={d['evidence']}")
        elif recs["sessions_considered"] < args.demote_min_sessions:
            print(f"  Demote: (skipped — only {recs['sessions_considered']} sessions, need ≥{args.demote_min_sessions})")
        else:
            print("  Demote: (none)")

    if args.dry_run:
        print("\n[dry-run] No changes written to learned.json")
    elif changed:
        print(f"\nWrote updated recommendations to {state_dir / 'learned.json'}")
        print("To activate the recommendations, set ``learned_mode: auto`` (or ``audit``) for the scope in config.yaml.")
    else:
        print("\nNo changes — recommendations match current learned.json content.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
