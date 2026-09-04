#!/usr/bin/env python3
"""Between-session tool-loadout shaper.

The principle in one line: usage-aware tool loading at per-tool granularity.
Adjustments happen at session boundaries (free cache moments), not per turn.

This script reads ``predictions.jsonl`` and ``tool_calls.jsonl`` for the
sessions active in the last N days per scope, then writes per-scope
promote/demote recommendations into ``learned.json``. The plugin's existing
``apply_to_preset`` machinery picks them up automatically when
``learned_mode`` is ``apply``.

What it computes
================

For each scope:

  Promote candidates — ``expand_only`` tools that the model reached for
    via ``expand_tools`` across recent sessions. Direct evidence the tool
    was wanted and wasn't being carried. Moved into the adaptive
    ``carry`` class so the next session's frozen ceiling carries them and
    the model can call them immediately without a round-trip.

  Demote candidates — adaptive ``carry`` residents that went unused
    across the recency window (``--window-days``) and whose carrying cost
    beats the on-demand expansion cost by the safety factor ``--demote-k``.
    Moving them to ``expand_only`` keeps the carried prefix tighter without
    losing real capability (they stay reachable via ``expand_tools`` /
    triggers), and a single use inside the window promotes them back. The
    immutable ``always_carry`` surface is never a demote candidate.

What it does NOT do
===================

  · No bundle-level decisions. The unit is the tool, not the toolset.
  · No per-turn adjustments. Run-frequency is on the order of daily
    or session-boundary, not per-message.
  · No clobbering of user-facing config. Hand-tuning happens through
    ``hermes config set`` (existing mechanism); this script owns the adaptive
    ``carry`` / ``expand_only`` assignment for a scope under its own
    ``shaping`` sub-key of ``learned.json``, written in the canonical v2
    shape (``carry`` / ``expand_only`` / ``shaping``) through
    ``learned.write_state`` — the sole owner of learned-state writes.

Usage
=====

  python3 scripts/shape-ceiling.py
  python3 scripts/shape-ceiling.py --dry-run            # report only
  python3 scripts/shape-ceiling.py --scope assistant-a:telegram
  python3 scripts/shape-ceiling.py --window-days 14     # widen the recency window to 14 days
  python3 scripts/shape-ceiling.py --json               # porcelain document on stdout
  python3 scripts/shape-ceiling.py --json-file out.json # porcelain document to a file

Threshold defaults resolve across the config layers: ``config.yaml``
``learning.shape_ceiling`` (the user layer) overrides the shipped
``policy.yaml`` ``learning.shape_ceiling`` defaults. CLI flags still win when
passed explicitly.

Output contract
===============

The prose printed to stdout is for humans and is **not** a contract: no
caller may parse it. Programs consume the porcelain document produced by
``--json`` / ``--json-file`` — a versioned JSON object (see
:data:`PORCELAIN_SCHEMA` / :data:`PORCELAIN_VERSION`) whose top-level
``wrote_learned_state`` boolean is the single authoritative answer to "did
this run rewrite ``learned.json``".

PyYAML is required (it ships in every Hermes virtualenv); running under an
interpreter without it exits loudly rather than degrading.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# The shaper's compute/merge implementation lives in the shared package module
# ``shaping.py`` (loaded below via :func:`_load_shaping`); this script owns
# only the CLI surface. Keeping the plugin dir on ``sys.path`` preserves the
# historical standalone-import behavior for anything else loaded from here.
_PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PLUGIN_DIR))

logger = logging.getLogger("tool_belt_plugin.shape_ceiling")

#: Porcelain (machine-readable) output identity. ``PORCELAIN_VERSION`` is
#: bumped whenever a key is removed or its meaning changes; additive keys do
#: not bump it. v2 replaced the session-count window keys
#: (``scopes[].window_requested``, ``thresholds.session_window``) with the
#: day window ``window_days``.
PORCELAIN_SCHEMA = "tool-belt/shape-ceiling"
PORCELAIN_VERSION = 2


def _load_shaping():
    """The shared shaping core — ``tool_belt_plugin.shaping``.

    The compute/merge implementation lives in the package module ``shaping.py``
    (shared with the plugin runtime's in-process auto-shape engine); this
    script is the CLI wrapper. ``shaping.py`` uses package-relative imports,
    so the hyphenated plugin directory is registered as the
    ``tool_belt_plugin`` package first (mirroring ``tests/conftest.py`` /
    ``configure.py`` so already-registered module objects are reused).
    """
    name = "tool_belt_plugin"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name,
            _PLUGIN_DIR / "__init__.py",
            submodule_search_locations=[str(_PLUGIN_DIR)],
        )
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise ImportError("cannot load tool-belt package")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{name}.shaping")


# Shared shaping core, re-exported under the script's historical names so
# existing importers (configure.py's ``load_shaper()``, the test suite) keep
# addressing the same single implementation.
_shaping = _load_shaping()
DEFAULTS = _shaping.DEFAULTS
default_state_dir = _shaping.default_state_dir
load_shape_ceiling_defaults = _shaping.load_shape_ceiling_defaults
load_jsonl = _shaping.load_jsonl
group_predictions_by_scope_session = _shaping.group_predictions_by_scope_session
index_tool_calls_by_prediction = _shaping.index_tool_calls_by_prediction
compute_scope_recommendations = _shaping.compute_scope_recommendations
merge_into_learned = _shaping.merge_into_learned
load_schema_sizes = _shaping.load_schema_sizes
read_cache_mode = _shaping.read_cache_mode
index_api_call_counts = _shaping.index_api_call_counts
measured_expand_penalty = _shaping.measured_expand_penalty


def _load_plugin_config() -> dict[str, Any]:
    """The profile's plugin settings (the config.yaml layer), for CLI default
    pre-fill.

    Fail-open: returns ``{}`` when the plugin package or the host config is
    unreadable (e.g. running outside a live Hermes), so the thresholds fall
    back to the ``policy.yaml`` layer. Explicit CLI flags override either way —
    this only sets the pre-filled defaults an ad-hoc run starts from.
    """
    try:
        plugin = sys.modules.get("tool_belt_plugin")
        if plugin is None:  # pragma: no cover - package loaded at import
            return {}
        plugin._load_user_config()
        cfg = getattr(plugin, "_CONFIG", {})
        return dict(cfg) if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _activation_hint(scopes: list[str]) -> str:
    """Guidance for turning written recommendations into a live loadout.

    Hermes owns its configuration file: activation goes through the guided
    command or through ``hermes config set``. This text must never tell an
    operator to hand-edit a config file — ``scripts/configure.py`` and
    ``scripts/README.md`` forbid exactly that.
    """
    agents = sorted({(s.split(":", 1)[0] or s) for s in scopes if s}) or ["<agent>"]
    lines = [
        "\nRecommendations are written but inert until the scope's learned_mode is 'apply'.",
        "  Guided:  python3 scripts/configure.py --agent %s --mode history" % agents[0],
    ]
    if len(agents) > 1:
        lines.append("           (repeat per agent: %s)" % ", ".join(agents[1:]))
    lines.append("  Direct:  hermes config set "
                 "plugins.entries.tool-belt.settings.channels.<agent>.<platform>.learned_mode apply")
    for scope in scopes[:5]:
        lines.append("             e.g. hermes config set "
                     f"plugins.entries.tool-belt.settings.channels.{scope.replace(':', '.')}.learned_mode apply")
    return "\n".join(lines)


def build_porcelain(
    per_scope: dict[str, dict[str, Any]],
    state_dir: Path,
    thresholds: dict[str, int],
    dry_run: bool,
    changed: bool,
) -> dict[str, Any]:
    """Assemble the machine-readable run document.

    This is the shaper's only output contract. ``wrote_learned_state`` is the
    stable top-level answer callers branch on: true exactly when this run
    persisted ``learned.json`` (i.e. there was a structural change *and* the
    run was not a dry run). ``changed`` reports the recommendation delta
    independently of whether it was written, so a dry run can still say
    "changes pending".
    """
    scopes: list[dict[str, Any]] = []
    for scope in sorted(per_scope):
        recs = per_scope[scope]
        considered = int(recs.get("sessions_considered") or 0)
        scopes.append({
            "scope": scope,
            "sessions_considered": considered,
            "window_days": int(recs.get("window_days") or 0),
            "computed_at": recs.get("computed_at"),
            # The per-event expansion cost that priced this scope's demote/
            # promote economics — the MEASURED figure (or the fallback on thin
            # data). Surfaced so a run's aggressiveness is auditable and so the
            # value is verifiably identical to what auto_shape_run applies.
            "expand_round_trip_tokens": int(recs.get("expand_round_trip_tokens") or 0),
            "promote": [
                {
                    "tool": str(p.get("tool") or ""),
                    "sessions": int(p.get("sessions") or 0),
                    "calls": int(p.get("calls") or 0),
                    "evidence": str(p.get("evidence") or ""),
                }
                for p in (recs.get("promote") or [])
            ],
            "demote": [
                {
                    "tool": str(d.get("tool") or ""),
                    "sessions_without_use": int(d.get("sessions_without_use") or 0),
                    "evidence": str(d.get("evidence") or ""),
                }
                for d in (recs.get("demote") or [])
            ],
            # True when the window held too FEW SESSIONS for the demote half
            # to run at all — distinct from "ran and found nothing". (The floor
            # is in sessions; the window is in days.)
            "demote_skipped_insufficient_sessions": (
                considered < int(thresholds.get("demote_min_sessions_no_use") or 0)
            ),
        })
    return {
        "schema": PORCELAIN_SCHEMA,
        "version": PORCELAIN_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "state_dir": str(state_dir),
        "learned_path": str(state_dir / "learned.json"),
        "dry_run": bool(dry_run),
        "changed": bool(changed),
        "wrote_learned_state": bool(changed and not dry_run),
        "thresholds": {
            "window_days": int(thresholds.get("window_days") or 0),
            "promote_min_sessions": int(thresholds.get("promote_min_sessions") or 0),
            "promote_min_calls": int(thresholds.get("promote_min_calls") or 0),
            "demote_min_sessions_no_use": int(
                thresholds.get("demote_min_sessions_no_use") or 0
            ),
        },
        "scopes": scopes,
    }


def main() -> int:
    # Pre-fill flag defaults from the resolved config layers (config.yaml
    # learning.shape_ceiling over policy.yaml over DEFAULTS); explicit flags
    # below still win for ad-hoc runs.
    defaults = load_shape_ceiling_defaults(plugin_config=_load_plugin_config() or None)
    # allow_abbrev=False: the retired ``--window`` (a session COUNT) is a
    # prefix of ``--window-days``, so abbreviation matching would silently
    # reinterpret an old invocation's 50 sessions as 50 days. A unit change
    # has to fail loudly, and no flag here is long enough to be worth
    # abbreviating.
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--state-dir", default=str(default_state_dir()))
    ap.add_argument("--scope", default="", help="filter to a specific scope (default: all)")
    ap.add_argument("--window-days", type=int, default=defaults["window_days"],
                    help="recency window in days: only sessions whose last "
                         "activity falls inside it are evidence")
    ap.add_argument("--promote-min-sessions", type=int, default=defaults["promote_min_sessions"])
    ap.add_argument("--promote-min-calls", type=int, default=defaults["promote_min_calls"])
    ap.add_argument("--demote-min-sessions", type=int, default=defaults["demote_min_sessions_no_use"])
    ap.add_argument("--demote-k", type=float, default=defaults["demote_k"],
                    help="economic safety factor: demote only when carrying costs "
                         "more than k x the on-demand expansion cost (tokens)")
    ap.add_argument("--dry-run", action="store_true", help="report only, don't write learned.json")
    ap.add_argument("--json", action="store_true",
                    help="emit the porcelain JSON document on stdout instead of prose")
    ap.add_argument("--json-file", default="",
                    help="also write the porcelain JSON document to this path")
    args = ap.parse_args()

    # Prose goes to stderr under --json so stdout stays a single JSON document.
    out = sys.stderr if args.json else sys.stdout

    state_dir = Path(args.state_dir)
    thresholds = {
        "window_days": args.window_days,
        "promote_min_sessions": args.promote_min_sessions,
        "promote_min_calls": args.promote_min_calls,
        "demote_min_sessions_no_use": args.demote_min_sessions,
        "demote_k": args.demote_k,
    }

    def emit_porcelain(
        per_scope: dict[str, dict[str, Any]],
        changed: bool,
        error: str = "",
    ) -> None:
        """Write the porcelain document wherever the flags asked for it.

        Emitted on every exit path, including the empty ones, so a caller can
        always parse a document rather than special-casing an error banner.
        """
        if not args.json and not args.json_file:
            return
        doc = build_porcelain(per_scope, state_dir, thresholds, args.dry_run, changed)
        if error:
            doc["error"] = error
        text = json.dumps(doc, indent=2, sort_keys=False)
        if args.json_file:
            path = Path(args.json_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + "\n", encoding="utf-8")
        if args.json:
            print(text)

    preds = load_jsonl(state_dir / "predictions.jsonl")
    tool_calls = load_jsonl(state_dir / "tool_calls.jsonl")
    if not preds:
        print(f"No predictions.jsonl rows under {state_dir}. Nothing to shape.", file=sys.stderr)
        emit_porcelain({}, changed=False, error="no_predictions")
        return 1

    grouped = group_predictions_by_scope_session(preds)
    calls_by_pred = index_tool_calls_by_prediction(tool_calls)
    schema_sizes = load_schema_sizes(state_dir)
    api_calls = load_jsonl(state_dir / "api_calls.jsonl")
    api_counts = index_api_call_counts(api_calls)

    per_scope: dict[str, dict[str, Any]] = {}
    for scope, sessions in grouped.items():
        if args.scope and scope != args.scope:
            continue
        per_scope[scope] = compute_scope_recommendations(
            scope=scope,
            sessions=sessions,
            calls_by_pred=calls_by_pred,
            window_days=args.window_days,
            promote_min_sessions=args.promote_min_sessions,
            promote_min_calls=args.promote_min_calls,
            demote_min_sessions_no_use=args.demote_min_sessions,
            demote_k=args.demote_k,
            schema_sizes=schema_sizes,
            cache_mode=read_cache_mode(state_dir, scope),
            api_call_counts=api_counts,
            expand_round_trip_tokens=measured_expand_penalty(
                preds, api_calls, tool_calls, scope),
        )

    if not per_scope:
        print(f"No scopes matched filter {args.scope!r}.", file=sys.stderr)
        emit_porcelain({}, changed=False, error="no_matching_scopes")
        return 1

    state, changed = merge_into_learned(state_dir, per_scope, args.dry_run)
    emit_porcelain(per_scope, changed=changed)

    # ── Human-readable report. Prose only: no caller parses any of this. ──
    for scope, recs in per_scope.items():
        print(f"\n=== {scope}  (sessions_considered={recs['sessions_considered']}) ===", file=out)
        if recs["promote"]:
            print("  Promote:", file=out)
            for p in recs["promote"]:
                print(f"    + {p['tool']:<30} sessions={p['sessions']:>2}  calls={p['calls']:>3}  evidence={p['evidence']}", file=out)
        else:
            print("  Promote: (none — no tools met the threshold)", file=out)
        if recs["demote"]:
            print("  Demote:", file=out)
            for d in recs["demote"]:
                econ = ""
                if "carry_tokens" in d:
                    econ = (f"  saves≈{d['carry_tokens']}tok vs"
                            f" ≈{d['demote_tokens']}tok in round-trips (k={d.get('k')})")
                print(f"    - {d['tool']:<30} used_in={d.get('sessions_with_use', 0)}/"
                      f"{d.get('sessions_with_use', 0) + d.get('sessions_without_use', 0)}"
                      f" sessions  evidence={d['evidence']}{econ}", file=out)
        elif recs["sessions_considered"] < args.demote_min_sessions:
            print(f"  Demote: (skipped — only {recs['sessions_considered']} sessions, need ≥{args.demote_min_sessions})", file=out)
        else:
            print("  Demote: (none)", file=out)

    if args.dry_run and changed:
        print("\n[dry-run] Recommendations differ from learned.json; nothing was written.", file=out)
        print("  Re-run without --dry-run to write them.", file=out)
    elif args.dry_run:
        print("\n[dry-run] No changes — recommendations match current learned.json content.", file=out)
    elif changed:
        print(f"\nWrote updated recommendations to {state_dir / 'learned.json'}", file=out)
        print(_activation_hint(sorted(per_scope)), file=out)
    else:
        print("\nNo changes — recommendations match current learned.json content.", file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
