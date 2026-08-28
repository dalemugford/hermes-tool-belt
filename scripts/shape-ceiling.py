#!/usr/bin/env python3
"""Between-session tool-loadout shaper.

The principle in one line: usage-aware tool loading at per-tool granularity.
Adjustments happen at session boundaries (free cache moments), not per turn.

This script reads ``predictions.jsonl`` and ``tool_calls.jsonl`` for the
last N sessions per scope, then writes per-scope promote/demote
recommendations into ``learned.json``. The plugin's existing
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
    across enough recent sessions. Moving them to ``expand_only`` keeps
    the carried prefix tighter without losing real capability (they stay
    reachable via ``expand_tools`` / triggers). The immutable
    ``always_carry`` surface is never a demote candidate. Conservative
    thresholds — easy to revert via a single use.

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
  python3 scripts/shape-ceiling.py --window 50          # consider last 50 sessions per scope
  python3 scripts/shape-ceiling.py --json               # porcelain document on stdout
  python3 scripts/shape-ceiling.py --json-file out.json # porcelain document to a file

Threshold defaults come from ``policy.yaml`` under
``learning.shape_ceiling``. CLI flags still win when passed explicitly.

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
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Read every telemetry row through the centralized v1/v2 normalizer so the
# shaper sees one canonical shape regardless of the on-disk schema version.
# The normalizer also owns the residency reconstruction (``residency_inferred``)
# the demote path depends on. Importing it puts the plugin dir on ``sys.path``
# first; the module has no import-time relative deps, so it loads standalone.
_PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PLUGIN_DIR))
from logger_io import normalize_prediction_row, normalize_tool_call_row  # noqa: E402
from yaml_required import require_yaml  # noqa: E402

logger = logging.getLogger("tool_belt_plugin.shape_ceiling")

#: Porcelain (machine-readable) output identity. ``PORCELAIN_VERSION`` is
#: bumped whenever a key is removed or its meaning changes; additive keys do
#: not bump it.
PORCELAIN_SCHEMA = "tool-belt/shape-ceiling"
PORCELAIN_VERSION = 1


def _load_learned():
    """The plugin's ``learned`` module — sole owner of learned-state I/O.

    ``learned.py`` uses package-relative imports, so the hyphenated plugin
    directory is registered as the ``tool_belt_plugin`` package first
    (mirroring ``tests/conftest.py`` / ``configure.py`` so already-registered
    module objects are reused).
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
    return importlib.import_module(f"{name}.learned")

# The immutable always_carry surface is *never* a shaping target: shaping only
# moves enabled built-ins between the adaptive ``carry`` and ``expand_only``
# classes. always_carry is excluded from demotion by construction (candidates
# are drawn only from the ``carry`` residency class, which the normalizer keeps
# disjoint from always_carry) and pinned by an explicit assertion in
# :func:`compute_scope_recommendations`.

# Thresholds — conservative defaults that won't fire on noise.
DEFAULTS = {
    "session_window": 20,
    "promote_min_sessions": 2,
    "promote_min_calls": 3,
    "demote_min_sessions_no_use": 20,
}

_POLICY_PATH = Path(__file__).resolve().parent.parent / "policy.yaml"


def default_state_dir() -> Path:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(home) / "state" / "tool-belt"


def load_shape_ceiling_defaults(policy_path: Path = _POLICY_PATH) -> dict[str, int]:
    """Load shaper defaults from ``policy.yaml``'s ``learning.shape_ceiling``.

    PyYAML is required: :func:`yaml_required.require_yaml` exits loudly when
    it is missing rather than degrading to a second, divergent parser. A
    missing or malformed policy file still falls back to :data:`DEFAULTS` —
    that is a policy-content question, not a wrong-interpreter one.
    """
    yaml = require_yaml()
    try:
        raw = policy_path.read_text(encoding="utf-8")
    except Exception:
        return dict(DEFAULTS)

    try:
        loaded = yaml.safe_load(raw) or {}
    except Exception:
        return dict(DEFAULTS)
    if not isinstance(loaded, dict):
        return dict(DEFAULTS)

    learning = loaded.get("learning")
    if isinstance(learning, dict):
        shape = learning.get("shape_ceiling")
        if isinstance(shape, dict):
            return _merge_shape_defaults(shape)
    return dict(DEFAULTS)


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
    """Returns {scope: {session_id: [normalized preds-in-order]}}.

    Every row is passed through :func:`normalize_prediction_row` first, so the
    shaper works exclusively against the canonical v2 shape (``carry_tools``,
    ``expand_only_tools``, ``residency`` / ``residency_inferred``, …). The
    normalizer preserves the scope/session/prediction identity fields the
    grouping reads, so v1, v2, and mixed streams group identically.
    """
    out: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for raw in preds:
        p = normalize_prediction_row(raw if isinstance(raw, dict) else {})
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
    """Index normalized tool-call rows by their ``prediction_id``.

    Rows are normalized through :func:`normalize_tool_call_row` so the promote
    path reads the canonical v2 expansion flags (``activated_by_expansion`` /
    ``expansion_provided_access``) regardless of the on-disk version.
    """
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in tool_calls:
        t = normalize_tool_call_row(raw if isinstance(raw, dict) else {})
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
    """Per-scope promote/demote analysis over canonical v2 rows.

    Promote evidence: a tool-call row whose ``activated_by_expansion`` or
    ``expansion_provided_access`` flag is set — direct evidence the model reached
    for an ``expand_only`` tool via ``expand_tools`` (an in-turn expansion or a
    sticky-carried use of an earlier one). Trigger activation is *never* promote
    evidence: a trigger-activated expand_only tool carries neither flag. Count
    distinct sessions per tool; promote when ≥ ``promote_min_sessions`` distinct
    sessions AND ≥ ``promote_min_calls`` total calls.

    Demote evidence: an adaptive ``carry`` resident (partition class C) that went
    unused across the window. Only rows whose residency the normalizer could
    reconstruct (``residency_inferred``) contribute carry candidates — a sparse
    v1 row cannot drive demotion. always_carry is excluded by construction (it is
    a distinct residency class) and pinned by an assertion below. Only fires when
    the window has ≥ ``demote_min_sessions_no_use`` sessions.

    Every candidate is validated against the concrete enabled tool names observed
    for the scope; a toolset/category-only value can never be stored in a per-tool
    carrying list (it is dropped with a warning).
    """
    session_ids_ordered = sorted(sessions.keys(), key=lambda sid: -max(
        (p.get("ts", 0) for p in sessions[sid]), default=0
    ))
    recent_session_ids = session_ids_ordered[:window]
    recent_sessions = {sid: sessions[sid] for sid in recent_session_ids}

    # Concrete enabled tool names seen for THIS scope — the validation domain
    # for candidate names. Union of every prediction tool list plus every
    # tool-call name observed under this scope's own predictions (never the
    # global call index: another agent's tools must not validate into this
    # scope's carrying lists). A category/toolset name never appears here (it
    # is a grouping key, not a concrete tool), so validating against this set
    # rejects one.
    enabled_names: set[str] = set()
    # always_carry residents observed — the demote assertion's forbidden set.
    always_carry_observed: set[str] = set()
    for plist in recent_sessions.values():
        for p in plist:
            for field in (
                "ceiling_tools", "always_carry_tools", "carry_tools",
                "expand_only_tools", "active_tools",
            ):
                for t in (p.get(field) or []):
                    enabled_names.add(str(t))
            for t in (p.get("always_carry_tools") or []):
                always_carry_observed.add(str(t))
            for tc in calls_by_pred.get(p.get("prediction_id", ""), []):
                name = str(tc.get("tool_name") or "")
                if name:
                    enabled_names.add(name)

    def _valid(tool_name: str, kind: str) -> bool:
        if tool_name and tool_name in enabled_names:
            return True
        logger.warning(
            "tool-belt: shaper rejecting %s candidate %r for scope %r — not a "
            "concrete enabled tool name (category/toolset names are never stored "
            "in a per-tool carrying list)",
            kind, tool_name, scope,
        )
        return False

    # ── Promote signals (expand_only → carry). ────────────────────────────
    sessions_with_tool: dict[str, set[str]] = defaultdict(set)
    calls_for_tool: Counter[str] = Counter()
    for sid, plist in recent_sessions.items():
        for p in plist:
            pid = p.get("prediction_id", "")
            for tc in calls_by_pred.get(pid, []):
                if tc.get("tool_name") == "expand_tools":
                    continue
                # Direct expansion/recovery evidence only. These flags are set
                # exclusively when the tool was reached via expand_tools; a
                # trigger activation leaves them unset.
                evidence = (
                    bool(tc.get("activated_by_expansion"))
                    or bool(tc.get("expansion_provided_access"))
                )
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
            if not _valid(tool_name, "promote"):
                continue
            promote.append({
                "tool": tool_name,
                "sessions": len(sids),
                "calls": calls_for_tool[tool_name],
                "evidence": "expansion",
            })
    promote.sort(key=lambda x: (-int(x["sessions"]), -int(x["calls"]), str(x["tool"])))

    # ── Demote signals (carry → expand_only). ─────────────────────────────
    # Only adaptive carry residents from residency-inferred rows are demotable.
    carry_observed: set[str] = set()
    tools_called: set[str] = set()
    for sid, plist in recent_sessions.items():
        for p in plist:
            if p.get("residency_inferred"):
                residency = p.get("residency") or {}
                carry_source = residency.get("carry") if isinstance(residency, dict) else None
                if carry_source is None:
                    carry_source = p.get("carry_tools") or []
                for t in carry_source:
                    carry_observed.add(str(t))
            pid = p.get("prediction_id", "")
            for tc in calls_by_pred.get(pid, []):
                name = str(tc.get("tool_name") or "")
                if name:
                    tools_called.add(name)

    demote: list[dict[str, Any]] = []
    if len(recent_sessions) >= demote_min_sessions_no_use:
        # Exclude the immutable always_carry surface by construction. For v2 rows
        # the normalizer already keeps the ``carry`` residency class disjoint from
        # always_carry, so this is a no-op. For *complete v1* rows the normalizer
        # collapses every resident into ``carry`` (v1 had no immutable split), so
        # an unused always_carry baseline resident would otherwise surface as a
        # demote candidate. Subtracting the observed always_carry set here makes
        # the exclusion genuinely by construction for both schema versions.
        demote_candidates = (carry_observed - tools_called) - always_carry_observed
        for tool_name in sorted(demote_candidates):
            if not _valid(tool_name, "demote"):
                continue
            demote.append({
                "tool": tool_name,
                "sessions_without_use": len(recent_sessions),
                "evidence": "carry_unused",
            })

    return {
        "scope": scope,
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sessions_considered": len(recent_sessions),
        "window_requested": window,
        "promote": promote,
        "demote": demote,
        "enabled_tool_names": sorted(enabled_names),
    }


def merge_into_learned(
    state_dir: Path,
    per_scope: dict[str, dict[str, Any]],
    dry_run: bool,
) -> tuple[dict[str, Any], bool]:
    """Merge shaping recommendations into ``learned.json`` as learned v2.

    For each shaped scope the recommendations are applied as *moves* across the
    adaptive ``carry`` ⇄ ``expand_only`` boundary, starting from the scope's
    current assignment:

      · a promotion moves its tool into ``carry`` and out of ``expand_only``;
      · a demotion moves its tool into ``expand_only`` and out of ``carry``.

    Promotions are applied after demotions so a tool named by both wins toward
    carrying, and any residual overlap is reconciled toward carry. The current
    assignment is read through ``learned.normalize_state`` — the central v1→v2
    adapter — and only the canonical v2 keys (``carry`` / ``expand_only`` /
    ``shaping``) are written; all v1 spelling stays inside ``learned.py``.
    Every candidate is validated against the scope's concrete enabled tool
    names — a category/toolset name is never written into a carrying list.

    Other scopes and all unrelated metadata (top-level and per-scope) are
    preserved, normalized to the v2 shape on write. Persistence goes through
    ``learned.write_state`` (atomic, fsynced, version-stamped — the single
    owner of learned-state writes). Returns the merged state and whether
    anything changed.
    """
    learned = _load_learned()
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

    state = learned.normalize_state(existing)  # central v1→v2 adapter
    scopes = dict(state.get("scopes") or {})

    changed = False
    for scope, recs in per_scope.items():
        entry = dict(scopes.get(scope) or {})
        enabled_names = set(recs.get("enabled_tool_names") or [])

        def _accept(tool: str, kind: str) -> bool:
            # An empty enabled ceiling means no candidate can be proven eligible:
            # there is no concrete enabled tool set to validate against, so refuse
            # every candidate (defense-in-depth for hand-built recs that bypass
            # ``compute``, whose ``_valid`` already drops candidates when the
            # enabled set is empty). Otherwise a candidate must be a concrete
            # enabled tool name — a category/toolset name never enters a per-tool
            # carrying list.
            if enabled_names and tool and tool in enabled_names:
                return True
            logger.warning(
                "tool-belt: refusing to write %s candidate %r into scope %r "
                "carrying list — not a concrete enabled tool name",
                kind, tool, scope,
            )
            return False

        # The entry is already normalized to v2 (carry/expand_only/shaping).
        carry_set = {str(t) for t in (entry.get("carry") or []) if str(t).strip()}
        expand_set = {str(t) for t in (entry.get("expand_only") or []) if str(t).strip()}
        prev_carry, prev_expand = set(carry_set), set(expand_set)

        demote_tools = [str(d.get("tool") or "") for d in (recs.get("demote") or [])]
        promote_tools = [str(p.get("tool") or "") for p in (recs.get("promote") or [])]

        # Demotions first (carry → expand_only) …
        for tool in demote_tools:
            if not _accept(tool, "demote"):
                continue
            expand_set.add(tool)
            carry_set.discard(tool)
        # … then promotions (expand_only → carry), so carrying wins any tie.
        for tool in promote_tools:
            if not _accept(tool, "promote"):
                continue
            carry_set.add(tool)
            expand_set.discard(tool)
        # Reconcile any residual overlap toward carry (belt-and-braces).
        expand_set -= carry_set

        new_carry = sorted(carry_set)
        new_expand = sorted(expand_set)

        # Structural change detection ignores the recomputed timestamp.
        prev_sig = (
            sorted(prev_carry), sorted(prev_expand),
            [(p.get("tool"), p.get("sessions"), p.get("calls"))
             for p in ((entry.get("shaping") or {}).get("promote") or [])],
            [(d.get("tool"), d.get("sessions_without_use"))
             for d in ((entry.get("shaping") or {}).get("demote") or [])],
        )
        new_sig = (
            new_carry, new_expand,
            [(p["tool"], p["sessions"], p["calls"]) for p in recs["promote"]],
            [(d["tool"], d["sessions_without_use"]) for d in recs["demote"]],
        )
        if prev_sig == new_sig:
            continue  # no change for this scope

        changed = True
        # v2 canonical fields only — no v1 mirror; learned.py owns v1 spelling.
        entry["carry"] = new_carry
        entry["expand_only"] = new_expand
        entry["shaping"] = recs
        scopes[scope] = entry

    state["scopes"] = scopes
    if changed:
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if changed and not dry_run:
        # learned.write_state is the single owner of learned-state persistence:
        # unique-temp atomic write, fsync, normalize-on-write, v2 version stamp.
        learned.write_state(state, learned_path)

    return state, changed


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
        "  Guided:  python3 scripts/configure.py --agent %s --path shape" % agents[0],
    ]
    if len(agents) > 1:
        lines.append("           (repeat per agent: %s)" % ", ".join(agents[1:]))
    lines.append("  Direct:  hermes config set "
                 "plugins.tool-belt.channels.<scope>.learned_mode apply")
    for scope in scopes[:5]:
        lines.append("             e.g. hermes config set "
                     f"plugins.tool-belt.channels.{scope}.learned_mode apply")
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
            "window_requested": int(recs.get("window_requested") or 0),
            "computed_at": recs.get("computed_at"),
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
            # True when the window was too short for the demote half to run at
            # all — distinct from "ran and found nothing".
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
            "session_window": int(thresholds.get("session_window") or 0),
            "promote_min_sessions": int(thresholds.get("promote_min_sessions") or 0),
            "promote_min_calls": int(thresholds.get("promote_min_calls") or 0),
            "demote_min_sessions_no_use": int(
                thresholds.get("demote_min_sessions_no_use") or 0
            ),
        },
        "scopes": scopes,
    }


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
    ap.add_argument("--json", action="store_true",
                    help="emit the porcelain JSON document on stdout instead of prose")
    ap.add_argument("--json-file", default="",
                    help="also write the porcelain JSON document to this path")
    args = ap.parse_args()

    # Prose goes to stderr under --json so stdout stays a single JSON document.
    out = sys.stderr if args.json else sys.stdout

    state_dir = Path(args.state_dir)
    thresholds = {
        "session_window": args.window,
        "promote_min_sessions": args.promote_min_sessions,
        "promote_min_calls": args.promote_min_calls,
        "demote_min_sessions_no_use": args.demote_min_sessions,
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
                print(f"    - {d['tool']:<30} sessions_without_use={d['sessions_without_use']}  evidence={d['evidence']}", file=out)
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
