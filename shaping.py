"""Shared shaping core — the single implementation of the between-session
tool-loadout shaper.

Extracted from ``scripts/shape-ceiling.py`` so the operator script and the
plugin runtime (in-process auto-shaping) share one implementation instead of
drifting apart. The script remains the CLI surface (flags, porcelain JSON,
human report) as a thin wrapper over this module.

Contents:

  · Threshold config loading (``policy.yaml`` → ``learning.shape_ceiling``,
    falling back to :data:`DEFAULTS`).
  · Telemetry ingestion helpers (``load_jsonl``, grouping, call indexing).
  · :func:`compute_scope_recommendations` — the promote/demote analysis.
  · :func:`apply_recommendations` / :func:`merge_into_learned` — merging
    recommendations into learned state. All persistence routes through
    ``learned.write_state`` — the sole owner of learned-state writes.
  · :func:`auto_shape_run` — the in-process auto-shape engine, called from
    the plugin's session-end path for scopes whose ``learned_mode`` resolves
    to ``apply``.

Import contract: this module is a submodule of the ``tool_belt_plugin``
package (the hyphenated plugin directory registered under that alias).
The plugin runtime imports it as ``from . import shaping``; standalone
scripts register the package first (``scripts/_plugin_loader.py`` or the
equivalent self-registration in ``scripts/shape-ceiling.py``) and then
import ``tool_belt_plugin.shaping``. Only relative imports are used here,
so both routes resolve identically to the same module objects.
"""

from __future__ import annotations

import calendar
import json
import logging
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import learned
from .logger_io import (
    DEFAULT_PER_TOOL_TOKENS,
    load_schema_sizes,
    normalize_prediction_row,
    normalize_tool_call_row,
)
from .savings import EXPAND_ROUND_TRIP_TOKENS
from .yaml_required import require_yaml

logger = logging.getLogger("tool_belt_plugin.shaping")

# Thresholds — conservative defaults that won't fire on noise. These are the
# shaper's evidence thresholds; the auto-shape engine reuses them unchanged.
# ``session_window`` is a ceiling, not a target: the shaper uses ALL available
# session history up to this many sessions. The floor is enforced separately by
# ``demote_min_sessions_no_use`` — with fewer sessions than that, demotion never
# fires. A wider window is strictly more conservative for demotion (evidence is
# zero uses across the entire window, so one use anywhere protects the tool).
DEFAULTS = {
    "session_window": 100,
    "promote_min_sessions": 2,
    "promote_min_calls": 3,
    "demote_min_sessions_no_use": 20,
    # Economic safety factor k: demote a carried tool only when carrying it
    # costs more than k× what reaching for it on demand would. Token-
    # denominated by design — fewer tokens is always cheaper on every route,
    # so the decision never consults a price table. k also absorbs the soft
    # costs (expansion latency, the risk the model doesn't reach for
    # expand_tools when it should).
    "demote_k": 1.5,
}

#: Auto-shape per-scope debounce default: at most one auto run per scope per
#: this many hours. Overridable via ``channels.<scope>.auto_shape_interval_hours``
#: (or the top-level ``auto_shape_interval_hours``) in the plugin config.
AUTO_SHAPE_DEFAULT_INTERVAL_HOURS = 24.0

# ── Inventory-reconciliation constants ──────────────────────────
#: Grace period between a tool's first observed absence from the install's
#: registry and its automatic cleanup from learned state + config pins. A
#: tool that reappears within the grace resets the clock; after cleanup it
#: starts a fresh journey (carried, full-start).
INVENTORY_GRACE_DAYS = 7

# ── Learned trigger-overlay constants ───────────────────────────
# Automatic application demands a stricter bar than the analyzer's
# recommend-only mining defaults (support 3 / precision 0.8): candidates are
# written into live trigger behavior with no human in the loop, so only
# obviously-good patterns qualify. Design principle, Dale verbatim: "A plugin
# promising to save you tokens shouldn't quietly spend them."
#: Minimum expansion-evidence messages containing the n-gram before it can be
#: auto-applied to the overlay.
OVERLAY_MINE_MIN_SUPPORT = 4
#: Minimum precision (evidence hits / (evidence hits + noise hits)) for
#: auto-application.
OVERLAY_MINE_MIN_PRECISION = 0.90
#: At most this many auto-applied keywords per (scope, tool).
OVERLAY_MINE_MAX_KEYWORDS_PER_TOOL = 3
#: Name-token trigger derivation (source b): minimum token length and the
#: generic-token stoplist. A token must be at least this long AND not in the
#: stoplist to count as distinctive; if nothing distinctive remains the
#: derivation is skipped entirely.
OVERLAY_NAME_TOKEN_MIN_LEN = 4
OVERLAY_NAME_TOKEN_STOPLIST = frozenset({
    "get", "set", "run", "list", "create", "delete", "update", "new", "add",
    "remove", "read", "write", "tool", "tools", "make", "fetch", "find",
    "show", "start", "stop", "open", "close", "file", "files", "data",
    "info", "status", "check", "view", "item", "items", "use", "call",
    "exec", "execute", "query", "send", "post", "put", "edit", "save",
    "load", "search", "name", "text", "path", "value", "count", "help",
    "with", "from", "into", "over", "this", "that", "auto", "mode",
})

_PLUGIN_DIR = Path(__file__).resolve().parent
_POLICY_PATH = _PLUGIN_DIR / "policy.yaml"

_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _utc_iso(ts: float | None = None) -> str:
    return time.strftime(_TS_FORMAT, time.gmtime(ts if ts is not None else time.time()))


def _parse_utc_iso(value: Any) -> float | None:
    """Parse the shaper's own UTC timestamp format; None when unparsable."""
    try:
        return float(calendar.timegm(time.strptime(str(value), _TS_FORMAT)))
    except Exception:
        return None


def default_state_dir() -> Path:
    # Same resolution as learned.state_dir(); kept callable without loading
    # any state.
    return learned.state_dir()


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


def _merge_shape_defaults(overrides: dict[str, Any]) -> dict[str, Any]:
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
    k = overrides.get("demote_k")
    if k is not None:
        try:
            parsed_k = float(k)
        except Exception:
            parsed_k = 0.0
        if parsed_k > 0:
            merged["demote_k"] = parsed_k
    return merged


def read_cache_mode(state_dir: Path, scope: str) -> str | None:
    """The scope's locked prompt-cache mode ('on'/'off'), or None when unknown.

    Read from ``cache_mode_detection.json`` (written by the hot path's cache
    detector). Unknown is treated by the economics as cache-on (fewest billable
    exposures) — the conservative direction, biasing toward carrying.
    """
    try:
        with (state_dir / "cache_mode_detection.json").open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        entry = doc.get(scope)
        if isinstance(entry, dict):
            mode = entry.get("mode")
            if mode in ("on", "off"):
                return str(mode)
    except Exception:
        pass
    return None


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


def index_api_call_counts(api_calls: list[dict[str, Any]]) -> dict[str, int]:
    """Count api_calls.jsonl rows per prediction_id.

    On a cache-off route every API call re-pays the tool manifest, and an
    agentic turn runs several calls under one prediction row — the economic
    test uses these counts as billable exposures (min 1 per prediction when
    a pid has no rows, e.g. api-call logging disabled or older data).
    """
    out: Counter[str] = Counter()
    for row in api_calls:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("prediction_id") or "")
        if pid:
            out[pid] += 1
    return dict(out)


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


def _bridge_names() -> frozenset[str]:
    """Tool Search bridge tool names — the shaper's never-recommend set.

    Mirrors ``__init__._bridge_tool_names``: sourced from
    ``tools.tool_search.BRIDGE_TOOL_NAMES`` when hermes-agent is importable,
    fail-OPEN to the hardcoded triple otherwise so the guard never silently
    vanishes in offline analysis.
    """
    try:
        from tools.tool_search import BRIDGE_TOOL_NAMES  # type: ignore[import-not-found]
        names = frozenset(str(n) for n in BRIDGE_TOOL_NAMES)
        if names:
            return names
    except Exception:
        pass
    return frozenset({"tool_search", "tool_describe", "tool_call"})


def compute_scope_recommendations(
    scope: str,
    sessions: dict[str, list[dict[str, Any]]],
    calls_by_pred: dict[str, list[dict[str, Any]]],
    window: int,
    promote_min_sessions: int,
    promote_min_calls: int,
    demote_min_sessions_no_use: int,
    demote_k: float = 1.5,
    schema_sizes: dict[str, int] | None = None,
    cache_mode: str | None = None,
    api_call_counts: dict[str, int] | None = None,
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
        # Assertion: Tool Search bridge tools are pass-through (outside the
        # partition, see __init__._is_bridge_tool) and must never appear in a
        # recommendation. Once passed through they never enter the evidence
        # domains at all — this guard is defensive, for rows written before
        # the pass-through (or by a foreign writer).
        if tool_name in _bridge_names():
            logger.warning(
                "tool-belt: shaper rejecting %s candidate %r for scope %r — "
                "Tool Search bridge tools are pass-through and never shaped",
                kind, tool_name, scope,
            )
            return False
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

    # Candidates pass the anti-flap gates here; the economic side of the
    # promotion test (observed expansion spend > what carrying would cost)
    # is applied below, once billable exposures are known.
    promote_candidates: list[str] = []
    for tool_name, sids in sessions_with_tool.items():
        if len(sids) >= promote_min_sessions and calls_for_tool[tool_name] >= promote_min_calls:
            if not _valid(tool_name, "promote"):
                continue
            if tool_name in always_carry_observed:
                # A pinned tool is carried unconditionally (class A) — a
                # learned-carry promotion would be redundant noise in diffs
                # and learned.json. The evidence is usually stale: fetches
                # from before the tool was pinned. Symmetric with the demote
                # arm's always_carry exclusion.
                continue
            promote_candidates.append(tool_name)

    # ── The economic test, priced per SESSION. ────────────────────────────
    # ``expand_tools`` is sticky: once expanded, the tool rides carried for
    # the rest of the session — so a session where the tool gets used costs
    # roughly the same whether it was carried or demoted, plus one round-trip.
    # The only place demotion saves anything is sessions where the tool is
    # never reached for:
    #
    #     saving  = schema_size(tool) × billable exposures in sessions WITHOUT use
    #     penalty = EXPAND_ROUND_TRIP_TOKENS × sessions WITH use
    #     demote when saving > k × penalty;  promote when penalty > saving
    #
    # Token-denominated by design — no price table. Zero uses makes penalty 0,
    # so the old binary unused-→-demote rule falls out as the limit case.
    # Billable exposures per session follow the scope's locked prompt-cache
    # mode: cache off pays the manifest on every API call (counted from
    # api_calls.jsonl per prediction, min 1); cache on (or unknown — the
    # conservative read) roughly once per session. Trigger activations don't
    # count as use: triggers stay free for a demoted tool. Only adaptive
    # carry residents from residency-inferred rows are demotable.
    carry_observed: set[str] = set()
    tools_called: set[str] = set()
    # Sessions where each tool saw a non-trigger call (v1 rows without
    # activation_source count as such — the conservative direction, biasing
    # toward carrying). ``demand_uses`` keeps raw call counts for reporting.
    demand_use_sessions: dict[str, set[str]] = defaultdict(set)
    demand_uses: Counter[str] = Counter()
    api_counts = api_call_counts or {}
    session_exposures: dict[str, int] = {}
    for sid, plist in recent_sessions.items():
        if cache_mode == "off":
            session_exposures[sid] = sum(
                max(1, int(api_counts.get(p.get("prediction_id", ""), 1) or 1))
                for p in plist
            )
        else:
            session_exposures[sid] = 1
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
                if not name:
                    continue
                tools_called.add(name)
                if str(tc.get("activation_source") or "") != "trigger":
                    demand_uses[name] += 1
                    demand_use_sessions[name].add(sid)

    total_exposures = sum(session_exposures.values())
    sizes = schema_sizes or {}
    k = float(demote_k) if demote_k and demote_k > 0 else 1.5

    def _economics(tool_name: str, use_sessions: set[str]) -> tuple[int, int]:
        """(saving, penalty) in tokens for carrying vs expanding this tool."""
        size = int(sizes.get(tool_name, DEFAULT_PER_TOOL_TOKENS))
        exposures_with_use = sum(
            session_exposures.get(sid, 0) for sid in use_sessions
        )
        saving = size * max(0, total_exposures - exposures_with_use)
        penalty = EXPAND_ROUND_TRIP_TOKENS * len(use_sessions)
        return saving, penalty

    # ── Promotion finalization: the reversed inequality. ──────────────────
    # Promote when the expansion penalty the tool is actually paying exceeds
    # the marginal cost of carrying it (its schema in the sessions it goes
    # unused). Together with the k-scaled demote test this forms a hysteresis
    # band — between the two thresholds a tool holds its current class.
    promote: list[dict[str, Any]] = []
    for tool_name in promote_candidates:
        saving, penalty = _economics(tool_name, sessions_with_tool[tool_name])
        if penalty <= saving:
            continue  # expanding on demand is still the cheaper side — hold
        promote.append({
            "tool": tool_name,
            "sessions": len(sessions_with_tool[tool_name]),
            "calls": calls_for_tool[tool_name],
            "carry_tokens": saving,
            "expansion_tokens": penalty,
            "evidence": "expansion",
        })
    promote.sort(key=lambda x: (-int(x["sessions"]), -int(x["calls"]), str(x["tool"])))

    demote: list[dict[str, Any]] = []
    if len(recent_sessions) >= demote_min_sessions_no_use:
        # Exclude the immutable always_carry surface by construction. For v2 rows
        # the normalizer already keeps the ``carry`` residency class disjoint from
        # always_carry, so this is a no-op. For *complete v1* rows the normalizer
        # collapses every resident into ``carry`` (v1 had no immutable split), so
        # an unused always_carry baseline resident would otherwise surface as a
        # demote candidate. Subtracting the observed always_carry set here makes
        # the exclusion genuinely by construction for both schema versions.
        for tool_name in sorted(carry_observed - always_carry_observed):
            use_sessions = demand_use_sessions.get(tool_name, set())
            saving, penalty = _economics(tool_name, use_sessions)
            if saving <= k * penalty:
                continue  # carrying is (or may be) the cheaper side — hold
            if not _valid(tool_name, "demote"):
                continue
            uses = int(demand_uses.get(tool_name, 0))
            demote.append({
                "tool": tool_name,
                "sessions_without_use": len(recent_sessions) - len(use_sessions),
                "sessions_with_use": len(use_sessions),
                "uses_in_window": uses,
                "carry_tokens": saving,
                "demote_tokens": penalty,
                "k": k,
                "evidence": "carry_unused" if uses == 0 else "carry_uneconomic",
            })

    return {
        "scope": scope,
        "computed_at": _utc_iso(),
        "sessions_considered": len(recent_sessions),
        "window_requested": window,
        "promote": promote,
        "demote": demote,
        "enabled_tool_names": sorted(enabled_names),
    }


def apply_recommendations(
    state: dict[str, Any],
    per_scope: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, list[str]]]]:
    """Apply shaping recommendations to a normalized v2 state, in memory.

    For each shaped scope the recommendations are applied as *moves* across the
    adaptive ``carry`` ⇄ ``expand_only`` boundary, starting from the scope's
    current assignment:

      · a promotion moves its tool into ``carry`` and out of ``expand_only``;
      · a demotion moves its tool into ``expand_only`` and out of ``carry``.

    Promotions are applied after demotions so a tool named by both wins toward
    carrying, and any residual overlap is reconciled toward carry. Only the
    canonical v2 keys (``carry`` / ``expand_only`` / ``shaping``) are written;
    all v1 spelling stays inside ``learned.py``. Every candidate is validated
    against the scope's concrete enabled tool names — a category/toolset name
    is never written into a carrying list.

    Other scopes and all unrelated metadata (top-level and per-scope) are
    preserved. Returns ``(state, changes)`` where ``changes`` maps each
    structurally changed scope to ``{"promoted": [...], "demoted": [...]}``
    (the tools that actually moved relative to the prior assignment; both may
    be empty when only the recorded evidence changed). No write happens here.
    """
    scopes = dict(state.get("scopes") or {})

    changes: dict[str, dict[str, list[str]]] = {}
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

        changes[scope] = {
            "promoted": sorted(carry_set - prev_carry),
            "demoted": sorted(expand_set - prev_expand),
        }
        # v2 canonical fields only — no v1 mirror; learned.py owns v1 spelling.
        entry["carry"] = new_carry
        entry["expand_only"] = new_expand
        entry["shaping"] = recs
        scopes[scope] = entry

    state["scopes"] = scopes
    return state, changes


def _read_learned_doc(learned_path: Path) -> dict[str, Any]:
    if learned_path.exists():
        try:
            existing = json.loads(learned_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                return existing
        except Exception:
            pass
    return {}


def merge_into_learned(
    state_dir: Path,
    per_scope: dict[str, dict[str, Any]],
    dry_run: bool,
    source: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Merge shaping recommendations into ``learned.json`` as learned v2.

    Reads the current document, normalizes it through ``learned.normalize_state``
    (the central v1→v2 adapter), applies :func:`apply_recommendations`, and —
    when something structurally changed and this is not a dry run — persists
    through ``learned.write_state`` (atomic, fsynced, version-stamped — the
    single owner of learned-state writes). Returns the merged state and whether
    anything changed.
    """
    learned_path = state_dir / "learned.json"
    state = learned.normalize_state(_read_learned_doc(learned_path))
    state, changes = apply_recommendations(state, per_scope)
    changed = bool(changes)

    if changed:
        state["updated_at"] = _utc_iso()
        # Stamp how/when this apply happened, symmetric with the auto
        # engine's source:"auto" — a status read must not depend on which
        # arm did the applying.
        if source:
            scopes = dict(state.get("scopes") or {})
            for scope in per_scope:
                entry = dict(scopes.get(scope) or {})
                meta = entry.get("shaping")
                meta = dict(meta) if isinstance(meta, dict) else {}
                meta["source"] = source
                meta["applied_at"] = state["updated_at"]
                entry["shaping"] = meta
                scopes[scope] = entry
            state["scopes"] = scopes

    if changed and not dry_run:
        learned.write_state(state, learned_path)

    return state, changed


def effective_always_carry(plugin_config: dict[str, Any], scope: str) -> set[str]:
    """The full undemotable surface for a scope.

    Shipped policy ``always_carry`` (structural baseline) ∪ the per-agent
    config pins (``plugins.tool-belt.always_carry`` plus the scope's additive
    ``channels.<scope>.always_carry`` — resolved via
    ``learned.config_always_carry``). Never raises; a load failure degrades
    to whatever half resolved.
    """
    protected: set[str] = set()
    try:
        from . import presets
        protected |= set(presets.load_base_policy().always_carry)
    except Exception as exc:  # pragma: no cover — policy load is fail-safe
        logger.warning("tool-belt: shaper could not load policy always_carry: %s", exc)
    try:
        protected |= set(learned.config_always_carry(plugin_config or {}, scope))
    except Exception as exc:  # pragma: no cover
        logger.warning("tool-belt: shaper could not resolve config pins: %s", exc)
    return protected


def filter_protected_demotions(
    plugin_config: dict[str, Any],
    per_scope: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Strip demote candidates naming an always_carry tool (policy ∪ config).

    always_carry — including config-pinned tools — is undemotable by
    construction. ``compute_scope_recommendations`` already excludes the
    *observed* always_carry residency class, but a config pin added after the
    telemetry window (whose rows still show the tool as adaptive carry) would
    otherwise surface as a candidate. This filter is the shaper-side half of
    the guarantee; the runtime half lives in ``learned.apply_to_preset``,
    which ignores any demotion signal naming an always_carry tool. Returns a
    new per-scope mapping; inputs are not mutated.
    """
    out: dict[str, dict[str, Any]] = {}
    for scope, recs in per_scope.items():
        protected = effective_always_carry(plugin_config, scope)
        demote = list(recs.get("demote") or [])
        kept = [d for d in demote if str(d.get("tool") or "") not in protected]
        dropped = sorted(
            {str(d.get("tool") or "") for d in demote} - {str(d.get("tool") or "") for d in kept}
        )
        if dropped:
            logger.warning(
                "tool-belt: dropping demote candidate(s) %s for scope %r — "
                "always_carry (policy baseline or config pin) is undemotable",
                ", ".join(dropped), scope,
            )
        # Promotions naming a pinned tool are redundant (class A already
        # carries it) — usually stale pre-pin fetch evidence. Drop them so
        # learned.json and the confirm diff never show a pinned tool moving.
        promote = list(recs.get("promote") or [])
        kept_promote = [p for p in promote
                        if str(p.get("tool") or "") not in protected]
        new_recs = dict(recs)
        new_recs["demote"] = kept
        new_recs["promote"] = kept_promote
        out[scope] = new_recs
    return out


# ─── Inventory reconciliation (auto cleanup) ────────────────────

def registry_tool_names() -> set[str] | None:
    """The install's full tool registry — every registered tool name.

    This is the AUTHORITATIVE "does the tool exist on this install" question
    : absence from one scope's platform ceiling is NOT absence
    from the install — a tool excluded by ``platform_toolsets`` on Slack may
    be fully present on Telegram. Only the registry answers install-wide.

    Resolved the way the runtime can — Hermes' in-process ``tools.registry``
    (the same import ``_is_mcp_tool`` in ``__init__`` uses). Returns ``None``
    when the registry is unavailable or empty (outside a gateway process, or
    before registration completes): the caller must FAIL OPEN and skip
    reconciliation entirely rather than treat every tool as missing.
    """
    try:
        from tools.registry import registry  # type: ignore[import-not-found]

        names = {str(e.name) for e in registry.get_all_entries() if getattr(e, "name", "")}
        return names or None
    except Exception as exc:
        logger.debug("tool-belt: tool registry unavailable (%s) — skipping "
                     "inventory reconciliation", exc)
        return None


def _inventory_path(state_dir: Path) -> Path:
    return state_dir / "inventory.json"


def read_inventory(state_dir: Path) -> dict[str, Any]:
    """Load the reconciliation sidecar; ``{"missing_since": {tool: iso}}``."""
    try:
        doc = json.loads(_inventory_path(state_dir).read_text(encoding="utf-8"))
        if isinstance(doc, dict) and isinstance(doc.get("missing_since"), dict):
            return {"missing_since": {
                str(k): str(v) for k, v in doc["missing_since"].items()
            }}
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("tool-belt: unreadable inventory sidecar (%s) — starting fresh", exc)
    return {"missing_since": {}}


def _write_inventory(state_dir: Path, doc: dict[str, Any]) -> None:
    target = _inventory_path(state_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp",
                                    dir=str(target.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _config_pin_tools(plugin_config: dict[str, Any]) -> set[str]:
    """Every always_carry config pin, global and per-channel (all channels)."""
    pins: set[str] = set()
    raw = plugin_config.get("always_carry")
    if isinstance(raw, list):
        pins |= {str(t).strip() for t in raw if str(t).strip()}
    channels = plugin_config.get("channels")
    if isinstance(channels, dict):
        for cfg in channels.values():
            if isinstance(cfg, dict) and isinstance(cfg.get("always_carry"), list):
                pins |= {str(t).strip() for t in cfg["always_carry"] if str(t).strip()}
    return pins


def _referenced_tools(state: dict[str, Any]) -> set[str]:
    """Every concrete tool name learned state still references."""
    out: set[str] = set()
    scopes = state.get("scopes")
    if not isinstance(scopes, dict):
        return out
    for entry in scopes.values():
        if not isinstance(entry, dict):
            continue
        for key in ("carry", "expand_only"):
            for t in entry.get(key) or []:
                if str(t).strip():
                    out.add(str(t))
        shaping_meta = entry.get("shaping")
        if isinstance(shaping_meta, dict):
            for key in ("promote", "demote"):
                for row in shaping_meta.get(key) or []:
                    if isinstance(row, dict) and str(row.get("tool") or "").strip():
                        out.add(str(row["tool"]))
        for group in entry.get("triggers") or []:
            if isinstance(group, dict):
                for t in group.get("tools") or []:
                    if str(t).strip():
                        out.add(str(t))
    return out


def default_config_pin_remover(tool: str) -> bool:
    """Remove ``tool`` from the profile's always_carry config pins on disk.

    Routes through Hermes' own config-write machinery — the same primitives
    ``hermes config set``/``unset`` use in-process (``hermes_cli.config.
    get_config_path`` + ``require_readable_config_before_write`` +
    ``utils.fast_safe_load``/``atomic_yaml_write`` — see
    ``set_config_value``/``unset_config_value`` in ``hermes_cli/config.py``).
    Outside a gateway process (tests, operator scripts) those modules are not
    importable; an equivalent standalone path against ``$HERMES_HOME/
    config.yaml`` with the same atomic-replace semantics is used instead.

    Edits ``plugins.tool-belt.always_carry`` and every
    ``plugins.tool-belt.channels.<scope>.always_carry`` list. Returns True
    when the file changed. Raises on failure — the caller logs a warning and
    skips (never propagates).
    """
    yaml = require_yaml()
    loader: Any = None
    writer: Any = None
    config_path: Path | None = None
    try:
        from hermes_cli.config import (  # type: ignore[import-not-found]
            get_config_path, require_readable_config_before_write,
        )
        from utils import atomic_yaml_write, fast_safe_load  # type: ignore[import-not-found]

        config_path = get_config_path()
        require_readable_config_before_write(config_path)
        loader = fast_safe_load
        writer = lambda p, d: atomic_yaml_write(p, d, sort_keys=False)  # noqa: E731
    except (Exception, SystemExit) as exc:
        if not isinstance(exc, ImportError):
            raise RuntimeError(f"hermes config machinery refused the write: {exc}") from exc
        # Standalone fallback: same semantics, no Hermes runtime available.
        home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
        config_path = Path(home) / "config.yaml"
        loader = yaml.safe_load

        def writer(p: Path, data: Any) -> None:
            text = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
            fd, tmp_name = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp",
                                            dir=str(p.parent))
            tmp = Path(tmp_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(text)
                    f.flush()
                    os.fsync(f.fileno())
                tmp.replace(p)
            except Exception:
                tmp.unlink(missing_ok=True)
                raise

    if config_path is None or not config_path.exists():
        return False
    with config_path.open("r", encoding="utf-8") as f:
        doc = loader(f) or {}
    if not isinstance(doc, dict):
        return False

    changed = False
    plugin_cfg = (doc.get("plugins") or {}).get("tool-belt") \
        if isinstance(doc.get("plugins"), dict) else None
    if isinstance(plugin_cfg, dict):
        pins = plugin_cfg.get("always_carry")
        if isinstance(pins, list) and tool in pins:
            plugin_cfg["always_carry"] = [t for t in pins if t != tool]
            changed = True
        channels = plugin_cfg.get("channels")
        if isinstance(channels, dict):
            for cfg in channels.values():
                if isinstance(cfg, dict):
                    ch_pins = cfg.get("always_carry")
                    if isinstance(ch_pins, list) and tool in ch_pins:
                        cfg["always_carry"] = [t for t in ch_pins if t != tool]
                        changed = True
    if changed:
        writer(config_path, doc)
    return changed


def reconcile_inventory(
    plugin_config: dict[str, Any],
    state_dir: Path,
    now: float | None = None,
    registry_names: set[str] | None = None,
    config_pin_remover: Any = None,
) -> dict[str, Any]:
    """One inventory-reconciliation pass. Returns a summary.

    Detects tools that learned state or config pins still reference but the
    install's REGISTRY no longer contains (per-scope platform absence is NOT
    missing — that distinction is load-bearing). First observed absence is
    stamped into ``inventory.json``; after :data:`INVENTORY_GRACE_DAYS` of
    continuous absence the tool is pruned automatically:

      · learned.json references (carry / expand_only / shaping evidence /
        trigger overlay) via ``learned.prune_tool`` + ``learned.write_state``;
      · config always_carry pins via Hermes' own config-write machinery
        (:func:`default_config_pin_remover`) — a machinery failure logs a
        warning and skips that tool's pin, never propagates;
      · one INFO log line per removal.

    A tool that reappears in the registry within the grace resets the clock;
    after cleanup nothing references it any more, so a later reappearance
    starts a fresh journey (carried, by the full-start contract). Fail-open:
    with no resolvable registry, nothing is touched.
    """
    now = time.time() if now is None else float(now)
    summary: dict[str, Any] = {"status": "ok", "pruned": [], "tracking": []}

    if registry_names is None:
        registry_names = registry_tool_names()
    if not registry_names:
        summary["status"] = "registry_unavailable"
        return summary

    learned_path = state_dir / "learned.json"
    state = learned.normalize_state(_read_learned_doc(learned_path))
    pins = _config_pin_tools(plugin_config or {})
    referenced = _referenced_tools(state) | pins

    inventory = read_inventory(state_dir)
    missing_since: dict[str, str] = inventory["missing_since"]
    inv_changed = False

    missing_now = {t for t in referenced if t not in registry_names}

    # Reappeared (or no longer referenced anywhere): reset the clock.
    for tool in [t for t in missing_since if t not in missing_now]:
        if tool in registry_names:
            logger.info(
                "tool-belt: %r is back in the tool registry — absence clock "
                "reset; it keeps its current journey", tool,
            )
        del missing_since[tool]
        inv_changed = True

    # Newly missing: start the grace clock.
    for tool in sorted(missing_now):
        if tool not in missing_since:
            missing_since[tool] = _utc_iso(now)
            inv_changed = True

    grace_seconds = INVENTORY_GRACE_DAYS * 86400.0
    expired = sorted(
        tool for tool in missing_now
        if (first := _parse_utc_iso(missing_since.get(tool))) is not None
        and (now - first) >= grace_seconds
    )

    state_changed = False
    for tool in expired:
        state, pruned = learned.prune_tool(state, tool)
        if pruned:
            state_changed = True
            logger.info(
                "tool-belt: pruned vanished tool %r from learned state "
                "(absent from the install's registry for %d+ days)",
                tool, INVENTORY_GRACE_DAYS,
            )
        if tool in pins:
            remover = config_pin_remover or default_config_pin_remover
            try:
                if remover(tool):
                    logger.info(
                        "tool-belt: removed stale always_carry config pin %r "
                        "(tool absent from the install's registry for %d+ days)",
                        tool, INVENTORY_GRACE_DAYS,
                    )
            except Exception as exc:
                logger.warning(
                    "tool-belt: could not remove stale config pin %r via the "
                    "config-write machinery (%s) — skipped, will retry on a "
                    "later pass", tool, exc,
                )
                continue  # keep the clock so a later pass retries the pin
            # Keep the in-memory plugin config consistent for the rest of
            # this process (the on-disk source of truth was just edited).
            try:
                raw = plugin_config.get("always_carry")
                if isinstance(raw, list) and tool in raw:
                    plugin_config["always_carry"] = [t for t in raw if t != tool]
                channels = plugin_config.get("channels")
                if isinstance(channels, dict):
                    for cfg in channels.values():
                        if isinstance(cfg, dict) and isinstance(cfg.get("always_carry"), list) \
                                and tool in cfg["always_carry"]:
                            cfg["always_carry"] = [t for t in cfg["always_carry"] if t != tool]
            except Exception:
                pass
        del missing_since[tool]
        inv_changed = True
        summary["pruned"].append(tool)

    summary["tracking"] = sorted(missing_since)

    if state_changed:
        learned.write_state(state, learned_path)
    if inv_changed:
        _write_inventory(state_dir, {"missing_since": missing_since})
    return summary


# ─── Learned trigger overlay (automatic anticipation) ───────────

def name_token_keywords(tool: str) -> list[str]:
    """Conservative word-boundary keyword regexes from a tool's name tokens.

    Source (b) of the trigger overlay: when demotion moves a tool that no
    policy or overlay trigger names, derive a trigger from its distinctive
    name tokens so the tool is never invisible to deterministic anticipation.
    Tokens are split on non-alphanumerics; generic tokens are filtered via
    :data:`OVERLAY_NAME_TOKEN_STOPLIST` and the
    :data:`OVERLAY_NAME_TOKEN_MIN_LEN` minimum; returns ``[]`` (skip the
    derivation entirely) when nothing distinctive remains.
    """
    tokens = [t.lower() for t in re.split(r"[^A-Za-z0-9]+", str(tool or "")) if t]
    distinctive = [
        t for t in dict.fromkeys(tokens)
        if len(t) >= OVERLAY_NAME_TOKEN_MIN_LEN
        and t not in OVERLAY_NAME_TOKEN_STOPLIST
        and not t.isdigit()
    ]
    return [rf"\b{re.escape(t)}\b" for t in distinctive]


def _overlay_entries(scope_entry: dict[str, Any]) -> list[dict[str, Any]]:
    raw = scope_entry.get("triggers")
    return [g for g in raw if isinstance(g, dict)] if isinstance(raw, list) else []


def _tools_with_trigger_coverage(
    policy_preset: Any, scope_entry: dict[str, Any]
) -> set[str]:
    """Tools any trigger (shipped policy OR overlay) already names."""
    covered: set[str] = set()
    for group in getattr(policy_preset, "triggers", None) or []:
        covered |= {str(t) for t in getattr(group, "tools", []) or []}
    for group in _overlay_entries(scope_entry):
        covered |= {str(t) for t in group.get("tools") or []}
    return covered


def _existing_patterns_for_tool(
    policy_preset: Any, scope_entry: dict[str, Any], tool: str
) -> tuple[list[re.Pattern[str]], list[str]]:
    """(compiled keyword patterns, raw exclude patterns) covering ``tool``.

    The compiled patterns keep the miner from re-learning a keyword an
    existing trigger already matches; the raw excludes are inherited onto a
    new overlay entry for the tool so dampeners apply to overlay triggers
    exactly as they do to the policy ones.
    """
    patterns: list[re.Pattern[str]] = []
    excludes: list[str] = []
    for group in getattr(policy_preset, "triggers", None) or []:
        if tool in (getattr(group, "tools", []) or []):
            patterns.extend(getattr(group, "keyword_patterns", []) or [])
            excludes.extend(
                p.pattern for p in (getattr(group, "exclude_patterns", []) or [])
            )
    for group in _overlay_entries(scope_entry):
        if tool in (group.get("tools") or []):
            for raw in group.get("keywords") or []:
                try:
                    patterns.append(re.compile(str(raw), flags=re.IGNORECASE))
                except re.error:
                    continue
            excludes.extend(str(x) for x in group.get("exclude_keywords") or [])
    return patterns, list(dict.fromkeys(excludes))


def _expansion_previews_by_tool(
    sessions: dict[str, list[dict[str, Any]]],
    calls_by_pred: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[str]], list[str]]:
    """(per-tool expansion-evidence message previews, all previews) for a scope.

    Evidence previews are the messages of predictions under which the model
    reached a tool via ``expand_tools`` (``activated_by_expansion`` /
    ``expansion_provided_access``) — the same evidence class the promote path
    trusts. The full preview list is the noise denominator for precision.
    """
    by_tool: dict[str, list[str]] = defaultdict(list)
    all_previews: list[str] = []
    for plist in sessions.values():
        for p in plist:
            preview = str(p.get("message_preview") or "").strip()
            if not preview:
                continue
            all_previews.append(preview)
            evidence_tools: set[str] = set()
            for tc in calls_by_pred.get(str(p.get("prediction_id") or ""), []):
                if tc.get("tool_name") == "expand_tools":
                    continue
                if bool(tc.get("activated_by_expansion")) or bool(
                    tc.get("expansion_provided_access")
                ):
                    name = str(tc.get("tool_name") or "")
                    if name:
                        evidence_tools.add(name)
            for name in evidence_tools:
                by_tool[name].append(preview)
    return by_tool, all_previews


def compute_overlay_updates(
    scope: str,
    scope_entry: dict[str, Any],
    sessions: dict[str, list[dict[str, Any]]],
    calls_by_pred: dict[str, list[dict[str, Any]]],
    protected: set[str],
    newly_demoted: list[str],
    policy_preset: Any = None,
) -> list[dict[str, Any]]:
    """Compute additive overlay-trigger updates for one scope (no writes).

    Source (a) — expansion-evidence mining: the analyzer's keyword miner
    (``analyze._suggest_keywords_for_expand_only_tool`` — the exact code the
    recommend-only flow uses) is run over live expansion-evidence previews
    and auto-applied only above the STRICT bar
    (:data:`OVERLAY_MINE_MIN_SUPPORT` / :data:`OVERLAY_MINE_MIN_PRECISION`),
    at most :data:`OVERLAY_MINE_MAX_KEYWORDS_PER_TOOL` keywords per tool.

    Source (b) — name-token derivation for tools just demoted with no
    trigger coverage anywhere (:func:`name_token_keywords`).

    Structural bounds: only ``expand_only`` (demoted) tools that are NOT
    protected (``always_carry`` ∪ config pins) ever get an overlay entry —
    the overlay can only ACTIVATE expand_only tools. Returns a list of
    normalized overlay entries to merge into the scope's ``triggers`` list.
    """
    if policy_preset is None:
        try:
            from . import presets
            policy_preset = presets.load_base_policy()
        except Exception:  # pragma: no cover — policy load is fail-safe
            policy_preset = None

    expand_only = {str(t) for t in scope_entry.get("expand_only") or []}
    eligible = expand_only - protected
    if not eligible:
        return []

    updates: list[dict[str, Any]] = []
    ts = _utc_iso()

    # ── Source (a): strict-bar keyword mining from expansion evidence. ─────
    previews_by_tool, all_previews = _expansion_previews_by_tool(sessions, calls_by_pred)
    mined_tools: set[str] = set()
    try:
        from . import analyze as analyze_mod
        miner = analyze_mod._suggest_keywords_for_expand_only_tool
    except Exception as exc:  # pragma: no cover — analyzer is optional here
        logger.warning("tool-belt: overlay keyword miner unavailable: %s", exc)
        miner = None
    if miner is not None:
        for tool in sorted(eligible):
            evidence = previews_by_tool.get(tool) or []
            if len(evidence) < OVERLAY_MINE_MIN_SUPPORT:
                continue
            existing_patterns, inherited_excludes = _existing_patterns_for_tool(
                policy_preset, scope_entry, tool,
            )
            evidence_set = set(evidence)
            noise = [p for p in all_previews if p not in evidence_set]
            try:
                candidates = miner(
                    expand_only_previews=evidence,
                    noise_previews=noise,
                    existing_patterns=existing_patterns,
                    min_n=2,
                    max_n=4,
                    min_support=OVERLAY_MINE_MIN_SUPPORT,
                    min_precision=OVERLAY_MINE_MIN_PRECISION,
                    max_candidates=OVERLAY_MINE_MAX_KEYWORDS_PER_TOOL,
                )
            except Exception as exc:
                logger.warning("tool-belt: overlay mining failed for %r/%r: %s",
                               scope, tool, exc)
                continue
            keywords = [str(c.get("suggested_regex") or "") for c in candidates
                        if str(c.get("suggested_regex") or "")]
            if not keywords:
                continue
            mined_tools.add(tool)
            updates.append({
                "name": f"auto:{tool}",
                "tools": [tool],
                "keywords": keywords,
                "exclude_keywords": inherited_excludes,
                "source": "mined",
                "created_at": ts,
                "evidence": {
                    "support": len(evidence),
                    "min_precision": OVERLAY_MINE_MIN_PRECISION,
                },
            })

    # ── Source (b): name-token triggers for uncovered fresh demotions. ─────
    covered = _tools_with_trigger_coverage(policy_preset, scope_entry) | mined_tools
    for tool in sorted({str(t) for t in newly_demoted} & eligible):
        if tool in covered:
            continue
        keywords = name_token_keywords(tool)
        if not keywords:
            logger.info(
                "tool-belt: no distinctive name tokens for demoted tool %r — "
                "skipping auto trigger derivation (recoverable via "
                "expand_tools only)", tool,
            )
            continue
        updates.append({
            "name": f"auto:{tool}",
            "tools": [tool],
            "keywords": keywords,
            "exclude_keywords": [],
            "source": "name_tokens",
            "created_at": ts,
        })
    return updates


def merge_overlay_updates(
    scope_entry: dict[str, Any], updates: list[dict[str, Any]]
) -> tuple[dict[str, Any], bool]:
    """Merge computed overlay entries into a scope entry, in memory.

    One overlay entry per (name) — an update for an existing entry extends
    its ``keywords``/``exclude_keywords`` (deduplicated, order-preserving)
    rather than duplicating the group. Returns ``(entry, changed)``.
    """
    if not updates:
        return scope_entry, False
    entry = dict(scope_entry)
    overlay = [dict(g) for g in _overlay_entries(entry)]
    by_name = {str(g.get("name") or ""): g for g in overlay}
    changed = False
    for update in updates:
        existing = by_name.get(str(update.get("name") or ""))
        if existing is None:
            overlay.append(dict(update))
            by_name[str(update.get("name") or "")] = overlay[-1]
            changed = True
            continue
        for key in ("keywords", "exclude_keywords"):
            merged = list(dict.fromkeys(
                [*(existing.get(key) or []), *(update.get(key) or [])]
            ))
            if merged != list(existing.get(key) or []):
                existing[key] = merged
                changed = True
    if changed:
        entry["triggers"] = overlay
    return entry, changed


# ─── In-process auto-shape engine ───────────────────────────────────────────

def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("false", "0", "no", "off"):
        return False
    if text in ("true", "1", "yes", "on"):
        return True
    return default


def auto_shape_interval_hours(plugin_config: dict[str, Any], scope: str) -> float:
    """Resolve the per-scope debounce interval.

    Lookup mirrors ``learned.learned_mode``: the scope's channel entry wins
    (with the platform fallback from ``learned.scope_candidates``), then the
    top-level ``auto_shape_interval_hours``, then the 24h default.
    """
    candidates: list[Any] = []
    channels = plugin_config.get("channels") or {}
    if isinstance(channels, dict):
        for key in learned.scope_candidates(scope):
            cfg = channels.get(key)
            if isinstance(cfg, dict) and "auto_shape_interval_hours" in cfg:
                candidates.append(cfg.get("auto_shape_interval_hours"))
                break
    candidates.append(plugin_config.get("auto_shape_interval_hours"))
    for raw in candidates:
        if raw is None:
            continue
        try:
            parsed = float(raw)
        except Exception:
            continue
        if parsed > 0:
            return parsed
    return AUTO_SHAPE_DEFAULT_INTERVAL_HOURS


def auto_shape_run(
    plugin_config: dict[str, Any],
    state_dir: Path | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """One auto-shape pass over every eligible scope. Returns a run summary.

    Called from the plugin's session-end path (never the request/dispatch
    path). Eligibility per scope:

      · ``learned_mode`` resolves to ``apply`` — the standing consent.
        Observe/recommend scopes are never auto-written.
      · The per-scope debounce interval has elapsed since the scope's
        ``shaping.last_auto_shape_at`` (default 24h,
        ``auto_shape_interval_hours`` to override).

    Evidence thresholds are the shaper's own (``policy.yaml`` →
    ``learning.shape_ceiling``, falling back to :data:`DEFAULTS`) — the auto
    engine changes *when* shaping runs, never *what* it decides. An attempt
    that finds nothing to change still stamps ``last_auto_shape_at`` (the
    debounce record) and writes nothing else.

    Persistence and metadata: on an actual assignment change the scope's
    ``shaping`` block (learned schema v2) gains ``source: "auto"`` and
    ``applied_at``; every attempted scope gets ``last_auto_shape_at``. All
    writes go through ``learned.write_state`` (atomic rename + fsync), which
    also makes a second *process* racing this one safe — last writer wins a
    whole consistent document; no cross-process locking is built here. The
    caller (``_maybe_auto_shape`` in ``__init__``) holds an in-process lock
    so two session-ends in one gateway can't run concurrently.

    Cache-on invariant: this function only writes ``learned.json``. It never
    touches any live session state (frozen tool sets, sticky residency, cache
    mode) — a live session keeps its frozen carrying; the new assignment is
    picked up naturally when a future session resolves its preset.
    """
    now = time.time() if now is None else float(now)
    summary: dict[str, Any] = {
        "ran": False,
        "attempted": [],
        "applied": {},
        "reason": "",
    }

    if not _coerce_bool(plugin_config.get("auto_shape"), default=True):
        summary["reason"] = "auto_shape_disabled"
        return summary

    sd = Path(state_dir) if state_dir is not None else default_state_dir()

    # Inventory reconciliation runs on every auto pass (before the
    # evidence gates — cleanup is due even when no new telemetry landed).
    # Fail-open: a reconciliation problem never blocks shaping.
    try:
        summary["inventory"] = reconcile_inventory(plugin_config, sd, now=now)
    except Exception as exc:
        logger.warning("tool-belt: inventory reconciliation failed (fail-open): %s", exc)
        summary["inventory"] = {"status": "error"}

    preds = load_jsonl(sd / "predictions.jsonl")
    if not preds:
        summary["reason"] = "no_predictions"
        return summary

    grouped = group_predictions_by_scope_session(preds)

    learned_path = sd / "learned.json"
    state = learned.normalize_state(_read_learned_doc(learned_path))
    scopes_state = state.get("scopes") or {}

    eligible: list[str] = []
    for scope in sorted(grouped):
        if learned.learned_mode(plugin_config, scope) != "apply":
            continue  # no standing consent — never auto-write
        entry = scopes_state.get(scope) or {}
        shaping_meta = entry.get("shaping") if isinstance(entry, dict) else {}
        last = None
        if isinstance(shaping_meta, dict):
            last = _parse_utc_iso(shaping_meta.get("last_auto_shape_at"))
        interval_s = auto_shape_interval_hours(plugin_config, scope) * 3600.0
        if last is not None and (now - last) < interval_s:
            continue  # debounced
        eligible.append(scope)

    if not eligible:
        summary["reason"] = "no_eligible_scopes"
        return summary

    thresholds = load_shape_ceiling_defaults()
    calls_by_pred = index_tool_calls_by_prediction(load_jsonl(sd / "tool_calls.jsonl"))
    schema_sizes = load_schema_sizes(sd)
    api_counts = index_api_call_counts(load_jsonl(sd / "api_calls.jsonl"))

    per_scope = {
        scope: compute_scope_recommendations(
            scope=scope,
            sessions=grouped[scope],
            calls_by_pred=calls_by_pred,
            window=thresholds["session_window"],
            promote_min_sessions=thresholds["promote_min_sessions"],
            promote_min_calls=thresholds["promote_min_calls"],
            demote_min_sessions_no_use=thresholds["demote_min_sessions_no_use"],
            demote_k=thresholds["demote_k"],
            schema_sizes=schema_sizes,
            cache_mode=read_cache_mode(sd, scope),
            api_call_counts=api_counts,
        )
        for scope in eligible
    }

    # always_carry (policy ∪ config pins) is undemotable by construction.
    per_scope = filter_protected_demotions(plugin_config, per_scope)

    state, changes = apply_recommendations(state, per_scope)

    # Explicit assertion of the immunity contract: nothing protected was
    # written toward expand_only by this run.
    for scope, delta in changes.items():
        protected = effective_always_carry(plugin_config, scope)
        leaked = protected & set(delta.get("demoted") or [])
        assert not leaked, (
            f"tool-belt: auto-shape demoted protected always_carry tool(s) "
            f"{sorted(leaked)} in scope {scope!r}"
        )
    ts_iso = _utc_iso(now)

    # Learned trigger overlay — automatic anticipation. Additive,
    # per scope, activation-only; fail-open so an overlay problem never
    # blocks the shaping write itself.
    overlay_changed: dict[str, int] = {}
    try:
        from . import presets as presets_mod
        policy_preset = presets_mod.load_base_policy()
        scopes_now = state.get("scopes") or {}
        for scope in eligible:
            entry = scopes_now.get(scope)
            if not isinstance(entry, dict):
                continue
            protected = effective_always_carry(plugin_config, scope)
            newly_demoted = list((changes.get(scope) or {}).get("demoted") or [])
            updates = compute_overlay_updates(
                scope=scope,
                scope_entry=entry,
                sessions=grouped[scope],
                calls_by_pred=calls_by_pred,
                protected=protected,
                newly_demoted=newly_demoted,
                policy_preset=policy_preset,
            )
            new_entry, did_change = merge_overlay_updates(entry, updates)
            if did_change:
                scopes_now = dict(scopes_now)
                scopes_now[scope] = new_entry
                overlay_changed[scope] = len(updates)
                logger.info(
                    "tool-belt: auto-learned %d trigger overlay update(s) for %s: %s",
                    len(updates), scope,
                    ", ".join(sorted({str(u.get("name")) for u in updates})),
                )
        if overlay_changed:
            state["scopes"] = scopes_now
    except Exception as exc:
        logger.warning("tool-belt: trigger-overlay learning failed (fail-open): %s", exc)

    scopes = dict(state.get("scopes") or {})
    for scope in eligible:
        entry = dict(scopes.get(scope) or {"carry": [], "expand_only": [], "shaping": {}})
        shaping_meta = entry.get("shaping")
        shaping_meta = dict(shaping_meta) if isinstance(shaping_meta, dict) else {}
        shaping_meta["last_auto_shape_at"] = ts_iso
        if scope in changes:
            shaping_meta["source"] = "auto"
            shaping_meta["applied_at"] = ts_iso
        entry["shaping"] = shaping_meta
        scopes[scope] = entry
    state["scopes"] = scopes
    if changes or overlay_changed:
        state["updated_at"] = ts_iso

    # One write covers both the applied assignments and the debounce stamps.
    learned.write_state(state, learned_path)

    for scope, delta in changes.items():
        promoted = delta.get("promoted") or []
        demoted = delta.get("demoted") or []
        logger.info(
            "tool-belt: auto-shaped %s: +%d carry%s, -%d to expand_only%s",
            scope,
            len(promoted),
            f" ({', '.join(promoted)})" if promoted else "",
            len(demoted),
            f" ({', '.join(demoted)})" if demoted else "",
        )

    summary.update({
        "ran": True,
        "attempted": list(eligible),
        "applied": changes,
        "overlay": overlay_changed,
        "reason": "ok",
    })
    return summary
