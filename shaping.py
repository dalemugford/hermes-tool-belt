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
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import learned
from .logger_io import normalize_prediction_row, normalize_tool_call_row
from .yaml_required import require_yaml

logger = logging.getLogger("tool_belt_plugin.shaping")

# Thresholds — conservative defaults that won't fire on noise. These are the
# shaper's evidence thresholds; the auto-shape engine reuses them unchanged.
DEFAULTS = {
    "session_window": 20,
    "promote_min_sessions": 2,
    "promote_min_calls": 3,
    "demote_min_sessions_no_use": 20,
}

#: Auto-shape per-scope debounce default: at most one auto run per scope per
#: this many hours. Overridable via ``channels.<scope>.auto_shape_interval_hours``
#: (or the top-level ``auto_shape_interval_hours``) in the plugin config.
AUTO_SHAPE_DEFAULT_INTERVAL_HOURS = 24.0

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

    if changed and not dry_run:
        learned.write_state(state, learned_path)

    return state, changed


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

    per_scope = {
        scope: compute_scope_recommendations(
            scope=scope,
            sessions=grouped[scope],
            calls_by_pred=calls_by_pred,
            window=thresholds["session_window"],
            promote_min_sessions=thresholds["promote_min_sessions"],
            promote_min_calls=thresholds["promote_min_calls"],
            demote_min_sessions_no_use=thresholds["demote_min_sessions_no_use"],
        )
        for scope in eligible
    }

    state, changes = apply_recommendations(state, per_scope)
    ts_iso = _utc_iso(now)

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
    if changes:
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
        "reason": "ok",
    })
    return summary
