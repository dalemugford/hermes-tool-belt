"""dynamic-tools — narrow tool definitions per message based on detected intent.

ARCHITECTURE NOTE — narrowing layer:
  We patch ``AIAgent._build_api_kwargs`` (called once per outbound API
  request), NOT ``__init__`` or ``get_tool_definitions``. Hermes caches
  AIAgent instances per session for up to 1 hour idle, so __init__ runs
  only on first message; subsequent messages reuse the cached agent.
  Per-call narrowing in _build_api_kwargs catches every message correctly.

  ``self.tools`` is left as the full ceiling — we filter the kwargs that
  go on the wire. Cleanest separation: agent state stays canonical;
  only the per-request payload is narrowed.

Lifecycle:
  · register(ctx)
      - install monkey-patch (AIAgent._build_api_kwargs across all bindings)
      - register the expand_tools meta-tool
      - register hooks (pre_gateway_dispatch, post_tool_call, on_session_end)

  · pre_gateway_dispatch(event, ...)
      - run predictor on message text + attachments
      - stash prediction state in a contextvars.ContextVar
      - state is mutable: expand_tools mutates the same dict to add expansions

  · patched _build_api_kwargs(self, api_messages)
      - call original to get the full kwargs
      - read prediction state from contextvar
      - filter kwargs["tools"] down to (always_on ∪ triggered ∪ expanded) ∩ ceiling
      - on first invocation per turn: write the prediction row to predictions.jsonl
      - return the filtered kwargs

  · post_tool_call(tool_name, ...)
      - append row to tool_calls.jsonl with prediction_id linkage

Failure mode: any exception anywhere falls through to no-narrowing.
"""

from __future__ import annotations

import contextvars
import hashlib
import logging
import os
import sys
import time
from typing import Any

from . import expand_tools as expand_tools_mod
from . import logger_io
from . import predictor as predictor_mod
from . import presets as presets_mod
from .presets import WILDCARD_ALWAYS_ON

logger = logging.getLogger(__name__)


# ─── Module state ──────────────────────────────────────────────────────────

# Per-message prediction state. Set by pre_gateway_dispatch, consumed by
# every _build_api_kwargs call within the message's reasoning loop.
# The dict IS mutable — expand_tools writes into it; subsequent
# _build_api_kwargs calls see the additions.
_PREDICTION_CV: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "dynamic_tools_prediction", default=None,
)

_CONFIG: dict[str, Any] = {
    "enabled": False,
    "log": True,
    "channels": {},
    "always_on_extra": [],
    "always_off": [],
    "agent": "",
    "learned_mode": "off",
    "sticky": {
        "enabled": True,
        "ttl_turns": 3,
        # Toolset names passed to expand_tools(category=...), NOT YAML trigger
        # group names. policy.yaml's trigger groups are named differently
        # (e.g. file_write, web_extract); sticky residency keys on the
        # model-facing toolset name resolved through Hermes' toolset table.
        "categories": ["terminal", "file", "browser", "web", "code_execution", "delegation"],
    },
    "predictor": {
        # Number of prior user messages from the same session to prepend to
        # the current message before regex classification. Catches signal in
        # short replies like "yes" / "go ahead" / "no, don't" whose intent
        # lives in the preceding turn. 0 disables, restoring the original
        # single-message behavior.
        "lookback_turns": 1,
    },
    # Fraction of sessions (deterministic by session_id hash) to bypass
    # narrowing entirely, providing an A/B baseline cohort. Bypassed
    # predictions still run for telemetry (triggers_fired, etc.) but
    # allowed_tool_names is set to WILDCARD and policy_source is "bypass".
    # 0.0 disables. Override per scope via channels.<scope>.bypass_rate.
    "bypass_rate": 0.0,
    # Cache-aware mode (Phase 0 of the cache-aware refactor — observation
    # only; no behavior change yet). Determines whether tools may be
    # mutated mid-session.
    #   auto — observe cache_read_tokens for the first few turns and
    #          self-select on|off. Bias the default to "on" once the
    #          empirical window concludes.
    #   on   — assume provider prefix-caching is active; freeze tools at
    #          session start (Phase 1 will honour this).
    #   off  — assume caching is inactive; per-turn narrowing (current
    #          behavior, kept alive for kimi/gpt-5.4-mini class providers).
    "cache_mode": "auto",
    "cache_auto": {
        # Observe at least this many API calls before locking the mode.
        # 5 (not 3) because call 1 of a turn is consistently cache-cold
        # (~30% hit rate even on cache-on providers); 3 turns of mixed
        # cold/warming would lock noise.
        "detect_calls": 5,
        # Also require this much accumulated input to lock. A 3-turn
        # "hi / what time / ok" opener has tiny inputs and a collapsed
        # hit-rate denominator — the resulting ratio is noise. Wait
        # until there's enough volume for the signal to be real.
        "detect_min_input_tokens": 5000,
        # hit_rate ≥ on_threshold → lock "on"; else lock "off". Phase 3
        # may switch this to a fresh-tokens-per-call quantity (the actual
        # economic driver) but ratio is sufficient for Phase 0 baseline.
        "on_threshold": 0.40,
        # Providers/models with known-bad provider-side caching skip the
        # detection window and lock "off" immediately. From the state.db
        # 14-day rollup: kimi at 2.5%, gpt-5.4-mini at 2.9%. Don't waste
        # 5 calls of observation to re-derive what we already measured.
        "providers_off_models": ["kimi-k2.6:cloud", "gpt-5.4-mini"],
    },
}

_ORIGINAL_BUILD_API_KWARGS: Any = None
_ORIGINAL_COMPRESS_CONTEXT: Any = None
_PATCHED = False
# Ephemeral sticky-residency state, keyed by sticky_key (a session-scoped
# opaque token derived from session_id — NOT the coarse policy scope).
# One entry per session; cleared on on_session_end.
_STICKY_BY_KEY: dict[str, dict[str, dict[str, Any]]] = {}
# Turn counter keyed by policy_scope (agent:platform) for telemetry only.
# NOT used for sticky residency decisions.
_POLICY_TURN_BY_SCOPE: dict[str, int] = {}
# Per-session ring of recent user-message texts for lookback prediction.
# Keyed by session_id. Cleared on on_session_end.
_PRIOR_MESSAGES_BY_SESSION: dict[str, list[str]] = {}

# ─── Cache-aware state (Phase 0: observation only, no behavior change) ────
# Frozen tool-set decisions per canonical session_key. Populated by Phase 1;
# Phase 0 leaves it empty but reserves the structure + eviction wiring so
# the test surface stays stable. Eviction: on_session_end, on_session_reset,
# and (Phase 4) on compaction.
_FROZEN_BY_SESSION: dict[str, dict[str, Any]] = {}

# Cross-session detection cache. Persisted JSON at
# ``~/.hermes/state/dynamic-tools/cache_mode_detection.json``. Keyed by
# scope (``agent:platform``) — model identity isn't reliable at
# pre_gateway_dispatch (set later by the AIAgent) and scope is the
# right granularity for "does this profile's provider cache?". Values:
#   {scope: {"mode": "on"|"off", "locked_at": ts, "lock_reason": str,
#            "hit_rate_at_lock": float, "sessions_locked": int,
#            "last_model": str}}
# Loaded once on register(); written on every per-session lock event.
_DETECTION_CACHE: dict[str, dict[str, Any]] = {}
_DETECTION_CACHE_LOADED = False

# Cache-mode detection state per canonical session_key. Each entry:
#   {
#     "mode": "pending" | "on" | "off",
#     "calls_observed": int,
#     "cache_read": int, "cache_write": int, "input": int,
#     "locked_at_call": int | None,
#     "lock_reason": str,  # "provider_blocklist" | "threshold_met" |
#                          # "threshold_failed" | "forced_on" | "forced_off"
#   }
# Populated by _on_post_api_request; consumed by Phase 1 freeze decision.
_CACHE_MODE_BY_SESSION: dict[str, dict[str, Any]] = {}


# ─── Config loading ────────────────────────────────────────────────────────

def _load_user_config() -> None:
    """Merge user config from ~/.hermes/config.yaml over the defaults."""
    try:
        from hermes_cli.config import load_config, cfg_get  # type: ignore[import-not-found]
        cfg = load_config()
        plugin_cfg = cfg_get(cfg, "plugins", "dynamic-tools", default={})
        if not isinstance(plugin_cfg, dict):
            return
        for key in ("enabled", "log", "agent", "learned_mode", "bypass_rate", "cache_mode"):
            if key in plugin_cfg:
                _CONFIG[key] = plugin_cfg[key]
        for key in ("channels", "always_on_extra", "always_off"):
            if key in plugin_cfg and plugin_cfg[key] is not None:
                _CONFIG[key] = plugin_cfg[key]
        if isinstance(plugin_cfg.get("sticky"), dict):
            sticky = dict(_CONFIG.get("sticky") or {})
            sticky.update(plugin_cfg["sticky"])
            _CONFIG["sticky"] = sticky
        if isinstance(plugin_cfg.get("predictor"), dict):
            predictor_cfg = dict(_CONFIG.get("predictor") or {})
            predictor_cfg.update(plugin_cfg["predictor"])
            _CONFIG["predictor"] = predictor_cfg
        if isinstance(plugin_cfg.get("cache_auto"), dict):
            cache_auto = dict(_CONFIG.get("cache_auto") or {})
            cache_auto.update(plugin_cfg["cache_auto"])
            _CONFIG["cache_auto"] = cache_auto
    except Exception as exc:
        logger.debug("dynamic-tools: config load failed (using defaults): %s", exc)


# ─── The narrowing patch ───────────────────────────────────────────────────

def _wrap_build_api_kwargs(original):
    """Wrap AIAgent._build_api_kwargs to narrow tools per API call."""
    def wrapped(self, api_messages):
        kwargs = original(self, api_messages)
        if not _CONFIG["enabled"]:
            return kwargs

        state = _PREDICTION_CV.get()
        if state is None:
            return kwargs

        try:
            tools = kwargs.get("tools")
            if not isinstance(tools, list) or not tools:
                return kwargs

            # Best-effort capture for cache-economics analysis. Both fields
            # are nullable in the prediction record; blank when the adapter
            # doesn't surface them at this layer. setdefault preserves the
            # value across intra-turn calls (e.g. post-expand_tools).
            state.setdefault("model", str(kwargs.get("model", "") or ""))
            state.setdefault(
                "provider",
                str(getattr(self, "provider", "") or kwargs.get("provider", "") or ""),
            )

            allowed = state.get("allowed_tool_names")
            expansions = state.get("expansions") or set()
            known = state.get("known_tool_names") or set()

            # Wildcard: don't narrow, but still log on first call
            if allowed == WILDCARD_ALWAYS_ON or allowed is None:
                names = _tool_names(tools)
                state["ceiling_tools"] = names
                # Snapshot once per prediction. Subsequent _build_api_kwargs
                # calls within the same turn (after the model used a tool
                # like expand_tools) must not overwrite the "what the model
                # initially saw" record — otherwise expansion-driven calls
                # look like they were available from the start.
                state.setdefault("initial_allowed_tools", names)
                state.setdefault("cut_tools", [])
                state.setdefault("unknown_kept_tools", [])
                state["last_tool_list_hash"] = _tool_list_hash(tools)
                state["last_system_hash"] = _system_message_hash(api_messages)
                state["api_call_idx"] = int(state.get("api_call_idx", 0)) + 1
                _maybe_log_prediction(state, ceiling=tools, narrowed=tools)
                return kwargs

            allowed_set = set(allowed) if isinstance(allowed, list) else set()
            allowed_set |= expansions  # user-expanded categories add tools

            # When expansions name a CATEGORY (toolset name) rather than tool
            # names directly, resolve them through Hermes' toolset table.
            for cat in expansions:
                allowed_set |= _resolve_category_to_tools(cat)

            # Apply safe-default for unknown tools: if a tool in the ceiling
            # is not mentioned anywhere in the preset, keep it on
            filtered = []
            ceiling_names = []
            cut_names = []
            unknown_kept_names = []
            for t in tools:
                if not isinstance(t, dict):
                    continue
                # Anthropic format: top-level "name". OpenAI format: function.name.
                name = _tool_name(t)
                if not name:
                    continue
                ceiling_names.append(name)
                # Hermes prefixes tool names with "mcp_" when sending to Anthropic
                # via the Claude Code OAuth path (anthropic_adapter.py:1774-1778).
                # Strip that prefix for matching against our preset, which uses
                # the canonical (unprefixed) tool names.
                base_name = _base_tool_name(name)
                if base_name in allowed_set:
                    filtered.append(t)
                elif base_name not in known:
                    filtered.append(t)
                    unknown_kept_names.append(name)
                else:
                    cut_names.append(name)

            # Diagnostic: log what's happening on the FIRST call per turn.
            if not state.get("logged"):
                logger.info(
                    "dynamic-tools: filter input tools=%d allowed_set=%d known=%d "
                    "→ kept=%d cut=%d (sample_names=%s, sample_cut=%s)",
                    len(tools), len(allowed_set), len(known),
                    len(filtered), len(cut_names),
                    ceiling_names[:5], cut_names[:5],
                )

            kwargs = dict(kwargs)
            kwargs["tools"] = filtered

            state["ceiling_tools"] = ceiling_names
            # Snapshot once per prediction — see wildcard branch above. The
            # filtered list on later calls reflects post-expansion state and
            # would otherwise pollute "initial" with expansion-added tools.
            state.setdefault("initial_allowed_tools", _tool_names(filtered))
            state.setdefault("cut_tools", cut_names)
            state.setdefault("unknown_kept_tools", unknown_kept_names)

            # Per-call hash for response-side correlation with
            # cache_read_tokens. Updated every API call so intra-turn
            # mutations (expand_tools) show up distinctly from the
            # snapshot stamped on the prediction row.
            state["last_tool_list_hash"] = _tool_list_hash(filtered)
            state["last_system_hash"] = _system_message_hash(api_messages)
            state["api_call_idx"] = int(state.get("api_call_idx", 0)) + 1

            # Phase 1: propagate any expansions back to the session's
            # frozen snapshot so the next dispatch reuses them. Without
            # this, expand_tools events would only live for one turn and
            # the model would have to re-expand on every subsequent
            # turn — defeating the freeze's whole point.
            sid = state.get("session_id", "")
            if sid and sid in _FROZEN_BY_SESSION:
                current_expansions = set(state.get("expansions") or set())
                if current_expansions:
                    frozen = _FROZEN_BY_SESSION[sid]
                    prior = set(frozen.get("expansions") or set())
                    if not current_expansions.issubset(prior):
                        frozen["expansions"] = prior | current_expansions

            _maybe_log_prediction(state, ceiling=tools, narrowed=filtered)
            return kwargs
        except Exception as exc:
            logger.warning(
                "dynamic-tools: narrowing failed (%s) — returning unfiltered tools",
                exc,
            )
            return kwargs

    wrapped.__wrapped__ = original  # type: ignore[attr-defined]
    return wrapped


def _resolve_category_to_tools(category: str) -> set[str]:
    """Resolve a Hermes toolset name (e.g. 'browser', 'cronjob') to its tool names.

    Used when expand_tools is called with a category — the model passes a
    toolset name like 'browser' and we need to know which individual tool
    names to allow.
    """
    try:
        from toolsets import resolve_toolset  # type: ignore[import-not-found]
        names = resolve_toolset(category)
        return set(names) if names else set()
    except Exception:
        return set()


def _tool_name(tool: Any) -> str:
    """Return the canonical tool name from Anthropic or OpenAI schema shapes."""
    if not isinstance(tool, dict):
        return ""
    name = tool.get("name") or (tool.get("function", {}) or {}).get("name", "")
    return str(name or "")


def _base_tool_name(name: str) -> str:
    """Strip transport-specific prefixes for preset matching."""
    return name[4:] if name.startswith("mcp_") else name


def _tool_names(tools: list) -> list[str]:
    """Return stable tool names from a tool-schema list."""
    return [name for name in (_tool_name(t) for t in tools) if name]


def _tool_list_hash(tools: list) -> str:
    """Truncated sha256 of the narrowed tool list in send-order.

    Captures membership, ordering, and schema-content drift — provider
    prefix-caches fingerprint the actual schema bytes, so a reorder or a
    description change invalidates the cache even when the name set is
    stable. Analysis derives "did the prefix change?" by comparing hashes
    across consecutive turns in the same session.
    """
    try:
        import json as _json
        payload = _json.dumps(tools, sort_keys=False, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


def _system_message_hash(api_messages: Any) -> str:
    """Hash of the system message content for Mnemosyne-injection detection.

    The system message sits BEFORE the tool block in the cached prefix.
    Anything that mutates it (Mnemosyne injecting recalled memory into
    the system prompt, prompt-caching strategy adjustments, etc.) busts
    the entire downstream cache regardless of how stable the tool list
    is. If this hash varies turn-over-turn within a session while
    tool_list_hash stays stable, the freeze is doing its job and
    something else upstream is the cache breaker.

    Returns truncated sha256 of the first message's content when its
    role is "system" or "developer"; empty string otherwise (some
    providers route system content differently).
    """
    try:
        if not isinstance(api_messages, list) or not api_messages:
            return ""
        first = api_messages[0]
        if not isinstance(first, dict):
            return ""
        role = str(first.get("role") or "").lower()
        if role not in ("system", "developer"):
            return ""
        content = first.get("content")
        # Content can be str OR list-of-parts (Anthropic multimodal shape).
        if isinstance(content, list):
            import json as _json
            payload = _json.dumps(content, sort_keys=False, ensure_ascii=False, separators=(",", ":"))
        elif isinstance(content, str):
            payload = content
        else:
            return ""
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


def _trigger_tools_by_group(preset: Any, fired: list[str]) -> dict[str, list[str]]:
    fired_set = set(fired or [])
    out: dict[str, list[str]] = {}
    for group in getattr(preset, "triggers", []) or []:
        if getattr(group, "name", "") in fired_set:
            out[str(group.name)] = list(getattr(group, "tools", []) or [])
    return out


def _sticky_config() -> dict[str, Any]:
    cfg = _CONFIG.get("sticky") or {}
    return cfg if isinstance(cfg, dict) else {}


def _sticky_enabled() -> bool:
    return bool(_sticky_config().get("enabled", True))


def _sticky_ttl_turns() -> int:
    try:
        return max(1, int(_sticky_config().get("ttl_turns", 3)))
    except Exception:
        return 3


def _sticky_allowed_categories() -> set[str]:
    raw = _sticky_config().get("categories")
    if raw in (None, "*"):
        return set()
    if not isinstance(raw, list):
        return set()
    return {str(item).strip() for item in raw if str(item).strip()}


def _sticky_key_for_session(session_id: str) -> str:
    """Derive a privacy-safe sticky-residency key from a session ID.

    The key is opaque (sha1 hex prefix) so no raw session/chat identifiers
    appear in logs. Returns empty string when session_id is empty, which
    causes all sticky helpers to no-op safely.
    """
    if not session_id:
        return ""
    digest = hashlib.sha1(session_id.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"stk:{digest}"


def _decay_sticky(sticky_key: str) -> None:
    """Decrement sticky entries once per inbound gateway turn for a sticky key."""
    if not _sticky_enabled() or not sticky_key:
        return
    entries = _STICKY_BY_KEY.get(sticky_key)
    if not entries:
        return
    for category in list(entries.keys()):
        entry = entries[category]
        try:
            entry["remaining_turns"] = int(entry.get("remaining_turns", 0)) - 1
        except Exception:
            entry["remaining_turns"] = 0
        if entry.get("remaining_turns", 0) <= 0:
            entries.pop(category, None)
    if not entries:
        _STICKY_BY_KEY.pop(sticky_key, None)


def _live_sticky(sticky_key: str) -> tuple[list[str], list[str], dict[str, int]]:
    """Return (tools, categories, remaining_turns) for active sticky entries."""
    if not _sticky_enabled() or not sticky_key:
        return [], [], {}
    entries = _STICKY_BY_KEY.get(sticky_key) or {}
    tools: set[str] = set()
    categories: list[str] = []
    remaining: dict[str, int] = {}
    for category, entry in entries.items():
        if int(entry.get("remaining_turns", 0)) <= 0:
            continue
        categories.append(category)
        remaining[category] = int(entry.get("remaining_turns", 0))
        tools.update(str(t) for t in (entry.get("tools") or set()) if str(t))
    return sorted(tools), sorted(categories), remaining


def _refresh_sticky(sticky_key: str, category: str, tools: list[str], *, policy_scope: str = "") -> dict[str, Any]:
    """Refresh in-memory sticky residency after successful expand_tools.

    sticky_key is the session-scoped opaque key (from _sticky_key_for_session).
    policy_scope (agent:platform) is used only for the turn-counter telemetry
    field; it is never stored as the residency index.
    """
    if not _sticky_enabled() or not sticky_key or not category or not tools:
        return {"enabled": False}

    allowed_categories = _sticky_allowed_categories()
    if allowed_categories and category not in allowed_categories:
        return {"enabled": True, "refreshed": False, "reason": "category_not_sticky"}

    entries = _STICKY_BY_KEY.setdefault(sticky_key, {})
    state = _PREDICTION_CV.get()
    entries[category] = {
        "tools": set(str(t) for t in tools if str(t)),
        "remaining_turns": _sticky_ttl_turns(),
        "updated_at": time.time(),
        "expanded_at_turn": _POLICY_TURN_BY_SCOPE.get(policy_scope, 0),
        "prediction_id": str((state or {}).get("prediction_id", "")),
    }

    sticky_tools, categories, remaining = _live_sticky(sticky_key)
    return {
        "enabled": True,
        "refreshed": True,
        "category": category,
        "sticky_tools": sticky_tools,
        "sticky_categories": categories,
        "sticky_remaining_turns": remaining,
    }


def _sticky_expansion_for_tool(sticky_key: str, policy_scope: str, tool: str) -> dict[str, Any]:
    """Return sticky expansion metadata if ``tool`` came from a live expansion.

    sticky_key is the session-scoped residency index.
    policy_scope (agent:platform) is used only to look up the turn counter
    for the turns_until_used telemetry field.
    """
    if not _sticky_enabled() or not sticky_key or not tool:
        return {}

    base_tool = _base_tool_name(tool)
    entries = _STICKY_BY_KEY.get(sticky_key) or {}
    current_turn = _POLICY_TURN_BY_SCOPE.get(policy_scope, 0)
    for category, entry in entries.items():
        try:
            remaining_turns = int(entry.get("remaining_turns", 0))
        except Exception:
            remaining_turns = 0
        if remaining_turns <= 0:
            continue
        tools = {str(t) for t in (entry.get("tools") or set()) if str(t)}
        if base_tool not in tools and tool not in tools:
            continue
        try:
            expanded_at_turn = int(entry.get("expanded_at_turn", current_turn) or current_turn)
        except Exception:
            expanded_at_turn = current_turn
        return {
            "category": category,
            "expanded_tool": tool,
            "expanded_at_turn": expanded_at_turn,
            "turns_until_used": max(0, current_turn - expanded_at_turn),
            "sticky_remaining_turns": remaining_turns,
            "expansion_prediction_id": str(entry.get("prediction_id", "")),
        }
    return {}


def _pending_expansion_for_tool(pending_expansion: dict[str, Any], tool: str) -> dict[str, Any]:
    """Return metadata when the current call uses the pending expansion."""
    if not pending_expansion or not tool:
        return {}
    resolved_tools = {str(t) for t in (pending_expansion.get("resolved_tools") or []) if str(t)}
    base_tool = _base_tool_name(tool)
    if tool not in resolved_tools and base_tool not in resolved_tools:
        return {}
    return {
        "category": pending_expansion.get("category", ""),
        "expanded_tool": tool,
        "turns_until_used": 0,
        "expansion_prediction_id": "",
    }


def _platform_name(value: Any) -> str:
    """Extract a privacy-safe platform name from gateway source shapes."""
    if value is None:
        return ""

    # Gateway SessionSource-like objects carry the real platform in .platform.
    nested = getattr(value, "platform", None)
    if nested is not None and nested is not value:
        name = _platform_name(nested)
        if name:
            return name

    raw = getattr(value, "value", value)
    if isinstance(raw, str):
        return raw.strip().lower()

    # Avoid logging repr(SessionSource), which can include chat/user details.
    return ""


def _profile_agent_name() -> str:
    """Derive the agent identity from profile/config context.

    Precedence mirrors :func:`_agent_platform_from_context` but stops at
    the agent slot. Returns ``""`` when no real profile identity is
    discoverable — callers MUST NOT substitute ``"default"`` for blank
    telemetry rows (it would conflate true-out-of-band calls with
    profile-less environments).
    """
    agent = str(_CONFIG.get("agent") or "").strip().lower()
    if agent:
        return agent
    home = os.environ.get("HERMES_HOME", "").rstrip("/")
    if home:
        parent_dir, name = os.path.split(home)
        if os.path.basename(parent_dir) == "profiles" and name:
            return name.strip().lower()
    return os.environ.get("HERMES_PROFILE", "").strip().lower()


def _platform_from_session_key(session_key: Any) -> str:
    """Return the platform segment of a canonical Hermes session key.

    Hermes' :func:`gateway.session.build_session_key` produces:
        ``agent:main:{platform}:{chat_type}[:...]``
    Position 1 is the *literal* string ``"main"`` — NOT a per-profile
    agent name (that lived in our imagination, not in the gateway).
    Position 2 is the platform.

    Returns ``""`` for anything that doesn't match the canonical shape
    (AIAgent's uuid/timestamp ``session_id``, blank inputs, anything
    without a ``"main"`` second segment).
    """
    if not isinstance(session_key, str) or not session_key:
        return ""
    parts = session_key.split(":")
    if len(parts) >= 3 and parts[0] == "agent" and parts[1] == "main":
        return parts[2].strip().lower()
    return ""


def _parse_session_key_scope(session_key: Any) -> tuple[str, str, str]:
    """Resolve ``(agent, platform, scope)`` from a canonical session key.

    Platform comes from the session key itself; agent comes from profile
    context (see :func:`_profile_agent_name`). Both must be non-empty for
    a populated result — otherwise returns ``("", "", "")`` so callers
    leave the row blank rather than synthesizing a misleading scope.

    The previous implementation read the agent from ``parts[1]``, which
    is always the literal ``"main"`` — that produced spurious
    ``main:telegram`` attribution for every profile.
    """
    platform = _platform_from_session_key(session_key)
    if not platform:
        return "", "", ""
    agent = _profile_agent_name()
    if not agent:
        return "", "", ""
    return agent, platform, f"{agent}:{platform}"


def _canonical_session_key(
    event: Any = None,
    session_store: Any = None,
    kwargs: dict[str, Any] | None = None,
) -> str:
    """Resolve the canonical Hermes session key for telemetry.

    Hermes' gateway is the source of truth for session identity. When
    ``session_store`` is reachable we ask it to build the key from
    ``event.source`` — that's the exact value the gateway itself uses to
    route the message. Falling back to ``HERMES_SESSION_KEY`` covers
    paths where ``session_store`` isn't handed in but the env var is set
    by the dispatch loop.

    Avoids inventing a session id from ``chat_id`` or other partial
    SessionSource fields — those would diverge from Hermes' own bookkeeping
    and break joins against gateway logs.
    """
    if session_store is not None and event is not None:
        try:
            source = getattr(event, "source", None)
            generator = getattr(session_store, "_generate_session_key", None)
            if source is not None and callable(generator):
                key = generator(source)
                if isinstance(key, str) and key:
                    return key
        except Exception:
            pass
    env_key = os.environ.get("HERMES_SESSION_KEY") or ""
    if env_key:
        return env_key
    if kwargs:
        for cand in ("session_key", "session_id"):
            val = kwargs.get(cand)
            if isinstance(val, str) and val:
                return val
    val = getattr(event, "session_id", None) if event is not None else None
    if isinstance(val, str) and val:
        return val
    return ""


def _agent_platform_from_context(event: Any = None, kwargs: dict[str, Any] | None = None) -> tuple[str, str, str]:
    """Return ``(agent, platform, scope)`` for per-agent/platform learning.

    Scope is ``{agent}:{platform}``. Agent identity comes from profile
    context (explicit config, HERMES_HOME profile dir, HERMES_PROFILE);
    platform comes from the canonical session key's platform segment, then
    from kwargs/event hints. Hermes' own ``HERMES_SESSION_KEY`` is shaped
    ``agent:main:{platform}:{chat_type}[:...]`` — position 1 is the
    literal ``"main"``, never the per-profile agent name, so we don't
    consult it for agent resolution. Falls through to ``default:unknown``
    only when nothing real is discoverable.
    """
    kwargs = kwargs or {}
    session_key = str(os.environ.get("HERMES_SESSION_KEY") or "")

    agent = _profile_agent_name()
    platform = _platform_from_session_key(session_key)

    for key in ("platform", "source", "transport"):
        name = _platform_name(kwargs.get(key))
        if name:
            platform = name
            break

    if event is not None:
        for attr in ("platform", "source", "transport"):
            try:
                value = getattr(event, attr, None)
            except Exception:
                value = None
            name = _platform_name(value)
            if name:
                platform = name
                break

    # Preserve blank attribution when either slot can't be filled with
    # real data. Synthesizing "default"/"unknown" hides legitimate
    # regressions (e.g. an explicit `agent:` config line removed by
    # accident) behind superficially valid scope strings — see the
    # post-tool-call recovery path at _recover_attribution_without_state
    # for the same invariant. The analyzer's normalize_scope() buckets
    # blank rows as "unknown" so reports stay readable.
    if not agent or not platform:
        return "", "", ""
    return agent, platform, f"{agent}:{platform}"


def _maybe_log_prediction(
    state: dict[str, Any],
    ceiling: list,
    narrowed: list,
) -> None:
    """Write the prediction row to predictions.jsonl on the FIRST API call
    per turn. Subsequent calls in the same turn don't re-log."""
    if state.get("logged"):
        return
    state["logged"] = True
    if not _CONFIG.get("log", True):
        return
    try:
        record = logger_io.PredictionRecord(
            ts=time.time(),
            prediction_id=state.get("prediction_id", ""),
            session_id=str(state.get("session_id", "")),
            channel=str(state.get("scope", state.get("channel", ""))),
            agent=str(state.get("agent", "")),
            platform=str(state.get("platform", "")),
            scope=str(state.get("scope", "")),
            message_hash=str(state.get("message_hash", "")),
            message_preview=str(state.get("message_preview", "")),
            preset=str(state.get("preset", "")),
            triggers_fired=list(state.get("triggers_fired", [])),
            triggers_suppressed=list(state.get("triggers_suppressed") or []),
            always_on_count=int(state.get("always_on_count", 0)),
            ceiling_count=len(ceiling),
            narrowed_count=len(narrowed),
            ceiling_tokens=logger_io.estimate_tokens(ceiling),
            narrowed_tokens=logger_io.estimate_tokens(narrowed),
            ceiling_tools=list(state.get("ceiling_tools") or _tool_names(ceiling)),
            allowed_tools=list(state.get("initial_allowed_tools") or _tool_names(narrowed)),
            cut_tools=list(state.get("cut_tools") or []),
            unknown_kept_tools=list(state.get("unknown_kept_tools") or []),
            always_on_tools=list(state.get("always_on_tools") or []),
            trigger_tools_by_group=dict(state.get("trigger_tools_by_group") or {}),
            expanded_tools=sorted(str(t) for t in (state.get("expansions") or set())),
            sticky_tools=list(state.get("sticky_tools") or []),
            sticky_categories=list(state.get("sticky_categories") or []),
            sticky_remaining_turns=dict(state.get("sticky_remaining_turns") or {}),
            policy_source=str(state.get("policy_source", "preset")),
            policy_version=str(state.get("policy_version", "")),
            learned_mode=str(state.get("learned_mode", "off")),
            learned_scope=str(state.get("learned_scope", "")),
            learned_changes=list(state.get("learned_changes") or []),
            lookback_used=int(state.get("lookback_used", 0)),
            lookback_turns_config=int(state.get("lookback_turns_config", 0)),
            tool_list_hash=_tool_list_hash(narrowed),
            provider=str(state.get("provider", "")),
            model=str(state.get("model", "")),
            frozen_reuse=bool(state.get("frozen_reuse", False)),
            frozen_reuse_count=int(state.get("frozen_reuse_count", 0)),
        )
        logger_io.log_prediction(record)
    except Exception as exc:
        logger.debug("dynamic-tools: log_prediction failed: %s", exc)


def _wrap_compress_context(original):
    """Phase 4: evict the session's frozen snapshot after compaction.

    Compaction rewrites the conversation history — the provider's
    cached prefix becomes worthless because the bytes after the system
    block differ. We treat that moment as a free reshape opportunity:
    drop the frozen tool set so the next ``pre_gateway_dispatch``
    runs the predictor fresh against the post-compaction state.

    The wrapper runs around the original to keep the AIAgent's
    compaction semantics intact even on failure paths — we evict only
    after the original returns successfully.
    """
    def wrapped(self, messages, system_message, *args, **kwargs):
        result = original(self, messages, system_message, *args, **kwargs)
        if not _CONFIG.get("enabled"):
            return result
        try:
            session_key = os.environ.get("HERMES_SESSION_KEY") or ""
            evicted_keys: list[str] = []
            if session_key and session_key in _FROZEN_BY_SESSION:
                _FROZEN_BY_SESSION.pop(session_key, None)
                _CACHE_MODE_BY_SESSION.pop(session_key, None)
                evicted_keys.append(session_key)
            # Defensive: also try the AIAgent's session_id (uuid). When
            # the env var isn't set but Phase 1 keyed under the uuid as
            # a fallback (it doesn't today, but if that changes we want
            # the patch resilient), the eviction here catches it.
            aid = getattr(self, "session_id", "") or ""
            if aid and aid != session_key and aid in _FROZEN_BY_SESSION:
                _FROZEN_BY_SESSION.pop(aid, None)
                _CACHE_MODE_BY_SESSION.pop(aid, None)
                evicted_keys.append(aid)
            if evicted_keys:
                logger.info(
                    "dynamic-tools: compaction evicted freeze for session(s) %s — "
                    "next dispatch will re-freeze",
                    evicted_keys,
                )
        except Exception as exc:
            logger.debug("dynamic-tools: compaction post-eviction failed: %s", exc)
        return result

    wrapped.__wrapped__ = original  # type: ignore[attr-defined]
    return wrapped


def _install_patches() -> bool:
    """Install the AIAgent monkey-patches.

    NOTE on the from-import / class-method situation:
      AIAgent is a class; ``_build_api_kwargs`` and ``_compress_context``
      are methods on it. Replacing methods on the class affects ALL
      instances (existing and new). No from-import binding problem
      here — methods are looked up via the instance's class at call
      time.
    """
    global _ORIGINAL_BUILD_API_KWARGS, _ORIGINAL_COMPRESS_CONTEXT, _PATCHED
    if _PATCHED:
        return True
    try:
        import run_agent  # type: ignore[import-not-found]
    except ImportError as exc:
        logger.warning("dynamic-tools: cannot import run_agent: %s", exc)
        return False

    AIAgent = getattr(run_agent, "AIAgent", None)
    if AIAgent is None or not hasattr(AIAgent, "_build_api_kwargs"):
        logger.warning("dynamic-tools: AIAgent._build_api_kwargs not found")
        return False

    _ORIGINAL_BUILD_API_KWARGS = AIAgent._build_api_kwargs
    AIAgent._build_api_kwargs = _wrap_build_api_kwargs(_ORIGINAL_BUILD_API_KWARGS)

    # Phase 4: compaction-driven freeze eviction. No `post_compaction`
    # hook exists in Hermes, so we wrap the forwarder method directly.
    # Best-effort — when _compress_context is missing (older Hermes
    # versions, custom builds) the freeze persists across compaction,
    # which costs a single cache break on the next dispatch but doesn't
    # corrupt anything.
    if hasattr(AIAgent, "_compress_context"):
        _ORIGINAL_COMPRESS_CONTEXT = AIAgent._compress_context
        AIAgent._compress_context = _wrap_compress_context(_ORIGINAL_COMPRESS_CONTEXT)
        logger.info("dynamic-tools: patches installed (_build_api_kwargs, _compress_context)")
    else:
        logger.info("dynamic-tools: patches installed (_build_api_kwargs only — _compress_context missing)")

    _PATCHED = True
    return True


# ─── Helpers ──────────────────────────────────────────────────────────────

def _all_known_tool_names(preset) -> set[str]:
    known: set[str] = set()
    if isinstance(preset.always_on, list):
        known.update(preset.always_on)
    for group in preset.triggers:
        known.update(group.tools)
    return known


def _attachments_from_event(event: Any) -> list[str]:
    out: list[str] = []
    try:
        atts = getattr(event, "attachments", None)
        if not atts:
            return out
        for a in atts:
            kind = None
            if isinstance(a, dict):
                kind = a.get("type") or a.get("kind") or a.get("mime_type") or ""
            else:
                kind = getattr(a, "type", None) or getattr(a, "mime_type", None) or ""
            kind = str(kind).lower()
            if "image" in kind or kind in {"photo", "picture"}:
                out.append("image")
            elif "audio" in kind or kind in {"voice", "ogg"}:
                out.append("audio")
    except Exception:
        return out
    return out


def _message_text_from_event(event: Any) -> str:
    for attr in ("text", "message", "content", "body"):
        try:
            v = getattr(event, attr, None)
        except Exception:
            v = None
        if isinstance(v, str) and v:
            return v
    return ""


def _predictor_lookback_turns() -> int:
    cfg = _CONFIG.get("predictor") or {}
    if not isinstance(cfg, dict):
        return 0
    try:
        return max(0, int(cfg.get("lookback_turns", 0)))
    except Exception:
        return 0


def _lookback_prior_messages(session_id: str, lookback_turns: int) -> list[str]:
    """Return up to ``lookback_turns`` prior user-message strings for a session."""
    if not session_id or lookback_turns <= 0:
        return []
    prior = _PRIOR_MESSAGES_BY_SESSION.get(session_id) or []
    return list(prior[-lookback_turns:])


def _record_lookback_message(session_id: str, message: str, lookback_turns: int) -> None:
    """Append the current message to the session's lookback ring, trimmed."""
    if not session_id or lookback_turns <= 0 or not message:
        return
    history = _PRIOR_MESSAGES_BY_SESSION.setdefault(session_id, [])
    history.append(message)
    # Keep the ring small. We need at most lookback_turns prior messages for
    # the *next* turn, plus the one we just appended.
    excess = len(history) - (lookback_turns + 1)
    if excess > 0:
        del history[:excess]


def _compose_predictor_input(prior: list[str], current: str) -> str:
    """Concatenate prior + current messages for the predictor.

    Prior messages are tagged with a `[prior]:` marker so future telemetry
    inspection can tell what the predictor actually saw. The marker itself
    is benign for regex matching — it doesn't introduce false-positive
    keywords.
    """
    parts: list[str] = []
    for msg in prior:
        if not msg:
            continue
        parts.append(f"[prior]: {msg}")
    if current:
        parts.append(current)
    return "\n".join(parts) if parts else current


def _bypass_rate_for_scope(scope: str) -> float:
    """Resolve the effective bypass rate for a scope.

    Per-scope ``channels.<scope>.bypass_rate`` wins over the global
    ``bypass_rate``. Looks up the full ``{agent}:{platform}`` scope first,
    then falls back to the bare platform for legacy channel configs.
    Values are clamped to [0.0, 1.0]; bad values fall through.
    """
    channels = _CONFIG.get("channels") or {}
    candidates: list[str] = []
    if scope:
        candidates.append(scope)
        if ":" in scope:
            platform = scope.rsplit(":", 1)[-1]
            if platform and platform not in candidates:
                candidates.append(platform)
    if isinstance(channels, dict):
        for key in candidates:
            cfg = channels.get(key) if key else None
            if isinstance(cfg, dict) and "bypass_rate" in cfg:
                try:
                    return max(0.0, min(1.0, float(cfg["bypass_rate"])))
                except (TypeError, ValueError):
                    continue
    try:
        return max(0.0, min(1.0, float(_CONFIG.get("bypass_rate", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def _should_bypass(scope: str, session_id: str) -> bool:
    """Deterministic, session-stable A/B bypass decision.

    Hashes (scope, session_id) so that a session is consistently in or out
    of the bypass cohort across all its turns. Without a session_id we can't
    keep a session coherent, so default to "do not bypass."
    """
    rate = _bypass_rate_for_scope(scope)
    if rate <= 0.0 or not session_id:
        return False
    if rate >= 1.0:
        return True
    digest = hashlib.sha1(f"{scope}|{session_id}".encode("utf-8", errors="replace")).digest()
    # First 4 bytes → uniform float in [0, 1)
    bucket = int.from_bytes(digest[:4], "big") / 0x1_0000_0000
    return bucket < rate


def _channel_from_event(event: Any) -> str:
    for attr in ("platform", "channel", "source", "transport"):
        try:
            v = getattr(event, attr, None)
        except Exception:
            v = None
        if isinstance(v, str) and v:
            return v.lower()
        # Handle enum-like values (e.g. Platform.TELEGRAM)
        val = getattr(v, "value", None) if v is not None else None
        if isinstance(val, str) and val:
            return val.lower()
    return ""


# ─── Cache-aware freeze (Phase 1) ──────────────────────────────────────────

def _resolve_cache_mode_for_session(session_id: str, scope: str = "") -> str:
    """Decide whether this dispatch should freeze tools at the session level.

    Returns "on" or "off":

      · Explicit ``cache_mode: off`` → "off" — keep per-turn narrowing.
      · Explicit ``cache_mode: on`` → "on" — freeze.
      · ``cache_mode: auto`` (default) → consult the cross-session
        detection cache (Phase 3). If a prior session of this scope
        locked to a definite mode, honor it. Otherwise default to "on"
        per the pivot doc's safe-default argument.

    Mid-session switching is intentionally not supported — switching
    itself busts the cache. The per-session detection telemetry from
    ``_CACHE_MODE_BY_SESSION`` is for the analyzer and the NEXT session,
    never for this one.
    """
    mode = str(_CONFIG.get("cache_mode") or "auto").lower()
    if mode == "off":
        return "off"
    if mode == "on":
        return "on"
    # auto: consult the cross-session cache; default "on" on a miss.
    cached = _cached_detection_mode(scope)
    if cached in ("on", "off"):
        return cached
    return "on"


def _freeze_session_snapshot(
    session_id: str,
    *,
    allowed_tool_names: Any,
    baseline_allowed_tools: Any,
    known_tool_names: set[str],
    preset_name: str,
    always_on_count: int,
    always_on_tools: list[str],
    trigger_tools_by_group: dict[str, list[str]],
    triggers_fired: list[str],
    triggers_suppressed: list[str],
    policy_source: str,
    policy_version: str,
    learned_mode: str,
    learned_scope: str,
    learned_changes: list[str],
) -> None:
    """Persist the first dispatch's tool-set decision for the session.

    Captured on the first dispatch under cache-on mode and reused on
    every subsequent dispatch in the same session. Eviction lives in
    ``_on_session_end`` and ``_on_session_reset``; Phase 4 adds a
    compaction-driven eviction.

    ``triggers_fired`` / ``trigger_tools_by_group`` are stored on the
    snapshot so subsequent turns' telemetry rows can describe the
    decision honestly (this is what was triggered when we froze) rather
    than blanking out as if the predictor never ran.
    """
    _FROZEN_BY_SESSION[session_id] = {
        "allowed_tool_names": allowed_tool_names,
        "baseline_allowed_tools": baseline_allowed_tools,
        "known_tool_names": known_tool_names,
        "expansions": set(),  # grown by expand_tools mid-session
        "preset_name": preset_name,
        "always_on_count": always_on_count,
        "always_on_tools": list(always_on_tools or []),
        "trigger_tools_by_group": dict(trigger_tools_by_group or {}),
        "triggers_fired_at_freeze": list(triggers_fired or []),
        "triggers_suppressed_at_freeze": list(triggers_suppressed or []),
        "policy_source": policy_source,
        "policy_version": policy_version,
        "learned_mode": learned_mode,
        "learned_scope": learned_scope,
        "learned_changes": list(learned_changes or []),
        "frozen_at": time.time(),
        "reuses": 0,
    }


def _build_state_from_frozen(
    frozen: dict[str, Any],
    *,
    session_id: str,
    scope: str,
    channel: str,
    agent: str,
    platform: str,
    message: str,
) -> dict[str, Any]:
    """Construct the contextvar state for a dispatch that reuses a frozen snapshot.

    Sticky residency and lookback are *not populated* — both were
    per-turn-adjustment mechanisms; the freeze makes them redundant by
    design (see [[dynamic-tools-design-principle]]). The fields are
    still present in the state dict with empty values so downstream
    consumers (telemetry, post_tool_call) don't need conditional reads.

    The frozen snapshot's ``expansions`` set carries forward — any
    expand_tools tools added in prior turns of this session stay
    available without a fresh expand_tools round-trip.
    """
    frozen["reuses"] = int(frozen.get("reuses", 0)) + 1
    return {
        "prediction_id": logger_io.new_prediction_id(),
        "allowed_tool_names": frozen["allowed_tool_names"],
        "baseline_allowed_tools": frozen["baseline_allowed_tools"],
        "known_tool_names": frozen["known_tool_names"],
        # Carry expansions forward in-session — this is the core of the
        # "stay frozen, including prior expansions" behavior.
        "expansions": set(frozen.get("expansions") or set()),
        "channel": channel,
        "agent": agent,
        "platform": platform,
        "scope": scope,
        # Sticky disabled under freeze — empty key short-circuits all
        # sticky helpers and the post_tool_call credit logic falls back
        # to the in-turn ``pending_expansion`` check, which is correct.
        "sticky_key": "",
        "session_id": session_id,
        "message_hash": logger_io.hash_message(message),
        "message_preview": logger_io.message_preview(message),
        "preset": frozen["preset_name"],
        # Triggers aren't recomputed per turn under freeze. We surface the
        # at-freeze values so the prediction row tells an honest story
        # ("this is what we froze on").
        "triggers_fired": list(frozen.get("triggers_fired_at_freeze") or []),
        "triggers_suppressed": list(frozen.get("triggers_suppressed_at_freeze") or []),
        "always_on_count": int(frozen.get("always_on_count", 0)),
        "always_on_tools": list(frozen.get("always_on_tools") or []),
        "trigger_tools_by_group": dict(frozen.get("trigger_tools_by_group") or {}),
        "sticky_tools": [],
        "sticky_categories": [],
        "sticky_remaining_turns": {},
        # ``frozen`` policy_source is set on first-dispatch (could be
        # "preset", "learned", or "bypass") — preserved on reuse so the
        # cohort analyzer doesn't see the same session split across
        # multiple sources.
        "policy_source": frozen.get("policy_source", "preset"),
        "policy_version": frozen.get("policy_version", ""),
        "learned_mode": frozen.get("learned_mode", "off"),
        "learned_scope": frozen.get("learned_scope", ""),
        "learned_changes": list(frozen.get("learned_changes") or []),
        # Lookback disabled under freeze.
        "lookback_used": 0,
        "lookback_turns_config": 0,
        "logged": False,
        # Marker for downstream telemetry — useful in the analyzer to
        # tell first-dispatch from reuse without joining on session.
        "frozen_reuse": True,
        "frozen_reuse_count": frozen["reuses"],
    }


# ─── Hook: pre_gateway_dispatch ────────────────────────────────────────────

def _on_pre_gateway_dispatch(event=None, gateway=None, session_store=None, **kwargs) -> None:
    if not _CONFIG["enabled"]:
        return None
    # Retry patch install if it was deferred at register() time. This runs
    # synchronously before the prediction work, so when _build_api_kwargs
    # fires for the same turn, the patch is already in place.
    if not _PATCHED:
        _install_patches()
    try:
        message = _message_text_from_event(event)
        attachments = _attachments_from_event(event)
        agent, platform, scope = _agent_platform_from_context(event, kwargs)
        _POLICY_TURN_BY_SCOPE[scope] = _POLICY_TURN_BY_SCOPE.get(scope, 0) + 1
        channel = scope
        session_id = _canonical_session_key(event, session_store, kwargs)

        # Phase 1: cache-on freeze. When this session already has a
        # frozen tool-set decision, reuse it verbatim and skip the
        # predictor entirely. The frozen path also disables sticky and
        # lookback by construction.
        cache_decision = _resolve_cache_mode_for_session(str(session_id), scope=scope)
        if cache_decision == "on" and session_id and session_id in _FROZEN_BY_SESSION:
            frozen = _FROZEN_BY_SESSION[session_id]
            _PREDICTION_CV.set(_build_state_from_frozen(
                frozen,
                session_id=str(session_id),
                scope=scope,
                channel=channel,
                agent=agent,
                platform=platform,
                message=message,
            ))
            return None

        # Narrow sticky-residency key: session-scoped, opaque. Distinct from the
        # coarse policy scope (agent:platform) used for telemetry/learned policy.
        sticky_key = _sticky_key_for_session(str(session_id))

        lookback_turns = _predictor_lookback_turns()
        prior_messages = _lookback_prior_messages(str(session_id), lookback_turns)
        predictor_input = _compose_predictor_input(prior_messages, message)

        preset = presets_mod.resolve_preset(_CONFIG, scope)
        prediction = predictor_mod.predict(predictor_input, attachments, preset)
        # Push the current message onto the session's lookback ring AFTER
        # prediction so we don't include the current message as its own prior.
        _record_lookback_message(str(session_id), message, lookback_turns)
        known = _all_known_tool_names(preset)
        always_on_tools = list(preset.always_on) if isinstance(preset.always_on, list) else []
        trigger_tools = _trigger_tools_by_group(preset, prediction.triggers_fired)
        _decay_sticky(sticky_key)
        sticky_tools, sticky_categories, sticky_remaining = _live_sticky(sticky_key)
        allowed_tool_names = prediction.allowed_tool_names
        # Capture the pre-sticky baseline. This is the set the model would
        # see *without* any prior expansion (sticky carries tools forward
        # only because a prior turn ran expand_tools). The credit decision
        # in _on_post_tool_call needs this to distinguish "model could have
        # called this anyway" from "this call is the payoff of an earlier
        # expand". WILDCARD is preserved (bypass cohort / wildcard always_on
        # already see every tool — nothing to attribute to expansion).
        if allowed_tool_names == WILDCARD_ALWAYS_ON or allowed_tool_names is None:
            baseline_allowed_tools: Any = WILDCARD_ALWAYS_ON
        else:
            baseline_allowed_tools = sorted(set(allowed_tool_names))
            allowed_tool_names = sorted(set(allowed_tool_names) | set(sticky_tools))

        # A/B baseline: deterministically bypass narrowing for a fraction of
        # sessions. The predictor still ran so we keep telemetry, but the
        # allowed set widens to the full ceiling and the policy_source is
        # stamped so the analyzer can compare cohorts.
        bypassed = _should_bypass(scope, str(session_id))
        if bypassed:
            allowed_tool_names = WILDCARD_ALWAYS_ON
        policy_source = "bypass" if bypassed else getattr(preset, "policy_source", "preset")

        _PREDICTION_CV.set({
            "prediction_id": logger_io.new_prediction_id(),
            "allowed_tool_names": allowed_tool_names,
            "baseline_allowed_tools": baseline_allowed_tools,
            "known_tool_names": known,
            "expansions": set(),
            "channel": channel,
            "agent": agent,
            "platform": platform,
            "scope": scope,
            "sticky_key": sticky_key,
            "session_id": session_id,
            "message_hash": logger_io.hash_message(message),
            "message_preview": logger_io.message_preview(message),
            "preset": prediction.preset_name,
            "triggers_fired": prediction.triggers_fired,
            "triggers_suppressed": prediction.triggers_suppressed,
            "always_on_count": prediction.always_on_count,
            "always_on_tools": always_on_tools,
            "trigger_tools_by_group": trigger_tools,
            "sticky_tools": sticky_tools,
            "sticky_categories": sticky_categories,
            "sticky_remaining_turns": sticky_remaining,
            "policy_source": policy_source,
            "policy_version": getattr(preset, "policy_version", ""),
            "learned_mode": getattr(preset, "learned_mode", _CONFIG.get("learned_mode", "off")),
            "learned_scope": getattr(preset, "learned_scope", ""),
            "learned_changes": list(getattr(preset, "learned_changes", []) or []),
            "lookback_used": len(prior_messages),
            "lookback_turns_config": lookback_turns,
            "logged": False,
        })

        # Phase 1: on the first dispatch under cache-on mode, freeze the
        # decision we just made. Every subsequent dispatch in this
        # session will reuse it via the frozen-path branch above. Skip
        # when there's no usable session_id (out-of-band dispatch can't
        # be re-identified, so caching the decision would only collide).
        if cache_decision == "on" and session_id:
            _freeze_session_snapshot(
                str(session_id),
                allowed_tool_names=allowed_tool_names,
                baseline_allowed_tools=baseline_allowed_tools,
                known_tool_names=known,
                preset_name=prediction.preset_name,
                always_on_count=prediction.always_on_count,
                always_on_tools=always_on_tools,
                trigger_tools_by_group=trigger_tools,
                triggers_fired=prediction.triggers_fired,
                triggers_suppressed=prediction.triggers_suppressed,
                policy_source=policy_source,
                policy_version=getattr(preset, "policy_version", ""),
                learned_mode=getattr(preset, "learned_mode", _CONFIG.get("learned_mode", "off")),
                learned_scope=getattr(preset, "learned_scope", ""),
                learned_changes=list(getattr(preset, "learned_changes", []) or []),
            )
    except Exception as exc:
        logger.warning("dynamic-tools: pre_gateway_dispatch failed: %s", exc)
    return None


# ─── Hook: post_tool_call ──────────────────────────────────────────────────

def _recover_attribution_without_state(
    session_id: Any,
    kwargs: dict[str, Any],
) -> tuple[str, str, str]:
    """Best-effort ``(agent, platform, scope)`` for out-of-band tool calls.

    Used when the contextvar has no per-turn state — typically background
    tools fired outside a gateway-dispatched message loop. Resolution:

      · Agent — from profile/config context only
        (:func:`_profile_agent_name`). The canonical session key's
        position 1 is always the literal ``"main"`` and must never be
        used as the agent identity.
      · Platform — from the passed ``session_id`` if it carries the
        canonical shape, else from ``HERMES_SESSION_KEY`` in env, else
        from kwargs hints (platform/source/transport).

    Returns ``("", "", "")`` when either slot can't be filled with real
    data — preserves blank attribution for legitimately untrackable rows
    rather than synthesizing ``default:unknown``.
    """
    platform = _platform_from_session_key(session_id)
    if not platform:
        platform = _platform_from_session_key(os.environ.get("HERMES_SESSION_KEY") or "")
    if not platform and kwargs:
        for key in ("platform", "source", "transport"):
            name = _platform_name(kwargs.get(key))
            if name:
                platform = name
                break
    agent = _profile_agent_name()
    if agent and platform:
        return agent, platform, f"{agent}:{platform}"
    return "", "", ""


def _on_post_tool_call(tool_name=None, args=None, result=None, task_id=None, **kwargs) -> None:
    if not _CONFIG["enabled"] or not _CONFIG.get("log", True):
        return None
    try:
        state = _PREDICTION_CV.get()
        prediction_id = state.get("prediction_id", "") if state else ""
        session_id = kwargs.get("session_id") or task_id or ""
        if tool_name:
            tool = str(tool_name)
            initial_allowed = set(state.get("initial_allowed_tools") or []) if state else set()
            # Baseline = the allowed set *before* sticky carry-over merged in.
            # Used for credit attribution: a tool only sticky-carries because
            # a prior expand_tools ran — those calls should be credited to
            # expansion, not classified as "initially available."
            # Falls back to initial_allowed for rows written before this
            # field existed; WILDCARD means "no narrowing", every tool is
            # initially available.
            baseline_raw = state.get("baseline_allowed_tools") if state else None
            if baseline_raw == WILDCARD_ALWAYS_ON:
                baseline_allowed: set[str] | None = None  # None means "everything"
            elif baseline_raw is None:
                baseline_allowed = set(initial_allowed)
            else:
                baseline_allowed = set(baseline_raw)
            cut_tools = set(state.get("cut_tools") or []) if state else set()
            expanded = set(state.get("expansions") or set()) if state else set()
            pending_expansion = dict(state.get("pending_expansion") or {}) if state else {}
            if state:
                attribution_agent = str(state.get("agent", ""))
                attribution_platform = str(state.get("platform", ""))
                attribution_scope = str(state.get("scope", ""))
            else:
                attribution_agent, attribution_platform, attribution_scope = (
                    _recover_attribution_without_state(session_id, kwargs)
                )
            # was_initially_available reflects the pre-sticky baseline so the
            # analyzer's "credit only when not initially available" filter
            # (analyze.py) lines up with the writer's credit decision below.
            base_name = _base_tool_name(tool)
            if baseline_allowed is None:
                was_initially_available = True
            else:
                was_initially_available = (
                    tool in baseline_allowed or base_name in baseline_allowed
                )
            extra = {
                "agent": attribution_agent,
                "platform": attribution_platform,
                "scope": attribution_scope,
                "was_initially_available": was_initially_available,
                "was_cut": tool in cut_tools,
                "was_expanded": tool in expanded,
            }
            if tool == "expand_tools":
                extra["expand_event"] = pending_expansion
            else:
                # Only credit a tool call as "expansion-driven" when the
                # tool was NOT already in the initial allowed set. A tool
                # the model could have called regardless gives no signal
                # about whether the expansion was useful — it would
                # inflate analyzer promotion signal with noise.
                # `after_expand_tools` and the round-trip metadata still
                # record when an expansion happened, so analyzers can
                # account for round-trip cost even when the eventual
                # tool call wasn't expansion-driven.
                already_available = was_initially_available
                expand_use: dict[str, Any] = {}
                if not already_available:
                    expand_use = _pending_expansion_for_tool(pending_expansion, tool)
                if pending_expansion:
                    extra["after_expand_tools"] = True
                    extra["expand_category"] = pending_expansion.get("category", "")
                    extra["expand_resolved_tools"] = pending_expansion.get("resolved_tools", [])
                    extra["expand_tools_added"] = pending_expansion.get("tools_added", [])
                if not expand_use and state and not already_available:
                    expand_use = _sticky_expansion_for_tool(
                        str(state.get("sticky_key", "")),
                        str(state.get("scope", "")),
                        tool,
                    )
                if expand_use:
                    extra["expand_tools_used"] = True
                    extra["expanded_tool"] = expand_use.get("expanded_tool", tool)
                    extra["turns_until_used"] = expand_use.get("turns_until_used", 0)
                    extra["expand_category"] = expand_use.get("category", extra.get("expand_category", ""))
                    extra["expansion_prediction_id"] = expand_use.get("expansion_prediction_id", "")
                    if "sticky_remaining_turns" in expand_use:
                        extra["sticky_remaining_turns"] = expand_use.get("sticky_remaining_turns")
                    if state:
                        state["expand_tools_used"] = True
                        state.setdefault("expanded_tools_used", []).append({
                            "tool": tool,
                            "category": extra.get("expand_category", ""),
                            "turns_until_used": extra.get("turns_until_used", 0),
                        })
                        if pending_expansion:
                            state["pending_expansion"] = None
                elif pending_expansion:
                    extra["expand_tools_used"] = False
            log_args = args if tool == "expand_tools" else None
            log_result = result if tool == "expand_tools" else None
            logger_io.log_tool_call(
                prediction_id, str(session_id), tool,
                args=log_args, result=log_result, extra=extra,
            )
    except Exception as exc:
        logger.debug("dynamic-tools: post_tool_call log failed: %s", exc)
    return None


# ─── Cache-mode detection ──────────────────────────────────────────────────

def _cache_auto_cfg() -> dict[str, Any]:
    cfg = _CONFIG.get("cache_auto") or {}
    return cfg if isinstance(cfg, dict) else {}


def _detection_cache_path():
    """Resolve the persisted detection cache path. Lazy import of os to keep
    the helper cheap; pathlib.Path import avoided to dodge a circular bind."""
    import os as _os
    home = _os.environ.get("HERMES_HOME") or _os.path.expanduser("~/.hermes")
    return _os.path.join(home, "state", "dynamic-tools", "cache_mode_detection.json")


def _load_detection_cache() -> None:
    """Read the persisted detection cache into memory exactly once.

    Failures are silent — a missing or corrupt file means we fall through
    to the safe-default "on" for every scope until the first lock writes
    a new entry. No telemetry is lost in either direction.
    """
    global _DETECTION_CACHE_LOADED
    if _DETECTION_CACHE_LOADED:
        return
    _DETECTION_CACHE_LOADED = True
    import json as _json
    try:
        with open(_detection_cache_path(), "r", encoding="utf-8") as f:
            data = _json.load(f)
        if isinstance(data, dict):
            for scope, entry in data.items():
                if isinstance(scope, str) and isinstance(entry, dict):
                    _DETECTION_CACHE[scope] = entry
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.debug("dynamic-tools: detection cache load failed: %s", exc)


def _save_detection_cache() -> None:
    """Persist the detection cache atomically."""
    import json as _json
    import os as _os
    path = _detection_cache_path()
    try:
        _os.makedirs(_os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            _json.dump(_DETECTION_CACHE, f, ensure_ascii=False, indent=2, sort_keys=True)
        _os.replace(tmp, path)
    except Exception as exc:
        logger.debug("dynamic-tools: detection cache save failed: %s", exc)


def _persist_detection_lock(
    scope: str,
    state: dict[str, Any],
    model: str,
) -> None:
    """Record a session's lock result to the cross-session cache.

    The cache is keyed by scope. Same-scope sessions that lock the same
    way bump ``sessions_locked``; a disagreeing lock overwrites with
    ``sessions_locked = 1`` — the most recent reality wins. We don't
    average or smooth because cache behavior is a property of the
    provider+model setup, not a noisy signal that benefits from
    averaging — when it changes, it changes hard (provider migration,
    model swap).

    Provider-blocklist locks are skipped: they're a config statement,
    not an observation, and don't need persistence to be effective.
    """
    if not scope or state.get("mode") not in ("on", "off"):
        return
    if state.get("lock_reason") == "provider_blocklist":
        return
    cached = _DETECTION_CACHE.get(scope)
    new_entry = {
        "mode": state["mode"],
        "locked_at": time.time(),
        "lock_reason": state.get("lock_reason", ""),
        "hit_rate_at_lock": state.get("hit_rate_at_lock", 0.0),
        "last_model": model,
    }
    if cached and cached.get("mode") == new_entry["mode"]:
        new_entry["sessions_locked"] = int(cached.get("sessions_locked", 0)) + 1
    else:
        new_entry["sessions_locked"] = 1
    _DETECTION_CACHE[scope] = new_entry
    _save_detection_cache()


def _cached_detection_mode(scope: str) -> str:
    """Consult the cross-session cache for a scope's prior locked mode.

    Returns "on", "off", or "" (no cache entry). Auto mode at dispatch
    time uses this to set the initial freeze decision intelligently —
    skipping the freeze entirely for known cache-off scopes.
    """
    if not scope:
        return ""
    entry = _DETECTION_CACHE.get(scope)
    if not entry:
        return ""
    mode = entry.get("mode")
    return mode if mode in ("on", "off") else ""


def _check_divergence(
    state: dict[str, Any],
    cache_read: int,
    input_tokens: int,
    cache_write: int,
    scope: str,
    session_key: str,
) -> None:
    """Post-lock observation: does the locked mode still match reality?

    A session locked "on" with consistently low hit rate post-lock is
    evidence that the freeze is the wrong call for this session —
    Mnemosyne is busting prefix, a tools mismatch we didn't anticipate,
    or simply a cold-cache window. We don't switch modes mid-session
    (the switch itself busts cache), but we surface the divergence so
    the analyzer can flag the session for forensic review.

    Threshold: 3 consecutive post-lock calls with hit rate below 30%
    when locked "on". 3 is short enough to fire on a real problem
    without flapping on transient noise; 30% is below the empirical
    "stable hash" floor (90.6% on Bernard), so true cache breakage
    sits well clear of it.
    """
    if state.get("mode") != "on" or state.get("locked_at_call") is None:
        return
    if state.get("divergence_warned"):
        return  # one warning per session is enough

    denom = cache_read + input_tokens + cache_write
    hit_rate = (cache_read / denom) if denom else 0.0

    if hit_rate < 0.30:
        state["divergence_streak"] = int(state.get("divergence_streak", 0)) + 1
    else:
        state["divergence_streak"] = 0

    if state["divergence_streak"] >= 3:
        state["divergence_warned"] = True
        logger.warning(
            "dynamic-tools: freeze divergence — scope=%s session=%s "
            "locked_mode=on but %d consecutive post-lock calls hit <30%% "
            "(last hit_rate=%.1f%%). Possible Mnemosyne prefix-break, "
            "provider cache regression, or unmodeled tool mutation. "
            "Session continues frozen; analyzer flags for review.",
            scope, session_key, state["divergence_streak"], hit_rate * 100,
        )


def _update_cache_mode_detection(
    session_key: str,
    model: str,
    cache_read: int,
    cache_write: int,
    input_tokens: int,
    scope: str = "",
) -> str:
    """Update detection state for a session and return the current mode.

    Phase 0: pure observation. Returns the mode string for the api_calls
    log line; nothing downstream gates on it yet. Phase 1 will read this
    state to decide whether to freeze the tool set at session start.

    Decision rules (per the revised plan, doc 16 Phase 3):
      · Forced mode (cache_mode != "auto") locks immediately.
      · Models in cache_auto.providers_off_models lock "off" immediately.
      · "pending" until BOTH `calls_observed >= detect_calls` AND
        `input_observed >= detect_min_input_tokens` are met. Then:
          hit_rate = read / (read + write + input)
          ≥ on_threshold → "on", else → "off".
      · Once locked, never switch mid-session.
    """
    if not session_key:
        return "off"

    forced = str(_CONFIG.get("cache_mode") or "auto").lower()
    state = _CACHE_MODE_BY_SESSION.get(session_key)
    if state is None:
        state = {
            "mode": "pending",
            "calls_observed": 0,
            "cache_read": 0,
            "cache_write": 0,
            "input": 0,
            "locked_at_call": None,
            "lock_reason": "",
        }
        _CACHE_MODE_BY_SESSION[session_key] = state

        # Forced modes resolve before the first observation lands.
        if forced == "on":
            state["mode"] = "on"
            state["lock_reason"] = "forced_on"
        elif forced == "off":
            state["mode"] = "off"
            state["lock_reason"] = "forced_off"

    # Update running stats regardless of lock state — useful for the
    # divergence detection Phase 3 will add. Cheap.
    state["calls_observed"] += 1
    state["cache_read"] += int(cache_read or 0)
    state["cache_write"] += int(cache_write or 0)
    state["input"] += int(input_tokens or 0)

    if state["mode"] != "pending":
        # Post-lock observation — divergence detection has teeth here.
        _check_divergence(state, cache_read, input_tokens, cache_write, scope, session_key)
        return state["mode"]

    # Provider blocklist: lock "off" the moment we see a known-bad model.
    blocklist = _cache_auto_cfg().get("providers_off_models") or []
    if model and isinstance(blocklist, list) and model in blocklist:
        state["mode"] = "off"
        state["locked_at_call"] = state["calls_observed"]
        state["lock_reason"] = "provider_blocklist"
        return state["mode"]  # not persisted — provider blocklist is config, not observation

    cfg = _cache_auto_cfg()
    try:
        detect_calls = max(1, int(cfg.get("detect_calls", 5)))
    except Exception:
        detect_calls = 5
    try:
        detect_min_input = max(0, int(cfg.get("detect_min_input_tokens", 5000)))
    except Exception:
        detect_min_input = 5000
    try:
        threshold = float(cfg.get("on_threshold", 0.40))
    except Exception:
        threshold = 0.40

    enough_calls = state["calls_observed"] >= detect_calls
    enough_input = (state["cache_read"] + state["cache_write"] + state["input"]) >= detect_min_input
    if enough_calls and enough_input:
        denom = state["cache_read"] + state["cache_write"] + state["input"]
        hit_rate = (state["cache_read"] / denom) if denom else 0.0
        state["hit_rate_at_lock"] = round(hit_rate, 4)
        if hit_rate >= threshold:
            state["mode"] = "on"
            state["lock_reason"] = "threshold_met"
        else:
            state["mode"] = "off"
            state["lock_reason"] = "threshold_failed"
        state["locked_at_call"] = state["calls_observed"]
        # Phase 3: persist the lock to the cross-session cache so the
        # next session of this scope can default correctly without
        # waiting for the observation window again.
        if scope:
            _persist_detection_lock(scope, state, model)

    return state["mode"]


# ─── Hook: post_api_request ────────────────────────────────────────────────

def _on_post_api_request(**kwargs) -> None:
    """Log per-API-call cache + token data for cache-economics analysis.

    Hermes fires this once per outbound API call with a normalized ``usage``
    dict (see AIAgent._usage_summary_for_api_request_hook). The tool-list
    hash captured in the contextvar is the wire-level hash sent on THIS
    call — intra-turn mutations from expand_tools produce distinct hashes
    within a single prediction_id, which is what makes the
    mutation→cache-miss correlation measurable.

    Out-of-band API calls (no prediction state) are skipped — they aren't
    gateway-dispatched and have no narrowing decision to attribute.
    """
    if not _CONFIG["enabled"] or not _CONFIG.get("log", True):
        return None
    try:
        state = _PREDICTION_CV.get()
        if state is None:
            return None
        usage = kwargs.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        session_key = str(state.get("session_id", ""))
        model = str(kwargs.get("model") or state.get("model", "") or "")
        cache_read = int(usage.get("cache_read_tokens") or 0)
        cache_write = int(usage.get("cache_write_tokens") or 0)
        input_tokens = int(usage.get("input_tokens") or 0)

        # Phase 0/3: detection state machine. Returns the current mode
        # (pending → on/off once the lock criteria are met). Phase 3
        # persists locks to the cross-session cache and runs divergence
        # checks on post-lock calls.
        cache_mode = _update_cache_mode_detection(
            session_key=session_key,
            model=model,
            cache_read=cache_read,
            cache_write=cache_write,
            input_tokens=input_tokens,
            scope=str(state.get("scope", "")),
        )

        # Phase 4: cache-TTL expiry detection. A stable-hash call with
        # zero cache_read after a long idle gap is the signature of
        # provider-side cache eviction (Anthropic's default 5m TTL).
        # The freeze logic did everything right — the TTL just lapsed
        # in between turns. Flagging this distinctly stops the analyzer
        # from misattributing it as freeze-quality regression.
        ttl_expired = False
        now = time.time()
        current_hash = str(state.get("last_tool_list_hash", ""))
        if session_key:
            mode_state = _CACHE_MODE_BY_SESSION.get(session_key) or {}
            prev_ts = mode_state.get("last_call_ts")
            prev_hash = mode_state.get("last_call_hash")
            if (
                prev_ts is not None
                and prev_hash
                and prev_hash == current_hash
                and cache_read == 0
                and (now - float(prev_ts)) > 300.0  # 5 min — Anthropic default TTL
            ):
                ttl_expired = True
            mode_state["last_call_ts"] = now
            mode_state["last_call_hash"] = current_hash
            _CACHE_MODE_BY_SESSION[session_key] = mode_state

        row = {
            "ts": now,
            "prediction_id": str(state.get("prediction_id", "")),
            "session_id": session_key,
            "scope": str(state.get("scope", "")),
            "api_call_idx": int(state.get("api_call_idx", 0)),
            "api_call_count": int(kwargs.get("api_call_count") or 0),
            "model": model,
            "provider": str(kwargs.get("provider") or state.get("provider", "") or ""),
            "api_mode": str(kwargs.get("api_mode") or ""),
            "tool_list_hash": current_hash,
            "system_hash": str(state.get("last_system_hash", "")),
            "input_tokens": input_tokens,
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
            "reasoning_tokens": int(usage.get("reasoning_tokens") or 0),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "api_duration": float(kwargs.get("api_duration") or 0.0),
            "finish_reason": str(kwargs.get("finish_reason") or ""),
            "message_count": int(kwargs.get("message_count") or 0),
            "cache_mode": cache_mode,
            "cache_ttl_expired": ttl_expired,
        }
        logger_io.log_api_call(row)
    except Exception as exc:
        logger.debug("dynamic-tools: post_api_request log failed: %s", exc)
    return None


# ─── Hook: on_session_end ──────────────────────────────────────────────────

def _on_session_end(session_id=None, **kwargs) -> None:
    """Clear all per-session state when a session ends.

    The contextvar is task-scoped so it auto-cleans, but we explicitly
    clear here to keep state hygiene clean if the same task processes
    another message. Sticky residency and lookback history are keyed by
    session so they must be evicted here to prevent cross-session bleed.

    Hermes invokes this hook with ``session_id`` set to the AIAgent's
    uuid/timestamp identifier — NOT the canonical gateway session key
    that pre_gateway_dispatch uses to index our per-session state. We
    therefore prefer the canonical key, which is still discoverable
    in-process via either a ``session_key`` kwarg (if Hermes ever passes
    one) or the ``HERMES_SESSION_KEY`` env var the gateway sets for the
    active dispatch. We evict under both shapes so legacy rows written
    under the uuid form (pre-fix) also get cleaned up.
    """
    if not _CONFIG["enabled"]:
        return None
    try:
        _PREDICTION_CV.set(None)
    except Exception:
        pass

    eviction_ids: list[str] = []
    canonical_key = ""
    if isinstance(kwargs.get("session_key"), str) and kwargs["session_key"]:
        canonical_key = str(kwargs["session_key"])
    if not canonical_key:
        env_key = os.environ.get("HERMES_SESSION_KEY") or ""
        if env_key:
            canonical_key = env_key
    if canonical_key:
        eviction_ids.append(canonical_key)
    if session_id:
        sid = str(session_id)
        if sid and sid not in eviction_ids:
            eviction_ids.append(sid)

    for sid in eviction_ids:
        sticky_key = _sticky_key_for_session(sid)
        if sticky_key:
            _STICKY_BY_KEY.pop(sticky_key, None)
        _PRIOR_MESSAGES_BY_SESSION.pop(sid, None)
        # NOTE: do NOT evict _FROZEN_BY_SESSION or _CACHE_MODE_BY_SESSION
        # here. ``on_session_end`` is a misleading hook name — Hermes
        # fires it at the end of every ``run_conversation`` call, which
        # means once per user message in multi-turn sessions
        # (conversation_loop.py:4680 warns about this explicitly).
        # Evicting the freeze here would nuke it after the first turn,
        # defeating the whole point of session-level freezing.
        # Eviction lives in _on_session_reset (true /reset, /new) only;
        # Phase 4 will add a compaction-driven eviction.
    return None


# ─── Hook: on_session_reset ────────────────────────────────────────────────

def _on_session_reset(session_id=None, **kwargs) -> None:
    """Discard cache-aware per-session state on /reset.

    `/reset` keeps the same canonical session_key but wipes the agent's
    message history — which, from the provider's cache perspective, is
    identical to a fresh session. The frozen tool set and the mode
    detection state from the prior conversation must be discarded so the
    next ``pre_gateway_dispatch`` re-freezes against the new (empty)
    history. Phase 1 will populate these dicts; Phase 0 verifies the
    eviction path runs correctly even when they're empty.

    Sticky/lookback state is also evicted by `on_session_end` semantics
    and intentionally cleared here too — `/reset` is a stronger signal
    than the gateway's per-message bookkeeping.
    """
    if not _CONFIG["enabled"]:
        return None

    eviction_ids: list[str] = []
    canonical_key = ""
    if isinstance(kwargs.get("session_key"), str) and kwargs["session_key"]:
        canonical_key = str(kwargs["session_key"])
    if not canonical_key:
        env_key = os.environ.get("HERMES_SESSION_KEY") or ""
        if env_key:
            canonical_key = env_key
    if canonical_key:
        eviction_ids.append(canonical_key)
    if session_id:
        sid = str(session_id)
        if sid and sid not in eviction_ids:
            eviction_ids.append(sid)

    for sid in eviction_ids:
        _FROZEN_BY_SESSION.pop(sid, None)
        _CACHE_MODE_BY_SESSION.pop(sid, None)
        sticky_key = _sticky_key_for_session(sid)
        if sticky_key:
            _STICKY_BY_KEY.pop(sticky_key, None)
        _PRIOR_MESSAGES_BY_SESSION.pop(sid, None)
    return None


# ─── Plugin entry point ────────────────────────────────────────────────────

def register(ctx) -> None:
    _load_user_config()
    if not _CONFIG.get("enabled"):
        logger.info("dynamic-tools: disabled in config")
        return

    # Phase 3: pre-load the cross-session detection cache so the first
    # dispatch under cache_mode: auto picks up prior-session decisions
    # instead of defaulting to a 5-call observation window every time.
    _load_detection_cache()

    # Try eager patch install. If it fails (typically a transient circular
    # import during Hermes startup), defer to first dispatch — by then
    # run_agent is fully loaded and the import will succeed.
    if not _install_patches():
        logger.info(
            "dynamic-tools: eager patch install deferred (will retry on first dispatch)"
        )

    # Register the expand_tools meta-tool
    try:
        ctx.register_tool(
            name="expand_tools",
            toolset="dynamic-tools",
            schema=expand_tools_mod.SCHEMA,
            handler=expand_tools_mod.make_handler(_PREDICTION_CV, sticky_refresh_fn=_refresh_sticky),
            description=expand_tools_mod.SCHEMA["description"],
        )
    except Exception as exc:
        logger.warning("dynamic-tools: expand_tools registration failed: %s", exc)

    # Hooks
    ctx.register_hook("pre_gateway_dispatch", _on_pre_gateway_dispatch)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_hook("on_session_reset", _on_session_reset)

    logger.info(
        "dynamic-tools: active (policy=policy.yaml, log=%s, learned_mode=%s, bypass_rate=%s)",
        _CONFIG.get("log"), _CONFIG.get("learned_mode"), _CONFIG.get("bypass_rate"),
    )
