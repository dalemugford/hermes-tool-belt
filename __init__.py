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
}

_ORIGINAL_BUILD_API_KWARGS: Any = None
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


# ─── Config loading ────────────────────────────────────────────────────────

def _load_user_config() -> None:
    """Merge user config from ~/.hermes/config.yaml over the defaults."""
    try:
        from hermes_cli.config import load_config, cfg_get  # type: ignore[import-not-found]
        cfg = load_config()
        plugin_cfg = cfg_get(cfg, "plugins", "dynamic-tools", default={})
        if not isinstance(plugin_cfg, dict):
            return
        for key in ("enabled", "log", "agent", "learned_mode", "bypass_rate"):
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

            allowed = state.get("allowed_tool_names")
            expansions = state.get("expansions") or set()
            known = state.get("known_tool_names") or set()

            # Wildcard: don't narrow, but still log on first call
            if allowed == WILDCARD_ALWAYS_ON or allowed is None:
                names = _tool_names(tools)
                state["ceiling_tools"] = names
                state["initial_allowed_tools"] = names
                state["cut_tools"] = []
                state["unknown_kept_tools"] = []
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
            state["initial_allowed_tools"] = _tool_names(filtered)
            state["cut_tools"] = cut_names
            state["unknown_kept_tools"] = unknown_kept_names

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

    agent = agent or "default"
    platform = platform or "unknown"
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
        )
        logger_io.log_prediction(record)
    except Exception as exc:
        logger.debug("dynamic-tools: log_prediction failed: %s", exc)


def _install_patches() -> bool:
    """Install the _build_api_kwargs monkey-patch.

    NOTE on the from-import / class-method situation:
      AIAgent is a class; ``_build_api_kwargs`` is a method on it. Replacing
      the method on the class affects ALL instances (existing and new). No
      from-import binding problem here — methods are looked up via the
      instance's class at call time.
    """
    global _ORIGINAL_BUILD_API_KWARGS, _PATCHED
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

    _PATCHED = True
    logger.info("dynamic-tools: patches installed (AIAgent._build_api_kwargs)")
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
        if allowed_tool_names != WILDCARD_ALWAYS_ON and allowed_tool_names is not None:
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
            extra = {
                "agent": attribution_agent,
                "platform": attribution_platform,
                "scope": attribution_scope,
                "was_initially_available": tool in initial_allowed,
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
                already_available = tool in initial_allowed
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
    return None


# ─── Plugin entry point ────────────────────────────────────────────────────

def register(ctx) -> None:
    _load_user_config()
    if not _CONFIG.get("enabled"):
        logger.info("dynamic-tools: disabled in config")
        return

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
    ctx.register_hook("on_session_end", _on_session_end)

    logger.info(
        "dynamic-tools: active (policy=policy.yaml, log=%s, learned_mode=%s, bypass_rate=%s)",
        _CONFIG.get("log"), _CONFIG.get("learned_mode"), _CONFIG.get("bypass_rate"),
    )
