"""tool-belt — usage-aware tool loading: carry what's needed, drop what isn't.

The plugin reduces per-API-call tool overhead by sending the model only the
tools likely to be useful, while respecting the user's ``platform_toolsets``
ceiling and keeping ``expand_tools`` as a safety valve for the cases the
plugin guessed wrong — on the providers where narrowing pays.

ARCHITECTURE — two postures, keyed by provider
==============================================

Whether narrowing helps or hurts is a property of the PROVIDER, per call:
some prompt-cache the prefix (OpenRouter, openai-codex — ~90%+ hit rates
measured) and some never do (ollama-cloud — 0% over thousands of calls).
A scope mixes both through primary/fallback/delegation, so the posture
is resolved against the provider a session is talking to, not the scope.
Detection state is keyed ``scope|provider``. The plugin runs one of two
postures per session:

  cache-on / carry-all (caching providers; ``cache_mode: on`` forces it)
    Ship the full enabled ceiling on every call, minus ``expand_tools``,
    and never touch the list again for the session's life. Any tool-list
    mutation on a caching provider busts the prefix cache and re-bills
    the whole conversation history at the full input rate (26K–54K
    tokens per event, measured) — far more than per-turn narrowing ever
    saves. Nothing to narrow means nothing to expand, so ``expand_tools``
    is stripped from the wire too: shipping it would only invite the one
    mutation the posture exists to prevent. The predictor, sticky
    residency, and lookback do not run; the prediction row is still
    written (ceiling == shipped, reduction 0) so the cohort stays
    visible in telemetry.

  cache-off (non-caching providers; ``cache_mode: off`` forces it)
    Per-turn narrowing with ``expand_tools`` shipped (carrying the
    expand-only manifest) and sticky residency carrying expanded tools
    forward a few turns. With no prefix cache to protect, every schema
    token not sent is a token saved. This is the value engine.

The posture is pinned at a session's first dispatch against the provider
it resolves to (observed, else the configured primary). A mid-session
provider change that would flip the posture evicts the pin so the next
dispatch re-resolves — the model switch already broke the cache, so the
re-resolution costs nothing extra.

See docs/ARCHITECTURE.md for the full design and docs/CONFIGURATION.md
for every config knob.

PATCH SURFACE — minimal, runtime-applied
========================================

  · ``AIAgent._build_api_kwargs`` — filters ``kwargs["tools"]`` per call.
    Under carry-all the filter only removes ``expand_tools``; under
    cache-off it applies the per-turn predictor's allowed set.
  · ``AIAgent._compress_context`` — evicts the session's posture pin on
    compaction so the next dispatch re-resolves against the
    post-compaction state (compaction already costs a cache break).

``self.tools`` on the AIAgent is left untouched — we narrow the on-the-wire
payload only.

LIFECYCLE
=========

  · register(ctx)
      - load the cross-session cache-mode detection state
      - install monkey-patches
      - register expand_tools meta-tool
      - register hooks: pre_gateway_dispatch, post_tool_call,
        post_api_request, on_session_end, on_session_reset

  · pre_gateway_dispatch(event, ...)
      - resolve the posture for the session (config + scope|provider
        detection cache), pinned at first dispatch
      - cache-on: carry-all state, no predictor
      - cache-off: run predictor (sticky + lookback enabled)
      - stash state in a contextvar (expand_tools mutates this same dict)

  · patched _build_api_kwargs(self, api_messages)
      - call original to get the full kwargs
      - read state from the contextvar
      - carry-all: strip expand_tools, ship everything else verbatim
      - cache-off: filter kwargs["tools"] down to the allowed set
      - on first call per turn: write the prediction row
      - on every call: write the api_calls row (cache tokens, system_hash, …)

  · post_tool_call(tool_name, …)
      - append row to tool_calls.jsonl with prediction_id linkage

  · post_api_request(usage, …)
      - per-call cache_read / cache_write capture
      - cache-mode detection state machine, per scope|provider bucket of
        the CALL's provider (locks "on"/"off")
      - provider-change eviction of the session's pin when the new
        provider's posture differs
      - cache-TTL expiry detection (stable hash + cold cache after >5min)
      - cross-session lock persistence to cache_mode_detection.json

  · on_session_end (fires per-turn — Hermes' hook semantics)
      - clears the turn contextvar only; sticky/lookback/pin/cache-mode
        state stay because this hook fires after every turn
      - then the debounced in-process auto-shape pass (_maybe_auto_shape)

  · on_session_reset (true /reset or /new)
      - evict all per-session state

Failure mode: any exception anywhere falls through to no-narrowing.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import threading
import time
from typing import Any

from . import carrying as carrying_mod
from . import expand_tools as expand_tools_mod
from . import logger_io
from . import predictor as predictor_mod
from . import presets as presets_mod
from .presets import NO_NARROWING

logger = logging.getLogger(__name__)


# ─── Module state ──────────────────────────────────────────────────────────

# Per-message prediction state. Set by pre_gateway_dispatch, consumed by
# every _build_api_kwargs call within the message's reasoning loop.
# The dict IS mutable — expand_tools writes into it; subsequent
# _build_api_kwargs calls see the additions.
_PREDICTION_CV: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "tool_belt_prediction", default=None,
)

_CONFIG: dict[str, Any] = {
    # Zero-config default: hooks are live as soon as Hermes loads the plugin
    # (the `plugins.enabled` list already gates whether the plugin loads at
    # all). `plugins.entries.tool-belt.settings.enabled: false` is the
    # explicit opt-out; an
    # UNSET key must not silently disable every hook while register() still
    # exposes expand_tools (seen live: a dangling expand_tools with no
    # narrowing behind it).
    "enabled": True,
    "log": True,
    "require_tool_call_id": True,
    # ^ Only log a tool_calls.jsonl row for a *primary* model dispatch — a
    # tool the model named in its assistant tool_use block. Hermes emits the
    # post_tool_call hook for those (carrying the provider ``tool_call_id``)
    # AND for nested/secondary dispatches routed straight through
    # ``model_tools.handle_function_call`` (code-execution sandboxes, the
    # code kernel, the MCP tools server, memory/mnemosyne batch fan-out).
    # Those nested calls never faced per-turn narrowing and arrive with an
    # EMPTY tool_call_id; logging them inflated ``uses_in_window`` in the
    # economic demotion engine. True = drop id-less nested dispatches (the
    # contract: a row means "the model dispatched this tool"). Escape hatch
    # for the unlikely case a transport emits primary calls without an id.
    #
    # KNOWN EDGE (reviewed 2026-09-03): degraded models at long context can
    # emit primary tool_use blocks with blank ids. Hermes synthesizes an id
    # only in ``build_assistant_message`` (for the replayable history); the
    # executor dispatches from the raw SDK message, so the hook receives an
    # empty id and the gate drops a REAL dispatch. Effect is a rare
    # under-count — safer for the demotion engine than over-counting (it
    # can only make a tool demote sooner), and ``require_tool_call_id:
    # false`` restores full logging if it ever matters in practice.
    "channels": {},
    "agent": "",
    # Zero-config default (full-start contract): evidence-driven shaping
    # applies automatically. `learned_mode: recommend` is the documented
    # opt-in observe/trial mode; explicit configs win as always.
    "learned_mode": "apply",
    # In-process periodic auto-shaping (session-end trigger). True by
    # default — automatic application for apply-mode scopes is the product
    # promise; `auto_shape: false` is the global opt-out. Per-scope debounce
    # interval (default 24h) overridable via `auto_shape_interval_hours`
    # (top-level or channels.<scope>.).
    "auto_shape": True,
    # Cache-off mode pipeline — sticky residency + per-turn predictor
    # lookback. Both are inert under cache-on (carry-all), where the
    # full ceiling ships on every call and there is nothing to carry
    # forward. The cache_off sub-section makes the dual-posture
    # structure explicit instead of layering MODE-ONLY notes on
    # top-level keys.
    "cache_off": {
        "sticky": {
            "enabled": True,
            "ttl_turns": 3,
            # Toolset names passed to expand_tools(category=...), NOT YAML
            # trigger group names. policy.yaml's trigger groups are named
            # differently (e.g. file_write, web_extract); sticky residency
            # keys on the model-facing toolset name resolved through
            # Hermes' toolset table.
            "categories": ["terminal", "file", "browser", "web", "code_execution", "delegation"],
        },
        "predictor": {
            # Number of prior user messages from the same session to prepend
            # to the current message before regex classification. Catches
            # signal in short replies like "yes" / "go ahead" / "no, don't"
            # whose intent lives in the preceding turn. 0 disables.
            "lookback_turns": 1,
        },
    },
    # Fraction of sessions (deterministic by session_id hash) to bypass
    # narrowing entirely, providing an A/B baseline cohort. Bypassed
    # predictions still run for telemetry (triggers_fired, etc.) but
    # active_tool_names is set to NO_NARROWING and policy_source is "bypass".
    # 0.0 disables. Override per scope via channels.<scope>.bypass_rate.
    "bypass_rate": 0.0,
    # Determines whether tools may be narrowed at all.
    #   auto — observe cache_read_tokens for the first few calls on each
    #          scope|provider bucket and self-select on|off; "on" until
    #          the empirical window concludes.
    #   on   — assume provider prefix-caching is active; carry the full
    #          ceiling (minus expand_tools) and never mutate the list.
    #   off  — assume caching is inactive; narrow tools per turn with
    #          expand_tools shipped.
    "cache_mode": "auto",
    # User-layer shaper thresholds. The five ``shape_ceiling`` keys
    # (session_window, promote_min_sessions, promote_min_calls,
    # demote_min_sessions_no_use, demote_k) resolve config.yaml → policy.yaml
    # → shaping.DEFAULTS via shaping.load_shape_ceiling_defaults(); this dict
    # is the config.yaml layer, empty by default so shipped policy.yaml
    # defaults apply unchanged. Nested (like cache_off/cache_auto), so the
    # five keys are NOT top-level settings and are not declared individually
    # in plugin.yaml's config_schema. Per-scope overrides are not supported
    # in v1 — the whole profile shares one threshold set.
    "learning": {},
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
        # hit_rate ≥ on_threshold → lock "on"; else lock "off".
        "on_threshold": 0.40,
        # Models (or provider names) empirically observed with negligible
        # provider-side cache hit rates; skip the detection window and
        # lock "off" immediately rather than spending 5 calls to
        # re-derive a known answer.
        "providers_off_models": ["kimi-k2.6:cloud", "gpt-5.4-mini"],
    },
}

# Host facts, NOT plugin settings: the configured primary provider/model
# from Hermes' top-level ``model`` block. Kept outside ``_CONFIG`` on
# purpose — ``_CONFIG`` is exactly the operator surface declared in
# plugin.yaml (locked by tests/test_config_path.py), and these are read
# from the host, never set by the operator through us. The provider is
# the posture-resolution fallback before a session has been observed on
# any provider; the model is telemetry only. Best-effort — "" when the
# host config is unreadable.
_HOST_MODEL: dict[str, str] = {"provider": "", "model": ""}

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

# ─── Cache-aware state ────────────────────────────────────────────────────
#: Posture pinned at a session's FIRST dispatch, with the provider it was
#: resolved against: ``{"mode": "on"|"off", "provider": str}``. Without the
#: pin, a detection lock landing mid-session flips the current session's
#: path turn-to-turn. Evicted on session reset, context compaction, and a
#: mid-session provider change whose posture differs (see
#: _on_post_api_request) — all three already cost a cache break.
_CACHE_DECISION_BY_SESSION: dict[str, dict[str, str]] = {}

# Cross-session detection cache. Persisted JSON at
# ``~/.hermes/state/tool-belt/cache_mode_detection.json``. Keyed by
# ``scope|provider`` (``agent:platform|provider``) — whether the prefix
# cache hits is a property of the provider a call went to, and a scope
# mixes providers via primary/fallback/delegation. Values:
#   {"scope|provider": {"mode": "on"|"off", "locked_at": ts,
#                       "lock_reason": str, "hit_rate_at_lock": float,
#                       "sessions_locked": int, "last_model": str}}
# Pre-1.0 files were keyed by bare scope; _load_detection_cache re-keys
# them under the configured primary provider. Loaded once on register();
# written on every lock event.
_DETECTION_CACHE: dict[str, dict[str, Any]] = {}
_DETECTION_CACHE_LOADED = False

# Cache-mode detection state per canonical session_key. Each entry:
#   {
#     "observed_provider": str,   # provider of the most recent API call
#     "last_call_ts": float, "last_call_hash": str,   # TTL-expiry probe
#     "buckets": {
#       provider: {
#         "mode": "pending" | "on" | "off",
#         "calls_observed": int,
#         "cache_read": int, "cache_write": int, "input": int,
#         "locked_at_call": int | None,
#         "lock_reason": str,  # "provider_blocklist" | "threshold_met" |
#                              # "threshold_failed" | "forced_on" | "forced_off"
#       }
#     }
#   }
# One bucket per provider the session has been observed on, so a failover
# call to a non-caching provider never drags a caching primary's numbers.
# Populated by _on_post_api_request; consulted by posture resolution.
_CACHE_MODE_BY_SESSION: dict[str, dict[str, Any]] = {}


# Most-recently-seen canonical session key per platform. Populated in
# _on_pre_gateway_dispatch (which sees the real session_store-derived key)
# and consumed by _on_session_reset to recover the canonical key the
# session's pin and detection state are stored under.
#
# Why this exists: Hermes' gateway fires on_session_reset with only the
# NEW session UUID and the platform string — not the canonical session_key
# the state is stored under. The new UUID is meaningless to us; without
# this back-reference, the eviction path is a silent no-op.
#
# Multi-chat tradeoff: under several active chats on the same platform,
# this map only tracks the latest. /new in one chat will evict that
# chat's state (correct) but not other chats' (also correct, because
# /new is per-chat and we only just touched this one). If chats fire
# back-to-back, the last-touched wins eviction; the others continue
# normally. Per-chat granularity would require Hermes core passing
# session_key to the hook.
_LAST_CANONICAL_BY_PLATFORM: dict[str, str] = {}

# ─── Auto-shape engine state ──────────────────────────────────────────────
# In-process throttle + lock for the session-end auto-shape trigger.
#   "not_before" — earliest wall-clock time the next full engine pass may
#     run. Comparing against it is the cheap guard: a single float compare
#     before any file I/O or module import. The real per-scope debounce
#     (default 24h, ``auto_shape_interval_hours``) is persisted in
#     learned.json's ``shaping.last_auto_shape_at`` and enforced by
#     ``shaping.auto_shape_run``.
#   "lock" — two session-ends in this process must not run the engine
#     concurrently. A second *gateway process* racing is already safe:
#     learned.write_state is an atomic whole-document rename, so the last
#     writer wins a consistent file; no cross-process locking is built.
_AUTO_SHAPE = {"lock": threading.Lock(), "not_before": 0.0}
# Re-run the (cheap) eligibility scan at most this often per process. The
# scan itself reads telemetry files, so it is not free — but per-scope
# debounce means a shorter recheck interval only costs reads, not writes.
_AUTO_SHAPE_RECHECK_SECONDS = 900.0


# ─── Config loading ────────────────────────────────────────────────────────

def _load_user_config() -> None:
    """Merge user config from ~/.hermes/config.yaml over the defaults."""
    try:
        from hermes_cli.config import load_config, cfg_get  # type: ignore[import-not-found]
        cfg = load_config()
        # Canonical: the plugin-settings subtree Hermes reserves for every
        # plugin. The pre-2026.9 `plugins.tool-belt` block is NOT read —
        # say so once rather than silently ignore an operator's settings.
        legacy = cfg_get(cfg, "plugins", "tool-belt", default=None)
        if isinstance(legacy, dict) and legacy:
            logger.warning(
                "tool-belt: ignoring settings at plugins.tool-belt — they now "
                "live under plugins.entries.tool-belt.settings (channels nest "
                "as channels.<agent>.<platform>); move them and remove the "
                "old block"
            )
        # Configured primary provider/model (Hermes' top-level ``model``
        # block) — the posture-resolution fallback before a session has
        # been observed on any provider. Read before the plugin-settings
        # gate so it lands even when the operator has no tool-belt block.
        try:
            _HOST_MODEL["provider"] = str(
                cfg_get(cfg, "model", "provider", default="") or ""
            ).strip().lower()
            _HOST_MODEL["model"] = str(
                cfg_get(cfg, "model", "default", default="") or ""
            ).strip()
        except Exception:
            pass
        plugin_cfg = cfg_get(cfg, "plugins", "entries", "tool-belt", "settings",
                             default={})
        if not isinstance(plugin_cfg, dict):
            return
        for key in (
            "enabled", "log", "require_tool_call_id", "agent", "learned_mode",
            "bypass_rate", "cache_mode",
            "auto_shape", "auto_shape_interval_hours", "always_carry",
        ):
            if key in plugin_cfg:
                _CONFIG[key] = plugin_cfg[key]
        # "always_on_extra"/"always_off" are REMOVED pre-1.0 knobs: copied
        # only so presets._warn_legacy_disable_inputs can surface a stale
        # value in user config. They are never applied.
        for key in ("channels", "always_on_extra", "always_off"):
            if key in plugin_cfg and plugin_cfg[key] is not None:
                _CONFIG[key] = plugin_cfg[key]
        if "channels" in plugin_cfg:
            from .learned import flatten_channels
            _CONFIG["channels"] = flatten_channels(plugin_cfg["channels"])
        if isinstance(plugin_cfg.get("cache_off"), dict):
            cache_off = dict(_CONFIG.get("cache_off") or {})
            user_cache_off = plugin_cfg["cache_off"]
            if isinstance(user_cache_off.get("sticky"), dict):
                sticky = dict(cache_off.get("sticky") or {})
                sticky.update(user_cache_off["sticky"])
                cache_off["sticky"] = sticky
            if isinstance(user_cache_off.get("predictor"), dict):
                predictor_cfg = dict(cache_off.get("predictor") or {})
                predictor_cfg.update(user_cache_off["predictor"])
                cache_off["predictor"] = predictor_cfg
            _CONFIG["cache_off"] = cache_off
        if isinstance(plugin_cfg.get("cache_auto"), dict):
            cache_auto = dict(_CONFIG.get("cache_auto") or {})
            cache_auto.update(plugin_cfg["cache_auto"])
            _CONFIG["cache_auto"] = cache_auto
        if isinstance(plugin_cfg.get("learning"), dict):
            learning = dict(_CONFIG.get("learning") or {})
            user_learning = plugin_cfg["learning"]
            if isinstance(user_learning.get("shape_ceiling"), dict):
                shape_ceiling = dict(learning.get("shape_ceiling") or {})
                shape_ceiling.update(user_learning["shape_ceiling"])
                learning["shape_ceiling"] = shape_ceiling
            _CONFIG["learning"] = learning
    except Exception as exc:
        logger.debug("tool-belt: config load failed (using defaults): %s", exc)


# ─── The narrowing patch ───────────────────────────────────────────────────

def _wrap_build_api_kwargs(original):
    """Wrap AIAgent._build_api_kwargs to narrow tools per API call.

    The wrapper must accept and forward every extra argument the host may
    pass (the deployed gateway calls ``_build_api_kwargs(api_messages,
    tools_for_api=...)`` on cache-plan branches): a fixed two-argument
    signature raises TypeError at call BINDING — before the fail-open
    try/except can engage — turning narrowing into an API-call crash.
    """
    def wrapped(self, api_messages, *args, **call_kwargs):
        kwargs = original(self, api_messages, *args, **call_kwargs)
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

            active_names = state.get("active_tool_names")
            expansions = state.get("expansions") or set()

            # Snapshot the live enabled ceiling once per turn. Split into the
            # built-in ceiling ``E`` (the partition domain) and MCP/plugin
            # passthrough names (outside the built-in partition). Captured here
            # — the first _build_api_kwargs call — so expand_tools has the
            # enabled ceiling recorded before it can report any activation.
            ceiling_names = _tool_names(tools)
            state["ceiling_tools"] = ceiling_names
            builtin_ceiling: set[str] = set()
            mcp_passthrough_names: list[str] = []
            for name in ceiling_names:
                base_name = _base_tool_name(name)
                # Hermes prefixes tool names with "mcp_" on the Claude Code
                # OAuth path (anthropic_adapter.py). Strip it before matching.
                # Tool Search bridge tools are pass-through like MCP tools:
                # they are the transport INTO the deferred catalog, not a
                # shapeable built-in capability.
                if (_is_mcp_tool(name) or _is_mcp_tool(base_name)
                        or _is_bridge_tool(name) or _is_bridge_tool(base_name)):
                    mcp_passthrough_names.append(name)
                else:
                    builtin_ceiling.add(base_name)
            state.setdefault("enabled_ceiling", sorted(builtin_ceiling))
            state.setdefault("mcp_passthrough_tools", mcp_passthrough_names)

            # No-narrowing (carry-all posture, no-narrowing preset, or A/B
            # bypass): don't narrow, but still log on the first call.
            if active_names == NO_NARROWING or active_names is None:
                shipped = tools
                if state.get("policy_source") == "cache_on_carry_all":
                    # Carry-all: everything ships EXCEPT expand_tools. It is
                    # a registered real tool, so Hermes always includes it;
                    # with nothing narrowed there is nothing to expand, and
                    # shipping it would invite the one mid-session mutation
                    # this posture exists to prevent. The list is otherwise
                    # untouched, so its hash is byte-stable across the
                    # session — the whole point on a caching provider.
                    shipped = [
                        t for t in tools
                        if not (isinstance(t, dict)
                                and _base_tool_name(_tool_name(t)) == "expand_tools")
                    ]
                    if len(shipped) != len(tools):
                        kwargs = dict(kwargs)
                        kwargs["tools"] = shipped
                # Snapshot once per prediction. Subsequent _build_api_kwargs
                # calls within the same turn (after the model used a tool
                # like expand_tools) must not overwrite the "what the model
                # initially saw" record.
                state.setdefault("initial_active_tools", _tool_names(shipped))
                state.setdefault("expand_only_tools", [])
                state["last_tool_list_hash"] = _tool_list_hash(shipped)
                state["last_system_hash"] = _system_message_hash(api_messages)
                state["api_call_idx"] = int(state.get("api_call_idx", 0)) + 1
                # ceiling == narrowed: reduction 0 by construction, and the
                # cohort stays visible in telemetry.
                _maybe_log_prediction(state, ceiling=shipped, narrowed=shipped)
                return kwargs

            # Resolve explicit expansions (expand_tools) — a category name
            # resolves through Hermes' toolset table to its member tool names.
            # Sticky residency (cache-off) is an activation source too, never a
            # residency change, so it joins the expanded set.
            expanded_names: set[str] = set(expansions)
            for item in expansions:
                expanded_names |= _resolve_category_to_tools(item)
            expanded_names |= set(state.get("sticky_tools") or [])

            triggered_names = set(state.get("triggered_tools") or [])

            # Finalize the three strata (A/C/X) and the per-message active set
            # against the LIVE enabled ceiling. No selection source can add a
            # tool absent from ``E``. Full-start contract: unknown enabled
            # built-ins are CARRIED; only the reconciled ``demoted`` loadout
            # (learned.apply_to_preset) moves a tool to expand_only.
            model = carrying_mod.resolve(
                enabled=builtin_ceiling,
                always_carry=set(state.get("resolved_always_carry") or []),
                carry=set(state.get("resolved_carry") or []),
                demoted=set(state.get("resolved_demoted") or []),
                triggered=triggered_names,
                expanded=expanded_names,
                passthrough=set(),  # passthrough handled directly on the tool list
                prior_active=set(state.get("prior_active_tools") or []),
            )
            active_set = model.active

            # Full-start: the pre-ceiling baseline (predictor residents ∪
            # trigger tools) understates what the model actually saw — most of
            # class C is implicit (E − A − demoted). Widen the credit baseline
            # with the resolved resident classes so expansion-credit
            # attribution never credits expand_tools for a tool that was
            # carried anyway.
            baseline = state.get("baseline_active_tools")
            if isinstance(baseline, list):
                state["baseline_active_tools"] = sorted(
                    set(baseline) | model.always_carry | model.carry
                )

            # Config always_carry pins absent from the enabled ceiling are
            # inert — warn once per scope, naming them.
            _warn_inert_pins(
                str(state.get("scope") or ""),
                state.get("config_always_carry") or [],
                builtin_ceiling,
            )

            filtered = []
            expand_only_names = []
            for t in tools:
                if not isinstance(t, dict):
                    continue
                name = _tool_name(t)
                if not name:
                    continue
                base_name = _base_tool_name(name)
                # MCP/plugin tools pass through without narrowing — Tool Search
                # owns their discovery; Tool Belt shapes built-in tools only.
                # The Tool Search bridge tools pass through for the same
                # reason: demoting the bridge would silently sever ALL access
                # to the deferred MCP/plugin catalog (mirror image of
                # _pin_expand_tools_visible, which protects the other layer's
                # hatch in the other direction).
                if (_is_mcp_tool(name) or _is_mcp_tool(base_name)
                        or _is_bridge_tool(name) or _is_bridge_tool(base_name)):
                    filtered.append(t)
                    continue
                if base_name in active_set:
                    filtered.append(t)
                else:
                    expand_only_names.append(name)

            # Diagnostic: log what's happening on the FIRST call per turn.
            if not state.get("logged"):
                logger.info(
                    "tool-belt: filter input tools=%d active=%d → kept=%d "
                    "expand_only=%d (sample_names=%s, sample_expand_only=%s)",
                    len(tools), len(active_set),
                    len(filtered), len(expand_only_names),
                    ceiling_names[:5], expand_only_names[:5],
                )

            # Append the expand-only discoverability manifest to a per-request
            # CLONE of the expand_tools schema. Built from ``model.expand_only``
            # (the residency stratum X), so it stays byte-identical across the
            # session even when a trigger later activates one of these tools —
            # residency never changes, only the active set grows. Fail OPEN:
            # any error leaves ``filtered`` and the original schema intact.
            try:
                manifest_text = expand_tools_mod.build_expand_only_manifest(
                    model.expand_only
                )
                if manifest_text:
                    filtered = [
                        expand_tools_mod.augment_schema_with_manifest(t, manifest_text)
                        if _base_tool_name(_tool_name(t)) == "expand_tools"
                        else t
                        for t in filtered
                    ]
            except Exception as exc:
                logger.debug(
                    "tool-belt: expand-only manifest skipped (%s) — original "
                    "expand_tools schema preserved", exc,
                )

            kwargs = dict(kwargs)
            kwargs["tools"] = filtered

            # Snapshot once per prediction — the filtered list on later calls
            # reflects post-expansion state and would otherwise pollute
            # "initial" with expansion-added tools.
            state.setdefault("initial_active_tools", _tool_names(filtered))
            state.setdefault("expand_only_tools", expand_only_names)
            # Record the resolved partition for telemetry / attribution.
            state["carry_always_carry"] = sorted(model.always_carry)
            state["carry_carry"] = sorted(model.carry)
            state["carry_expand_only"] = sorted(model.expand_only)

            # Per-call hash for response-side correlation with
            # cache_read_tokens. Updated every API call so intra-turn
            # mutations (expand_tools) show up distinctly from the
            # snapshot stamped on the prediction row.
            state["last_tool_list_hash"] = _tool_list_hash(filtered)
            state["last_system_hash"] = _system_message_hash(api_messages)
            state["api_call_idx"] = int(state.get("api_call_idx", 0)) + 1

            _maybe_log_prediction(state, ceiling=tools, narrowed=filtered)
            return kwargs
        except Exception as exc:
            logger.warning(
                "tool-belt: narrowing failed (%s) — returning unfiltered tools",
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


def _bridge_tool_names() -> frozenset[str]:
    """Names of Hermes' Tool Search bridge tools (tiered-disclosure access
    infrastructure): ``tool_search`` / ``tool_describe`` / ``tool_call``.

    Sourced from ``tools.tool_search.BRIDGE_TOOL_NAMES`` so a upstream rename
    tracks automatically; fail-OPEN to the hardcoded triple when hermes-agent
    isn't importable (tests, offline analysis) so pass-through immunity never
    silently vanishes.
    """
    try:
        from tools.tool_search import BRIDGE_TOOL_NAMES  # type: ignore[import-not-found]
        names = frozenset(str(n) for n in BRIDGE_TOOL_NAMES)
        if names:
            return names
    except Exception:
        pass
    return frozenset({"tool_search", "tool_describe", "tool_call"})


_BRIDGE_TOOL_NAMES: frozenset[str] = _bridge_tool_names()


def _is_bridge_tool(name: str) -> bool:
    """True for Hermes' Tool Search bridge tools.

    The bridge is the transport for the deferred MCP/plugin catalog — tools
    Tool Belt has explicitly disclaimed ownership of. Like MCP tools, the
    bridge sits OUTSIDE the built-in partition: never demotable, never counted
    as adaptive carry, never listed in the expand-only manifest. The
    ``always_carry`` pin in policy.yaml is the pure-data backup for the case
    where this structural check can't see ``tools.tool_search``.
    """
    return name in _BRIDGE_TOOL_NAMES


def _is_mcp_tool(name: str) -> bool:
    """Return True if a tool name belongs to an MCP server.

    Hermes names MCP tools ``mcp__<server>__<tool>`` (see
    ``mcp_tool.MCP_TOOL_NAME_PREFIX``). The Anthropic adapter path
    uses ``mcp_<tool>`` (single underscore). Both are checked.

    Also consults the Hermes tool registry for an ``mcp-`` toolset
    prefix, matching the logic in ``tools.tool_search.is_deferrable_tool_name``.
    Returns False if the registry is unavailable.
    """
    if name.startswith("mcp__") or name.startswith("mcp_"):
        return True
    try:
        from tools.registry import registry
        entry = registry.get_entry(name)
        if entry is not None and entry.toolset.startswith("mcp-"):
            return True
    except Exception:
        pass
    return False


def _tool_names(tools: list[dict[str, Any]]) -> list[str]:
    """Return stable tool names from a tool-schema list."""
    return [name for name in (_tool_name(t) for t in tools) if name]


def _tool_list_hash(tools: list[dict[str, Any]]) -> str:
    """Truncated sha256 of the narrowed tool list in send-order.

    Captures membership, ordering, and schema-content drift — provider
    prefix-caches fingerprint the actual schema bytes, so a reorder or a
    description change invalidates the cache even when the name set is
    stable. Analysis derives "did the prefix change?" by comparing hashes
    across consecutive turns in the same session.
    """
    try:
        payload = json.dumps(tools, sort_keys=False, ensure_ascii=False, separators=(",", ":"))
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
    tool_list_hash stays stable, carry-all is doing its job and
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
            payload = json.dumps(content, sort_keys=False, ensure_ascii=False, separators=(",", ":"))
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
    cache_off = _CONFIG.get("cache_off") or {}
    if not isinstance(cache_off, dict):
        return {}
    cfg = cache_off.get("sticky") or {}
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
    the agent slot. Hermes reserves ``default`` for the root profile, so a
    root ``HERMES_HOME`` resolves to that neutral identity. Truly
    profile-less calls still return ``""`` rather than inventing context.
    """
    agent = str(_CONFIG.get("agent") or "").strip().lower()
    if agent:
        return agent
    home = os.environ.get("HERMES_HOME", "").rstrip("/")
    if home:
        parent_dir, name = os.path.split(home)
        if os.path.basename(parent_dir) == "profiles" and name:
            return name.strip().lower()
        return os.environ.get("HERMES_PROFILE", "").strip().lower() or "default"
    return os.environ.get("HERMES_PROFILE", "").strip().lower()


def _platform_from_session_key(session_key: Any) -> str:
    """Return the platform segment of a canonical Hermes session key.

    Hermes' :func:`gateway.session.build_session_key` produces:
        ``agent:main:{platform}:{chat_type}[:...]``
    Position 1 is the *literal* string ``"main"`` — NOT a per-profile
    agent name. Position 2 is the platform.

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


def _looks_like_hermes_session_uuid(value: Any) -> bool:
    """True when *value* looks like a Hermes internal session UUID.

    Hermes' AIAgent session ids are shaped ``YYYYMMDD_HHMMSS_<hex>``
    (e.g. ``20260823_204133_466974f4``) — the id that rotates on
    ``/new`` / ``/reset`` / auto-reset. This is distinct from the
    canonical *session key* (``agent:main:{platform}:{chat_type}:...``)
    which carries a ``":"`` and stays constant across ``/new``.

    We only need to *exclude* the session key (and blanks): any non-empty
    string without a ``":"`` is accepted. That keeps us robust to future
    id-shape tweaks while never mistaking a session key for a UUID — which
    would defeat the per-session shaper grouping this field exists for.
    """
    return isinstance(value, str) and bool(value) and ":" not in value


def _hermes_session_uuid(
    event: Any = None,
    session_store: Any = None,
    kwargs: dict[str, Any] | None = None,
    session_key: str = "",
) -> str:
    """Resolve the Hermes internal session UUID for telemetry grouping.

    Telemetry-only. Unlike :func:`_canonical_session_key` (the load-bearing
    session *key* used for pin/bypass/sticky/lookback keying), this is the
    per-session UUID that changes on ``/new``. The shaper groups by it so a
    single long-lived chat still yields the distinct-session count its demote
    threshold needs.

    Resolution order (all read-only, no session side effects):

    1. ``session_store._entries[session_key].session_id`` — the live routing
       entry for this chat. Reflects the current active session and rotates
       on ``/new``. Preferred because it's guaranteed UUID-shaped and doesn't
       depend on process-global env state.
    2. ``HERMES_SESSION_ID`` env var — the id the gateway/CLI binds for the
       active turn (mirrors how :func:`_canonical_session_key` reads
       ``HERMES_SESSION_KEY``). May lag by one turn at a ``/new`` boundary,
       which is harmless for distinct-session counting.
    3. ``kwargs``/``event`` ``session_id`` — only when UUID-shaped, so we
       never capture a session key here.

    Returns ``""`` when nothing UUID-shaped is reachable; the shaper falls
    back to ``session_id`` for such rows.
    """
    if session_store is not None and session_key:
        try:
            entries = getattr(session_store, "_entries", None)
            entry = entries.get(session_key) if isinstance(entries, dict) else None
            sid = getattr(entry, "session_id", "") if entry is not None else ""
            if _looks_like_hermes_session_uuid(sid):
                return sid
        except Exception:
            pass

    env_sid = os.environ.get("HERMES_SESSION_ID") or ""
    if _looks_like_hermes_session_uuid(env_sid):
        return env_sid

    if kwargs:
        for cand in ("session_id", "session_uuid"):
            val = kwargs.get(cand)
            if _looks_like_hermes_session_uuid(val):
                return str(val)
    val = getattr(event, "session_id", None) if event is not None else None
    if _looks_like_hermes_session_uuid(val):
        return str(val)
    return ""


def _agent_platform_from_context(event: Any = None, kwargs: dict[str, Any] | None = None) -> tuple[str, str, str]:
    """Return ``(agent, platform, scope)`` for the gateway-dispatch path.

    Companion to :func:`_recover_attribution_without_state` (post-tool-call
    recovery path); the two intentionally use different precedence rules:

      · This function: env session key → kwargs hints → event attrs, with
        each later source *overwriting* the earlier — the event is
        authoritative when a message is being dispatched (we trust the
        gateway over any leftover env state).
      · Recovery path: passed-in session_id → env key → kwargs hints, with
        each later source only used when the earlier returned blank — when
        we have no context, the first plausible answer wins.

    Scope is ``{agent}:{platform}``. Agent comes from profile context
    (explicit config, HERMES_HOME profile dir, HERMES_PROFILE); the
    platform comes from ``HERMES_SESSION_KEY`` — see
    :func:`_platform_from_session_key` for that key's shape.
    Returns ``("", "", "")`` when either slot can't be filled with real
    data (analyzer buckets blank rows as "unknown").
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


#: Scopes already warned about inert config always_carry pins (per process).
_WARNED_INERT_PINS: set[tuple[str, frozenset]] = set()


def _warn_inert_pins(scope: str, pins: Any, enabled: set[str]) -> None:
    """Warn once, naming config always_carry pins absent from ``E``.

    A pinned name that is not an enabled tool is inert — the ceiling stays
    authoritative and the pin cannot re-enable anything (``∩ E`` in
    ``carrying.resolve``). Never fails; never raises.
    """
    try:
        missing = frozenset(str(t) for t in (pins or []) if str(t) not in enabled)
        if not missing:
            return
        key = (str(scope), missing)
        if key in _WARNED_INERT_PINS:
            return
        _WARNED_INERT_PINS.add(key)
        logger.warning(
            "tool-belt: config always_carry pin(s) %s for scope %r are not in "
            "the enabled tool ceiling — inert (a disabled/absent tool cannot "
            "be re-enabled by pinning; check the name or enable the tool)",
            ", ".join(sorted(missing)), scope,
        )
    except Exception:  # pragma: no cover — warning must never break dispatch
        pass


def _maybe_log_prediction(
    state: dict[str, Any],
    ceiling: list,
    narrowed: list,
) -> None:
    """Write the prediction row to predictions.jsonl.

    Runs on the FIRST API call per turn; subsequent calls in the same turn
    don't re-log.
    """
    if state.get("logged"):
        return
    state["logged"] = True
    if not _CONFIG.get("log", True):
        return
    try:
        # Residency partition finalized against the live enabled ceiling in
        # _build_api_kwargs (carrying.resolve). Empty on the no-narrowing/bypass
        # path (no narrowing ran) — counts derive from the actual A/C/X sets so
        # telemetry reflects the schema the model really saw, never the preset's
        # pre-ceiling loadout.
        always_carry_tools = list(state.get("carry_always_carry") or [])
        carry_tools = list(state.get("carry_carry") or [])
        expand_only_tools = list(
            state.get("carry_expand_only") or state.get("expand_only_tools") or []
        )
        # T = trigger-activated expand_only tools (triggered ∩ X), flattened.
        expand_only_set = set(expand_only_tools)
        trigger_activated = sorted(
            t for t in (state.get("triggered_tools") or []) if t in expand_only_set
        )
        record = logger_io.PredictionRecord(
            ts=time.time(),
            prediction_id=state.get("prediction_id", ""),
            session_id=str(state.get("session_id", "")),
            hermes_session_id=str(state.get("hermes_session_id", "")),
            channel=str(state.get("scope", state.get("channel", ""))),
            agent=str(state.get("agent", "")),
            platform=str(state.get("platform", "")),
            scope=str(state.get("scope", "")),
            message_hash=str(state.get("message_hash", "")),
            message_preview=str(state.get("message_preview", "")),
            preset=str(state.get("preset", "")),
            triggers_fired=list(state.get("triggers_fired", [])),
            triggers_suppressed=list(state.get("triggers_suppressed") or []),
            always_carry_count=len(always_carry_tools),
            carry_count=len(carry_tools),
            ceiling_count=len(ceiling),
            narrowed_count=len(narrowed),
            ceiling_tokens=logger_io.estimate_tokens(ceiling),
            narrowed_tokens=logger_io.estimate_tokens(narrowed),
            tokens_estimator=logger_io.token_estimator_name(),
            ceiling_tools=list(state.get("ceiling_tools") or _tool_names(ceiling)),
            active_tools=list(state.get("initial_active_tools") or _tool_names(narrowed)),
            expand_only_tools=expand_only_tools,
            mcp_passthrough_tools=list(state.get("mcp_passthrough_tools") or []),
            always_carry_tools=always_carry_tools,
            carry_tools=carry_tools,
            trigger_tools_by_group=dict(state.get("trigger_tools_by_group") or {}),
            trigger_activated_tools=trigger_activated,
            expanded_tools=sorted(str(t) for t in (state.get("expansions") or set())),
            sticky_tools=list(state.get("sticky_tools") or []),
            sticky_categories=list(state.get("sticky_categories") or []),
            sticky_remaining_turns=dict(state.get("sticky_remaining_turns") or {}),
            policy_source=str(state.get("policy_source", "preset")),
            policy_version=str(state.get("policy_version", "")),
            learned_mode=str(state.get("learned_mode", "apply")),
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
        # Per-tool schema size snapshot for the shaper's token-denominated
        # demotion test. The ceiling defs only exist in-process; the sidecar
        # write is debounced (1h) and never raises.
        logger_io.update_schema_sizes(ceiling)
    except Exception as exc:
        logger.debug("tool-belt: log_prediction failed: %s", exc)


def _evict_session_cache_state(session_id: str, keep_provider: str = "") -> bool:
    """Drop a session's posture pin and detection state.

    The single eviction path shared by context compaction, ``/reset`` /
    ``/new``, and a mid-session provider change (see
    ``_on_post_api_request``) — every one of them is a moment the
    provider's cached prefix is already gone, so re-resolving the posture
    on the next dispatch costs nothing extra. Returns True when anything
    was actually evicted.

    ``keep_provider`` is the provider-change form: the pin and every OTHER
    provider's bucket go, but the session stays marked as observed on
    ``keep_provider`` with that bucket intact — so the re-resolution lands
    on the provider the session is actually talking to, not the
    configured primary, and this call's observation isn't thrown away.
    """
    sid = str(session_id or "")
    if not sid:
        return False
    had_pin = _CACHE_DECISION_BY_SESSION.pop(sid, None) is not None
    if not keep_provider:
        had_mode = _CACHE_MODE_BY_SESSION.pop(sid, None) is not None
        return had_pin or had_mode
    mode_state = _CACHE_MODE_BY_SESSION.get(sid) or {}
    buckets = mode_state.get("buckets") or {}
    kept = buckets.get(keep_provider)
    _CACHE_MODE_BY_SESSION[sid] = {
        "observed_provider": keep_provider,
        "buckets": {keep_provider: kept} if kept is not None else {},
    }
    return True


def _wrap_compress_context(original):
    """Evict the session's posture pin after compaction.

    Compaction rewrites the conversation history — the provider's
    cached prefix becomes worthless because the bytes after the system
    block differ. We treat that moment as a free re-resolution
    opportunity: drop the pin and detection state so the next
    ``pre_gateway_dispatch`` resolves the posture fresh (a detection
    lock that landed mid-session can take effect here rather than
    waiting for the next session).

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
            # Only carry-all sessions re-resolve here. Cache-off sessions
            # are narrowing per turn already and their detection counters
            # must survive compaction unchanged.
            if session_key and _pinned_posture(session_key) == "on":
                if _evict_session_cache_state(session_key):
                    evicted_keys.append(session_key)
            # Defensive: also try the AIAgent's session_id (UUID) in case
            # the canonical environment key is unavailable.
            aid = getattr(self, "session_id", "") or ""
            if aid and aid != session_key and _pinned_posture(aid) == "on":
                if _evict_session_cache_state(aid):
                    evicted_keys.append(aid)
            if evicted_keys:
                logger.info(
                    "tool-belt: compaction evicted posture pin for session(s) "
                    "%s — next dispatch will re-resolve",
                    evicted_keys,
                )
        except Exception as exc:
            logger.debug("tool-belt: compaction post-eviction failed: %s", exc)
        return result

    wrapped.__wrapped__ = original  # type: ignore[attr-defined]
    return wrapped


def _pin_expand_tools_visible() -> None:
    """Keep ``expand_tools`` out of Hermes' native tiered-disclosure bridge.

    Hermes' Tool Search bridge (``model_tools._compute_tool_definitions`` →
    ``tools.tool_search.assemble_tool_defs``) runs at tool-definition time,
    *upstream* of our ``_build_api_kwargs`` narrowing. Its classifier,
    ``tool_search.is_deferrable_tool_name``, treats ``expand_tools`` as
    deferrable (a non-core, non-MCP plugin tool) and replaces it with the
    ``tool_search``/``describe``/``call`` bridge tools — hiding the one tool
    whose whole job is expanding tools. Our narrowing only ever subtracts
    from the on-the-wire list, so once the bridge has removed ``expand_tools``
    we can never add it back, and ``always_carry`` (policy.yaml) never gets a
    chance to protect it (``A = always_carry ∩ E`` is empty when the tool is
    absent from the enabled ceiling ``E``).

    Fix: wrap the classifier so ``expand_tools`` is never classified
    deferrable. It then survives into ``E`` and our existing ``always_carry``
    entry keeps it resident. Every caller of ``is_deferrable_tool_name`` lives
    inside ``tool_search`` and resolves it via the module global at call time
    (``classify_tools`` included), so rebinding the module attribute reaches
    them all. Idempotent (guarded by a marker attribute) and fail-open: any
    error leaves the bridge untouched and ``expand_tools`` merely deferred.
    """
    try:
        import tools.tool_search as _ts  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("tool-belt: tool_search unavailable, expand_tools pin skipped (%s)", exc)
        return

    original = getattr(_ts, "is_deferrable_tool_name", None)
    if original is None or getattr(original, "_tool_belt_expand_pin", False):
        return

    pinned_name = expand_tools_mod.SCHEMA["name"]

    def _is_deferrable_keep_expand(name, *args, **kwargs):
        # Only expand_tools is pinned — any other tool-belt tool keeps the
        # native classifier's default treatment.
        if name == pinned_name:
            return False
        return original(name, *args, **kwargs)

    _is_deferrable_keep_expand._tool_belt_expand_pin = True  # type: ignore[attr-defined]
    _is_deferrable_keep_expand.__wrapped__ = original  # type: ignore[attr-defined]
    _ts.is_deferrable_tool_name = _is_deferrable_keep_expand
    logger.info("tool-belt: expand_tools pinned visible against the Tool Search bridge")


def _install_patches() -> bool:
    """Install the AIAgent monkey-patches.

    Patching the methods on the ``AIAgent`` class affects all existing and
    future instances.
    """
    global _ORIGINAL_BUILD_API_KWARGS, _ORIGINAL_COMPRESS_CONTEXT, _PATCHED
    if _PATCHED:
        return True

    # Pin expand_tools visible first — the Tool Search bridge runs at
    # tool-definition time and is independent of run_agent, so install this
    # even on builds where the run_agent patch below can't (yet) load.
    _pin_expand_tools_visible()

    try:
        import run_agent  # type: ignore[import-not-found]
    except ImportError as exc:
        logger.warning("tool-belt: cannot import run_agent: %s", exc)
        return False

    AIAgent = getattr(run_agent, "AIAgent", None)
    if AIAgent is None or not hasattr(AIAgent, "_build_api_kwargs"):
        logger.warning("tool-belt: AIAgent._build_api_kwargs not found")
        return False

    _ORIGINAL_BUILD_API_KWARGS = AIAgent._build_api_kwargs
    AIAgent._build_api_kwargs = _wrap_build_api_kwargs(_ORIGINAL_BUILD_API_KWARGS)

    # Compaction-driven posture-pin eviction. No `post_compaction`
    # hook exists in Hermes, so we wrap the forwarder method directly.
    # Best-effort — when _compress_context is missing (older Hermes
    # versions, custom builds) the pin persists across compaction, which
    # only delays a mid-session detection lock to the next session and
    # doesn't corrupt anything.
    if hasattr(AIAgent, "_compress_context"):
        _ORIGINAL_COMPRESS_CONTEXT = AIAgent._compress_context
        AIAgent._compress_context = _wrap_compress_context(_ORIGINAL_COMPRESS_CONTEXT)
        logger.info("tool-belt: patches installed (_build_api_kwargs, _compress_context)")
    else:
        logger.info("tool-belt: patches installed (_build_api_kwargs only — _compress_context missing)")

    _PATCHED = True
    return True


# ─── Helpers ──────────────────────────────────────────────────────────────

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
    cache_off = _CONFIG.get("cache_off") or {}
    cfg = cache_off.get("predictor") if isinstance(cache_off, dict) else None
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


# ─── Cache-aware posture ───────────────────────────────────────────────────

def _pinned_posture(session_id: str) -> str:
    """The session's pinned posture ("on"/"off"), or "" when unpinned."""
    pin = _CACHE_DECISION_BY_SESSION.get(str(session_id or ""))
    if not isinstance(pin, dict):
        return ""
    mode = pin.get("mode")
    return mode if mode in ("on", "off") else ""


def _session_provider(session_id: str) -> str:
    """Provider this session was most recently observed on ("" if none yet).

    Set by ``_on_post_api_request`` from the per-call ``provider`` Hermes
    reports, so it reflects failover and ``/model`` switches — not just the
    configured primary.
    """
    mode_state = _CACHE_MODE_BY_SESSION.get(str(session_id or "")) or {}
    return str(mode_state.get("observed_provider") or "")


def _bucket_mode(scope: str, provider: str, session_key: str = "") -> str:
    """Locked mode ("on"/"off") for a scope|provider bucket, or "" if pending.

    The session's in-process bucket wins when it has locked — forced
    modes and provider-blocklist locks live only there, never in the
    persisted cache — then the cross-session cache.
    """
    provider = str(provider or "").strip().lower()
    buckets = (_CACHE_MODE_BY_SESSION.get(str(session_key or "")) or {}).get("buckets") or {}
    mode = (buckets.get(provider) or {}).get("mode")
    if mode in ("on", "off"):
        return mode
    return _cached_detection_mode(scope, provider)


def _resolve_posture_for_provider(scope: str, provider: str, session_key: str = "") -> str:
    """Posture ("on"/"off") for a scope|provider bucket, unpinned.

      · Explicit ``cache_mode: off`` → "off" — per-turn narrowing.
      · Explicit ``cache_mode: on`` → "on" — carry-all.
      · ``cache_mode: auto`` (default) → consult the bucket (this
        session's in-process lock, then the cross-session detection
        cache). If it locked to a definite mode, honor it. Otherwise
        default to "on": until proven uncached, protect prefix-cache
        stability.
    """
    mode = str(_CONFIG.get("cache_mode") or "auto").lower()
    if mode in ("on", "off"):
        return mode
    locked = _bucket_mode(scope, provider, session_key)
    return locked if locked in ("on", "off") else "on"


def _resolve_cache_mode_for_session(
    session_id: str, scope: str = "", provider: str = "",
) -> str:
    """Decide this session's posture: carry-all ("on") or narrow ("off").

    The posture is a property of the PROVIDER the session talks to, so it
    resolves against one, in order: (a) the provider this session has
    already been observed on (failover / ``/model`` aware), (b) the
    explicit ``provider`` argument, (c) the configured primary provider,
    (d) "" — the bare-scope entry ``_cached_detection_mode`` falls back
    to for pre-1.0 detection files.

    The decision is PINNED at the session's first dispatch
    (``_CACHE_DECISION_BY_SESSION``) together with the provider it was
    resolved against: a detection lock landing mid-session applies to
    the NEXT session, never this one. The one mid-session re-resolution
    is a provider change whose posture differs — ``_on_post_api_request``
    evicts the pin then, because the switch already broke the cache.
    """
    sid = str(session_id or "")
    pinned = _pinned_posture(sid)
    if pinned:
        return pinned
    resolved_provider = (
        _session_provider(sid)
        or str(provider or "").strip().lower()
        or str(_HOST_MODEL.get("provider") or "")
    )
    decision = _resolve_posture_for_provider(scope, resolved_provider, session_key=sid)
    if sid:
        _CACHE_DECISION_BY_SESSION[sid] = {
            "mode": decision, "provider": resolved_provider,
        }
    return decision


def _build_carry_all_state(
    preset: Any,
    *,
    session_id: str,
    hermes_session_id: str = "",
    scope: str,
    channel: str,
    agent: str,
    platform: str,
    message: str,
) -> dict[str, Any]:
    """Construct the contextvar state for a carry-all (cache-on) dispatch.

    Nothing is narrowed, so the predictor, sticky residency, and lookback
    do not run and there is no residency partition to record. The fields
    are still present with empty values so downstream consumers
    (telemetry, post_tool_call) don't need conditional reads, and
    ``policy_source`` is stamped ``"cache_on_carry_all"`` — distinct from
    ``"bypass"``, the internal A/B control cohort — so the analyzer can
    tell the two unnarrowed cohorts apart.
    """
    return {
        "prediction_id": logger_io.new_prediction_id(),
        "active_tool_names": NO_NARROWING,
        "baseline_active_tools": NO_NARROWING,
        "resolved_always_carry": list(getattr(preset, "always_carry", []) or []),
        "resolved_carry": list(getattr(preset, "carry", []) or []),
        "resolved_demoted": list(getattr(preset, "demoted", []) or []),
        "config_always_carry": list(getattr(preset, "config_always_carry", []) or []),
        "triggered_tools": [],
        "prior_active_tools": [],
        "expansions": set(),
        "channel": channel,
        "agent": agent,
        "platform": platform,
        "scope": scope,
        # Sticky disabled — empty key short-circuits all sticky helpers.
        "sticky_key": "",
        "session_id": session_id,
        "hermes_session_id": hermes_session_id,
        "message_hash": logger_io.hash_message(message),
        "message_preview": logger_io.message_preview(message),
        "preset": str(getattr(preset, "name", "") or ""),
        "triggers_fired": [],
        "triggers_suppressed": [],
        "trigger_tools_by_group": {},
        "sticky_tools": [],
        "sticky_categories": [],
        "sticky_remaining_turns": {},
        "policy_source": "cache_on_carry_all",
        "policy_version": getattr(preset, "policy_version", ""),
        "learned_mode": getattr(preset, "learned_mode", _CONFIG.get("learned_mode", "apply")),
        "learned_scope": getattr(preset, "learned_scope", ""),
        "learned_changes": list(getattr(preset, "learned_changes", []) or []),
        "lookback_used": 0,
        "lookback_turns_config": 0,
        "logged": False,
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
        # Telemetry-only: the Hermes internal session UUID that rotates on
        # /new. Distinct from session_id (the session key) — see
        # _hermes_session_uuid. Flows into the prediction record and the
        # shaper's per-session grouping; touches nothing load-bearing.
        hermes_session_id = _hermes_session_uuid(
            event=event,
            session_store=session_store,
            kwargs=kwargs,
            session_key=str(session_id),
        )

        # Record the canonical session_key for this platform so that the
        # next on_session_reset hook (which Hermes fires with only the
        # NEW session UUID + platform) can find what to evict.
        if platform and session_id:
            _LAST_CANONICAL_BY_PLATFORM[str(platform)] = str(session_id)

        # Slash-command messages (/new, /reset, /resume, /model, ...) are
        # intercepted by the gateway's command router AFTER this hook
        # fires but BEFORE any LLM call. A pin resolved here would never
        # reach an API call to be logged. Skip the whole pipeline for
        # these — we've already tracked the canonical key above so the
        # subsequent on_session_reset can evict any pre-existing pin
        # cleanly.
        if isinstance(message, str) and message.strip().startswith("/"):
            return None

        # Carry-all (cache-on): the provider prompt-caches, so the full
        # ceiling ships on every call and the list is never mutated
        # mid-session. There is nothing to predict or narrow — the
        # predictor, sticky residency, and lookback all stay off. The
        # preset still resolves so the prediction row can name the policy
        # the cohort ran under.
        cache_decision = _resolve_cache_mode_for_session(str(session_id), scope=scope)
        if cache_decision == "on":
            preset = presets_mod.resolve_preset(_CONFIG, scope)
            _PREDICTION_CV.set(_build_carry_all_state(
                preset,
                session_id=str(session_id),
                hermes_session_id=hermes_session_id,
                scope=scope,
                channel=channel,
                agent=agent,
                platform=platform,
                message=message,
            ))
            return None

        # Cache-off: per-turn narrowing. Narrow sticky-residency key:
        # session-scoped, opaque. Distinct from the coarse policy scope
        # (agent:platform).
        sticky_key = _sticky_key_for_session(str(session_id))
        lookback_turns = _predictor_lookback_turns()
        prior_messages = _lookback_prior_messages(str(session_id), lookback_turns)
        predictor_input = _compose_predictor_input(prior_messages, message)

        preset = presets_mod.resolve_preset(_CONFIG, scope)
        prediction = predictor_mod.predict(predictor_input, attachments, preset)
        resolved_always_carry = list(getattr(preset, "always_carry", []) or [])
        resolved_carry = list(getattr(preset, "carry", []) or [])
        resolved_demoted = list(getattr(preset, "demoted", []) or [])
        config_pins = list(getattr(preset, "config_always_carry", []) or [])
        trigger_tools = _trigger_tools_by_group(preset, prediction.triggers_fired)
        triggered_tool_names = sorted({t for tools in trigger_tools.values() for t in tools})

        # Push the current message onto the session's lookback ring AFTER
        # prediction so we don't include the current message as its own prior.
        _record_lookback_message(str(session_id), message, lookback_turns)
        _decay_sticky(sticky_key)
        sticky_tools, sticky_categories, sticky_remaining = _live_sticky(sticky_key)

        active_tool_names = prediction.active_tool_names
        # Capture the pre-sticky baseline. This is the set the model would
        # see *without* any prior expansion (sticky carries tools forward
        # only because a prior turn ran expand_tools). The credit decision
        # in _on_post_tool_call needs this to distinguish "model could have
        # called this anyway" from "this call is the payoff of an earlier
        # expand". NO_NARROWING is preserved (bypass cohort / no-narrowing
        # preset already see every tool — nothing to attribute to expansion).
        if active_tool_names == NO_NARROWING or active_tool_names is None:
            baseline_active_tools: Any = NO_NARROWING
        else:
            baseline_active_tools = sorted(set(active_tool_names))
            if sticky_tools:
                active_tool_names = sorted(set(active_tool_names) | set(sticky_tools))

        # A/B baseline: deterministically bypass narrowing for a fraction of
        # sessions. The predictor still ran so we keep telemetry, but the
        # active set widens to the full ceiling and the policy_source is
        # stamped so the analyzer can compare cohorts.
        bypassed = _should_bypass(scope, str(session_id))
        if bypassed:
            active_tool_names = NO_NARROWING
            # Bypass means no narrowing — sticky/lookback aren't narrowing
            # anything, so their telemetry fields would falsely imply
            # narrowing signal in the bypass cohort. Zero them out for
            # cohort-comparison cleanliness.
            sticky_tools = []
            sticky_categories = []
            sticky_remaining = {}
            prior_messages = []
        policy_source = "bypass" if bypassed else getattr(preset, "policy_source", "preset")

        _PREDICTION_CV.set({
            "prediction_id": logger_io.new_prediction_id(),
            "active_tool_names": active_tool_names,
            "baseline_active_tools": baseline_active_tools,
            # Residency strata carried separately so the partition can be
            # finalized against the live enabled ceiling in _build_api_kwargs.
            "resolved_always_carry": resolved_always_carry,
            "resolved_carry": resolved_carry,
            "resolved_demoted": resolved_demoted,
            "config_always_carry": config_pins,
            "triggered_tools": triggered_tool_names,
            "prior_active_tools": [],
            "expansions": set(),
            "channel": channel,
            "agent": agent,
            "platform": platform,
            "scope": scope,
            "sticky_key": sticky_key,
            "session_id": session_id,
            "hermes_session_id": hermes_session_id,
            "message_hash": logger_io.hash_message(message),
            "message_preview": logger_io.message_preview(message),
            "preset": prediction.preset_name,
            "triggers_fired": prediction.triggers_fired,
            "triggers_suppressed": prediction.triggers_suppressed,
            "trigger_tools_by_group": trigger_tools,
            "sticky_tools": sticky_tools,
            "sticky_categories": sticky_categories,
            "sticky_remaining_turns": sticky_remaining,
            "policy_source": policy_source,
            "policy_version": getattr(preset, "policy_version", ""),
            "learned_mode": getattr(preset, "learned_mode", _CONFIG.get("learned_mode", "apply")),
            "learned_scope": getattr(preset, "learned_scope", ""),
            "learned_changes": list(getattr(preset, "learned_changes", []) or []),
            "lookback_used": len(prior_messages),
            "lookback_turns_config": lookback_turns,
            "logged": False,
        })
    except Exception as exc:
        logger.warning("tool-belt: pre_gateway_dispatch failed: %s", exc)
    return None


# ─── Hook: post_tool_call ──────────────────────────────────────────────────

def _recover_attribution_without_state(
    session_id: Any,
    kwargs: dict[str, Any],
) -> tuple[str, str, str]:
    """Best-effort ``(agent, platform, scope)`` for out-of-band tool calls.

    Companion to :func:`_agent_platform_from_context` (the dispatch path);
    see that docstring for why the two functions have different
    precedence semantics. Used when the contextvar has no per-turn
    state — typically background tools fired outside a gateway-dispatched
    message loop.

    Resolution order (each later source only used when earlier returned
    blank — "first plausible answer wins"):

      · Platform — passed ``session_id`` if it carries the canonical
        ``agent:main:{platform}:...`` shape, else ``HERMES_SESSION_KEY``
        env, else kwargs hints (platform/source/transport).
      · Agent — :func:`_profile_agent_name` only.

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
        # Provider-assigned id for the model's tool_use request. Present only
        # on a *primary* dispatch: Hermes' executor emits post_tool_call for
        # the model's assistant tool_use blocks via
        # ``tool_executor._emit_terminal_post_tool_call``, threading the real
        # tool_call_id (and session_id). Nested/secondary dispatches — tools
        # invoked straight through ``model_tools.handle_function_call`` by
        # code-execution sandboxes, the code kernel, the MCP tools server, or
        # memory/mnemosyne batch fan-out — pass neither, so it arrives empty.
        tool_call_id = str(kwargs.get("tool_call_id") or "")
        if tool_name:
            tool = str(tool_name)
            # A row in tool_calls.jsonl means "the model dispatched this tool"
            # — nothing else. Nested dispatches (empty tool_call_id) never
            # faced per-turn narrowing; counting them as calls inflated
            # ``uses_in_window`` in the shaper's economic demotion engine
            # (a resolved-but-not-dispatched tool looked heavily used, so the
            # penalty side ballooned and the tool was over-carried). Drop them.
            # ``require_tool_call_id: false`` restores the pre-fix behavior.
            if _CONFIG.get("require_tool_call_id", True) and not tool_call_id:
                return None
            initial_allowed = set(state.get("initial_active_tools") or []) if state else set()
            # Baseline = the active set *before* sticky carry-over merged in.
            # Used for credit attribution: a tool only sticky-carries because
            # a prior expand_tools ran — those calls should be credited to
            # expansion, not classified as "initially available."
            # Falls back to initial_allowed when no prediction state is
            # attached; NO_NARROWING means every tool is initially
            # available.
            baseline_raw = state.get("baseline_active_tools") if state else None
            if baseline_raw == NO_NARROWING:
                baseline_allowed: set[str] | None = None  # None means "everything"
            elif baseline_raw is None:
                baseline_allowed = set(initial_allowed)
            else:
                baseline_allowed = set(baseline_raw)
            expand_only = set(state.get("expand_only_tools") or []) if state else set()
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
            # Tag the call's source so the analyzer can filter cleanly.
            #   gateway  — a real prediction context was attached (the
            #              normal path: pre_gateway_dispatch ran for this
            #              user message and the tool call belongs to it).
            #   cron     — Hermes' cron runner uses `cron_<hash>_<ts>`
            #              session ids and bypasses pre_gateway_dispatch
            #              entirely, so there's no prediction context to
            #              attribute against. These calls are NOT narrowed.
            #   subagent — A subagent call inside a parent gateway session.
            #              The parent has a prediction context; the
            #              subagent doesn't see it. These calls inherit
            #              the parent's tool ceiling but aren't subject
            #              to narrowing themselves.
            sid_str = str(session_id or "")
            if sid_str.startswith("cron_"):
                source = "cron"
            elif prediction_id:
                source = "gateway"
            else:
                source = "subagent"
            # ``activation_source`` — the carrying class that made this tool
            # active this turn (v2). Derived from the resolved A/C/X partition
            # captured in state; refined to expansion/sticky_expansion below
            # when the credit logic confirms an expand-driven call. MCP/plugin
            # pass-through and out-of-band (cron/subagent) calls resolve to
            # ``external`` — outside the built-in partition, never shaping
            # evidence. A no-narrowing/bypass turn is ``full_ceiling``.
            partition_ac = set(state.get("carry_always_carry") or []) if state else set()
            partition_c = set(state.get("carry_carry") or []) if state else set()
            triggered_set = set(state.get("triggered_tools") or []) if state else set()
            if not state or source in ("cron", "subagent"):
                activation_source = "external"
            elif baseline_allowed is None:
                activation_source = "full_ceiling"
            elif base_name in partition_ac or tool in partition_ac:
                activation_source = "always_carry"
            elif base_name in partition_c or tool in partition_c:
                activation_source = "carry"
            elif base_name in triggered_set or tool in triggered_set:
                activation_source = "trigger"
            elif tool in expanded or base_name in expanded:
                activation_source = "expansion"
            else:
                activation_source = "external"
            extra = {
                "agent": attribution_agent,
                "platform": attribution_platform,
                "scope": attribution_scope,
                "source": source,
                "tool_call_id": tool_call_id,
                "was_initially_active": was_initially_available,
                "was_expand_only": tool in expand_only,
                "activated_by_expansion": tool in expanded,
                "activation_source": activation_source,
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
                    extra["expansion_provided_access"] = True
                    extra["expanded_tool"] = expand_use.get("expanded_tool", tool)
                    extra["turns_until_used"] = expand_use.get("turns_until_used", 0)
                    extra["expand_category"] = expand_use.get("category", extra.get("expand_category", ""))
                    extra["expansion_prediction_id"] = expand_use.get("expansion_prediction_id", "")
                    # A prior-turn sticky recovery carries remaining_turns; a
                    # same-turn expand_tools round-trip does not. Distinguish the
                    # two activation sources for truthful attribution.
                    if "sticky_remaining_turns" in expand_use:
                        extra["sticky_remaining_turns"] = expand_use.get("sticky_remaining_turns")
                        extra["activation_source"] = "sticky_expansion"
                    else:
                        extra["activation_source"] = "expansion"
                    if state and pending_expansion:
                        state["pending_expansion"] = None
                elif pending_expansion:
                    extra["expansion_provided_access"] = False
            log_args = args if tool == "expand_tools" else None
            log_result = result if tool == "expand_tools" else None
            logger_io.log_tool_call(
                prediction_id, str(session_id), tool,
                args=log_args, result=log_result, extra=extra,
            )
    except Exception as exc:
        logger.debug("tool-belt: post_tool_call log failed: %s", exc)
    return None


# ─── Cache-mode detection ──────────────────────────────────────────────────

def _cache_auto_cfg() -> dict[str, Any]:
    cfg = _CONFIG.get("cache_auto") or {}
    return cfg if isinstance(cfg, dict) else {}


def _detection_cache_path():
    """Resolve the persisted detection cache path."""
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return os.path.join(home, "state", "tool-belt", "cache_mode_detection.json")


def _detection_key(scope: str, provider: str = "") -> str:
    """Detection-cache key: ``scope|provider`` (bare ``scope`` when unknown)."""
    scope = str(scope or "")
    provider = str(provider or "").strip().lower()
    return f"{scope}|{provider}" if provider else scope


def _load_detection_cache() -> None:
    """Read the persisted detection cache into memory exactly once.

    Pre-1.0 files were keyed by bare scope (no ``|``). Those entries are
    re-keyed under the configured primary provider — the only provider a
    scope-level lock could have been observing — and persisted in the new
    format on the next save. When the primary is unknown the bare key is
    kept as-is; ``_cached_detection_mode`` falls back to it.

    Failures are silent — a missing or corrupt file means we fall through
    to the safe-default "on" for every bucket until the first lock writes
    a new entry. No telemetry is lost in either direction.
    """
    global _DETECTION_CACHE_LOADED
    if _DETECTION_CACHE_LOADED:
        return
    _DETECTION_CACHE_LOADED = True
    try:
        with open(_detection_cache_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return
        primary = str(_HOST_MODEL.get("provider") or "")
        migrated = 0
        for key, entry in data.items():
            if not (isinstance(key, str) and isinstance(entry, dict)):
                continue
            if "|" not in key and primary:
                new_key = _detection_key(key, primary)
                # A new-format entry for the same bucket is newer evidence.
                if new_key not in data and new_key not in _DETECTION_CACHE:
                    _DETECTION_CACHE[new_key] = entry
                    migrated += 1
                continue
            _DETECTION_CACHE[key] = entry
        if migrated:
            logger.info(
                "tool-belt: re-keyed %d scope-only detection entr%s under "
                "primary provider %r", migrated, "y" if migrated == 1 else "ies",
                primary,
            )
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.debug("tool-belt: detection cache load failed: %s", exc)


def _save_detection_cache() -> None:
    """Persist the detection cache atomically."""
    path = _detection_cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_DETECTION_CACHE, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception as exc:
        logger.debug("tool-belt: detection cache save failed: %s", exc)


def _persist_detection_lock(
    scope: str,
    state: dict[str, Any],
    model: str,
    provider: str = "",
) -> None:
    """Record a bucket's lock result to the cross-session cache.

    The cache is keyed by ``scope|provider``. Same-bucket sessions that
    lock the same way bump ``sessions_locked``; a disagreeing lock
    overwrites with ``sessions_locked = 1`` — the most recent reality
    wins. We don't average or smooth because cache behavior is a property
    of the provider+model setup, not a noisy signal that benefits from
    averaging — when it changes, it changes hard (provider migration,
    model swap).

    Provider-blocklist locks are skipped: they're a config statement,
    not an observation, and don't need persistence to be effective.
    """
    if not scope or state.get("mode") not in ("on", "off"):
        return
    if state.get("lock_reason") == "provider_blocklist":
        return
    key = _detection_key(scope, provider)
    cached = _DETECTION_CACHE.get(key)
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
    _DETECTION_CACHE[key] = new_entry
    _save_detection_cache()


def _cached_detection_mode(scope: str, provider: str = "") -> str:
    """Consult the cross-session cache for a scope|provider's locked mode.

    Returns "on", "off", or "" (no cache entry). Auto mode at dispatch
    time uses this to resolve the posture without repeating the
    observation window. A bare-scope entry (pre-1.0 file whose primary
    provider was unknown at load) is the fallback lookup.
    """
    if not scope:
        return ""
    entry = _DETECTION_CACHE.get(_detection_key(scope, provider))
    if not entry and provider:
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
    provider: str = "",
    provider_changed: bool = False,
) -> None:
    """Post-lock observation: does the locked mode still match reality?

    A bucket locked "on" with consistently low hit rate post-lock is
    evidence that carry-all is the wrong call for this session — a
    changing prompt prefix, a tools mismatch we didn't anticipate, or
    simply a cold-cache window. We don't switch postures mid-session
    (the switch itself busts cache), but we surface the divergence so
    the analyzer can flag the session for forensic review.

    Operates per scope|provider bucket. ``provider_changed`` marks a call
    that landed on a different provider than the session's previous call
    (failover, ``/model``, or a switch back): it is EXPECTED to miss —
    that provider has no prefix for this history — so the streak resets
    and the call never counts toward a warning.

    Threshold: 3 consecutive post-lock calls with hit rate below 30%
    when locked "on". 3 is short enough to fire on a real problem
    without flapping on transient noise; 30% is a deliberately low floor,
    so true cache breakage sits well clear of it.
    """
    if state.get("mode") != "on" or state.get("locked_at_call") is None:
        return
    if state.get("divergence_warned"):
        return  # one warning per bucket is enough
    if provider_changed:
        state["divergence_streak"] = 0
        return

    denom = cache_read + input_tokens + cache_write
    hit_rate = (cache_read / denom) if denom else 0.0

    if hit_rate < 0.30:
        state["divergence_streak"] = int(state.get("divergence_streak", 0)) + 1
    else:
        state["divergence_streak"] = 0

    if state["divergence_streak"] >= 3:
        state["divergence_warned"] = True
        logger.warning(
            "tool-belt: cache divergence — scope=%s provider=%s session=%s "
            "locked_mode=on but %d consecutive post-lock calls hit <30%% "
            "(last hit_rate=%.1f%%). Possible Mnemosyne prefix-break, "
            "provider cache regression, or unmodeled tool mutation. "
            "Session continues carry-all; analyzer flags for review.",
            scope, provider, session_key, state["divergence_streak"],
            hit_rate * 100,
        )


def _update_cache_mode_detection(
    session_key: str,
    model: str,
    cache_read: int,
    cache_write: int,
    input_tokens: int,
    scope: str = "",
    provider: str = "",
) -> str:
    """Update detection state for a session|provider and return its mode.

    Accumulation and the lock happen in the bucket of the CALL's
    provider (``_CACHE_MODE_BY_SESSION[session_key]["buckets"][provider]``),
    so a failover call to a non-caching provider never drags a caching
    primary's numbers. The session-level ``observed_provider`` is updated
    on every call.

    Decision rules:
      · Forced mode (cache_mode != "auto") locks immediately.
      · Models — or provider names — in cache_auto.providers_off_models
        lock "off" immediately.
      · "pending" until BOTH `calls_observed >= detect_calls` AND
        `input_observed >= detect_min_input_tokens` are met. Then:
          hit_rate = read / (read + write + input)
          ≥ on_threshold → "on", else → "off".
      · Once locked, a bucket never switches mid-session.
    """
    if not session_key:
        return "off"

    forced = str(_CONFIG.get("cache_mode") or "auto").lower()
    provider = str(provider or "").strip().lower()
    session_state = _CACHE_MODE_BY_SESSION.setdefault(session_key, {})
    prev_provider = session_state.get("observed_provider")
    provider_changed = prev_provider is not None and prev_provider != provider
    session_state["observed_provider"] = provider
    buckets = session_state.setdefault("buckets", {})
    state = buckets.get(provider)
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
        buckets[provider] = state

        # Forced modes resolve before the first observation lands.
        if forced == "on":
            state["mode"] = "on"
            state["lock_reason"] = "forced_on"
        elif forced == "off":
            state["mode"] = "off"
            state["lock_reason"] = "forced_off"

    # Update running stats regardless of lock state so post-lock divergence
    # remains observable.
    state["calls_observed"] += 1
    state["cache_read"] += int(cache_read or 0)
    state["cache_write"] += int(cache_write or 0)
    state["input"] += int(input_tokens or 0)

    if state["mode"] != "pending":
        # Post-lock observation — divergence detection has teeth here.
        _check_divergence(
            state, cache_read, input_tokens, cache_write, scope, session_key,
            provider=provider, provider_changed=provider_changed,
        )
        return state["mode"]

    # Provider blocklist: lock "off" the moment we see a known-bad model
    # or provider.
    blocklist = _cache_auto_cfg().get("providers_off_models") or []
    if isinstance(blocklist, list):
        listed = {str(x).strip().lower() for x in blocklist if x}
        if (model and str(model).strip().lower() in listed) or (provider and provider in listed):
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
        # Persist the lock so the next session can choose correctly without
        # repeating the observation window.
        if scope:
            _persist_detection_lock(scope, state, model, provider=provider)

    return state["mode"]


def _provider_caches(scope: str, provider: str, session_key: str = "") -> bool | None:
    """Locked verdict for a scope|provider bucket: True ("on"), False ("off"),
    None while pending.

    This is the ``provider_caches`` contract on every ``api_calls`` row —
    the savings code reads it, so its meaning is exactly "does this
    provider prefix-cache" (per ``_bucket_mode``), never the session
    posture.
    """
    mode = _bucket_mode(scope, provider, session_key)
    if mode == "on":
        return True
    if mode == "off":
        return False
    return None


def _maybe_evict_on_provider_change(
    session_key: str, scope: str, provider: str,
) -> None:
    """Re-resolve the session's posture when its provider changes.

    Dale's edge case: ``/model`` in Telegram, or a failover, moves a
    session from a caching provider to a non-caching one (or back). The
    pin was resolved against the OLD provider; if the NEW provider's
    posture differs, keep-pinning would narrow on a caching provider or
    carry-all on a non-caching one for the rest of the session. The
    switch already busted the provider's prefix cache, so evicting the
    pin — through the same path compaction and ``/reset`` use — and
    letting the next dispatch re-resolve costs nothing. Detection
    counters for the old provider go with it; the new provider's bucket
    starts fresh.
    """
    pin = _CACHE_DECISION_BY_SESSION.get(session_key)
    if not isinstance(pin, dict):
        return
    provider = str(provider or "").strip().lower()
    pinned_provider = str(pin.get("provider") or "")
    if not provider or provider == pinned_provider:
        return
    new_posture = _resolve_posture_for_provider(scope, provider, session_key=session_key)
    if new_posture == pin.get("mode"):
        # Same posture on the new provider — the pin stays, but record
        # the provider so a later switch compares against reality.
        pin["provider"] = provider
        return
    _evict_session_cache_state(session_key, keep_provider=provider)
    logger.info(
        "tool-belt: session %s moved from provider %r to %r — posture %s → %s; "
        "pin evicted, next dispatch re-resolves",
        session_key, pinned_provider, provider, pin.get("mode"), new_posture,
    )


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
        scope = str(state.get("scope", ""))
        model = str(kwargs.get("model") or state.get("model", "") or "")
        provider = str(kwargs.get("provider") or state.get("provider", "") or "")
        cache_read = int(usage.get("cache_read_tokens") or 0)
        cache_write = int(usage.get("cache_write_tokens") or 0)
        input_tokens = int(usage.get("input_tokens") or 0)

        # Detection returns pending/on/off for the CALL's provider bucket,
        # persists settled scope|provider decisions, and checks post-lock
        # divergence.
        cache_mode = _update_cache_mode_detection(
            session_key=session_key,
            model=model,
            cache_read=cache_read,
            cache_write=cache_write,
            input_tokens=input_tokens,
            scope=scope,
            provider=provider,
        )
        provider_caches = _provider_caches(scope, provider, session_key)
        # Mid-session provider change (failover, /model): re-resolve the
        # posture if the new provider's differs. Runs AFTER detection so
        # the new provider's bucket keeps this call's observation.
        if session_key:
            _maybe_evict_on_provider_change(session_key, scope, provider)

        # Cache-TTL expiry detection. A stable-hash call with
        # zero cache_read after a long idle gap is the signature of
        # provider-side cache eviction (Anthropic's default 5m TTL).
        # Carry-all did everything right — the TTL just lapsed in
        # between turns. Flagging this distinctly stops the analyzer
        # from misattributing it as a tool-list mutation.
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
            "provider": provider,
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
            # Session posture as detected for this call's bucket (pending/on/
            # off) — unchanged semantics; the savings code classifies on it.
            "cache_mode": cache_mode,
            # Provider verdict: True/False once the scope|provider bucket has
            # locked "on"/"off", None while pending. Additive contract.
            "provider_caches": provider_caches,
            "cache_ttl_expired": ttl_expired,
        }
        logger_io.log_api_call(row)
    except Exception as exc:
        logger.debug("tool-belt: post_api_request log failed: %s", exc)
    return None


# ─── Hook: on_session_end ──────────────────────────────────────────────────

def _on_session_end(session_id=None, **kwargs) -> None:
    """WARNING: this hook fires PER TURN, not at actual session end.

    ``on_session_end`` is a misleading hook name — Hermes fires it at the
    end of every ``run_conversation`` call, which means once per user
    message in multi-turn sessions. Hermes' own conversation loop calls
    this out explicitly in a source comment.

    NO per-session state is evicted here — deliberately. Sticky residency
    and lookback history are SESSION-scoped features that must survive
    across turns: sticky decays on its own per-dispatch turn budget
    (``_decay_sticky``) and the lookback ring self-trims, so neither
    grows unbounded. All per-session memory (posture pin, cache-mode,
    sticky, lookback) is cleared by :func:`_on_session_reset` (true /reset
    or /new). An earlier revision evicted sticky/lookback here keyed on the
    ``HERMES_SESSION_KEY`` env var; the deployed gateway migrated session
    identity to ContextVars and never sets that var, so the eviction was
    a live no-op — and anywhere the var DID exist it wiped sticky state
    at the end of the turn that created it, defeating the feature.

    What this hook does: clear the turn's prediction contextvar, then run
    the debounced auto-shape pass.
    """
    if not _CONFIG["enabled"]:
        return None
    try:
        _PREDICTION_CV.set(None)
    except Exception:
        pass

    # The sanctioned between-session moment
    # for the in-process auto-shape engine. Fail-open by construction — see
    # _maybe_auto_shape.
    _maybe_auto_shape()
    return None


def _maybe_auto_shape() -> None:
    """Periodic in-process auto-shaping trigger (session-end path only).

    Applies shaping automatically for scopes whose ``learned_mode`` resolves
    to ``apply`` — the operator's standing consent — without any system-level
    scheduler. Everything heavy lives in ``shaping.auto_shape_run``; this
    wrapper only does the cheap gating:

      · single timestamp compare (``_AUTO_SHAPE["not_before"]``) before any
        file I/O or import;
      · global opt-out ``auto_shape: false`` (default true — automatic is
        the product promise);
      · non-blocking in-process lock so two session-ends can't run the
        engine twice (a second gateway process is covered by learned.py's
        atomic whole-document write — no cross-process locking here);
      · lazy import of the shaping module, paid at session-end time, never
        at plugin load.

    FAIL-OPEN ABSOLUTELY: any exception is logged as a warning and never
    propagates into the gateway. The engine only writes learned.json — live
    sessions keep their in-flight carrying untouched; the new assignment
    applies to future cache-off sessions when their preset resolves at
    dispatch (carry-all sessions ship the whole ceiling regardless).
    """
    try:
        if not _CONFIG["enabled"]:
            return
        if _CONFIG.get("auto_shape", True) is False or str(
            _CONFIG.get("auto_shape", True)
        ).strip().lower() in ("false", "0", "no", "off"):
            return
        now = time.time()
        if now < float(_AUTO_SHAPE["not_before"]):
            return  # cheap guard: one float compare, no I/O
        if not _AUTO_SHAPE["lock"].acquire(blocking=False):
            return  # another session-end is already running the engine
        try:
            _AUTO_SHAPE["not_before"] = now + _AUTO_SHAPE_RECHECK_SECONDS
            from . import shaping as shaping_mod  # lazy: session-end time only

            shaping_mod.auto_shape_run(_CONFIG)
        finally:
            _AUTO_SHAPE["lock"].release()
    except Exception as exc:
        logger.warning("tool-belt: auto-shape pass failed (fail-open): %s", exc)


# ─── Hook: on_session_reset ────────────────────────────────────────────────

def _on_session_reset(session_id=None, **kwargs) -> None:
    """Discard cache-aware per-session state on /reset.

    `/reset` keeps the same canonical session_key but wipes the agent's
    message history — which, from the provider's cache perspective, is
    identical to a fresh session. The posture pin and the mode detection
    state from the prior conversation must be discarded so the next
    ``pre_gateway_dispatch`` re-resolves against the new (empty) history.

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

    # Hermes' on_session_reset hook passes only the NEW session UUID
    # (post-rotation) and the platform string. The pin is keyed by
    # the canonical session_key (agent:main:<platform>:...) which the
    # hook does NOT pass. Recover it from the platform back-reference
    # we populate in _on_pre_gateway_dispatch — that hook fires for
    # /new BEFORE the command router routes it, so the most-recent
    # canonical key for the platform is the one being reset.
    platform = kwargs.get("platform") or ""
    if platform:
        last_canonical = _LAST_CANONICAL_BY_PLATFORM.get(str(platform), "")
        if last_canonical and last_canonical not in eviction_ids:
            eviction_ids.append(last_canonical)

    for sid in eviction_ids:
        _evict_session_cache_state(sid)
        sticky_key = _sticky_key_for_session(sid)
        if sticky_key:
            _STICKY_BY_KEY.pop(sticky_key, None)
        _PRIOR_MESSAGES_BY_SESSION.pop(sid, None)
    return None


# ─── Plugin entry point ────────────────────────────────────────────────────

def register(ctx) -> None:
    _load_user_config()

    # One-shot visibility for the config-disabled state: registration still
    # happens (Plugin Doctor needs it), but every hook guards on
    # ``_CONFIG["enabled"]`` — a config-disabled plugin exposes expand_tools
    # while narrowing nothing. That state should never be silent.
    if not _CONFIG.get("enabled", True):
        logger.warning(
            "tool-belt: registered but disabled by config "
            "(plugins.entries.tool-belt.settings.enabled is false) — hooks "
            "inert, no narrowing, "
            "no telemetry; expand_tools stays registered but idle"
        )

    # NOTE: registration is unconditional. Hermes' plugin manager already
    # gates load/no-load via ``plugins.enabled`` in config.yaml, so an
    # internal ``enabled`` check here would be redundant for load control —
    # and it blocked Plugin Doctor from validating that the declared tool
    # and hooks actually register. The plugin stays functionally disable-able
    # via config: each hook and the narrowing patch guard on
    # ``_CONFIG["enabled"]`` internally, so with ``enabled: False`` the
    # registrations exist but do nothing.

    # Pre-load cross-session cache-mode decisions so auto mode need not
    # repeat the observation window for every session.
    _load_detection_cache()

    # Try eager patch install. If it fails (typically a transient circular
    # import during Hermes startup), defer to first dispatch — by then
    # run_agent is fully loaded and the import will succeed.
    if not _install_patches():
        logger.info(
            "tool-belt: eager patch install deferred (will retry on first dispatch)"
        )

    # Register the expand_tools meta-tool
    try:
        ctx.register_tool(
            name="expand_tools",
            toolset="tool-belt",
            schema=expand_tools_mod.SCHEMA,
            handler=expand_tools_mod.make_handler(_PREDICTION_CV, sticky_refresh_fn=_refresh_sticky),
            description=expand_tools_mod.SCHEMA["description"],
        )
    except Exception as exc:
        logger.warning("tool-belt: expand_tools registration failed: %s", exc)

    # Hooks
    ctx.register_hook("pre_gateway_dispatch", _on_pre_gateway_dispatch)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_hook("on_session_reset", _on_session_reset)

    # `hermes tool-belt ...` terminal subcommand. Optional and fail-open: a
    # Hermes without register_cli_command (or a registration that raises) must
    # never cost the tool/hooks above — the bare `tool-belt` launcher on PATH
    # remains the guaranteed entry point either way. Imported lazily so plugin
    # load doesn't pay for a module only the CLI path uses.
    try:
        if hasattr(ctx, "register_cli_command"):
            from . import cli as cli_mod

            ctx.register_cli_command(
                name="tool-belt",
                help="Tool Belt savings report and onboarding",
                setup_fn=cli_mod.register_cli,
                handler_fn=cli_mod.tool_belt_command,
                description=(
                    "Hermes Tool Belt operator CLI. `savings` prints the "
                    "read-only token-savings report; `configure` runs "
                    "onboarding. Flags are passed through unchanged to the "
                    "same CLI the `tool-belt` launcher runs — see "
                    "`hermes tool-belt savings --help` / "
                    "`hermes tool-belt configure --help`."
                ),
            )
    except Exception as exc:
        logger.warning("tool-belt: CLI command registration failed: %s", exc)

    logger.info(
        "tool-belt: active (policy=policy.yaml, log=%s, learned_mode=%s, bypass_rate=%s)",
        _CONFIG.get("log"), _CONFIG.get("learned_mode"), _CONFIG.get("bypass_rate"),
    )
