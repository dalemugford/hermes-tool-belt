"""The ``expand_tools`` meta-tool — model's recourse when a category was
gated out of the current turn.

NEW DESIGN (post-cached-agent fix):
  expand_tools no longer mutates ``agent.tools`` directly. Instead it
  appends a category to the prediction state's ``expansions`` set
  (stored in a ContextVar, mutable in place). The patched
  ``_build_api_kwargs`` reads this set on the next API call and unions
  the category's tools into the allowed set.

  This works regardless of whether the AIAgent was freshly constructed
  or pulled from the gateway's per-session cache — narrowing happens at
  request-build time, not at agent construction time.
"""

from __future__ import annotations

import difflib
import json
import logging

logger = logging.getLogger(__name__)


SCHEMA = {
    "name": "expand_tools",
    "description": (
        "Load a category of tools that wasn't included in this turn's tool list. "
        "Use this when you need a tool that isn't available — for example: "
        "'browser' for web automation, 'image_gen' for drawing, 'cronjob' for "
        "scheduling, 'tts' for text-to-speech, 'delegation' for spawning subagents, "
        "'code_execution' for batch processing scripts. After calling, the new "
        "tools are available on the very next tool call and may remain available "
        "briefly during the same active task. Expanded tools are ephemeral; call "
        "this again if a needed tool is absent later. Cheap and safe to call "
        "preemptively if you're unsure whether a tool is loaded. "
        "Accepts a 'category' (toolset name) or a 'tool' (specific tool name). "
        "If you know the tool name but not its category, pass it as 'tool' and "
        "the handler will find the right category for you."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": (
                    "The toolset name to load. Common values: browser, image_gen, "
                    "tts, vision, cronjob, delegation, code_execution, terminal, "
                    "file, web, skills, homeassistant. Any toolset name from "
                    "Hermes' toolset table is accepted."
                ),
            },
            "tool": {
                "type": "string",
                "description": (
                    "A specific tool name to load (e.g. 'mnemosyne_diagnose', 'browser_exec'). "
                    "If you know the tool name but not its category, use this instead of 'category'. "
                    "The handler will resolve it to the correct toolset automatically."
                ),
            },
        },
        "required": [],
    },
}


def _available_toolset_names() -> list[str]:
    """Return the live list of toolset names, or an empty list on failure."""
    try:
        from toolsets import get_toolset_names  # type: ignore[import-not-found]
        return list(get_toolset_names() or [])
    except Exception:
        return []


def _resolve_category(category: str) -> tuple[str, list[str], list[str]]:
    """Resolve a user-supplied category to (canonical_name, tools, available).

    Forgiving of surrounding whitespace and case differences against the
    live toolset table. Returns ``(canonical, [], available)`` when the
    category cannot be resolved — callers use ``available`` to render a
    helpful error.
    """
    available = _available_toolset_names()
    try:
        from toolsets import resolve_toolset  # type: ignore[import-not-found]
    except Exception:
        return category, [], available

    # As-given first — preserves whatever the model passed when valid.
    try:
        resolved = list(resolve_toolset(category) or [])
    except Exception:
        resolved = []
    if resolved:
        return category, resolved, available

    # Case-insensitive lookup against the live table.
    lower = category.lower()
    for name in available:
        if name.lower() == lower:
            try:
                resolved = list(resolve_toolset(name) or [])
            except Exception:
                resolved = []
            if resolved:
                return name, resolved, available
            break

    return category, [], available


def _resolve_tool_to_category(tool_name: str) -> str | None:
    """Reverse-lookup a tool name to its parent toolset category.

    Iterates all toolset names from ``_available_toolset_names()``, calls
    ``resolve_toolset()`` on each, and returns the first category whose
    tool list contains the requested ``tool_name`` (case-insensitive).
    Returns ``None`` if no match.
    """
    available = _available_toolset_names()
    try:
        from toolsets import resolve_toolset  # type: ignore[import-not-found]
    except Exception:
        return None
    lower = tool_name.lower()
    for cat in available:
        try:
            tools = list(resolve_toolset(cat) or [])
        except Exception:
            continue
        if lower in (t.lower() for t in tools):
            return cat
    return None


def _tool_not_found_message(tool_name: str, available: list[str]) -> str:
    """Compose an error for an unresolvable tool name, with fuzzy suggestions.

    Enumerates every (category, tool) pair from the live toolset table and
    offers the closest tool-name matches, each annotated with its category.
    """
    all_tools: list[tuple[str, str]] = []
    try:
        from toolsets import resolve_toolset  # type: ignore[import-not-found]
        for cat in available:
            try:
                all_tools.extend((cat, t) for t in (resolve_toolset(cat) or []))
            except Exception:
                continue
    except Exception:
        pass

    suggestions = difflib.get_close_matches(
        tool_name.lower(), [t.lower() for _, t in all_tools], n=3, cutoff=0.6
    )
    parts = [f"tool {tool_name!r} not found in any toolset."]
    if suggestions:
        # Map suggestion-lowercase back to canonical tool names + categories.
        canonical: list[str] = []
        seen: set[str] = set()
        for s in suggestions:
            for cat, t in all_tools:
                if t.lower() == s and t not in seen:
                    canonical.append(f"{t} (category: {cat})")
                    seen.add(t)
                    break
        if canonical:
            parts.append(f"Did you mean: {', '.join(canonical)}?")
    return " ".join(parts)


def _not_found_message(category: str, available: list[str]) -> str:
    """Compose an error message that surfaces dynamic categories + suggestions."""
    suggestions = difflib.get_close_matches(category.lower(), [n.lower() for n in available], n=3, cutoff=0.6)
    parts = [f"toolset {category!r} not found."]
    if suggestions:
        # Map suggestion-lowercase back to the canonical-cased names.
        canonical: list[str] = []
        seen: set[str] = set()
        for s in suggestions:
            for name in available:
                if name.lower() == s and name not in seen:
                    canonical.append(name)
                    seen.add(name)
                    break
        if canonical:
            parts.append(f"Did you mean: {', '.join(canonical)}?")
    if available:
        # Cap the listing — some Hermes installs have 40+ toolsets.
        preview = sorted(available)
        head = preview[:24]
        tail_note = f" (+{len(preview) - len(head)} more)" if len(preview) > len(head) else ""
        parts.append(f"Available categories: {', '.join(head)}{tail_note}.")
    else:
        parts.append("Available categories could not be enumerated.")
    return " ".join(parts)


def make_handler(prediction_cv, sticky_refresh_fn=None):
    """Build the handler closure with access to the prediction ContextVar.

    The handler appends the requested category to the contextvar's
    ``expansions`` set. The patched ``_build_api_kwargs`` (in __init__.py)
    reads this set on every API call to widen the narrowed tool list. If sticky
    residency is enabled, successful expansions also refresh short-lived
    in-memory residency for the active scope.
    """

    def handle(args: dict, **kwargs) -> str:
        try:
            return _handle_inner(args, kwargs, prediction_cv, sticky_refresh_fn)
        except Exception as exc:
            logger.warning("expand_tools handler error: %s", exc)
            return json.dumps({
                "success": False,
                "error": f"expand_tools failed: {type(exc).__name__}",
            })

    return handle


def _handle_inner(args: dict, kwargs: dict, prediction_cv, sticky_refresh_fn=None) -> str:
    raw_category = str(args.get("category") or "").strip()
    raw_tool = str(args.get("tool") or "").strip()

    if not raw_category and not raw_tool:
        return json.dumps({
            "success": False,
            "error": (
                "category or tool is required "
                "(e.g. category='browser' or tool='mnemosyne_diagnose')"
            ),
        })

    # If only a tool name was given, resolve it to its parent category.
    # An explicit category always wins — explicit is better than a guess.
    if not raw_category and raw_tool:
        resolved_category = _resolve_tool_to_category(raw_tool)
        if resolved_category is None:
            return json.dumps({
                "success": False,
                "error": _tool_not_found_message(raw_tool, _available_toolset_names()),
            })
        raw_category = resolved_category

    state = prediction_cv.get()
    if state is None:
        # No prediction context — either we're in a code path that doesn't
        # narrow (CLI, subagent, cron) or pre_gateway_dispatch never fired.
        # Either way the tool the model wants is either already available
        # or unreachable; return a graceful error.
        return json.dumps({
            "success": False,
            "error": (
                "no prediction context for this turn. "
                "All available tools should already be loaded — try the call "
                "you wanted to make."
            ),
        })

    category, resolved, available = _resolve_category(raw_category)

    if not resolved:
        return json.dumps({
            "success": False,
            "error": _not_found_message(raw_category, available),
        })

    initial_allowed = set(state.get("initial_allowed_tools") or [])
    cut_tools = set(state.get("cut_tools") or [])

    # Append to the contextvar's expansions set — mutable in place
    expansions = state.setdefault("expansions", set())
    if not isinstance(expansions, set):
        expansions = set(expansions or [])
        state["expansions"] = expansions

    # Honest accounting: a tool counts as "newly added by this call" only
    # if it wasn't already on the model's initial allowed list and wasn't
    # already carried in from a prior expand_tools invocation. The previous
    # logic only checked the second condition, so re-expanding a category
    # that overlapped the initial allowed set would falsely report the
    # overlap as "added".
    already_available_tools = [t for t in resolved if t in initial_allowed]
    recovered_cut_tools = [t for t in resolved if t in cut_tools]
    new_additions = [
        t for t in resolved
        if t not in initial_allowed and t not in expansions
    ]
    expansions.add(category)  # also store the category name itself
    expansions.update(resolved)

    expand_event = {
        "category": category,
        "resolved_tools": resolved,
        "tools_added": new_additions,
        "recovered_cut_tools": recovered_cut_tools,
        "already_available_tools": already_available_tools,
        "triggers_fired": list(state.get("triggers_fired") or []),
    }
    sticky_event = None
    if sticky_refresh_fn is not None:
        try:
            sticky_event = sticky_refresh_fn(
                str(state.get("sticky_key", "")),
                category,
                resolved,
                policy_scope=str(state.get("scope", "")),
            )
            expand_event["sticky"] = sticky_event
            if isinstance(sticky_event, dict) and sticky_event.get("refreshed"):
                state["sticky_tools"] = sticky_event.get("sticky_tools", [])
                state["sticky_categories"] = sticky_event.get("sticky_categories", [])
                state["sticky_remaining_turns"] = sticky_event.get("sticky_remaining_turns", {})
        except Exception as exc:
            logger.debug("expand_tools sticky refresh failed: %s", exc)
            sticky_event = {"enabled": False, "error": type(exc).__name__}
            expand_event["sticky"] = sticky_event
    state.setdefault("expansion_events", []).append(expand_event)
    state["pending_expansion"] = expand_event

    logger.info(
        "expand_tools: category=%s added=%d already=%d (expansions size=%d)",
        category, len(new_additions), len(already_available_tools), len(expansions),
    )
    response = {
        "success": True,
        "category": category,
        "resolved_tools": resolved,
        "tools_added": new_additions,
        "recovered_cut_tools": recovered_cut_tools,
        "already_available_tools": already_available_tools,
        "message": _success_message(category, new_additions, already_available_tools),
    }
    if sticky_event is not None:
        response["sticky"] = sticky_event
    return json.dumps(response)


def _success_message(category: str, new_additions: list[str], already_available: list[str]) -> str:
    """Compose a message that matches what actually happened."""
    if not new_additions and already_available:
        return (
            f"Category {category!r} was already loaded "
            f"({len(already_available)} tool(s)); no new tools added. "
            "Make the call you wanted to make."
        )
    if new_additions and already_available:
        return (
            f"Loaded {len(new_additions)} new tool(s) from category {category!r} "
            f"({len(already_available)} were already available). "
            "They're available on the next tool call."
        )
    return (
        f"Loaded {len(new_additions)} tool(s) from category {category!r}. "
        "They're available on the next tool call."
    )
