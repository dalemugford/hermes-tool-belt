"""The ``expand_tools`` meta-tool — model's recourse when a category was
gated out of the current turn.

Appends the requested category to the prediction state's ``expansions`` set (a
ContextVar mutated in place). ``_build_api_kwargs`` unions that set into the
allowed tools on the next API call, so expansion works whether the AIAgent was
freshly constructed or served from the gateway's per-session cache — narrowing
happens at request-build time, not at agent construction time.
"""

from __future__ import annotations

import difflib
import json
import logging
from typing import Any, Iterable

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


# ─── Expand-only discoverability manifest ───────────────────────────────────
#
# Lets the model *discover* which enabled built-in tools live in the
# ``expand_only`` stratum ``X`` without carrying their full schemas. A compact,
# deterministic, name-only manifest is appended to a per-request CLONE of the
# carried ``expand_tools`` description — the registered SCHEMA object is never
# mutated. Grouping uses Hermes' live toolset table; names with no known
# category land in an explicitly labeled ungrouped bucket.

_UNGROUPED_LABEL = "(ungrouped)"

_MANIFEST_HEADER = (
    "Enabled expand-only tools (full schemas omitted this turn to save "
    "context). A trigger may auto-activate one when your message needs it; "
    "otherwise call expand_tools(tool=NAME) — or expand_tools(category=NAME) — "
    "to load it explicitly. Grouped by toolset:"
)


def _build_category_index() -> dict[str, set[str]]:
    """Return ``{category: {tool_name, ...}}`` from the live toolset table.

    Best-effort: an unresolvable toolset is skipped, and a missing ``toolsets``
    module yields an empty index (every name then falls to the ungrouped bucket).
    """
    index: dict[str, set[str]] = {}
    available = _available_toolset_names()
    try:
        from toolsets import resolve_toolset  # type: ignore[import-not-found]
    except Exception:
        return index
    for cat in available:
        try:
            tools = resolve_toolset(cat) or []
        except Exception:
            continue
        members = {str(t) for t in tools if t}
        if members:
            index[str(cat)] = members
    return index


def build_expand_only_manifest(
    expand_only_names: Iterable[Any] | None,
    *,
    category_index: dict[str, set[str]] | None = None,
) -> str:
    """Compose the compact, deterministic expand-only manifest text.

    ``expand_only_names`` is the effective stratum ``X = E − (always_carry ∪
    carry)`` — already built-in-only (MCP/plugin pass-through names are excluded
    upstream). Names are grouped by toolset category (a name with no known
    category is placed under an explicitly labeled ungrouped bucket, sorted
    last). Groups and names sort deterministically, so the manifest is stable
    across differently ordered source schema lists. Returns ``""`` when there
    are no expand-only names.

    ``category_index`` (``{category: {tool_name}}``) may be injected for tests;
    when ``None`` it is resolved from the live toolset table.
    """
    names = sorted({str(n) for n in (expand_only_names or []) if n})
    if not names:
        return ""

    if category_index is None:
        category_index = _build_category_index()

    # Reverse map tool -> category. Iterate categories in sorted order so a
    # tool that (pathologically) belongs to more than one toolset resolves to
    # a single, deterministic category.
    tool_to_cat: dict[str, str] = {}
    for cat in sorted(category_index):
        for tool in category_index[cat]:
            tool_to_cat.setdefault(str(tool), str(cat))

    groups: dict[str, list[str]] = {}
    for name in names:
        cat = tool_to_cat.get(name, _UNGROUPED_LABEL)
        groups.setdefault(cat, []).append(name)

    # Known categories alphabetical; the ungrouped bucket always sorts last.
    def _group_key(cat: str) -> tuple[int, str]:
        return (1, "") if cat == _UNGROUPED_LABEL else (0, cat)

    lines = [_MANIFEST_HEADER]
    for cat in sorted(groups, key=_group_key):
        members = ", ".join(sorted(groups[cat]))
        lines.append(f"{cat}: {members}")
    return "\n".join(lines)


def augment_schema_with_manifest(schema: Any, manifest_text: str) -> Any:
    """Return a per-request CLONE of ``schema`` with ``manifest_text`` appended
    to its description. The input schema object is never mutated.

    Handles both the Anthropic tool shape (top-level ``description``) and the
    OpenAI shape (``function.description``). When ``manifest_text`` is empty the
    original object is returned unchanged.
    """
    if not manifest_text or not isinstance(schema, dict):
        return schema
    clone = dict(schema)
    fn = clone.get("function")
    if isinstance(fn, dict):
        fn_clone = dict(fn)
        base = str(fn_clone.get("description") or "")
        fn_clone["description"] = (base + "\n\n" + manifest_text).strip()
        clone["function"] = fn_clone
    else:
        base = str(clone.get("description") or "")
        clone["description"] = (base + "\n\n" + manifest_text).strip()
    return clone


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


def make_handler(prediction_cv: Any, sticky_refresh_fn: Any = None):
    """Build the handler closure with access to the prediction ContextVar.

    The handler appends the requested category to the contextvar's
    ``expansions`` set. The patched ``_build_api_kwargs`` (in __init__.py)
    reads this set on every API call to widen the narrowed tool list. If sticky
    residency is enabled, successful expansions also refresh short-lived
    in-memory residency for the active scope.
    """

    def handle(args: dict, **_kwargs) -> str:
        try:
            return _handle_inner(args, prediction_cv, sticky_refresh_fn)
        except Exception as exc:
            logger.warning("expand_tools handler error: %s", exc)
            return json.dumps({
                "success": False,
                "error": f"expand_tools failed: {type(exc).__name__}",
            })

    return handle


def _handle_inner(args: dict, prediction_cv: Any, sticky_refresh_fn: Any = None) -> str:
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

    initial_active = set(state.get("initial_active_tools") or [])
    expand_only = set(state.get("expand_only_tools") or [])

    # Ceiling gate: the enabled built-in ceiling ``E`` is captured on the first
    # request of the turn (state["enabled_ceiling"]). A tool that resolves
    # globally but is absent from ``E`` is not enabled for THIS scope — it can
    # never be activated here, so it is reported "unavailable", never "added".
    # When the ceiling hasn't been captured (e.g. no prediction/narrowing path
    # ran), we can't gate — treat every resolved tool as activatable.
    enabled_ceiling = state.get("enabled_ceiling")
    if enabled_ceiling is None:
        activatable = list(resolved)
        unavailable_tools: list[str] = []
    else:
        ceiling_set = set(enabled_ceiling)
        activatable = [t for t in resolved if t in ceiling_set]
        unavailable_tools = [t for t in resolved if t not in ceiling_set]

    # Append to the contextvar's expansions set — mutable in place. Only
    # ceiling-present (activatable) tools are added; a ceiling-absent tool
    # must not leak into the active set on the next request.
    expansions = state.setdefault("expansions", set())
    if not isinstance(expansions, set):
        expansions = set(expansions or [])
        state["expansions"] = expansions

    # Honest accounting: a tool counts as "newly added by this call" only
    # if it's activatable here, wasn't already on the model's initial active
    # list, and wasn't already carried in from a prior expand_tools call.
    already_available_tools = [t for t in activatable if t in initial_active]
    recovered_cut_tools = [t for t in activatable if t in expand_only]
    new_additions = [
        t for t in activatable
        if t not in initial_active and t not in expansions
    ]
    expansions.add(category)  # also store the category name itself
    expansions.update(activatable)

    expand_event = {
        "category": category,
        "resolved_tools": resolved,
        "tools_added": new_additions,
        "recovered_cut_tools": recovered_cut_tools,
        "already_available_tools": already_available_tools,
        "unavailable_tools": unavailable_tools,
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
        "expand_tools: category=%s added=%d already=%d unavailable=%d (expansions size=%d)",
        category, len(new_additions), len(already_available_tools),
        len(unavailable_tools), len(expansions),
    )
    response = {
        "success": True,
        "category": category,
        "resolved_tools": resolved,
        "tools_added": new_additions,
        "recovered_cut_tools": recovered_cut_tools,
        "already_available_tools": already_available_tools,
        "unavailable_tools": unavailable_tools,
        "message": _success_message(
            category, new_additions, already_available_tools, unavailable_tools
        ),
    }
    if sticky_event is not None:
        response["sticky"] = sticky_event
    return json.dumps(response)


def _success_message(
    category: str,
    new_additions: list[str],
    already_available: list[str],
    unavailable: list[str] | None = None,
) -> str:
    """Compose a message that matches what actually happened."""
    unavailable = unavailable or []
    unavailable_note = ""
    if unavailable:
        unavailable_note = (
            f" {len(unavailable)} tool(s) in {category!r} are not enabled for "
            "this scope and could not be loaded."
        )
    if not new_additions and already_available:
        return (
            f"Category {category!r} was already loaded "
            f"({len(already_available)} tool(s)); no new tools added.{unavailable_note} "
            "Make the call you wanted to make."
        )
    if not new_additions and unavailable:
        return (
            f"No tools from category {category!r} could be loaded — "
            f"{len(unavailable)} resolved tool(s) are not enabled for this scope."
        )
    if new_additions and already_available:
        return (
            f"Loaded {len(new_additions)} new tool(s) from category {category!r} "
            f"({len(already_available)} were already available).{unavailable_note} "
            "They're available on the next tool call."
        )
    return (
        f"Loaded {len(new_additions)} tool(s) from category {category!r}.{unavailable_note} "
        "They're available on the next tool call."
    )
