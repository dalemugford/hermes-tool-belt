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
        "preemptively if you're unsure whether a tool is loaded."
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
            }
        },
        "required": ["category"],
    },
}


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
    category = str(args.get("category") or "").strip()
    if not category:
        return json.dumps({
            "success": False,
            "error": "category is required (e.g. 'browser', 'image_gen', 'cronjob')",
        })

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

    # Resolve the category to actual tool names so we can report what got added
    try:
        from toolsets import resolve_toolset  # type: ignore[import-not-found]
        resolved = list(resolve_toolset(category))
    except Exception:
        resolved = []

    if not resolved:
        return json.dumps({
            "success": False,
            "error": (
                f"toolset {category!r} not found. Available categories: "
                "browser, image_gen, tts, vision, cronjob, delegation, "
                "code_execution, terminal, file, web, skills, homeassistant. "
                "(Other plugin-provided toolsets may also exist.)"
            ),
        })

    initial_allowed = set(state.get("initial_allowed_tools") or [])
    cut_tools = set(state.get("cut_tools") or [])

    # Append to the contextvar's expansions set — mutable in place
    expansions = state.setdefault("expansions", set())
    if not isinstance(expansions, set):
        expansions = set(expansions or [])
        state["expansions"] = expansions
    new_additions = [t for t in resolved if t not in expansions]
    recovered_cut_tools = [t for t in resolved if t in cut_tools]
    already_available_tools = [t for t in resolved if t in initial_allowed]
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
            sticky_event = sticky_refresh_fn(str(state.get("scope", "")), category, resolved)
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
        "expand_tools: category=%s added=%d (expansions size=%d)",
        category, len(new_additions), len(expansions),
    )
    response = {
        "success": True,
        "category": category,
        "tools_added": new_additions,
        "recovered_cut_tools": recovered_cut_tools,
        "already_available_tools": already_available_tools,
        "message": (
            f"Loaded {len(new_additions)} tool(s) from category {category!r}. "
            "They're available on the next tool call."
        ),
    }
    if sticky_event is not None:
        response["sticky"] = sticky_event
    return json.dumps(response)
