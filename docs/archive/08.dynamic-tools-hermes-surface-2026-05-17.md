# dynamic-tools — Hermes Patch Surface

Reference for what the plugin touches in Hermes' code. Read this **before any Hermes upstream upgrade** — if any of these patch points have moved or changed shape, the plugin needs adjustment (or will silently no-op).

## TL;DR

- **Hermes source files**: zero modified on disk. The plugin lives entirely under `~/.hermes/plugins/dynamic-tools/`.
- **Runtime patches applied at gateway startup**: **one** method on **one** class.
- **Hermes internals the plugin reads (not patches)**: a few. Listed below.
- **Failure mode if upstream changes break the patch**: plugin disables itself with a warning. Sessions continue working with the full unfiltered tool set. Nothing breaks; we just lose savings until the plugin is updated.

## The one runtime patch

```
agent.run_agent.AIAgent._build_api_kwargs    →  wrapped by plugin
```

Source location of the patch: [`plugins/dynamic-tools/__init__.py`](../plugins/dynamic-tools/__init__.py) `_install_patches()`.

What the wrapper does: calls original, reads the prediction context from a `ContextVar`, filters `kwargs["tools"]` against the prediction's allowed-set, and returns the modified kwargs. Falls back to original kwargs unchanged on any error.

## Hermes internals the plugin reads (no patches, but read at runtime)

The plugin doesn't modify these — but it depends on their shape and behavior. If any of them change in an upstream update, the plugin may silently no-op or behave incorrectly. None will crash Hermes.

| What | Where | What we depend on |
|---|---|---|
| `AIAgent._build_api_kwargs` | [run_agent.py:8126](../hermes-agent/run_agent.py#L8126) | Method exists on the class. Returns a dict with `kwargs["tools"]` containing the per-call tool list. |
| `kwargs["tools"]` shape (Anthropic) | [agent/transports/anthropic.py:35](../hermes-agent/agent/transports/anthropic.py#L35) | List of dicts with top-level `name`. |
| `kwargs["tools"]` shape (Codex Responses) | [agent/codex_responses_adapter.py:205](../hermes-agent/agent/codex_responses_adapter.py#L205) | List of dicts with top-level `name`. |
| `mcp_` prefix on Anthropic tools | [agent/anthropic_adapter.py:1774-1778](../hermes-agent/agent/anthropic_adapter.py#L1774-L1778) | Tool names get prefixed with `_MCP_TOOL_PREFIX = "mcp_"` when sent via Claude Code OAuth. Plugin strips this prefix before matching. |
| `pre_gateway_dispatch` hook firing | [gateway/run.py:3604-3613](../hermes-agent/gateway/run.py#L3604) | Hook fires for inbound user messages, kwargs `event=, gateway=, session_store=`. |
| `post_tool_call` hook | [hermes_cli/plugins.py:62](../hermes-agent/hermes_cli/plugins.py#L62) | Standard Hermes plugin hook. |
| `on_session_end` hook | [hermes_cli/plugins.py:71](../hermes-agent/hermes_cli/plugins.py#L71) | Standard Hermes plugin hook. |
| `ctx.register_tool` | [hermes_cli/plugins.py:220](../hermes-agent/hermes_cli/plugins.py#L220) | Registers `expand_tools` meta-tool. Standard plugin API. |
| `hermes_cli.config.load_config / cfg_get` | [hermes_cli/config.py:3491, 3574](../hermes-agent/hermes_cli/config.py#L3491) | Read plugin config from `~/.hermes/config.yaml`. |
| `toolsets.resolve_toolset(name)` | [toolsets.py](../hermes-agent/toolsets.py) | Resolve a category/toolset name (e.g. `"browser"`) to its tool name list. Used by `expand_tools`. |
| `_run_in_executor_with_context` propagates `ContextVar` | [gateway/run.py](../hermes-agent/gateway/run.py) (`copy_context()` + `ctx.run`) | Required for the prediction `ContextVar` to reach the agent's executor thread. |
| `session_store._generate_session_key(event.source)` | [gateway/session.py](../hermes-agent/gateway/session.py) `SessionStore` | Resolve the canonical Hermes session key for telemetry. Added 2026-05-17 (audit fix #3) so `predictions.jsonl` rows carry the same session id Hermes uses for routing, instead of the AIAgent uuid. Plugin gracefully falls back to `HERMES_SESSION_KEY` env var, then `kwargs.session_id`, then `event.session_id`. Failure mode: `session_id` field is blank, which disables the bypass cohort + session-windowed FP classification but doesn't break narrowing. |
| Canonical session-key shape `agent:main:{platform}:{chat_type}[:...]` | [gateway/session.py:build_session_key](../hermes-agent/gateway/session.py) | Position 1 is the literal string `"main"`, NOT the per-profile agent name. The plugin's `_parse_session_key_scope` validates this and extracts platform from position 2. If Hermes changes the key shape, scope attribution falls back to blank rather than synthesizing `main:<platform>`. |

## What an upstream update could break

In rough order of "likeliest to bite us":

1. **`AIAgent._build_api_kwargs` renamed or signature changed.** Patch fails to install. Plugin logs `dynamic-tools: AIAgent._build_api_kwargs not found` and disables itself. Behavior: full tool set sent, no narrowing. **Easy to detect from logs.**

2. **`mcp_` prefix changes** (different constant, different prefix, prefix moved to a different code path). Plugin's prefix-strip becomes wrong. Behavior: filter loses some narrowing, depending on direction of change. **Detectable via the diagnostic log line that prints `sample_names=...`** (compare to preset).

3. **`kwargs["tools"]` key changes** (e.g., Hermes moves to `kwargs["functions"]` or similar for some provider). Filter sees no tools, no-ops. **Detectable**: prediction rows show `ceiling=0`.

4. **`pre_gateway_dispatch` kwargs change.** Hook still fires but with different shape. Plugin's `_message_text_from_event` returns empty, predictor falls back to wildcard, no narrowing. **Detectable**: prediction rows show all triggers empty and `narrowed == ceiling`.

5. **`toolsets.resolve_toolset` removed/renamed.** `expand_tools` returns errors. Narrowing still works (preset controls always-on/triggers). **Detectable**: `expand_tools` log warnings.

6. **`pre_gateway_dispatch` hook itself removed** (less likely, but possible if Hermes restructures plugin API). Plugin loads but never narrows. **Detectable**: no prediction rows written despite messages being processed.

## How to verify the patch is healthy after a Hermes upgrade

Run after `hermes upgrade` (or `pip install -U` of the agent):

```bash
# 1. Restart and watch for the plugin's startup line
hermes gateway restart
sleep 6
grep "dynamic-tools" ~/.hermes/logs/agent.log | tail -3
# Expected:
#   dynamic-tools: patches installed (AIAgent._build_api_kwargs)
#   dynamic-tools: active (mode=aggressive, log=True)

# 2. Send a test message that should NOT trigger anything
#    (e.g. "hi" to Bernard on Telegram)

# 3. Verify a prediction row landed AND filter actually narrowed
tail -1 ~/.hermes/state/dynamic-tools/predictions.jsonl | jq
# Expected: narrowed_count < ceiling_count, tokens_saved > 0

# 4. Spot-check the diagnostic log
grep "filter input tools" ~/.hermes/logs/agent.log | tail -1
# Expected: "kept=N cut=M" with both N and M > 0
```

If any of those checks fail, the plugin silently broke during the upgrade. The recovery path:

```bash
# Disable while you investigate
hermes plugins disable dynamic-tools
hermes gateway restart
# Bernard / Sue keep working with full toolsets

# Investigate which patch point moved by re-reading the table above
# and grep'ing for renamed symbols in the upgraded Hermes source
```

## Re-applying after upstream changes

If Hermes upstream renames `_build_api_kwargs` (the most likely break), the fix is one line in [`plugins/dynamic-tools/__init__.py`](../plugins/dynamic-tools/__init__.py) `_install_patches()`:

```python
# Change this:
AIAgent._build_api_kwargs = _wrap_build_api_kwargs(_ORIGINAL_BUILD_API_KWARGS)

# To match the new method name:
AIAgent._new_method_name = _wrap_build_api_kwargs(_ORIGINAL_BUILD_API_KWARGS)
```

(Also update the `getattr(AIAgent, "_build_api_kwargs", None)` check above it.)

Most other failure modes (kwargs shape, prefix changes, hook signature) require corresponding small adjustments in `_wrap_build_api_kwargs` or the helper functions — all in `__init__.py`. None require deep rearchitecture; each affected line in this doc points at exactly the file:line that would need to change.

## Roll-back plan

The plugin is fully reversible without touching Hermes' source:

```bash
# Quick disable — keeps the install but disables on next restart
hermes plugins disable dynamic-tools  # OR set plugins.dynamic-tools.enabled: false in config.yaml
hermes gateway restart

# Full removal
rm -rf ~/.hermes/plugins/dynamic-tools
rm -rf ~/.hermes/state/dynamic-tools           # optional — drops telemetry
hermes gateway restart
```

The `AIAgent._build_api_kwargs` monkey-patch only exists in the running gateway process's memory. A restart wipes it cleanly.
