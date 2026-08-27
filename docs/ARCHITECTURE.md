# tool-belt Architecture

`tool-belt` is a Hermes plugin that reduces per-request tool overhead by
sending the model only the tools likely to matter, while leaving the user's
`platform_toolsets` ceiling intact. This document describes how the plugin
makes that decision, where it intercepts the request lifecycle, and how its
behavior changes depending on whether the active provider supports prefix
caching.

The code is the source of truth. Where a file is referenced inline, that's
where the named mechanism actually lives.

## The problem

Every API call ships the full tool block — names, descriptions, JSON
schemas — to the model. For a profile with a handful of plugins, that's
8–15k tokens of overhead per message, regardless of whether the answer is
`"yes"` or `"run the build and check the logs"`.

The naive fix is to narrow the tool block per turn based on intent. That
fix conflicts with how modern providers bill multi-turn conversations.
Anthropic and OpenAI (auto-cache) hash the prefix of each request —
roughly `system + tools + history` — and bill the cached span at a steep
discount. Mutating the tool block between turns invalidates the cache
boundary that sits in the middle of the prefix. The provider then
re-bills the conversation history at the full input rate. On a multi-turn
session, that loss exceeds the schema-token savings from narrowing.

`tool-belt` resolves this by separating the *goal* from the *cadence*:

- **Goal**: usage-aware tool loading at per-tool granularity. Tools the
  model needs are loaded; tools it doesn't are dropped.
- **Cadence**: when the adjustment happens. Per turn under providers
  without prefix caching; per session (with model-driven mid-session
  expansion) under providers with prefix caching.

The plugin auto-detects which cadence the active provider supports and
operates accordingly.

## Two modes

The plugin runs in one of two modes per session.

### Cache-on mode (default)

Used for providers with active prefix caching (Anthropic, OpenAI
auto-cache).

1. On the **first dispatch** in a session, the predictor
   ([predictor.py](../predictor.py)) runs against the resolved policy
   ([policy.yaml](../policy.yaml), layered with per-scope overrides and
   any active learned overlay) and produces a tool set.
2. That set is **frozen** into a per-session snapshot keyed by canonical
   session key.
3. Every subsequent dispatch in the same session **reuses the snapshot
   verbatim**. The predictor does not run. Sticky residency and lookback
   are skipped — both were per-turn-adjustment mechanisms made redundant
   by the freeze.
4. The model can call `expand_tools(category=...)` to admit a tool group
   that wasn't in the frozen set. The new tools are appended to the
   snapshot in place and persist for the rest of the session. This pays
   one cache break in exchange for permanent access.
5. The snapshot is evicted only at moments that already cost a cache
   break: true `/reset` or `/new`, true end of session, and compaction
   (see [The freeze mechanism](#the-freeze-mechanism) below).

### Cache-off mode

Used for providers without effective prefix caching.

1. On **every dispatch**, the predictor runs against the resolved policy.
   It produces an allowed set from the union of `always_on` and any
   triggered groups.
2. Recently expanded categories carry forward via **sticky residency** —
   tools admitted by a prior turn's `expand_tools` stay loaded for
   `ttl_turns` further turns without needing re-expansion.
3. The predictor optionally concatenates the **last N user messages**
   (lookback) before running its regex classifier, so short replies like
   `"yes"` or `"go ahead"` inherit intent from the prior turn.
4. The on-the-wire tool list is the intersection of
   `(always_on ∪ triggered ∪ expanded ∪ sticky)` with the user's
   ceiling.

Both modes share the safety properties: the ceiling is never widened
beyond what the user's `platform_toolsets` already allowed, and any
exception anywhere falls through to no narrowing.

## Mode detection

The mode is decided per-session, never switched mid-session (since the
switch itself would bust the cache).

Configuration:

```yaml
plugins:
  tool-belt:
    cache_mode: auto    # auto | on | off
```

- `cache_mode: on` and `cache_mode: off` are honored verbatim.
- `cache_mode: auto` (default) consults a cross-session detection cache.
  If a prior session of this scope locked to a definite mode, that mode
  is reused immediately. Otherwise, the plugin defaults to `on` and
  starts an observation window.

The auto path observes `cache_read_tokens` across early API calls. Once
enough calls and enough cumulative input have been observed, the
hit-rate ratio decides:

```
hit_rate = cache_read / (cache_read + cache_write + input)
hit_rate ≥ on_threshold  → lock "on"
hit_rate <  on_threshold  → lock "off"
```

Tunable under `cache_auto`:

```yaml
plugins:
  tool-belt:
    cache_auto:
      detect_calls: 5              # min calls before locking
      detect_min_input_tokens: 5000 # min cumulative input before locking
      on_threshold: 0.40
      providers_off_models:        # known-uncached models, lock immediately
        - kimi-k2.6:cloud
        - gpt-5.4-mini
```

Locked decisions persist per-scope (`{agent}:{platform}`) to
`~/.hermes/state/tool-belt/cache_mode_detection.json`. Subsequent
sessions of the same scope skip the observation window.

The detection code is `_update_cache_mode_detection` in
[__init__.py](../__init__.py); the per-dispatch lookup is
`_resolve_cache_mode_for_session`.

## The freeze mechanism

Under cache-on mode, the per-session snapshot is stored in the module-level
`_FROZEN_BY_SESSION` dict, keyed by the canonical Hermes session key
(`agent:main:{platform}:...`). The snapshot is built by
`_freeze_session_snapshot` on first dispatch and consumed by
`_build_state_from_frozen` on subsequent dispatches.

### Eviction

The plugin registers two distinct hooks:

- `on_session_end` — fires after every `run_conversation` call (i.e.
  once per turn in multi-turn sessions; the hook's name in Hermes is
  misleading). The handler **does not evict** the freeze. It only
  resets the per-turn contextvar and clears sticky / lookback state.
- `on_session_reset` — fires on a true `/reset` or `/new`. The handler
  **evicts** the freeze, the cache-mode detection state, and all
  per-session sticky / lookback state.

The freeze is also evicted on compaction. The plugin wraps
`AIAgent._compress_context` so that after the compressor rewrites
conversation history (which invalidates the provider's cached prefix
regardless of plugin behavior), the snapshot is dropped. The next
dispatch then re-freezes against the post-compaction state. This wrapper
is installed by `_wrap_compress_context`.

### Divergence detection

A session locked `on` whose post-lock cache hit rate stays low is
evidence that something upstream of the tool block is busting cache —
typically a system-prompt mutation (e.g. memory injection), a model
swap, or an unmodeled tool mutation. The plugin tracks consecutive
post-lock calls with hit rate below 30%; after 3 in a row it logs a
divergence warning. The session continues frozen (mid-session mode
switches are not supported by construction), but the analyzer can flag
the session for review.

### Hash fingerprints

Each API call records:

- `tool_list_hash` — truncated sha256 of the tool list in send order.
  Captures membership, ordering, and schema-byte drift. Provider caches
  fingerprint the actual schema bytes, so a description change
  invalidates the cache even when names are stable.
- `system_hash` — truncated sha256 of the first message when its role
  is `system` or `developer`. Lets analysis distinguish "the freeze
  held but the system prompt changed" from "the freeze itself
  changed."

Both hashes are written to `api_calls.jsonl` alongside `cache_read_tokens`
and `cache_write_tokens`, which is how cache economics get measured
directly rather than inferred.

## expand_tools

[expand_tools.py](../expand_tools.py) registers a meta-tool the model can
call when it wants a tool group that wasn't loaded. The schema is:

```json
{
  "name": "expand_tools",
  "parameters": {
    "type": "object",
    "properties": {
      "category": {
        "type": "string",
        "description": "browser, image_gen, tts, vision, cronjob, delegation, code_execution, terminal, file, web, skills, homeassistant, ..."
      }
    },
    "required": ["category"]
  }
}
```

The handler resolves the category to its tool names via Hermes' toolset
table and appends them to the prediction context's `expansions` set.
The patched `_build_api_kwargs` reads that set on the next API call and
unions the resolved tools into the allowed list.

Under cache-on mode, successful expansions are also written back to the
session's frozen snapshot, so the expanded tools persist for the rest
of the session without further round-trips.

Under cache-off mode, successful expansions refresh sticky residency for
the category — the tools stay loaded for `ttl_turns` further turns and
then decay.

Every `expand_tools` event is logged as a learning signal. The
between-session shaper reads these to decide which tools to promote
into future sessions' starting ceilings.

## Between-session shaping

[scripts/shape-ceiling.py](../scripts/shape-ceiling.py) reads
`predictions.jsonl` and `tool_calls.jsonl`, groups by scope and session,
and produces per-scope promote / demote recommendations.

Promote evidence:

- A tool call tagged `was_expanded: true` (the in-turn expansion path) or
  `expand_tools_used: true` (the sticky-carry path) is direct evidence
  the model reached for a tool that wasn't initially available.
- A tool is promoted when it appears across `promote_min_sessions`
  distinct sessions **and** receives `promote_min_calls` total calls
  within the window.

Demote evidence:

- A tool that appears in `always_on_tools` across the window but is
  never actually called.
- Demotion fires only when the window contains at least
  `demote_min_sessions_no_use` sessions, so capability isn't pulled on
  thin evidence.

Defaults are inherited from `policy.yaml` under `learning.shape_ceiling`:

```yaml
learning:
  shape_ceiling:
    session_window: 20
    promote_min_sessions: 2
    promote_min_calls: 3
    demote_min_sessions_no_use: 20
```

CLI flags override per-run. Output is merged into
`~/.hermes/state/tool-belt/learned.json` under
`scopes[].cache_aware`, with `promote` mirrored to `scopes[].always_on`
and `demote` to `scopes[].always_off` so the existing
`apply_to_preset` reader picks them up.

The shaper writes recommendations to `learned.json`, but the runtime does
not merge them under the default `learned_mode: recommend`. To activate
them, set `learned_mode: apply` on the scope.

## Patch surface in the request lifecycle

The plugin monkey-patches two methods on `AIAgent` at registration time.

### `AIAgent._build_api_kwargs`

Called by the agent to assemble the outbound API request. The plugin's
wrapper:

1. Calls the original to get the full `kwargs`.
2. Reads the current prediction state from a `ContextVar`. If absent
   (e.g. an out-of-band call), returns `kwargs` unchanged.
3. Filters `kwargs["tools"]` down to the prediction's allowed set,
   plus any expansions added by `expand_tools` since the prediction
   was set up.
4. Tools whose names aren't in any policy entry are kept on by default
   (safe fallback for plugin-provided tools).
5. Records per-call hashes for response-side cache correlation.
6. Under cache-on mode, propagates new expansions back into the
   session's frozen snapshot so they survive into later dispatches.

`self.tools` on the AIAgent is never mutated. Only the on-the-wire
payload is narrowed.

### `AIAgent._compress_context`

Called when the agent compacts conversation history. The plugin's
wrapper:

1. Calls the original to perform the compaction.
2. If the call succeeds and the active session has a frozen snapshot,
   evicts it.

The next dispatch re-runs the predictor against the post-compaction
state and freezes the result. This wrapper is best-effort: on a Hermes
build without `_compress_context`, the freeze persists across
compaction, which costs one cache break on the next dispatch but does
not corrupt anything.

### Hooks

In addition to the patches, the plugin registers:

- `pre_gateway_dispatch` — resolve cache mode, run predictor or reuse
  frozen snapshot, populate the prediction context.
- `post_tool_call` — log each actual tool call linked by
  `prediction_id`, with attribution flags (`was_initially_available`,
  `was_cut`, `was_expanded`, `after_expand_tools`, etc.).
- `post_api_request` — capture per-call cache_read / cache_write
  tokens, run the cache-mode detection state machine, detect cache TTL
  expiry (stable hash + cold cache after >5min idle).
- `on_session_end` — per-turn cleanup only; does **not** evict the
  freeze or cache-mode state.
- `on_session_reset` — true session reset; evicts the freeze, the
  cache-mode state, and all per-session sticky / lookback state.

## Policy resolution

Each dispatch resolves an effective policy through this chain
([presets.py](../presets.py) and [learned.py](../learned.py)):

```
User ceiling (platform_toolsets)
  → Base policy (policy.yaml)
  → Global always_on_extra / always_off
  → Per-scope channels.<scope> overrides
  → Learned overlay (only when learned_mode is apply)
  → expand_tools admissions (per session under cache-on; per turn under cache-off)
  → Sticky residency (cache-off only)
  → A/B bypass (deterministic per (scope, session_id) when bypass_rate > 0)
```

The ceiling is the upper bound. The plugin can only narrow within it,
never widen past it.

## State on disk

```
~/.hermes/state/tool-belt/
├── predictions.jsonl              # one row per inbound dispatch
├── tool_calls.jsonl               # one row per tool call, linked by prediction_id
├── api_calls.jsonl                # one row per outbound API call, with cache + hash data
├── cache_mode_detection.json      # per-scope locked mode (cross-session)
└── learned.json                   # promote / demote recommendations from the shaper
```

The plugin never writes `learned.json` from the prediction path. Writes
come from the analyzer or the shaper script — an intentional boundary
that keeps adaptive changes reviewable.

## Failure modes

By design, any failure path returns the unfiltered tool list:

- `policy.yaml` missing or malformed → wildcard preset, no narrowing.
- Predictor exception → wildcard result.
- Filter exception in `_build_api_kwargs` → original `kwargs` returned.
- `learned.json` load error → base policy returned as-is.
- `expand_tools` called with no prediction context → structured error
  JSON returned to the model.

The plugin is built to be invisible when broken, never destructive.
