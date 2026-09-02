# tool-belt Architecture

`tool-belt` is a Hermes plugin that reduces per-request tool overhead by
sending the model only the tools likely to matter, while leaving the user's
`platform_toolsets` ceiling intact. This document describes how the plugin
makes that decision, where it intercepts the request lifecycle, and how its
behavior changes depending on whether the provider a session is talking to
supports prefix caching.

The code is the source of truth. Where a file is referenced inline, that's
where the named mechanism actually lives.

## The problem

Every API call ships the full tool block — names, descriptions, JSON
schemas — to the model. For a profile with a handful of plugins, that's
8–15k tokens of overhead per message, regardless of whether the answer is
`"yes"` or `"run the build and check the logs"`.

The naive fix is to narrow the tool block per turn based on intent. That
fix conflicts with how modern providers bill multi-turn conversations.
Anthropic, OpenAI (auto-cache) and OpenRouter hash the prefix of each
request — roughly `system + tools + history` — and bill the cached span at
a steep discount. Mutating the tool block between turns invalidates the
cache boundary that sits in the middle of the prefix. The provider then
re-bills the conversation history at the full input rate. Measured on the
operator install this plugin was built against, one such break cost
≈42.8K tokens per event on a mixed-provider history and ≈54.4K on an
all-Codex one — 28–36× the flat 1,500-token estimate the plugin used to
charge. On a caching provider the tool schemas themselves are only ~5% of
cost, because they sit in the cheap cached prefix; the cost driver is
uncached input, mostly cache-miss re-bills. Narrowing there saves pennies
and risks a large break.

On a provider that never caches (ollama-cloud: 0% hits over 4,154
measured calls) the arithmetic flips: there is no prefix to protect, and
every schema token not sent is a full-price token saved, on every call.

`tool-belt` resolves this by making the *posture* a property of the
provider a session is talking to, per call:

- **Goal**: usage-aware tool loading at per-tool granularity. Tools the
  model needs are loaded; tools it doesn't are dropped — on the providers
  where dropping them pays.
- **Posture**: on a non-caching provider, narrow per turn. On a caching
  provider, carry everything and never touch the list — get out of the
  cache's way.

The plugin detects which posture each provider deserves and operates
accordingly. Essentially all of its honest savings come from the
non-caching posture.

## Two postures

The plugin runs in one of two postures per session, pinned at the
session's first dispatch.

### Cache-on / carry-all (caching providers)

Used for providers with active prefix caching (Anthropic, OpenAI
auto-cache, OpenRouter — 90%+ hit rates measured). `cache_mode: on`
forces it.

1. On every dispatch the plugin ships the **full enabled ceiling minus
   `expand_tools`**. Nothing is narrowed, so the predictor
   ([predictor.py](../predictor.py)), sticky residency, and lookback do
   not run. The state is built by `_build_carry_all_state` in
   [__init__.py](../__init__.py).
2. The tool list is **never mutated for the session's life**. Any
   mutation on a caching provider busts the prefix cache and re-bills
   the whole history — far more than per-turn narrowing ever saves. With
   nothing narrowed there is nothing to expand, so `expand_tools` is
   stripped from the wire too: shipping it would only invite the one
   mutation the posture exists to prevent.
3. A prediction row is still written per dispatch (`policy_source:
   "cache_on_carry_all"`, `ceiling == narrowed`, reduction 0) so the
   cohort stays visible in telemetry and feeds usage learning (which
   tools were actually called).
4. The posture pin is evicted only at moments that already cost a cache
   break: true `/reset` or `/new`, compaction, and a mid-session provider
   change that would flip the posture (see
   [The carry-all posture](#the-carry-all-posture) below).

### Cache-off (non-caching providers)

Used for providers without effective prefix caching (ollama-cloud, any
route with `provider_caches: false`). `cache_mode: off` forces it. This is
the value engine.

1. On **every dispatch**, the predictor runs against the resolved policy
   ([policy.yaml](../policy.yaml), layered with per-scope overrides and
   any active learned overlay). It produces an allowed set from the
   carried strata (`always_carry` pins plus every tool not demoted to
   expand-only) and any triggered groups.
2. `expand_tools` ships, carrying the expand-only manifest, so the model
   can recover a tool the predictor left out.
3. Recently expanded categories carry forward via **sticky residency** —
   tools admitted by a prior turn's `expand_tools` stay loaded for
   `ttl_turns` further turns without needing re-expansion.
4. The predictor optionally concatenates the **last N user messages**
   (lookback) before running its regex classifier, so short replies like
   `"yes"` or `"go ahead"` inherit intent from the prior turn.
5. The on-the-wire tool list is the intersection of
   `(always_carry ∪ carry ∪ triggered ∪ expanded ∪ sticky)` with the
   user's ceiling.

Both postures share the safety properties: the ceiling is never widened
beyond what the user's `platform_toolsets` already allowed, and any
exception anywhere falls through to no narrowing. The bypass cohort
(`bypass_rate`, the internal A/B control) is unchanged by the posture and
is still excluded from savings.

## Posture detection

Whether a route caches is a property of the provider, not the scope: a
scope mixes providers through primary, fallback, and delegation. So
detection state is keyed **per `scope|provider`**, one bucket per route
the scope has been served on, and a session's posture is resolved against
the provider it is talking to.

Configuration:

```yaml
plugins:
  entries:
    tool-belt:
      settings:
        cache_mode: auto    # auto | on | off
```

- `cache_mode: on` and `cache_mode: off` are honored verbatim for every
  session.
- `cache_mode: auto` (default) consults the `scope|provider` bucket —
  this session's in-process lock first, then the cross-session detection
  cache. A locked bucket is reused immediately. Otherwise the plugin
  defaults to `on` (until proven uncached, protect prefix-cache
  stability) and starts an observation window.

The auto path observes `cache_read_tokens` across early API calls,
accumulating in the bucket of **the call's** provider — a failover call
to a non-caching provider never drags a caching primary's numbers. Once
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
  entries:
    tool-belt:
      settings:
        cache_auto:
          detect_calls: 5              # min calls before locking
          detect_min_input_tokens: 5000 # min cumulative input before locking
          on_threshold: 0.40
          providers_off_models:        # known-uncached models OR providers, lock immediately
            - kimi-k2.6:cloud
            - gpt-5.4-mini
```

Locked decisions persist per bucket to
`~/.hermes/state/tool-belt/cache_mode_detection.json` under the key
`"<scope>|<provider>"` (provider lower-cased). Pre-1.0 files keyed by bare
`"<scope>"` are re-keyed under the configured primary provider on load —
the only provider a scope-level lock could have been observing — or kept
bare when the primary is unknown. Subsequent sessions on the same route
skip the observation window.

### Which provider a session resolves against

The posture is pinned at a session's first dispatch, resolved in order
against: (a) the provider the session has already been observed on (set
from the per-call `provider` Hermes reports, so it follows failover and
`/model`), (b) the configured primary — Hermes' top-level
`model.provider`, (c) the bare-scope legacy entry. A detection lock that
lands mid-session applies to the *next* session, never the current one.

The one mid-session re-resolution is a provider change whose posture
differs. When `/model` or a failover moves a session from a caching
provider to a non-caching one (or back), `_maybe_evict_on_provider_change`
evicts the pin so the next dispatch re-resolves against the provider the
session is actually talking to. The switch already busted the provider's
prefix cache, so the re-resolution costs nothing extra. A switch to a
provider with the *same* posture keeps the pin and just records the new
provider.

The detection code is `_update_cache_mode_detection` in
[__init__.py](../__init__.py); the per-dispatch lookup is
`_resolve_cache_mode_for_session`, over `_resolve_posture_for_provider`.

## The carry-all posture

There is no per-session frozen snapshot. Under cache-on the plugin holds
only a posture pin (`_CACHE_DECISION_BY_SESSION`, keyed by the canonical
Hermes session key `agent:main:{platform}:...`, recording the mode and the
provider it was resolved against) plus the per-session detection buckets
(`_CACHE_MODE_BY_SESSION`). The tool list shipped is whatever Hermes
assembled for the call, minus `expand_tools` — so it is byte-stable
across the session by construction, not by bookkeeping.

### Eviction

The plugin registers two distinct hooks:

- `on_session_end` — fires after every `run_conversation` call (i.e.
  once per turn in multi-turn sessions; the hook's name in Hermes is
  misleading). The handler **evicts nothing** — not the pin, not sticky
  residency, not the lookback ring (those are session-scoped and
  self-trim). It resets the per-turn contextvar and runs the debounced
  auto-shape pass.
- `on_session_reset` — fires on a true `/reset` or `/new`. The handler
  **evicts** the posture pin, the cache-mode detection state, and all
  per-session sticky / lookback state.

The pin is also evicted on compaction. The plugin wraps
`AIAgent._compress_context` so that after the compressor rewrites
conversation history (which invalidates the provider's cached prefix
regardless of plugin behavior), a carry-all session's pin is dropped. The
next dispatch re-resolves the posture against the post-compaction state —
a detection lock that landed mid-session can take effect here. Cache-off
sessions are narrowing per turn already; their pin and detection counters
survive compaction unchanged. This wrapper is installed by
`_wrap_compress_context`.

All three paths — compaction, `/reset`, provider change — go through
`_evict_session_cache_state`; every one is a moment the cached prefix is
already gone.

### Divergence detection

A bucket locked `on` whose post-lock cache hit rate stays low is
evidence that something upstream of the tool block is busting cache —
typically a system-prompt mutation (e.g. memory injection) or a provider
cache regression. The plugin tracks consecutive post-lock calls with hit
rate below 30% per bucket; after 3 in a row it logs one divergence
warning. A call that landed on a different provider than the session's
previous call is *expected* to miss (that provider has no prefix for this
history), so it resets the streak and never counts. The session continues
carry-all (mid-session posture switches are not supported by
construction, except the provider-change eviction above), but the
analyzer can flag the session for review.

### Hash fingerprints

Each API call records:

- `tool_list_hash` — truncated sha256 of the tool list in send order.
  Captures membership, ordering, and schema-byte drift. Provider caches
  fingerprint the actual schema bytes, so a description change
  invalidates the cache even when names are stable.
- `system_hash` — truncated sha256 of the first message when its role
  is `system` or `developer`. Lets analysis distinguish "the tool list
  held but the system prompt changed" from "the tool list itself
  changed."

Both hashes are written to `api_calls.jsonl` alongside `cache_read_tokens`,
`cache_write_tokens`, and `provider_caches` (the call's bucket verdict),
which is how cache economics get measured directly rather than inferred.

## expand_tools

[expand_tools.py](../expand_tools.py) registers a meta-tool the model can
call when it wants a tool group that wasn't loaded. It ships only under
the cache-off posture. The schema is:

```json
{
  "name": "expand_tools",
  "parameters": {
    "type": "object",
    "properties": {
      "category": {
        "type": "string",
        "description": "browser, image_gen, tts, vision, cronjob, delegation, code_execution, terminal, file, web, skills, homeassistant, ..."
      },
      "tool": {
        "type": "string",
        "description": "a specific tool name; the handler finds its category"
      }
    },
    "required": []
  }
}
```

The handler resolves the category to its tool names via Hermes' toolset
table and appends them to the prediction context's `expansions` set.
The patched `_build_api_kwargs` reads that set on the next API call and
unions the resolved tools into the allowed list.

Successful expansions refresh sticky residency for the category — the
tools stay loaded for `ttl_turns` further turns and then decay.

Under carry-all the tool is stripped from the wire (see above); a
carry-all session therefore produces no expand events. Every
`expand_tools` event on a cache-off session is logged as a learning
signal. The between-session shaper reads these to decide which tools to
promote into future sessions' starting ceilings.

Hermes' own Tool Search bridge (`tool_search` / `tool_describe` /
`tool_call`) is complementary, not an alternative: it defers MCP and
plugin tools — the cold long tail — and core tools never defer by its
design. `tool-belt` shapes the hot core, and only on non-caching
providers. The bridge tools are pass-through for the plugin: outside the
partition, never shaped.

## Between-session shaping

The shaping engine lives in [shaping.py](../shaping.py). It reads
`predictions.jsonl` and `tool_calls.jsonl`, groups by scope and session,
and produces per-scope promote / demote assignments. It runs in three
places: the in-process auto-shape pass at session end (the normal path —
debounced per scope via `auto_shape_stamp.json`, apply-mode scopes only),
`tool-belt configure --mode history` (interactive, diffed and confirmed),
and [scripts/shape-ceiling.py](../scripts/shape-ceiling.py) (the CLI
wrapper for on-demand or scripted runs).

Shaping is **moot under carry-all**. A scope whose primary provider is
locked `on` ships no `expand_tools`, so demotion saves nothing and
promotion has nothing to recover from: `compute_scope_recommendations`
returns an explicit no-op (`reason: "caching_provider_carry_all"`) and
`auto_shape_run` lists the scope under `skipped_carry_all`, unstamped, so
a posture flip re-qualifies it at once. A scope's effective posture for
shaping is its primary provider's bucket (`read_cache_mode` →
`resolve_primary_detection_entry`): the profile's own `config.yaml`
`model.provider` first, then the Hermes config loader, then the
most-locked `scope|provider` bucket, then a legacy bare key. Carry-all
prediction rows still count as *usage* evidence (which tools were called)
but can never be expansion evidence.

Promote evidence (cache-off scopes):

- A tool call tagged `activated_by_expansion: true` (the in-turn
  expansion path) or `expansion_provided_access: true` (the sticky-carry
  path) is direct evidence the model reached for a tool that wasn't
  initially available. Trigger activation is never promote evidence.
- A tool is promoted when it clears the anti-flap gates
  (`promote_min_sessions` distinct sessions **and** `promote_min_calls`
  total calls) **and** the economics favor carrying: the observed
  expansion spend (`sessions with use × per-event cost`) exceeds what
  carrying the tool would have cost over the same window.

Demote evidence — the economic test (token-denominated by design; the
decision never consults a price table, because fewer tokens is always
cheaper on every route):

`expand_tools` is sticky — expand once and the tool rides carried for the
rest of the session — so a session *with* use costs about the same either
way, plus one expansion. Demotion only saves in sessions *without* use:

```
saving  = schema_size(tool) × billable exposures in sessions WITHOUT use
penalty = per_event_cost × sessions WITH use      (one expand_tools event each)
demote when saving > demote_k × penalty
promote when penalty > saving                     (after the anti-flap gates)
```

- `per_event_cost` is the scope's **measured** expansion cost
  (`savings.measure_expand_overhead`: for a non-caching scope, the median
  input size of the expand meta-call). `EXPAND_ROUND_TRIP_TOKENS = 1500`
  is only the thin-data fallback (fewer than 5 expand events); the
  shaper's rationale block records the value it used as
  `expand_round_trip_tokens`.
- Per-tool schema sizes come from the `schema_sizes.json` sidecar
  (snapshotted in-process, since only the live gateway sees real tool
  definitions); unmeasured tools fall back to a 388-token average.
- Billable exposures: a cache-off scope pays the manifest on every API
  call (counted per prediction from `api_calls.jsonl`, min 1). A scope
  whose posture is *unknown* is charged one exposure per session — the
  conservative direction, biasing toward carrying. (A scope locked `on`
  never reaches the arithmetic.)
- Trigger-activated uses don't defend a carry slot: they stay free for a
  demoted tool.
- A tool with zero uses is the limit case (`demote_tokens = 0`) — it
  always demotes, which is the old binary rule.
- Between the demote and promote thresholds is a hysteresis band where a
  tool holds its current class, so assignments don't flap.
- Demotion fires only when the window contains at least
  `demote_min_sessions_no_use` sessions, so capability isn't pulled on
  thin evidence.

Defaults are inherited from `policy.yaml` under `learning.shape_ceiling`:

```yaml
learning:
  shape_ceiling:
    session_window: 100
    promote_min_sessions: 2
    promote_min_calls: 3
    demote_min_sessions_no_use: 20
    demote_k: 1.5
```

CLI flags override per-run. Output is merged into
`~/.hermes/state/tool-belt/learned.json` as schema v2: promotions land in
`scopes[].carry`, demotions in `scopes[].expand_only`, and the shaper's
rationale (counts, economics, timestamps) in `scopes[].shaping`.

The shaper writes assignments to `learned.json`, and under the default
`learned_mode: apply` the runtime merges them into the preset on every
cache-off dispatch (carry-all sessions ship the whole ceiling regardless).
`learned_mode: recommend` is the opt-out observe mode: the overlay is kept
on disk but not applied.

## Patch surface in the request lifecycle

The plugin monkey-patches two methods on `AIAgent` at registration time.

### `AIAgent._build_api_kwargs`

Called by the agent to assemble the outbound API request. The plugin's
wrapper:

1. Calls the original to get the full `kwargs`.
2. Reads the current prediction state from a `ContextVar`. If absent
   (e.g. an out-of-band call), returns `kwargs` unchanged.
3. Under carry-all: removes `expand_tools` from `kwargs["tools"]` and
   ships everything else verbatim.
4. Under cache-off: filters `kwargs["tools"]` down to the prediction's
   allowed set, plus any expansions added by `expand_tools` since the
   prediction was set up.
5. Tools no policy or learned entry names are carried (full-start): a
   newly shipped plugin tool is on the wire until evidence demotes it.
6. Records per-call hashes for response-side cache correlation and, on
   the first call of the turn, writes the prediction row.

`self.tools` on the AIAgent is never mutated. Only the on-the-wire
payload is narrowed.

### `AIAgent._compress_context`

Called when the agent compacts conversation history. The plugin's
wrapper:

1. Calls the original to perform the compaction.
2. If the call succeeds and the active session is pinned carry-all,
   evicts the pin and detection state.

The next dispatch re-resolves the posture against the post-compaction
state. This wrapper is best-effort: on a Hermes build without
`_compress_context`, the pin persists across compaction, which only means
a mid-session lock waits for the next session to take effect — nothing is
corrupted.

### Hooks

In addition to the patches, the plugin registers:

- `pre_gateway_dispatch` — resolve the posture (pinned at first
  dispatch), build carry-all state or run the predictor, populate the
  prediction context.
- `post_tool_call` — log each actual tool call linked by
  `prediction_id`, with attribution (`source`, `was_initially_active`,
  `was_expand_only`, `activated_by_expansion`, `activation_source`).
- `post_api_request` — capture per-call cache_read / cache_write
  tokens, run the cache-mode detection state machine in the call's
  `scope|provider` bucket, stamp `provider_caches`, evict the pin on a
  posture-changing provider switch, detect cache TTL expiry (stable
  hash + cold cache after >5min idle).
- `on_session_end` — per-turn: clears the prediction contextvar and
  runs the debounced in-process auto-shape pass. Evicts **nothing**
  (pin, cache-mode, sticky, lookback all survive across turns).
- `on_session_reset` — true session reset; evicts the pin, the
  cache-mode state, and all per-session sticky / lookback state.

## Policy resolution

Each cache-off dispatch resolves an effective policy through this chain
([presets.py](../presets.py) and [learned.py](../learned.py)); a
carry-all dispatch resolves the preset for telemetry but ships the whole
ceiling:

```
User ceiling (platform_toolsets)                      E
  → Base policy (policy.yaml always_carry + triggers)
  → Config pins (always_carry, global ∪ per-channel)  A = pins ∩ E
  → Per-channel settings (channels.<agent>.<platform>)
  → Learned overlay (only when learned_mode is apply)  C / X split
  → Trigger activation (policy ∪ learned overlay)     T = triggered ∩ X
  → expand_tools admissions (cache-off only)
  → Sticky residency (cache-off only)
  → Bypass (deterministic per (scope, session_id) when bypass_rate > 0)
```

The ceiling is the upper bound. The plugin can only narrow within it,
never widen past it.

## State on disk

```
~/.hermes/state/tool-belt/
├── predictions.jsonl              # one row per inbound dispatch
├── tool_calls.jsonl               # one row per tool call, linked by prediction_id
├── api_calls.jsonl                # one row per outbound API call, with cache + hash data + provider_caches
├── cache_mode_detection.json      # per scope|provider locked mode (cross-session)
├── schema_sizes.json              # measured per-tool schema token sizes
├── auto_shape_stamp.json          # per-scope auto-shape debounce stamps
├── inventory.json                 # tool-absence tracking (reconciliation grace)
├── configure-state.json           # configure's pre-observation bypass memo
├── learned_recommendations.json   # analyzer's reviewable recommendation report
├── harvest/                       # session-history replay artifacts
└── learned.json                   # the learned carrying assignment (v2)
```

`cache_mode_detection.json` is a flat object keyed `"<scope>|<provider>"`
(legacy bare `"<scope>"` keys are migrated on load); each entry carries
`mode`, `locked_at`, `lock_reason`, `hit_rate_at_lock`, `sessions_locked`,
`last_model`. Every `api_calls.jsonl` row carries `cache_mode` — that
*call's* bucket outcome (`pending` / `on` / `off`), not the session
posture — and `provider_caches` (`true` / `false` / `null` while pending),
the field the savings engine prices against.

The plugin never writes `learned.json` from the prediction path. Writes
come from the in-process auto-shape pass (session end, debounced), the
shaper, and the configure flows — every one routed through
`learned.write_state`.

## Failure modes

By design, any failure path returns the unfiltered tool list:

- `policy.yaml` missing or malformed → wildcard preset, no narrowing.
- Predictor exception → wildcard result.
- Filter exception in `_build_api_kwargs` → original `kwargs` returned.
- `learned.json` load error → base policy returned as-is.
- `expand_tools` called with no prediction context → structured error
  JSON returned to the model.

The plugin is built to be invisible when broken, never destructive.
