# CONFIGURATION

Reference for every configuration surface in `tool-belt`. For
behavior — what the plugin does at runtime, when freezes happen, how
the cache-mode state machine evolves — see
[ARCHITECTURE.md](ARCHITECTURE.md).

## Config layering

Configuration is assembled from three files with distinct ownership:

| File | Owner | Purpose |
|---|---|---|
| `policy.yaml` (plugin dir) | Plugin | Shipped preset. Always-on baseline, trigger rules, shaper thresholds. |
| `config.yaml` (user) | User | Enable the plugin, set mode flags, add per-scope overrides. |
| `learned.json` (state dir) | `shape-ceiling.py` | Auto-tuned promote/demote per scope. Generally not hand-edited. |

State files (`learned.json`, `cache_mode_detection.json`, JSONL
telemetry) live under `~/.hermes/state/tool-belt/` by default. When
`HERMES_HOME` is set (per-profile installs), all state paths resolve
relative to it — so per-profile state lands under
`$HERMES_HOME/state/tool-belt/`.

Resolution order at dispatch time (later layers override earlier):

1. `policy.yaml` — base preset (`always_on`, `triggers`).
2. `config.yaml` global overrides — `always_on_extra`, `always_off`.
3. `config.yaml` per-scope overrides — `channels.<scope>.*`.
4. `learned.json` — applied only when `learned_mode` is `apply`.

See [`presets.py:resolve_preset`](../presets.py) for the resolver.

---

## Managed configuration (`scripts/configure.py`)

Most users never edit `config.yaml` for this plugin.
[`scripts/configure.py`](../scripts/configure.py) writes the per-scope keys
for them, through `hermes config set` / `hermes config unset` — Hermes owns
`config.yaml`, so the file is never edited directly. Every write is shown as
a `before → after` line first and requires confirmation.

The command writes exactly two per-scope keys, both of which are ordinary
documented settings — it introduces no key of its own:

| Key | Written when | Value |
|---|---|---|
| `plugins.tool-belt.channels.<scope>.learned_mode` | shape path | `apply` |
| `plugins.tool-belt.channels.<scope>.learned_mode` | recommend path, reset | `recommend` |
| `plugins.tool-belt.channels.<scope>.bypass_rate` | recommend path | `1.0` (full observation) |
| `plugins.tool-belt.channels.<scope>.bypass_rate` | shape path, on acceptance | `0.0` (narrow immediately) |
| `plugins.tool-belt.channels.<scope>.bypass_rate` | reset | the value observation mode replaced, default `0.0` |

Scope keys are the same `agent:platform` identifiers used by telemetry — see
[`channels`](#channels).

### Observation mode

The recommend path sets a scope's [`bypass_rate`](#bypass_rate) to `1.0`.
Every session then ships the untouched tool ceiling while telemetry still
records what the predictor *would* have selected — so a later shaping run has
real evidence without any narrowing having happened in the meantime. The
command reports how many more sessions each scope needs; that minimum is
derived from [`learning.shape_ceiling`](#learningshape_ceiling) in
`policy.yaml`, never hardcoded.

### Reset behavior

`configure.py --reset <agent>` returns a shaped scope to its pre-shaping
state:

1. The scope's entry is removed from [`learned.json`](#learnedjson-reference).
   Other scopes and the global block are left untouched, and the file itself
   is preserved.
2. `channels.<scope>.learned_mode` is set to `recommend`, so the overlay stops
   being merged even if one is written again later.
3. `channels.<scope>.bypass_rate` is restored to whatever it was before
   observation mode replaced it.

Step 3 reads a small sidecar, `configure-state.json`, written next to
`learned.json` in the scope's state directory when observation mode is
enabled. It records only the previous `bypass_rate` per scope. It is not part
of the learned-state schema, nothing at runtime reads it, and deleting it is
harmless — a reset then restores the shipped default of `0.0`.

---

## `config.yaml` reference

User config lives under the `plugins.tool-belt` key of Hermes'
`config.yaml`. The loader is
[`_load_user_config`](../__init__.py); defaults come from the
`_CONFIG` dict in the same file.

```yaml
plugins:
  tool-belt:
    enabled: true
    log: true
    cache_mode: auto
    learned_mode: recommend
    bypass_rate: 0.0
    always_on_extra: []
    always_off: []
    cache_off:
      sticky:
        enabled: true
        ttl_turns: 3
        categories: [terminal, file, browser, web, code_execution, delegation]
      predictor:
        lookback_turns: 1
    cache_auto:
      detect_calls: 5
      detect_min_input_tokens: 5000
      on_threshold: 0.40
      providers_off_models: ["kimi-k2.6:cloud", "gpt-5.4-mini"]
    channels: {}
```

### `enabled`

Type: `bool`. Default: `false`.

Master switch. When `false`, `register()` short-circuits and no
monkey-patches are installed — the plugin is a no-op.

### `log`

Type: `bool`. Default: `true`.

Toggles all JSONL telemetry writers
(`predictions.jsonl`, `tool_calls.jsonl`, `api_calls.jsonl`). Setting
`false` keeps the narrowing behavior but disables observation.

### `cache_mode`

Type: `string`. One of `auto`, `on`, `off`. Default: `auto`.

Decides whether the plugin freezes tools per session or re-narrows per
turn. See [ARCHITECTURE.md](ARCHITECTURE.md) for the two-mode model.

- `on` — freeze tools at session start. Skips sticky + lookback.
- `off` — per-turn narrowing with sticky residency.
- `auto` — consult the cross-session detection cache; default `on` on a
  miss. The per-session detection state machine still runs and can
  lock the session to `on`/`off` after enough calls.

Resolved by
[`_resolve_cache_mode_for_session`](../__init__.py).

### `learned_mode`

Type: `string`. One of `apply`, `recommend`. Default:
`apply`. Resolved by [`learned.py:learned_mode`](../learned.py).

Controls whether [`learned.json`](#learnedjson-reference) is applied
to the active preset:

- `apply` (default) — merge the learned overlay (demotions/promotions)
  into the preset during resolution; evidence-driven shaping lands
  automatically. Prediction rows stamp `learned_mode: apply`.
- `recommend` — opt-in observe/trial mode: the runtime does **not** merge
  `learned.json`. Recommendations flow through
  `analyze.py --write-recommendations` and the shaper for human review;
  behavior is unchanged.

Legacy values normalize at load: `off` → `recommend`, and `auto` /
`audit` → `apply`.

May be overridden per scope via `channels.<scope>.learned_mode`.

### `always_carry`

Type: `list[string]`. Default: `[]`. Also accepted under
`channels.<scope>.always_carry` — union semantics: a scope entry adds
pins, it never removes global ones.

Per-agent always-carry pins. The effective immutable resident baseline is
the shipped `policy.yaml` `always_carry` ∪ this list, intersected with the
enabled tool ceiling at resolution — a name Hermes has disabled (or a typo)
is inert and logged as a warning, never an error. Pinned tools are
undemotable by construction: the runtime ignores any learned demotion
naming one, and the shaper/auto-shape engine drops such candidates before
writing.

### `bypass_rate`

Type: `float` in `[0.0, 1.0]`. Default: `0.0`.

Fraction of sessions (deterministic by `sha1(scope|session_id)`) that
bypass narrowing and ship the full tool ceiling. Bypassed sessions
still run the predictor (so telemetry is preserved) but
`policy_source` is stamped `bypass` and `allowed_tool_names` widens to
the ceiling. Use `1.0` to disable narrowing on a scope without
disabling the plugin.

Per-scope override: `channels.<scope>.bypass_rate`. See
[`_bypass_rate_for_scope`](../__init__.py).

### `always_on_extra`

Type: `list[str]`. Default: `[]`.

Tool names *added* to the preset's `always_on` baseline. Applied
globally; layered before per-scope overrides.

### `always_off`

Type: `list[str]`. Default: `[]`.

Tool names *removed* from the resolved always-on list and from every
trigger group's tool set. Applied globally; layered before per-scope
overrides.

### `cache_off`

Controls the per-turn pipeline used when the session is in cache-off
mode. Inert under cache-on. Default values are baked into `_CONFIG`.

```yaml
cache_off:
  sticky:
    enabled: true
    ttl_turns: 3
    categories: [terminal, file, browser, web, code_execution, delegation]
  predictor:
    lookback_turns: 1
```

#### `cache_off.sticky.enabled`

Type: `bool`. Default: `true`.

When `true`, tools added via `expand_tools` persist for `ttl_turns`
additional turns before being dropped from the per-turn allow set.

#### `cache_off.sticky.ttl_turns`

Type: `int`. Default: `3`.

Number of *additional* turns a sticky-resident category carries forward.
Coerced to `max(1, ttl_turns)` by [`_sticky_ttl_turns`](../__init__.py).

#### `cache_off.sticky.categories`

Type: `list[str]` or `"*"`. Default:
`[terminal, file, browser, web, code_execution, delegation]`.

Toolset categories eligible for sticky residency. These are
expand_tools category names (`browser`, `terminal`, etc.) — NOT
`policy.yaml` trigger group names. Set to `"*"` (or omit) to allow all
categories.

#### `cache_off.predictor.lookback_turns`

Type: `int >= 0`. Default: `1`. Resolved by
[`_predictor_lookback_turns`](../__init__.py).

Number of prior user messages prepended to the current message before
the predictor classifies. Catches signal in short replies like `"yes"`
/ `"go ahead"` whose intent lives in the preceding turn. Set to `0` to
disable.

### `cache_auto`

Controls the cache-mode detection state machine
([`_update_cache_mode_detection`](../__init__.py)). Inert when
`cache_mode` is forced to `on` or `off`.

```yaml
cache_auto:
  detect_calls: 5
  detect_min_input_tokens: 5000
  on_threshold: 0.40
  providers_off_models: ["kimi-k2.6:cloud", "gpt-5.4-mini"]
```

#### `cache_auto.detect_calls`

Type: `int >= 1`. Default: `5`.

Minimum API calls to observe before locking the session's mode.

#### `cache_auto.detect_min_input_tokens`

Type: `int >= 0`. Default: `5000`.

Minimum accumulated `cache_read + cache_write + input` token volume
required before locking, in addition to `detect_calls`. Prevents
short-opener sessions from locking on noise.

#### `cache_auto.on_threshold`

Type: `float`. Default: `0.40`.

Cache-hit ratio at which the session locks to `on`. Below the threshold
the session locks `off`. Ratio is
`cache_read / (cache_read + cache_write + input)`.

#### `cache_auto.providers_off_models`

Type: `list[str]`. Default: `["kimi-k2.6:cloud", "gpt-5.4-mini"]`.

Model identifiers that skip the observation window and lock `off`
immediately. Provider-blocklist locks are *not* persisted to the
cross-session detection cache — they're a config statement, not an
observation.

### `channels`

Type: `dict[str, dict]`. Default: `{}`.

Per-scope overrides. Scope keys are `{agent}:{platform}` (e.g.
`default:telegram` for the root profile). Lookups try the full scope first, then
fall back to the bare platform segment — so a `telegram` entry still applies
to `default:telegram` when no more specific entry exists.

An explicit plugin `agent` value remains the highest-precedence label. Named
profiles use their directory name, the root profile uses Hermes' reserved
`default` label, and calls without any profile context remain unattributed.

Each scope value mirrors the top-level shape and may set any of:

- `always_on_extra`
- `always_off`
- `learned_mode`
- `bypass_rate`

```yaml
channels:
  cli:
    always_on_extra: [terminal, execute_code]
  default:telegram:
    learned_mode: recommend
    always_on_extra: [terminal]
  slack:
    bypass_rate: 1.0
```

---

## `policy.yaml` reference

The single shipped policy lives at the plugin root
([`policy.yaml`](../policy.yaml)) and is loaded by
[`presets.py:load_base_policy`](../presets.py). One file ships with
the plugin; for new shapes either edit it in place or fork it.

```yaml
name: aggressive
description: >
  Minimum always-on; everything else trigger-gated.

learning:
  shape_ceiling:
    session_window: 20
    promote_min_sessions: 2
    promote_min_calls: 3
    demote_min_sessions_no_use: 20

always_on:
  - mnemosyne_remember
  - clarify
  - read_file
  - send_message
  - expand_tools

triggers:
  - name: shell
    tools: [terminal]
    keywords:
      - '\b(run|install|execute|shell|terminal|build|deploy)\b'
    exclude_keywords:
      - '\b(should we|might|maybe|eventually)\b'

  - name: vision
    tools: [vision_analyze]
    has_attachment: image
    keywords:
      - '\b(this (image|picture|photo|screenshot))\b'
```

### Top-level keys

- `name` (`string`) — preset identifier; surfaces in `predictions.jsonl`.
- `description` (`string`) — free text, ignored by the loader.
- `always_on` (`list[str]` or `"*"`) — tools loaded on every message in
  this scope, regardless of message content. The literal `"*"` means
  "no narrowing — load everything in the user's `platform_toolsets`
  ceiling".
- `triggers` (`list[dict]`) — trigger groups, evaluated against each
  inbound message. Multiple groups may fire on one message (additive).
- `learning.shape_ceiling` (`dict`) — default thresholds for the
  between-session shaper. See below.

Tools in the user's ceiling that are not listed in `always_on` and not
named under any trigger are considered "unknown" and kept on (safe
fallback for plugin-provided tools the policy author didn't list).

### Trigger group shape

```yaml
- name: shell
  tools: [terminal, process]
  keywords:
    - '\b(run|install|deploy)\b'
    - '\$\s*\w+'
  exclude_keywords:
    - '\b(should we|might|maybe)\b'
  has_attachment: image
```

| Key | Type | Required | Meaning |
|---|---|---|---|
| `name` | `string` | yes | Identifier surfaced in telemetry (`triggers_fired`). |
| `tools` | `list[str]` | yes | Tool names added when the group fires. |
| `keywords` | `list[str]` | no | Regex patterns. Group fires if any match (case-insensitive). |
| `exclude_keywords` | `list[str]` | no | Veto regexes. Any match suppresses the group, even if positive signals matched. |
| `has_attachment` | `string` | no | When set to `image` or `audio`, the group fires if the event carries a matching attachment. Combined with `keywords` as an OR. |

Patterns are compiled with `re.IGNORECASE`. Bad regexes are logged and
skipped — they don't break policy load. See
[`presets.py:_compile_keywords`](../presets.py).

### `learning.shape_ceiling`

Default thresholds for [`scripts/shape-ceiling.py`](../scripts/shape-ceiling.py).
The script reads them via `load_shape_ceiling_defaults`. CLI flags
override per-run.

| Key | Default | Meaning |
|---|---|---|
| `session_window` | `20` | Look back at most this many recent sessions per scope. |
| `promote_min_sessions` | `2` | A tool needs `expand_tools`-driven use across this many distinct sessions to promote. |
| `promote_min_calls` | `3` | And this many total calls. |
| `demote_min_sessions_no_use` | `20` | Window must contain this many sessions before any always-on tool can be demoted for non-use. |

### Authoring a new preset

The shipped loader points at the single `policy.yaml` at the plugin
root. To run a fork:

1. Copy `policy.yaml` to a new file.
2. Edit `name`, `always_on`, `triggers`, and (optionally) `learning`.
3. Replace `policy.yaml` in place — the loader path is hard-coded to
   `Path(__file__).parent / "policy.yaml"` in
   [`presets.py`](../presets.py).

To disable narrowing on a scope without removing it, set
`channels.<scope>.bypass_rate: 1.0` in `config.yaml` rather than
forking the preset.

---

## `learned.json` reference

Path: `$HERMES_HOME/state/tool-belt/learned.json` (or
`~/.hermes/state/tool-belt/learned.json` when `HERMES_HOME` is
unset). Read-mostly: owned by `scripts/shape-ceiling.py`; consumed by
[`learned.py:apply_to_preset`](../learned.py) when
`learned_mode` is `apply`.

```json
{
  "version": 1,
  "updated_at": "2026-06-01T00:00:00Z",
  "scopes": {
    "default:telegram": {
      "cache_aware": {
        "scope": "default:telegram",
        "computed_at": "2026-06-01T00:00:00Z",
        "sessions_considered": 20,
        "window_requested": 20,
        "promote": [
          {"tool": "browser_navigate", "sessions": 4, "calls": 7, "evidence": "expand_tools"}
        ],
        "demote": [
          {"tool": "image_generate", "sessions_without_use": 20, "evidence": "always_on_unused"}
        ]
      },
      "always_on": ["browser_navigate"],
      "always_off": ["image_generate"]
    }
  }
}
```

### Schema

- `version` (`int`) — must equal `1` (`LEARNED_VERSION` in
  [`learned.py`](../learned.py)). Mismatched versions are dropped.
- `updated_at` (`string`, ISO 8601 UTC) — last write timestamp.
- `scopes` (`dict[str, dict]`) — per-scope overlays, keyed by
  `{agent}:{platform}` (with bare-platform fallback).

Per scope:

- `cache_aware` — full shaper output (`promote`, `demote`,
  `sessions_considered`, `computed_at`). Owned by the shaper;
  human-readable rationale for the lists below.
- `always_on` — tools promoted into the always-on baseline. Mirrored
  from `cache_aware.promote` by the shaper.
- `always_off` — tools demoted from the always-on baseline and removed
  from every trigger group. Mirrored from `cache_aware.demote`.
- `trigger_adjustments` (optional) — `{<trigger_name>: {action:
  demote|disable}}`. Disables a trigger group entirely. Currently
  written only by manual edits.

### Global block

```json
{
  "global": {
    "always_off": ["some_tool"]
  }
}
```

A top-level `global.always_off` list applies across all scopes — useful
for blocklisting a tool everywhere without touching `config.yaml`.

### Hand-editing

`learned.json` is rewritten atomically on every shaper run. Manual
edits survive until the next `scripts/shape-ceiling.py` run that
produces a different recommendation for the same scope, at which point
the shaper-owned fields (`cache_aware`, `always_on`, `always_off`) are
overwritten. Hand edits belong in `config.yaml` (`always_on_extra` /
`always_off` / `channels.<scope>.*`); the learned file is for
auto-tuned state.

---

## Cache mode detection state

Path: `$HERMES_HOME/state/tool-belt/cache_mode_detection.json`.
Owned by the runtime; written on every per-session lock event by
[`_persist_detection_lock`](../__init__.py).

```json
{
  "default:telegram": {
    "mode": "on",
    "locked_at": 1717200000.0,
    "lock_reason": "threshold_met",
    "hit_rate_at_lock": 0.82,
    "sessions_locked": 14,
    "last_model": "claude-opus-4-7"
  }
}
```

### Fields

- `mode` — `on` or `off`.
- `locked_at` — Unix timestamp of the most recent lock for this scope.
- `lock_reason` — `threshold_met`, `threshold_failed`, `forced_on`,
  `forced_off`. `provider_blocklist` locks are intentionally *not*
  persisted.
- `hit_rate_at_lock` — cache-hit ratio at the lock decision.
- `sessions_locked` — count of consecutive same-mode locks for the
  scope. Resets to `1` when the mode flips.
- `last_model` — model identifier at the most recent lock.

### Override precedence

- Setting `cache_mode: on` or `cache_mode: off` in `config.yaml` wins
  over any persisted state — the explicit mode is forced for every
  session and no detection runs.
- Under `cache_mode: auto`, this cache supplies the initial freeze
  decision; the per-session detection state machine still runs and may
  rewrite the entry at lock time.
- Deleting the file is safe. The next dispatch defaults to `on` and
  rebuilds entries as sessions lock.

---

## Telemetry outputs

All telemetry writes through [`logger_io.py`](../logger_io.py) and is
controlled by `log: true`. Files are append-only JSONL at
`$HERMES_HOME/state/tool-belt/`.

### `predictions.jsonl`

One row per inbound gateway message. Written on the first API call of
the turn by [`_maybe_log_prediction`](../__init__.py). Schema is the
`PredictionRecord` dataclass in [`logger_io.py`](../logger_io.py).

Key fields:

- `ts`, `prediction_id`, `session_id`, `scope`, `agent`, `platform`
- `message_hash`, `message_preview` — sha1 prefix + first 80 chars.
  Full text is never logged.
- `preset`, `triggers_fired`, `triggers_suppressed`, `always_on_count`
- `ceiling_count` / `narrowed_count` and `ceiling_tokens` /
  `narrowed_tokens` — before/after the narrowing filter.
- `ceiling_tools`, `allowed_tools`, `cut_tools`, `unknown_kept_tools`,
  `always_on_tools`, `expanded_tools`, `sticky_tools`,
  `sticky_categories`, `sticky_remaining_turns`
- `policy_source` — `preset`, `learned`, or `bypass`.
- `policy_version`, `learned_mode`, `learned_scope`, `learned_changes`
- `lookback_used`, `lookback_turns_config`
- `tool_list_hash` — sha256 prefix of the wire-level tool schemas.
- `provider`, `model`
- `frozen_reuse`, `frozen_reuse_count` — true on cache-on dispatches
  after the first, with index within the session.

### `tool_calls.jsonl`

One row per actual tool call during a session. Written by
[`_on_post_tool_call`](../__init__.py).

Key fields:

- `ts`, `prediction_id`, `session_id`, `tool_name`
- `agent`, `platform`, `scope`
- `was_initially_available` — was the tool in the pre-sticky baseline?
- `was_cut` — was the tool present in the ceiling but removed by the
  narrower?
- `was_expanded` — did the tool get added via in-turn `expand_tools`?
- `expand_tools_used`, `expanded_tool`, `expand_category`,
  `turns_until_used`, `expansion_prediction_id` — populated when the
  call is a payoff of a prior expansion. Only credited when the tool
  was *not* initially available.
- `after_expand_tools`, `expand_resolved_tools`, `expand_tools_added` —
  context fields when an expansion happened in this turn.
- For `tool_name == "expand_tools"`: the row also carries `args` and
  `result` of the meta-tool call.

### `api_calls.jsonl`

One row per outbound API call. Written by
[`_on_post_api_request`](../__init__.py). Schema is intentionally
loose (free-form dict) — keys may evolve with the post-API hook
payload.

Key fields:

- `ts`, `prediction_id`, `session_id`, `scope`
- `api_call_idx` — index within the turn (1 = first call after user
  message, 2+ = follow-up calls within the same reasoning loop).
- `api_call_count` — Hermes' global call counter.
- `model`, `provider`, `api_mode`
- `tool_list_hash`, `system_hash` — fingerprints of the wire-level
  tool block and system message on this call.
- `input_tokens`, `output_tokens`, `cache_read_tokens`,
  `cache_write_tokens`, `reasoning_tokens`, `prompt_tokens`,
  `total_tokens`
- `api_duration`, `finish_reason`, `message_count`
- `cache_mode` — current detection state (`pending`, `on`, `off`).
- `cache_ttl_expired` — true when a stable-hash call returned zero
  `cache_read_tokens` after a >5 minute idle gap (provider-side TTL
  expiry, not a freeze regression).

Joining `tool_calls.jsonl` and `api_calls.jsonl` to `predictions.jsonl`
on `prediction_id` is the supported analysis path.

## Analyzing telemetry with `jq`

Quick recipes for spot-checks. For proper per-scope precision/recall and
promotion candidates, use [`analyze.py`](../analyze.py) — it does the
joins and applies the thresholds correctly.

**Watch live as messages flow in:**

```bash
tail -f ~/.hermes/state/tool-belt/predictions.jsonl | \
  jq -c '{scope, msg: .message_preview, triggers: .triggers_fired, suppressed: .triggers_suppressed, saved: .tokens_saved, pct: .reduction_pct}'
```

**Spot-check the last 5 predictions:**

```bash
tail -5 ~/.hermes/state/tool-belt/predictions.jsonl | \
  jq -c '{scope, msg: .message_preview[:50], triggers: .triggers_fired, ceiling: .ceiling_count, narrowed: .narrowed_count, saved: .tokens_saved, pct: .reduction_pct}'
```

**Aggregate savings — total and average:**

```bash
jq -s 'map(.tokens_saved) | {n: length, total_saved: add, avg_saved: (add/length | floor), avg_pct: (map(.reduction_pct) | add/length | (.*10|floor)/10)}' \
  ~/.hermes/state/tool-belt/predictions.jsonl
```

**Trigger frequency — which triggers fire most often:**

```bash
jq -r '.triggers_fired | if length == 0 then "(none)" else join(",") end' \
  ~/.hermes/state/tool-belt/predictions.jsonl | sort | uniq -c | sort -rn
```

**Trigger dampener audit — what got suppressed by `exclude_keywords`:**

```bash
jq -c 'select((.triggers_suppressed // []) | length > 0) | {msg: .message_preview[:60], suppressed: .triggers_suppressed, fired: .triggers_fired}' \
  ~/.hermes/state/tool-belt/predictions.jsonl | tail -20
```

**Tool calls per session — what actually got used:**

```bash
jq -r '.tool_name' ~/.hermes/state/tool-belt/tool_calls.jsonl | sort | uniq -c | sort -rn
```

**Expansion success rate — when did `expand_tools` lead to a real call?**

```bash
jq -c 'select(.expand_tools_used == true) | {scope, category: .expand_category, tool: .expanded_tool, turns_until_used}' \
  ~/.hermes/state/tool-belt/tool_calls.jsonl | tail -20
```

## Analyzer thresholds (`analyze.py`)

All thresholds are CLI flags, not hard-coded policy. The deterministic
telemetry pass reads `predictions.jsonl` + `tool_calls.jsonl`, prints a
summary, writes a dated markdown report under `reports/`, and can
optionally emit `learned_recommendations.json` for review
([`scripts/README.md`](../scripts/README.md) has the workflow).

| Flag | Default | Meaning |
|---|---|---|
| `--per-tool-tokens` | 388 | Estimated tokens per tool per API call (for carrying-cost math) |
| `--min-expansions` | 2 | Minimum expansion events before a category is eligible for a recommendation |
| `--promote-expand-rate` | 0.50 | Expand rate above which a category is a promotion candidate |
| `--promote-use-rate` | 0.80 | Downstream-use share above which a category is a promotion candidate |
| `--unused-carry-turns` | 10 | Always-on tools carried this many turns with zero calls become demotion candidates |
| `--trigger-min-fires` | 3 | Minimum fires before a trigger gets a precision/recall recommendation |
| `--expand-round-trip-tokens` | 1500 | Estimated total cost of one `expand_tools` round-trip (model output + result + extra API call). Folded into per-category net-value math and the headline net-savings number. |
| `--harvest-min-cuts` | 3 | Minimum was_cut count before a tool surfaces as a harvest-driven promotion candidate. |

Dampener/trigger-keyword suggestion flags (`--suggest-dampeners`,
`--suggest-trigger-keywords` and their tuning flags) are documented in
[`scripts/README.md`](../scripts/README.md).
