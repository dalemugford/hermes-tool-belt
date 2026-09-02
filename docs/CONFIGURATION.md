# CONFIGURATION

Reference for every configuration surface in `tool-belt`. For
behavior — what the plugin does at runtime, when freezes happen, how
the cache-mode state machine evolves — see
[ARCHITECTURE.md](ARCHITECTURE.md).

## Config layering

Configuration is assembled from three files with distinct ownership:

| File | Owner | Purpose |
|---|---|---|
| `policy.yaml` (plugin dir) | Plugin | Shipped preset. Always-carry baseline, trigger rules, shaper thresholds. |
| `config.yaml` (user) | User | Enable the plugin, set mode flags, add per-scope overrides. |
| `learned.json` (state dir) | shaper / auto-shape / configure | The learned carrying assignment per scope (schema v2). Generally not hand-edited. |

State files (`learned.json`, `cache_mode_detection.json`, JSONL
telemetry) live under `~/.hermes/state/tool-belt/` by default. When
`HERMES_HOME` is set (per-profile installs), all state paths resolve
relative to it — so per-profile state lands under
`$HERMES_HOME/state/tool-belt/`.

Resolution order at dispatch time (later layers override earlier):

1. `policy.yaml` — base preset (`always_carry`, `triggers`).
2. `config.yaml` pins — `always_carry` (global and additive per-scope
   `channels.<scope>.always_carry`).
3. `config.yaml` per-scope settings — `channels.<scope>.*`.
4. `learned.json` — applied under `learned_mode: apply` (the default);
   kept but not applied under `recommend`.

See [`presets.py:resolve_preset`](../presets.py) for the resolver.

---

## Managed configuration (`scripts/configure.py`)

Most users never edit `config.yaml` for this plugin.
[`scripts/configure.py`](../scripts/configure.py) writes the per-scope keys
for them, through `hermes config set` / `hermes config unset` — Hermes owns
`config.yaml`, so the file is never edited directly. Every write is shown as
a `before → after` line first and requires confirmation.

`configure` is a mode-setter: pick the agent, then either its
**Protected tools** (a picker whose selections are written to
`plugins.entries.tool-belt.settings.always_carry`) or **Tool shaping options** — channels,
then a mode. Non-interactively: `--agent <name> --mode learning|history|off`,
optionally `--channel <platform>` to target one channel (defaults to all).

It writes only ordinary documented settings — it introduces no key of
its own:

| Key | Written when | Value |
|---|---|---|
| `plugins.entries.tool-belt.settings.channels.<agent>.<platform>.learned_mode` | mode learning / history | `apply` |
| `plugins.entries.tool-belt.settings.channels.<agent>.<platform>.learned_mode` | mode off, reset | `recommend` |
| `plugins.entries.tool-belt.settings.channels.<agent>.<platform>.bypass_rate` | observation (legacy recommend path) | `1.0` (full observation) |
| `plugins.entries.tool-belt.settings.channels.<agent>.<platform>.bypass_rate` | mode history, on acceptance | `0.0` (narrow immediately) |
| `plugins.entries.tool-belt.settings.channels.<agent>.<platform>.bypass_rate` | reset | the value observation mode replaced, default `0.0` |
| `plugins.entries.tool-belt.settings.always_carry` | Protected-tools picker, on confirmation | the selected tool list |

Scope keys are the same `agent:platform` identifiers used by telemetry,
written on disk as `channels.<agent>.<platform>` — see [`channels`](#channels).

### Observation mode

The recommend path sets a scope's [`bypass_rate`](#bypass_rate) to `1.0`.
Every session then ships the untouched tool ceiling while telemetry still
records what the predictor *would* have selected — so a later shaping run has
real evidence without any narrowing having happened in the meantime. The
command reports how many more sessions each scope needs; that minimum is
derived from [`learning.shape_ceiling`](#learningshape_ceiling) in
`policy.yaml`, never hardcoded.

### Reset behavior

`tool-belt configure --mode off --agent <agent>` pauses shaping for an agent
(the learned overlay is kept but not applied — turning shaping back on
resumes where it left off). The deeper `configure.py --reset <agent>` returns
a shaped scope all the way to its pre-shaping state:

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

User config lives under `plugins.entries.tool-belt.settings` in Hermes'
`config.yaml` — the plugin-settings subtree Hermes reserves for every
plugin (`ctx.get_config`, `config_schema` validation and `hermes plugins`
all key off it). Per-channel entries nest as
`channels.<agent>.<platform>` because Hermes forbids `:` in a settings
key segment; the loader flattens them to the `agent:platform` scope
string every other file (telemetry, `learned.json`) uses. A block left at
the pre-2026.9 location `plugins.tool-belt` is **not read**; the plugin
logs one warning at load naming the move. The loader is
[`_load_user_config`](../__init__.py); defaults come from the
`_CONFIG` dict in the same file.

```yaml
plugins:
  entries:
    tool-belt:
      settings:
        enabled: true
        log: true
        agent: ""
        cache_mode: auto
        learned_mode: apply
        auto_shape: true
        auto_shape_interval_hours: 24
        always_carry: []
        bypass_rate: 0.0
        always_on_extra: []   # removed knob — warned about, never applied
        always_off: []        # removed knob — warned about, never applied
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
        channels:
          bernard:            # <agent>
            telegram:         # <platform>
              learned_mode: apply
```

### `enabled`

Type: `bool`. Default: `true`.

Shipped ON: a fresh install narrows and records telemetry from first
use (see [PRIVACY.md](PRIVACY.md)). Set `false` to opt out entirely.

Master switch. When `false`, `register()` short-circuits and no
monkey-patches are installed — the plugin is a no-op.

### `agent`

Type: `str`. Default: `""`.

Display name for this profile's agent in `configure --status` rows and the
savings report (e.g. `bernard` on a root profile whose directory identity
is `default`). `configure --agent` accepts either name. Empty means the
profile name is used.

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

### `auto_shape`

Type: `bool`. Default: `true`.

Runs the in-process shaping pass from the session-end hook, for every
scope whose `learned_mode` resolves to `apply` — the operator's standing
consent. Debounced per scope (stamps live in the `auto_shape_stamp.json`
sidecar); a pass that changes nothing writes nothing. Set `false` to
shape only via `tool-belt configure` or the shaper script.

### `auto_shape_interval_hours`

Type: `number`. Default: `24`. Also accepted per scope under
`channels.<scope>.auto_shape_interval_hours`.

Minimum hours between auto-shape attempts for a scope.

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

**When to pin — an honest note.** The shaper's economics optimize token
cost, and for most tools that's the whole story. But a tool's *presence in
the manifest is also an invitation*: the model reaches for tools it can
see. For proactive tools — ones the model should use on its own initiative
rather than when the user asks — demotion doesn't just add a fetch
round-trip, it trains the behavior away entirely. Low usage then looks
like evidence, and the shaping becomes self-confirming. Memory tools are
the canonical case: an agent whose `remember` is buried behind
`expand_tools` quietly stops forming memories, and nothing errors. If your
agent runs a memory plugin, pin its core write/read/correct surface (for
Mnemosyne: `mnemosyne_remember`, `mnemosyne_recall`, `mnemosyne_update`,
`mnemosyne_invalidate`) and let the specialist remainder ride the
evidence. The same reasoning applies to any tool your agent's own
instructions (SOUL/system prompt) tell it to use habitually: if your
directives insist on a tool, pin it.

### `bypass_rate` (internal testing feature)

Type: `float` in `[0.0, 1.0]`. Default: `0.0`. **Not surfaced by
`configure` — leave it at 0.0 unless you are validating Tool Belt.**

Fraction of sessions (deterministic by `sha1(scope|session_id)`) that
bypass narrowing and ship the full tool ceiling. Bypassed sessions
still run the predictor (so telemetry is preserved) but
`policy_source` is stamped `bypass` and `allowed_tool_names` widens to
the ceiling.

Why you don't need it: savings are measured *directly* — every narrowed
turn records `ceiling_tokens − narrowed_tokens`, so no control cohort is
required, and bypassed sessions are excluded from the savings headline
(they never narrow). A standing bypass fraction therefore only *spends*
tokens (each bypassed session ships the full manifest at full price)
without improving any measurement. Its legitimate uses are temporary:
validating prediction quality on a critical agent before trusting
shaping (set a small rate, watch whether bypassed sessions use tools
narrowing would have hidden, then set it back to 0.0), and A/B trials
during Tool Belt development. `1.0` effectively disables narrowing on a
scope without disabling the plugin — observation mode uses this
internally.

Per-scope override: `channels.<scope>.bypass_rate`. See
[`_bypass_rate_for_scope`](../__init__.py).

### `always_on_extra` / `always_off` (removed)

Type: `list[str]`. Default: `[]`. **No effect.**

Pre-1.0 knobs from before the full-start carrying model. They are still
parsed so that a value left in an old config produces one warning at
load naming the replacement, and are never applied. Use `always_carry`
to pin a tool; there is no disable list — under full-start a tool
leaves the carried set only through evidence-driven demotion, and
Hermes' own `platform_toolsets` is where a tool is removed entirely.

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

Per-channel overrides, nested on disk as `channels.<agent>.<platform>`
(Hermes forbids `:` in a settings key segment). The loader flattens them
to the `agent:platform` scope string telemetry and `learned.json` use.
Lookups try the full scope first, then fall back to a bare
`channels.<platform>` entry — so a `slack` entry still applies to
`default:slack` when no more specific entry exists. Scalar/list keys at
the agent level (`channels.<agent>.always_carry`) are that agent's
defaults for every platform; a platform entry wins on conflict.

An explicit plugin `agent` value remains the highest-precedence label. Named
profiles use their directory name, the root profile uses Hermes' reserved
`default` label, and calls without any profile context remain unattributed.

Each platform value mirrors the top-level shape and may set any of:

- `learned_mode`
- `bypass_rate`
- `always_carry` (additive — union with the global pins, never removes)
- `auto_shape_interval_hours`
- `always_on_extra`, `always_off` (removed knobs — warned about, never applied)

```yaml
channels:
  default:                 # agent (root profile)
    always_carry: [terminal]   # every platform of this agent
    telegram:              # platform
      learned_mode: recommend
    cli:
      auto_shape_interval_hours: 6
  slack:                   # bare platform: platform-wide fallback
    bypass_rate: 1.0
```

A bare-platform entry is recognised by its children being scalars/lists;
an entry whose children are all mappings is read as an agent.

---

## `policy.yaml` reference

The single shipped policy lives at the plugin root
([`policy.yaml`](../policy.yaml)) and is loaded by
[`presets.py:load_base_policy`](../presets.py). One file ships with
the plugin; for new shapes either edit it in place or fork it.

```yaml
name: aggressive
description: >
  Full-start with a structural always-carry baseline; demotions are
  evidence-driven; expand_only tools are trigger-gated.

learning:
  shape_ceiling:
    session_window: 100
    promote_min_sessions: 2
    promote_min_calls: 3
    demote_min_sessions_no_use: 20
    demote_k: 1.5

always_carry:
  - clarify
  - skill_view
  - skills_list
  - expand_tools
  - tool_search
  - tool_describe
  - tool_call

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
- `always_carry` (`list[str]`) — immutable residents: carried on every
  message and never shaped by the learned layer. (A legacy policy file's
  `always_on` list — or its `"*"` no-narrowing spelling — still loads,
  folded in at read time.)
- `triggers` (`list[dict]`) — trigger groups, evaluated against each
  inbound message. Multiple groups may fire on one message (additive).
- `learning.shape_ceiling` (`dict`) — default thresholds for the
  between-session shaper. See below.

Every enabled tool outside `always_carry` starts CARRIED (full-start);
evidence-driven demotion is the only way a tool moves to expand-only,
so tools the policy author didn't list are safe by construction.

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
| `session_window` | `100` | Ceiling on history the shaper reads — it uses all available sessions up to this many. `demote_min_sessions_no_use` is the floor: below that, demotion never fires. |
| `promote_min_sessions` | `2` | Anti-flap gate: a tool needs `expand_tools`-driven use across this many distinct sessions to be a promote candidate. |
| `promote_min_calls` | `3` | And this many total calls. Candidates then promote only when their observed expansion spend exceeds what carrying them would have cost. |
| `demote_min_sessions_no_use` | `20` | Window must contain this many sessions before any carried tool can be demoted. |
| `demote_k` | `1.5` | Economic safety factor: demote a carried tool only when carrying it costs more than k × what expanding it on demand would (token-denominated — no price table). |

### Authoring a new preset

The shipped loader points at the single `policy.yaml` at the plugin
root. To run a fork:

1. Copy `policy.yaml` to a new file.
2. Edit `name`, `always_carry`, `triggers`, and (optionally) `learning`.
3. Replace `policy.yaml` in place — the loader path is hard-coded to
   `Path(__file__).parent / "policy.yaml"` in
   [`presets.py`](../presets.py).

To disable narrowing on a scope without removing it, set
`channels.<agent>.<platform>.bypass_rate: 1.0` in `config.yaml` rather
than forking the preset.

---

## `learned.json` reference

Path: `$HERMES_HOME/state/tool-belt/learned.json` (or
`~/.hermes/state/tool-belt/learned.json` when `HERMES_HOME` is
unset). Written by the shaper (manual and session-end auto-shape),
the configure flows, and inventory reconciliation — all through
`learned.write_state`; consumed by
[`learned.py:apply_to_preset`](../learned.py) when
`learned_mode` is `apply` (the default).

```json
{
  "version": 2,
  "updated_at": "2026-08-01T00:00:00Z",
  "scopes": {
    "default:telegram": {
      "carry": ["browser_navigate"],
      "expand_only": ["image_generate"],
      "shaping": {
        "scope": "default:telegram",
        "computed_at": "2026-08-01T00:00:00Z",
        "sessions_considered": 30,
        "window_requested": 100,
        "source": "auto",
        "applied_at": "2026-08-01T00:00:00Z",
        "last_auto_shape_at": "2026-08-01T00:00:00Z",
        "enabled_tool_names": ["browser_navigate", "image_generate", "..."],
        "promote": [
          {"tool": "browser_navigate", "sessions": 4, "calls": 7,
           "carry_tokens": 9800, "expansion_tokens": 10500, "evidence": "expansion"}
        ],
        "demote": [
          {"tool": "image_generate", "sessions_without_use": 29, "sessions_with_use": 1,
           "uses_in_window": 1, "carry_tokens": 17385, "demote_tokens": 1500,
           "k": 1.5, "evidence": "carry_uneconomic"}
        ]
      }
    }
  }
}
```

### Schema

- `version` (`int`) — `2` (`LEARNED_VERSION` in
  [`learned.py`](../learned.py)). Documents declaring a NEWER version are
  refused on read and never rewritten. A v1 document (`always_on` /
  `always_off` / `cache_aware` keys) is normalized to this shape in
  memory on read — the file itself is only rewritten on the next real
  write.
- `updated_at` (`string`, ISO 8601 UTC) — last write timestamp.
- `scopes` (`dict[str, dict]`) — per-scope overlays, keyed by
  `{agent}:{platform}` (with bare-platform fallback).

Per scope:

- `carry` — learned promotions into adaptive residency.
- `expand_only` — evidence-driven demotions out of residency; the only
  way an enabled tool leaves the full-start carry set.
- `shaping` — the shaper's rationale block: `promote` / `demote` with
  per-tool economics (`carry_tokens` vs `expansion_tokens` /
  `demote_tokens`, `k`, `evidence` ∈ `expansion | carry_unused |
  carry_uneconomic`), `enabled_tool_names` (the ceiling the pass saw —
  also what `configure --status` counts "carried" against),
  `computed_at`, and the apply stamp (`source` `auto|configure`,
  `applied_at`, `last_auto_shape_at`). Human-readable; nothing at runtime
  branches on it.

> **Learned trigger overlay (schema v2, supersedes the above for triggers).**
> Under the 1.0 carrying model a scope's learned entry may carry a
> `triggers` list — an additive, auto-learned overlay of trigger groups
> (`{name, tools, keywords, exclude_keywords, source}`) that UNIONS with the
> shipped `policy.yaml` triggers at preset resolution; the shipped file is
> never edited, and v1 `trigger_adjustments` are ignored. Entries are
> written only by the session-end auto pass: keyword mining from expansion
> evidence auto-applies above a strict bar (support ≥ 4, precision ≥ 0.90),
> and a demotion with no trigger coverage mints a conservative name-token
> trigger. The overlay is activation-only by construction — it can only
> activate `expand_only` tools inside the enabled ceiling, never demote,
> disable, or touch `always_carry`. `configure --status` shows
> "auto-learned triggers: N" per scope; a scope reset clears the overlay;
> inventory reconciliation prunes entries for vanished tools. The runtime
> stays deterministic — overlay authorship is pure arithmetic, no model
> calls anywhere: "A plugin promising to save you tokens shouldn't quietly
> spend them."

### Global block

A top-level `global` block is preserved through reads and writes for
forward compatibility, but it is inert: no consumer reads it. Per-agent
blocklisting belongs in `config.yaml` scope settings or `policy.yaml`.

### Hand-editing

`learned.json` is rewritten atomically whenever a shaping pass actually
changes an assignment (a quiet auto pass writes nothing — its debounce
stamp lives in the `auto_shape_stamp.json` sidecar). Manual edits
survive until the next changing pass overwrites the shaper-owned fields
(`carry`, `expand_only`, `shaping`). Hand configuration belongs in
`config.yaml` (`always_carry` pins, `channels.<scope>.*`); the learned
file is for auto-tuned state.

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

- `ts`, `prediction_id`, `session_id` (the session *key*, constant per
  chat), `hermes_session_id` (Hermes' rotating session UUID — the
  shaper's distinct-session unit), `scope`, `agent`, `platform`, `channel`
- `message_hash`, `message_preview` — sha1 prefix + first 80 chars.
  Full text is never logged.
- `preset`, `policy_version`, `triggers_fired`, `triggers_suppressed`
- `ceiling_count` / `narrowed_count` and `ceiling_tokens` /
  `narrowed_tokens` / `tokens_saved` / `reduction_pct` — before/after
  the narrowing filter; `tokens_estimator` names the estimator.
- Carrying strata (the 1.0 carrying model): `always_carry_tools` (A) and
  `always_carry_count`, `carry_tools` (C) and `carry_count`,
  `expand_only_tools` (X = E − (A ∪ C)), `ceiling_tools` (E),
  `mcp_passthrough_tools` (outside the partition, never shaped).
- Activation this turn: `active_tools` (what the model saw: A ∪ C ∪ T ∪ R),
  `trigger_tools_by_group` and `trigger_activated_tools` (T),
  `expanded_tools` (R), `sticky_tools`, `sticky_categories`,
  `sticky_remaining_turns`.
- `policy_source` — `preset`, `learned`, or `bypass`.
- `learned_mode`, `learned_scope`, `learned_changes`
- `lookback_used`, `lookback_turns_config`
- `tool_list_hash` — sha256 prefix of the wire-level tool schemas.
- `provider`, `model`, `schema_version`
- `frozen_reuse`, `frozen_reuse_count` — true on cache-on dispatches
  after the first, with index within the session.

### `tool_calls.jsonl`

One row per actual tool call during a session. Written by
[`_on_post_tool_call`](../__init__.py).

Key fields:

- `ts`, `prediction_id`, `session_id`, `tool_name`, `schema_version`
- `agent`, `platform`, `scope`
- `source` — `gateway` (narrowing applied; counts toward savings),
  `cron`, or `subagent` (never narrowed; excluded — see
  [SAVINGS.md](SAVINGS.md#whats-excluded)).
- `was_initially_active` — was the tool in the turn's active set before
  any expansion?
- `was_expand_only` — was the tool assigned to the expand-only stratum
  (the shaper's demotion evidence reads this).
- `activated_by_expansion` — did an `expand_tools` call make this tool
  available? The promotion signal.
- `activation_source` — how the tool came to be active:
  `always_carry`, `carry`, `full_ceiling` (bypass / no narrowing), or
  `external` (expansion or trigger).
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
