# Hermes Tool Belt

### Adaptive tool loadouts for agents. Carry only what you need, drop the rest. Sharpens with use.

A plugin for [Hermes Agent](https://hermes-agent.nousresearch.com).

## Why

Every message ships tool definitions to the model. For a heavy profile —
web, terminal, file, browser, MCP servers, custom skills — that's
8–15k tokens of overhead on a message whose answer is `"yes."` Your
`"hi"` costs the same as `"deploy the build script"` in tool overhead.

The naive fix — re-narrow the tool list every turn — fights provider
prefix-caching: changing the tool block between turns busts the cache
and re-bills the conversation history at full input rate. On a
multi-turn session, lost cache costs more than schema savings.

Tool Belt resolves the tension. Three features, working together.

## 1. `expand_tools` — the safety valve that learns

A meta-tool the model calls when it needs something that wasn't loaded.
Narrowing never strands the agent: it asks, gets it, the conversation
continues. Every call is logged as evidence the tool was wanted —
between sessions, the shaper folds repeatedly-reached-for tools into
the next session's ceiling.

**The model's reach IS the promotion vote.** Your loadout grows toward
what you actually use, with no hand-tuning required.

```text
User: "search the docs for X and read me the relevant section"
  loadout:     14 tools (file, web, search)
  model calls: expand_tools(category="browser")
  result:      12 browser tools join the loadout
  → next session, browser is pre-loaded (shaper promoted it)
```

## 2. Deterministic shaping — rules over routing

The narrowing decision is a regex match on the message, an attachment
check, and a YAML lookup — **no LLM-as-router.** No extra API call,
no non-deterministic mis-route. Every decision is auditable in
`predictions.jsonl`.

Hand-tune triggers in `policy.yaml`, layer per-scope overrides in
`config.yaml`, and the telemetry feeds the analyzer back recommendations
— all the same shape. The YAML is the truth.

```yaml
# policy.yaml
always_on:
  - read_file
  - send_message
  - expand_tools
triggers:
  - name: shell
    tools: [terminal]
    keywords: ['\b(run|install|execute|deploy)\b']
    exclude_keywords: ['\b(should we|might|maybe)\b']
  - name: vision
    tools: [vision_analyze]
    has_attachment: image
```

## 3. Cache-aware adaptation — one goal, two cadences

Same predictor, same policy, same `expand_tools`. What changes is
**when** the narrowing happens.

**Cache-on session** (Anthropic, OpenAI auto-cache):

```text
User: "morning, what's on the agenda?"
  ceiling:    37 tools (~3,500 tok)
  frozen:     14 tools (~1,500 tok)   ← session-frozen — predictor ran once
  cache:      cold on turn 1

User: "summarize the inbox briefly"
  frozen:     14 tools                ← reused verbatim
  cache:      hot — ~95% prefix cache hit, only the new message billed fresh

User: "open https://example.com and tell me the title"
  model calls: expand_tools(category="browser")
  frozen:     14 + 12 browser tools   ← expanded, persists for rest of session
  cache:      one break this turn; subsequent turns reuse expanded set
```

**Cache-off session** (kimi, gpt-5.4-mini):

```text
User: "run the build script and check the logs"
  ceiling:    37 tools
  shipped:    14 tools                ← per-turn: always-on + `shell` trigger
  saved:      ~1,800 tok / message    (terminal+process loaded; browser cut)
```

Mode is **auto-detected per scope** from observed `cache_read` tokens
and locked. Anthropic / OpenAI auto-cache → cache-on by default.
Kimi / gpt-5.4-mini → cache-off. You set nothing.

| Shared in both modes | Cache-ON | Cache-OFF |
|---|---|---|
| Policy + always_on + triggers | Freeze at session start | Re-narrow each turn |
| `expand_tools` safety valve | Persists for the session | Sticky for N turns |
| Telemetry + savings math | (predictor skipped on reuse) | Lookback to prior message active |
| Bypass cohort (A/B baseline) | ✓ | ✓ |
| Strict `platform_toolsets` ceiling | ✓ | ✓ |
| Honest tokenizer (tiktoken when available) | ✓ | ✓ |

The headline: the same plugin saves tokens regardless of provider — and
picks the right cadence on its own.

## Three things that make this safe to install

- **Fails invisibly.** Any error — bad YAML, predictor exception, missing lookup — falls through to "no narrowing." Tool Belt can save you tokens; it can never break your agent.
- **Strict ceiling.** Can only *narrow* what your `platform_toolsets` config already allowed. Never adds a tool you didn't sanction.
- **Honest telemetry.** Every prediction logged with the savings math. Token counts via `tiktoken-cl100k` when installed, `chars/4` heuristic otherwise — every row records which estimator was used. Provider-billed tokens recorded directly from API responses. Anyone can verify your savings claims from the raw JSONL. See [docs/SAVINGS.md](docs/SAVINGS.md) for the methodology.

## Install

```bash
hermes plugins install dalemugford/hermes-tool-belt
```

Enable in `~/.hermes/config.yaml`:

```yaml
plugins:
  tool-belt:
    enabled: true
```

Restart the gateway. From the next message onward, the plugin starts
adapting tool payloads based on detected intent. Cache-aware mode is
chosen automatically per scope.

### Optional: real-tokenizer counts

For exact token counts in the savings report instead of the chars/4
heuristic:

```bash
pip install tiktoken
```

Auto-detected on next gateway restart; every row in `predictions.jsonl`
records which estimator was used (`tokens_estimator` field). The savings
delta is honest either way — both sides use the same estimator — but
absolute counts are more accurate with `tiktoken`. See
[docs/SAVINGS.md](docs/SAVINGS.md#which-tokenizer) for the full story.

### Day-one warm start (if you already use Hermes)

```bash
python3 ~/.hermes/plugins/tool-belt/scripts/bootstrap.py
```

Inspects your existing telemetry per profile. Under cache-on scopes,
runs the between-session shaper (`scripts/shape-ceiling.py`) to
identify per-tool promote/demote candidates from real `expand_tools`
evidence. Under cache-off scopes, runs harvest-replay + analyzer to
mine trigger-keyword tweaks. Either path produces a ranked **TOP
ACTIONS** summary tailored to your usage.

---

## Releases

This branch (`main`) is the trunk; everything ships from here. Adaptive
features that aren't ready to be enabled by default are gated behind
config flags (`enabled: false`, `learned_mode: off`, etc.) so an
install of `main` is always safe.

Release tags use CalVer (`YYYY.M.D`). For stability, pin a tag. For
bleeding-edge, track `main`. Current release: **2026.5.17-beta**.

## How it stays honest

- **Append-only telemetry** (`predictions.jsonl` + `tool_calls.jsonl`)
  — every narrowing decision is recorded; tune from real numbers, not
  vibes.
- **A/B baseline cohort** — set `bypass_rate: 0.05` and a deterministic
  5% of sessions ship the full toolset, so headline savings get a real
  control to compare against.
## Companion docs

The code is the source of truth. The active doc set is intentionally small:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the plugin works: the two modes, the freeze mechanism, `expand_tools`, between-session shaping, where in the request lifecycle the patches sit.
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) — every knob in `config.yaml`, `policy.yaml`, and `learned.json`, with defaults.
- [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) — behaviors that look like bugs but aren't (Codex reasoning cache, gateway-restart freeze loss).

## Policy

The plugin ships **one hand-curated policy** at
[policy.yaml](policy.yaml). It targets messaging-style scopes
(Telegram, Slack, Discord) — minimum always-on tools, everything else
trigger-gated. Edit the YAML directly to customize, or layer per-scope
overrides via `always_on_extra` / `always_off` / `channels.<scope>.*` in
config.

To **disable narrowing on a specific scope**, set `bypass_rate: 1.0`
for that scope — see
[A/B baseline cohort](#ab-baseline-cohort) below. This routes the scope
through the predictor for telemetry but ships the full tool ceiling to
the model.

## Per-channel / per-scope override

The plugin keys policy on `{agent}:{platform}` (e.g. `bernard:telegram`,
`sue:slack`). Lookups try the full scope first, then fall back to the
bare platform — so an override on `bernard:telegram` wins over a generic
`telegram` entry.

```yaml
plugins:
  tool-belt:
    enabled: true
    channels:
      cli:
        always_on_extra: [terminal, execute_code]
      bernard:telegram:           # agent-scoped override
        learned_mode: recommend
        always_on_extra: [terminal]
      slack:                       # platform-wide fallback
        bypass_rate: 1.0          # disable narrowing on Slack entirely
```

## How it works

The plugin runs in one of two modes per scope. The mode is detected
automatically from observed cache behavior over the first few API
calls of a session, then locked for the session's lifetime. The
locked decision persists across sessions in
`~/.hermes/state/tool-belt/cache_mode_detection.json` so subsequent
sessions skip the observation window.

### Cache-on mode (default for Anthropic, OpenAI auto-cache)

1. **First dispatch in a session** — `pre_gateway_dispatch` runs the
   predictor ([predictor.py](predictor.py)) against the active policy
   ([policy.yaml](policy.yaml) → per-scope overrides → learned overlay
   via [presets.py](presets.py) and [learned.py](learned.py)), then
   *freezes* the resulting tool set into a per-session snapshot
   keyed by canonical session key.

2. **Every subsequent dispatch in the same session** reuses the frozen
   snapshot verbatim. The predictor doesn't run. Sticky residency and
   lookback are disabled — the frozen set carries forward in-session
   expansions, making per-turn carry-over redundant.

3. **`expand_tools` is the safety valve.** When the model calls
   `expand_tools(category="browser")` mid-session, the resolved tools
   are added to the frozen snapshot in-place and stay available for
   the rest of the session. This trades one cache break (model-driven,
   for a tool the model demonstrably needed) for permanent access to
   that tool group within the session.

4. **The freeze is evicted only at moments that already cost a cache
   break**: true `/reset` or `/new`, end of session, and compaction
   (the plugin patches `AIAgent._compress_context` to re-freeze
   against the post-compaction state).

5. **Between sessions**, [`scripts/shape-ceiling.py`](scripts/shape-ceiling.py)
   reads accumulated `expand_tools` evidence and writes per-tool
   promote/demote recommendations into `learned.json`. Tools the model
   reached for repeatedly join the next session's frozen ceiling;
   always-on tools that went unused for many sessions get demoted.

### Cache-off mode (kimi, gpt-5.4-mini, anything provider-side caching doesn't reach)

The original per-turn pipeline. On every dispatch:

1. **`pre_gateway_dispatch`** runs the predictor on message text +
   attachments + the last few user messages (lookback), produces an
   allowed set, merges with sticky residency from prior expansions.

2. **Patched `_build_api_kwargs`** filters `kwargs["tools"]` down to
   `(always_on ∪ triggered ∪ expanded ∪ sticky) ∩ ceiling` on every
   outbound API call.

3. **`expand_tools`** mid-turn adds the category to the contextvar's
   expansions set. Successful expansions refresh sticky residency so
   the category stays available for `ttl_turns` further turns without
   re-expanding.

### Shared pieces (both modes)

- **`post_tool_call` hook** logs every actual tool call linked by
  `prediction_id`, with flags for whether the tool was initially
  available, expanded, sticky-carried, etc. This is what makes the
  analyzer's precision and expansion-success math possible.
- **`post_api_request` hook** captures per-call cache_read /
  cache_write tokens and a hash of the tool block, so cache economics
  can be measured directly rather than inferred.
- **`self.tools` on the AIAgent is left untouched.** The plugin only
  narrows the on-the-wire payload via the `_build_api_kwargs` patch;
  agent state stays canonical.

## Configuration reference

```yaml
plugins:
  tool-belt:
    enabled: true
    log: true

    # Global tweaks layered on top of policy.yaml
    always_on_extra: []
    always_off: []

    # Cache-off mode pipeline — inert under cache-on (the default).
    # Both sticky residency and predictor lookback only apply when the
    # session resolves to cache_mode: off (kimi, gpt-5.4-mini, etc.).
    cache_off:
      sticky:
        enabled: true
        ttl_turns: 3
        # Toolset names passed to expand_tools(category=...). NOT YAML
        # trigger group names. Empty / "*" / null disables filtering.
        categories: [terminal, file, browser, web, code_execution, delegation]
      predictor:
        # Concat last N user messages from the same session into the
        # regex input. Catches signal in short replies like "yes",
        # "go ahead", "no don't". Set 0 to disable.
        lookback_turns: 1

    # A/B baseline cohort — deterministically bypass narrowing for a
    # fraction of sessions so the analyzer can compare narrowed vs
    # unnarrowed cohorts. Set 1.0 on a scope to fully disable narrowing
    # there. Deterministic via
    # hash(scope|session_id).
    bypass_rate: 0.0

    # Adaptive / learned layer (see "Learned policy" below)
    learned_mode: off          # off | recommend | auto | audit

    # Per-scope overrides
    channels:
      bernard:telegram:        # agent-scoped override
        learned_mode: recommend
        bypass_rate: 0.05      # 5% baseline cohort on this scope only
        always_on_extra: [terminal]
      slack:                    # platform-wide fallback
        bypass_rate: 1.0       # disable narrowing on Slack entirely
```

## Telemetry

When `log: true`, two append-only JSONL files at `~/.hermes/state/tool-belt/`:

- **predictions.jsonl** — one row per inbound gateway message:
  `prediction_id`, `session_id`, `agent`, `platform`, `scope`,
  `message_hash` + 80-char preview, `preset`, `triggers_fired`,
  `triggers_suppressed`, `ceiling_count` + `ceiling_tokens`,
  `narrowed_count` + `narrowed_tokens`, `tokens_saved`, `reduction_pct`,
  full `ceiling_tools` / `allowed_tools` / `cut_tools` / `unknown_kept_tools`,
  `always_on_tools`, `trigger_tools_by_group`, `expanded_tools`,
  `sticky_tools` + `sticky_categories` + `sticky_remaining_turns`,
  `policy_source` (preset | learned | bypass), `policy_version`,
  `learned_mode`, `learned_scope`, `learned_changes`, `lookback_used`,
  `lookback_turns_config`. Full message text is never logged.

- **tool_calls.jsonl** — one row per actual tool call linked by
  `prediction_id`: `tool_name`, plus flags `was_initially_available`,
  `was_cut`, `was_expanded`, `after_expand_tools`, `expand_category`,
  `expand_tools_used`, `expanded_tool`, `turns_until_used`,
  `expansion_prediction_id`, `sticky_remaining_turns`. For `expand_tools`
  rows: the full `expand_event` payload (resolved tools, sticky refresh
  result).

### Viewing savings — quick `jq` recipes

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

Use [analyze.py](analyze.py) (below) for proper per-scope precision/recall
and promotion candidates — it does the join correctly and applies the
adjustable thresholds.

## Trigger dampeners (`exclude_keywords`)

Positive regex matches can still be vetoed by an `exclude_keywords` list on
the same trigger group. This handles the most common false positives —
discussion/meta phrasing like "should we", "talk through", "don't write" —
without growing the regex DSL into a full scoring framework.

Currently enabled on `file_write`, `delegation`, and `cronjob` in
[policy.yaml](policy.yaml). When an exclude vetoes
an otherwise-firing trigger, the group name is recorded in
`predictions.jsonl` under `triggers_suppressed` so you can audit dampener
effectiveness separately from "no positive match."

Regression test:

```bash
cd ~/.hermes/plugins/tool-belt
~/.hermes/hermes-agent/venv/bin/python3 -m unittest tests.test_trigger_dampeners -v
```

### Auto-suggesting new dampeners from telemetry

Dampeners are hand-written today, but the analyzer can mine candidate
patterns from telemetry. When a trigger fires (positive match) but no
matching tool ever gets called, that message preview is a false-positive
data point. The analyzer collects 80-char previews per (scope, trigger),
extracts 2–4 word n-grams, and surfaces n-grams that show up frequently
in false-positive messages but rarely (or never) in true-positive ones.

```bash
hermes-agent/venv/bin/python3 plugins/tool-belt/analyze.py \
  --suggest-dampeners
```

The output includes regex-ready patterns plus a few sample previews per
candidate so you can sanity-check before pasting into the YAML. Candidates
already covered by an existing `exclude_keywords` pattern are filtered out
automatically — no point suggesting something already vetoed.

Tuning flags:

| Flag | Default | Meaning |
|---|---|---|
| `--dampener-min-support` | 3 | Minimum false-positive occurrences for a candidate |
| `--dampener-min-precision` | 0.8 | Minimum `fp / (fp + tp)` ratio |
| `--dampener-min-n` | 2 | Shortest n-gram length (words) |
| `--dampener-max-n` | 4 | Longest n-gram length (words) |
| `--dampener-max-candidates` | 10 | Cap on suggestions per (scope, trigger) |

This closes the loop: telemetry → false-positive mining → human-reviewed
candidate → tighter YAML → fewer false positives next batch.

## Learned policy (`learned_mode`)

Beyond the static preset, the plugin supports a persistent per-scope overlay
in `~/.hermes/state/tool-belt/learned.json`. Shape:

```json
{
  "version": 1,
  "updated_at": "...",
  "scopes": {
    "bernard:telegram": {
      "always_on": ["terminal"],
      "always_off": [],
      "trigger_adjustments": {
        "shell": {"action": "demote", "reason": "78% false-positive rate"}
      },
      "promotions": { "...": "audit trail" }
    }
  },
  "global": { "always_off": ["homeassistant"] }
}
```

`learned_mode` controls how the file is consumed:

| Mode | Behavior |
|---|---|
| `off` (default) | `learned.json` is ignored entirely. Safe default. |
| `recommend` | Plugin does **not** apply `learned.json`. Use [analyze.py](analyze.py) with `--write-recommendations` to emit `learned_recommendations.json` for human review. |
| `auto` | Plugin applies `always_on` / `always_off` / `trigger_adjustments` from `learned.json` during preset resolution. |
| `audit` | Same merge as `auto`. Each prediction row stamps `learned_mode: audit` so audit-mode runs can be greppped or aggregated separately from `auto`. No additional per-decision flag today. |

Resolution order, late-binding wins:

```
User ceiling (platform_toolsets)
  → Base policy (policy.yaml)
  → Global always_on_extra / always_off
  → Per-scope channels.<scope> overrides
  → Learned overlay (only when learned_mode ∈ {auto, audit})
  → expand_tools admissions (per-turn)
  → Sticky residency (per-turn, in-memory)
  → A/B bypass (when bypass_rate fires for this session)
```

`learned.json` is read-only at prediction time. The plugin never writes it.
Writing is reserved for the analyzer or explicit human action — keep that
boundary so promotions remain reviewable.

## Analyzer (`analyze.py`)

Deterministic telemetry pass. Reads `predictions.jsonl` + `tool_calls.jsonl`,
prints a summary, writes a dated markdown report under `reports/`, and can
optionally emit `learned_recommendations.json` for review.

```bash
hermes-agent/venv/bin/python3 plugins/tool-belt/analyze.py
hermes-agent/venv/bin/python3 plugins/tool-belt/analyze.py --format json
hermes-agent/venv/bin/python3 plugins/tool-belt/analyze.py --write-recommendations
hermes-agent/venv/bin/python3 plugins/tool-belt/analyze.py --include-archives
```

All thresholds are CLI flags, not hard-coded policy:

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

The analyzer does **not** write `learned.json`. It only ever writes the
markdown report and (with `--write-recommendations`) the recommendations
JSON. Auto-apply is deliberately a separate, future step.

Per-category recommendations now include net-value math: `saved_round_trip_tokens`
(`expansions × expand_round_trip_tokens`) minus `added_carry_tokens`
(`predictions × tools_in_category × per_tool_tokens`). When a category
clears the expand-rate/use-rate thresholds but has negative net value
under the current defaults, the action becomes `review_net_negative`
instead of `promote_candidate`.

### Harvest-aware mode

When the state dir contains `policy_source: harvest` rows (produced by
[scripts/harvest-replay.py](scripts/harvest-replay.py)), the analyzer
auto-activates two additional recommendation flows:

**Per-tool promotion candidates** (`kind: harvest_tool_promotion`):
direct was_cut-based ranking. Each historical tool call that current
narrowing would have cut represents a counterfactual `expand_tools`
round-trip. The math:

    net_savings = cuts × expand_round_trip_tokens
                − predictions × per_tool_tokens

| Action | Trigger |
|---|---|
| `promote_always_on` | net positive — tool should join always-on |
| `broaden_trigger_recall` | net negative but cut_rate ≥ 10% — fix the trigger keywords, don't pay per-turn carry |
| `keep_gated` | sparse cuts; not worth either action |

**Trigger-keyword candidates** (`--suggest-trigger-keywords`):
symmetric inverse of `--suggest-dampeners`. Mines was_cut message
previews for n-grams that would have fired a trigger to cover the
tool. For each candidate, decides whether to suggest `add_keywords_to_trigger`
(an existing group already claims the tool) or `create_new_trigger`
(no group claims it yet — proposes a new group name from the tool's
underscore prefix).

```bash
# Generate the harvest data, then run the analyzer against it
python3 scripts/harvest-replay.py
python3 analyze.py --state-dir ~/.hermes/state/tool-belt/harvest \
    --suggest-trigger-keywords --suggest-dampeners
```

Or use [scripts/bootstrap.py](scripts/bootstrap.py) to do all of this
in one shot with a ranked TOP ACTIONS summary.

## A/B baseline cohort

Headline savings ("57% reduction") are tautological without a control. Set
`bypass_rate > 0` globally or per-scope to deterministically send a
fraction of sessions through with no narrowing:

```yaml
plugins:
  tool-belt:
    bypass_rate: 0.05   # 5% global baseline
    channels:
      bernard:telegram:
        bypass_rate: 0.1
```

Bypass is sticky per session — the same `session_id` is always in or out,
so a session's behavior stays internally consistent. Bypassed turns still
run the predictor for telemetry but stamp `policy_source: "bypass"` so the
analyzer's `cohorts` payload can compare:

```bash
hermes-agent/venv/bin/python3 plugins/tool-belt/analyze.py --format json \
  | jq '.scopes["bernard:telegram"].cohorts'
```

Each cohort reports `predictions`, `sessions`, `tool_calls`,
`tool_calls_per_prediction`, `expand_tools_events`, and `expand_rate`.
First slice is roll-ups only; per-session latency/turn-count deltas are
future work.

## Predictor lookback

The predictor optionally prepends the last N user messages from the same
session before regex matching. This recovers intent for short replies like
"yes" / "go ahead" / "no, don't" whose signal lives in the prior turn.

```yaml
plugins:
  tool-belt:
    predictor:
      lookback_turns: 1   # 0 disables
```

Prior turns are tagged with `[prior]:` in the composed input so dampener
regexes anchor cleanly. `exclude_keywords` dampeners still apply across
the composite — if the prior turn was conversational ("should we update
the README later?") and the current is "sure", the existing dampener for
`should we` vetoes the trigger and stamps `triggers_suppressed` as before.

Telemetry: each prediction row gets `lookback_used` (count of prior
messages actually concatenated this turn — zero on session-first turns)
and `lookback_turns_config`.

## Sticky residency

When `expand_tools` succeeds, the resolved tools stay admitted for the
scope for `sticky.ttl_turns` further inbound gateway turns (default: 3).
This avoids the model paying for an extra `expand_tools` round-trip on
every turn during an active multi-turn task. Sticky state is in-memory
only — lost on gateway restart, by design. Repeated sticky refreshes feed
the analyzer's promotion signal.

Sticky decrements happen once per inbound user turn (in
`pre_gateway_dispatch`), not per LLM API call inside a tool loop.

## Verifying it works

```bash
# After restarting Hermes:
tail -f ~/.hermes/state/tool-belt/predictions.jsonl
# Send a message in any gateway channel. You should see a row appear with
# the triggers that fired and the counts of tools loaded.

# Or check Hermes' own logs:
grep "tool-belt" ~/.hermes/logs/agent.log
# Should show "patches installed" at startup and "active (policy=policy.yaml, ...)"
```

## Editing the policy

The policy YAML lives at [policy.yaml](policy.yaml) inside the plugin.
To customize:

1. Edit the YAML directly. `keywords` and `exclude_keywords` are
   case-insensitive Python regexes.
2. Restart Hermes — policy.yaml is parsed each time the predictor runs,
   so changes take effect on the next turn.

The shipped policy targets messaging-style scopes. Significant
deviations (CLI power-user shape, MCP-heavy profile, etc.) are better
expressed as per-scope `always_on_extra` / `always_off` overrides in
your config than as edits to this file — that keeps your customizations
out of the way of plugin updates.

## Limitations and known sharp edges

- **CLI sessions don't fire `pre_gateway_dispatch`.** CLI messages bypass
  the gateway, so neither narrowing nor freeze applies. No telemetry written.
- **Subagents are not narrowed.** When `delegate_task` spawns a subagent,
  the parent's tool curation is inherited; the predictor doesn't run again.
- **Cron jobs are not narrowed.** Cron sessions get explicit per-job
  toolsets via Hermes' own `_resolve_cron_enabled_toolsets`.
- **Unknown plugin tools default to always-on.** Tools whose names aren't
  in the policy's known sets are kept on (safer for shipping new plugins).
- **In-memory session state doesn't survive gateway restarts.** Both the
  cache-on frozen snapshot and cache-off sticky residency live in process
  memory. A restart between turns will re-freeze (cache-on) or clear
  sticky carry-over (cache-off). The cross-session detection cache and
  `learned.json` are on disk; only the per-session memory is volatile.
- **Codex (gpt-5.4) reasoning trace breaks cache mid-session.** When the
  model produces reasoning tokens, the resulting message-history mutation
  busts cache on the next call regardless of what the plugin does.
  Anthropic-side cache is more deterministic. See
  [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).
- **`learned_mode: auto`/`audit` is not the default.** Adaptive promotion
  is deliberately opt-in until recommendations have been reviewed for a
  given scope. Start with `recommend`.

## Failure modes

By design, **anything that goes wrong falls through to no-narrowing**:

- Bad YAML syntax in policy.yaml → wildcard fallback (no narrowing)
- policy.yaml missing → wildcard fallback (no narrowing)
- Predictor exception → returns wildcard (no narrowing)
- Filter exception in patched `_build_api_kwargs` → returns unfiltered tools
- Learned-state load error → base policy returned as-is, plugin keeps running
- `agent` not in registry when `expand_tools` is called → returns honest
  error JSON to the model

The plugin should be invisible when broken, never destructive.

## Disabling

Either:

```yaml
plugins:
  tool-belt:
    enabled: false
```

…or remove the directory entirely:

```bash
rm -rf ~/.hermes/plugins/tool-belt
```

## Files

```
~/.hermes/plugins/tool-belt/
├── plugin.yaml       # Hermes plugin manifest
├── __init__.py       # runtime hooks, cache-mode detection, frozen loadouts
├── predictor.py      # deterministic intent matching
├── presets.py        # policy loading and per-scope resolution
├── learned.py        # learned overlay loading and merging
├── expand_tools.py   # dynamic recovery meta-tool
├── logger_io.py      # telemetry writers
├── analyze.py        # telemetry analyzer
├── policy.yaml       # shipped policy and shaping thresholds
├── AGENTS.md         # repository engineering rules
├── docs/             # architecture, configuration, savings, known issues
├── scripts/          # supported operator and verification commands
└── tests/            # regression and integration tests
```