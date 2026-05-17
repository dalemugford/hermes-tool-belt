# dynamic-tools

A [Hermes Agent](https://hermes-agent.nousresearch.com) plugin that
narrows the tools sent to the model on each message — so the prompt
budget pays only for the tools the current turn actually needs.

## The problem

Every inbound message ships **every** tool definition the agent has.
For a heavy profile (web, terminal, file, browser, MCP servers, custom
skills) that can be 8–15k tokens of overhead on a message whose answer
is "yes." Most of those tools aren't relevant to what's being asked
right now — but the model still pays to read them.

## What it does

On every inbound gateway message, the plugin:

1. **Reads the message** (and recent context, if lookback is enabled).
2. **Predicts** which tool categories the agent will need this turn
   using a configurable regex/keyword classifier.
3. **Filters** the outbound API call's `tools` payload down to
   `(always_on ∪ triggered ∪ expanded ∪ sticky) ∩ ceiling`. Your
   existing `platform_toolsets` config remains a hard ceiling — the
   plugin never *adds* tools you didn't sanction.
4. **Hands the model an escape hatch.** An `expand_tools` meta-tool is
   always loaded. If the predictor guesses wrong, the model calls
   `expand_tools(category="browser")` (or any category) and the next
   API call unions the resolved tools into the allowed set.
5. **Logs everything** to append-only JSONL so you can audit and tune
   the policy with real numbers, not vibes.

Any failure — bad YAML, predictor exception, lookup miss — falls
through to **no narrowing**. The plugin is invisible when broken, never
destructive.

## Measured impact

Telemetry on Bernard's Telegram scope (pre-2026-05-17 audit reset):
**3.37M tokens saved across 109 sessions, ~57% reduction vs.
baseline**, at an estimated **~388 tokens per tool per API call**.

> The 2026-05-17 audit found two attribution bugs in the analyzer
> (`expand_tools_used` was 94% over-credited; trigger FPs were ~32%
> under-counted by missing late-bound TPs) and reset live telemetry
> for clean post-fix measurement. The savings figure above is
> unaffected — token math doesn't depend on the buggy fields — but
> per-trigger precision/recall numbers from before the fix should be
> treated as approximate.
>
> Until clean post-fix data accumulates, the harvest-based view
> ([scripts/bootstrap.py](scripts/bootstrap.py)) is the most reliable
> source of per-profile auto-apply recommendations.

## Releases

This branch (`main`) is the trunk; everything ships from here. Lab
features that aren't ready to be enabled by default are gated behind
config flags (`enabled: false`, `learned_mode: off`, etc.) so an
install of `main` is always safe.

Release tags (`vX.Y.Z`) mark known-good states. For stability, pin a
tag. For bleeding-edge, track `main`.

Companion docs:
- **[STRATEGY.md](STRATEGY.md)** — the north star: the promise, release strategy, calibration flow, official-plugin readiness. **Start here** if you want to understand where this plugin is going.
- [PLAN.md](PLAN.md) — current architecture, what's built vs. planned, validation status
- [CALIBRATION-PLAN.md](CALIBRATION-PLAN.md) — design for the observe → handoff → steady-state flow that makes "install, activate, done" honest
- [AUTO-APPLY-PLAN.md](AUTO-APPLY-PLAN.md) — the writer-side script that closes the loop (telemetry-correctness audit cleared mechanically; see [docs/telemetry-audit-2026-05-17.md](docs/telemetry-audit-2026-05-17.md))
- [TRIGGER-SCORING-PLAN.md](TRIGGER-SCORING-PLAN.md) — historical full-scoring plan; current direction is the lighter `exclude_keywords` dampener slice
- [docs/telemetry-audit-2026-05-17.md](docs/telemetry-audit-2026-05-17.md) — first-pass telemetry-correctness audit (2 bugs found + fixed, 2 items clean, 3 items pending data accumulation)
- [docs/harvest-followups.md](docs/harvest-followups.md) — open auto-apply candidates surfaced by harvest; resolve by edit or via the trigger-keyword suggester
- [docs/dynamic-tools-hermes-surface.md](docs/dynamic-tools-hermes-surface.md) — patch-point reference; read before any Hermes upgrade
- [docs/dynamic-tools-plan.md](docs/dynamic-tools-plan.md) — original design doc and categorization audit

## Quick start

The plugin is **off by default**. To enable globally, edit
`~/.hermes/config.yaml`:

```yaml
plugins:
  dynamic-tools:
    enabled: true
    log: true                 # write per-prediction telemetry
    learned_mode: off         # off | recommend | auto | audit (see below)
```

Restart Hermes (or the gateway) so the plugin's `register()` runs. From the
next message onward, tool definitions are narrowed per-message. The model
always gets `expand_tools` as recourse if it needs a category that wasn't
loaded.

### Day-one recommendations (warm start)

If you have any existing Hermes sessions under `~/.hermes/sessions/`,
run the bootstrap script once after install to get concrete promotion
candidates and trigger-keyword suggestions tailored to your own usage
— no need to wait for live telemetry to accumulate:

```bash
python3 ~/.hermes/plugins/dynamic-tools/scripts/bootstrap.py
```

The script replays your session history through the predictor, runs
the analyzer in harvest-aware mode, and prints a ranked **TOP ACTIONS**
summary. See [scripts/README.md](scripts/README.md) for details.

## Policy

The plugin ships **one hand-curated policy** at
[policy.yaml](policy.yaml). It targets messaging-style scopes
(Telegram, Slack, Discord) — minimum always-on tools, everything else
trigger-gated. Edit the YAML directly to customize, or layer per-scope
overrides via `always_on_extra` / `always_off` / `channels.<scope>.*` in
config.

To **disable narrowing on a specific scope** (the old `conservative`
mode), set `bypass_rate: 1.0` for that scope — see
[A/B baseline cohort](#ab-baseline-cohort) below. This routes the scope
through the predictor for telemetry but ships the full tool ceiling to
the model.

> A future "calibration mode" (see [CALIBRATION-PLAN.md](CALIBRATION-PLAN.md))
> will replace the hand-curated policy with one learned from each
> profile's own usage. The shipped YAML is the warm-start seed for that
> flow, not the destination.

## Per-channel / per-scope override

The plugin keys policy on `{agent}:{platform}` (e.g. `bernard:telegram`,
`sue:slack`). Lookups try the full scope first, then fall back to the
bare platform — so an override on `bernard:telegram` wins over a generic
`telegram` entry.

```yaml
plugins:
  dynamic-tools:
    enabled: true
    channels:
      cli:
        always_on_extra: [terminal, execute_code]
      bernard:telegram:           # agent-scoped override
        learned_mode: recommend
        always_on_extra: [terminal]
      slack:                       # legacy platform-only key still works
        bypass_rate: 1.0          # disable narrowing on Slack entirely
```

## How it works

1. **`pre_gateway_dispatch` hook** runs on every inbound gateway message. It
   reads the message text + attachments, runs a regex/keyword classifier
   ([predictor.py](predictor.py)) against the active policy (loaded from
   [policy.yaml](policy.yaml) and resolved through the global → per-scope →
   learned overlay chain in [presets.py](presets.py) and
   [learned.py](learned.py)), and stashes the prediction in a
   `contextvars.ContextVar`.

2. **Monkey-patched `AIAgent._build_api_kwargs`** ([__init__.py](__init__.py))
   runs once per outbound API call. It reads the contextvar and filters
   `kwargs["tools"]` down to `(always_on ∪ triggered ∪ expanded ∪ sticky) ∩ ceiling`.
   This is the right hook because Hermes caches `AIAgent` instances per
   session for up to an hour; patching `__init__` would only narrow the first
   message. `self.tools` is left as the full ceiling — only the per-request
   payload is narrowed.

3. **`expand_tools` meta-tool** is always loaded. The model can call it
   (`expand_tools(category="browser")`) when it needs something gated. The
   handler appends the category to the contextvar's `expansions` set; the
   next `_build_api_kwargs` call unions the resolved tools into the allowed
   set. Expansions also refresh in-memory **sticky residency** for the active
   session via a private sticky key, so a category stays available for
   `ttl_turns` further turns without re-expanding.

4. **`post_tool_call` hook** logs every actual tool call alongside the
   prediction id — including whether the tool came from the initial allowed
   set, from `expand_tools`, or via sticky residency. This is what lets the
   analyzer measure precision (did the trigger fire and the tool actually
   get used?) and expansion success rate (did the expansion lead to a real
   tool call?).

## Configuration reference

```yaml
plugins:
  dynamic-tools:
    enabled: true
    log: true

    # Global tweaks layered on top of policy.yaml
    always_on_extra: []
    always_off: []

    # Sticky residency (in-memory only, per active session)
    sticky:
      enabled: true
      ttl_turns: 3
      # Toolset names passed to expand_tools(category=...). NOT YAML trigger
      # group names. Empty list / "*" / null disables filtering.
      categories: [terminal, file, browser, web, code_execution, delegation]

    # Predictor lookback — concat last N user messages from the same
    # session into the regex input. Catches signal in short replies like
    # "yes", "go ahead", "no don't". Set 0 to disable.
    predictor:
      lookback_turns: 1

    # A/B baseline cohort — deterministically bypass narrowing for a
    # fraction of sessions so the analyzer can compare narrowed vs
    # unnarrowed cohorts. Set 1.0 on a scope to fully disable narrowing
    # there (the old `conservative mode`). Deterministic via
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
      slack:                    # legacy platform-only key still works
        bypass_rate: 1.0       # disable narrowing on Slack entirely
```

## Telemetry

When `log: true`, two append-only JSONL files at `~/.hermes/state/dynamic-tools/`:

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
tail -f ~/.hermes/state/dynamic-tools/predictions.jsonl | \
  jq -c '{scope, msg: .message_preview, triggers: .triggers_fired, suppressed: .triggers_suppressed, saved: .tokens_saved, pct: .reduction_pct}'
```

**Spot-check the last 5 predictions:**

```bash
tail -5 ~/.hermes/state/dynamic-tools/predictions.jsonl | \
  jq -c '{scope, msg: .message_preview[:50], triggers: .triggers_fired, ceiling: .ceiling_count, narrowed: .narrowed_count, saved: .tokens_saved, pct: .reduction_pct}'
```

**Aggregate savings — total and average:**

```bash
jq -s 'map(.tokens_saved) | {n: length, total_saved: add, avg_saved: (add/length | floor), avg_pct: (map(.reduction_pct) | add/length | (.*10|floor)/10)}' \
  ~/.hermes/state/dynamic-tools/predictions.jsonl
```

**Trigger frequency — which triggers fire most often:**

```bash
jq -r '.triggers_fired | if length == 0 then "(none)" else join(",") end' \
  ~/.hermes/state/dynamic-tools/predictions.jsonl | sort | uniq -c | sort -rn
```

**Trigger dampener audit — what got suppressed by `exclude_keywords`:**

```bash
jq -c 'select((.triggers_suppressed // []) | length > 0) | {msg: .message_preview[:60], suppressed: .triggers_suppressed, fired: .triggers_fired}' \
  ~/.hermes/state/dynamic-tools/predictions.jsonl | tail -20
```

**Tool calls per session — what actually got used:**

```bash
jq -r '.tool_name' ~/.hermes/state/dynamic-tools/tool_calls.jsonl | sort | uniq -c | sort -rn
```

**Expansion success rate — when did `expand_tools` lead to a real call?**

```bash
jq -c 'select(.expand_tools_used == true) | {scope, category: .expand_category, tool: .expanded_tool, turns_until_used}' \
  ~/.hermes/state/dynamic-tools/tool_calls.jsonl | tail -20
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

Smoke check:

```bash
hermes-agent/venv/bin/python3 plugins/dynamic-tools/scripts/check_trigger_dampeners.py
```

### Auto-suggesting new dampeners from telemetry

Dampeners are hand-written today, but the analyzer can mine candidate
patterns from telemetry. When a trigger fires (positive match) but no
matching tool ever gets called, that message preview is a false-positive
data point. The analyzer collects 80-char previews per (scope, trigger),
extracts 2–4 word n-grams, and surfaces n-grams that show up frequently
in false-positive messages but rarely (or never) in true-positive ones.

```bash
hermes-agent/venv/bin/python3 plugins/dynamic-tools/analyze.py \
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
in `~/.hermes/state/dynamic-tools/learned.json`. Shape:

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
hermes-agent/venv/bin/python3 plugins/dynamic-tools/analyze.py
hermes-agent/venv/bin/python3 plugins/dynamic-tools/analyze.py --format json
hermes-agent/venv/bin/python3 plugins/dynamic-tools/analyze.py --write-recommendations
hermes-agent/venv/bin/python3 plugins/dynamic-tools/analyze.py --include-archives
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
python3 analyze.py --state-dir ~/.hermes/state/dynamic-tools/harvest \
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
  dynamic-tools:
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
hermes-agent/venv/bin/python3 plugins/dynamic-tools/analyze.py --format json \
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
  dynamic-tools:
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
tail -f ~/.hermes/state/dynamic-tools/predictions.jsonl
# Send a message in any gateway channel. You should see a row appear with
# the triggers that fired and the counts of tools loaded.

# Or check Hermes' own logs:
grep "dynamic-tools" ~/.hermes/logs/agent.log
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
  the gateway, so the predictor isn't run and no narrowing happens.
- **Subagents are not narrowed.** When `delegate_task` spawns a subagent,
  the parent's tool curation is inherited; the predictor doesn't run again.
- **Cron jobs are not narrowed.** Cron sessions get explicit per-job
  toolsets via Hermes' own `_resolve_cron_enabled_toolsets`.
- **Unknown plugin tools default to always-on.** Tools whose names aren't
  in the policy's known sets are kept on (safer for shipping new plugins).
- **Sticky residency is per-process.** A gateway restart clears it. This is
  intentional — sticky is a session-grain optimization, not a learned policy.
- **`learned_mode: auto`/`audit` is not the default.** Adaptive promotion
  is deliberately opt-in until the analyzer's recommendations have been
  reviewed for a given scope. Start with `recommend`.

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
  dynamic-tools:
    enabled: false
```

…or remove the directory entirely:

```bash
rm -rf ~/.hermes/plugins/dynamic-tools
```

## Files

```
~/.hermes/plugins/dynamic-tools/
├── plugin.yaml                      # manifest
├── __init__.py                      # register(); _build_api_kwargs patch; hooks; sticky residency
├── predictor.py                     # regex/keyword classifier with dampener support
├── presets.py                       # YAML loader + per-scope resolver
├── learned.py                       # learned.json loader + preset merge (mtime-cached)
├── expand_tools.py                  # the expand_tools meta-tool
├── logger_io.py                     # predictions.jsonl + tool_calls.jsonl
├── analyze.py                       # deterministic telemetry analyzer
├── policy.yaml                      # the single shipped tool policy
├── CLAUDE.md                        # in-dir editing rules (scope, no back-compat, labs/main)
├── README.md                        # this file
├── STRATEGY.md                      # north star — promise, lab/live split, official-plugin readiness
├── PLAN.md                          # current architecture + status
├── AUTO-APPLY-PLAN.md               # writer-side closure plan; prereqs partially cleared
├── CALIBRATION-PLAN.md              # observe → handoff → steady-state flow
├── TRIGGER-SCORING-PLAN.md          # historical full-scoring plan (deferred)
├── docs/
│   ├── telemetry-audit-2026-05-17.md  # first audit pass — 2 bugs fixed, 3 pending data
│   ├── harvest-followups.md           # open auto-apply candidates surfaced by harvest
│   ├── dynamic-tools-hermes-surface.md  # patch-point reference
│   └── dynamic-tools-plan.md          # original design doc + categorization
├── scripts/
│   ├── bootstrap.py                 # first-install warm-start UX wrapper
│   ├── harvest-replay.py            # replay sessions/*.jsonl → synthetic telemetry
│   ├── smoke-test.py                # end-to-end mechanical validation
│   ├── daily-analysis.sh            # twice-daily analyzer cron
│   ├── rotate-telemetry.sh          # archive live JSONL files (gateway-safe)
│   ├── check_trigger_dampeners.py   # smoke checks for exclude_keywords + learned merge
│   └── README.md                    # ops scripts inventory + invocation
├── tests/
│   ├── test_session_attribution.py  # 43 unit tests across attribution/expand/harvest/keywords
│   ├── test_harvest_privacy.py      # privacy invariants for harvest output
│   └── run_tests.py
└── reports/                         # analyzer markdown output (dated)
```

## Review

```
# one line per run: timestamp, status, prediction count, scopes, expand events,
# net savings, recommendation count, dampener candidates
cat ~/.hermes/state/dynamic-tools/cron-logs/daily-summary.log
```