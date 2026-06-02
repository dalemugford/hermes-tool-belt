# Launch: Hermes Tool Belt

A release-ready narrative for blog / Show HN / launch announcement.
Tighten or expand the sections as needed for each channel — there's a
~140-char hook, a 300-word elevator, a long-form blog draft, and a
12-tweet thread variant at the bottom.

The product page lives at [README.md](../README.md). The methodology
lives at [SAVINGS.md](SAVINGS.md). This file is the **story**.

---

## The hook (≤140 chars)

> Hermes Tool Belt — adaptive tool loadouts for agents. Carry only what
> you need, drop the rest. Sharpens with use.

Or:

> Your agent ships 8–15k tokens of tool definitions every message,
> even when the answer is "hi". Hermes Tool Belt fixes that without
> breaking the cache.

---

## The 300-word elevator

Every message your agent sends to an LLM ships the **full tool block** —
names, descriptions, JSON schemas — even when the user just typed `"hi"`.
For a heavy profile (web, terminal, file, browser, MCP, skills), that's
8–15k tokens of overhead per message.

The naive fix — narrow the tools each turn — busts the provider's prefix
cache. On a multi-turn session, the lost cache costs more than the
schema savings.

**Hermes Tool Belt** resolves the tension with three features that work
together:

1. **`expand_tools`** — a meta-tool the model calls when it needs
   something that wasn't loaded. Narrowing never strands the agent: it
   asks, gets it, the conversation continues. Every call is logged as
   evidence that tool was wanted — between sessions, the shaper folds
   repeatedly-reached-for tools into the next session's ceiling. The
   model's reach IS the promotion vote.

2. **Deterministic shaping** — regex triggers in `policy.yaml`, an
   `always_on` baseline, per-scope overrides, and a bypass-rate A/B
   knob. No LLM-as-router. No extra API call. No non-deterministic
   mis-route. Every decision auditable in `predictions.jsonl`.

3. **Cache-aware adaptation** — same predictor, same policy, same
   `expand_tools`. What changes is **when**: under cache-on providers
   (Anthropic, OpenAI auto-cache), tools are frozen for the session so
   the cached prefix stays intact. Under cache-off providers (kimi,
   gpt-5.4-mini), tools are re-narrowed per turn. Mode is auto-detected
   from observed `cache_read` tokens. You set nothing.

Falls through to no-narrowing on any error. Never adds a tool your
config didn't already allow. Token counts use `tiktoken` when installed
(chars/4 fallback). Every savings number is reproducible from raw JSONL.

`hermes plugins install dalemugford/hermes-tool-belt`

---

## The long-form blog draft (~700 words)

### Your agent's `hi` costs as much as `deploy the build script`

Open the network panel on any LLM-powered agent and look at one request.
You'll find tool definitions stuffed into the request body: names,
descriptions, JSON schemas, a `required` list per tool. Multiplied
across a profile that can browse, run terminals, read files, talk to
MCP servers, and use custom skills, that block is **8–15k tokens of
overhead** — shipped on every message.

It's the same overhead whether the message is `"hi"` or `"deploy the
build script and tail the logs."`

The naive fix is obvious: predict the user's intent and ship only the
tools the model is likely to need. We tried that. It cost more than it
saved.

### Why per-turn narrowing fights prefix caching

Anthropic, OpenAI auto-cache, and most modern providers hash the prefix
of each request — roughly `system + tools + history` — and bill the
cached portion at a steep discount (~10% of full rate on Anthropic).
That cache discount is *massive*: it's often the single biggest
billing item on a multi-turn session.

Per-turn narrowing **mutates the tools block between turns**. That puts
a cache-invalidation point in the middle of the prefix. The provider
re-bills the entire conversation history at the full input rate. On a
3-turn session, the cache loss exceeds the schema savings within the
first reused turn.

So we had two bad options: ship too much overhead, or break the cache.

### A third option: separate the goal from the cadence

**Goal**: usage-aware tool loading at per-tool granularity. Tools the
model needs are loaded; tools it doesn't are dropped.

**Cadence**: when the adjustment happens. Per turn when there's no
prefix cache to protect. Per session when there is — adjusting at a
moment the cache would have refreshed anyway.

Tool Belt auto-detects which cadence the active provider supports and
operates accordingly. The detection observes `cache_read_tokens` over
the first few API calls of a scope, locks the mode, and persists the
decision across sessions.

### Three features doing the work

**`expand_tools` — the safety valve that learns.** A meta-tool the
model can call when it wants a tool that wasn't loaded. The handler
resolves the category, the tools join the request, the conversation
continues. The narrowing decision never strands the model.

Every call is logged. Between sessions, the shaper reads
`expand_tools` events as evidence and promotes repeatedly-reached-for
tools into the next session's starting ceiling. The model's reach IS
the promotion vote — no hand-tuning required.

**Deterministic shaping — rules over routing.** The narrowing decision
is a regex match on message text, an attachment check, and a YAML
lookup. No LLM-as-router. No extra API call. No non-deterministic
mis-route. Hand-tune triggers in `policy.yaml`, layer per-scope
overrides in `config.yaml`, and the telemetry feeds the analyzer back
recommendations — all the same shape. The YAML is the truth.

**Cache-aware adaptation — one goal, two cadences.** Same predictor,
same policy, same `expand_tools`. Under cache-on providers, tools are
frozen at session start so the cached prefix stays intact. Under
cache-off providers, tools are re-narrowed per turn with sticky
residency carrying expanded categories forward a few turns. Same goal,
different cadence, picked automatically per scope.

### How it stays honest

Tool Belt never adds a tool your `platform_toolsets` config didn't
already allow. Any error — bad YAML, predictor exception, missing
lookup — falls through to no narrowing. Token counts use `tiktoken`
when installed (chars/4 fallback), and every prediction row records
which estimator was used. Provider-billed tokens are recorded directly
from API responses in `api_calls.jsonl`. The savings math is
reproducible from raw JSONL with a `jq` one-liner.

### Install

```bash
hermes plugins install dalemugford/hermes-tool-belt
pip install tiktoken    # optional — exact token counts in the savings report
```

The plugin runs cache-on by default for providers it doesn't recognize,
falls back to cache-off on detection failure, falls back to no
narrowing on any error. Restart the gateway, send a message, watch the
savings report.

---

## The Twitter thread (12 tweets)

**1/** Your agent ships 8–15k tokens of tool definitions on every
message — names, descriptions, JSON schemas, the whole block. Even when
the answer is `"hi"`. We built **Hermes Tool Belt** to fix that without
breaking the cache.

**2/** The naive fix — narrow tools per turn — fights provider prefix
caching. Modern providers cache the system+tools+history block at a
steep discount. Mutate the tools mid-prefix and the cache invalidates,
re-billing the whole conversation at full rate.

**3/** Tool Belt resolves the tension with three features.

**4/** **#1: `expand_tools`** — a meta-tool the model calls when it
needs something that wasn't loaded. Narrowing never strands the agent.
Asks → gets → continues. Every call is logged.

**5/** Between sessions, those `expand_tools` events become **promotion
votes**. Tools the model reached for get folded into the next session's
ceiling. The model's reach IS the signal. No hand-tuning.

**6/** **#2: Deterministic shaping** — regex triggers in `policy.yaml`,
an `always_on` baseline, per-scope overrides, an A/B bypass-rate knob.
**No LLM-as-router.** No extra API call. No non-deterministic mis-route.
Every decision auditable in `predictions.jsonl`.

**7/** The YAML is the truth. You hand-tune triggers, telemetry feeds
the analyzer back recommendations, all the same shape. Hot-reload on
gateway restart.

**8/** **#3: Cache-aware adaptation** — same predictor, same policy,
same `expand_tools`. What changes is **when**.

**9/** Cache-on providers (Anthropic, OpenAI auto-cache): freeze tools
at session start. Cached prefix stays intact. ~95% prefix cache hit on
subsequent turns.

**10/** Cache-off providers (kimi, gpt-5.4-mini): re-narrow per turn.
Sticky residency carries expanded categories forward. Lookback to prior
user message catches signal from short replies.

**11/** Mode is **auto-detected per scope** from observed `cache_read`
tokens. Locked decision persists across sessions. You set nothing.

**12/** Falls through to no-narrowing on any error. Never widens past
your `platform_toolsets` ceiling. Token counts use `tiktoken` when
installed (chars/4 fallback). Savings reproducible from raw JSONL.

`hermes plugins install dalemugford/hermes-tool-belt`

→ [github.com/dalemugford/hermes-tool-belt](https://github.com/dalemugford/hermes-tool-belt)
