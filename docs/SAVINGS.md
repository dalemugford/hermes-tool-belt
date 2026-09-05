# Savings — what we measure and how

Hermes Tool Belt's headline number is the **net input-token-equivalent
saved**: the tool-schema tokens the model never saw, that your config
would otherwise have shipped, priced at what they were worth on the route
that would have billed them, minus the measured cost of the expansions
that recovered from narrowing. This file describes what that means
precisely, what it doesn't include, why the number is smaller than the
one earlier releases printed, and how to reproduce the math from the raw
telemetry.

The single source of truth is the code; this doc explains its decisions
in plain language.

## The baseline

Every prediction row in `predictions.jsonl` records two counts:

| Field | What it means |
|---|---|
| `ceiling_count` | Number of tools that *would have* shipped in this turn's API call without Tool Belt — the full `platform_toolsets` set your config allowed. |
| `narrowed_count` | Number of tools Tool Belt actually shipped after narrowing. |
| `ceiling_tokens` | Tokenized size of the ceiling toolset (name + description + JSON schema, per tool). |
| `narrowed_tokens` | Tokenized size of the narrowed toolset that actually went out. |
| `tokens_saved` | `ceiling_tokens − narrowed_tokens` — the count of tool-schema tokens the provider never saw. |
| `tokens_estimator` | Which estimator produced the counts: `tiktoken-cl100k` (preferred) or `chars-div-4` (fallback). |

The comparison is always **against your config-allowed ceiling**. We don't
compare against some hypothetical "best case" — we compare against what
your agent *would* have sent if Tool Belt weren't installed.

The arithmetic is exact: every row, `ceiling - narrowed == tokens_saved`.
The observed-cohort math lives in [savings.py](../savings.py)
(`cohort_stats`, `compute_observed`) and `./tool-belt savings --json`
emits the per-cohort totals if you want to verify on your own data.

### Which tokenizer

Tool Belt prefers **tiktoken's `cl100k_base`** (OpenAI's BPE) when the
`tiktoken` package is installed. Falls back to a **chars/4 heuristic**
otherwise. Either way, each row records which estimator was used in the
`tokens_estimator` field, and the savings report surfaces this in its
header so you can see at a glance which estimator your numbers came from.

```bash
pip install tiktoken    # optional but recommended for exact counts
```

| Estimator | Accuracy | When used |
|---|---|---|
| `tiktoken-cl100k` | Exact for GPT-family. ~5% approximate for Claude (different BPE; cl100k tends to slightly *over*-count Claude's JSON). | When `tiktoken` is importable. |
| `chars-div-4` | `len(json.dumps(payload)) // 4`. Within ~5–10% of real tokens for English JSON. Tends to *under*-count vs both real tokenizers. | When `tiktoken` isn't installed. |

For both estimators, the **delta** between ceiling and narrowed is more
honest than the absolute counts: both sides go through the same
estimator, so its offset cancels in the reduction percentage. A "47%
reduction" survives any tokenizer switch.

For provider-billed reality (what you actually pay), see
`api_calls.jsonl` → `input_tokens`. That field comes directly from the
provider's API response and is exact.

## Cache-on and cache-off

Tool Belt operates differently depending on whether the provider a
session is talking to caches the request prefix. The posture is a
property of the *provider*, per call, not of the scope — see
[ARCHITECTURE.md](ARCHITECTURE.md#two-postures).

### Cache-on (Anthropic, OpenAI auto-cache, OpenRouter)

On a caching provider Tool Belt **does not narrow**. It ships the full
enabled ceiling minus `expand_tools` on every call and never mutates the
list for the session's life (carry-all). Prediction rows are written with
`policy_source: "cache_on_carry_all"` and `ceiling == narrowed`.

In this mode:
- **Schema savings claimed: none.** `tokens_saved` is 0 on every
  carry-all row, and the cohort contributes nothing to the headline.
- **Tool Belt's value here is what it doesn't do.** On a caching route
  the tool block sits in the cheap cached prefix (only ~5% of cost on
  the measured install); the cost driver is uncached input, mostly
  cache-miss re-bills. A mid-session tool-list mutation re-bills the
  whole history at the full input rate — measured at ≈42.8K tokens per
  event on a mixed-provider history and ≈54.4K on an all-Codex one, and
  57–87% of all cache-miss re-bill cost measured on a narrowing route.
  Not busting the cache is the whole contribution, and it is not
  reported as savings.
- The cache hit rate is reported from provider-returned
  `cache_read_tokens` — it is the provider's discount, not ours.

### Cache-off (ollama-cloud, and any non-caching provider)

The predictor runs on every turn. There is no cache to protect, just
per-turn schema savings — this is where essentially all of Tool Belt's
honest value lives.

In this mode:
- **Per-turn savings (vs ceiling)**: `tokens_saved`, recurring every
  call, at the full input rate.
- **expand_tools overhead**: when the model needs a tool that wasn't
  loaded, it calls `expand_tools`, and the model's expand turn is one
  extra full-price API call carrying the whole history. The report
  charges each event its **measured** cost — the median `input_tokens`
  of the expand meta-call on that route (≈24–37K tokens on the measured
  install) — and subtracts the total from the gross. Only when a route
  has fewer than 5 expand events does it fall back to
  `EXPAND_ROUND_TRIP_TOKENS = 1500`, and the headline says so.

### Detection state

Detection is keyed per `scope|provider` bucket. A bucket locks to one
mode (`on` or `off`) after a short observation window — usually within
the first 5 API calls on that route. Each `api_calls.jsonl` row carries:

- `cache_mode` — that **call's** bucket outcome: `"pending"`, `"on"`, or
  `"off"`. It is not the session posture; a failover call to a
  non-caching provider is tagged by its own bucket.
- `provider_caches` — `true` / `false` once the call's bucket has locked,
  `null` while pending. This is the field the pricing reads.

Predictions during the pending window count toward a separate "pending"
cohort in the savings report and are priced at the full input rate
(unknown = conservative). They get re-classified once the bucket locks.

Locked decisions persist across sessions in
`~/.hermes/state/tool-belt/cache_mode_detection.json`, keyed
`"<scope>|<provider>"`, so subsequent sessions on the same route skip the
observation window. See
[ARCHITECTURE.md](ARCHITECTURE.md#posture-detection) for the full state
machine.

## Pricing: input-token equivalents

A saved schema token is not worth the same everywhere. On a caching
provider it would have been billed at the cache-read rate, so each one is
worth `1 / miss_premium` input tokens (~0.1 for the models in
`PRICE_TABLE`); on a non-caching provider every saved token was a
full-price input token (factor 1.0). `price_factor_for` in
[savings.py](../savings.py) applies this per prediction, and the result
is `saved_input_equiv_total` — the gross the headline nets from. The raw
schema figure (`realized_schema_token_reduction`) is kept as a detail
line.

### How a prediction's cache status is resolved

`prediction_provider_caches` looks at the prediction's API calls:

1. The per-call `provider_caches` field (rows written from this release
   on) is authoritative — the last call's status wins, then any call with
   a known status.
2. Rows that predate the field fall back to `PROVIDER_CACHE_HINTS`, a
   small table keyed by provider name: `ollama-cloud` and `ollama` are
   `false` (0% observed cache_read hits). A provider absent from the
   table has no hint.
3. Otherwise the cohort the prediction was classified into (`on` →
   caching, `off` → not). Pending and bypass rows with no per-call facts
   stay unknown, are priced at full rate, and are excluded from overhead
   measurement.

`PROVIDER_CACHE_HINTS` covers rows without a per-call cache status: it
names the routes observed never to cache, so a row whose session-level
`cache_mode` tag disagrees with its actual route is still priced right.

## Forward-aware headline, uncached-scoped denominators

The headline `NET TOKENS SAVED` is `net_forward` — the non-caching cohort's
ongoing net, `saved_input_equiv_noncaching − overhead_noncaching`
(`ObservedCohort` in [savings.py](../savings.py)). It does **not** blend in
caching-provider expand breaks (`net_caching_historical`), which are
one-time costs the carry-all posture does not incur. That figure is kept
in `--json` for diagnostics only, alongside `net_token_reduction`, the
blended figure over every cohort. Neither appears in the text report.

Because the headline is scoped to non-caching sessions, the numbers built
from it are too: the per-session average, the 12-month pace, the headline
session count, and each agent's row in `PER AGENT SAVINGS` all divide by /
span the **uncached sessions only** — `compute_observed` walks the same
`caches is not True` rows that price `net_forward` and records
`n_sessions_noncaching`, `first_ts_noncaching`, and `last_ts_noncaching`
alongside them. A caching session contributes nothing to the net and so
does not dilute the per-session average.

An agent whose sessions are caching-dominant — `savings_cli`'s
`_carried_whole` classifies this from the cohort's prediction counts — is
excluded from the total entirely and named instead in a neutral line:
`Agent(s) currently excluded from Tool Belt (no shaping): <names>.` followed
by `Agents using caching providers exclusively do not require Tool Belt.`
Names are comma-separated and, on a terminal with color, italicized
(`savings_cli._ital`).

## Measured expansion overhead

`measure_expand_overhead` in [savings.py](../savings.py) prices one
expansion per cohort from the telemetry itself (the report's
`overhead_measurement` block shows its work):

- **Caching cohort** — `per_event = max(0, expand_miss_rate −
  baseline_miss_rate) × median_rebill_premium`. A prediction *missed*
  when any of its calls returned `cache_read_tokens == 0` with a prompt
  above 2,000 tokens; the baseline is over caching predictions that made
  no `expand_tools` call and weren't the session's cold first turn; the
  re-bill premium is the missed prompt × `(input − cache_read) / input`
  at the call model's rates. This is the causal cost of the cache break,
  net of the misses that would have happened anyway. A caching session
  under carry-all produces no expand events, so this prices only rows
  that carry them.
- **Non-caching cohort** — `per_event` = median `input_tokens` of the
  expand meta-call (the prediction's call with the smallest output).
- **Fallback** — a cohort with fewer than 5 expand events, or (caching)
  fewer than 10 baseline predictions, is charged
  `EXPAND_ROUND_TRIP_TOKENS = 1500` and marked `"fallback"`.

The headline is tagged with its basis: `measured overhead`,
`fallback: 1,500 tok/event`, or `mixed` when the two cohorts differ.
`ObservedCohort` exposes `overhead_per_event_caching`,
`overhead_per_event_noncaching`, `overhead_basis`, and
`overhead_measurement`.

## Bypassed sessions

When `bypass_rate > 0` is set (an internal testing feature — see
[CONFIGURATION.md](CONFIGURATION.md#bypass_rate-internal-testing-feature)),
a deterministic fraction of sessions ship the **full toolset** with no
narrowing. The hash `sha1(scope|session_id)` decides — same session is
always in or out so behavior stays internally consistent.

Bypass rows are written to `predictions.jsonl` with `policy_source:
"bypass"` and `ceiling_count == narrowed_count` (nothing was narrowed
by design):

- **They are excluded from the savings figures.** The headline reflects
  only rows where narrowing actually applied; a bypassed session spends
  the full manifest and improves no measurement — savings are measured
  directly per narrowed turn, so no control cohort is needed.
- **They are counted separately in `--json`** (`bypass` and `pending`
  keys per agent) for anyone reconciling the raw rows; the text report
  does not print them.
- They are distinct from carry-all rows (`policy_source:
  "cache_on_carry_all"`), which are the plugin's deliberate posture on a
  caching provider, not a control.

`bypass_rate: 1.0` is how observation mode turns narrowing off on a
scope without disabling the plugin.

## What's excluded

Three classes of tool calls are *not* subject to narrowing and are
explicitly excluded from the savings figures. They're still logged to
`tool_calls.jsonl`, tagged via the `source` field:

| `source` value | When | Why it's excluded |
|---|---|---|
| `gateway` | The normal path: a user message hit `pre_gateway_dispatch`, the posture resolved, and the per-turn narrowed set (cache-off) or the carry-all list (cache-on) applied to the API call. | Subject to the posture. **These count toward savings.** |
| `cron` | Hermes' cron runner uses `cron_<hash>_<ts>` session ids and bypasses `pre_gateway_dispatch` entirely; cron jobs get explicit per-job tool sets from Hermes itself. | Tool Belt's narrowing never ran for these. |
| `subagent` | A subagent (`delegate_task`-spawned) made a tool call inside a parent gateway session. The subagent inherits the parent's tools but doesn't see the prediction context. | Tool Belt's narrowing doesn't apply transitively to subagents. |

Cron and subagent rows never enter the savings arithmetic — they are
filtered on `source` before any cohort is built.

## What we don't claim

- **We don't claim cost savings without showing the math.** Token counts
  → dollars is the user's call (the report can be re-aggregated against
  any provider's rate card; the JSON output makes this easy). The
  illustrative dollar lines in the text report use public input list
  prices from `PRICE_TABLE` and say which rate basis they came from.
- **We don't claim the provider's cache discount as savings.** On a
  caching route Tool Belt narrows nothing and claims nothing; the cache
  hit rate is shown from provider-returned tokens so you can see whether
  the cache is warm, and that's all.
- **We don't double-count.** The gross is `Σ(ceiling − narrowed)` across
  logged prediction rows — one row per inbound user message — weighed
  once per row by its route's price factor. Carry-all rows contribute 0.
- **We don't claim the matched-counterfactual upper bound as the
  headline.** [cache_replay.py](../cache_replay.py) computes a per-call
  counterfactual against the stable-cohort mean at the same
  `api_call_idx` — useful for deep analysis, deliberately not the
  headline number. The headline is the simple, defensible priced
  `ceiling − narrowed` less measured overhead. `analyze.py` imports this
  module to produce the cache-aware section of its report; it is also a
  standalone diagnostic CLI.

## Reproducing the numbers from raw telemetry

The savings report runs entirely on three JSONL files:

```bash
~/.hermes/state/tool-belt/predictions.jsonl
~/.hermes/state/tool-belt/api_calls.jsonl     # for cache_mode, provider_caches + cache_read tokens
~/.hermes/state/tool-belt/tool_calls.jsonl    # for expand_tools events + cron/subagent exclusion
```

Quick reproduction of the raw cache-off schema figure with `jq` (the
gross before pricing and overhead):

```bash
# Raw schema tokens saved across cache-off predictions (last api_call's cache_mode determines bucket)
jq -s --slurpfile api ~/.hermes/state/tool-belt/api_calls.jsonl \
  '($api[0] | map(select(.cache_mode == "off")) | map(.prediction_id) | unique) as $off_preds
   | map(select(.prediction_id as $p | $off_preds | index($p))) | map(.tokens_saved) | add' \
  ~/.hermes/state/tool-belt/predictions.jsonl
```

The same query with `"on"` returns 0 for any session run under carry-all.
The canonical command packages the full priced math into a single
read-only report:

```bash
./tool-belt savings                      # every enabled agent + aggregate
./tool-belt savings --agent=default      # one agent, all its platforms
./tool-belt savings --json               # stable machine-readable schema
./tool-belt savings --since 2026-05-15
```

The text report leads with `NET TOKENS SAVED` (`net_forward`, the
uncached-only ongoing net, with its overhead basis), then the raw schema
tokens unsent and what they were worth after cache pricing, then the
`expand_tools` overhead and event count. `--json` exposes `net_forward`,
`saved_input_equiv_noncaching`, `overhead_noncaching`,
`n_sessions_noncaching` / `first_ts_noncaching` / `last_ts_noncaching`,
the blended-history `net_token_reduction` and `net_caching_historical`,
`saved_input_equiv_total`, `realized_schema_token_reduction`,
`expansion_overhead`, `overhead_per_event_caching` / `_noncaching`,
`overhead_basis`, and `overhead_measurement` per agent.

`tool-belt savings` reports two separately-labeled cohorts that are **never
summed together**:

- **Observed** — realized savings from organic telemetry (provider-returned
  usage is authoritative). This is what actually happened.
- **Projected** — a counterfactual replay of historical Hermes sessions through
  the current effective carrying assignment. Every projected figure is labeled
  counterfactual until matched by organic post-apply telemetry and carries a
  confidence (high/medium/low). Dollars are suppressed: Hermes records a
  *transport* label rather than a billing route, so `classify_cost` never
  reaches the `known` metered class and the report falls back to a net
  input-reduction percentage.

The engine lives in [`savings.py`](../savings.py) and is the single home for the
price table, the provider cache hints, the token estimator, the overhead
measurement, and the thin-data fallback constant.
`cache_replay.py` imports the price table from it, so there is no duplicate
table or constant.

## Field reference (predictions.jsonl)

For the full list, see [CONFIGURATION.md → Telemetry outputs](CONFIGURATION.md#telemetry-outputs).
The fields most relevant to savings:

| Field | Type | Meaning |
|---|---|---|
| `ts` | float | Unix timestamp |
| `prediction_id` | string | Join key for `tool_calls.jsonl` and `api_calls.jsonl` |
| `session_id` | string | Canonical Hermes session key (`agent:main:<platform>:...`), constant per chat |
| `hermes_session_id` | string | Hermes' rotating session UUID — the shaper's distinct-session unit |
| `scope` | string | `<agent>:<platform>` (e.g. `default:telegram`) |
| `tokens_estimator` | string | `tiktoken-cl100k` or `chars-div-4` |
| `ceiling_count` | int | Tools in the platform_toolsets ceiling |
| `narrowed_count` | int | Tools actually shipped |
| `ceiling_tokens` | int | Tokenized size of the ceiling toolset |
| `narrowed_tokens` | int | Tokenized size of the narrowed toolset |
| `tokens_saved` | int | `ceiling_tokens − narrowed_tokens` (0 on carry-all rows) |
| `reduction_pct` | float | `tokens_saved / ceiling_tokens × 100` |
| `frozen_reuse` | bool | Vestigial — always `false` on current rows. Historical rows with `true` were cache-on reuse dispatches under the removed per-session freeze; the classifier still reads them as cache-on. |
| `frozen_reuse_count` | int | Vestigial — always `0` on current rows. |
| `policy_source` | string | `preset`, `learned`, `cache_on_carry_all` (caching provider, nothing narrowed), or `bypass` (A/B control) |
| `model` | string | Provider model name on this dispatch |

## Limitations honestly stated

- **Cron and subagent calls aren't narrowed.** That's by design (cron sets
  its own toolset; subagents inherit). They're tagged via `source` so
  they're easy to filter out, but they show up in `tool_calls.jsonl`.
- **On caching providers Tool Belt saves nothing it can count.** Its
  contribution there is not busting the cache, which shows up as an
  absence of re-bills, not as a number. The report does not invent one.
- **Predictor false-positives cost `expand_tools` events.** When the
  model needs a tool we didn't ship, it has to ask, and on a non-caching
  route that ask is one extra full-price call. We log every event and
  subtract its **measured** cost (`measure_expand_overhead` in
  [savings.py](../savings.py)). With fewer than 5 events on a route (or
  fewer than 10 baseline predictions for the caching cohort) the figure
  is the `EXPAND_ROUND_TRIP_TOKENS = 1500` fallback and the headline is
  tagged `fallback`; [analyze.py](../analyze.py)'s
  `--expand-round-trip-tokens` overrides that fallback, not the
  measurement.
- **Historical cache-status classification is a hint, not a fact.** Rows
  written before `provider_caches` existed are classified by the
  provider-name hint table, which only knows `ollama-cloud` / `ollama`.
  Any other pre-field row falls to its cohort posture, which for a
  fallback route may be wrong in the direction of over-claiming.
- **Cohort splits are noisy on small samples.** A few sessions per mode is
  enough to make the per-turn average wobble. The total is robust; the
  per-turn average is noisier.
