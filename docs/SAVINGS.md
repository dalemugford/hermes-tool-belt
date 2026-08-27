# Savings — what we measure and how

Hermes Tool Belt's headline number is the count of **tokens the model never
saw, that your config would otherwise have shipped**. This file describes
what that means precisely, what it doesn't include, and how to reproduce
the math from the raw telemetry.

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
There's a one-line cross-check in [scripts/savings-report.py](../scripts/savings-report.py)
if you want to verify on your own data.

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

Tool Belt operates differently depending on whether the active provider
caches the request prefix.

### Cache-on (Anthropic, OpenAI auto-cache)

The predictor runs once at session start, selects a tool set, and **freezes**
it for the rest of the session. Subsequent turns reuse the same tools so
the provider's cached prefix stays valid.

In this mode:
- **Per-turn savings (vs ceiling)**: `tokens_saved`, the same as cache-off.
  The narrowed tools-block is smaller than the ceiling would have been.
- **Cache amortization (bonus)**: the narrowed block, once cached, gets
  re-billed at the cache-hit rate on every subsequent turn. The provider
  reports this directly as `cache_read_tokens` in `api_calls.jsonl`.

The bonus is **enabled** by Tool Belt (by holding the prefix stable so the
cache doesn't get busted) but the discount itself comes from the provider.
We report the cache hit rate from provider-returned tokens — it's not an
estimate.

### Cache-off (kimi, gpt-5.4-mini, and any non-caching provider)

The predictor runs on every turn. There's no freeze, no cache to amortize
against, just per-turn schema savings.

In this mode:
- **Per-turn savings (vs ceiling)**: `tokens_saved`, recurring every turn.
- **expand_tools overhead**: when the model needs a tool that wasn't loaded,
  it calls `expand_tools`, paying a round-trip cost (~1500 tokens for the
  output token, the result, and the next API call). The savings report
  subtracts `(expand_tools_events × 1500)` from the gross savings to give
  a net figure.

### Detection state

A scope is locked to one mode (`on` or `off`) after a short observation
window — usually within the first 5 API calls. Before that, the
`cache_mode` field in `api_calls.jsonl` is `"pending"`.

Predictions during the pending window count toward a separate "pending"
cohort in the savings report. They get re-classified once the scope locks.

The locked decision persists across sessions in
`~/.hermes/state/tool-belt/cache_mode_detection.json` so subsequent
sessions skip the observation window. See
[ARCHITECTURE.md](ARCHITECTURE.md#mode-detection) for the full state
machine.

## The A/B baseline cohort

When `bypass_rate > 0` is set (globally or per-scope in `config.yaml`),
a deterministic fraction of sessions ship the **full toolset** with no
narrowing. The hash `sha1(scope|session_id)` decides — same session is
always in or out of the cohort so behavior stays internally consistent.

Bypass rows are written to `predictions.jsonl` with `policy_source:
"bypass"` and `ceiling_count == narrowed_count` (nothing was narrowed
by design). They serve as a real control group:

- **They are excluded from the cache-on and cache-off savings figures.**
  Including them would tank the average with rows that intentionally
  didn't narrow. The headline numbers reflect only the rows where
  narrowing actually applied.
- **They appear as their own cohort in the savings report.** The
  "Bypass cohort" section shows how many sessions ran without
  narrowing and the full-toolset token volume — the apples-to-apples
  baseline for "what would the bill look like without Tool Belt?"

To turn narrowing off entirely on a scope without disabling the plugin,
set `bypass_rate: 1.0` on it. To run a small control sample for
comparison, set `bypass_rate: 0.05` (5% of sessions).

## What's excluded

Three classes of tool calls are *not* subject to narrowing and are
explicitly excluded from the savings figures. They're still logged to
`tool_calls.jsonl`, tagged via the `source` field:

| `source` value | When | Why it's excluded |
|---|---|---|
| `gateway` | The normal path: a user message hit `pre_gateway_dispatch`, the predictor ran, and the freeze (or per-turn narrowed set) applied to the API call. | Subject to narrowing. **These count toward savings.** |
| `cron` | Hermes' cron runner uses `cron_<hash>_<ts>` session ids and bypasses `pre_gateway_dispatch` entirely; cron jobs get explicit per-job tool sets from Hermes itself. | Tool Belt's narrowing never ran for these. |
| `subagent` | A subagent (`delegate_task`-spawned) made a tool call inside a parent gateway session. The subagent inherits the parent's tools but doesn't see the prediction context. | Tool Belt's narrowing doesn't apply transitively to subagents. |

If you run the savings report on a profile with heavy cron use, you'll
see the cron row count as an "Excluded" line. The savings numbers above
don't include those rows.

## What we don't claim

- **We don't claim cost savings without showing the math.** Token counts
  → dollars is the user's call (the report can be re-aggregated against
  any provider's rate card; the JSON output makes this easy).
- **We don't claim cache amortization is "savings vs no-Tool-Belt".**
  The cache discount is a provider feature. Tool Belt enables it by not
  invalidating the cache mid-session. We report cache hit rate from
  provider-returned tokens so you can see directly whether the cache is
  warm.
- **We don't double-count.** The savings number is `Σ(ceiling - narrowed)`
  across all logged prediction rows — one row per inbound user message,
  including freeze-reuse rows in cache-on mode. The arithmetic is honest
  even if a session reuses the same freeze across many turns.
- **We don't claim the matched-counterfactual upper bound as the
  headline.** [scripts/cache-freeze-replay.py](../scripts/cache-freeze-replay.py)
  computes a per-call counterfactual against the stable-cohort mean at
  the same `api_call_idx` — useful for deep analysis, deliberately not
  the headline number. The headline is the simple, defensible
  `ceiling - narrowed` calculation.

## Reproducing the numbers from raw telemetry

The savings report runs entirely on three JSONL files:

```bash
~/.hermes/state/tool-belt/predictions.jsonl
~/.hermes/state/tool-belt/api_calls.jsonl     # for cache_mode + cache_read tokens
~/.hermes/state/tool-belt/tool_calls.jsonl    # for cron/subagent exclusion counts
```

Quick reproduction of the headline cache-on number with `jq`:

```bash
# Tokens saved across cache-on predictions (last api_call's cache_mode determines bucket)
jq -s --slurpfile api ~/.hermes/state/tool-belt/api_calls.jsonl \
  '($api[0] | map(select(.cache_mode == "on")) | map(.prediction_id) | unique) as $on_preds
   | map(select(.prediction_id as $p | $on_preds | index($p))) | map(.tokens_saved) | add' \
  ~/.hermes/state/tool-belt/predictions.jsonl
```

Same approach for `cache-off`. The savings report just packages this
into a single command:

```bash
python3 scripts/savings-report.py
python3 scripts/savings-report.py --scope default:telegram
python3 scripts/savings-report.py --since 2026-05-15
python3 scripts/savings-report.py --json   # machine-readable
```

## Field reference (predictions.jsonl)

For the full list, see [CONFIGURATION.md → Telemetry outputs](CONFIGURATION.md#telemetry-outputs).
The fields most relevant to savings:

| Field | Type | Meaning |
|---|---|---|
| `ts` | float | Unix timestamp |
| `prediction_id` | string | Join key for `tool_calls.jsonl` and `api_calls.jsonl` |
| `session_id` | string | Canonical Hermes session key (`agent:main:<platform>:...`) |
| `scope` | string | `<agent>:<platform>` (e.g. `default:telegram`) |
| `ceiling_count` | int | Tools in the platform_toolsets ceiling |
| `narrowed_count` | int | Tools actually shipped |
| `ceiling_tokens` | int | Tokenized size of the ceiling toolset |
| `narrowed_tokens` | int | Tokenized size of the narrowed toolset |
| `tokens_saved` | int | `ceiling_tokens − narrowed_tokens` |
| `reduction_pct` | float | `tokens_saved / ceiling_tokens × 100` |
| `frozen_reuse` | bool | True when this dispatch reused an existing freeze |
| `frozen_reuse_count` | int | How many times this freeze has been reused |
| `policy_source` | string | `preset`, `learned`, or `bypass` (A/B baseline) |
| `model` | string | Provider model name on this dispatch |

## Limitations honestly stated

- **Cron and subagent calls aren't narrowed.** That's by design (cron sets
  its own toolset; subagents inherit). They're tagged via `source` so
  they're easy to filter out, but they show up in `tool_calls.jsonl`.
- **The cache amortization figure isn't "savings vs no-Tool-Belt"** — it's
  the provider's cache benefit, which exists with or without us. We report
  it because Tool Belt *enables* it by not invalidating the cache, but
  the proper attribution is "cache discount the provider gave you" not
  "savings Tool Belt created."
- **Predictor false-positives cost `expand_tools` round-trips.** When the
  model needs a tool we didn't ship, it has to ask. We log every event
  and subtract a ~1500-token estimate from the cache-off net figure. The
  estimate is configurable in [analyze.py](../analyze.py) (`--expand-round-trip-tokens`).
- **Cohort splits are noisy on small samples.** A few sessions per mode is
  enough to make the per-turn average wobble. The total `tokens_saved`
  is robust; the per-turn average is noisier.
