# tool-belt scripts

Operational scripts that live next to the plugin. Each is safe to run by
hand or under a scheduler. The shaper (`shape-ceiling.py`) writes to
`learned.json` under its own `cache_aware` sub-key; nothing else mutates
policy state.

## Cache-aware default (current)

Under cache-on providers (Anthropic, OpenAI auto-cache — ~80% of typical
traffic), tools are frozen at session start. The scripts most relevant
to that path are at the top:

| Script | Purpose | When to run |
|---|---|---|
| [`shape-ceiling.py`](shape-ceiling.py) | Between-session shaper (Phase 2 of doc 16). Reads `predictions.jsonl` + `tool_calls.jsonl` per scope, identifies per-tool **promote** candidates (tools reached via `expand_tools` in recent sessions) and **demote** candidates (always-on tools unused across enough sessions), writes to `learned.json` under `scopes[].cache_aware`. Conservative thresholds. | After ~20 sessions of organic use per scope. Run with `--dry-run` first. |
| [`cache-freeze-replay.py`](cache-freeze-replay.py) | Phase 0/5 replay harness. Reports freeze efficacy (would-break mutations eliminated) and Phase 5 matched-counterfactual cache savings with per-model price table. | After each meaningful change to the freeze logic; also useful for periodic health checks. |
| [`mnemosyne-prefix-check.py`](mnemosyne-prefix-check.py) | Verification harness (pre-Phase-1 gate). Confirms that `system_hash` stays stable across turns — i.e., Mnemosyne memory injection lands in a cache-friendly position rather than mutating the prefix. Gate passed on both profiles 2026-05-31; retained for regression detection. | After Mnemosyne config changes or upstream Mnemosyne upgrades. |

## Cache-off mode (kimi, gpt-5.4-mini, anything provider-side caching doesn't reach)

The per-turn predictor stays alive for cache-off providers, and the
following scripts support tuning that path:

| Script | Purpose | When to run |
|---|---|---|
| [`bootstrap.py`](bootstrap.py) | First-install warm-start. Runs `harvest-replay.py` then the analyzer in harvest-aware mode and prints a ranked **TOP ACTIONS** summary derived from your existing Hermes sessions. Useful primarily on cache-off scopes — on cache-on, `shape-ceiling.py` consumes actual `expand_tools` evidence and produces stronger recommendations. | Optional. Once after install, if you have any cache-off-mode scopes. |
| [`harvest-replay.py`](harvest-replay.py) | Replay user messages from `~/.hermes/sessions/*` through the per-turn predictor and emit synthetic telemetry tagged `policy_source: harvest`. | When tuning per-turn triggers on cache-off scopes. |
| [`check_trigger_dampeners.py`](check_trigger_dampeners.py) | Smoke checks: dampeners still veto + dampeners survive the learned-merge path. The learned-merge regression guard applies to all modes; the trigger-firing checks are cache-off-mode validation. | Before committing changes to policy.yaml or learned merge logic. |

## Ops utilities (mode-agnostic)

| Script | Purpose | When to run |
|---|---|---|
| [`smoke-test.py`](smoke-test.py) | End-to-end mechanical validation. Synthetic sessions through the plugin in an isolated tempdir. Currently covers the per-turn path; cache-on freeze-path assertions are a TODO. | After any gateway restart that loads new plugin code, or before committing changes that touch the hook surface. |
| [`rotate-telemetry.sh`](rotate-telemetry.sh) | Archive current `predictions.jsonl` + `tool_calls.jsonl` + `api_calls.jsonl` to `state/tool-belt/archive/reset-<ts>/`. Gateway-safe. | After a meaningful change to plugin behavior, when you want a clean measurement window. |
| [`daily-analysis.sh`](daily-analysis.sh) | Run analyzer with full recommendations + dampener mining, drop a markdown report + recommendations JSON, then run `shape-ceiling.py` to update `learned.json` for scopes with enough evidence. | Scheduled twice daily via launchd, or by hand any time. |
| [`com.dalemugford.hermes.tool-belt-analyzer.plist`](com.dalemugford.hermes.tool-belt-analyzer.plist) | launchd LaunchAgent template; runs `daily-analysis.sh` at 00:00 and 12:00 local time. | Install once. |

## Shape next session's frozen ceiling

```bash
python3 scripts/shape-ceiling.py --dry-run        # report
python3 scripts/shape-ceiling.py                  # write learned.json
```

Reads recent live telemetry per scope and reports per-tool promote /
demote candidates. Default thresholds are inherited from `policy.yaml`
under `learning.shape_ceiling` (the shipped values remain promote: ≥2
sessions and ≥3 calls; demote: ≥20 sessions of evidence). CLI flags
still override for ad-hoc runs. Writes to `learned.json` under
`scopes[].cache_aware` and mirrors to `scopes[].always_on` /
`scopes[].always_off` so the existing `apply_to_preset` reader picks
them up when `learned_mode` is `auto` or `audit`.

## Replay live data through the freeze policy

```bash
python3 scripts/cache-freeze-replay.py
```

Reports freeze coverage (matches vs would_break mutations), per-model
cache-adjusted savings under matched counterfactual, and dollar
estimates from the per-model price table. Add `--scope bernard:telegram`
to filter; `--markdown` to emit a report-friendly format.

## Confirm Mnemosyne stays cache-friendly

```bash
python3 scripts/mnemosyne-prefix-check.py
```

Verdict: `PREFIX_STABLE` or `MNEMOSYNE_MUTATES_PREFIX`. Run after any
Mnemosyne config change or upstream Mnemosyne upgrade.

## Cache-off fallback: bootstrap + replay

```bash
python3 scripts/bootstrap.py                       # cache-off scopes only
python3 scripts/harvest-replay.py                  # replay through per-turn predictor
```

These remain useful for tuning cache-off-mode scopes (kimi, gpt-5.4-mini).
On cache-on scopes, prefer `shape-ceiling.py` — actual `expand_tools`
events are stronger evidence than regex trigger guesses.

## Validate the plugin's runtime behavior

```bash
python3 scripts/smoke-test.py
```

Eight integration assertions covering the high-leverage invariants:
session_id population, bypass cohort math, expand_tools_used
attribution, cross-session sticky isolation, on_session_end eviction.
Run after any restart that loads new plugin code.

## Rotate telemetry to start a clean window

```bash
./rotate-telemetry.sh [tag]
```

Moves the live JSONL files into `~/.hermes/state/tool-belt/archive/reset-<UTC-timestamp>[-<tag>]/`.
The plugin re-creates the files on the next write (logger_io opens in
append mode every call, so an `mv` is atomic and gateway-safe).

If you pass a tag, it's appended to the archive folder name —
e.g. `./rotate-telemetry.sh pre-simplification` produces
`reset-2026-05-12-032645-pre-simplification/`.

## Schedule the daily analyzer

```bash
# install
cp com.dalemugford.hermes.tool-belt-analyzer.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) \
    ~/Library/LaunchAgents/com.dalemugford.hermes.tool-belt-analyzer.plist

# confirm it's scheduled
launchctl print gui/$(id -u)/com.dalemugford.hermes.tool-belt-analyzer \
    | grep -E "state|next run"

# run once on demand (smoke test or ad-hoc)
launchctl kickstart gui/$(id -u)/com.dalemugford.hermes.tool-belt-analyzer
```

Outputs land in four places:

- `~/.hermes/plugins/tool-belt/reports/<label>/YYYY-MM-DD-HHMMSS-analysis.md`
  — the human-readable per-run report for root / each profile.
- `<state-dir>/learned_recommendations.json` — the latest
  machine-readable recommendation set for that telemetry source.
  Overwritten each run.
- `<state-dir>/learned.json` — the applied learned overlay written by
  `shape-ceiling.py` when the evidence changes. Consumed on the next
  session when `learned_mode` is `auto` or `audit`.
- `~/.hermes/state/tool-belt/cron-logs/` — per-run JSON output,
  shaper logs for writes/failures, any stderr captured during a failed
  run, and `daily-summary.log` with one line per stage for at-a-glance
  review.

## Review a week's worth of data

```bash
# Scan the summary log
cat ~/.hermes/state/tool-belt/cron-logs/daily-summary.log

# Open the most recent markdown report (now includes Harvest-Driven and
# Trigger-Keyword sections when harvest data is present)
ls -t ~/.hermes/plugins/tool-belt/reports/*.md | head -1 | xargs open

# Inspect the latest recommendations JSON — includes harvest_tool_promotion
# kinds alongside expanded_category and trigger_group recs
jq '.dampener_candidates, .recommendations, .trigger_keyword_candidates' \
   ~/.hermes/state/tool-belt/learned_recommendations.json
```

## Uninstall

```bash
launchctl bootout gui/$(id -u) \
    ~/Library/LaunchAgents/com.dalemugford.hermes.tool-belt-analyzer.plist
rm ~/Library/LaunchAgents/com.dalemugford.hermes.tool-belt-analyzer.plist
```

The script files and their outputs aren't touched by uninstall —
the LaunchAgent just stops being scheduled.
