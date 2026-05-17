# dynamic-tools scripts

Operational scripts that live next to the plugin. Each is safe to run by
hand or under a scheduler. None mutate `learned.json` or other policy
state — those changes remain manual (see [AUTO-APPLY-PLAN.md](../AUTO-APPLY-PLAN.md)).

## Inventory

| Script | Purpose | When to run |
|---|---|---|
| [`bootstrap.py`](bootstrap.py) | First-install warm-start. Runs `harvest-replay.py` then the analyzer in harvest-aware mode and prints a ranked **TOP ACTIONS** summary derived from your existing Hermes sessions. Closes the cold-start gap so day-one is useful, not "wait a week." | Once, immediately after `hermes plugins install`. Idempotent — re-run any time to refresh recommendations against current session history. |
| [`harvest-replay.py`](harvest-replay.py) | Replay user messages from `~/.hermes/sessions/*` (and per-profile `profiles/*/sessions/*`) through the dynamic-tools predictor and emit synthetic `predictions.jsonl` + `tool_calls.jsonl` tagged `policy_source: harvest` into a `state/dynamic-tools/harvest/` subdir. | Called by `bootstrap.py`. Run by hand to refresh harvest data after a policy.yaml change, or when investigating a specific window with `--window-days N`. |
| [`smoke-test.py`](smoke-test.py) | End-to-end mechanical validation. Runs 109 synthetic session scenarios through the plugin in an isolated tempdir; asserts session_id population, bypass-cohort distribution, expand_tools_used attribution, cross-session isolation, on_session_end eviction. | After any gateway restart that loads new plugin code, or before committing changes that touch the hook surface. Pass/fail in <1 second, no side effects. |
| [`rotate-telemetry.sh`](rotate-telemetry.sh) | Archive current `predictions.jsonl` + `tool_calls.jsonl` to `state/dynamic-tools/archive/reset-<ts>/`. Gateway-safe. | After a meaningful change to plugin behavior, when you want a clean measurement window. |
| [`daily-analysis.sh`](daily-analysis.sh) | Run analyzer with full recommendations + dampener mining, drop a markdown report + recommendations JSON, append a one-line summary to `cron-logs/daily-summary.log`. | Scheduled twice daily via launchd, or by hand any time. Skips cleanly when telemetry is empty. |
| [`check_trigger_dampeners.py`](check_trigger_dampeners.py) | Smoke checks: dampeners still veto + dampeners survive the learned-merge path. | Before committing changes to policy.yaml or learned merge logic. |
| [`com.dalemugford.hermes.dynamic-tools-analyzer.plist`](com.dalemugford.hermes.dynamic-tools-analyzer.plist) | launchd LaunchAgent template; runs `daily-analysis.sh` at 00:00 and 12:00 local time. | Install once; the system handles scheduling. |

## First install — run `bootstrap.py`

```bash
python3 scripts/bootstrap.py
```

Produces a ranked TOP ACTIONS summary like:

```
================================================================
  TOP ACTIONS
================================================================
  Tool promotions (edit policy.yaml or channels.<scope>.always_on_extra):
    1. [PROMOTE  ] bernard:telegram  terminal  cuts= 516  net=+499,684 tok
    2. [PROMOTE  ] bernard:slack     terminal  cuts=  13  net=+12,128 tok
    3. [BROADEN  ] bernard:telegram  patch     cuts= 132  net=-76,316 tok

  Trigger keyword candidates (add to the named trigger's `keywords` list):
    1. bernard:telegram  shell       ← "deploy the app"
    2. bernard:telegram  file_write  ← "load the hermes"
    ...
```

Flags: `--profile <name>` (one profile only), `--window-days N`
(recent sessions only), `--quiet` (suppress phase output).

## Replay sessions to refresh harvest

```bash
python3 scripts/harvest-replay.py
```

Reads `sessions/*.jsonl` per profile, runs each user message through
the predictor against the session's recorded toolset, and writes
synthetic telemetry to `state/dynamic-tools/harvest/`. Output rows are
stamped `policy_source: harvest` so the analyzer can weight them
differently from live data. Tool call arguments are NEVER written —
only `message_hash` + 80-char `message_preview`. See
[tests/test_harvest_privacy.py](../tests/test_harvest_privacy.py) for
the enforced privacy invariants.

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

Moves the live JSONL files into `~/.hermes/state/dynamic-tools/archive/reset-<UTC-timestamp>[-<tag>]/`.
The plugin re-creates the files on the next write (logger_io opens in
append mode every call, so an `mv` is atomic and gateway-safe).

If you pass a tag, it's appended to the archive folder name —
e.g. `./rotate-telemetry.sh pre-simplification` produces
`reset-2026-05-12-032645-pre-simplification/`.

## Schedule the daily analyzer

```bash
# install
cp com.dalemugford.hermes.dynamic-tools-analyzer.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) \
    ~/Library/LaunchAgents/com.dalemugford.hermes.dynamic-tools-analyzer.plist

# confirm it's scheduled
launchctl print gui/$(id -u)/com.dalemugford.hermes.dynamic-tools-analyzer \
    | grep -E "state|next run"

# run once on demand (smoke test or ad-hoc)
launchctl kickstart gui/$(id -u)/com.dalemugford.hermes.dynamic-tools-analyzer
```

Outputs land in three places:

- `~/.hermes/plugins/dynamic-tools/reports/YYYY-MM-DD-HHMMSS-analysis.md`
  — the human-readable per-run report.
- `~/.hermes/state/dynamic-tools/learned_recommendations.json` —
  the latest machine-readable recommendation set. Overwritten each run.
- `~/.hermes/state/dynamic-tools/cron-logs/` — per-run JSON output,
  any stderr captured during a failed run, and `daily-summary.log` with
  one line per run for at-a-glance review.

## Review a week's worth of data

```bash
# Scan the summary log
cat ~/.hermes/state/dynamic-tools/cron-logs/daily-summary.log

# Open the most recent markdown report (now includes Harvest-Driven and
# Trigger-Keyword sections when harvest data is present)
ls -t ~/.hermes/plugins/dynamic-tools/reports/*.md | head -1 | xargs open

# Inspect the latest recommendations JSON — includes harvest_tool_promotion
# kinds alongside expanded_category and trigger_group recs
jq '.dampener_candidates, .recommendations, .trigger_keyword_candidates' \
   ~/.hermes/state/dynamic-tools/learned_recommendations.json
```

## Uninstall

```bash
launchctl bootout gui/$(id -u) \
    ~/Library/LaunchAgents/com.dalemugford.hermes.dynamic-tools-analyzer.plist
rm ~/Library/LaunchAgents/com.dalemugford.hermes.dynamic-tools-analyzer.plist
```

The script files and their outputs aren't touched by uninstall —
the LaunchAgent just stops being scheduled.
