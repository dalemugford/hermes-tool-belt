# dynamic-tools scripts

Operational scripts that live next to the plugin. Each is safe to run by
hand or under a scheduler. None mutate `learned.json` or other policy
state — those changes remain manual (see [AUTO-APPLY-PLAN.md](../AUTO-APPLY-PLAN.md)).

## Inventory

| Script | Purpose | When to run |
|---|---|---|
| [`rotate-telemetry.sh`](rotate-telemetry.sh) | Archive current `predictions.jsonl` + `tool_calls.jsonl` to `state/dynamic-tools/archive/reset-<ts>/`. Gateway-safe. | After a meaningful change to plugin behavior, when you want a clean measurement window. |
| [`daily-analysis.sh`](daily-analysis.sh) | Run analyzer with full recommendations + dampener mining, drop a markdown report + recommendations JSON, append a one-line summary to `cron-logs/daily-summary.log`. | Scheduled twice daily via launchd, or by hand any time. Skips cleanly when telemetry is empty. |
| [`check_trigger_dampeners.py`](check_trigger_dampeners.py) | Smoke checks: dampeners still veto + dampeners survive the learned-merge path. | Before committing changes to policy.yaml or learned merge logic. |
| [`com.dalemugford.hermes.dynamic-tools-analyzer.plist`](com.dalemugford.hermes.dynamic-tools-analyzer.plist) | launchd LaunchAgent template; runs `daily-analysis.sh` at 00:00 and 12:00 local time. | Install once; the system handles scheduling. |

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

# Open the most recent markdown report
ls -t ~/.hermes/plugins/dynamic-tools/reports/*.md | head -1 | xargs open

# Inspect the latest recommendations JSON
jq '.dampener_candidates, .recommendations' \
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
