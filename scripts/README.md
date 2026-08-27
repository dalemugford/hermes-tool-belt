# Tool Belt scripts

Supported operator and verification commands shipped with the plugin. Runtime
state lives under `$HERMES_HOME/state/tool-belt/`; generated reports and logs
are not tracked by Git.

## Commands

| Script | Purpose | Typical use |
|---|---|---|
| [`bootstrap.py`](bootstrap.py) | Mode-aware first-install warm start. Uses live `expand_tools` evidence for cache-on scopes and session replay for cache-off scopes. | Optional, once after installation. |
| [`shape-ceiling.py`](shape-ceiling.py) | Builds per-scope promote/demote recommendations from recent sessions and writes the learned overlay. | Run after enough organic sessions; inspect with `--dry-run` first. |
| [`harvest-replay.py`](harvest-replay.py) | Replays existing Hermes sessions through the per-turn predictor and writes privacy-reduced synthetic telemetry. | Tune trigger coverage for cache-off scopes. |
| [`cache-freeze-replay.py`](cache-freeze-replay.py) | Measures frozen-tool-list efficacy and estimates cache cost from matched API-call positions. | Investigate cache behavior or verify analyzer savings. |
| [`savings-report.py`](savings-report.py) | Produces the compact, independently checkable token-savings report documented in `docs/SAVINGS.md`. | Inspect all scopes or a selected date/scope window. |
| [`check-tool-drift.py`](check-tool-drift.py) | Finds tool names present in the observed ceiling but absent from policy. | After Hermes/plugin upgrades or toolset changes. |
| [`smoke-test.py`](smoke-test.py) | Exercises cache-on and cache-off behavior in isolated temporary state. | Before committing hook, freeze, expansion, or telemetry changes. |
| [`rotate-telemetry.sh`](rotate-telemetry.sh) | Moves live JSONL telemetry into a timestamped archive without stopping the gateway. | Start a clean measurement window. |
| [`daily-analysis.sh`](daily-analysis.sh) | Runs the analyzer and shaper for the root profile and named profiles with telemetry. | Run manually or from a scheduler. |

Trigger-dampener regression coverage lives in
[`tests/test_trigger_dampeners.py`](../tests/test_trigger_dampeners.py) and runs
with the normal test suite.

## Common workflows

### Preview or apply between-session shaping

```bash
python3 scripts/shape-ceiling.py --dry-run
python3 scripts/shape-ceiling.py
```

Defaults come from `policy.yaml` under `learning.shape_ceiling`; command-line
flags override them for an individual run. Applied recommendations are written
to `$HERMES_HOME/state/tool-belt/learned.json`.

### Analyze cache behavior

```bash
python3 scripts/cache-freeze-replay.py
python3 scripts/cache-freeze-replay.py --scope assistant-a:telegram
python3 scripts/savings-report.py --json
```

The standard `analyze.py` report already includes cache-aware matched-
counterfactual figures. Run these scripts directly when you need focused or
machine-readable output.

### Warm-start an existing installation

```bash
python3 scripts/bootstrap.py
python3 scripts/bootstrap.py --profile assistant-a
```

The bootstrap command discovers the root Hermes profile and named profiles
under `$HERMES_HOME/profiles/`. It does not modify plugin policy; the shaper
writes only the learned overlay.

### Validate behavior

```bash
python3 tests/run_tests.py
python3 scripts/smoke-test.py
python3 scripts/check-tool-drift.py
```

The smoke test currently checks eight cache-off invariants and five cache-on
invariants, including frozen snapshot reuse and expansion persistence.

### Rotate telemetry

```bash
scripts/rotate-telemetry.sh optional-tag
```

The script moves live `predictions.jsonl`, `tool_calls.jsonl`, and
`api_calls.jsonl` files into
`$HERMES_HOME/state/tool-belt/archive/reset-<timestamp>[-<tag>]/`. The plugin
recreates each file on the next append.

### Run periodic analysis

```bash
scripts/daily-analysis.sh
```

Generated output is written to:

- `reports/<profile>/` inside the local plugin checkout (ignored by Git)
- `<state-dir>/learned_recommendations.json`
- `<state-dir>/learned.json` when shaping evidence changes
- `$HERMES_HOME/state/tool-belt/cron-logs/`

Scheduler setup is environment-specific. Invoke `daily-analysis.sh` from the
scheduler appropriate to the host; no machine-specific scheduler configuration
is shipped in the repository.
