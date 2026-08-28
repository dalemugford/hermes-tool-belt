# Tool Belt scripts

Supported operator and verification commands shipped with the plugin. Runtime
state lives under `$HERMES_HOME/state/tool-belt/`; generated reports and logs
are not tracked by Git.

Examples use `python3`; any Python 3 environment with PyYAML works. Set
`HERMES_HOME` for a non-default Hermes home. `daily-analysis.sh` also accepts
`HERMES_PYTHON` when a scheduler does not inherit the intended Python path.
The root Hermes profile is labelled `default`; named profiles keep their
directory names.

## Commands

| Script | Purpose | Typical use |
|---|---|---|
| [`configure.py`](configure.py) | Guided onboarding. Detects the state of every agent scope, explains what shaping would change, and writes the configuration through `hermes config`. | The first command to run after installing. Re-run any time. |
| [`bootstrap.py`](bootstrap.py) | Mode-aware first-install warm start. Uses live `expand_tools` evidence for cache-on scopes and session replay for cache-off scopes. | Optional, once after installation. |
| [`shape-ceiling.py`](shape-ceiling.py) | Builds per-scope promote/demote recommendations from recent sessions and writes the learned overlay. | Run after enough organic sessions; inspect with `--dry-run` first. |
| [`harvest-replay.py`](harvest-replay.py) | Replays existing Hermes sessions through the per-turn predictor and writes privacy-reduced synthetic telemetry. | Tune trigger coverage for cache-off scopes. |
| [`cache-freeze-replay.py`](cache-freeze-replay.py) | Measures frozen-tool-list efficacy and estimates cache cost from matched API-call positions. **Also a hard library dependency of `analyze.py`**, which imports it via `importlib` for the cache-aware savings section; its CLI is an optional focused diagnostic. | Investigate cache behavior or verify analyzer savings. |
| [`savings-report.py`](savings-report.py) | **Deprecated wrapper.** Kept for backward compatibility; delegates to the canonical engine in [`../savings.py`](../savings.py). Prefer `tool-belt savings`. | Legacy per-scope cache-on/off view. |
| [`../tool-belt`](../tool-belt) `savings` | Canonical, read-only savings command. Reports every enabled agent (or one via `--agent`) with separately-labeled **observed** and **projected** cohorts; `--json` for a stable schema. Backed by [`../savings.py`](../savings.py). | The supported way to inspect savings. |
| [`smoke-test.py`](smoke-test.py) | Exercises cache-on and cache-off behavior in isolated temporary state. | Before committing hook, freeze, expansion, or telemetry changes. |
| [`rotate-telemetry.sh`](rotate-telemetry.sh) | Moves live JSONL telemetry into a timestamped archive without stopping the gateway. | Start a clean measurement window. |
| [`daily-analysis.sh`](daily-analysis.sh) | Runs the analyzer and shaper for the root profile and named profiles with telemetry. | Run manually or from a scheduler. |
| [`../tests/seed_sessions.py`](../tests/seed_sessions.py) | Populates a throwaway Hermes home with telemetry generated from the scripted conversations in `tests/scripts/`, using the real policy resolver and predictor. | Demo or debug onboarding without a live gateway: `.venv/bin/python tests/seed_sessions.py --home /tmp/demo-home`, then run `configure.py` against it with `HERMES_HOME` set. Requires the development environment from `CONTRIBUTING.md` (PyYAML). |

Trigger-dampener regression coverage lives in
[`tests/test_trigger_dampeners.py`](../tests/test_trigger_dampeners.py) and runs
with the normal test suite. End-to-end onboarding coverage lives in
[`tests/test_onboarding_e2e.py`](../tests/test_onboarding_e2e.py).

## Common workflows

### Configure the plugin (start here)

```bash
python3 scripts/configure.py            # interactive
python3 scripts/configure.py --status   # read-only state report
```

`configure.py` discovers every `agent:platform` scope from telemetry, reports
which of four states it is in — `fresh`, `observing`, `ready`, `shaped` — and
offers the step that fits. Two paths are available on a fresh scope:

- **shape** — analyze the history already on disk, print a plain-language
  summary of what would become always-on and what would move to on-demand,
  and on confirmation write the learned overlay and set `learned_mode: apply`
  for that scope.
- **recommend** — leave tool loading untouched while telemetry accumulates,
  then re-run later. The command prints how many more sessions each scope
  needs; the minimum comes from `policy.yaml` `learning.shape_ceiling`.

Config is written only through `hermes config set` / `hermes config unset`;
`config.yaml` is never edited directly. Every write is preceded by its
`before → after` line and requires confirmation. If `hermes` is not on PATH
the command prints the exact commands to run by hand and exits 0.

Non-interactive flags for scripting and tests:

```bash
python3 scripts/configure.py --status
python3 scripts/configure.py --agent default --path recommend --yes
python3 scripts/configure.py --agent default --path shape --dry-run
python3 scripts/configure.py --reset default
```

`--status` never writes. `--dry-run` prints every diff and writes nothing —
neither files nor `hermes config` calls.

### Mine dampener and trigger-keyword candidates

```bash
python3 analyze.py --suggest-dampeners
python3 analyze.py --state-dir ~/.hermes/state/tool-belt/harvest \
    --suggest-trigger-keywords --suggest-dampeners
```

The analyzer mines 80-char message previews per (scope, trigger), extracts
2–4 word n-grams, and surfaces n-grams that show up frequently in
false-positive messages but rarely in true-positive ones (the inverse flow
mines `was_cut` previews for keyword candidates that would have covered a
cut tool). Output includes regex-ready patterns plus sample previews;
candidates already covered by an existing pattern are filtered out.
Tuning flags:

| Flag | Default | Meaning |
|---|---|---|
| `--dampener-min-support` | 3 | Minimum false-positive occurrences for a candidate |
| `--dampener-min-precision` | 0.8 | Minimum `fp / (fp + tp)` ratio |
| `--dampener-min-n` | 2 | Shortest n-gram length (words) |
| `--dampener-max-n` | 4 | Longest n-gram length (words) |
| `--dampener-max-candidates` | 10 | Cap on suggestions per (scope, trigger) |

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
./tool-belt savings                       # canonical: all agents, both cohorts
./tool-belt savings --agent=default --json # machine-readable, one agent
python3 scripts/cache-freeze-replay.py
python3 scripts/cache-freeze-replay.py --scope assistant-a:telegram
python3 scripts/savings-report.py --json   # deprecated wrapper
```

`tool-belt savings` is the public entry point. Pricing (`PRICE_TABLE`), the
token estimator, and the expand-round-trip overhead constant are single-sourced
in `savings.py`; `cache-freeze-replay.py` imports the price table from there and
`savings-report.py` re-exports the observed-cohort math — no duplicate tables.

The standard `analyze.py` report already includes cache-aware matched-
counterfactual figures. It computes them by importing `cache-freeze-replay.py`
as a library (via `importlib`, because of the hyphenated filename), so the
script is a hard analyzer dependency, not just a standalone tool — do not move
or remove it. Run these scripts directly when you need focused or
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
scripts/daily-analysis.sh              # analyze + shape in dry-run (default)
SHAPE_APPLY=1 scripts/daily-analysis.sh  # also let the shaper write learned.json
```

The shaper step runs with `--dry-run` by default: an unattended run reports
what it would change and logs `shape_pending`, but never rewrites the
`learned.json` you reviewed. Set `SHAPE_APPLY=1` to opt in to the write (its
effect on the live loadout is still gated by each scope's `learned_mode`).

Generated output is written to:

- `reports/<profile>/` inside the local plugin checkout (ignored by Git)
- `<state-dir>/learned_recommendations.json`
- `<state-dir>/learned.json` — only under `SHAPE_APPLY=1`, and only when
  shaping evidence changes
- `$HERMES_HOME/state/tool-belt/cron-logs/` (per-run shaper log and its
  machine-readable `*.shape.json`, kept when a run wrote or has changes
  pending)

Scheduler setup is environment-specific. Invoke `daily-analysis.sh` from the
scheduler appropriate to the host; no machine-specific scheduler configuration
is shipped in the repository. Set `HERMES_PYTHON` explicitly in sparse
scheduler environments when `python3` is not the interpreter with PyYAML.
