# Tool Belt scripts

Supported operator and verification commands shipped with the plugin. Runtime
state lives under `$HERMES_HOME/state/tool-belt/`; generated reports and logs
are not tracked by Git.

Examples use `python3`; any Python 3 environment with PyYAML works. Set
`HERMES_HOME` for a non-default Hermes home. The root Hermes profile is
labelled `default`; named profiles keep their directory names.

## Commands

| Script | Purpose | Typical use |
|---|---|---|
| [`configure.py`](configure.py) | Mode-setter. Detects every agent scope, sets the shaping mode (learning / history / off / observe / reset) and protected tools, and writes the configuration through `hermes config`. | The first command to run after installing. Re-run any time. |
| [`bootstrap.py`](bootstrap.py) | Posture-aware first-install warm start for narrowing (non-caching) scopes: live `expand_tools` evidence via the shaper plus session-history replay. Caching scopes are carry-all and are reported as "nothing to bootstrap". | Optional, once after installation. |
| [`shape-ceiling.py`](shape-ceiling.py) | Builds per-scope promote/demote recommendations from recent sessions and writes the learned overlay. | Run after enough organic sessions; inspect with `--dry-run` first. |
| [`replay-shaping.py`](replay-shaping.py) | Read-only chronological replay of one scope's telemetry, from an empty learned state through the real shaper. Reports convergence, ramp cost, implied `expand_tools` events, promotions and flap; `--window-days` and `--floor` accept comma lists to sweep. | Check a shaping cadence against your own history before changing the defaults. |
| [`harvest-replay.py`](harvest-replay.py) | Replays existing Hermes sessions through the per-turn predictor and writes privacy-reduced synthetic telemetry. | Tune trigger coverage for cache-off scopes. |
| [`../tool-belt`](../tool-belt) `savings` | Canonical, read-only savings command. Reports every enabled agent (or one via `--agent`) with separately-labeled **observed** and **projected** cohorts; `--json` for a stable schema. Backed by [`../savings.py`](../savings.py). | The supported way to inspect savings. |
| [`smoke-test.py`](smoke-test.py) | Exercises the carry-all (cache-on) and narrowing (cache-off) postures in isolated temporary state. | Before committing hook, carry-all, expansion, or telemetry changes. |
| [`rotate-telemetry.sh`](rotate-telemetry.sh) | Moves live JSONL telemetry into a timestamped archive without stopping the gateway. | Start a clean measurement window. |
| [`../tests/seed_sessions.py`](../tests/seed_sessions.py) | Populates a throwaway Hermes home with telemetry generated from the scripted conversations in `tests/scripts/`, using the real policy resolver and predictor. | Demo or debug onboarding without a live gateway: `.venv/bin/python tests/seed_sessions.py --home /tmp/demo-home`, then run `configure.py` against it with `HERMES_HOME` set. Requires the development environment from `CONTRIBUTING.md` (PyYAML). |

Trigger-dampener regression coverage lives in
[`tests/test_trigger_dampeners.py`](../tests/test_trigger_dampeners.py) and runs
with the normal test suite. End-to-end onboarding coverage lives in
[`tests/test_onboarding_e2e.py`](../tests/test_onboarding_e2e.py).

## Common workflows

### Configure the plugin (start here)

```bash
hermes tool-belt configure              # canonical form
python3 scripts/configure.py            # same code, run directly
python3 scripts/configure.py --status   # read-only state report
```

`configure.py` is a mode-setter. It discovers every `agent:platform` scope
from telemetry; interactively you pick an agent, then either **Protected
tools** (a picker over the agent's inventory — selections are written to
`plugins.entries.tool-belt.settings.always_carry`, always carried and never shaped) or
**Tool shaping options** (pick channels, then a mode):

- **learning** — shaping on (`learned_mode: apply`); the plugin shapes
  automatically from future usage. No history run.
- **history** — shaping on, plus a shaping pass over the sessions already
  recorded, shown as a plain-language diff and applied on confirmation.
- **off** — observation mode (`learned_mode: recommend`): every enabled
  tool is carried, telemetry keeps accumulating, and the learned overlay
  is kept but not applied.
- **observe** — `off` plus the scope's `bypass_rate` set to `1.0`: a
  full-ceiling observation baseline.
- **reset** — `off` plus the agent's learned overlay cleared
  (`--mode reset --agent <agent>`).

`--status` classifies each scope from its settings: the row shows
`shaping ON/OFF` with, once a scope is shaped, how many tools it carries
each session beyond its pins and how many are available by expansion;
`learning` while evidence accumulates; `observing` when shaping is off.

Config is written only through `hermes config set` / `hermes config unset`;
`config.yaml` is never edited directly. Every write is preceded by its
`before → after` line and requires confirmation. If `hermes` is not on PATH
the command prints the exact commands to run by hand and exits 0.

Non-interactive flags for scripting and tests:

```bash
python3 scripts/configure.py --status
python3 scripts/configure.py --agent default --mode learning --yes
python3 scripts/configure.py --agent default --mode history --dry-run
python3 scripts/configure.py --agent default --mode off --yes
python3 scripts/configure.py --agent default --channel slack --mode off --yes   # one channel only
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
mines `was_expand_only` previews for keyword candidates that would have
covered an expand-only tool). Output includes regex-ready patterns plus sample previews;
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

Defaults come from `policy.yaml` under `learning.shape_ceiling`
(`window_days: 7`, `demote_min_sessions_no_use: 2`, `promote_min_sessions: 1`,
`promote_min_calls: 2`, `demote_k: 1.5`); command-line flags override them for
an individual run. `--window-days N` sets the day window for the run. Applied
recommendations are written to `$HERMES_HOME/state/tool-belt/learned.json`.

Every demote/promote decision is priced at the scope's **measured** per-event
expand cost (`shaping.measured_expand_penalty`) — the same pricing
`auto_shape_run` uses in production — not the flat `EXPAND_ROUND_TRIP_TOKENS`
fallback. `configure.py`'s history-mode preview uses the identical call, so an
interactive review and an automatic pass never disagree on economics. The
`--json` / `--json-file` porcelain document includes `expand_round_trip_tokens`
per scope so the priced basis is inspectable, not just the resulting
recommendation.

### Replay a shaping cadence against your own history

```bash
python3 scripts/replay-shaping.py --scope default:telegram
python3 scripts/replay-shaping.py --scope default:telegram --window-days 7,14,30
python3 scripts/replay-shaping.py --scope default:telegram --floor 2,5,20
```

Read-only: it starts from an empty learned state and walks one scope's
telemetry forward in time through the real shaper, session by session, so the
result is what the shipped engine would actually have done. Per setting it
reports where the carried set converges and what it costs per turn, the ramp
cost of getting there, how many `expand_tools` events the cadence implies
(counting **primary** model dispatches only — nested and secondary dispatches
are not evidence of the model reaching for a tool), how many promotions fired,
and whether any tool flapped. Passing comma-separated values to `--window-days`
or `--floor` sweeps them and prints the settings side by side.

Its blind spot is worth stating: the replay can only score what the recorded
traffic did, and a carried tool is an invitation — a tool on the wire is
sometimes used *because* it is visible. A replay cannot show the sessions that
would have gone differently had a demoted tool still been present, so read it
as a cost model, not a behavioral one.

### Analyze cache behavior

```bash
./tool-belt savings                       # canonical: all agents, both cohorts
./tool-belt savings --agent=default --json # machine-readable, one agent
python3 cache_replay.py
python3 cache_replay.py --scope assistant-a:telegram
```

`tool-belt savings` is the public entry point. Pricing (`PRICE_TABLE`), the
token estimator, and the expansion overhead (measured per cohort, with
`EXPAND_ROUND_TRIP_TOKENS` as the thin-data fallback) are single-sourced in
`savings.py`; `cache_replay.py` imports the price table from there — no
duplicate tables.

On a caching scope the plugin is carry-all (full ceiling, no `expand_tools`,
no mid-session mutation), so `cache_replay.py --scope <caching scope>` should
report zero mutations of every kind; that is the quickest check that the
posture held.

The standard `analyze.py` report already includes cache-aware matched-
counterfactual figures, computed by importing `cache_replay.py` as a library.
Run these commands directly when you need focused or machine-readable
output.

### Warm-start an existing installation

```bash
python3 scripts/bootstrap.py
python3 scripts/bootstrap.py --profile assistant-a
```

The bootstrap command discovers the root Hermes profile and named profiles
under `$HERMES_HOME/profiles/`. It does not modify plugin policy; the shaper
writes only the learned overlay. Scopes whose primary provider prompt-caches
are carry-all and produce no expansion evidence by design; bootstrap reports
them as "nothing to bootstrap" rather than waiting for data that never comes.

### Validate behavior

```bash
python3 tests/run_tests.py
python3 scripts/smoke-test.py
```

The smoke test checks the cache-off (narrowing) invariants — sticky
residency, expansion attribution, cross-session isolation, session lifecycle
— and the cache-on (carry-all) contract: every prediction row is
`cache_on_carry_all` with ceiling == shipped, `expand_tools` is absent from
the wire, exactly one `tool_list_hash` per session, no `expand_tools` calls,
and `provider_caches` present on every `api_calls.jsonl` row.

### Rotate telemetry

```bash
scripts/rotate-telemetry.sh optional-tag
```

The script moves live `predictions.jsonl`, `tool_calls.jsonl`, and
`api_calls.jsonl` files into
`$HERMES_HOME/state/tool-belt/archive/reset-<timestamp>[-<tag>]/`. The plugin
recreates each file on the next append.

### Ongoing shaping and analysis

Scopes with `learned_mode: apply` are auto-shaped in-process by the plugin
itself at session end (default once per 24h per scope; opt out with
`auto_shape: false`, interval via `auto_shape_interval_hours`). No scheduled
task is needed. For on-demand diagnostics, run the analyzer or shaper
directly: `python3 analyze.py`, `python3 scripts/shape-ceiling.py --dry-run`.
