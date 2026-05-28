# CLAUDE.md — dynamic-tools

Rules for working in this directory. The entry point for understanding what
this plugin is and where it's going is [STRATEGY](docs/02.strategy-2026-05-12.md);
current architecture and status live in [PLAN](docs/03.plan-2026-05-12.md);
user-facing docs in [README.md](README.md).

## Scope of edits

- Changes belong **only** inside `plugins/dynamic-tools/`. Do not edit
  Hermes core, other plugins, profiles, or anything else in `~/.hermes/`
  while working on this plugin.
- The plugin is not yet publicly released. No back-compat is required —
  prefer the clean change over a shim.

## Branch discipline (trunk-based, as of 2026-05-17)

- Develop on `main`. It's the trunk and the single source of truth.
- Release stability comes from tags (`vX.Y.Z`), not from a parallel branch.
- Features that aren't safe to enable by default ship behind config flags
  (`enabled: false`, `learned_mode: off`, etc.), not behind branch quarantine.
- The `labs` branch is reserved for the rare case where something is
  genuinely unsafe to ship even off-by-default (e.g., live soak-testing
  `apply.py` against real `learned.json`). Otherwise it stays unused.

The previous split (`labs` = active dev, `main` = curated runtime) was
abandoned because every feature was already gated by config or by being
an opt-in script; the branch separation added cherry-pick friction without
adding real safety.

## Where things live

- Mechanism: [__init__.py](__init__.py), [predictor.py](predictor.py),
  [presets.py](presets.py), [expand_tools.py](expand_tools.py),
  [logger_io.py](logger_io.py), [policy.yaml](policy.yaml)
- Adaptive layer (off-by-default): [learned.py](learned.py)
- Dev / ops tools: [analyze.py](analyze.py), [scripts/](scripts/)
- Telemetry: `~/.hermes/state/dynamic-tools/predictions.jsonl` +
  `tool_calls.jsonl`
- Harvest output (separate from live telemetry):
  `~/.hermes/state/dynamic-tools/harvest/`
