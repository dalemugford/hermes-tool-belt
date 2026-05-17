# CLAUDE.md — dynamic-tools

Rules for working in this directory. The entry point for understanding what
this plugin is and where it's going is [STRATEGY.md](STRATEGY.md); current
architecture and status live in [PLAN.md](PLAN.md); user-facing docs in
[README.md](README.md).

## Scope of edits

- Changes belong **only** inside `plugins/dynamic-tools/`. Do not edit
  Hermes core, other plugins, profiles, or anything else in `~/.hermes/`
  while working on this plugin.
- The plugin is not yet publicly released. No back-compat is required —
  prefer the clean change over a shim.

## Branch discipline

- Two branches: `labs` (active work, data collection, experimentation) and
  `main` (runtime-only, what would ship).
- Develop on `labs`. Cherry-pick runtime-only changes to `main` when they
  earn promotion under the criteria in [STRATEGY.md](STRATEGY.md) §"Lab vs.
  Live."
- Check the current branch before editing.

## Where things live

- Mechanism (live-eligible): [__init__.py](__init__.py),
  [predictor.py](predictor.py), [presets.py](presets.py),
  [expand_tools.py](expand_tools.py), [logger_io.py](logger_io.py),
  [policy.yaml](policy.yaml)
- Adaptive layer (lab): [learned.py](learned.py), [analyze.py](analyze.py)
- Telemetry: `~/.hermes/state/dynamic-tools/predictions.jsonl` +
  `tool_calls.jsonl`
