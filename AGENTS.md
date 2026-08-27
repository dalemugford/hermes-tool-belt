# Tool Belt repository guide

The code is the source of truth. Keep behavior and documentation aligned in
the same change.

## Scope

- Make changes only inside this repository unless the task explicitly expands
  scope.
- Do not edit Hermes core, profile configuration, or live Tool Belt state while
  changing the plugin.
- Preserve fail-open behavior: an internal Tool Belt failure must leave the
  original Hermes tool set available.
- Preserve the configured `platform_toolsets` ceiling: Tool Belt may narrow the
  ceiling and restore tools from it, never add unsanctioned tools.
- Treat public configuration keys, telemetry fields, and learned-state shapes
  as contracts. Prefer a clean migration over indefinite compatibility code
  when a contract must change.

## Branch and release discipline

- Develop on `main`; it is the trunk and source of truth.
- Use release tags for stable snapshots.
- Keep incomplete or experimental behavior disabled by configuration rather
  than maintaining a second implementation branch.
- Update `CHANGELOG.md` under `Unreleased` for user-facing changes once the
  changelog exists.

## Repository map

- Runtime hooks and cache behavior: `__init__.py`
- Prediction and policy: `predictor.py`, `presets.py`, `policy.yaml`
- Dynamic recovery: `expand_tools.py`
- Learned overlay: `learned.py`
- Telemetry writes: `logger_io.py`
- Telemetry analysis: `analyze.py`
- Supported operator commands: `scripts/`
- Tests: `tests/`
- Architecture: `docs/ARCHITECTURE.md`
- Configuration reference: `docs/CONFIGURATION.md`
- Savings methodology: `docs/SAVINGS.md`
- Known limitations: `docs/KNOWN_ISSUES.md`

## Verification

Run the complete suite for every code change:

```bash
~/.hermes/hermes-agent/venv/bin/python3 tests/run_tests.py
~/.hermes/hermes-agent/venv/bin/python3 scripts/smoke-test.py
```

Run targeted diagnostics when their mechanism changes:

```bash
~/.hermes/hermes-agent/venv/bin/python3 scripts/check-tool-drift.py
~/.hermes/hermes-agent/venv/bin/python3 scripts/shape-ceiling.py --dry-run
```

Never commit generated telemetry, reports, caches, bytecode, learned state, or
machine-specific scheduler configuration.
