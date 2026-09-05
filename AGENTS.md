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
- Update `CHANGELOG.md` under `Unreleased` for every user-facing change.

## Repository map

- Runtime hooks and cache posture (carry-all on caching providers, per-turn
  narrowing on non-caching ones): `__init__.py`
- Carrying-strata partition helpers: `carrying.py`
- Prediction and policy: `predictor.py`, `presets.py`, `policy.yaml`
- Dynamic recovery: `expand_tools.py`
- Learned overlay: `learned.py`
- Telemetry writes: `logger_io.py`
- Telemetry analysis: `analyze.py`
- Analyzer cache-cost library: `cache_replay.py` (imported by `analyze.py`;
  also a standalone CLI)
- Savings engine and CLI: `savings.py`, `savings_cli.py`, `tool-belt` launcher
- Supported operator commands: `scripts/`
- Tests: `tests/`
- Architecture: `docs/ARCHITECTURE.md`
- Configuration reference: `docs/CONFIGURATION.md`
- Savings methodology: `docs/SAVINGS.md`
- Known limitations: `docs/KNOWN_ISSUES.md`
- Privacy contract: `docs/PRIVACY.md`
- Release checklist: `docs/RELEASING.md`
- Surface audit summary (partly superseded, historical): `docs/SURFACE_AUDIT.md`

## Verification

Create the repository development environment as described in
`CONTRIBUTING.md`, then run the complete suite for every code change:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python tests/run_tests.py
.venv/bin/python scripts/smoke-test.py
```

Run targeted diagnostics when their mechanism changes:

```bash
.venv/bin/python scripts/shape-ceiling.py --dry-run
```

Never commit generated telemetry, reports, caches, bytecode, learned state, or
machine-specific scheduler configuration.
