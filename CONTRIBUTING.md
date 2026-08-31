# Contributing to Hermes Tool Belt

Thanks for your interest in improving the plugin. This guide covers what the
repository is for and what a change is expected to include.

## Scope

Tool Belt is a plugin for [Hermes Agent](https://hermes-agent.nousresearch.com)
that adaptively narrows the tool set sent to the model. Contributions should
stay within that purpose:

- Change only files inside this repository. Do not edit Hermes core, profile
  configuration, or live Tool Belt state as part of a plugin change.
- Preserve fail-open behavior: an internal failure must leave the original
  Hermes tool set available.
- Preserve the configured `platform_toolsets` ceiling. The plugin may narrow
  within it and restore from it; it must never admit unsanctioned tools.
- Treat public configuration keys, telemetry fields, and learned-state shapes
  as contracts. Prefer a clean migration over indefinite compatibility code.

## Branches and commits

- `main` is the trunk and source of truth; develop against it.
- Keep incomplete or experimental behavior disabled by configuration rather
  than living on a separate branch.
- Write focused commits with clear messages that describe the user-facing or
  behavioral effect.

## Changelog

Every user-facing change updates `CHANGELOG.md` under `Unreleased`, using the
Keep a Changelog headings (`Added`, `Changed`, `Fixed`, `Removed`).

## Development environment

The plugin and its tests depend only on the Python standard library plus
PyYAML. To run the suite outside the Hermes runtime, create a virtual
environment and install the dev requirements:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python tests/run_tests.py
```

Python 3.11+ is the realistic target (the code uses
`from __future__ import annotations` and `dict[str, Any]` syntax).

## Verification

Run the full suite and the smoke test for every change:

```bash
.venv/bin/python tests/run_tests.py
.venv/bin/python scripts/smoke-test.py
```

`tests/run_tests.py` also runs in the clean venv above; the smoke test and
diagnostics below exercise the plugin against the full Hermes runtime.

Run targeted diagnostics when the mechanism they cover changes:

```bash
.venv/bin/python scripts/shape-ceiling.py --dry-run
```

Verify results independently — read the output, don't assume a pass. Never
commit generated telemetry, reports, caches, bytecode, learned state, or
machine-specific scheduler configuration.
