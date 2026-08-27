# Changelog

All notable changes to Hermes Tool Belt are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses CalVer (`YYYY.M.D` with an optional prerelease suffix).

## [Unreleased]

The initial public capabilities of the plugin:

### Added

- **Cache-aware session freezing.** Under prefix-cache-friendly providers,
  the selected tool set stays stable across normal turns. Model-requested
  expansion is the explicit cache-breaking exception.
- **Cache-off per-turn narrowing.** Under cache-off providers, the tool set is
  narrowed per turn, where re-selection carries no cache penalty.
- **`expand_tools` recovery.** A meta-tool that restores tools mid-session on
  demand, by category or by tool name, with fuzzy suggestions for near matches.
- **Deterministic policy and trigger shaping.** A hand-curated policy
  (`policy.yaml`) resolves the per-scope loadout from always-on tools and
  intent triggers, with no model call in the hot path.
- **Learned between-session shaping.** An optional overlay promotes and demotes
  tools for future sessions from real `expand_tools` evidence and is applied
  only when `learned_mode: apply` is set.
- **Telemetry, analyzer, and savings reporting.** Append-only telemetry records
  every narrowing decision; the analyzer summarizes usage and reports token
  savings with an optional A/B baseline cohort.
- **Fail-open behavior.** Any internal failure leaves the original Hermes tool
  set fully available.
- **Strict `platform_toolsets` ceiling.** The plugin only narrows within, and
  restores from, the operator's configured ceiling; it never admits
  unsanctioned tools.
- **Clean native test environment.** A minimal `requirements-dev.txt` (PyYAML
  only) lets contributors run `python tests/run_tests.py` in a fresh virtual
  environment without the full Hermes runtime.
- **Continuous integration.** A GitHub Actions workflow
  (`.github/workflows/ci.yml`) runs the test suite on Python 3.11 and 3.12 for
  every push and pull request to `main`.

### Changed

- **Simplified `learned_mode` to two values: `recommend` and `apply`.** The
  default is `recommend`, which never merges `learned.json` into the live
  preset (recommendations still flow through the analyzer and shaper for human
  review). `apply` merges the learned overlay during preset resolution.
  Existing configs keep working: legacy values migrate automatically at load —
  `off` → `recommend`, and `auto` / `audit` → `apply`. No config edit is
  required.

### Fixed

- **Plugin Doctor registration validation.** `register()` no longer returns
  early when the plugin is disabled in config, so `hermes plugins doctor`
  can validate that the declared `expand_tools` tool and all five hooks are
  actually registered. Functional disabling is unchanged — each hook and the
  narrowing patch still guard on `enabled` internally.
- **Portable profile discovery.** Root-profile telemetry and session harvests
  now use Hermes' neutral `default` identity instead of an installation-specific
  agent name. Named profiles, custom `HERMES_HOME` paths (including spaces),
  and missing state/session directories are covered by regression tests.
- **Portable Python selection.** Bootstrap child processes reuse the invoking
  interpreter, while scheduled analysis uses `python3` or an explicit
  `HERMES_PYTHON` instead of assuming a Hermes source checkout and venv live
  under the user's home directory.
- **Surface audit summary.** `docs/SURFACE_AUDIT.md` documents the audited
  product surface — core features, secondary capabilities, operator tiers —
  and the cleanup decisions carried into this release cycle.
