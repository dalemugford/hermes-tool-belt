# Changelog

All notable changes to Hermes Tool Belt are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses CalVer (`YYYY.M.D` with an optional prerelease suffix).

## [Unreleased]

The initial public capabilities of the plugin:

### Added

- **Audited always-carry pin set (2026-08-30 audit).** The shipped
  `policy.yaml` `always_carry` is now exactly seven tools: `expand_tools`,
  `tool_search`, `tool_describe`, `tool_call`, `clarify`, `skills_list`,
  `skill_view`. Dropped: `send_message` (not agent-callable on current
  Hermes — the pin was permanently inert) and `todo` (evidence-ruled;
  users can pin it back via `plugins.tool-belt.always_carry`).
- **Bridge tools join the pass-through.** Hermes' Tool Search bridge
  (`tool_search`/`tool_describe`/`tool_call`) is now treated like MCP
  pass-through: outside the built-in partition, never demotable, never
  counted as adaptive carry, never listed in the expand-only manifest, and
  never named by a shaper recommendation. Previously ordinary no-use
  evidence could demote the bridge and silently sever ALL access to the
  deferred MCP/plugin catalog. The policy pin remains as pure-data backup
  for environments where `tools.tool_search` isn't importable — the mirror
  image of `_pin_expand_tools_visible`, which protects `expand_tools`
  against the bridge in the opposite direction. Bridge tools are recorded
  in the pass-through telemetry field, not the carrying strata.
- **Zero-config full-start default.** On a scope with no learned state,
  Tool Belt now carries **everything Hermes enables** — active == the
  enabled ceiling — and evidence-driven demotion (the existing no-use
  thresholds) is the shaping motion that moves tools to expand-only over
  time. This flips the old defaults: unknown enabled built-ins are carried
  (not expand-only), the shipped curated `carry` warm-start list is retired
  from `policy.yaml` and runtime resolution, and the default `learned_mode`
  is now `apply` (automatic shaping); `recommend` becomes the documented
  opt-in observe/trial mode. Explicit configs win as always, legacy aliases
  are unchanged, and triggers/expand_tools/fail-open semantics are
  untouched. The plugin's internal `enabled` default also flips to `true`
  (Hermes' `plugins.enabled` list still gates loading; a config-disabled
  plugin now logs a one-shot warning instead of going silently inert).
- **Per-agent `always_carry` config pinning.** New key
  `plugins.tool-belt.always_carry: [tool, ...]` (plus additive
  `channels.<scope>.always_carry` — union semantics, scope adds and never
  removes). Pins union with the shipped structural baseline, are validated
  against the enabled ceiling (a disabled/unknown name is inert, with one
  clear warning naming it), and are undemotable by construction — the
  runtime ignores learned demotions naming a pin and the shaper/auto-shape
  engine filters such candidates before writing. The Protected-tools picker
  in `tool-belt configure` reads and writes these pins interactively.
- **In-process periodic auto-shaping.** Scopes whose `learned_mode` resolves
  to `apply` are now shaped automatically by the plugin itself — triggered
  from the session-end path, debounced to once per 24h per scope
  (`auto_shape_interval_hours` to override), with the same evidence
  thresholds as the shaper script. No system-level scheduler is needed;
  `auto_shape: false` is the global opt-out. Observe/recommend scopes are
  never auto-written. Each automatic apply is logged and recorded in the
  scope's learned `shaping` block (`source: "auto"`, `applied_at`,
  `last_auto_shape_at`); `configure --status` shows a shaped
  scope's learned carry/expansion counts. The shaper's compute/merge core moved
  into the shared package module `shaping.py`, with
  `scripts/shape-ceiling.py` remaining as the CLI wrapper (flags, porcelain
  JSON, and human output unchanged).
- **Automatic inventory reconciliation.** The session-end auto pass now
  reconciles learned state and config pins against the install's tool
  REGISTRY (absence from one scope's platform ceiling is not "missing" —
  the tool may be live on another platform). A referenced tool absent from
  the registry has its first observed absence tracked
  (`state/tool-belt/inventory.json`); after a 7-day grace it is pruned
  automatically from `learned.json` (carry/expand-only assignments, shaping
  evidence, trigger-overlay entries) and from `always_carry` config pins —
  the config edit goes through Hermes' own config-write machinery
  (in-process, atomic; a machinery failure logs a warning and skips, never
  propagates), with one INFO line per removal. A tool that reappears within
  the grace resets the clock; after cleanup a reappearing tool starts a
  fresh journey — carried, per the full-start contract (newly added tools
  need no special handling for the same reason, now regression-locked for
  already-shaped scopes). Fail-open throughout: no resolvable registry
  means no reconciliation.
- **Learned trigger overlay (automatic anticipation).** Deterministic
  trigger anticipation now learns automatically across arbitrary
  installs/tools: an additive per-scope overlay stored in the scope's
  learned `triggers` list (schema v2) unions with the shipped `policy.yaml`
  trigger groups at preset resolution — the shipped file is never edited.
  Two sources, both in the session-end auto pass: (a) the analyzer's
  expansion-evidence keyword mining is auto-applied above a STRICT bar
  (support ≥ 4, precision ≥ 0.90, ≤ 3 keywords per tool — stricter than the
  recommend-only defaults); (b) a tool demoted with no trigger coverage
  anywhere gets a conservative word-boundary name-token trigger (generic
  tokens stoplisted, minimum length 4; skipped entirely when nothing
  distinctive remains). The overlay is structurally activation-only — it
  can only activate `expand_only` tools (`T = triggered ∩ X`), never
  demote, disable, or touch `always_carry`, and can never reference a tool
  outside the enabled ceiling. Exclude-keyword dampeners apply to overlay
  triggers exactly as to policy ones (a mined entry inherits the covering
  policy group's excludes). Inspectable: `configure --status` shows
  "auto-learned triggers: N" per scope; a scope reset clears its overlay;
  inventory reconciliation prunes entries for vanished tools. The hot path
  stays model-free and deterministic — "A plugin promising to save you
  tokens shouldn't quietly spend them."
- **`expand_tools` response clarity.** The "not enabled" note now states
  the cause and its fixability — the tools are excluded by the operator's
  `platform_toolsets` ceiling, Tool Belt cannot restore them, and enabling
  them requires a Hermes config change. Persistence wording is honest per
  cache mode: cache-on responses say the expansion "persists for this
  session" (and no longer emit a misleading `sticky: {enabled: false}`
  block); sticky wording is reserved for cache-off, where the sticky block
  still reports residency.
- **`hermes tool-belt ...` subcommand.** `register(ctx)` now calls Hermes' `ctx.register_cli_command()`, so `hermes tool-belt savings` and `hermes tool-belt configure` work alongside the bare `tool-belt` launcher. Flags are forwarded verbatim to the same `savings_cli` entry point the launcher execs, so the two invocation forms cannot drift. Registration is optional and fail-open: on a Hermes without `register_cli_command`, the tool and hooks register exactly as before.
- **Canonical savings CLI (`tool-belt savings`).** Reports observed and counterfactual token savings across all enabled agents or one selected agent, with deterministic JSON, full-schema replay, confidence-aware percentages, and dollars only for provably metered routes. Reports echo the honored `--since` window (`Window: since …` in text, a `since` key in JSON). The launcher honors `HERMES_PYTHON` and the local Hermes environment without importing runtime hooks. The session-input percentage is shown only against provider-reported usage; when historical sessions predate telemetry, only the explicitly-labeled schema-only reduction is shown (reconstructed context omits tool results, system prompt, and per-API-call accumulation, so it never backs a percentage).
- **configure = mode-setter (`tool-belt configure`).** One command from
  install to a working configuration: pick the agent, then Protected tools
  (a spacebar picker over the agent's observed inventory, written to
  `plugins.tool-belt.always_carry`) or Tool shaping — channels, then a mode:
  On (learning: shape from future usage), On (use history: shape from
  recorded sessions now, with a compact named-tools diff), or Off
  (observation; the learned overlay is kept but not applied).
  Non-interactive: `--mode learning|history|off` (the pre-1.0
  `--path`/`--reset` spellings remain as hidden aliases). `--status` prints
  `Tool Belt: enabled|disabled` plus one `shaping ON/OFF (…)` row per
  agent scope. Configuration is written only through
  `hermes config set`/`unset`, every write is preceded by a
  `before → after` diff and an explicit confirmation, and
  `--status`/`--dry-run` never write; guidance echoes the invocation form
  actually used. Degrades cleanly to printing hand-run commands when the
  `hermes` CLI is unavailable.
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
- **Release checklist (`docs/RELEASING.md`).** The reproducible steps for
  cutting a release: pre-release verification on a clean clone, a
  clean-profile install test under a throwaway `HERMES_HOME`, behavioral
  spot-checks, documentation reconciliation, version and changelog
  finalization, tagging, and post-release confirmation.
- **Privacy documentation (`docs/PRIVACY.md`).** What telemetry records and
  what it never records, where state lives and under whose permissions, the
  append-only retention model, how to disable collection, and how to erase
  every file the plugin writes.
- **Onboarding test harness (`tests/seed_sessions.py`, `tests/scripts/`,
  `tests/test_onboarding_e2e.py`).** Scripted conversations are run through the
  real policy resolver and predictor to synthesise telemetry into a throwaway
  Hermes home, so the whole onboarding arc — scope discovery, the four-state
  machine, both paths, the learned-overlay write, and per-scope independence —
  is exercised end to end with no gateway, no provider, and no messaging
  platform. Each script declares the trigger groups it expects, so a policy
  regression fails the seed rather than producing quietly-wrong fixtures, and
  rows are built through `logger_io.PredictionRecord` so the harness inherits
  the production schema by construction. The seeder doubles as an operator
  command for demoing onboarding by hand.

### Removed

- **`scripts/daily-analysis.sh`.** In-process auto-shaping replaces the
  scheduled shaper run, and the analyzer reports it generated had no
  scheduled consumer. Run `analyze.py` or `scripts/shape-ceiling.py`
  directly for on-demand diagnostics.

### Changed

- **Simplified `learned_mode` to two values: `recommend` and `apply`.** The
  default is `apply` (full-start — see the Added entry above), which merges
  the learned overlay during preset resolution. `recommend` is the opt-out
  observe mode: the overlay is kept but never merged, and recommendations
  flow through the analyzer and shaper for human review. Existing configs
  keep working: legacy values migrate automatically at load — `off` →
  `recommend`, and `auto` / `audit` → `apply`. No config edit is required.

### Fixed

- **Test-harness demo interpreter.** The manual session-seeding command now
  uses the project development environment (`.venv/bin/python`), where the
  declared PyYAML dependency is installed, instead of assuming the system
  Python provides it.

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
