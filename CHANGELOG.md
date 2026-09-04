# Changelog

All notable changes to Hermes Tool Belt are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses CalVer (`YYYY.M.D` with an optional prerelease suffix).

## [Unreleased]

## [2026.9.5] - 2026-09-04

### Fixed

- **Blocklisted providers/models now narrow from the first session.** A route
  named in `cache_auto.providers_off_models` used to ship one full carry-all
  session before the `post_api_request` lock landed, because the session's
  posture was pinned at first dispatch under the unlocked-bucket default
  (`on`). `_resolve_posture_for_provider` now consults the blocklist when a
  `scope|provider` bucket is still unlocked, so a known-uncached provider or
  model resolves `off` before any API call has run. The detection-lock path
  and the resolver now share one predicate (`_provider_or_model_blocklisted`),
  so they can't disagree about what counts as blocklisted. The configured
  primary model is attributed only to the primary provider
  (`_model_for_provider`), so an observed failover provider is never narrowed
  on a stale primary-model match. Removes the corresponding KNOWN_ISSUES entry.

### Documentation

- **RELEASING.md §7 now includes reformatting the published release note.**
  The release workflow publishes the raw CHANGELOG section; the post-release
  checklist now calls out editing it into the standing format (summary →
  `## ✨ Highlights` → `## 🔧 Core changes`).
- **README copyedit pass.** Fixed typos (`OpenRouter`, `optimizes`), a broken
  Markdown bold in the "What Tool Belt Adds" list, a run-on sentence, and
  stray whitespace — no content changes.

## [2026.9.4] - 2026-09-04

### Added

- **User-configurable shaper thresholds via `config.yaml`.** The five
  `learning.shape_ceiling` keys (`session_window`, `promote_min_sessions`,
  `promote_min_calls`, `demote_min_sessions_no_use`, `demote_k`) can now be
  tuned under `plugins.entries.tool-belt.settings.learning.shape_ceiling`
  without forking the plugin-owned `policy.yaml`. Resolution is per-key,
  highest layer first: `config.yaml` (user) → `policy.yaml` (shipped preset)
  → `shaping.DEFAULTS`. Every entrypoint threads the layer through
  `load_shape_ceiling_defaults` — the in-process auto-shape engine,
  `scripts/shape-ceiling.py` (CLI flags still override per-run), and
  `scripts/configure.py`. Invalid values (non-positive, non-numeric, wrong
  type) fall through to the layer below and never raise (fail-open). Enables
  a tighter rolling demote/promote cadence (smaller window, lower idle floor)
  on busy installs; per-scope overrides are not supported yet. See
  [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

### Documentation

- **Documented the `codex_app_server` runtime as not shaped.** That runtime
  hands the turn to a Codex subprocess and exposes Hermes tools through a
  fixed MCP tool list the plugin does not control, so those sessions are not
  narrowed and not recorded in telemetry — the same family of limitation as
  CLI, subagent, and cron sessions. Added to the README Limitations list,
  with the distinction that Codex *as a provider* (`api_mode:
  codex_responses`) runs through the normal tool loop and is shaped as usual.

### Fixed

- **`tool_calls.jsonl` now logs only primary model dispatches** (behind
  `require_tool_call_id`, default `true`). Hermes emits `post_tool_call`
  both for the model's assistant `tool_use` blocks and for nested/secondary
  dispatches (code-execution sandboxes, the MCP tools server, memory/
  mnemosyne batch fan-out) that never faced per-turn narrowing; logging the
  latter inflated `uses_in_window` in the economic demotion engine, so
  resolved-but-not-dispatched tools looked heavily used and were
  systematically over-carried (observed: the `memory` tool showed 136
  telemetry rows vs 5 real dispatches ever). Known edge: degraded models at
  long context can emit primary calls with blank ids; the default gate
  under-counts in that rare case (safe for the demotion engine — set
  `require_tool_call_id: false` to restore legacy logging). Historical
  inflated rows age out of the shaper window by rotation; no migration.

## [2026.9.3] - 2026-09-03

### Added

- **`configure --channel <platform>`** restricts a non-interactive `--mode`
  run to one (or several) of the agent's channels; a name matching nothing
  exits 2 and lists the channels found. `--platform` stays what it was — a
  hint for profiles with no telemetry, not a filter.
- **`plugin.yaml` declares `python_dependencies` and a `config_schema`**
  covering every top-level setting, so Hermes warns at gateway load when a
  setting has the wrong type or a declared dependency is missing, and
  `hermes plugins doctor` validates the schema's type names.
- **Releases are automated, and installing is documented.** Pushing a CalVer
  tag runs `.github/workflows/release.yml`, which re-runs the full gate
  (tests + `scripts/smoke-test.py` on Python 3.11 and 3.12), hard-fails if
  the tag does not match `plugin.yaml`'s `version`, and publishes a GitHub
  Release with notes taken from this changelog. The smoke test now also runs
  in CI on every push to `main` — the branch `hermes plugins install` clones
  — so `main` stays installable. The README documents installing from `main`
  and pinning a known-good commit with
  `hermes plugins install … --ref <commit-sha>`.

### Changed

- **Tool Belt is reframed as tool shaping for uncached APIs; `tool-belt
  savings` reports only the value that's actually there.** The report drops
  the caching/non-caching/historical jargon and reads as plain "savings from
  Tool Belt." The headline was already `net_forward` (uncached-only), but the
  per-session average and 12-month pace still divided by / spanned *all*
  sessions, diluting both with caching sessions that contribute nothing;
  `compute_observed` now also records `n_sessions_noncaching` /
  `first_ts_noncaching` / `last_ts_noncaching` from the same non-caching rows
  that price `net_forward`, and the CLI scopes the session count, average,
  pace span, and per-agent row to them. Agents whose sessions are
  caching-dominant are excluded from the total and named in a separate
  neutral line ("Agent(s) currently excluded from Tool Belt (no shaping):
  …") — their names italicized and comma-separated when the terminal
  supports color. `net_token_reduction` (the old blended-history figure)
  stays available in `--json` for diagnostics but no longer appears in the
  headline.
- **Expansion cost is now measured consistently everywhere a scope's
  shaping is recommended.** Production `auto_shape_run` already priced an
  `expand_tools` round-trip at the scope's measured per-event cost
  (`measure_expand_overhead`); the other two callers of
  `compute_scope_recommendations` — `scripts/shape-ceiling.py` (manual CLI)
  and `scripts/configure.py` (interactive review) — omitted
  `expand_round_trip_tokens` and silently fell back to the flat 1,500
  (26× too cheap on a long uncached session), so manual/interactive review
  could recommend demotions production itself rejects. The compute is now
  shared as `shaping.measured_expand_penalty()`, used by all three
  entrypoints; `shape-ceiling`'s porcelain output surfaces
  `expand_round_trip_tokens` per scope.
- **The cache-mode axis is now per provider, not per scope, and the caching
  posture is carry-all (behavior change; expect the savings headline to drop
  ~3×).** Whether narrowing helps or hurts is a property of the *provider*,
  per call — a caching provider re-bills the whole conversation history when
  the tool set changes mid-session, so on it Tool Belt saves cache-read
  pennies of schema and risks a large break. Measured across two live
  profiles, the old scope-keyed model over-sold by ~3.3× (a scope that fails
  over between a caching primary and a non-caching fallback was mis-locked in
  both directions), and on caching providers Tool Belt was net-negative
  because `expand_tools` mutations were 57–87% of all cache-miss cost. So:
  - **Caching providers → carry-all.** The whole tool ceiling ships every
    turn (minus `expand_tools`, which has nothing to recover when everything
    is already carried), the tool list is byte-stable for the session, and no
    predictor runs. This removes the per-session **frozen snapshot** and the
    "one cache break per `expand_tools` call" model entirely.
  - **Non-caching providers → unchanged.** Per-turn narrowing, sticky
    residency, lookback, and `expand_tools` — this is where the honest
    savings are, and it is untouched.
  - **Detection is keyed `scope|provider`.** Legacy `cache_mode_detection.json`
    keys migrate to `scope|<configured primary>` on load; a failover call to
    a non-caching provider no longer drags a caching primary's bucket. A
    mid-session provider change (e.g. `/model`) that flips the posture
    re-resolves it (the switch already busted the cache, so it's free).
    `cache_auto.providers_off_models` now also matches provider *names*.
  - **`api_calls.jsonl` rows gain `provider_caches`** (`true`/`false`/`null`);
    `frozen_reuse`/`frozen_reuse_count` remain but are vestigial (always
    `false`/`0`).
- **Savings accounting is cache-aware.** Gross schema savings are priced at
  the cache-read rate on caching providers (they live in the cached prefix)
  and full input rate otherwise; the flat `EXPAND_ROUND_TRIP_TOKENS = 1500`
  per-event overhead is replaced by a per-cohort *measured* cost (caching:
  the causal prefix-break re-bill; non-caching: the extra full-price call),
  with 1500 kept only as the thin-data fallback. The `tool-belt savings`
  headline is now a single honest input-token-equivalent net, tagged with
  its basis (`measured` / `fallback`). Under a caching posture the shaper is
  a no-op (nothing to demote when everything is carried).

- **Config moved to Hermes' plugin-settings subtree.** Tool Belt's settings
  now live at `plugins.entries.tool-belt.settings.*` — the path Hermes
  reserves for every plugin, which `ctx.get_config`, `config_schema`
  validation and `hermes plugins` all key off — instead of the ad-hoc
  `plugins.tool-belt.*` block. Per-channel entries nest as
  `channels.<agent>.<platform>` (Hermes forbids `:` in a settings key
  segment); the loader flattens them to the `agent:platform` scope string
  the rest of the plugin uses, and scalar keys at the agent level are
  defaults for every platform of that agent. A block left at the old path
  is not read; the plugin logs one warning at load. `configure` writes
  only the new path.

### Fixed

- **`configure` no longer drops out of the terminal UI at the confirm.**
  The "Apply these changes?" question is now the same radio list as the
  pickers (Apply / Back, with the disclosed diff repeated as its
  description; the diff also stays in the scrollback), and declining walks
  back to the mode picker instead of ending the run. The
  `Launcher already present` line after a confirmed apply is gone — a
  current launcher is silent; only the missing-PATH note still prints.
- **"N carried" now counts what the agent actually carries.** The status
  row and the shaping preview reported `len(learned.carry)` — only tools
  promoted *back* from expansion, which is empty right after a shape — so a
  freshly shaped scope read `0 carried, 53 by expansion`. "Carried" is now
  the real per-session set beyond pins: enabled ceiling − expand-only −
  (policy + config) pins. The preview's per-list diff keeps the learned
  `carry` line under its honest name, `Promoted back to carried`, beneath a
  `Carried each session (beyond pins): N → M` headline.
- **`configure` mode picker opens on the current mode.** The "Shaping mode"
  radio list always opened on "On — learning" regardless of the scope's
  `learned_mode`; it now opens on the mode in effect (or, when the chosen
  channels disagree, on row 0 with a `Currently: slack OFF, telegram ON`
  line). The numbered fallback marks the current row `(current)`.
- **`configure` channel checklist no longer pre-checks every channel.**
  ENTER with the cursor on one channel used to apply the mode to all of
  them. Rows now start unchecked and show each channel's current
  `shaping ON/OFF`; confirming with nothing checked walks back like ESC.
- **Color-gating test no longer fails on a bare clone / CI.** The test
  asserting that excluded-agent names italicize only when color is on
  imported `hermes_cli.colors` unconditionally, raising
  `ModuleNotFoundError` where the Hermes runtime isn't installed. Standalone,
  `savings_cli` falls back to its own no-op color function with no gating
  hook to flip, so italics never render there — the test now skips in that
  case; the comma-separation behavior is still locked everywhere by a
  separate test.

## [2026.8.31] - 2026-08-31

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
