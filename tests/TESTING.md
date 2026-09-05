# Test suite design

The suite is built around one rule:

> **A test lives only if it can state, in one sentence: the promise it
> protects, the plausible regression it catches, and why no other test would
> catch it.** New tests must meet the same bar at review time — put the
> sentence in the docstring.

Concretely, a test earns its place when a targeted mutation of the code
isolates it, when it is the sole or strongest guard of a named invariant, when
it locks a contract (CLI flags and exit codes, config schema symmetry, file
formats, porcelain, policy pins, `HERMES_HOME` containment, no-write
guarantees), or when it is an end-to-end wiring test through real components.
It does not when it only re-proves what a sibling proves through a bigger
fixture, when it asserts code-owned prose, or when its assertions cannot fail.
Between two siblings, the survivor is the one that goes through the real code
path rather than hand-built state.

The rule exists because a suite of per-fix regression locks can pass in full
while the deployed system drifts underneath it — every test exercising
functions with explicit fixtures, none exercising the wiring. The tiers below
answer that shape directly.

## Tiers

| Tier | Files | Protects |
|---|---|---|
| 0 — deployed defaults & shipped artifacts | `test_tier0_deployed_defaults.py`, `test_config_path.py` | The in-code defaults and policy.yaml ARE the promise: enabled/apply/auto defaults across their separate homes; the audited pin set exactly; thresholds; regex compile; the config path contract (`plugins.entries.tool-belt.settings`, nested channels ↔ scope keys, `config_schema` defaults == `_CONFIG`). Zero mocking. |
| 1 — promise wire contracts | `test_promise_wire.py` (crown jewels), plus the promise-graded survivors: `test_economic_demotion.py`, `test_auto_shape.py`, `test_inventory_reconciliation.py`, `test_trigger_overlay.py`, `test_full_start_contract.py`, `test_bridge_passthrough.py`, `test_expand_tools_real_bridge.py`, `test_harvest_privacy.py`, savings-honesty classes in `test_savings_cli.py` | Each shipped promise end-to-end through real wiring: real policy.yaml, real hooks (`_on_pre_gateway_dispatch` → wrapped `_build_api_kwargs`), real JSONL writers, Hermes faked only at import seams (`toolsets`, `tools.tool_search`, `hermes_cli.config`). |
| 2 — property invariants | `test_properties.py` | The algebra for ANY input, ~200 seeded-random cases per property: partition disjoint/covering, active ⊆ ceiling, full-start supremum, pin immunity under hostile learned states, fail-open at every wrapper seam, never-narrow-to-∅. |
| 3 — retained locks | everything else in `tests/` | Bugs subtle enough that Tiers 0–2 would not re-catch them (session-pricing math, sticky decay, trigger false-positive dampeners, attribution inflation, learned-write durability, `/new` boundary). |
| 4 — live-wire walkthrough | NOT pytest — see below | What the suite cannot honestly cover. |

## Meta-rules

1. **Real wiring has a precise definition.** Fake Hermes only at the import
   seams the code already treats as boundaries. Anything past that seam is
   Tier 4 by definition — the suite says so instead of simulating it.
2. **Fixtures assert their own preconditions.** A recovery test first proves
   the tool was actually absent; a shaped-path test proves something was
   shaped. A fixture that silently stops exercising its path must fail, not
   pass vacuously. (Origin: a savings smoke reported 5/5 while its expansion
   block was 7/8 because full-start left nothing demoted in the fixture.)
3. **No constant mirrors.** Asserting that a literal in a test equals the same
   literal in the code catches nothing; asserting that two *independent
   sources* agree (e.g. `FallbackBaselineDriftTests`) catches drift. Tier 0
   asserts shipped values only where they are the documented promise.

## Tier 4 — what pytest cannot cover (walkthrough probes, run at release)

1. `expand_tools` reaches the wire on a live gateway (fresh profile → one
   message → newest `predictions.jsonl` row contains it in the active set),
   and one end-to-end expansion recovery driven by the model itself.
2. The running gateway is actually narrowing (`tool-belt configure --status`,
   telemetry row count grows after a message).
3. Provider prefix-cache reality: hit rates in `api_calls.jsonl` after a
   3-turn session; cache-mode detection locking on real traffic.
4. Monkey-patch compatibility with the installed Hermes version
   (`_build_api_kwargs` signature, hook kwarg shapes, `register_cli_command`).
5. `hermes config set` pin-removal writing the real config.yaml cleanly.
6. Whether the model reaches for `expand_tools` when it needs a demoted tool
   (wording efficacy) — field telemetry only.
7. Gateway restart/launchd state survival.

## Running

`python3 -m unittest discover tests` from the plugin dir, or
`tests/run_tests.py`, or `python -m pytest -q -p no:cacheprovider` run from
inside `tests/` (system `python3` has no pytest installed; use a venv
interpreter that does). Bare `pytest tests/` from the plugin root does NOT
work (the hyphenated plugin dir breaks its import of the package `__init__`)
— run it from inside `tests/`, or use unittest instead.

The full suite, including every hermes-dependent case, runs with zero skips
under the Hermes interpreter:

```bash
~/.hermes/hermes-agent/venv/bin/python tests/run_tests.py
```

Under the plugin's own `.venv` — and in the shipped GitHub CI, which builds a
bare venv with no Hermes runtime — about a dozen cases self-skip: every case
gated on `hermes_cli` or `tools.tool_search` not being importable (the
real-bridge tests and `configure.py`'s curses *contract* tests, which pin the
signatures of an upstream module CI does not have), plus a `tiktoken`-gated
case when that package isn't installed. Any *other* skip line means something
changed and needs explaining.

## Known hygiene debt

`test_cache_aware._seed_plugin_config` and
`test_bridge_passthrough._seed_config` clear-and-replace the module `_CONFIG`
without restoring it, poisoning later files in discovery order. Tier 0 and the
wire tests are immunized via `conftest.PRISTINE_CONFIG` (snapshotted before
any test runs); fixing the seeders to save/restore is welcome cleanup.
