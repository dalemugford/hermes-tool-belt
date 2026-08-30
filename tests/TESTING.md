# Test suite design

Rebuilt from first principles (2026-08-30) around one rule:

> **A test lives only if it can state, in one sentence: the promise it
> protects, the plausible regression it catches, and why no other test would
> catch it.** New tests must meet the same bar at review time — put the
> sentence in the docstring.

The suite grew organically as per-fix regression locks during release prep,
and all of them passed while the week's real incidents (a config-default flip
that left the plugin silently inert; demotable Tool Search bridge tools that
would have severed all deferred MCP access; `expand_tools` itself deferred
upstream) were found by audit and field reports instead. Those incidents
shared one shape: *the deployed system drifted while every test exercised
functions with explicit fixtures.* The rebuild answers that shape directly.

## Tiers

| Tier | Files | Protects |
|---|---|---|
| 0 — deployed defaults & shipped artifacts | `test_tier0_deployed_defaults.py` | The in-code defaults and policy.yaml ARE the promise: enabled/apply/auto defaults across their separate homes; the audited pin set exactly; thresholds; regex compile. Zero mocking. |
| 1 — promise wire contracts | `test_promise_wire.py` (crown jewels), plus the promise-graded survivors: `test_economic_demotion.py`, `test_auto_shape.py`, `test_inventory_reconciliation.py`, `test_trigger_overlay.py`, `test_full_start_contract.py`, `test_bridge_passthrough.py`, `test_expand_tools_real_bridge.py`, `test_harvest_privacy.py`, savings-honesty classes in `test_savings_cli.py` | Each shipped promise end-to-end through real wiring: real policy.yaml, real hooks (`_on_pre_gateway_dispatch` → wrapped `_build_api_kwargs`), real JSONL writers, Hermes faked only at import seams (`toolsets`, `tools.tool_search`, `hermes_cli.config`). |
| 2 — property invariants | `test_properties.py` | The algebra for ANY input, ~200 seeded-random cases per property: partition disjoint/covering, active ⊆ ceiling, full-start supremum, pin immunity under hostile learned states, fail-open at every wrapper seam, never-narrow-to-∅. |
| 3 — retained locks | everything else in `tests/` | Curated survivors of the original 462: bugs subtle enough that Tiers 0–2 would not re-catch them (session-pricing math, sticky decay, v1→v2 normalization edges, trigger false-positive dampeners, attribution inflation, learned-write durability, `/new` boundary). |
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
`tests/run_tests.py`. Bare `pytest tests/` does NOT work (the hyphenated
plugin dir breaks its import of the package `__init__`); use unittest.
The real-bridge tests require the Hermes venv interpreter — under a bare
python they self-skip and the file becomes a silent no-op, so CI must use
`~/.hermes/hermes-agent/venv/bin/python3`.

## Known hygiene debt

`test_cache_aware._seed_plugin_config` and
`test_bridge_passthrough._seed_config` clear-and-replace the module `_CONFIG`
without restoring it, poisoning later files in discovery order. Tier 0 and the
wire tests are immunized via `conftest.PRISTINE_CONFIG` (snapshotted before
any test runs); fixing the seeders to save/restore is welcome cleanup.
