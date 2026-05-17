# dynamic-tools: Strategy

**Last updated:** 2026-05-12
**Status:** North-star strategy doc. Short on purpose. Detail lives in the
sub-plans linked at the bottom; this document is the place to align on
what the plugin is trying to *be*.

---

## The promise

> **Install, activate, done — saves tokens on tool overhead, regardless
> of your agent's toolset.**

That's the user-facing pitch. It's a strong claim because it has to work
for users we've never met, with toolsets we've never seen, on profiles
we didn't tune.

**Current honest state:** the plugin delivers "install, **tune**, done"
today. One hand-curated policy (`policy.yaml`) targets messaging-shaped
scopes, and a knowledgeable user can layer per-scope overrides via
`always_on_extra` / `always_off` / `bypass_rate`. The universal promise
— same plugin producing the right policy for *any* profile without
human tuning — is a research target, not a shipped feature.

The rest of this doc is the path from where we are to where the promise
is real.

---

## Lab vs. Live

To make the promise honest without freezing all current experimentation,
split the plugin into two surfaces:

### Live (`main` branch / shipped plugin)

What we'd be comfortable telling other Hermes users to install today.
Stable, narrow scope, predictable behavior.

Includes:
- The single shipped policy (`policy.yaml`) and the per-scope override surface
- `predictor.py` (regex + `exclude_keywords` dampeners)
- `_build_api_kwargs` patch — the actual narrowing mechanism
- `expand_tools` meta-tool — the safety valve
- Sticky residency (in-memory, per-session-key, TTL'd) — corrected from
  the original per-scope keying after the 2026-05-17 audit found
  cross-conversation leakage on the old design
- Predictor lookback (1-turn default)
- `triggers_suppressed` telemetry — minimal, useful for tuning
- Basic `predictions.jsonl` / `tool_calls.jsonl` for jq spot-checks
- Smoke-test harness (`scripts/smoke-test.py`) — mechanical validation
  for the runtime surface above

Approximately 600 lines. Everything in this set has been in production
under Bernard:telegram and has measurable savings.

### Lab (`labs` branch / not yet shipped)

Where the adaptive ideas live until they prove themselves across profiles.

Includes:
- `learned.json` overlay (`learned.py`, `learned_mode`)
- Analyzer's recommendation engine (`recommendation_rows`,
  `harvest_recommendation_rows`, `dampener_candidates`,
  `trigger_keyword_candidates`, cohort comparison, net-value math)
- A/B bypass cohort sampler
- Harvest infrastructure — `scripts/harvest-replay.py` and the
  harvest-aware analyzer modes (new since 2026-05-17)
- First-install bootstrap UX (`scripts/bootstrap.py`) — currently
  manually invoked; auto-bootstrap on `register()` is future work
- The auto-apply layer described in [AUTO-APPLY-PLAN.md](AUTO-APPLY-PLAN.md) — still unimplemented but with mechanical prereqs now cleared
- The calibration-mode flow described in [CALIBRATION-PLAN.md](CALIBRATION-PLAN.md)

### Promotion criteria (lab → live)

A feature graduates to live when **all** of the following hold:

1. It works on at least one non-Bernard profile (see Cross-Profile
   Validation below) without per-scope tuning.
2. Its failure modes are bounded — worst case is "no narrowing happens,"
   never "the agent breaks."
3. Its config surface is small enough to document in one paragraph.
4. There's a way to disable it that doesn't require the user to
   understand the feature.

Anything that fails a criterion stays in lab.

---

## Calibration flow (the path to the universal promise)

The promise "install, activate, done" can't mean "we guess your policy
on day one." A profile we've never seen needs an observation period
before we can shape its tool availability. So the honest version of the
promise is:

> Install. Plugin runs in observation mode and learns your usage.
> After N sessions, it proposes a profile-specific policy. You accept
> (or it auto-applies, under strict gates). From then on, you save
> tokens on tool overhead.

Three phases:

1. **Observe** — wildcard-with-telemetry. No narrowing yet. Honest in
   the logs about what it's doing.
2. **Handoff** — analyzer produces a starter policy from this profile's
   own data. User reviews + accepts, or auto-apply commits it under the
   gates in [AUTO-APPLY-PLAN.md](AUTO-APPLY-PLAN.md).
3. **Steady state** — active narrowing with the personalized policy.
   Telemetry keeps flowing; periodic re-analysis catches drift and
   proposes adjustments.

The shipped policy becomes the *seed* for the observe phase (or the
fallback when calibration hasn't completed), not the destination. Full
design in [CALIBRATION-PLAN.md](CALIBRATION-PLAN.md).

**Update 2026-05-17:** `scripts/bootstrap.py` is the first concrete
implementation of the handoff phase. It runs harvest-replay against
existing `~/.hermes/sessions/*.jsonl` to produce synthetic telemetry,
then runs the analyzer in harvest-aware mode to surface ranked TOP
ACTIONS — concrete promotion candidates and trigger-keyword
suggestions. Day-one recommendations exist; the user reviews and
applies them by hand. Auto-bootstrap on plugin first activation (the
"register() detects empty live telemetry → runs harvest →
surfaces recommendations" magic UX) is the next step toward making
the handoff fully automatic.

---

## Cross-profile validation

Before the lab → live promotion gate accepts an adaptive feature, the
feature has to survive contact with profiles we didn't tune for. The
validation matrix:

| Profile | Shape | What it stresses |
|---|---|---|
| `bernard:telegram` | Messaging, conversational, high volume | Lookback, dampeners, expand round-trip cost |
| `sue:*` (cron + DM mix) | Mixed scheduled + ad-hoc, lower volume | Mixed-traffic scope behavior, low-data confidence |
| CLI-heavy power user | Long sessions, broad tool usage | Sticky, always-on stability |
| Custom MCP-heavy profile | Unknown plugin tools dominate | `unknown_kept_tools` handling, safety net |
| First-week new user | Cold start, no telemetry history | Calibration phase + handoff UX |

A feature passes validation when it produces sensible behavior on at
least three of these without per-profile tuning. "Sensible" is defined
per-feature in the sub-plan that owns it.

The first-week new-user profile is special: it's the test of whether the
calibration flow itself works. Nothing graduates without passing it.

---

## Success bar — when the promise is real

Concrete, measurable criteria the plugin must meet to credibly make the
universal promise:

1. **30-day live run** on three non-Bernard profiles produces a
   personalized learned.json each, with no entries a human would
   reject on review.
2. **No auto-applied change** has caused a measurable degradation in
   turn count or task-completion success in any scope's bypass cohort.
3. **First-week new-user** profile completes calibration → handoff →
   steady-state without manual intervention. End state: a policy that
   beats the shipped `policy.yaml` for that profile on net tokens.
4. **Disabling the plugin** (set `enabled: false`) returns the agent
   to baseline behavior with no residual state — no leaked learned
   policy, no stale telemetry that breaks something else.
5. **Documentation passes the "smart colleague" test**: a Hermes user
   who's never seen this plugin can `hermes plugins install` it, read
   the README, and have it working in five minutes without asking
   questions.

---

## Official Hermes-plugin readiness

For dynamic-tools to ship as an official plugin (or be considered for
core inclusion), it needs to conform to Hermes' plugin authoring
conventions. Tracking criteria here so they don't drift:

### Manifest (plugin.yaml)

Already in shape — `name`, `version`, `description`, `provides_tools`,
`provides_hooks`. Verify on every release that the manifest accurately
reflects what `register()` actually does.

### Entry point (`__init__.py:register(ctx)`)

- Tools registered via `ctx.register_tool(name=, toolset=, schema=, handler=, description=)`. ✅
- Hooks registered via `ctx.register_hook(event, callback)`. ✅
- All hook callbacks accept `**kwargs` for forward compatibility. ✅
  (Verified: `_on_pre_gateway_dispatch`, `_on_post_tool_call`,
  `_on_session_end` all have `**kwargs`.)

### Distribution path

When ready for public install, the plugin needs to be installable via:

```bash
hermes plugins install <user>/<repo>            # prompts to enable
hermes plugins install <user>/<repo> --enable   # install + enable
```

Implies:
- A standalone Git repo (the plugin can't live in `~/.hermes/plugins/`
  forever; it needs its own home).
- Repo layout matches the [hermes-example-plugins](https://github.com/NousResearch/hermes-example-plugins)
  shape: one plugin per repo, `plugin.yaml` + `__init__.py` at the root,
  sub-modules as needed.
- A `README.md` at the repo root that opens with "install, activate,
  done."

### Discovery hierarchy

Hermes searches in this order, later wins on name collision:
bundled → `~/.hermes/plugins/` → `.hermes/plugins/` → pip → NixOS.

Implication: while we develop in `~/.hermes/plugins/dynamic-tools/`, a
later upstream version would override it. That's fine for testing but
needs explicit handling at release time — pick one home for the plugin
and stop maintaining the other.

### Config-key conventions

The plugin's config lives under `plugins.dynamic-tools.*` in
`config.yaml`. All knobs should:
- Have a documented default that produces useful behavior with no
  user tuning (the "install, activate, done" promise).
- Fail safe: invalid values fall back to the default with a logged
  warning, never break the gateway.
- Be discoverable via the README's Configuration Reference section,
  with one paragraph per knob explaining when to change it.

### CLI surface (future)

Per the original PLAN.md Phase 4 sketch, an eventual
`hermes dynamic-tools <subcommand>` CLI for inspection and manual
overrides. Not required for first public release; nice to have once the
adaptive layer is durable.

---

## Sub-plans

The detail lives in the linked plans. This document is the index.

| Plan | What it covers | Status |
|---|---|---|
| [PLAN.md](PLAN.md) | Original adaptive layer plan; shipped capabilities through 2026-05-12 | Current |
| [AUTO-APPLY-PLAN.md](AUTO-APPLY-PLAN.md) | The writer-side script that closes the loop; telemetry-correctness prereqs | Design, not implemented |
| [CALIBRATION-PLAN.md](CALIBRATION-PLAN.md) | The observe → handoff → steady-state flow that makes the universal promise honest | Design, not implemented |
| [TRIGGER-SCORING-PLAN.md](TRIGGER-SCORING-PLAN.md) | Historical full-scoring framework; deferred in favor of `exclude_keywords` dampeners | Historical |

---

*Update this document when the strategic shape changes — promotion of a
lab feature to live, validation passing on a new profile, or a change
in the official-plugin readiness criteria. Operational detail belongs
in the sub-plans, not here.*
