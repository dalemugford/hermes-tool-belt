# dynamic-tools: Calibration Mode — Plan

**Date:** 2026-05-12
**Status:** Design only; not implemented. Depends on the telemetry
correctness audit in [AUTO-APPLY-PLAN.md](AUTO-APPLY-PLAN.md) and
cross-profile validation criteria in [STRATEGY.md](STRATEGY.md).

---

## Goal

Make the strategy's universal promise — *install, activate, done; saves
tokens regardless of toolset* — intellectually honest.

The shipped policy can't deliver this on its own: it's shaped for
messaging-style scopes and actively misfits profiles we didn't tune
for. To work on a profile we've never seen, the plugin has to observe
what that profile actually does **before** it tries to shape policy.

This document is the design for that observation → adaptation → steady
state flow.

---

## Three phases

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Phase 1    │───▶│   Phase 2    │───▶│   Phase 3    │
│   OBSERVE    │    │   HANDOFF    │    │ STEADY STATE │
└──────────────┘    └──────────────┘    └──────────────┘
  wildcard +         analyzer            personalized
  telemetry          proposes policy     narrowing +
  (no narrowing)     user (or apply)     periodic
                     commits             re-analysis
```

Per scope, independently. A user with three active scopes can be in
phase 1 on one and phase 3 on another. Calibration is local to each
scope's data.

---

## Phase 1 — Observe

### Behavior

- `allowed_tool_names = WILDCARD_ALWAYS_ON` regardless of policy
- `policy_source = "calibrating"` stamped on every prediction row
- Full telemetry still written (predictions.jsonl, tool_calls.jsonl)
- `expand_tools` not registered (nothing's gated, no recourse needed)
- Sticky residency, learned overlay, dampeners — all dormant

### Entry conditions

- Plugin enabled for the first time on this scope, OR
- User explicitly enters calibration mode (`hermes dynamic-tools recalibrate <scope>`), OR
- Tool ceiling for the scope changes significantly (e.g. a new MCP
  server registers >5 tools we've never seen) — invalidates prior
  policy, restart calibration

### Exit conditions

Move to Phase 2 when **all** of the following hold for this scope:

| Signal | Floor | Rationale |
|---|---|---|
| Predictions logged | ≥100 | Sample size for stable trigger fire-rate estimates |
| Sessions observed | ≥10 | Trigger fire-rate distributions stabilize across distinct sessions, not within one |
| Distinct tools called | ≥3 | Need actual tool-use data, not just "the agent talked" |
| Days of data | ≥7 | Covers weekly cadence (cron jobs, weekend traffic) |

These are floors, not thresholds the analyzer optimizes for. They're a
"is this enough to make a call" gate. Specific numbers calibrate from
the telemetry audit; the shape is correct.

### Privacy / honesty

Calibration mode shows up in:
- The plugin's startup log line: `dynamic-tools: scope <s> in calibration (X/100 predictions, Y/10 sessions)`
- The README's "How it works" section
- A `hermes dynamic-tools status` CLI when that lands

No silent observation. The user sees that the plugin is learning.

---

## Phase 2 — Handoff

### Behavior

Calibration's exit conditions cleared. The analyzer produces a starter
policy by running its existing pipeline on the scope's observe-phase
telemetry:

1. Per-trigger precision: how often did each trigger fire *and* its
   tool actually get called?
2. Per-category expand patterns: which categories does this scope
   actually need? (Trick question — in observe mode nothing gets cut,
   so we don't have expand events. Use actual tool-call frequency
   instead: tools called in ≥X% of sessions are always-on candidates;
   tools called rarely are gate candidates.)
3. Dampener candidates: phrases that look like signal but don't lead
   to tool calls.

Output: a proposed learned.json patch + a "calibration report" markdown
summarizing what the plugin learned about this scope.

### Two handoff paths

**Path A — User review (default).**

The calibration report lands somewhere the user will see — a notice in
the dashboard, a Slack/Telegram DM, or a CLI prompt next time they
`hermes` command anything. Format:

```
dynamic-tools: calibration complete for bernard:telegram

Observed: 142 predictions, 18 sessions, 12 distinct tools, 9 days.

Proposed policy:
  +5 tools moved to always-on (used in >70% of sessions)
  +14 tools moved to trigger-gated (used in <10% of sessions)
  +3 dampener phrases learned

See: ~/.hermes/state/dynamic-tools/calibration-bernard-telegram.md
Apply: hermes dynamic-tools accept bernard:telegram
Reject: hermes dynamic-tools recalibrate bernard:telegram
```

User reviews the markdown report. `accept` writes the proposal into
learned.json. `recalibrate` returns to Phase 1.

**Path B — Auto-apply.**

If `learned_mode: auto` is set in config AND the proposed policy passes
the auto-apply thresholds from [AUTO-APPLY-PLAN.md](AUTO-APPLY-PLAN.md),
the handoff commits without human review. The calibration report still
gets written for audit; the user just doesn't have to click anything.

This is the path that delivers the literal "install, activate, done"
promise. Path A is the conservative default while we're still earning
trust.

### Exit

Successful handoff: scope transitions to Phase 3. The starter policy
is now in learned.json under the scope's keys, with full audit trail
(when, why, source metrics, calibration window).

Failed handoff: user rejects (or auto-apply gate denies). Scope returns
to Phase 1 with the existing telemetry intact — the next handoff
attempt uses observation + the rejected data combined.

---

## Phase 3 — Steady state

### Behavior

Active narrowing using the personalized policy. Same as today's "plugin
on" behavior:

- Preset YAML provides the base shape
- Per-scope learned overlay (now populated from calibration) refines it
- `expand_tools`, sticky residency, dampeners all active
- Telemetry continues to flow

### Drift detection

Periodically (default: weekly), re-run the analyzer on telemetry from
the last N days for each Phase-3 scope. Look for:

- **Promoted tools that stopped getting used.** Auto-demote candidate
  (per AUTO-APPLY-PLAN.md auto-demote rules).
- **Cut tools that the model keeps expanding.** Auto-promote candidate
  if expand round-trip cost > carry cost.
- **Dampeners over- or under-firing.** Surfaced as recommendations.
- **New tool categories** (new MCP server). If significant, may
  invalidate calibration and re-enter Phase 1.

### User controls

The user can always:
- View current policy: `hermes dynamic-tools show <scope>`
- Recalibrate: `hermes dynamic-tools recalibrate <scope>` (returns
  scope to Phase 1)
- Manually adjust: edit learned.json directly; manual entries get a
  `manual: true` flag so auto-apply won't override them

---

## Cold-start behavior

The instant a new user installs the plugin with `enabled: true`:

1. No prior telemetry for any scope.
2. First inbound message creates a scope and enters Phase 1.
3. Behavior is wildcard-with-telemetry until Phase 1 exit conditions
   hit — so the user pays full token cost during calibration.

This is honest but lossy. To avoid the "feels like nothing's happening
for a week" UX:

**Option: warm start from `policy.yaml`.** During Phase 1, also apply
the shipped policy's narrowing as a guess. Telemetry records what
*would* have been wildcard alongside what actually shipped. After Phase
2, the personalized policy supersedes.

This is the design that delivers immediate savings AND honest
calibration. Tradeoff: telemetry includes a mix of observed-narrowed
and observed-wildcard signal during Phase 1; the analyzer has to handle
both. Worth doing — the alternative is a long zero-value onboarding
window.

**Open: does warm-start change the calibration math?** If the warm
policy cuts a tool the user actually needed, the analyzer sees an
expand_tools event instead of a direct call, and may misclassify. Need
to think through whether to use warm-start telemetry for handoff
decisions or only observe-mode (forced-wildcard) telemetry.

---

## Connection to other plans

- **[AUTO-APPLY-PLAN.md](AUTO-APPLY-PLAN.md)** — the writer that commits
  the handoff under strict gates. Path B (auto-handoff) is literally
  `apply.py` running once Phase 1 exits. The two plans share the same
  thresholds, audit trail format, and idempotency contract.
- **[STRATEGY.md](STRATEGY.md)** — the cross-profile validation
  requirement is what determines when calibration mode itself
  graduates from lab to live. The first-week-new-user profile is the
  canonical test for this flow.
- **[PLAN.md](PLAN.md)** — the current shipped capabilities are the
  building blocks. Calibration mode is the user-facing flow that
  composes them.

---

## What this plan does NOT do

- **Doesn't replace the shipped policy.** `policy.yaml` becomes the
  warm-start seed and the fallback when no telemetry exists. It's not
  the destination.
- **Doesn't require auto-apply.** Path A (user review) works fine
  without auto-apply enabled. Auto-apply is just the way to make the
  handoff invisible.
- **Doesn't compete with manual override.** Users who hand-tune
  learned.json keep that ability; calibration just gives users who
  don't want to tune a way to get value without doing the work.
- **Doesn't observe forever.** Phase 1 has hard exit conditions. A
  scope can't get stuck in observation just because telemetry is
  ambiguous — the floors are about "enough data," not "perfect data."

---

## Open questions

- **Per-scope vs. cross-scope learning.** Does calibration for
  `bernard:telegram` benefit from looking at `bernard:slack` data?
  Probably yes for trigger precision (the model behaves similarly across
  messaging platforms), probably no for tool-frequency (each platform's
  conversation shape is its own thing). May need a "shared trigger
  learning, scope-specific tool policy" split.
- **Warm-start telemetry honesty.** See open question in §Cold-start.
  Do we use warm-start data for handoff or wait for clean
  observe-mode data?
- **Recalibration cadence.** Drift detection is weekly by default —
  but a heavy-development week might invalidate a policy in days.
  Should drift trigger automatic Phase-1 re-entry, or just surface a
  recommendation?
- **What if Phase 1 never exits?** Low-volume scope, never accumulates
  100 predictions in 7 days. Options: extend the window, lower the
  floors for low-volume scopes (with weaker confidence), or stay in
  Phase 1 indefinitely (honest but lossy). Lean: stay in Phase 1, but
  expose the state so the user can manually apply the shipped policy
  if they prefer that to wildcard.
- **Multi-agent profiles.** When does `sue:telegram` calibration
  benefit from `bernard:telegram` calibration data? Cross-pollination
  could accelerate cold start, but risks transferring Bernard-specific
  habits to Sue. Probably no shared learning by default; explicit
  opt-in if a user runs many agents with similar workloads.

---

## Implementation sketch (when we get there)

Rough sequencing once the prereqs in AUTO-APPLY-PLAN.md and
STRATEGY.md's cross-profile validation are satisfied:

1. **Phase markers.** Add a `phase: "observe" | "handoff_ready" | "steady" | "manual"`
   field per scope in learned.json. Plugin reads phase at policy
   resolution time and short-circuits to wildcard if observe.
2. **Phase 1 exit checker.** Background or on-demand check that walks
   recent telemetry and bumps scopes from observe → handoff_ready when
   floors are cleared.
3. **Calibration report writer.** Reuses analyzer output; produces a
   per-scope markdown file in `state-dir/calibrations/`.
4. **CLI hooks.** `hermes dynamic-tools status`, `accept`, `recalibrate`,
   `show`. Reads phase, runs handoff, transitions state.
5. **Auto-handoff (Path B).** Integration point with `apply.py` from
   AUTO-APPLY-PLAN.md. Runs the handoff under auto-apply gates when
   `learned_mode: auto` is set.
6. **Drift detection.** Scheduled re-analysis with recommendations
   piped back through the recommendation system.
7. **Warm-start.** Shipped `policy.yaml` narrowing during Phase 1,
   with telemetry honest about whether each row was observed-narrowed
   or forced-wildcard.

Each step is independently committable. The first three are enough to
deliver Path A; the rest add the "no human in the loop" path.

---

*Parked here until the telemetry-correctness prereqs clear and at least
one cross-profile validation point is available. Until then, treat this
as the destination design, not the build target.*
