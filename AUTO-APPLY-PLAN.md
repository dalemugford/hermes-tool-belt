# dynamic-tools: Auto-Apply Layer — Plan

**Date:** 2026-05-12
**Status:** design only; not implemented. Telemetry-correctness validation
is a prerequisite before any auto-apply code runs against live state.

---

## Goal

Close the loop the rest of the plugin opens. The pipeline today is:

```
telemetry → analyzer → recommendations (human pastes) → tighter policy
```

Auto-apply replaces the middle step:

```
telemetry → analyzer → apply.py (gated writes) → learned.json → tighter policy
```

This lets the plugin **shape tool availability based on agent/client
habits** without a human in the loop for every iteration, while keeping
all the safety boundaries the manual flow already enforces.

---

## Prerequisite: telemetry correctness audit

**Before any apply code touches learned.json**, we have to be confident
the analyzer's view of the world matches reality. Open questions:

1. **`expand_tools_used` accounting.** Does the post-tool-call hook
   correctly distinguish "the model used a tool from the expanded set" vs.
   "the model called an unrelated tool after expand_tools fired"? Sticky
   residency complicates this — a tool may show as used because of sticky,
   not because of the current expansion event.
2. **Trigger false-positive classification.** A row is FP if the trigger
   fired and no matching tool was called *within the same prediction*. But
   the model often calls the tool 2–3 turns later (sticky carries the
   expansion). Are we miscounting late-bound uses as FPs?
3. **`session_id` coverage.** Several existing telemetry rows show
   `session_id: ""` — confirm this is fixed in current code and that
   future rows always carry a session id (necessary for cohort tracking
   and revert logic).
4. **Cohort populations.** Once `bypass_rate: 0.05` is live, confirm the
   bypass cohort actually accumulates rows on `bernard:telegram` and
   that `policy_source: bypass` shows up correctly. If bypass turns out
   net-negative or noisy, recalibrate before trusting any apply gate
   that depends on cohort comparison.
5. **`tools_in_category` accuracy.** The expand-cost math uses
   `expand_event.resolved_tools` for tools-per-category. Confirm this
   matches the actual resolved tool list at the time of expansion across
   different categories (some categories like `browser` have 12 tools;
   small misses change net-value math meaningfully).
6. **Dampener candidate sample sizes.** Auto-apply for dampeners requires
   a high precision floor (≥0.95) and meaningful support (≥5). Validate
   on at least two weeks of live telemetry that high-precision candidates
   are *consistently* high-precision across non-overlapping windows — not
   just a one-week artifact.

**Gate:** until each of these is answered, treat the apply layer as a
design exercise, not a build target.

---

## Architecture

### Separation of concerns

| File | Role | Side effects |
|---|---|---|
| `analyze.py` | Read telemetry, compute stats and recommendations | None (read-only) |
| `apply.py` (new) | Read recommendations, gate by strict thresholds, write learned.json | Writes learned.json + change log |
| `learned.py` | Read learned.json at prediction time, merge into preset | Read-only at prediction time |

`analyze.py` stays observable. `apply.py` is the only writer besides
explicit user action. `learned_mode` (whether the plugin consumes
learned.json at prediction time) remains a user-controlled config knob —
`apply.py` never changes it.

### Why a separate script vs. an `--apply` flag on `analyze.py`

- Cleaner audit: "what did the analyzer say" vs. "what did we apply" stay
  distinct files and distinct invocations.
- Easier to schedule: `apply.py` can run on a different cadence than
  analysis (e.g. analyze daily, apply weekly).
- Easier to gate: a future hook or permission system can require explicit
  approval for `apply.py` invocation without restricting `analyze.py`.
- Reduces the risk of an accidental `--apply` flag flipping live policy
  during a routine analysis run.

---

## Two change types

### 1. Promote/demote (use existing `learned.json` schema)

The current schema already supports this:

```json
{
  "scopes": {
    "bernard:telegram": {
      "always_on": ["terminal"],
      "always_off": ["browser_cdp"],
      "trigger_adjustments": {"shell": {"action": "demote"}},
      "promotions": {
        "terminal": {
          "promoted_at": "2026-05-12T...",
          "triggered_by": "apply",
          "source_metrics": {
            "expand_rate": 0.62,
            "success_rate": 0.91,
            "net_value_tokens": 4200,
            "predictions": 124
          },
          "thresholds_used": {
            "promote_expand_rate": 0.50,
            "promote_use_rate": 0.80,
            "expand_round_trip_tokens": 1500
          }
        }
      }
    }
  }
}
```

`learned.py.apply_to_preset` already handles `always_on` / `always_off` /
`trigger_adjustments`. No schema change needed for this axis.

### 2. Learned dampeners (new field)

New per-scope field:

```json
{
  "scopes": {
    "bernard:telegram": {
      "learned_excludes": {
        "file_write": [
          {
            "pattern": "\\bdown the road\\b",
            "added_at": "2026-05-12T...",
            "source_metrics": {
              "fp_count": 14,
              "tp_count": 0,
              "precision": 1.0,
              "sample_previews": ["...", "...", "..."]
            },
            "thresholds_used": {
              "apply_min_precision": 0.95,
              "apply_min_support": 5
            }
          }
        ]
      }
    }
  }
}
```

`learned.py.apply_to_preset` extends each `TriggerGroup.exclude_patterns`
with the learned regexes during merge. Pattern is the same as the bug fix
that landed in the earlier review — copy `exclude_patterns` when rebuilding
TriggerGroup, plus add the learned ones.

**Why a separate field instead of editing the YAML:**
- Preset YAML stays human-curated; users can read it without seeing
  machine-generated noise.
- Audit trail lives next to the change.
- Reverting an auto-applied dampener is just `del scope.learned_excludes.<trigger>.<pattern>`,
  no risk of corrupting the YAML.
- Reload mechanics already work for learned.json (mtime cache); no need
  to invalidate preset cache on every change.

---

## Auto-apply thresholds (stricter than human-review)

Auto-apply must be **more conservative** than the analyzer's review
defaults. A human reviewing recommendations can use judgment on borderline
cases. The applier has no judgment — so the threshold for "definitely
right" should be visibly above "probably worth looking at."

| Axis | Review default (analyze.py) | Auto-apply default (apply.py) |
|---|---|---|
| Promote: confidence floor | n/a | medium (block `low`) |
| Promote: status floor | `review` allowed | block `insufficient_data` |
| Promote: net-value | flagged | **required positive** |
| Promote: min predictions | n/a | 50 (don't promote from a thin sample) |
| Demote: unused carry turns | 10 | 50 (don't demote on noise) |
| Dampener: precision | 0.80 | **0.95** |
| Dampener: support (FP count) | 3 | **5** |
| Dampener: max per run | 10 surfaced | **3 applied** |
| Max changes per run | n/a | **5 total** |
| Idle window | n/a | last 7 days of telemetry only |

The "max changes per run" rate limit matters: one bad apply that breaks a
scope shouldn't be hidden inside 20 other applies in the same audit log.

---

## CLI design (apply.py)

```bash
# Default: dry-run. Print proposed diff, write nothing.
python3 apply.py

# Actually write. Refuses to run if state-dir telemetry is empty or
# obviously stale.
python3 apply.py --apply

# Tune thresholds (each defaults to the auto-apply column above).
python3 apply.py --apply-min-precision 0.95 \
                 --apply-min-support 5 \
                 --apply-min-predictions 50 \
                 --apply-max-changes 5

# Scope to a single agent or platform.
python3 apply.py --apply --scope-filter "bernard:telegram:*"

# Window — default last 7 days, opt in to wider history.
python3 apply.py --apply --window-days 14
```

**Exit codes:**
- `0` — ran cleanly (dry-run or apply)
- `1` — apply blocked by safety guard (e.g. no telemetry, telemetry too
  thin, threshold mismatch with existing learned.json schema)
- `2` — unexpected error

**Always emits** a change log at `state-dir/apply-changes-<ts>.jsonl`
(append-only) regardless of dry-run/apply, so we have a record of what
*would have been* applied even on dry runs.

---

## Audit trail

Every change writes two records:

1. **In learned.json** under the scope's `promotions` map. Includes the
   applied state, the source recommendation metrics, and the exact
   thresholds in effect at apply time.
2. **In `apply-changes-<ts>.jsonl`** under state-dir. One row per change:
   ```json
   {
     "ts": "2026-05-12T...",
     "change_type": "promote_always_on" | "demote_always_off" | "add_learned_exclude" | "remove_learned_exclude" | "demote_unused",
     "scope": "bernard:telegram",
     "item": "terminal",
     "applied": true,
     "dry_run": false,
     "source_metrics": {...},
     "thresholds_used": {...},
     "previous_state": {...},
     "new_state": {...}
   }
   ```

Replays for accountability: "show me every change applied to
bernard:telegram in the last 30 days" is a single `jq` over the change
log files.

---

## Idempotency

A second `apply.py --apply` run with no new telemetry must produce zero
changes. Implementation:

- Promote: skip if item is already in `learned.always_on` (or already
  in the preset's static always_on for this scope).
- Demote: skip if item is already in `learned.always_off`.
- Dampener: skip if the regex pattern already appears in
  `learned_excludes[trigger]`, or matches an existing
  `exclude_keywords` entry in the preset (use the same overlap check
  the analyzer's `--suggest-dampeners` flow already implements).

Idempotency is a hard requirement, not a nice-to-have — without it,
running apply on a cron would re-promote the same item every interval.

---

## Auto-demote (the "is it actually working" check)

Auto-promote without auto-demote is a one-way ratchet — once a tool ends
up always-on, it stays there even if usage drops. The applier needs a
revert path:

For each entry in `learned.always_on` (or `learned_excludes`):

1. Look at telemetry from after `promoted_at`.
2. For always-on tools: if the tool was carried for ≥50 turns with zero
   calls, queue a `demote_unused` change.
3. For learned dampeners: if the dampener pattern matched ≥5 messages
   AND any of those matched messages also produced a real tool call from
   the trigger's group, the dampener is over-aggressive — queue removal.
4. Apply demotions under the same strict gates and rate limits.

This is the "stop-the-bleed" mechanism. It also serves as the auto-apply
layer's self-test: if the analyzer's recommendations are bad, demotion
fires and cleans up.

---

## What auto-apply does NOT do

Things deliberately out of scope:

- **Editing the preset YAML.** Hand-curated content stays hand-curated.
- **Flipping `learned_mode` in config.** That's an explicit human opt-in.
  apply.py can write learned.json all day; if the user's config has
  `learned_mode: off`, the plugin still ignores it. This is the master
  off-switch.
- **Modifying static presets.** Even for the rare case where a hand-edit
  to aggressive.yaml would be cleaner, that's a human call.
- **Touching sticky residency.** Sticky is intentionally session-grain;
  promoting always-on is its durable cousin.
- **Auto-flipping `bypass_rate`.** The cohort sampler is a measurement
  tool; changing it is a measurement-strategy decision.

---

## Implementation order (when we get there)

Rough sequencing once the telemetry audit clears:

1. **Schema extension**: add `learned_excludes` to learned.json. Update
   `learned.py.apply_to_preset` to merge it into `TriggerGroup.exclude_patterns`.
   Regression-guard with a smoke check (mirror the existing
   `_check_learned_merge_preserves_dampeners` pattern).
2. **apply.py skeleton**: CLI, telemetry loading, dry-run output. No
   writes yet. Validate the proposed-diff output looks right against
   live telemetry.
3. **Promotion path**: read recommendations, filter under strict
   thresholds, propose promotes/demotes. Still dry-run only.
4. **Dampener path**: read dampener candidates, filter under strict
   thresholds, propose dampener adds.
5. **Audit trail + idempotency**: implement the change log and the
   skip-if-already-present checks. Dry-run still.
6. **Wire `--apply`**: actually write learned.json + change log. Smoke
   test idempotency by running twice in a row.
7. **Demote path**: read post-promotion telemetry, propose
   demote/remove-learned-exclude. Dry-run first, then wire.
8. **Cohort gate** (optional, depends on telemetry audit outcome): only
   apply if the bypass cohort comparison shows the recommendation
   doesn't degrade behavior on the narrowed cohort vs the bypass cohort.

Each step is independently committable. Stop at any boundary if telemetry
or behavior surfaces unexpected.

---

## Open questions

- **Cron vs. manual invocation.** Should `apply.py` run on a schedule
  (e.g. weekly), or only when Dale invokes it? Lean: manual until the
  telemetry validation has produced confidence; then maybe weekly with
  a notification.
- **Per-scope opt-in.** Should `apply.py` only act on scopes that have
  `learned_mode: auto`/`audit` set in the user's config? Lean: yes —
  if the user hasn't opted in to applying learned state at prediction
  time, the applier shouldn't write to a learned.json that won't be
  consumed anyway. (Caveat: dry-run should ignore this check so users
  can preview without opting in.)
- **Threshold calibration.** The defaults in this plan are reasonable
  guesses. After the telemetry audit, derive empirical thresholds from
  the live data — e.g. "in the last 60 days, the lowest-precision
  dampener that survived human review was 0.93, so set the apply floor
  to 0.95 with margin."
- **What about the unknown-kept-tools class?** When a new MCP server
  registers tools the plugin doesn't know about, those pass through. If
  auto-apply ever observes a never-used always-on entry from this class,
  is it safe to demote? Lean: no, treat unknown tools as protected like
  `expand_tools` / `memory` / `send_message` already are.

---

## Success criteria (for when the layer eventually ships)

1. A 30-day live run with `--apply` enabled produces a learned.json that,
   when reviewed manually, contains no entries a human would have
   rejected.
2. No auto-applied change causes a measurable degradation in turn count
   or completion success in the affected scope's bypass cohort.
3. Auto-demote correctly reverts at least one promotion within 60 days,
   demonstrating the self-cleaning loop works.
4. The change log under `apply-changes-*.jsonl` is sufficient to answer
   "why does this scope have this policy" without consulting the
   recommendations history.
5. Disabling `learned_mode` in config produces immediate rollback to
   static-preset behavior — no residual state from the applier survives.

---

*Plan parked here; resume when telemetry-correctness audit clears the
prerequisite list at the top.*
