# Archive

Pre-pivot design docs. Kept for historical context (git archaeology, "why did
we drop the per-turn predictor?") but no longer load-bearing for the current
architecture.

The current architecture target is
[`docs/16.cache-aware-refactor-plan-2026-05-30.md`](../16.cache-aware-refactor-plan-2026-05-30.md).
The pivot decision is in
[`docs/15.cache-aware-pivot-2026-05-28.md`](../15.cache-aware-pivot-2026-05-28.md).

## What's here

- **`01.review-2026-05-07.md`** — early external code review. Findings were
  applied in subsequent commits; doc has no actionable content remaining.
- **`04.auto-apply-plan-2026-05-12.md`** — design for a per-turn `apply.py`
  writer. Replaced by the session-boundary shaper at
  [`scripts/shape-ceiling.py`](../../scripts/shape-ceiling.py) (Phase 2 of
  doc 16). The shaper consumes actual `expand_tools` events instead of
  regex-trigger inferences — same intent, stronger signal, cheaper moment.
- **`05.calibration-plan-2026-05-12.md`** — observe → handoff → steady-state
  flow for the per-turn predictor. Doesn't apply to session-boundary
  freezing. Cache-off mode keeps the original behavior so the calibration
  steps still work for kimi / gpt-5.4-mini profiles if needed.
- **`06.trigger-scoring-plan-2026-05-12.md`** — full-scoring proposal,
  marked deferred at the time. Per-turn-only by design. Dead under
  cache-on; rendered unnecessary by the shift in evidence axis (model-
  driven `expand_tools` instead of regex precision).
- **`10.harvest-followups-2026-05-17.md`** — concrete trigger-keyword
  broadenings identified by the harvest analyzer. Actionable under
  per-turn narrowing only. The shaper learns the equivalent from
  expand_tools events directly, no manual trigger edits required.

## What stayed visible in `docs/`

The legacy docs that are still load-bearing or useful as cost-categorization
references — 07 (dynamic-tools-plan), 08 (hermes-surface), 09 (telemetry-
audit), 11 (roadmap), 12 (expand-tools-evolution) — stayed in the active set
with disclaimers added at the top where the contents predate the pivot.
