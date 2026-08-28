# Tool Belt — Surface Audit Summary

**Date:** 2026-08-27
**Commit audited:** `13c953e` (Task 6, read-only)
**Sources:** Claude Code runtime-feature audit (`workspace/delegations/2026-08-27-084655-task-6-runtime-feature-audit/report.md`) and supported-surface audit (`workspace/delegations/2026-08-27-084655-task-6-supported-surface-audit/report.md`), reconciled against source by Bernard.

---

## Verdict

The repository is already lean. Both audits found no dead tracked files, orphaned
scripts, or defensible removal candidates. The test suite was green throughout
(119 passed, 1 optional-dependency skip; smoke 8/8 cache-off + 5/5 cache-on).
No repository changes were made during the audit.

---

## Core product features

The public product story: **safe, cache-aware tool loadouts with an explicit
recovery valve.**

1. **Adaptive tool narrowing**
   Filters each API call's tool list down from the user-configured Hermes
   ceiling. Unknown tools stay available, so Tool Belt never silently removes
   newly introduced tools.

2. **Lightweight intent prediction**
   A regex/keyword/dampener/attachment classifier chooses relevant tool groups.
   The shipped `policy.yaml` remains inspectable and deterministic.

3. **Cache-aware session loadouts**
   Tool sets freeze for prefix-caching models and remain stable through the
   session. Auto-detection reads actual cache telemetry and remembers whether
   each model/scope should use cache-on or cache-off behavior.

4. **`expand_tools` recovery**
   The model can restore a category or individual tool when prediction misses.
   Expansion persists through the frozen session (cache-on) or temporarily via
   sticky residency (cache-off).

5. **Safety guarantees**
   Tool Belt never widens the configured ceiling, keeps unfamiliar tools
   available, isolates state by canonical session key, and fails open to the
   original tool list when anything breaks.

---

## Secondary capabilities

Working and supported, but not the opening pitch:

- **Cache-off refinements:** short-term sticky tool residency and one-turn
  intent lookback.
- **Learned policy overlay:** the offline shaper can promote/demote tools per
  scope via `learned.json`; disabled by default.
- **Privacy-reduced telemetry:** append-only prediction, tool-call, and
  cache-economics records.
- **Analysis feedback loop:** telemetry feeds the analyzer and shaper; the
  runtime never rewrites learned policy by itself.
- **Deterministic bypass cohort:** unnarrowed baseline for measurement; doubles
  as a full kill switch at `bypass_rate: 1.0`.

---

## Operator surface

**Routine operator tools**

- `bootstrap.py` — optional warm start from existing sessions and telemetry
- `analyze.py` — comprehensive effectiveness and recommendation report
- `shape-ceiling.py` — writes reviewed per-scope learned adjustments
- `tool-belt savings` — canonical, read-only savings command (observed +
  projected cohorts); backed by `savings.py`
- `savings-report.py` — deprecated compatibility wrapper over `savings.py`

**Advanced operations**

- `harvest-replay.py` — privacy-reduced synthetic telemetry from session history
- `daily-analysis.sh` — scheduled multi-profile analyzer/shaper wrapper
- `rotate-telemetry.sh` — starts a clean measurement window
- `cache-freeze-replay.py` — cache-cost engine used internally by `analyze.py`,
  with an optional diagnostic CLI

**Maintainer tooling**

- `smoke-test.py` — eight cache-off + five cache-on invariants
- `check-tool-drift.py` — detects ceiling tools missing from policy
- Test suite, CI workflow, contributor material

---

## Cleanup decisions (Task 6 implementation list)

Recommendations agreed for discussion:

- Rename "Pass A predictor" to simply "predictor" — no Pass B exists.
- Simplify `learned_mode` from four names to two effective behaviors:
  recommendation generation becomes an analyzer operation; runtime config
  reduces to `off | auto`.
- Remove `memory` from the protected-always-on set in `analyze.py` — the shipped
  policy deliberately forces the legacy `memory` tool off, making its protected
  status contradictory.
- Keep `cache-freeze-replay.py` where it is; document the analyzer dependency
  rather than extracting shared math before release.
- Change "twice-daily" to "periodic" in `daily-analysis.sh` — cadence belongs to
  the operator.
- Accept and document the multi-chat reset limitation (last-writer-wins) until
  Hermes core passes the canonical session key to `on_session_reset`.

**Overall view:** Task 6 should clarify support tiers and remove stale
semantics. File deletion and broad extraction would add risk without improving
the product.