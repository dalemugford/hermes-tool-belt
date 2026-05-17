# Harvest-Driven Follow-ups (Open)

Concrete policy/trigger edits surfaced by the harvest analyzer's first
run on real session data. Each item is backed by measured was_cut
signal — not speculation. Resolve by either applying the edit manually
or letting the (future) trigger-keyword suggester surface and apply it.

This file lives next to the harvest artifact ([telemetry-audit-2026-05-17.md](telemetry-audit-2026-05-17.md))
so the evidence and the proposed action stay co-located. When a
follow-up is resolved, move its block to a `resolved/` history section
with the commit SHA + before/after metrics.

---

## Open

### 1. Broaden the `browser` trigger group's keywords (CORRECTED)

**Correction:** the `browser` trigger group EXISTS in policy.yaml
(I was wrong in an earlier message). Its keywords are just too narrow
to catch how Dale actually asks Bernard for browser things — it fires
zero times across all of Bernard's harvested sessions despite 73
historical browser tool calls.

**Evidence (harvest, bernard:telegram):**

| Tool | Cuts |
|---|---:|
| browser_console | 18 |
| browser_navigate | 16 |
| browser_click | 14 |
| browser_snapshot | 14 |
| browser_vision | 7 |
| browser_type | 4 |

**Current keywords (policy.yaml `browser` group):**
- `\b(browse|navigate to|open .* in (a |the )?browser|browser automation)\b`
- `\b(fill (out|in)|click on|select|hover|scroll)\b.*\b(form|page|button|link|element)\b`
- `\bweb scrap`
- `\bautomate.*browser\b`

**Suggester output gives these candidates:**
- `"look up what time"` (cut_count=6, precision=1.0)
- `"can you check"` patterns (in browser_console previews)

The trigger-keyword suggester should be used to mine real keywords
from Dale's actual phrasing. Run:
`python3 analyze.py --state-dir state/dynamic-tools/harvest --suggest-trigger-keywords`

**Source:** [harvest analyzer 2026-05-17](../../state/dynamic-tools/harvest/predictions.jsonl)

---

### 2. Broaden `skills_authoring` trigger to cover `skill_manage`

**Evidence (harvest):**
- bernard:telegram — `skill_manage` cut 13 times
- sue:slack — `skill_manage` cut 8 times

**Why current trigger misses these:** the `skills_authoring` trigger
exists in policy.yaml but fires only ONCE across all bernard:telegram
sessions and never on sue:slack. Its keywords are too narrow.

**Proposed:** broaden the trigger's keyword set or add `skill_manage`
to a sibling group keyed on:
- "create a skill", "save a skill", "skill called", "new skill"
- "update the skill", "modify the skill"
- "promote the skill", "ship the skill"
- "skill for X"

**Source:** [harvest analyzer 2026-05-17](../../state/dynamic-tools/harvest/predictions.jsonl)

---

## Resolved

(empty — pending first follow-up resolution)
