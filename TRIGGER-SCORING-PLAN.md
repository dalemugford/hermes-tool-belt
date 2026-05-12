# Trigger Dampeners and Near-Miss Telemetry Plan

> **For Hermes:** Use the `subagent-driven-development` skill or Bernard's Claude Code artifact wrapper to implement this plan task-by-task. For each task, keep the patch small, verify behavior with focused tests/fixtures, and avoid changing runtime defaults unless Dale explicitly approves.

**Goal:** Improve dynamic-tools trigger precision without building the full scored-trigger framework yet.

**Architecture:** Keep the existing binary regex trigger model. Add a small negative-veto layer (`exclude_keywords`) and light trigger-shape refinements for the worst false-positive groups. Add near-miss telemetry later so future tuning is based on observed recovery events rather than guesses.

**Tech Stack:** Python stdlib, YAML presets, existing dynamic-tools modules (`presets.py`, `predictor.py`, `logger_io.py`, `analyze.py`), JSONL telemetry.

---

## Decision Update

The previous version of this plan proposed a full scored trigger framework: score signals, thresholds, per-signal telemetry, and analyzer lab reports. That design is still intellectually sound, but it is too heavy for the current problem.

Current telemetry already shows the plugin is working:

- `bernard:telegram` predictions: 272
- tool-call rows: 1,417+
- `expand_tools` events: 65
- expansions with downstream use: 45 / 65 (69.2%)
- logged first-call tokens saved: ~1.79M

The weak spot is not the whole architecture. It is a small set of trigger false positives:

- `file_write`: useful but imprecise; about 47% same-prediction precision in the latest analyzer report.
- `delegation`: low volume, 0 same-prediction hits in current telemetry.
- `cronjob`: low volume, 0 same-prediction hits in current telemetry.

The next round should reduce those false positives with the smallest understandable mechanism.

---

## Current Context

Plugin folder:

```text
/Users/macmini/.hermes/plugins/dynamic-tools
```

Important files:

- `presets.py` — defines `TriggerGroup`, YAML parsing, and trigger matching.
- `predictor.py` — evaluates trigger groups and returns `Prediction`.
- `logger_io.py` — writes prediction/tool-call telemetry.
- `analyze.py` — reads telemetry and writes reports/recommendations.
- `presets/aggressive.yaml` — primary Telegram/messaging preset.
- `PLAN.md` — main dynamic-tools adaptive-layer plan.
- `reports/2026-05-11-030456-analysis.md` — latest useful live analyzer report.

Claude Code review artifact behind this revision:

```text
/Users/macmini/Agents/Bernard/delegations/2026-05-10-232212-dynamic-tools-trigger-improvement-review/report.md
```

---

## Revised Strategy

Use three small improvements before considering scoring:

1. **Negative vetoes / dampeners** — suppress a trigger when the message is clearly discussion, planning, or meta-talk.
2. **Tighter trigger shapes** — prefer action-plus-object or explicit vocabulary over broad topical keywords.
3. **Near-miss telemetry** — log when a gated category was later expanded so future refinements can target real misses.

Do not promote `terminal` to always-on in this pass. Terminal is heavily used, but current analyzer output still recommends keeping it gated. Revisit only after a more precise cost calculation that accounts for conversational Telegram turns.

Do not implement the full score/threshold framework in this pass.

---

## Behavior Model

Current model:

```text
trigger fires if ANY positive keyword matches
```

New model:

```text
if ANY exclude keyword matches:
    trigger does not fire
else if ANY positive keyword matches:
    trigger fires
else:
    trigger does not fire
```

This is deliberately simpler than scoring. It is easy to explain, easy to audit in YAML, and easy to revert.

---

## Task 1: Add `exclude_keywords` to trigger groups

**Objective:** Let a trigger define negative regex patterns that veto the trigger before positive keyword matching.

**Files:**

- Modify: `presets.py`
- Test/verify: existing parser/matcher tests if present; otherwise add focused lightweight tests or run a direct Python fixture.

**Step 1: Extend `TriggerGroup`**

Add a field beside `keyword_patterns`:

```python
exclude_patterns: list[re.Pattern[str]] = field(default_factory=list)
```

**Step 2: Veto before positive matching**

At the top of `TriggerGroup.matches()` after any normalization but before attachment/keyword positives:

```python
for pat in self.exclude_patterns:
    if pat.search(message):
        return False
```

Keep attachment handling after the veto. If a trigger has an attachment and a discussion dampener, the dampener should still suppress only when the text explicitly indicates non-action/meta discussion. For image attachments, do not add broad excludes.

**Step 3: Parse YAML**

Where `_parse_triggers()` builds `TriggerGroup`, parse the new optional key with the same helper used for `keywords`:

```python
exclude_patterns=_compile_keywords(entry.get("exclude_keywords")),
```

**Step 4: Verify backward compatibility**

Run:

```bash
cd /Users/macmini/.hermes/plugins/dynamic-tools
python3 -m py_compile presets.py
```

Expected: no output, exit 0.

Also verify that presets without `exclude_keywords` behave exactly as before.

---

## Task 2: Add focused dampeners to `file_write`

**Objective:** Stop loading write/patch tools when Dale is discussing files, plans, or approaches rather than asking for an edit.

**Files:**

- Modify: `presets/aggressive.yaml`

**Implementation:**

Add `exclude_keywords` under the `file_write` trigger. Prefer phrase-level dampeners. Avoid single-word dampeners that suppress valid action requests too often.

Suggested first pass:

```yaml
  - name: file_write
    tools: [write_file, patch]
    keywords:
      - '\b(write|create|save|update|edit|modify|patch|append|fix|change|delete|rewrite)\b'
      - '\.(md|py|js|ts|tsx|jsx|json|yaml|yml|toml|sh|html|css|sql|txt)\b'
      - '/\S+\.\S+'
      - '~/\S+'
    exclude_keywords:
      - '\b(should we|would we|could we|might|maybe|eventually|later|future)\b'
      - '\b(what do you think|talk through|approach|strategy|idea|thinking about)\b'
      - '\b(not asking you to|don\x27t|do not)\b.*\b(write|edit|change|patch|save|commit)\b'
      - '^\s*>'
      - '```'
```

Notes:

- `should`, `would`, and `could` alone may be too broad. Use phrase-level patterns where possible.
- Do not suppress direct requests like “Can you write this to a file?”
- Code-block suppression is useful because pasted examples often contain verbs and filenames that are not instructions.

**Verification fixture ideas:**

Should fire:

- “Save notes on this to a markdown file.”
- “Patch `presets.py` with that change.”
- “Update `/Users/macmini/.hermes/plugins/dynamic-tools/PLAN.md`.”

Should not fire:

- “Should we save this as a file later?”
- “What do you think about the file-write trigger?”
- “Talk through the approach before editing anything.”
- “Don’t patch it yet.”

---

## Task 3: Tighten `delegation` trigger vocabulary

**Objective:** Avoid loading delegation tools when Dale is talking about delegation, scale, or concurrency conceptually.

**Files:**

- Modify: `presets/aggressive.yaml`

**Implementation:**

Make delegation more explicit. Remove or dampen weak scope/concurrency-only triggers like “big task,” “massive,” and “both at once” unless paired with a delegation/action verb.

Suggested first pass:

```yaml
  - name: delegation
    tools: [delegate_task]
    keywords:
      - '\b(delegate|sub-?agents?|spawn|orchestrat(e|or))\b'
      - '\b(use|run|spin up|ask)\b.{0,40}\b(Claude Code|CC|worker|sub-?agent)\b'
      - '\b(in parallel|simultaneously|concurrent(ly)?)\b.{0,80}\b(do|run|work on|build|research|implement|fix)\b'
    exclude_keywords:
      - '\b(should we|would we|could we|might|maybe|eventually|later|future)\b'
      - '\b(idea|approach|strategy|thinking about|talk through|discussion)\b'
      - '\b(not yet|not now|don\x27t delegate|do not delegate)\b'
```

**Verification fixture ideas:**

Should fire:

- “Delegate this to Claude Code using your artifact wrapper.”
- “Spin up CC to review this plugin.”
- “Use subagents to research these in parallel.”

Should not fire:

- “We had ideas about delegation.”
- “Should we use subagents for this eventually?”
- “This is a big project, but let’s talk first.”

---

## Task 4: Tighten `cronjob` trigger shape

**Objective:** Avoid loading cron tools when the user is discussing schedules, habits, or cron conceptually rather than asking to schedule something.

**Files:**

- Modify: `presets/aggressive.yaml`

**Implementation:**

Make cron triggers require scheduling intent plus time/frequency/action. Avoid plain `daily`, `weekly`, or `every day` as standalone positives.

Suggested first pass:

```yaml
  - name: cronjob
    tools: [cronjob]
    keywords:
      - '\b(schedule|recurring|repeating)\b.*\b(task|job|check|run|send|reminder|notify)\b'
      - '\bremind me\b.*\b(every|daily|weekly|at|tomorrow|on monday|on tuesday|on wednesday|on thursday|on friday|on saturday|on sunday)\b'
      - '\bnotify me\b.*\b(daily|weekly|every|when|at)\b'
      - '\bset (a |an )?(alarm|reminder|timer)\b'
      - '\bcron(job)?\b.*\b(create|add|schedule|run|edit|remove|pause|resume)\b'
    exclude_keywords:
      - '\b(should we|would we|could we|might|maybe|eventually|later|future)\b'
      - '\b(how does|how do|what is|explain|talk through|approach)\b.*\b(cron|schedule|scheduled|reminder)\b'
      - '\b(I|we)\b.*\b(every day|daily|weekly)\b'
```

**Verification fixture ideas:**

Should fire:

- “Remind me every Friday at 9am to check this.”
- “Schedule a weekly report.”
- “Create a cron job to run this every morning.”

Should not fire:

- “I check this every day.”
- “How does our cron setup work?”
- “Should we schedule this later?”

---

## Task 5: Add suppressed-trigger telemetry

**Objective:** Make dampener behavior auditable without full scored telemetry.

**Files:**

- Modify: `predictor.py` and/or `presets.py` depending on current structure.
- Modify: `logger_io.py` only if the prediction record schema needs an explicit field.
- Modify: `analyze.py` to summarize suppression counts only if cheap.

**Implementation direction:**

Extend trigger evaluation so a group can report why it did not fire:

```json
"suppressed_triggers": [
  {
    "name": "file_write",
    "matched_exclude": "discussion_framing"
  }
]
```

Keep the first pass simple. It is acceptable to log raw exclude pattern strings or signal names if the YAML grows names later. If logging raw regexes is too noisy, log only the trigger name and a count.

**Success criteria:**

- Prediction rows show when `file_write`, `delegation`, or `cronjob` was suppressed.
- Analyzer can later answer: “did suppression prevent a later `expand_tools` call for the same category?”

---

## Task 6: Add near-miss telemetry after dampeners settle

**Objective:** Identify false negatives where the model had to call `expand_tools` for a category that a trigger failed to load.

**Files:**

- Modify: `__init__.py` or expand-tools post-call path.
- Modify: `logger_io.py` if a new field is persisted.
- Modify: `analyze.py` to surface near-miss counts.

**Implementation direction:**

When `expand_tools(category="file")`, `expand_tools(category="terminal")`, etc. is called, link it back to the current prediction and log candidate trigger groups that did not fire.

Possible `tool_calls.jsonl` field on the `expand_tools` row:

```json
"evaluated_triggers_not_fired": ["file_write"]
```

Possible analyzer output:

```text
file_write: 6 suppressed, 1 later expanded file tools within sticky window
cronjob: 3 suppressed, 0 later expanded cronjob tools
```

This makes future scoring work data-driven if dampeners are not enough.

---

## Task 7: Update analyzer report wording

**Objective:** Keep reports aligned with the lighter strategy.

**Files:**

- Modify: `analyze.py`
- Reports generated under `reports/`

**Implementation direction:**

Add a short section when fields exist:

```text
Suppressed triggers:
- file_write: suppressed 12, later same-category expansion 1
- delegation: suppressed 3, later same-category expansion 0
- cronjob: suppressed 2, later same-category expansion 0
```

Keep recommendation language conservative:

- “tighten trigger” when false positives dominate
- “review dampener” when suppression is followed by same-category expansion
- “keep gated” when expansions are useful but not frequent enough for always-on

Do not auto-recommend terminal always-on from this pass unless the configured promotion thresholds are met.

---

## Testing Commands

Run from the plugin directory:

```bash
cd /Users/macmini/.hermes/plugins/dynamic-tools
python3 -m py_compile presets.py predictor.py logger_io.py analyze.py
python3 analyze.py --write-recommendations --format text
```

If tests exist for the plugin, run them. If no formal tests exist, add a small temporary fixture script or unit test that loads `presets/aggressive.yaml`, evaluates representative messages, and asserts expected trigger names.

Suggested fixture cases are listed in Tasks 2–4.

---

## Rollback Strategy

All changes are reversible with one git revert if kept in a tight commit.

Rollback levers:

- Remove `exclude_keywords` entries from YAML to disable dampeners while keeping code support.
- Revert `presets.py` to remove the code path entirely.
- Ignore new telemetry fields; JSONL consumers must tolerate missing/extra fields.
- Keep `learned_mode` unchanged; do not turn on adaptive auto-apply as part of this plan.

---

## Deferred: Full Scored Trigger Framework

The full scored framework should remain deferred until both conditions are true:

1. Dampeners and tighter trigger shapes still leave meaningful false positives or false negatives.
2. There are enough `message_preview`, suppression, and near-miss examples to tune weights from real traffic.

When that time comes, resurrect the old design ideas:

- named positive/negative signals
- thresholds per trigger group
- `trigger_scores` telemetry
- analyzer examples grouped by true positive, false positive, false negative, and near miss

For now, the system does not need another brain. It needs a few better reflexes.
