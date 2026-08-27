# Test Harness — Scripted Session Seeding for Onboarding Simulation

**Status:** design proposal (Task 7c). No harness code exists yet; nothing in this
document has been implemented.

**Problem.** `scripts/configure.py` discovers `agent:platform` scopes from
telemetry that the plugin's runtime hooks write during real gateway sessions.
Today the only way to exercise onboarding end-to-end is to send messages through
a live gateway on a real messaging platform and then run `configure.py` by hand.
There is no way to simulate that from the test suite.

**Goal.** A harness that stands up an isolated, throwaway Hermes home, populates
it with telemetry produced from *scripted conversations*, and lets
`configure.py` run against that populated state — exercising scope discovery,
the four-state machine, both onboarding paths, and the learned overlay write.

---

## Contents

- [1. What actually produces telemetry](#1-what-actually-produces-telemetry)
- [2. What `configure.py` and the shaper actually need](#2-what-configurepy-and-the-shaper-actually-need)
- [A. Approach recommendation](#a-approach-recommendation)
- [B. Scripted conversation format](#b-scripted-conversation-format)
- [C. Harness implementation plan](#c-harness-implementation-plan)
- [D. Integration with the test suite](#d-integration-with-the-test-suite)
- [E. Cost and CI feasibility](#e-cost-and-ci-feasibility)
- [F. Findings, blockers, and open questions](#f-findings-blockers-and-open-questions)

---

## 1. What actually produces telemetry

This section is the load-bearing one. The approach recommendation follows from
it directly.

### 1.1 `predictions.jsonl` requires a real LLM API dispatch

`predictions.jsonl` rows are written by exactly one function,
`_maybe_log_prediction` (`__init__.py:972`). That function has exactly two call
sites, both inside `_wrap_build_api_kwargs` (`__init__.py:337` for the wildcard
branch, `__init__.py:441` for the narrowed branch).

`_wrap_build_api_kwargs` is the monkey-patch installed over
`AIAgent._build_api_kwargs`. It runs when the agent is assembling the payload
for an outbound provider call.

> **The `pre_gateway_dispatch` hook does not write telemetry.** It runs the
> predictor and stashes the result in a contextvar; the row is written later,
> by the API-kwargs patch, on the first API call of the turn.

So a prediction row requires a genuine provider round-trip. Running the
predictor alone writes nothing.

### 1.2 …and the dispatch must come from the *gateway*, not the CLI

`_wrap_build_api_kwargs` returns early when the contextvar is empty
(`__init__.py:299-301`):

```python
state = _PREDICTION_CV.get()
if state is None:
    return kwargs
```

`_PREDICTION_CV.set(...)` appears at only two places, `__init__.py:1478`
(frozen-snapshot reuse) and `__init__.py:1565` (the normal path). **Both are
inside `_on_pre_gateway_dispatch`** (`__init__.py:1427`).

Consequences, verified against the installed CLI:

| Entry point | Fires `pre_gateway_dispatch`? | Writes telemetry? |
|---|---|---|
| `hermes chat` / `hermes -z PROMPT` | No — CLI path, not gateway dispatch | **No** |
| `hermes send --to telegram "..."` | No — its own `--help` states "no LLM, no agent loop, no running gateway required" | **No** |
| Inbound message to a running `hermes gateway` on a real platform | Yes | Yes |

There is no non-interactive CLI flag that drives the gateway dispatch path.

### 1.3 Scope attribution needs a real platform

`_agent_platform_from_context` (`__init__.py:916`) resolves `agent` from the
profile context and `platform` from `HERMES_SESSION_KEY` / kwargs hints /
`event.platform`. It deliberately refuses to synthesise a fallback
(`__init__.py:967`):

```python
if not agent or not platform:
    return "", "", ""
```

A blank scope is dropped by the shaper's grouping
(`shape-ceiling.py:231`: `if scope and sid`), so it never reaches
`configure.py`. Producing a usable scope through the real chain therefore
requires an actual gateway event carrying a platform.

**Net:** the "full gateway" approach needs a live provider *and* a running
gateway *and* a real messaging-platform account delivering inbound messages —
not merely a cheap model.

### 1.4 `tool_calls.jsonl`

Written by `logger_io.log_tool_call` from `_on_post_tool_call`
(`__init__.py:1790`). The promote-evidence flags (`was_expanded`,
`expand_tools_used`) are computed there from live expansion state, so the same
gateway dependency applies.

---

## 2. What `configure.py` and the shaper actually need

Read in the opposite direction — from the consumer back — the requirement is
much smaller than the producer chain.

### 2.1 Directory discovery

`discover_state_dirs` (`scripts/configure.py:190`) walks:

- the root scope: `<HERMES_HOME>/state/tool-belt`, surfaced when *either* that
  directory or `<HERMES_HOME>/sessions` exists;
- each named profile: `<HERMES_HOME>/profiles/<name>/state/tool-belt`, same
  either-or rule, skipping a child literally named `default`.

Nothing else on disk is consulted. No `config.yaml`, no session files, no
profile manifest.

### 2.2 Session counting

`_session_counts_by_scope` (`scripts/configure.py:218`) delegates to
`shaper.group_predictions_by_scope_session` (`shape-ceiling.py:219`), which
reads exactly three fields per row:

```python
scope = str(p.get("scope") or "")
sid   = str(p.get("hermes_session_id") or p.get("session_id") or "")
if scope and sid:
    out[scope][sid].append(p)
...
sessions[sid].sort(key=lambda p: p.get("ts", 0))
```

A "session" is **one distinct `hermes_session_id`** (falling back to
`session_id` on older rows). Turn count within a session is irrelevant to the
count.

### 2.3 The minimum viable prediction row

For `configure.py` + the shaper, a prediction row needs only:

| Field | Consumer | Required |
|---|---|---|
| `scope` | grouping (`shape-ceiling.py:229`) | **yes** — non-blank |
| `hermes_session_id` *or* `session_id` | grouping (`shape-ceiling.py:230`) | **yes** — non-blank |
| `ts` | recency sort + window (`shape-ceiling.py:235, 270`) | effectively yes (defaults to `0`, ties break arbitrarily) |
| `prediction_id` | join to `tool_calls.jsonl` (`shape-ceiling.py:281`) | yes, if tool calls matter |
| `always_on_tools` | demote evidence (`shape-ceiling.py:315`) | for demote |
| `unknown_kept_tools` | demote evidence (`shape-ceiling.py:318`, `mcp__`-prefixed entries skipped) | optional |

Every other field in `PredictionRecord.to_dict()` (`logger_io.py:168`) — the
token counts, `triggers_fired`, `policy_source`, `tool_list_hash`, `provider`,
`frozen_reuse`, and the rest — is **unread** by `configure.py` and
`shape-ceiling.py`. Those fields are consumed by `analyze.py` and
`scripts/savings-report.py` instead.

### 2.4 The minimum viable tool-call row

`index_tool_calls_by_prediction` (`shape-ceiling.py:239`) keys on
`prediction_id`. `compute_scope_recommendations` then reads `tool_name` plus
the boolean evidence flags (`shape-ceiling.py:286-291`):

- Rows named `expand_tools` are skipped outright.
- Promote evidence = `was_expanded` **or** `expand_tools_used` truthy.
- Any row at all (evidence or not) marks its `tool_name` as "used", which
  suppresses demotion.

### 2.5 Thresholds and the state machine

From `policy.yaml:14-18`:

```yaml
learning:
  shape_ceiling:
    session_window: 20
    promote_min_sessions: 2
    promote_min_calls: 3
    demote_min_sessions_no_use: 20
```

`required_sessions` (`scripts/configure.py:402`) is
`max(promote_min_sessions, demote_min_sessions_no_use)` = **20**.

`classify_scope` (`scripts/configure.py:416`) resolves, in order:

1. `learned_mode` normalises to `apply` → **`shaped`**
2. else effective `bypass_rate >= 1.0` → **`ready`** if `sessions >= 20`, else **`observing`**
3. else not configured and `sessions < 20` → **`fresh`**
4. else **`ready`** if `sessions >= 20`, else **`fresh`**

Note branches 1 and 2 read *config*, not disk state. The harness must be able to
control what `hermes config get` returns — see §D.2.

**Window interaction worth knowing:** discovery counts *all* sessions, but
`compute_scope_recommendations` truncates to the 20 most recent
(`shape-ceiling.py:270-274`). Seeding 22 sessions yields a status line reading
`22/20` while the shaping summary says "from 20 recorded session(s)". Both are
correct; a test asserting on both numbers must expect the difference.

### 2.6 Empirical confirmation

A throwaway probe (outside the repo) wrote hand-built rows carrying only the
fields in §2.3–§2.4 into a temp `HERMES_HOME`, then ran the real
`scripts/configure.py` against it. Results:

- `--status` reported `testagent:telegram   ready   ready to shape (22/20 sessions)`
- `--path shape` produced a full shaping summary, promoted `run_command`
  (expand evidence) and demoted `never_used_tool` (always-on, never called)
- the real shaper wrote a real `learned.json` with the expected
  `scopes.<scope>.{always_on, always_off, cache_aware}` shape
- a follow-up `--status` reported `shaped   shaping applied`

**Synthesised rows are indistinguishable from real ones at the `configure.py`
boundary.** Neither `configure.py` nor `shape-ceiling.py` contains any
provenance check, signature, or required-field validation that a synthesised row
could fail.

---

## A. Approach recommendation

### Recommended: **synthesised telemetry**, driven through the real predictor

Write `predictions.jsonl` and `tool_calls.jsonl` directly, but derive the
*content* of each row by calling `presets.resolve_preset()` and
`predictor.predict()` on the scripted message — the same two calls
`_on_pre_gateway_dispatch` makes (`__init__.py:1513-1514`). Only the transport
(gateway → provider → API-kwargs patch) is replaced.

Verified working standalone in-process:

```text
['shell']                      run the tests and check git status
['file_write', 'shell']        now commit the fix with git
['browser']                    browse to example.com and click on the login button
['web_extract', 'browser']     web scrape the pricing page
['history_search']             what did we decide about pricing yesterday?
```

This is the important refinement over "just fabricate rows": `triggers_fired`,
`always_on_tools`, and the narrowed tool set come from **real policy
evaluation against `policy.yaml`**, so a policy change that breaks trigger
matching will surface in the harness rather than being papered over by
hardcoded fixture values.

### Comparison

| | Full gateway | Synthesised (recommended) |
|---|---|---|
| Requires a provider/API key | Yes | No |
| Requires a running gateway | Yes | No |
| Requires a real platform account (Telegram/Slack/…) | **Yes** — see §1.3 | No |
| Network | Yes | No |
| Deterministic | No | Yes |
| Runtime for 20 sessions × 3 turns | minutes; rate limits; flaky | milliseconds |
| CI-viable | No | Yes |
| Exercises `configure.py` state machine | Yes | Yes |
| Exercises the shaper | Yes | Yes |
| Exercises real trigger matching | Yes | Yes |
| Exercises the gateway hook chain | Yes | **No** |
| Exercises the API-kwargs narrowing patch | Yes | **No** |

### Why the gap is acceptable

The uncovered surface is precisely `_on_pre_gateway_dispatch` →
`_wrap_build_api_kwargs` → `_maybe_log_prediction`. That surface is already
covered by the existing suite — `tests/test_session_attribution.py` (1,146
lines) drives attribution and hook behaviour directly, and
`tests/test_cache_aware.py` (664 lines) covers the freeze/reuse paths. The
thing this harness is meant to test is **onboarding**, which begins at the
telemetry files.

### The one honest risk, and how to close it

The harness fabricates the row *shape*. If `PredictionRecord.to_dict()`
(`logger_io.py:168`) gains, renames, or drops a field the shaper later reads,
a hand-rolled writer would drift silently.

**Mitigation — build rows through `logger_io.PredictionRecord` itself,** not
from a dict literal. Construct the dataclass, call `.to_dict()`, and append.
The harness then inherits the production schema by construction; a renamed
field becomes a `TypeError` at seed time rather than a silent test blind spot.
This costs nothing and removes the only structural objection to the approach.

Add one cheap guard test alongside it (§D.4): assert the field names the shaper
reads are present in `PredictionRecord.to_dict()` output.

### Not recommended: a hybrid tier

A gated, opt-in "real gateway" tier (`TOOLBELT_E2E_LIVE=1`) was considered and
is **not** recommended. Because of §1.3 it would need a provisioned messaging
platform account and a running gateway — CI-hostile, credential-bearing, and
manual to keep alive. The manual end-to-end check Dale already performs covers
the same ground at lower cost. If that judgement changes, the scripted format
in §B is transport-agnostic and could feed a live driver unchanged.

---

## B. Scripted conversation format

YAML (PyYAML is already the sole runtime dependency — `requirements-dev.txt`).
One file per conversation shape, under `tests/scripts/`.

### Schema

```yaml
name: <str>            # script id, used in test failure messages
agent: <str>           # scope left half
platform: <str>        # scope right half  -> scope == "<agent>:<platform>"
profile: <str|null>    # null/"default" -> root state dir;
                       # otherwise <HERMES_HOME>/profiles/<profile>/state/tool-belt
sessions: <int>        # how many distinct hermes_session_ids to emit
turns:
  - user: <str>              # message text, fed to the real predictor
    expect_triggers: [<str>] # optional; asserted at seed time (see below)
    calls:                   # optional; tool_calls.jsonl rows for this turn
      - <str>                # bare string  -> plain call, no expand evidence
      - tool: <str>          # mapping form
        expanded: <bool>     # sets was_expanded (promote evidence)
        count: <int>         # repeat this call N times (default 1)
```

`expect_triggers` is the deterministic trigger-testing hook the brief asks for.
The seeder compares it against the real `prediction.triggers_fired` and raises
on mismatch, so a policy regression fails the *seed*, loudly, instead of
producing quietly-wrong telemetry.

### Example 1 — terminal-heavy (`tests/scripts/terminal-heavy.yaml`)

Drives the promote arm: `terminal` is reached for via expansion, repeatedly.

```yaml
name: terminal-heavy
agent: testagent
platform: telegram
profile: null
sessions: 20
turns:
  - user: "run the tests and check git status"
    expect_triggers: [shell]
    calls:
      - read_file
      - tool: terminal
        expanded: true
        count: 2
  - user: "now commit the fix with git"
    expect_triggers: [file_write, shell]
    calls:
      - tool: terminal
        expanded: true
      - write_file
```

### Example 2 — browser-heavy (`tests/scripts/browser-heavy.yaml`)

A second scope in the same home, so multi-scope discovery and per-scope
independence get covered.

```yaml
name: browser-heavy
agent: testagent
platform: slack
profile: null
sessions: 20
turns:
  - user: "browse to example.com and click on the login button"
    expect_triggers: [browser]
    calls:
      - tool: browser_exec
        expanded: true
        count: 2
  - user: "web scrape the pricing page"
    expect_triggers: [web_extract, browser]
    calls: [browser_exec]
```

### Example 3 — chat-heavy (`tests/scripts/chat-heavy.yaml`)

Almost no tool use. Drives the **demote** arm: always-on tools sit in every
prediction row and never appear in `tool_calls.jsonl`. Also lands in a named
profile, covering `discover_state_dirs`' profile branch.

```yaml
name: chat-heavy
agent: assistant-b
platform: telegram
profile: assistant-b
sessions: 20
turns:
  - user: "what did we decide about pricing yesterday?"
    expect_triggers: [history_search]
    calls: [mnemosyne_recall]
  - user: "draft a reply to that email"
    expect_triggers: []
    calls: []
```

### A fourth, sub-threshold script

`tests/scripts/observing.yaml` — identical to `chat-heavy` but
`sessions: 5`. Needed to assert the `observing` and `fresh` states, which by
definition cannot be reached from a 20-session script.

### Design notes

- **`sessions` repeats the whole `turns` block.** Session *i* emits one
  prediction row per turn, all sharing `hermes_session_id = f"{name}-sess-{i}"`.
  A single script run therefore produces all 20 sessions; the harness does not
  loop externally.
- **`ts` is monotonic and seeded, never `time.time()`** — `ts = base + i*1000 + turn_index`,
  with `base` a fixed constant. This keeps the recency window in
  `compute_scope_recommendations` deterministic across runs.
- **Demote requires restraint in `calls`.** Any tool named in a `calls` list
  anywhere in the window is marked used and becomes undemotable
  (`shape-ceiling.py:326-328`). Scripts intended to drive demotion must leave
  the target tools uncalled.
- **Protected tools never demote.** `effective_protected_always_on`
  (`shape-ceiling.py:336`) shields `expand_tools`, `send_message`, `memory`,
  and friends. A test asserting "tool X was demoted" must not pick one of
  these.

---

## C. Harness implementation plan

### C.1 `tests/seed_sessions.py`

Importable module *and* a `__main__` entry point (so it can also be used by
hand to populate a scratch home for manual `configure.py` play).

Proposed surface:

```python
@dataclass
class SeedResult:
    hermes_home: Path
    state_dir: Path
    scope: str
    sessions: int
    predictions: int
    tool_calls: int

def load_script(path: Path) -> dict: ...

def seed(script: dict, hermes_home: Path, *, verify_triggers: bool = True) -> SeedResult: ...

def seed_all(scripts: Sequence[Path], hermes_home: Path) -> list[SeedResult]: ...
```

`seed()` steps:

1. Resolve `state_dir` from `profile` — root `<home>/state/tool-belt` when
   null/`"default"`, else `<home>/profiles/<profile>/state/tool-belt`. `mkdir(parents=True)`.
2. **Set `HERMES_HOME` for the duration of the preset/predictor calls.**
   `learned.state_dir()` (`learned.py:66-67`) reads the env var, and
   `resolve_preset` merges `learned.json` when `learned_mode` is `apply`
   (`presets.py:187`). Without this the seeder can read the *live* user's
   learned overlay. Non-negotiable for isolation.
3. Load the base policy once: `presets.load_base_policy()`.
4. Per session `i`, per turn `t`:
   - `preset = presets.resolve_preset(plugin_config, scope)`
   - `prediction = predictor.predict(turn["user"], attachments=None, preset=preset)`
   - if `verify_triggers` and `expect_triggers` is present, compare sorted sets
     and raise `ScriptMismatch` on divergence
   - build a `logger_io.PredictionRecord` (see C.2) and append `.to_dict()`
   - append a `tool_calls.jsonl` row per entry in `calls`, expanded by `count`
5. Write both files with a single `write_text` of joined JSON lines — cheaper
   than 1,200 appends, and byte-identical to what append-mode produces.

### C.2 Building the prediction row

Populate `PredictionRecord` from the real prediction so the row is
production-shaped, not minimal:

```python
allowed = prediction.allowed_tool_names
narrowed = [] if allowed is presets.WILDCARD_ALWAYS_ON else list(allowed)
always_on = list(preset.always_on) if isinstance(preset.always_on, list) else []

record = logger_io.PredictionRecord(
    ts=base_ts + i * 1000 + t,
    prediction_id=f"{script_name}-s{i}-t{t}",
    session_id=f"{agent}:main:{platform}:seed",
    hermes_session_id=f"{script_name}-sess-{i}",
    channel=scope,
    agent=agent,
    platform=platform,
    scope=scope,
    message_hash=logger_io.hash_message(text),
    message_preview=logger_io.message_preview(text),
    preset=prediction.preset_name,
    triggers_fired=list(prediction.triggers_fired),
    triggers_suppressed=list(prediction.triggers_suppressed),
    always_on_count=prediction.always_on_count,
    always_on_tools=always_on,
    ceiling_count=len(ceiling_tools),
    narrowed_count=len(narrowed),
    ceiling_tokens=..., narrowed_tokens=...,   # see the caveat below
    ...
)
rows.append(record.to_dict())
```

**Token counts must not use `logger_io.estimate_tokens()` on a real schema
payload.** The probe confirmed `token_estimator_name()` returns
`tiktoken-cl100k` when tiktoken happens to be installed and `chars-div-4`
otherwise (`logger_io.py:217-233`), so counts differ between machines and
between CI and a dev laptop. Since neither `configure.py` nor the shaper reads
these fields (§2.3), the harness should write **fixed synthetic integers** and
record `tokens_estimator="seeded"` — honest provenance, and deterministic.

> If a later task wants the harness to feed `savings-report.py` too, that
> decision needs revisiting; a savings assertion on seeded token counts would
> be measuring the seed constant, not the plugin. Out of scope here.

### C.3 Producing enough sessions

One script run produces `sessions` distinct sessions — 20 by default, matching
`required_sessions()`. No external loop. Tests should read the threshold from
`configure.required_sessions(configure.shape_thresholds())` rather than
hardcoding `20`, mirroring `tests/test_configure.py:139`, so a `policy.yaml`
change doesn't silently invalidate the fixtures.

For sub-threshold states, override at call time:
`seed(script, home, sessions_override=5)`.

### C.4 Running `configure.py` against the populated state

Two supported drivers, both already precedented in the suite:

**In-process (preferred for tests).** Load `configure.py` by path, patch the
CLI seam, call `main()` with `--hermes-home`:

```python
rc = configure.main(["--status", "--hermes-home", str(home)])
```

**Subprocess (for a smoke check).** `python3 scripts/configure.py --status`
with `env["HERMES_HOME"]` set. Works, but see the blocker in §F.1 — a
subprocess without a fake `hermes` on PATH cannot reach the overlay write.

---

## D. Integration with the test suite

### D.1 Reuse what exists

`tests/test_configure.py` already has most of the isolation scaffolding, and
the new tests should reuse it rather than reinvent it:

| Helper | Location | Role |
|---|---|---|
| `TempHomeTestCase` | `tests/test_configure.py:128` | temp home whose path deliberately contains a space |
| `FakeRunner` | `tests/test_configure.py:38` | records argv, replays canned `hermes config get` stdout |
| `isolate(runner, which, sink)` | `tests/test_configure.py:59` | patches `subprocess.run`, `_default_runner`, `shutil.which`, and output |
| `make_ctx` | `tests/test_configure.py:122` | builds a `RunContext` with a silent `out` |
| `_fs_snapshot` | `tests/test_configure.py:510` | "wrote nothing" assertions |
| `seed_telemetry` | `tests/test_configure.py:86` | the existing 5-key fixture writer |

`seed_telemetry` stays as-is — the unit tests around it are fine and fast.
`seed_sessions.py` is the richer, script-driven complement, not a replacement.
Nothing in `tests/test_configure.py` needs to change.

### D.2 New file: `tests/test_onboarding_e2e.py`

`FakeRunner` returns `"Config key not set: <key>"` for every read by default,
so config-dependent states must be driven by seeding `get_values`. The
`classify_scope` branches map to fixtures like this:

| Target state | Seeded sessions | `FakeRunner.get_values` for `plugins.tool-belt.channels.<scope>.*` |
|---|---|---|
| `fresh` | 5 | *(none)* |
| `observing` | 5 | `bypass_rate: "1.0"` |
| `ready` | 20 | `bypass_rate: "1.0"` (or none — branch 4 also yields `ready`) |
| `shaped` | 20 | `learned_mode: "apply"` |

Proposed cases:

1. **`test_fresh_scope_reports_fresh`** — seed `observing.yaml` (5 sessions),
   no config. `--status` output contains `fresh` and `5/20`.
2. **`test_recommend_path_moves_scope_to_observing`** — seed 5 sessions, run
   `--path recommend --yes`, assert the emitted writes are exactly
   `learned_mode=recommend` and `bypass_rate=1.0`, assert
   `configure-state.json` was written (`remember_previous_bypass`,
   `scripts/configure.py:462`), then re-classify with those values fed back
   through `FakeRunner.get_values` and assert `observing`.
3. **`test_threshold_crossing_moves_observing_to_ready`** — same config, seed
   the full 20-session script, assert `ready` and `20/20`.
4. **`test_shape_path_writes_overlay_and_flips_to_shaped`** — the spine.
   Seed `terminal-heavy.yaml`, run `--agent default --path shape --yes`,
   assert: the summary names the promoted tool; `learned.json` exists with
   `scopes[scope].always_on` containing `terminal`; the two config writes are
   `learned_mode=apply` and `bypass_rate=0.0`; re-classifying with
   `learned_mode: "apply"` yields `shaped`.
5. **`test_demote_arm_from_a_chat_heavy_script`** — seed `chat-heavy.yaml`
   into a named profile, shape it, assert `scopes[scope].always_off` is
   non-empty, that it excludes every member of
   `effective_protected_always_on()`, and that the overlay landed in the
   *profile* state dir and not the root one.
6. **`test_multiple_scopes_are_discovered_and_shaped_independently`** — seed
   `terminal-heavy` and `browser-heavy` into one home; shape only the first;
   assert the second stays `fresh`/`ready` and gains no `learned.json` entry.
7. **`test_dry_run_writes_nothing`** — `--dry-run` with `_fs_snapshot()`
   before/after, and `runner.writes == []`.
8. **`test_status_is_read_only`** — `--status` over every seeded scope;
   `runner.calls` contains no `config set` / `config unset`.

### D.3 Determinism and isolation checklist

- **No network, no model, no provider key.** Nothing in the seed path opens a
  socket.
- **No `time.time()` in seeded rows.** Fixed `base_ts`. Note the *shaper* still
  stamps `computed_at`/`updated_at` with real time
  (`shape-ceiling.py:348, 381`); assertions must not compare those.
- **No `HERMES_HOME` leakage.** Wrap the predictor calls in
  `mock.patch.dict(os.environ, {"HERMES_HOME": str(home)})`.
- **Clear `learned.py`'s mtime cache between tests.** `learned.py:112` caches
  `learned.json` by mtime; two tests writing different overlays inside the same
  mtime granularity could otherwise cross-contaminate. Add an explicit cache
  reset in `setUp`.
- **`shutil.which` always patched**, so a developer's real `hermes` binary can
  never be reached (§F.1).
- **Temp home per test** via `TempHomeTestCase`; `addCleanup` handles teardown.

### D.4 Schema-drift guard

One small test, cheap insurance for the §A risk:

```python
def test_seeded_rows_carry_every_field_the_shaper_reads(self):
    row = seed_sessions.build_prediction_row(...)  # via PredictionRecord.to_dict()
    for field in ("scope", "hermes_session_id", "ts", "prediction_id",
                  "always_on_tools", "unknown_kept_tools"):
        self.assertIn(field, row)
```

### D.5 Discovery

`tests/run_tests.py` discovers `pattern="test_*.py"` from `tests/`.
`test_onboarding_e2e.py` is picked up automatically.

`seed_sessions.py` does **not** match the pattern, so it will not be
collected as a test module — correct, since it is a helper. The
`tests/scripts/*.yaml` files are inert data.

---

## E. Cost and CI feasibility

**Cost: zero.** No provider call, no API key, no token spend. The "cheap model"
question in the brief is moot under the recommended approach — the model is
never invoked. (For the record, had the gateway approach been chosen the
blocker would not have been model price but the platform-account requirement in
§1.3.)

**CI: yes, unconditionally.** `.github/workflows/ci.yml` runs
`.venv/bin/python tests/run_tests.py` on Python 3.11 and 3.12 with
`requirements-dev.txt` (PyYAML only). The harness adds no dependency: PyYAML for
the scripts, `unittest` and `tempfile` from the stdlib. No secrets, no network,
no services.

**Runtime estimate.** The dominant cost is `predictor.predict()` — 14 trigger
groups of regexes per turn. At 3 scripts × 20 sessions × 2 turns = 120
predictions, plus ~1,200 JSON lines written and re-read by the shaper. The
probe runs described in §2.6 (22 sessions, two full `configure.py` invocations
including a real shaper pass and overlay write) completed in well under a
second each.

**Estimate: under 2 seconds for the whole new test file.** This should be
confirmed by measurement once implemented, not taken on faith.

One easy optimisation if it ever matters: `resolve_preset` is called per turn
in C.1 step 4, but its result is constant for a given `(plugin_config, scope)`.
Hoisting it out of the loop is safe and removes most of the per-turn cost.

---

## F. Findings, blockers, and open questions

### F.1 Blocker (for the subprocess driver): no `hermes` on PATH ⇒ no overlay

Discovered empirically. Running `configure.py --path shape --yes` in a
subprocess with `hermes` removed from `PATH` prints the full shaping summary and
then the manual-commands fallback — but **never writes `learned.json`**.

The cause is `flow_shape` (`scripts/configure.py:901-906`):

```python
writes = plan_shape_writes(info, ctx.settings(info.scope))
if not _confirm_writes(ctx, f"Config changes for {info.scope}:", writes):
    ctx.out(f"  Skipped {info.scope}. Nothing written.")
    continue
changed = write_learned_overlay(info, recs, dry_run=ctx.dry_run)
```

`_confirm_writes` returns `False` in degraded mode (`scripts/configure.py:876-878`),
so the `continue` fires before the overlay write. This is correct product
behaviour — don't half-apply when config can't be written — but it means:

> **Any test asserting on `learned.json` must fake the `hermes` CLI.** Patching
> `shutil.which` (via `isolate`) plus `_default_runner` is mandatory, not
> optional. Stripping `PATH` is not a substitute.

Confirmed working: with `hermes_available` forced true and a fake runner
installed, the same seeded state produced a correct `learned.json` and the
`ready → shaped` transition.

### F.2 Finding: config reads and writes do not round-trip in `FakeRunner`

In the probe, `--path recommend --yes` reported `Applied: hermes config set …
bypass_rate 1.0`, but a subsequent `--status` in the *same process* still read
`ready`, not `observing` — the fake store never fed the writes back into
`read_plugin_config`.

This is a fake-fidelity issue, not a product bug (`main()` reads config once at
`scripts/configure.py:1083`, before any flow runs — real `hermes` would persist
to disk and the *next* invocation would see it). But it shapes the test design:
**state transitions must be asserted by re-invoking `main()` with the new values
seeded into `FakeRunner.get_values`,** not by expecting one process to observe
its own writes. Cases 2 and 3 in §D.2 are written accordingly.

An optional upgrade — a `RecordingConfigStore` where `config set` mutates the
dict that `config get` reads — would make multi-invocation tests read more
naturally. Worth building only if §D.2 gets unwieldy.

### F.3 Finding: `resolve_preset` reads the live `learned.json` without `HERMES_HOME`

`learned.state_dir()` (`learned.py:66-67`) falls back to
`os.path.expanduser("~/.hermes")`. If the harness calls `resolve_preset` with
`learned_mode: apply` and no `HERMES_HOME` set, it reads **the developer's real
overlay** — non-deterministic, and machine-dependent. Hence the hard
requirement in §C.1 step 2. Read-only, so no corruption risk, but a real
correctness trap.

### F.4 Finding: profile scoping is asymmetric

`configure.py` discovers profiles at `<HERMES_HOME>/profiles/<name>/state/tool-belt`
(`scripts/configure.py:205-215`), but `learned.state_dir()` is the flat
`<HERMES_HOME>/state/tool-belt`. Both are right: at *runtime* a profile runs
with `HERMES_HOME` pointed at the profile directory, while `configure.py`
deliberately scans from the root so it can report on every profile at once.

The harness must honour both: seed to the nested profile path, but if it ever
needs profile-accurate preset resolution, set `HERMES_HOME` to the *profile*
directory for that call. For the recommended design this only matters when a
script sets `profile:` **and** the test exercises `learned_mode: apply`
resolution — none of the §D.2 cases do, but it is a live edge for future work.

### F.5 Finding: `hermes profile create` cannot be driven into a temp home

`hermes profile create <name>` has `--clone`, `--clone-from`, `--no-alias`,
`--no-skills`, `--description` — but no `--home` / `--root` flag. It writes to
the ambient Hermes home. The recommended design sidesteps this entirely by
creating the directory layout directly (§2.1 shows `configure.py` needs nothing
more than the directories), which is also why "throwaway profile" needs no
`hermes` invocation at all.

**Unknown, not verified:** whether `hermes profile create` respects a
`HERMES_HOME` env override. Not investigated, because the recommended design
does not need it. It would need answering before any live-gateway tier.

### F.6 Observation: a trigger dampener gap the harness would surface

While validating script messages, `"make a picture of a cat"` fired
`['shell', 'image_gen']` — the `shell` group matched on `\bmake\b`
(`policy.yaml:98`, intended for the `make` build tool).

This is **not** a finding about the harness; it is an existing policy
behaviour, and `tests/test_trigger_dampeners.py` already covers this class of
problem. It is noted here as evidence for the §A argument: running the *real*
predictor over scripted messages surfaces policy behaviour that hand-written
fixtures hide. Whether this specific match is worth a dampener is a separate
question, out of scope for Task 7c.

### F.7 Open questions for Bernard/Dale

1. **Should the harness double as a `savings-report.py` fixture source?** If
   yes, the token-count decision in §C.2 needs revisiting — seeded constants
   would make savings assertions self-referential.
2. **Should `seed_sessions.py` live in `tests/` or `scripts/`?** `tests/` per
   the brief. But it is genuinely useful for manually populating a scratch home
   to demo onboarding, which argues for `scripts/`. `tests/` with a documented
   `__main__` is the middle path proposed here.
3. **Is a `RecordingConfigStore` (§F.2) worth building now, or deferred until
   the §D.2 fixtures prove awkward?** Recommendation: defer.

---

## Summary

| Question | Answer |
|---|---|
| Recommended approach | Synthesised telemetry, driven through the real predictor and built via `logger_io.PredictionRecord` |
| Does telemetry need a real API call? | Yes in production (`__init__.py:337, 441`) — which is exactly why the harness bypasses it |
| Can the gateway hook be synthesised? | Not usefully — it needs a running gateway on a real platform (§1.3) |
| Are synthesised rows indistinguishable to `configure.py`? | Yes — empirically verified (§2.6); no provenance or validation check exists |
| Sessions needed | 20 (`max(promote_min_sessions=2, demote_min_sessions_no_use=20)`, `policy.yaml:14-18`) |
| New files | `tests/seed_sessions.py`, `tests/scripts/*.yaml` (4), `tests/test_onboarding_e2e.py` |
| Existing files changed | None |
| New dependencies | None |
| CI-viable | Yes — no secrets, no network |
| Cost | Zero |
| Hard requirement | Fake the `hermes` CLI, or the overlay is never written (§F.1) |
