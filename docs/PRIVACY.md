# Privacy

What Tool Belt writes to disk, what it deliberately does not write, and
how to turn it off or erase it. Every claim here is checkable against
the source files cited beside it — if a property isn't stated, assume
the plugin does not implement it.

The plugin is local-only. It has no network code of its own: telemetry
is never transmitted anywhere, and nothing is phoned home.

---

## What is collected

When `log: true` (the default), the plugin appends to three JSONL files
under `$HERMES_HOME/state/tool-belt/` — `~/.hermes/state/tool-belt/`
when `HERMES_HOME` is unset ([`logger_io.py:_state_dir`](../logger_io.py)):

| File | One row per | Records |
|---|---|---|
| `predictions.jsonl` | inbound gateway message | which preset and triggers fired, the tool names before and after narrowing, estimated token counts, the message hash and preview |
| `tool_calls.jsonl` | tool call in a session | the tool name and attribution flags (was it cut, was it expanded, was it available already) |
| `api_calls.jsonl` | outbound API call | token and cache-hit counters, model/provider identifiers, and fingerprint hashes of the tool block and system message |

The purpose is measurement: the analyzer joins the three files on
`prediction_id` to answer whether narrowing predicted the tools the
model actually reached for, and what it cost or saved.

The full field-by-field schema is in
[CONFIGURATION.md § Telemetry outputs](CONFIGURATION.md#telemetry-outputs).
It is not repeated here.

## What is not collected

**Message text is never written.** The central privacy property. Each
prediction row carries two derived values and nothing else:

- `message_hash` — the first 16 hex characters of a sha1 of the message
  ([`logger_io.py:hash_message`](../logger_io.py)).
- `message_preview` — at most 80 characters, whitespace-collapsed
  (`" ".join(text.split())`, which strips newlines, tabs, and runs of
  spaces), truncated with an ellipsis
  ([`logger_io.py:message_preview`](../logger_io.py)).

The preview exists for log triage. It is a real excerpt of the user's
message: the first 80 characters of it. Treat it as such — it is short,
but it is not anonymised.

Both are derived from the current user message only, not from the
predictor's lookback window. Prior-turn messages are held in an
in-memory ring (`_PRIOR_MESSAGES_BY_SESSION` in
[`__init__.py`](../__init__.py)) that is never serialised to disk and
does not survive a gateway restart.

**Tool arguments and results are not logged, with one exception.** The
`tool_calls.jsonl` writer is passed `args` and `result` only when the
tool is `expand_tools`
([`__init__.py:_on_post_tool_call`](../__init__.py) — `log_args = args
if tool == "expand_tools" else None`). `expand_tools` is the plugin's
own meta-tool and its input schema accepts exactly two string
properties, `category` and `tool`
([`expand_tools.py`](../expand_tools.py)); its result is the plugin's
own expansion outcome. No other tool's arguments or results reach
disk.

**Also not written:** API keys, auth tokens, message attachments,
assistant output text, or system-prompt contents.

**Tool names are written; tool definitions are not.** Prediction rows
list tool names — the ceiling set, the allowed set, the cut set, and so
on. The schemas themselves (descriptions, parameter definitions) appear
only as `tool_list_hash`, a 16-hex-character sha256 prefix of the
serialized narrowed tool list ([`__init__.py:_tool_list_hash`](../__init__.py)).
Likewise `system_hash` in `api_calls.jsonl` is a 16-hex-character
sha256 prefix of the system message content
([`__init__.py:_system_message_hash`](../__init__.py)) — the hash is
recorded so cache-busting upstream changes are detectable; the content
is not.

## Where it lives, and how it is protected

State resolves to `$HERMES_HOME/state/tool-belt/`, defaulting to
`~/.hermes/state/tool-belt/`. Per-profile installs set `HERMES_HOME`,
so each profile's telemetry lands under that profile's own tree.

Two things worth being plain about:

- **Files inherit the process umask.** The writers call
  `path.open("a")` with no `chmod` and no umask adjustment
  ([`logger_io.py`](../logger_io.py)). On a typical default umask that
  means world-readable files. The plugin applies no permission
  hardening of its own. If the state directory needs to be
  unreadable to other local users, set that up yourself.
- **There is no encryption at rest and no access control.** The JSONL
  files are plain text. Anything that can read the directory can read
  the previews.

Telemetry never leaves the machine by way of this plugin. There are no
HTTP clients, sockets, or upload paths in the plugin's runtime modules.

## Retention

Telemetry is append-only by design and **the plugin never deletes
it**. There is no expiry, no size cap, and no automatic rotation. Files
grow until an operator intervenes.

Rotation is manual: [`scripts/rotate-telemetry.sh`](../scripts/rotate-telemetry.sh)
moves the live files into `state-dir/archive/reset-<UTC-timestamp>/`
and the next plugin write recreates fresh ones at the original paths.
Two caveats:

- Rotation **archives, it does not delete**. Rotated data stays on disk
  under `archive/` until removed by hand.
- The script rotates all three files — `predictions.jsonl`,
  `tool_calls.jsonl`, and `api_calls.jsonl`.

## Turning collection off

The shipped defaults are `enabled: true` and `log: true`
([`__init__.py:_CONFIG`](../__init__.py)): telemetry — including the
`message_preview` excerpt (up to 80 characters) and `message_hash`
described above — is recorded from first use. Two independent opt-out
switches in the `plugins.tool-belt` block of Hermes' `config.yaml`
(see [CONFIGURATION.md](CONFIGURATION.md#log)):

- `log: false` — stops every JSONL writer. All three logging hooks
  check it before doing any work
  ([`__init__.py`](../__init__.py): `_maybe_log_prediction`,
  `_on_post_tool_call`, `_on_post_api_request`). Narrowing continues to
  work; you simply stop observing it. Note that the analyzer and the
  learned overlay have nothing to learn from after this, so the shaping
  workflow effectively stops too.
- `enabled: false` — the master switch. `register()` short-circuits, no
  patches are installed, and the plugin is a no-op. Nothing is narrowed
  and nothing is logged.

## Erasing everything

There is no "forget me" command. Erasure is manual file deletion:

1. Stop the Hermes gateway.
2. Delete `$HERMES_HOME/state/tool-belt/` (the per-profile path for a
   profile install; `~/.hermes/state/tool-belt/` otherwise).

That removes:

- `predictions.jsonl`, `tool_calls.jsonl`, `api_calls.jsonl`
- `archive/` — every rotated snapshot
- `harvest/` — synthetic rows written by
  [`scripts/harvest-replay.py`](../scripts/harvest-replay.py), which
  carry the same hash-plus-preview shape as live rows
- `learned.json` and `learned_recommendations.json`
- `cache_mode_detection.json`
- `configure-state.json` (the `scripts/configure.py` sidecar)

One derived artifact lives **outside** the state directory: the
analyzer writes dated markdown reports to the plugin's own `reports/`
directory ([`analyze.py`](../analyze.py)), and those reports — along
with `learned_recommendations.json` — can quote message previews as
`sample_previews`. That only happens under the opt-in keyword-mining
flags (`--suggest-dampeners`, `--suggest-trigger-keywords`), which are
off by default. Delete `reports/` too for a complete erasure.
(`reports/` is in `.gitignore`, so these never enter version control.)

Deleting Tool Belt state does not touch Hermes core state — session
files, config, and any other plugin's data are unaffected. It is also
non-destructive to behavior: the plugin rebuilds its state files on the
next dispatch and falls back to shipped defaults for anything it can no
longer read.

## The learned overlay and the detection cache

Neither derived state file contains message content.

`learned.json` holds per-scope promote/demote decisions computed from
telemetry: tool-name lists (`carry`, `expand_only`), the shaper's
rationale block (`shaping`, with counts and a timestamp), and
optional trigger adjustments (`triggers`) — see
[CONFIGURATION.md § `learned.json` reference](CONFIGURATION.md#learnedjson-reference).
Tool names and counts, nothing more.

`cache_mode_detection.json` records, per scope, whether the provider
appears to honour prefix caching: `mode`, `locked_at`, `lock_reason`,
`hit_rate_at_lock`, `sessions_locked`, `last_model`
([CONFIGURATION.md § Cache mode detection state](CONFIGURATION.md#cache-mode-detection-state)).
Deleting it is safe; detection re-runs.

`configure-state.json` records one number per scope — the `bypass_rate`
that observation mode replaced, so a reset can restore it
([`scripts/configure.py`](../scripts/configure.py)).

## What reaches the model provider

Tool Belt only ever *removes* things from the outbound request. It
rewrites the tool-definition block to a narrowed subset before the
request is sent, and does nothing else to the payload: it adds no
fields, injects no context, and transmits no telemetry or derived state
to the provider. The one thing it contributes to the request is its own
`expand_tools` meta-tool, registered with Hermes at load
([`__init__.py`](../__init__.py) — `ctx.register_tool`) so the model
has a way to ask for what was cut. That is a tool definition, not user
data.

Net effect on outbound traffic: strictly less is sent than would have
been without the plugin.

## Scope and profile isolation

Telemetry rows are keyed by scope, formatted `agent:platform`
([`__init__.py`](../__init__.py)). Learned state, the detection cache,
and the configure sidecar are all keyed the same way, so one scope's
decisions never apply to another unless a bare-platform fallback key is
written deliberately (see
[`learned.py:scope_candidates`](../learned.py)).

Profiles are isolated by path: every state helper resolves through
`HERMES_HOME`, so a profile install reads and writes only its own
directory. No file is shared between profiles.

In-memory session state — the frozen tool set, the lookback ring, the
per-session cache-mode machine — is keyed by canonical session key and
held in plain dicts. It is process-local and does not survive a gateway
restart.

---

See also: [CONFIGURATION.md](CONFIGURATION.md) for the full telemetry
schema and every config knob, and
[ARCHITECTURE.md](ARCHITECTURE.md) for where in the request lifecycle
the narrowing happens.
