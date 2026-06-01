# Known Issues

A short list of behaviors that look like bugs but aren't — at least not
ones `tool-belt` can fix from where it sits. If you're seeing
something here, the plugin is doing what it should; the cost lives
elsewhere.

For general limitations (CLI sessions bypassing the gateway, subagents
inheriting parent tools, etc.) see the "Limitations and known sharp
edges" section in the [README](../README.md).

---

### Codex Responses API: reasoning trace breaks prefix cache mid-session

**Symptom.** On the `gpt-5.4` family routed through `api_mode:
codex_responses`, a session shows `cache_read_tokens=0` on a turn even
though `system_hash` and `tool_list_hash` (recorded by this plugin in
`api_calls.jsonl`) are identical to the previous turn. Cache hit rates
for reasoning-producing models look noticeably worse than for non-
reasoning models on the same workload.

**Cause.** Codex Responses attaches a reasoning trace to the assistant
message. That trace is included in the next turn's API payload as part
of the conversation history, so the bytes after the cached prefix
mutate from one turn to the next. Two compatible mechanisms have been
observed:

1. The server-side response chain (`previous_response_id`) appears to
   have a shorter TTL than the documented 5–10 minute prompt-cache
   window, so a gap between turns can drop the chain even when prompt
   cache would still serve.
2. Reasoning-item content carried into the next request is not
   byte-stable across turns, so the prefix hash the provider matches
   against changes even when the tool block and system block don't.

**Impact.** Higher input-token billing on reasoning turns and the turn
that follows them. Savings figures for `gpt-5.4`-family scopes will be
noisier than for Anthropic models, where cache behavior is more
deterministic. The freeze mechanism itself is unaffected: the tool list
stays stable across the session, which is everything `tool-belt`
can do here.

**Workaround.** None owed by this plugin. The issue is upstream of
where tool curation lives, and no in-plugin change can prevent the
reasoning trace from mutating the prefix. If consistent prefix caching
matters more than reasoning output, route the scope to an Anthropic
model instead.

---

### Gateway restarts drop in-memory freeze state

**Symptom.** After a `hermes gateway restart` (or any other process
restart), the first message in a previously-active session comes back
with a cache miss — `cache_read_tokens=0` despite the session having
been mid-conversation. Subsequent turns in the same session cache
normally again.

**Cause.** The per-session frozen tool snapshot lives in
`_FROZEN_BY_SESSION`, an in-process dictionary in `__init__.py`. It is
not persisted to disk. A restart wipes it, so the next dispatch for
that session runs the predictor fresh and writes a new freeze. The
freshly computed tool block is functionally equivalent in most cases,
but the provider's cached prefix was keyed on the prior block's bytes,
so the first post-restart turn is a cold-cache turn.

Cache-mode detection state (`cache_mode_detection.json` at
`~/.hermes/state/tool-belt/`) IS persisted, so the re-freeze uses
the same cache mode as before without paying for another detection
window. Cache-off sticky residency is also in-memory and is cleared on
restart in the same way.

**Impact.** One cache-miss turn per active session, per restart. On
short sessions the cost is negligible; on long sessions with large
histories it's a one-time hit on whichever turn happens to arrive first
after the restart.

**Workaround.** Don't restart the gateway mid-session unless you need
to. Restarts to apply config or code changes are unavoidable; expect to
pay one cache-miss turn per session that resumes afterward.
