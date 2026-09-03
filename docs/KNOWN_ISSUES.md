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
that follows them. Cache hit rates for `gpt-5.4`-family scopes will be
noisier than for Anthropic models, where cache behavior is more
deterministic. The carry-all posture itself is unaffected: the tool
list stays byte-stable across the session, which is everything
`tool-belt` can do here. (Savings figures are not affected either —
on a caching route the plugin narrows nothing and claims nothing.)

**Workaround.** None owed by this plugin. The issue is upstream of
where tool curation lives, and no in-plugin change can prevent the
reasoning trace from mutating the prefix. If consistent prefix caching
matters more than reasoning output, route the scope to an Anthropic
model instead.

---

### `providers_off_models` doesn't prevent a scope's first session from running carry-all

**Symptom.** A provider (or model) is listed in `cache_auto.providers_off_models` —
a known non-caching route — but the very first session tool-belt runs for that
`scope|provider` bucket still ships the full ceiling under `cache_on_carry_all`.
Narrowing only starts on the next session.

**Cause.** The blocklist lock is written by `_update_cache_mode_detection`
(`__init__.py`), which only fires from the `post_api_request` hook — after at
least one API call has already gone out. The session's posture, though, is
resolved once, at the session's first dispatch
(`_resolve_cache_mode_for_session` → `_resolve_posture_for_provider`), before
any call has happened. Under `cache_mode: auto`, an unlocked bucket defaults
to `"on"` (protect prefix-cache stability until proven uncached), so a fresh
`scope|provider` bucket runs its first session carry-all even when the
provider is blocklisted. The blocklist lock lands during that first session's
calls and, like any other detection lock, takes effect starting the next one.

**Impact.** One session of missed narrowing per new `scope|provider` bucket on
a blocklisted provider. Moot once the bucket has locked, which the blocklist
forces almost immediately — and moot entirely for a scope whose bucket is
already locked off from prior data. Not a correctness bug: the carry-all
session still logs full telemetry and usage evidence.

**Workaround.** None needed for steady state. If the very first session's
token cost matters, set `cache_mode: off` for that scope explicitly rather
than relying on the blocklist to catch it in time.

---

### Removed: in-memory freeze state (gateway restarts, concurrent `/new`)

Earlier releases kept a per-session frozen tool snapshot in process
memory, which produced two documented sharp edges: a gateway restart
re-froze every active session (one cold-cache turn each), and `/new` in
one chat could evict a sibling chat's snapshot on the same platform. The
snapshot no longer exists — on a caching provider the plugin ships the
full ceiling minus `expand_tools` on every call, and the list is stable
by construction rather than by bookkeeping. A restart or a mis-targeted
reset now only drops the posture pin, which re-resolves on the next
dispatch from the on-disk detection cache with no change to the tool
list. Cache-off sticky residency is still in-memory and is cleared by a
restart, as before.
