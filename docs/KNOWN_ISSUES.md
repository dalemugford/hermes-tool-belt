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

### Shell-bypass detection: terminal-only, and detection-only (v1)

**Symptom.** The model runs code in an `execute_code` sandbox — or a
subprocess spawned from one — that shells out to a native tool's CLI
shadow, and no `bypass_events.jsonl` row appears. Or: a bypass is
detected and recorded, but nothing at the time of the call nudges the
model toward the native tool; the shell path still runs.

**Why.** The v1 detector inspects only the `terminal` tool's `command`
string ([`bypass_map.py`](../bypass_map.py),
[`_maybe_log_shell_bypass`](../__init__.py)). `execute_code` carries a
code body, not a single command line, so catching a CLI shadow inside it
needs deeper code-content inspection (import graphs, subprocess-argv
reconstruction) that is deliberately out of scope for v1 — a narrow,
high-precision terminal matcher earns trust before a fuzzier one is
added. The detector is also **detection-only**: it produces a promotion
signal for the offline shaper, and does not intervene at call time.

**Workaround / future direction.** None owed today — a missed
`execute_code` bypass is a lost learning signal, not a correctness or
privacy problem, and the promotion path still learns from every
`terminal` bypass and from the expand-only harvest path. Two natural v2
extensions, both opt-in: (1) extend the pattern table to inspect
`execute_code` bodies; (2) a runtime intervention that, on a detected
bypass, prefers the native tool — behind a config flag, off by default,
since intervention changes behavior rather than just observing it.
