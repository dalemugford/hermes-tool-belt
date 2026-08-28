# Hermes Tool Belt

### Adaptive tool loadouts for agents. Carry only what you need, drop the rest. Sharpens with use.

A plugin for [Hermes Agent](https://hermes-agent.nousresearch.com).

## What it does

Tool Belt narrows the tool list shipped on every API call to the tools
the current message actually needs — and never the other direction.
Your configured `platform_toolsets` ceiling is the contract: Tool Belt
may remove tools from a payload and restore them on demand, nothing
more. Five capabilities:

1. **Adaptive tool narrowing.** Each API call ships only the tools
   predicted to be relevant. Unfamiliar tools stay available — Tool
   Belt never silently removes something it doesn't recognize.

2. **Deterministic intent prediction.** A regex/keyword classifier with
   dampeners and an attachment check chooses the relevant tool groups.
   No LLM-as-router, no extra API call, no non-deterministic
   mis-route — every decision is auditable in telemetry, and the
   shipped [policy.yaml](policy.yaml) is the truth.

3. **Cache-aware session loadouts.** For prefix-caching providers, the
   tool set freezes at session start and stays stable, so the provider
   prefix cache keeps hitting. Cache behavior is auto-detected per
   model and scope; non-caching providers get per-turn narrowing
   instead. You set nothing.

4. **`expand_tools` recovery.** The model's escape hatch: one call
   loads a missing category or tool mid-session, and repeated use of a
   category gets it promoted into future sessions. **The model's reach
   IS the promotion vote.**

5. **Safety guarantees.** Fails open — any internal error leaves your
   full tool set untouched. Never widens the ceiling. Keeps unknown
   tools available. Isolates all state per session.

```text
User: "search the docs for X and read me the relevant section"
  loadout:     14 tools (file, web, search)
  model calls: expand_tools(category="browser")
  result:      12 browser tools join the loadout
  → next session, browser is pre-loaded (shaper promoted it)
```

## Why narrow at all

Every message ships tool definitions to the model. For a heavy profile —
web, terminal, file, browser, MCP servers, custom skills — that's
8–15k tokens of overhead on a message whose answer is `"yes."` Your
`"hi"` costs the same as `"deploy the build script"` in tool overhead.

The naive fix — re-narrow every turn — fights provider
prefix-caching: changing the tool block between turns busts the cache
and re-bills the conversation history at full input rate. On a
multi-turn session, lost cache costs more than schema savings.

Tool Belt resolves the tension by cadence:

- **Cache-on** (Anthropic, OpenAI auto-cache): the tool set freezes at
  session start and is reused verbatim every turn, so the provider
  prefix cache stays hot. Illustrative shape: 37-tool ceiling →
  ~14-tool frozen loadout → ~95% cache hit on subsequent turns.
- **Cache-off** (providers without prefix caching): per-turn narrowing
  with short sticky residency for expanded tools. Illustrative: a
  37-tool ceiling narrowed to 14 on a shell-intent message saves
  roughly a third of the tool-schema overhead per message.

The numbers above are illustrative, not measured guarantees — the
methodology for measuring your own is in
[docs/SAVINGS.md](docs/SAVINGS.md).

## What's safe about it

- **Fails invisibly.** Any error — bad YAML, predictor exception, missing lookup — falls through to "no narrowing." Tool Belt can save you tokens; it can never break your agent.
- **Strict ceiling.** Can only *narrow* what your `platform_toolsets` config already allowed. Never adds a tool you didn't sanction.
- **Honest telemetry.** Every prediction logged with the savings math. Token counts via `tiktoken-cl100k` when installed, `chars/4` heuristic otherwise — every row records which estimator was used. Provider-billed tokens recorded directly from API responses. Anyone can verify your savings claims from the raw JSONL. See [docs/SAVINGS.md](docs/SAVINGS.md) for the methodology.

## Install and configure

Two commands. Install, then configure. The examples assume the launcher is
installed (onboarding offers this after your first apply; or link it by hand:
`mkdir -p ~/.local/bin && ln -s ~/.hermes/plugins/tool-belt/tool-belt ~/.local/bin/tool-belt`).

```bash
hermes plugins install dalemugford/hermes-tool-belt
tool-belt configure
```

`configure.py` is the front door. It finds every agent and platform you
run, reads the telemetry already on disk, and asks how you want to
start. It writes configuration only through `hermes config set` — never
by editing `config.yaml` — shows you a `before → after` line for every
change, and applies nothing without your explicit confirmation.

You pick one of two paths, per agent:

**Shape now** — for when you already have Hermes history. The command
analyzes it, then shows you in plain language what would change: which
tools stay loaded on every message, which move to on-demand, and which
trigger groups still fire. Nothing is written until you say yes. On
confirmation it writes the learned overlay and turns shaping on for the
agents you chose.

**Recommend first** — for when you would rather watch before narrowing.
The command puts the selected agents into observation mode: tool loading
stays exactly as it is today while telemetry accumulates. It tells you
how many more sessions each agent needs. Re-run the command later and it
offers the same review-and-confirm step for whichever agents are ready.

Either way, restart the gateway afterward so Hermes picks up the new
configuration.

Re-run `configure.py` any time. It always detects the current state and
offers what fits — shape an agent that just became ready, add agents,
review what shaping did, or reset an agent back to observation mode.

```bash
tool-belt configure --status
```

`--status` is read-only: per-agent state and how much data each has.

### Optional: real-tokenizer counts

For exact token counts in the savings report instead of the chars/4
heuristic:

```bash
pip install tiktoken
```

Auto-detected on next gateway restart; every row in `predictions.jsonl`
records which estimator was used (`tokens_estimator` field). The savings
delta is honest either way — both sides use the same estimator — but
absolute counts are more accurate with `tiktoken`. See
[docs/SAVINGS.md](docs/SAVINGS.md#which-tokenizer) for the full story.

### First-session check

After restarting the gateway, confirm it's alive:

```bash
tool-belt configure --status
```

To watch decisions as they happen:

```bash
tail -f ~/.hermes/state/tool-belt/predictions.jsonl | \
  jq -c '{scope, msg: .message_preview, triggers: .triggers_fired, saved: .tokens_saved, pct: .reduction_pct}'
```

The full telemetry schema is in
[docs/CONFIGURATION.md](docs/CONFIGURATION.md#telemetry-outputs); the
analyzer ([analyze.py](analyze.py)) does the joins properly when you
want real precision/recall numbers.

## Daily use

Nothing to do. The plugin runs on every message; `expand_tools` is the
model's recovery valve; between sessions the shaper folds real usage
back into the policy. Re-run `configure.py` whenever you want to review
what shaping did, promote a scope from observation, or reset an agent.

## If something feels off

**"A tool I need isn't loading."** The model should call
`expand_tools(category=...)` to recover it — that's the designed path,
and repeated use gets the category promoted in future sessions. For an
immediate fix, add it per-scope with `always_on_extra` in config. Check
`configure.py --status` to see what shaping is active for that agent.

**"Tools load that I never use."** That's carry cost the shaper will
eventually demote — or help it along: run
`python3 analyze.py` for demotion candidates, or
`configure.py --reset <agent>` to return an agent to observation mode.

**"I want it off right now."** Two switches, per scope or global:

```yaml
plugins:
  tool-belt:
    channels:
      default:telegram:
        bypass_rate: 1.0   # narrowing off on this scope; telemetry stays on
    # or the plugin entirely:
    enabled: false
```

Restart the gateway after either. `bypass_rate: 1.0` keeps telemetry
flowing (useful if you plan to turn narrowing back on); `enabled: false`
stops everything.

**"The savings numbers look wrong."** Check the `tokens_estimator` field
in your telemetry — `chars/4` without tiktoken, `tiktoken-cl100k` with.
Run `./tool-belt savings --json` for the independently checkable math
(the deprecated `python3 scripts/savings-report.py --json` still works).
Methodology and caveats: [docs/SAVINGS.md](docs/SAVINGS.md).

## Advanced configuration

`configure.py` covers the normal path. Everything it writes is a
documented config key; the underlying knobs stay available for anyone
who wants to drive them directly.

```yaml
plugins:
  tool-belt:
    enabled: true
    log: true

    # Global tweaks layered on top of policy.yaml
    always_on_extra: []
    always_off: []

    # A/B baseline cohort — deterministically ship the full toolset for
    # a fraction of sessions so savings have a control to compare
    # against. 1.0 on a scope fully disables narrowing there.
    bypass_rate: 0.0

    # Learned overlay: recommend (default) never applies learned.json;
    # apply merges it during preset resolution.
    learned_mode: recommend    # recommend | apply

    # Per-scope overrides
    channels:
      default:telegram:        # agent-scoped override
        learned_mode: recommend
        bypass_rate: 0.05      # 5% baseline cohort on this scope only
        always_on_extra: [terminal]
      slack:                    # platform-wide fallback
        bypass_rate: 1.0       # disable narrowing on Slack entirely
```

Scope keys are `{agent}:{platform}`; Hermes calls the root profile
`default`. Lookups try the full scope first, then fall back to the bare
platform. The legacy `off` / `auto` / `audit` values for `learned_mode`
migrate automatically (`off` → `recommend`, `auto`/`audit` → `apply`).

Every knob — cache-off sticky residency, predictor lookback, dampener
tuning, learned.json shape, full telemetry field reference — is
documented in [docs/CONFIGURATION.md](docs/CONFIGURATION.md). The full
operator command set (bootstrap, shaper, analyzer, savings report,
drift check) is in [scripts/README.md](scripts/README.md).

```bash
python3 ~/.hermes/plugins/tool-belt/scripts/bootstrap.py   # inspect recommendations, writes nothing
```

## How it works

One mechanism, two cadences, chosen per scope:

- **Cache-on** (prefix-caching providers): the first dispatch of a
  session runs the predictor and *freezes* the resulting tool set.
  Every later dispatch reuses it verbatim, so the prefix cache stays
  hot. `expand_tools` additions are folded into the frozen set for the
  rest of the session — one deliberate cache break, then stability.
  The freeze is evicted only at moments that already cost one:
  `/reset`, `/new`, session end, or compaction.
- **Cache-off** (no provider caching): per-turn narrowing — always-on
  tools, plus every trigger group whose regexes match the message,
  plus expanded/sticky tools, intersected with your configured
  ceiling.
- **Between sessions**, the shaper reads accumulated `expand_tools`
  evidence and writes per-tool promote/demote recommendations into
  `learned.json`, applied at runtime only when `learned_mode: apply`.

The full lifecycle — hooks, freeze mechanics, compaction handling,
telemetry joins — is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Limitations and known sharp edges

- **CLI sessions don't fire `pre_gateway_dispatch`.** CLI messages bypass
  the gateway, so neither narrowing nor freeze applies. No telemetry written.
- **Subagents are not narrowed.** When `delegate_task` spawns a subagent,
  the parent's tool curation is inherited; the predictor doesn't run again.
- **Cron jobs are not narrowed.** Cron sessions get explicit per-job
  toolsets via Hermes' own `_resolve_cron_enabled_toolsets`.
- **Unknown plugin tools default to always-on.** Tools whose names aren't
  in the policy's known sets are kept on (safer for shipping new plugins).
- **In-memory session state doesn't survive gateway restarts.** Both the
  cache-on frozen snapshot and cache-off sticky residency live in process
  memory. A restart between turns will re-freeze (cache-on) or clear
  sticky carry-over (cache-off). The cross-session detection cache and
  `learned.json` are on disk; only the per-session memory is volatile.
- **Concurrent chats on one platform** can leave a sibling chat's freeze
  un-evicted on `/new`. Deliberate tradeoff; see
  [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).
- **Codex (gpt-5.4) reasoning trace breaks cache mid-session.** When the
  model produces reasoning tokens, the resulting message-history mutation
  busts cache on the next call regardless of what the plugin does.
  Anthropic-side cache is more deterministic. See
  [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).
- **`learned_mode: apply` is not the default.** Adaptive promotion
  is deliberately opt-in until recommendations have been reviewed for a
  given scope. The default `recommend` never merges `learned.json`.

## Failure modes

By design, **anything that goes wrong falls through to no-narrowing**:

- Bad YAML syntax in policy.yaml → wildcard fallback (no narrowing)
- policy.yaml missing → wildcard fallback (no narrowing)
- Predictor exception → returns wildcard (no narrowing)
- Filter exception in patched `_build_api_kwargs` → returns unfiltered tools
- Learned-state load error → base policy returned as-is, plugin keeps running
- `agent` not in registry when `expand_tools` is called → returns honest
  error JSON to the model

The plugin should be invisible when broken, never destructive.

## Disabling

Either:

```yaml
plugins:
  tool-belt:
    enabled: false
```

…or remove the directory entirely:

```bash
rm -rf ~/.hermes/plugins/tool-belt
```

## Releases

Versions use CalVer: `YYYY.M.D`, with an optional prerelease suffix
(for example `2026.5.17-beta`). The authoritative version is the one in
[`plugin.yaml`](plugin.yaml); each stable snapshot is marked with a
matching release tag. Pin a tag for stability, or track `main` for the
latest changes. See [CHANGELOG.md](CHANGELOG.md) for what has changed.

## Files

```
~/.hermes/plugins/tool-belt/
├── plugin.yaml       # Hermes plugin manifest
├── __init__.py       # runtime hooks, cache-mode detection, frozen loadouts
├── predictor.py      # deterministic intent matching
├── presets.py        # policy loading and per-scope resolution
├── learned.py        # learned overlay loading and merging
├── expand_tools.py   # dynamic recovery meta-tool
├── logger_io.py      # telemetry writers
├── analyze.py        # telemetry analyzer
├── policy.yaml       # shipped policy and shaping thresholds
├── AGENTS.md         # repository engineering rules
├── CHANGELOG.md      # unreleased and tagged user-facing changes
├── CONTRIBUTING.md   # contributor workflow and verification rules
├── LICENSE           # MIT license
├── docs/             # architecture, configuration, savings, known issues
├── scripts/          # configure (front door) + operator and verification commands
└── tests/            # regression and integration tests
```

## Companion docs

The code is the source of truth. The active doc set is intentionally small:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the plugin works: the two modes, the freeze mechanism, `expand_tools`, between-session shaping, where in the request lifecycle the patches sit.
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) — every knob in `config.yaml`, `policy.yaml`, and `learned.json`, with defaults; full telemetry field reference.
- [docs/SAVINGS.md](docs/SAVINGS.md) — how the token-savings numbers are computed and how to verify them on your own data.
- [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) — behaviors that look like bugs but aren't (Codex reasoning cache, gateway-restart freeze loss, concurrent-chat resets).
- [docs/PRIVACY.md](docs/PRIVACY.md) — what telemetry records and what it never records, where state lives, and how to disable collection or erase it.
- [docs/RELEASING.md](docs/RELEASING.md) — the maintainer release checklist: pre-release verification, clean-profile install test, behavioral spot-checks, version and tag steps.
- [docs/TEST_HARNESS.md](docs/TEST_HARNESS.md) — how onboarding is tested end to end from scripted conversations, without a gateway, a provider, or a messaging platform.

## License

[MIT](LICENSE) © 2026 Dale Mugford.