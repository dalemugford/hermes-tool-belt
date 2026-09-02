# Hermes Tool Belt

### Adaptive tool loadouts for agents. Carry only what you need, drop the rest. Sharpens with use.

A plugin for [Hermes Agent](https://hermes-agent.nousresearch.com).

> On the install it was built against, Tool Belt has saved **9.5 million
> tokens** of measured overhead — about 60k tokens and three cents of API
> cost per conversation, with every tool still one ask away. Run
> `hermes tool-belt savings` to see yours.

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

Two commands. Install, then configure:

```bash
hermes plugins install dalemugford/hermes-tool-belt
hermes tool-belt configure
```

`hermes tool-belt configure` and `hermes tool-belt savings` are the
canonical commands. Onboarding also offers to install a `tool-belt`
launcher after your first apply (or link it by hand: `mkdir -p
~/.local/bin && ln -s ~/.hermes/plugins/tool-belt/tool-belt
~/.local/bin/tool-belt`); the launcher takes identical flags and runs
identical code — pure convenience.

`configure` is the front door: a mode-setter. It finds every agent and
platform you run, reads the telemetry already on disk, and lets you set
the shaping mode per agent and channel. It writes configuration only
through `hermes config set` — never by editing `config.yaml` — shows
you a `before → after` line for every change, and applies nothing
without your explicit confirmation.

Interactively you pick an agent, then one of two areas:

1. **Protected tools** — a picker over the agent's tool inventory.
   Selected tools are written to `plugins.tool-belt.always_carry`:
   always carried, never shaped.
2. **Tool shaping options** — pick channels, then a mode:
   - **learning** — shaping on (`learned_mode: apply`); the plugin
     shapes automatically from future usage.
   - **history** — shaping on, and a shaping pass runs over your
     recorded sessions right away, with a review-and-confirm diff.
   - **off** — observation mode (`learned_mode: recommend`): every
     enabled tool is carried, telemetry keeps accumulating, and the
     learned overlay is kept but not applied.

Non-interactively: `--agent <name> --mode learning|history|off`, optionally
`--channel <platform>` to target one of the agent's channels (defaults to all).
Restart the gateway afterward so Hermes picks up the new
configuration. Re-run `configure` any time — it always detects the
current state.

```bash
hermes tool-belt configure --status
```

`--status` is read-only. It prints `Tool Belt: enabled|disabled`, then
an `Agents` list with one row per scope:
`N. scope  shaping ON/OFF (…)` — the parenthetical shows the learned
carry/expansion counts once a scope is shaped, `learning` while
evidence is still accumulating, or `observing` when shaping is off.

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
model's recovery valve; at session end the in-process auto-shape pass
folds real usage back into `learned.json`. Re-run `hermes tool-belt
configure` whenever you want to review what shaping did, protect a
tool, or switch an agent's mode.

## If something feels off

**"A tool I need isn't loading."** The model should call
`expand_tools(category=...)` to recover it — that's the designed path,
and repeated use gets the category promoted in future sessions. For an
immediate, permanent fix, pin it with `plugins.tool-belt.always_carry:
[tool_name]` (global, or additively under `channels.<scope>.`) — the
Protected tools picker in `hermes tool-belt configure` writes the same
key. Check `hermes tool-belt configure --status` to see what shaping is
active for that agent.

**"Tools load that I never use."** That's carry cost the shaper will
eventually demote — or help it along: run
`python3 ~/.hermes/plugins/tool-belt/analyze.py` for demotion candidates, or
`tool-belt configure --mode off --agent <agent>` to switch an agent to
observation mode.

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

    # A/B baseline cohort — deterministically ship the full toolset for
    # a fraction of sessions so savings have a control to compare
    # against. 1.0 on a scope fully disables narrowing there.
    bypass_rate: 0.0

    # Per-agent always-carry pins: union with the shipped structural
    # baseline, never demotable, inert for tools Hermes has disabled.
    always_carry: []

    # Learned overlay: apply (the default) merges learned.json during
    # preset resolution; recommend is the opt-in observe/trial mode that
    # never applies it.
    learned_mode: apply        # apply (default) | recommend

    # Per-scope overrides
    channels:
      default:telegram:        # agent-scoped override
        learned_mode: recommend
        always_carry: []       # adds to the global pins (union, never removes)
        bypass_rate: 0.05      # 5% baseline cohort on this scope only
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
telemetry rotation) is in [scripts/README.md](scripts/README.md).

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
- **`learned_mode: apply` IS the default** (zero-config contract): a fresh
  scope carries every enabled tool, and evidence-driven demotions apply
  automatically as they accumulate. Set `learned_mode: recommend` on a
  scope to opt into observe/trial mode, where `learned.json` is never
  merged and shaping stays a human-reviewed proposal.

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
├── carrying.py       # resident-partition carrying model
├── shaping.py        # evidence-driven shaping + in-process auto-shape pass
├── expand_tools.py   # dynamic recovery meta-tool
├── logger_io.py      # telemetry writers
├── analyze.py        # telemetry analyzer
├── savings.py        # savings-report computation
├── savings_cli.py    # savings-report rendering and CLI
├── cli.py            # `hermes tool-belt` subcommand registration
├── yaml_required.py  # PyYAML availability guard for the operator scripts
├── tool-belt         # public command launcher (configure, savings, …)
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

## License

[MIT](LICENSE) © 2026 Dale Mugford.