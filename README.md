# Hermes Tool Belt

### Adaptive tool loadouts for uncached agents. Carry only what you need, drop the rest. Sharpens with use. Reduces tool token overhead.

A plugin for [Hermes Agent](https://hermes-agent.nousresearch.com).

Tool definitions are cheap to carry on cached APIs (OpenAI, Anthropic) yet *costly on uncached ones.* Tool Belt shapes tool carry for Agents using uncached APIs (like Ollama Cloud, OpenRouter).

## Why Tool Belt
On a **cached API** tool definitions live in the
prompt prefix the provider caches. You send them once. Every turn reads
them back at the cache-read rate. Tool-definition cost is cheap.

However on **uncached APIs** there is no prefix to cache, and *every request re-sends every
tool definition at full input price*... whether an agent uses that tool or not in the session.

Which means agents might carry
8–15k tokens of tool overhead in a session which doesn't use any tools at all. _What if you could significantly save tool overhead tokens while still making tools available to your agents?_

## What Tool Belt Adds

Tool Belt optimizes Agent tool use in three main ways:
1. **It shapes tool loadouts based on real usage.** Tool Belt adds and reads agent telemetry. It learns which tools each agent actually reaches for, and decides (per tool and per agent) whether carrying that tool is worth the token carry cost.
2. **It defers tools that aren't worth carrying.** Tool Belt drops such a tool from the carried loadout and tucks it behind its own custom tool, `expand_tools`. A single call recovers it, and a tool the agent keeps reaching for is promoted back into the loadout on its own once it's invoked enough.
3. **It includes a deterministic prediction loader to provide tools just in time.** A deterministic predictor reads each incoming message and pre-loads tools it guesses your agent is about to need, before the agent reaches for them. This further saves tokens, preventing your agent from needing `expand_tools` at all in many cases. There is no LLM router, no extra API call. Every decision is written to telemetry, and used to shape your agent's toolset.

## Everything is Automatic

Tool Belt knows when your agent is using an uncached or cached provider.
It picks up when new tools have been added in Hermes. It picks up when tools are removed. And Tool Belt fails open if anything goes wrong.

## How it works

Tool Belt runs on every message, before your agent runs. It follows the same
four steps each time.

1. **It checks the provider.** If the provider caches (OpenAI, Anthropic), Tool
   Belt carries every tool and stops. There is nothing to save on a cached API.
2. **It builds the loadout.** On an uncached provider, Tool Belt ships a small
   set of tools for this message: your always-on tools, the tools the message
   looks like it needs, and any tools the agent asked for earlier in the
   session.
3. **Your agent runs.** If the agent needs a tool that is not loaded, it calls
   `expand_tools` to load it. The tool arrives in the same session.
4. **The session ends.** Tool Belt records what the agent used and what it
   reached for. Over time it promotes tools the agent uses and defers tools the
   agent ignores.

Every step is written to telemetry, so you can see why each tool was loaded. If
any step fails, Tool Belt ships your full tool set and the message runs as
normal.

The full lifecycle — hooks, the carry-all posture, compaction handling,
telemetry joins — is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## How to use it

Install the plugin, restart the gateway, and configure it:

```bash
hermes plugins install dalemugford/hermes-tool-belt   # add --enable to skip the enable prompt
hermes gateway restart
hermes tool-belt configure
```

Tool Belt is opt-in. The installer asks before it enables the plugin. If you
say no, enable it later with `hermes plugins enable tool-belt`. The gateway
loads plugins when it starts, so restart it after install.

`hermes tool-belt configure` is the front door. It finds every agent and
platform you run. It reads the telemetry already on disk. It lets you set the
shaping mode for each agent and channel. It writes settings only through
`hermes config set`, never by editing `config.yaml` by hand. It shows you a
`before → after` line for every change, and it applies nothing without your
confirmation.

Pick an agent, then pick one of two areas:

1. **Protected tools.** A picker over the agent's tools. Selected tools are
   always carried and never shaped. They are written to
   `plugins.entries.tool-belt.settings.always_carry`.
2. **Tool shaping.** Pick channels, then pick a mode:
   - **learning** — shaping is on. The plugin shapes from future usage.
   - **history** — shaping is on, and a shaping pass runs over your recorded
     sessions now. You review a diff and confirm before anything applies.
   - **off** — observation only. Every tool is carried, telemetry keeps
     recording, and nothing is applied.

To set a mode without the menus:

```bash
hermes tool-belt configure --agent <name> --mode learning|history|off|observe|reset
```

Add `--channel <platform>` to target one channel. It defaults to all channels.
Restart the gateway afterward so Hermes picks up the change. You can re-run
`configure` any time. It always reads the current state.

To check status without changing anything:

```bash
hermes tool-belt configure --status
```

`--status` prints whether Tool Belt is enabled, then one line per agent scope.
Each line shows the shaping state (`ON`, `OFF`, `learning`, or `observing`) and
how many tools the scope carries and defers.

### Daily use

There is nothing to do. The plugin runs on every message. `expand_tools` is the
model's recovery valve. At session end, Tool Belt folds real usage back into
`learned.json` on its own. Re-run `hermes tool-belt configure` when you want to
review what shaping did, protect a tool, or change a mode.

To see what you have saved:

```bash
hermes tool-belt savings
```

The numbers are measured from your own traffic and priced at what the tokens
were worth. Anyone can check them against the raw telemetry. The method is in
[docs/SAVINGS.md](docs/SAVINGS.md).

### Optional: exact token counts

By default Tool Belt estimates tokens with a `chars/4` heuristic. For exact
counts in the savings report, install `tiktoken`:

```bash
pip install tiktoken
```

Tool Belt detects it on the next gateway restart. The savings figure is honest
either way, because both sides of the math use the same estimator. `tiktoken`
just makes the absolute counts more accurate. Every telemetry row records which
estimator it used, in the `tokens_estimator` field.

## If something feels off

**A tool I need is not loading.** The model should call
`expand_tools(category=...)` to load it. That is the designed path, and repeated
use promotes the tool for later sessions. For an immediate, permanent fix, pin
the tool with `always_carry`. The Protected tools picker in
`hermes tool-belt configure` writes the same key. Run
`hermes tool-belt configure --status` to see what shaping is active.

**Tools load that I never use.** That is carry cost. Tool Belt demotes those
tools once it has the evidence — by default, a tool goes expand-only when it
sees no use across at least 2 sessions inside a rolling 7-day window. Anything
demoted in error costs one `expand_tools` round-trip and promotes itself back.
To shape from your recorded sessions now instead of waiting, run
`hermes tool-belt configure --mode history --agent <agent>` and review the diff.

**I want it off right now.** Two switches, per scope or global:

```yaml
plugins:
  entries:
    tool-belt:
      settings:
        channels:
          default:              # agent (Hermes calls the root profile "default")
            telegram:           # platform
              bypass_rate: 1.0  # narrowing off on this scope; telemetry stays on
        # or turn off the whole plugin:
        enabled: false
```

Set these through `hermes config set`, then restart the gateway.
`bypass_rate: 1.0` keeps telemetry recording, which is useful if you plan to
turn narrowing back on. `enabled: false` stops everything.

**The savings numbers look wrong.** Check the `tokens_estimator` field in your
telemetry. It is `chars/4` without tiktoken and `tiktoken-cl100k` with it. Run
`hermes tool-belt savings --json` for the checkable math. Method and caveats are
in [docs/SAVINGS.md](docs/SAVINGS.md).

## What's safe about it

- **It fails invisibly.** Any error — bad YAML, a predictor exception, a missing
  lookup — falls through to "no narrowing." A broken plugin is a silent no-op,
  not a broken agent.
- **It has a strict ceiling.** Tool Belt can only narrow what your
  `platform_toolsets` config already allowed. It never adds a tool you did not
  sanction.
- **Its telemetry is honest.** Every prediction is logged with the savings math.
  Provider-billed tokens are recorded straight from the API responses. Anyone
  can verify your savings from the raw JSONL. See
  [docs/SAVINGS.md](docs/SAVINGS.md).

## Advanced configuration

`hermes tool-belt configure` covers the normal path. Everything it writes is a
documented config key. You can also set those keys directly.

```yaml
plugins:
  entries:
    tool-belt:
      settings:
        enabled: true
        log: true

        # Internal testing feature — fraction of sessions shipped with the
        # full toolset. Savings are measured directly, so leave it at 0.0.
        # 1.0 on a scope turns narrowing off there (observation mode).
        bypass_rate: 0.0

        # Per-agent always-carry pins. Never demoted. Inert for tools Hermes
        # has disabled.
        always_carry: []

        # Learned overlay. apply (the default) uses learned.json. recommend is
        # the observe-only mode that never applies it.
        learned_mode: apply    # apply (default) | recommend

        # Per-channel overrides: channels.<agent>.<platform>
        channels:
          default:               # agent
            always_carry: []     # agent-wide default for every platform below
            telegram:            # platform
              learned_mode: recommend
          slack:                 # bare platform = platform-wide fallback
            bypass_rate: 1.0     # turn narrowing off on Slack entirely
```

A scope on disk is `channels.<agent>.<platform>`. Hermes calls the root profile
`default` and does not allow `:` in a settings key. A bare `channels.<platform>`
entry is the platform-wide fallback.

Every knob — sticky residency, predictor lookback, dampener tuning, the
`learned.json` shape, the full telemetry field reference — is in
[docs/CONFIGURATION.md](docs/CONFIGURATION.md). The full operator command set is
in [scripts/README.md](scripts/README.md).

```bash
python3 ~/.hermes/plugins/tool-belt/scripts/bootstrap.py       # inspect recommendations, writes nothing
python3 ~/.hermes/plugins/tool-belt/scripts/replay-shaping.py  # replay a scope's history through the shaper, writes nothing
```

`replay-shaping.py` is how you check the shaping cadence against your own
traffic before changing it: it replays one scope's telemetry chronologically,
from an empty learned state through the real shaper, and reports where the
carried set converges, what the ramp cost, how many `expand_tools` events the
settings imply, and whether anything flapped. `--window-days` and `--floor`
take comma-separated lists to sweep several settings in one run.

## Limitations

- **CLI sessions are not shaped.** CLI messages bypass the gateway, so no
  shaping and no telemetry apply to them.
- **Subagents are not shaped.** A subagent inherits the parent's tools. The
  predictor does not run again for it.
- **Cron jobs are not shaped.** Cron sessions get their tools from Hermes
  directly.
- **The Codex app-server runtime is not shaped.** When an agent runs on the
  `codex_app_server` runtime, Codex drives the turn and reaches Hermes tools
  through a fixed MCP tool list in an isolated subprocess. The plugin never
  sees or controls that toolset, so those sessions are not narrowed, save
  nothing, and are not recorded in telemetry. This is the runtime, not the
  provider: Codex used the normal way — as a provider, `api_mode:
  codex_responses` — runs through Hermes' own tool loop and is shaped like
  any other provider. Only `codex_app_server` bypasses the plugin.
- **New tools start carried.** A tool that no policy or learned entry names is
  carried on every call until the evidence demotes it. This is safe for new
  plugins, at some carry cost at first.
- **Session memory does not survive a gateway restart.** The in-process state
  (cache posture, sticky tools) is rebuilt after a restart. `learned.json` and
  the detection cache are on disk and are not affected.
- **Codex reasoning can break the cache mid-session.** When the model produces
  reasoning tokens, the message history changes and breaks the cache on the next
  call, whatever the plugin does. See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).

## Failure modes

By design, anything that goes wrong falls through to no narrowing:

- Bad YAML in policy.yaml → full tool set (no narrowing)
- policy.yaml missing → full tool set (no narrowing)
- Predictor exception → full tool set (no narrowing)
- Filter exception → unfiltered tools returned
- Learned-state load error → base policy returned, plugin keeps running
- Agent not in the registry when `expand_tools` is called → honest error JSON to
  the model

The plugin is invisible when it breaks. It is never destructive.

## Disabling

Turn it off in config:

```yaml
plugins:
  entries:
    tool-belt:
      settings:
        enabled: false
```

…or remove the directory:

```bash
rm -rf ~/.hermes/plugins/tool-belt
```

## Releases

`hermes plugins install` clones `main`, so the default install always gets the
latest commit. That is the intended path — `main` is kept installable.

To pin a known-good snapshot instead, install a specific commit:

```bash
hermes plugins install dalemugford/hermes-tool-belt --ref <commit-sha>
```

`--ref` takes a full 40-character commit SHA, not a tag name. Each release lists
its commit on the [Releases](https://github.com/dalemugford/hermes-tool-belt/releases)
page; copy that SHA to pin.

Versions use CalVer: `YYYY.M.D`, with an optional prerelease suffix (for example
`2026.5.17-beta`). The authoritative version is the one in
[`plugin.yaml`](plugin.yaml), and each release tag is the same string. See
[CHANGELOG.md](CHANGELOG.md) for what has changed.

## Files

```
~/.hermes/plugins/tool-belt/
├── plugin.yaml       # Hermes plugin manifest
├── __init__.py       # runtime hooks, cache-mode detection, carry-all + narrowing
├── predictor.py      # deterministic intent matching
├── presets.py        # policy loading and per-scope resolution
├── learned.py        # learned overlay loading and merging
├── carrying.py       # resident-partition carrying model
├── shaping.py        # evidence-driven shaping + in-process auto-shape pass
├── expand_tools.py   # dynamic recovery meta-tool
├── logger_io.py      # telemetry writers
├── analyze.py        # telemetry analyzer
├── savings.py        # savings computation
├── savings_cli.py    # savings rendering and CLI
├── cache_replay.py   # analyzer cache-cost library and diagnostic CLI
├── cli.py            # `hermes tool-belt` subcommand registration
├── yaml_required.py  # PyYAML availability guard for the operator scripts
├── tool-belt         # public command launcher (configure, savings, …)
├── policy.yaml       # shipped policy and shaping thresholds
├── AGENTS.md         # repository engineering rules
├── CHANGELOG.md      # unreleased and tagged user-facing changes
├── CONTRIBUTING.md   # contributor workflow and verification rules
├── LICENSE           # MIT license
├── docs/             # architecture, configuration, savings, privacy, known issues, releasing
├── scripts/          # configure (front door) + operator and verification commands
└── tests/            # regression and integration tests
```

## Companion docs

The code is the source of truth. The doc set is intentionally small:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the plugin works: the two postures, carry-all vs narrowing, `expand_tools`, between-session shaping, where the patches sit in the request lifecycle.
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) — every knob in `config.yaml`, `policy.yaml`, and `learned.json`, with defaults; the full telemetry field reference.
- [docs/SAVINGS.md](docs/SAVINGS.md) — how the token-savings numbers are computed and how to verify them on your own data.
- [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) — behaviors that look like bugs but are not (the Codex reasoning trace breaking prefix cache mid-session).
- [docs/PRIVACY.md](docs/PRIVACY.md) — what telemetry records and never records, where state lives, and how to disable or erase it.
- [docs/RELEASING.md](docs/RELEASING.md) — the maintainer release checklist.

## License

[MIT](LICENSE) © 2026 Dale Mugford.
