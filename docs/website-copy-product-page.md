# dynamic-tools website copy — product page draft

## Hero

**Ship only the tools this turn actually needs.**

`dynamic-tools` is a Hermes Agent plugin that cuts per-message tool overhead by narrowing the tool definitions sent to the model on each gateway message.

**Measured: 57% reduction across 109 production sessions, with 3.37M tokens saved.**

## Short pitch

Every Hermes message normally carries the full tool catalog for that profile. `dynamic-tools` changes that. It predicts which tools are relevant for the current turn, sends only that smaller set, and keeps a built-in recovery path when the model needs something that was gated.

The result is lower tool overhead, better token efficiency, and a cleaner path to running richer agents without paying full freight on every casual message.

## Top five features

### 1. Per-message tool narrowing
`dynamic-tools` trims the tool payload on each gateway turn instead of shipping the full catalog every time. That means short conversational turns stop paying the same overhead as heavy operational tasks.

### 2. Safe on-demand recovery with `expand_tools`
If the model needs a gated category, it can call `expand_tools` and bring those tools back in on the next API call. Recovery is part of the design, so misses degrade gracefully instead of breaking the workflow.

### 3. Strict respect for user tool policy
The plugin works inside the user's existing `platform_toolsets` ceiling. It can narrow what is already allowed. It cannot silently enable tools the user disabled.

### 4. Sticky residency for active work
When a gated category is expanded, those tools can remain available for a few more turns. That keeps multi-turn work from paying the same expand round-trip again and again.

### 5. Built-in telemetry and warm-start tuning
Every prediction and actual tool call is logged for analysis. Existing Hermes sessions can also be replayed through the predictor, so a user can get day-one recommendations from real history instead of waiting for fresh telemetry to accumulate.

## Security and privacy

`dynamic-tools` was built to be aggressive about savings and conservative about risk.

- **No full message logging:** telemetry records a message hash plus an 80-character preview, not the full message body.
- **Historical replay preserves privacy:** harvest output is tested to ensure long message content and tool-call arguments do not leak into derived files.
- **User-configured ceiling stays authoritative:** the plugin never grants access beyond what the user already allowed in Hermes.
- **Session-scoped sticky state:** temporary tool admissions live in memory for the active session and clear on restart.
- **Fail-open behavior:** if parsing, prediction, config, or patching fails, the plugin falls back to no narrowing and the agent keeps working normally.
- **Minimal patch surface:** the plugin lives out of tree and applies a single runtime wrapper rather than modifying Hermes source files on disk.

## Performance

The value proposition is simple: stop paying token cost for tools the model was never going to use.

- **Measured savings:** 57% reduction across 109 production sessions.
- **Large leverage per tool removed:** the plugin docs estimate about 388 tokens per tool, per API call.
- **Cheap decision path:** tool selection uses deterministic triggers and dampeners rather than another heavyweight model pass.
- **Lower repeat expansion cost:** sticky residency keeps active categories available across follow-up turns.
- **Control-group ready:** deterministic bypass cohorts give users a way to compare narrowed traffic against full-tool baseline traffic.
- **Faster path to useful tuning:** bootstrap replay mines existing session history and surfaces ranked recommendations immediately.

## Ease of use

Install the plugin, enable it in Hermes config, restart the gateway, and it begins narrowing tool payloads on the next message. The best part of the experience is that it mostly stays out of the way.

Users can keep the shipped policy, add a few per-scope overrides, or move into learned overlays later. The warm-start replay path means they do not need to hand-curate everything from scratch. Today the honest claim is closer to *install, activate, tune lightly, done* than full autopilot, and that honesty helps the product.

## Extra highlights

### Observability first
Every narrowing decision, suppression, expansion, sticky carry, and cohort source can be inspected. The plugin is measurable by design.

### Built for messaging-shaped traffic
The sweet spot is Telegram, Slack, Discord, and similar conversational flows where the full tool universe is often overkill.

### Clear path to adaptive policy
Static policy works today. Learned overlays, analyzer recommendations, and historical replay already exist. Auto-apply remains gated behind stricter validation, which is exactly where it should be.

### Failure mode users can live with
When something goes wrong, the plugin steps aside and Hermes sends the full toolset. Savings disappear for a moment. Capability does not.

## Closing copy options

### Option A
**dynamic-tools cuts Hermes tool overhead by shipping only the tools each turn actually needs — with measured savings, built-in recovery, and zero risk of silently exceeding the user’s allowed toolset.**

### Option B
**Your agent should pay for capability when it needs capability. dynamic-tools makes that real.**

### Option C
**For conversational traffic, the full tool catalog is usually dead weight. dynamic-tools turns that waste back into usable budget.**
