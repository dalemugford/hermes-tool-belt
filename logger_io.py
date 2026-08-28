"""Append-only JSONL logging for predictions and tool calls.

Two log files at ``~/.hermes/state/tool-belt/``:

  predictions.jsonl
    One row per inbound gateway message (canonical schema v2):
      {schema_version, ts, prediction_id, channel, message_hash,
       message_preview, preset, triggers_fired, always_carry_count,
       carry_count, ceiling_count, narrowed_count, active_tools,
       expand_only_tools, always_carry_tools, carry_tools, …}

  tool_calls.jsonl
    One row per actual tool call during a session:
      {ts, prediction_id, session_id, tool_name}

Both are opened in append mode (``"a"``) and written one JSON line at a
time, which on POSIX is effectively atomic for short writes. Files
self-heal — a corrupt row never breaks the next write. Privacy: full
message text is never logged; only a sha1 hash and a short preview
(first 80 chars).
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Canonical telemetry schema version emitted by the producers below. Bumped to
# 2 for the Tool Belt 1.0 carrying model: predictions split residency into
# always_carry/carry, rename allowed/cut to active/expand_only, and drop the
# retired unknown_kept concept; tool-call rows rename availability/cut flags and
# add an explicit ``activation_source``. See ``normalize_prediction_row`` /
# ``normalize_tool_call_row`` for the in-memory v1→v2 canonicalization that
# downstream consumers will read through once they migrate onto it (a later
# phase; today's consumers still read raw rows).
SCHEMA_VERSION = 2

# Fallback immutable-resident baseline used by the v1 normalizer to split a
# historical resident set into the v2 always_carry / carry classes. The live
# baseline is read from policy.yaml (see ``_always_carry_baseline``); this
# mirror keeps the normalizer working even if that load fails.
_FALLBACK_ALWAYS_CARRY = frozenset(
    {"clarify", "skill_view", "skills_list", "todo", "send_message", "expand_tools"}
)


@functools.lru_cache(maxsize=1)
def _always_carry_baseline() -> frozenset[str]:
    """The permanent always_carry tool names, read from policy (cached).

    Lazily imports presets to avoid an import cycle and never raises — on any
    failure it returns the hardcoded fallback baseline so historical-row
    normalization stays deterministic.
    """
    try:
        from . import presets as _presets  # local import: avoid import cycle
        base = _presets.load_base_policy()
        names = getattr(base, "always_carry", None) or []
        result = frozenset(str(t) for t in names if isinstance(t, str) and t)
        if result:
            return result
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("tool-belt: always_carry baseline load failed: %s", exc)
    return _FALLBACK_ALWAYS_CARRY


def _state_dir() -> Path:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(home) / "state" / "tool-belt"


def predictions_path() -> Path:
    return _state_dir() / "predictions.jsonl"


def tool_calls_path() -> Path:
    return _state_dir() / "tool_calls.jsonl"


def api_calls_path() -> Path:
    return _state_dir() / "api_calls.jsonl"


def new_prediction_id() -> str:
    return uuid.uuid4().hex[:16]


def hash_message(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="replace")).hexdigest()[:16]


def message_preview(text: str, max_chars: int = 80) -> str:
    """Return a short, safe preview of the message text for log triage.

    Newlines and control chars stripped, length capped. Used to make
    log triage easier; the full text never enters the log.
    """
    if not text:
        return ""
    safe = " ".join(str(text).split())
    if len(safe) > max_chars:
        return safe[:max_chars] + "…"
    return safe


@dataclass
class PredictionRecord:
    ts: float
    prediction_id: str
    session_id: str
    channel: str
    message_hash: str
    message_preview: str
    preset: str
    triggers_fired: list[str]
    # Carrying residency counts, split by class (Tool Belt 1.0 carrying model):
    #   always_carry_count — immutable residents (partition class A ∩ E)
    #   carry_count        — adaptive residents  (partition class C)
    # These replace the v1 single ``always_on_count`` (which conflated the two).
    always_carry_count: int
    carry_count: int
    # Counts of tools sent to the model
    ceiling_count: int       # what would have been sent without the plugin
    narrowed_count: int      # what actually got sent after our filter
    # Token counts — computed by estimate_tokens(). The estimator used is
    # recorded in `tokens_estimator` per row so the savings report can
    # surface provenance honestly. See estimate_tokens() for the algo.
    ceiling_tokens: int
    narrowed_tokens: int
    triggers_suppressed: list[str] | None = None
    # Hermes internal session UUID (shaped ``YYYYMMDD_HHMMSS_<hex>``) that
    # rotates on ``/new``/``/reset``, distinct from ``session_id`` (the
    # session *key*, which is constant per chat). Telemetry-only: the shaper
    # groups predictions by this so a single long-lived chat still produces
    # the distinct-session count its demote threshold needs. Blank on older
    # rows and whenever the UUID isn't reachable — the shaper falls back to
    # ``session_id`` for those.
    hermes_session_id: str = ""
    # Flat policy scope for future per-agent/platform learning.
    agent: str = ""
    platform: str = ""
    scope: str = ""
    # Tool-name sets for heuristic analysis (Tool Belt 1.0 carrying model).
    ceiling_tools: list[str] | None = None
    # active_tools replaces the v1 ``allowed_tools`` — the per-message active
    # set A ∪ C ∪ T ∪ R (what the model actually sees this turn).
    active_tools: list[str] | None = None
    # expand_only_tools replaces the v1 ``cut_tools`` — the residency stratum X
    # (E − (A ∪ C)). These describe carrying *assignment*, not activation: a
    # tool here may still be active this turn via trigger (T) or expansion (R).
    expand_only_tools: list[str] | None = None
    mcp_passthrough_tools: list[str] | None = None
    # Residency split (replaces the v1 single ``always_on_tools`` resident list).
    always_carry_tools: list[str] | None = None   # class A ∩ E
    carry_tools: list[str] | None = None           # class C
    trigger_tools_by_group: dict[str, list[str]] | None = None
    # Flattened trigger-activated expand_only tools (T = triggered ∩ X). Kept
    # alongside ``trigger_tools_by_group`` so activation is legible without
    # re-deriving it from the per-group map.
    trigger_activated_tools: list[str] | None = None
    expanded_tools: list[str] | None = None
    sticky_tools: list[str] | None = None
    sticky_categories: list[str] | None = None
    sticky_remaining_turns: dict[str, int] | None = None
    policy_source: str = "preset"
    policy_version: str = ""
    learned_mode: str = "recommend"
    learned_scope: str = ""
    learned_changes: list[str] | None = None
    # Predictor lookback: how many prior messages were concatenated into the
    # input, and the configured ceiling. lookback_used == 0 with
    # lookback_turns_config > 0 means a session-first turn (no history yet).
    lookback_used: int = 0
    lookback_turns_config: int = 0
    # sha256 of canonical-JSON-serialized narrowed tool schemas (truncated to
    # 16 hex chars). Captures membership, order, and schema-content drift —
    # provider prefix-caches fingerprint the actual schema bytes, so order
    # and description changes invalidate the cache even when the tool name
    # set is stable. Analysis derives "did the prefix change?" by comparing
    # hashes across consecutive turns in the same session.
    tool_list_hash: str = ""
    # Provider and model identifiers when reachable from the API kwargs.
    # Cache economics differ by provider; populated best-effort, blank when
    # the kwargs shape doesn't expose them.
    provider: str = ""
    model: str = ""
    # True when this turn reused a frozen session snapshot rather than
    # re-running the predictor.
    # frozen_reuse_count is the index within the session — 1 on the
    # second dispatch, 2 on the third, etc. Both blank for cache-off
    # mode and for the first dispatch under cache-on (which freezes).
    frozen_reuse: bool = False
    frozen_reuse_count: int = 0
    # Which tokenizer produced ceiling_tokens / narrowed_tokens.
    #   "tiktoken-cl100k": real BPE tokenizer (OpenAI). Exact for
    #     GPT-family, ~5% approximate for Claude (different BPE).
    #   "chars-div-4":     len(json.dumps) // 4 fallback when tiktoken
    #     isn't installed. ~5–10% off either way; the delta still tracks
    #     the real delta because both sides use the same estimator.
    tokens_estimator: str = "chars-div-4"

    @property
    def tokens_saved(self) -> int:
        return max(0, self.ceiling_tokens - self.narrowed_tokens)

    @property
    def reduction_pct(self) -> float:
        if not self.ceiling_tokens:
            return 0.0
        return (self.tokens_saved / self.ceiling_tokens) * 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "ts": self.ts,
            "prediction_id": self.prediction_id,
            "session_id": self.session_id,
            "hermes_session_id": self.hermes_session_id,
            "channel": self.channel,  # legacy alias; prefer scope for new analysis
            "agent": self.agent,
            "platform": self.platform,
            "scope": self.scope,
            "message_hash": self.message_hash,
            "message_preview": self.message_preview,
            "preset": self.preset,
            "triggers_fired": self.triggers_fired,
            "triggers_suppressed": self.triggers_suppressed or [],
            "always_carry_count": self.always_carry_count,
            "carry_count": self.carry_count,
            "ceiling_count": self.ceiling_count,
            "narrowed_count": self.narrowed_count,
            "ceiling_tokens": self.ceiling_tokens,
            "narrowed_tokens": self.narrowed_tokens,
            "tokens_saved": self.tokens_saved,
            "reduction_pct": round(self.reduction_pct, 2),
            "ceiling_tools": self.ceiling_tools or [],
            "active_tools": self.active_tools or [],
            "expand_only_tools": self.expand_only_tools or [],
            "mcp_passthrough_tools": self.mcp_passthrough_tools or [],
            "always_carry_tools": self.always_carry_tools or [],
            "carry_tools": self.carry_tools or [],
            "trigger_tools_by_group": self.trigger_tools_by_group or {},
            "trigger_activated_tools": self.trigger_activated_tools or [],
            "expanded_tools": self.expanded_tools or [],
            "sticky_tools": self.sticky_tools or [],
            "sticky_categories": self.sticky_categories or [],
            "sticky_remaining_turns": self.sticky_remaining_turns or {},
            "policy_source": self.policy_source,
            "policy_version": self.policy_version,
            "learned_mode": self.learned_mode,
            "learned_scope": self.learned_scope,
            "learned_changes": self.learned_changes or [],
            "lookback_used": self.lookback_used,
            "lookback_turns_config": self.lookback_turns_config,
            "tool_list_hash": self.tool_list_hash,
            "provider": self.provider,
            "model": self.model,
            "frozen_reuse": self.frozen_reuse,
            "frozen_reuse_count": self.frozen_reuse_count,
            "tokens_estimator": self.tokens_estimator,
        }


@functools.lru_cache(maxsize=1)
def _get_encoder() -> tuple[Any, str]:
    """Resolve the token encoder. Returns (encoder, estimator_name).

    Prefers tiktoken's cl100k_base (OpenAI's BPE — exact for GPT, ~5%
    approximate for Claude). Falls back to a chars/4 heuristic when
    tiktoken isn't installed.

    Cached on first call: loading the cl100k vocab costs ~50ms one-time;
    every subsequent encode is 2–5ms. The cache keeps that cost outside
    the hot path.
    """
    try:
        import tiktoken  # type: ignore[import-not-found]
        return tiktoken.get_encoding("cl100k_base"), "tiktoken-cl100k"
    except Exception:
        return None, "chars-div-4"


def token_estimator_name() -> str:
    """The estimator that estimate_tokens() will use on this process."""
    _, name = _get_encoder()
    return name


def estimate_tokens(payload: Any) -> int:
    """Estimate token count of a JSON-serializable payload.

    Two-tier:
      1. tiktoken (cl100k_base) when installed — real BPE counts.
      2. chars/4 fallback — `len(json.dumps(payload)) // 4`. The
         heuristic understates JSON tool-schema density slightly but
         is consistent on both sides of a comparison, so deltas track
         the real delta even when absolute counts wobble.

    The provider's billed `input_tokens` (in api_calls.jsonl) remains
    the source of truth for actual billing; this function is for the
    `ceiling_tokens` / `narrowed_tokens` deltas that the savings
    report cites against the user's config-allowed ceiling.
    """
    try:
        import json
        text = json.dumps(payload, ensure_ascii=False)
    except Exception:
        return 0
    encoder, _ = _get_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass  # fall through to char heuristic
    return max(0, len(text) // 4)


def _string_list(value: Any) -> list[str] | None:
    """Coerce a value to ``list[str]``. ``None`` stays ``None`` (absent field).

    A lone string becomes a single-element list. Non-iterables return ``None``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return [value] if value else []
    try:
        return [str(t) for t in value if str(t)]
    except TypeError:
        return None


def _is_mcp_name(name: str) -> bool:
    """Cheap MCP/plugin tool-name predicate for normalization (no plugin import).

    Mirrors the runtime's ``_is_mcp_tool`` heuristic closely enough for the
    normalizer's ``unknown_kept_tools`` filter: Hermes MCP tools are prefixed
    ``mcp__`` (or ``mcp_`` on the Claude Code OAuth path).
    """
    return name.startswith("mcp__") or name.startswith("mcp_")


def normalize_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize any prediction telemetry row into the canonical v2 shape.

    Pure, non-mutating, and non-raising: returns a new dict, never rewrites any
    file. Intended to become the shared choke point downstream consumers read
    through so v1, v2, and mixed-version streams all present the same canonical
    fields; consumer migration onto it lands in a later phase (today
    ``shape-ceiling.py`` and friends still read raw rows). Handles malformed
    input by returning a best-effort dict.

    Canonical v2 fields (always present on the returned row): ``schema_version``,
    ``always_carry_tools``, ``carry_tools``, ``active_tools``,
    ``expand_only_tools``, ``always_carry_count``, ``carry_count``, and
    ``trigger_activated_tools``.

    v1 rows (``always_on_tools`` / ``allowed_tools`` / ``cut_tools``) are mapped
    forward: the resident set is ``always_on_tools`` plus any non-MCP
    ``unknown_kept_tools`` (kept-in tools were resident), split against the
    permanent ``always_carry`` baseline into the v2 always_carry / carry
    classes; ``cut_tools`` becomes ``expand_only_tools``.

    Residency reconstruction (``residency`` mapping + ``residency_inferred``):
    only complete membership — ceiling plus residents plus active — permits it,
    because the A/C/X partition is unknowable from a sparse row. A sparse row
    stays valid for savings/promotion analysis but sets
    ``residency_inferred: false`` and cannot drive demotion. The v1 residency
    mapping keeps class A empty (v1 had no immutable split — all residents map
    to ``carry``); a v2 row's mapping reflects its authoritative A/C/X.

    Legacy v1 field names (``always_on_tools`` / ``allowed_tools`` /
    ``cut_tools`` / ``always_on_count``) are retained as computed aliases only
    while the not-yet-migrated analyzer/shaper still read them; they are not the
    canonical surface and will be dropped once those consumers move to v2.
    """
    try:
        out = dict(row or {})
    except (TypeError, ValueError):
        return {"schema_version": SCHEMA_VERSION, "residency_inferred": False,
                "residency": None}

    is_v2 = out.get("schema_version") == SCHEMA_VERSION or "always_carry_tools" in out

    ceiling = _string_list(out.get("ceiling_tools"))

    if is_v2:
        always_carry_tools = _string_list(out.get("always_carry_tools")) or []
        carry_tools = _string_list(out.get("carry_tools")) or []
        residents = [*always_carry_tools, *carry_tools]
        active = _string_list(out.get("active_tools"))
        if active is None:
            active = _string_list(out.get("allowed_tools"))
        expand_only = _string_list(out.get("expand_only_tools"))
        if expand_only is None:
            expand_only = _string_list(out.get("cut_tools"))
    else:
        v1_residents = _string_list(out.get("always_on_tools")) or []
        # Kept-in unknown tools were resident too; MCP/plugin pass-through is
        # NOT residency evidence and is filtered out (guarantee: MCP can never
        # become shaping evidence).
        unknown_kept = [t for t in (_string_list(out.get("unknown_kept_tools")) or [])
                        if not _is_mcp_name(t)]
        baseline = _always_carry_baseline()
        residents = list(dict.fromkeys([*v1_residents, *unknown_kept]))
        always_carry_tools = [t for t in residents if t in baseline]
        carry_tools = [t for t in residents if t not in baseline]
        active = _string_list(out.get("allowed_tools"))
        if active is None:
            active = _string_list(out.get("active_tools"))
        expand_only = _string_list(out.get("cut_tools"))
        if expand_only is None:
            expand_only = _string_list(out.get("expand_only_tools"))

    # ── Canonical v2 fields (always set explicitly). ──────────────────────
    out["schema_version"] = SCHEMA_VERSION
    out["always_carry_tools"] = always_carry_tools
    out["carry_tools"] = carry_tools
    out["active_tools"] = active or []
    out["expand_only_tools"] = expand_only or []
    out["always_carry_count"] = out.get("always_carry_count", len(always_carry_tools))
    out["carry_count"] = out.get("carry_count", len(carry_tools))
    out.setdefault("trigger_activated_tools",
                   _string_list(out.get("trigger_activated_tools")) or [])
    out.pop("unknown_kept_tools", None)

    # ── Transitional legacy aliases for not-yet-migrated consumers. ───────
    out["always_on_tools"] = residents
    out["allowed_tools"] = active or []
    out["cut_tools"] = expand_only or []

    # ── Residency reconstruction (complete membership only). ──────────────
    residents_known = bool(out.get("always_on_tools") is not None
                           and (always_carry_tools or carry_tools or
                                out.get("always_on_tools")))
    complete = bool(ceiling) and residents_known and (active is not None)
    if complete:
        ceiling_set = [str(t) for t in ceiling]
        ceiling_members = set(ceiling_set)
        resident_set = set(residents) & ceiling_members
        if is_v2:
            a_set = set(always_carry_tools) & ceiling_members
            c_set = (set(carry_tools) & ceiling_members) - a_set
        else:
            a_set = set()
            c_set = resident_set
        x_set = [t for t in ceiling_set if t not in (a_set | c_set)]
        out["residency_inferred"] = True
        out["residency"] = {
            "always_carry": sorted(a_set),
            "carry": sorted(c_set),
            "expand_only": sorted(x_set),
        }
    else:
        out["residency_inferred"] = False
        out.setdefault("residency", None)
    return out


def normalize_tool_call_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize any tool-call telemetry row into the canonical v2 shape.

    Pure, non-mutating, non-raising — the tool-call counterpart to
    :func:`normalize_prediction_row`. v1 rows use ``was_initially_available`` /
    ``was_cut`` / ``was_expanded`` / ``expand_tools_used``; v2 renames these to
    ``was_initially_active`` / ``was_expand_only`` / ``activated_by_expansion``
    / ``expansion_provided_access`` and adds an explicit ``activation_source``.

    Both directions of the availability/cut/expansion flags are retained as
    aliases so a not-yet-migrated reader sees consistent values regardless of
    the on-disk version. ``activation_source`` is only set when present (v2) or
    confidently derivable; a v1 row without it keeps it blank rather than
    guessing a residency class the old schema never recorded.
    """
    try:
        out = dict(row or {})
    except (TypeError, ValueError):
        return {"schema_version": SCHEMA_VERSION}

    def _pick(*names, default=None):
        for n in names:
            if n in out and out[n] is not None:
                return out[n]
        return default

    was_initially_active = _pick("was_initially_active", "was_initially_available")
    was_expand_only = _pick("was_expand_only", "was_cut")
    activated_by_expansion = _pick("activated_by_expansion", "was_expanded")
    expansion_provided_access = _pick("expansion_provided_access", "expand_tools_used")
    activation_source = _pick("activation_source", default="")

    out["schema_version"] = SCHEMA_VERSION
    # Canonical v2 flags (only set when the row carried the information — a
    # counterfactual harvest row deliberately omits expansion flags and must
    # not gain a spurious ``False``).
    if was_initially_active is not None:
        out["was_initially_active"] = bool(was_initially_active)
        out["was_initially_available"] = bool(was_initially_active)  # legacy alias
    if was_expand_only is not None:
        out["was_expand_only"] = bool(was_expand_only)
        out["was_cut"] = bool(was_expand_only)  # legacy alias
    if activated_by_expansion is not None:
        out["activated_by_expansion"] = bool(activated_by_expansion)
        out["was_expanded"] = bool(activated_by_expansion)  # legacy alias
    if expansion_provided_access is not None:
        out["expansion_provided_access"] = bool(expansion_provided_access)
        out["expand_tools_used"] = bool(expansion_provided_access)  # legacy alias
    out["activation_source"] = str(activation_source or "")
    return out


def log_prediction(record: PredictionRecord) -> None:
    """Append a prediction row. Errors are swallowed."""
    try:
        path = predictions_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception as exc:
        logger.debug("tool-belt: log_prediction failed: %s", exc)


def log_api_call(row: dict[str, Any]) -> None:
    """Append one row per outbound API call. Errors are swallowed.

    Schema is kept loose (free-form dict) because the post_api_request hook
    payload evolves across providers; pinning a dataclass forces lossy
    coercion. Callers are expected to include at minimum: ts, prediction_id,
    session_id, api_call_idx, cache_read_tokens, cache_write_tokens,
    input_tokens, output_tokens, tool_list_hash. Analysis joins to
    predictions.jsonl on prediction_id.
    """
    try:
        path = api_calls_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception as exc:
        logger.debug("tool-belt: log_api_call failed: %s", exc)


def log_tool_call(
    prediction_id: str,
    session_id: str,
    tool_name: str,
    args: dict | None = None,
    result: Any = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append a tool-call row. Errors are swallowed."""
    try:
        path = tool_calls_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "ts": time.time(),
            "prediction_id": prediction_id,
            "session_id": session_id,
            "tool_name": tool_name,
        }
        if extra:
            row.update(extra)
        if args is not None:
            row["args"] = args
        if result is not None:
            # result may arrive as a raw JSON string from the tool handler
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except Exception:
                    pass
            row["result"] = result
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception as exc:
        logger.debug("tool-belt: log_tool_call failed: %s", exc)
