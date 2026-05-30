"""Append-only JSONL logging for predictions and tool calls.

Two log files at ``~/.hermes/state/dynamic-tools/``:

  predictions.jsonl
    One row per inbound gateway message:
      {ts, prediction_id, channel, message_hash, message_preview,
       preset, triggers_fired, always_on_count, allowed_count,
       ceiling_count, est_tokens_saved}

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


def _state_dir() -> Path:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(home) / "state" / "dynamic-tools"


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
    always_on_count: int
    # Counts of tools sent to the model
    ceiling_count: int       # what would have been sent without the plugin
    narrowed_count: int      # what actually got sent after our filter
    # Approximate token counts (chars/4 — fast, no tokenizer dep)
    ceiling_tokens: int
    narrowed_tokens: int
    triggers_suppressed: list[str] | None = None
    # Flat policy scope for future per-agent/platform learning.
    agent: str = ""
    platform: str = ""
    scope: str = ""
    # Tool-name sets for heuristic analysis.
    ceiling_tools: list[str] | None = None
    allowed_tools: list[str] | None = None
    cut_tools: list[str] | None = None
    unknown_kept_tools: list[str] | None = None
    always_on_tools: list[str] | None = None
    trigger_tools_by_group: dict[str, list[str]] | None = None
    expanded_tools: list[str] | None = None
    sticky_tools: list[str] | None = None
    sticky_categories: list[str] | None = None
    sticky_remaining_turns: dict[str, int] | None = None
    policy_source: str = "preset"
    policy_version: str = ""
    learned_mode: str = "off"
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
            "ts": self.ts,
            "prediction_id": self.prediction_id,
            "session_id": self.session_id,
            "channel": self.channel,  # legacy alias; prefer scope for new analysis
            "agent": self.agent,
            "platform": self.platform,
            "scope": self.scope,
            "message_hash": self.message_hash,
            "message_preview": self.message_preview,
            "preset": self.preset,
            "triggers_fired": self.triggers_fired,
            "triggers_suppressed": self.triggers_suppressed or [],
            "always_on_count": self.always_on_count,
            "ceiling_count": self.ceiling_count,
            "narrowed_count": self.narrowed_count,
            "ceiling_tokens": self.ceiling_tokens,
            "narrowed_tokens": self.narrowed_tokens,
            "tokens_saved": self.tokens_saved,
            "reduction_pct": round(self.reduction_pct, 2),
            "ceiling_tools": self.ceiling_tools or [],
            "allowed_tools": self.allowed_tools or [],
            "cut_tools": self.cut_tools or [],
            "unknown_kept_tools": self.unknown_kept_tools or [],
            "always_on_tools": self.always_on_tools or [],
            "trigger_tools_by_group": self.trigger_tools_by_group or {},
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
        }


def estimate_tokens(payload: Any) -> int:
    """Cheap char-based token estimate (chars / 4). Good for comparing
    before/after sizes without taking a tokenizer dependency. Anthropic's
    real tokenizer differs by ~5%, but the *delta* this estimator reports
    tracks the actual delta closely enough for the savings dashboard.
    """
    try:
        import json
        return max(0, len(json.dumps(payload, ensure_ascii=False)) // 4)
    except Exception:
        return 0


def log_prediction(record: PredictionRecord) -> None:
    """Append a prediction row. Errors are swallowed."""
    try:
        path = predictions_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception as exc:
        logger.debug("dynamic-tools: log_prediction failed: %s", exc)


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
        logger.debug("dynamic-tools: log_api_call failed: %s", exc)


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
        logger.debug("dynamic-tools: log_tool_call failed: %s", exc)
