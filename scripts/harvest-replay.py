#!/usr/bin/env python3
"""Harvest synthetic telemetry from existing Hermes session JSONLs.

The replay reads ``~/.hermes/sessions/*.jsonl`` (and each profile's
``profiles/*/sessions/*.jsonl``), runs every user message through the
tool-belt predictor against the session's recorded toolset, then
emits synthetic ``predictions.jsonl`` and ``tool_calls.jsonl`` tagged
``source: harvest`` / ``policy_source: harvest``. The analyzer reads
these alongside live telemetry and weights them differently.

This unlocks two things:

  1. Trigger tuning without waiting — precision/recall and dampener-candidate
     questions are answered from the session history already on disk, with no
     organic accumulation period.
  2. Public-release warm start — on first install, the plugin runs harvest
     against the user's existing sessions so day-one recommendations are
     real, not "wait a week".

What harvest CAN validate:
  · Trigger precision/recall against real intent
  · Dampener candidate mining from real false-positive messages
  · Coverage gaps (tools the model frequently called that no trigger predicts)

What harvest CANNOT validate (counterfactual):
  · Actual token savings (the model historically saw the full toolset)
  · expand_tools_used round-trip frequency (no narrowing was active)
  · bypass cohort comparison (no bypass existed)

Privacy invariants (enforced by code, not vibes):
  · Output rows contain ONLY message hash + 80-char preview — matching
    what the live writer already produces.
  · Tool call arguments are NEVER written to output.
  · A regression test in tests/test_harvest_privacy.py asserts no 64+ char
    substring of any input message ever appears in derived files.

Usage:
  python3 scripts/harvest-replay.py                       # all discovered profiles
  python3 scripts/harvest-replay.py --profile default     # root profile only
  python3 scripts/harvest-replay.py --window-days 60      # last N days only
  python3 scripts/harvest-replay.py --dry-run             # parse + count, no writes
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re as _re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
PLUGIN_DIR = HERE.parent
sys.path.insert(0, str(PLUGIN_DIR.parent))
sys.path.insert(0, str(PLUGIN_DIR))
sys.path.insert(0, str(HERE))
from _plugin_loader import load_plugin_package  # noqa: E402

load_plugin_package()

predictor = importlib.import_module("tool_belt_plugin.predictor")
presets_mod = importlib.import_module("tool_belt_plugin.presets")
logger_io = importlib.import_module("tool_belt_plugin.logger_io")
savings_mod = importlib.import_module("tool_belt_plugin.savings")
require_yaml = importlib.import_module("tool_belt_plugin.yaml_required").require_yaml


# How far ahead in JSONL order to look for tool calls that "respond" to
# a given user message. Bounded by the next user message — that's the
# natural end-of-response marker.
LOOKAHEAD_TURN_LIMIT = 50


@dataclass
class HarvestedSession:
    """The parsed shape of one session JSONL relevant to replay."""
    session_file: Path
    profile_agent: str           # "default" or a named profile, derived from path
    platform: str                # from session_meta
    ceiling_tools: list[str]     # tool names visible to the historical model
    tool_defs: dict[str, Any]    # name -> the COMPLETE session_meta.tools entry
    turns: list[dict[str, Any]]  # ordered list of {role, content, tool_calls?, ts}


def _profile_agent_from_path(session_file: Path) -> str:
    """Derive the agent identity from the session file's path.

    ``~/.hermes/sessions/*.jsonl`` belongs to the root profile (``default``).
    ``~/.hermes/profiles/<name>/sessions/*.jsonl`` belongs to <name>.
    Paths outside a named ``profiles/<name>`` directory use ``default``.
    """
    parts = session_file.resolve().parts
    if "profiles" in parts:
        i = len(parts) - 1 - parts[::-1].index("profiles")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "default"


def parse_session(session_file: Path) -> HarvestedSession | None:
    """Parse one session JSONL into the harvest-ready shape.

    Returns None if the session is unusable (no meta header, no user
    messages, malformed JSON).
    """
    try:
        lines = [
            json.loads(l)
            for l in session_file.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
    except Exception:
        return None

    meta = next((row for row in lines if row.get("role") == "session_meta"), None)
    if not meta:
        return None

    raw_tools = meta.get("tools") or []
    ceiling_names: list[str] = []
    tool_defs: dict[str, Any] = {}
    for entry in raw_tools:
        if not isinstance(entry, dict):
            continue
        # Codex/OpenAI shape: {type: function, function: {name, ...}}
        if "function" in entry and isinstance(entry["function"], dict):
            name = entry["function"].get("name")
        # Anthropic shape: {name, ...}
        else:
            name = entry.get("name")
        if isinstance(name, str) and name:
            ceiling_names.append(name)
            # Preserve the COMPLETE definition — descriptions and JSON-Schema
            # parameter blocks dominate real schema-token cost, so token counts
            # must be taken over full defs, never `[{"name": ...}]` placeholders.
            if name not in tool_defs:
                tool_defs[name] = entry

    turns = [row for row in lines if row.get("role") in ("user", "assistant")]
    if not any(t.get("role") == "user" for t in turns):
        return None

    return HarvestedSession(
        session_file=session_file,
        profile_agent=_profile_agent_from_path(session_file),
        platform=str(meta.get("platform") or "unknown"),
        ceiling_tools=ceiling_names,
        tool_defs=tool_defs,
        turns=turns,
    )


# Hermes wraps user messages with system-injected framing — quote/reply
# context, thread context summaries, speaker attribution. The mining
# flows (dampener and trigger-keyword) operate on message previews; if
# the framing eats the 80-char preview window, candidates become noise
# (matches on "prior messages in this thread", "you are assistant-a
# working on a task" — system text, not user intent). Strip the framing
# BEFORE preview computation so the preview captures actual content.
_FRAMING_BLOCK = _re.compile(
    r"^\s*\[(?:Replying to:[^\]]*|Thread context[^\]]*)\]\s*",
    flags=_re.DOTALL,
)
_SPEAKER_PREFIX = _re.compile(r"^\s*\[[a-zA-Z0-9_.-]{1,40}\]\s*")


def _strip_message_framing(text: str) -> str:
    """Strip Hermes-injected framing prefixes from a user message.

    Removes ``[Replying to: "..."]`` quote blocks, ``[Thread context — ...]``
    context summaries, and leading ``[username]`` speaker prefixes (often
    repeated multiple times when reply chains nest). Idempotent — calling
    on an unframed message is a no-op.
    """
    if not text:
        return text
    prev = None
    while text != prev:
        prev = text
        text = _FRAMING_BLOCK.sub("", text)
        text = _SPEAKER_PREFIX.sub("", text)
    return text


def _message_text(row: dict[str, Any]) -> str:
    """Extract user message text from the row's content field, with
    Hermes' system-injected framing stripped."""
    content = row.get("content")
    if isinstance(content, str):
        return _strip_message_framing(content)
    if isinstance(content, list):
        # Some assistant frameworks use a content-block list. User messages
        # in Hermes are typically plain strings, but be defensive.
        parts = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("text") or block.get("content")
                if isinstance(t, str):
                    parts.append(t)
        return _strip_message_framing("\n".join(parts))
    return ""


def _tool_calls_for_response(turns: list[dict[str, Any]], user_idx: int) -> list[str]:
    """Return every tool name the model called in response to the user
    message at index ``user_idx``, bounded by the next user message.

    "In response" = the assistant turns that fall between this user
    message and the next user message (or end of session, or
    LOOKAHEAD_TURN_LIMIT — whichever is first).
    """
    called: list[str] = []
    limit = min(len(turns), user_idx + 1 + LOOKAHEAD_TURN_LIMIT)
    for j in range(user_idx + 1, limit):
        row = turns[j]
        if row.get("role") == "user":
            break  # next user turn — stop accumulating
        if row.get("role") != "assistant":
            continue
        for call in row.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") if isinstance(call.get("function"), dict) else call
            name = fn.get("name") if isinstance(fn, dict) else None
            if isinstance(name, str) and name:
                called.append(name)
    return called


def replay_session(
    session: HarvestedSession,
    preset: Any,
    out_predictions: list[dict],
    out_tool_calls: list[dict],
    harvest_run_ts: float,
) -> None:
    """Run the predictor over each user message; emit synthetic rows."""
    scope = f"{session.profile_agent}:{session.platform}"
    # Synthetic but stable session_id derived from the file stem — keeps
    # harvest rows from one session linked together for windowed analysis.
    harvest_session_id = f"harvest:{session.session_file.stem}"
    ceiling_count = len(session.ceiling_tools)

    def _defs_for(names: list[str]) -> list[Any]:
        """Full definitions for the given names, in order. Falls back to a
        name-only stub only for a name with no recorded definition (should not
        happen for a well-formed session_meta)."""
        return [session.tool_defs.get(n, {"name": n}) for n in names]

    # Tokenize the COMPLETE tool definitions (name + description + parameter
    # schema), via the canonical estimator — not `[{"name": ...}]` placeholders.
    ceiling_tokens = savings_mod.schema_tokens(_defs_for(list(session.ceiling_tools)))

    for i, row in enumerate(session.turns):
        if row.get("role") != "user":
            continue
        message = _message_text(row)
        if not message.strip():
            continue

        prediction = predictor.predict(message, attachments=[], preset=preset)
        prediction_id = logger_io.new_prediction_id()

        # Compute the narrowed view: intersection of the resolved active set
        # with the session's ceiling. Under the 1.0 carrying model an enabled
        # built-in outside the active set is derived expand_only.
        if prediction.no_narrowing:
            allowed_names = list(session.ceiling_tools)
            expand_only_names: list[str] = []
        else:
            allowed_set = set(prediction.active_tool_names)  # type: ignore[arg-type]
            allowed_names = []
            expand_only_names = []
            for name in session.ceiling_tools:
                # Under the 1.0 carrying model, an enabled built-in outside the
                # resolved active set is expand_only — it activates only on
                # trigger activation or explicit expansion. Only names in the
                # resolved active set stay resident/active.
                if name in allowed_set:
                    allowed_names.append(name)
                else:
                    expand_only_names.append(name)

        narrowed_tokens = savings_mod.schema_tokens(_defs_for(allowed_names))

        # Ground truth: tool names the model actually called responding to this msg
        called_tools = _tool_calls_for_response(session.turns, i)

        # v2 residency split over the active set: the preset's immutable
        # always_carry baseline and adaptive carry loadout classify the
        # resident tools; ``expand_only_names`` is the expand_only stratum X.
        ac_base = set(getattr(preset, "always_carry", []) or [])
        c_base = set(getattr(preset, "carry", []) or [])
        always_carry_tools = [t for t in allowed_names if t in ac_base]
        carry_tools = [t for t in allowed_names if t in c_base and t not in ac_base]

        record = logger_io.PredictionRecord(
            ts=harvest_run_ts,
            prediction_id=prediction_id,
            session_id=harvest_session_id,
            channel=scope,
            agent=session.profile_agent,
            platform=session.platform,
            scope=scope,
            message_hash=logger_io.hash_message(message),
            message_preview=logger_io.message_preview(message),
            preset=prediction.preset_name,
            triggers_fired=prediction.triggers_fired,
            triggers_suppressed=prediction.triggers_suppressed,
            always_carry_count=len(always_carry_tools),
            carry_count=len(carry_tools),
            always_carry_tools=always_carry_tools,
            carry_tools=carry_tools,
            ceiling_count=ceiling_count,
            narrowed_count=len(allowed_names),
            ceiling_tokens=ceiling_tokens,
            narrowed_tokens=narrowed_tokens,
            ceiling_tools=list(session.ceiling_tools),
            active_tools=list(allowed_names),
            expand_only_tools=list(expand_only_names),
            policy_source="harvest",
            policy_version="harvest-emitter-1",
        )
        out_predictions.append(record.to_dict())

        # Synthetic tool_call rows for each historical call. Args/results
        # stripped — privacy invariant.
        for tool_name in called_tools:
            out_tool_calls.append({
                "schema_version": logger_io.SCHEMA_VERSION,
                "ts": harvest_run_ts,
                "prediction_id": prediction_id,
                "session_id": harvest_session_id,
                "tool_name": tool_name,
                "agent": session.profile_agent,
                "platform": session.platform,
                "scope": scope,
                "was_initially_active": tool_name in allowed_names,
                "was_expand_only": tool_name in expand_only_names,
                "policy_source": "harvest",
                # No expansion_provided_access field — counterfactual; analyzer
                # must not treat absence as False (which the read-side filter
                # already handles via `is True`).
            })


def iter_session_files(root_or_profile_dir: Path, window_days: int | None) -> Iterator[Path]:
    sessions_dir = root_or_profile_dir / "sessions"
    if not sessions_dir.is_dir():
        return
    cutoff_ts = None
    if window_days is not None:
        cutoff_ts = time.time() - (window_days * 86400)
    for path in sorted(sessions_dir.glob("*.jsonl")):
        if cutoff_ts is not None and path.stat().st_mtime < cutoff_ts:
            continue
        yield path


def _load_plugin_config(profile_home: Path) -> dict[str, Any]:
    """Load the tool-belt section from the profile's config.yaml.

    Matters because per-profile plugin settings and ``channels.*`` overrides
    change the predictor's carrying behavior. Running the harvest with the base
    policy alone can misclassify profiles with meaningful scope-specific
    configuration.

    Returns a config dict shaped exactly like what the plugin's
    register() builds from ``cfg_get("plugins.tool-belt.*")``, with
    ``enabled: True`` so resolve_preset proceeds. Returns ``{"enabled":
    True}`` (no overrides) if the config file is missing or unparseable —
    matches the live fail-safe behavior.

    A missing PyYAML is *not* one of those cases: it would silently drop
    every per-profile override and replay the wrong policy, so
    :func:`require_yaml` exits instead.
    """
    config_path = profile_home / "config.yaml"
    if not config_path.is_file():
        return {"enabled": True}
    yaml = require_yaml()
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {"enabled": True}
    plugins = (data.get("plugins") or {}) if isinstance(data.get("plugins"), dict) else {}
    dt_cfg = plugins.get("tool-belt") or {}
    if not isinstance(dt_cfg, dict):
        return {"enabled": True}
    # Force-enable for harvest so resolve_preset doesn't short-circuit on
    # users who haven't activated the plugin yet (the whole point of
    # harvest is to back-test what WOULD have happened).
    out = dict(dt_cfg)
    out["enabled"] = True
    return out


def discover_profiles(hermes_home: Path) -> list[tuple[str, Path]]:
    """Return ``[(agent_name, profile_home), ...]`` covering root + every
    profiles/* dir that has a sessions directory."""
    out: list[tuple[str, Path]] = []
    if (hermes_home / "sessions").is_dir():
        out.append(("default", hermes_home))
    profiles_dir = hermes_home / "profiles"
    if profiles_dir.is_dir():
        for child in sorted(profiles_dir.iterdir()):
            if (child.name != "default" and child.is_dir()
                    and (child / "sessions").is_dir()):
                out.append((child.name, child))
    return out


def write_outputs(
    state_dir: Path,
    predictions: list[dict],
    tool_calls: list[dict],
    *,
    dry_run: bool,
) -> None:
    harvest_dir = state_dir / "harvest"
    if dry_run:
        print(f"  [dry-run] would write {len(predictions)} predictions, "
              f"{len(tool_calls)} tool_calls to {harvest_dir}")
        return
    harvest_dir.mkdir(parents=True, exist_ok=True)
    pred_path = harvest_dir / "predictions.jsonl"
    call_path = harvest_dir / "tool_calls.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row) + "\n")
    with call_path.open("w", encoding="utf-8") as f:
        for row in tool_calls:
            f.write(json.dumps(row) + "\n")
    print(f"  wrote {pred_path} ({len(predictions)} rows)")
    print(f"  wrote {call_path} ({len(tool_calls)} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="restrict to one profile (e.g. default or assistant-a)")
    parser.add_argument("--window-days", type=int, default=None,
        help="only process sessions modified within the last N days")
    parser.add_argument("--dry-run", action="store_true",
        help="parse + count only; write nothing")
    parser.add_argument("--hermes-home", type=Path,
        default=Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes"))
    args = parser.parse_args()

    profiles = discover_profiles(args.hermes_home)
    if args.profile:
        profiles = [(n, p) for n, p in profiles if n == args.profile]
    if not profiles:
        print(f"error: no profiles found under {args.hermes_home}", file=sys.stderr)
        return 1

    grand_total_sessions = 0
    grand_total_predictions = 0
    grand_total_calls = 0
    harvest_run_ts = time.time()

    for agent_name, profile_home in profiles:
        print(f"\n=== {agent_name} ({profile_home}) ===")
        plugin_config = _load_plugin_config(profile_home)
        predictions: list[dict] = []
        tool_calls: list[dict] = []
        sessions_processed = 0
        sessions_skipped = 0

        for path in iter_session_files(profile_home, args.window_days):
            session = parse_session(path)
            if session is None:
                sessions_skipped += 1
                continue
            # Resolve preset per-scope so per-channel overrides apply.
            scope = f"{session.profile_agent}:{session.platform}"
            preset = presets_mod.resolve_preset(plugin_config, channel=scope)
            replay_session(session, preset, predictions, tool_calls, harvest_run_ts)
            sessions_processed += 1

        print(f"  sessions: {sessions_processed} processed, {sessions_skipped} skipped")
        print(f"  user messages → predictions: {len(predictions)}")
        print(f"  historical tool calls: {len(tool_calls)}")

        state_dir = profile_home / "state" / "tool-belt"
        write_outputs(state_dir, predictions, tool_calls, dry_run=args.dry_run)

        grand_total_sessions += sessions_processed
        grand_total_predictions += len(predictions)
        grand_total_calls += len(tool_calls)

    print(f"\n{'='*60}\n"
          f"  Total: {grand_total_sessions} sessions, "
          f"{grand_total_predictions} predictions, "
          f"{grand_total_calls} tool calls"
          f"\n{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
