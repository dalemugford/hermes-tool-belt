#!/usr/bin/env python3
"""Deterministic telemetry analyzer for the tool-belt plugin.

Reads append-only JSONL telemetry and emits a human-readable summary plus a
markdown report. The analyzer is intentionally conservative: it tolerates older
rows with missing fields, treats recommendations as suggestions, and keeps all
thresholds as adjustable defaults rather than permanent policy.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

DEFAULT_PER_TOOL_TOKENS = 388
DEFAULT_MIN_EXPANSIONS = 2
DEFAULT_PROMOTE_EXPAND_RATE = 0.50
DEFAULT_PROMOTE_USE_RATE = 0.80
DEFAULT_UNUSED_CARRY_TURNS = 10
DEFAULT_TRIGGER_MIN_FIRES = 3
# Estimated total token cost of one expand_tools round-trip: the model emits a
# tool-call (~150 output tokens), the gateway hands the result back (~250),
# and the next API call resends system + messages + widened tool list. The
# largest term is "extra full API request body," dominated by message history
# and the widened tool schemas. 1500 is a conservative single-call estimate;
# tune via --expand-round-trip-tokens once we have a measured baseline.
DEFAULT_EXPAND_ROUND_TRIP_TOKENS = 1500
# Dampener suggestion defaults. These are conservative — tuning down min
# support or precision quickly produces noise; tuning up shrinks the
# candidate set to obviously-good patterns only.
DEFAULT_DAMPENER_MIN_SUPPORT = 3
DEFAULT_DAMPENER_MIN_PRECISION = 0.8
DEFAULT_DAMPENER_MIN_N = 2
DEFAULT_DAMPENER_MAX_N = 4
DEFAULT_DAMPENER_MAX_CANDIDATES = 10

BASE_PROTECTED_ALWAYS_ON = {
    "memory",
    "session_search",
    "clarify",
    "skill_view",
    "skills_list",
    "todo",
    "send_message",
    "expand_tools",
    # Hermes' native deferred-loading bridge tools. Tool Search manages
    # its own activation; Tool Belt should never gate or demote them.
    "tool_search",
    "tool_describe",
    "tool_call",
}


@dataclass
class ScopeStats:
    scope: str
    predictions: int = 0
    sessions: set[str] = field(default_factory=set)
    total_tokens_saved: int = 0
    total_ceiling_tokens: int = 0
    total_narrowed_tokens: int = 0
    expansion_events: int = 0
    expansions_by_category: Counter[str] = field(default_factory=Counter)
    expansion_ids_by_category: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # Tools resolved by each category (from expand_tools result payload).
    # Used by expand-cost accounting to estimate carry cost if we promote.
    tools_per_category: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    used_expansion_rows: int = 0
    used_expansion_event_ids_by_category: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    expanded_uses_by_category: Counter[str] = field(default_factory=Counter)
    expanded_uses_by_tool: Counter[str] = field(default_factory=Counter)
    turns_until_used: list[int] = field(default_factory=list)
    tool_calls: Counter[str] = field(default_factory=Counter)
    always_on_carry_turns: Counter[str] = field(default_factory=Counter)
    cut_turns: Counter[str] = field(default_factory=Counter)
    trigger_fires: Counter[str] = field(default_factory=Counter)
    trigger_hits: Counter[str] = field(default_factory=Counter)
    trigger_needs: Counter[str] = field(default_factory=Counter)
    trigger_misses: Counter[str] = field(default_factory=Counter)
    trigger_false_positives: Counter[str] = field(default_factory=Counter)
    # Per-trigger lists of message previews. fp = trigger fired but no
    # matching tool got called (false positive); tp = trigger fired and a
    # matching tool did get called (true positive). Used by the dampener
    # suggester to mine candidate exclude_keywords. Capped per trigger to
    # avoid unbounded growth on big telemetry archives.
    trigger_fp_previews: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    trigger_tp_previews: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    learned_modes: Counter[str] = field(default_factory=Counter)
    # Per-policy_source cohort counters. Used by A/B baseline comparison.
    # ``policy_sources`` counts prediction rows by source ("preset" / "learned"
    # / "bypass" / "manual"); the cohort dicts roll up tool calls and
    # expansions tied to each cohort's prediction rows.
    policy_sources: Counter[str] = field(default_factory=Counter)
    cohort_sessions: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    cohort_tool_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cohort_expansions: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # Harvest-specific signal: per-tool count of historical calls that
    # WOULD have been cut by current narrowing (was_cut == True on the
    # tool_call row). Populated only when harvest-source rows are present.
    # The corresponding `harvest_predictions` is the denominator for
    # carry-cost math when proposing always-on promotion.
    harvest_was_cut: Counter[str] = field(default_factory=Counter)
    harvest_predictions: int = 0
    harvest_tool_calls_total: int = 0
    # Per-tool sample of message previews that led to a was_cut call.
    # Used by the trigger-keyword suggester to mine candidate keywords
    # for adding cut tools into existing or new trigger groups.
    harvest_cut_previews: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    # All harvest-prediction previews for this scope — the noise denominator
    # for keyword mining precision. Sized cap protects unbounded growth on
    # big harvests; sampling preserves precision math accuracy.
    harvest_all_previews: list[str] = field(default_factory=list)
    # Tool-list stability metrics. A mutation = tool_list_hash differs from
    # the previous turn in the same session (turn > 1). Mutations bust
    # provider prefix-cache on tool schemas, billing the conversation
    # history prefix at full input rate. Stable turns keep the cache warm.
    # Turn 1 is excluded from both counts — first turns always warm a cold
    # cache regardless of narrowing.
    mutation_turns: int = 0
    stable_turns: int = 0
    first_turns: int = 0
    hash_missing_turns: int = 0  # rows where tool_list_hash is unavailable
    sessions_with_mutation: set[str] = field(default_factory=set)
    # Per-provider/model rollups for stability. Empty key when the adapter
    # didn't surface the value.
    mutation_by_provider: Counter[str] = field(default_factory=Counter)
    mutation_by_model: Counter[str] = field(default_factory=Counter)
    stable_by_provider: Counter[str] = field(default_factory=Counter)
    stable_by_model: Counter[str] = field(default_factory=Counter)


def state_dir_from_env() -> Path:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(home) / "state" / "tool-belt"


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def string_list(value: Any) -> list[str]:
    return [str(item).strip() for item in as_list(value) if str(item).strip()]


def normalize_scope(row: dict[str, Any]) -> str:
    scope = str(row.get("scope") or "").strip().lower()
    if scope:
        return scope
    agent = str(row.get("agent") or "").strip().lower()
    platform = str(row.get("platform") or "").strip().lower()
    if agent and platform:
        return f"{agent}:{platform}"
    channel = str(row.get("channel") or "").strip().lower()
    if channel:
        return channel
    if platform:
        return platform
    return "unknown"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                row.setdefault("_line_no", line_no)
                row.setdefault("_source_file", str(path))
                yield row


def telemetry_paths(state_dir: Path, filename: str, include_archives: bool) -> list[Path]:
    paths = [state_dir / filename]
    if include_archives:
        archive_dir = state_dir / "archive"
        if archive_dir.exists():
            paths.extend(sorted(archive_dir.glob(filename)))
            stem = Path(filename).stem
            paths.extend(sorted(archive_dir.glob(f"{stem}-*.jsonl")))
            paths.extend(sorted(archive_dir.glob(f"{stem}_*.jsonl")))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve() if path.exists() else path
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(iter_jsonl(path))
    return rows


def category_from_expand_row(row: dict[str, Any]) -> str:
    event = row.get("expand_event") if isinstance(row.get("expand_event"), dict) else {}
    args = row.get("args") if isinstance(row.get("args"), dict) else {}
    result = row.get("result") if isinstance(row.get("result"), dict) else {}
    for source in (event, args, result, row):
        category = str(source.get("category") or source.get("toolset") or "").strip()
        if category:
            return category
    return "unknown"


def _resolved_tools_from_expand_row(row: dict[str, Any]) -> list[str]:
    """Return the canonical resolved-tools list for an expand_tools row.

    The plugin records this redundantly under ``expand_event.resolved_tools``
    on the source side and ``result.resolved_tools`` on the response side.
    Older telemetry may only have one of them.
    """
    for key in ("expand_event", "result"):
        source = row.get(key) if isinstance(row.get(key), dict) else {}
        tools = source.get("resolved_tools") if isinstance(source, dict) else None
        if isinstance(tools, list) and tools:
            return [str(t).strip() for t in tools if str(t).strip()]
    return []


def expansion_event_key(row: dict[str, Any], category: str) -> str:
    prediction_id = str(row.get("prediction_id") or "").strip()
    line_no = row.get("_line_no", "")
    source_file = row.get("_source_file", "")
    return f"{prediction_id}:{category}" if prediction_id else f"{source_file}:line:{line_no}:{category}"


def trigger_tools(row: dict[str, Any]) -> dict[str, set[str]]:
    raw = row.get("trigger_tools_by_group")
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, set[str]] = {}
    for group, tools in raw.items():
        group_name = str(group).strip()
        tool_names = set(string_list(tools))
        if group_name and tool_names:
            parsed[group_name] = tool_names
    return parsed


def collect_stats(predictions: list[dict[str, Any]], tool_calls: list[dict[str, Any]]) -> dict[str, ScopeStats]:
    stats: dict[str, ScopeStats] = {}

    def get(scope: str) -> ScopeStats:
        if scope not in stats:
            stats[scope] = ScopeStats(scope=scope)
        return stats[scope]

    prediction_scope: dict[str, str] = {}
    prediction_triggers: dict[str, dict[str, set[str]]] = {}
    prediction_tools_called: dict[str, set[str]] = defaultdict(set)
    prediction_policy_source: dict[str, str] = {}
    prediction_preview: dict[str, str] = {}
    tool_groups_by_scope: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for row in predictions:
        scope = normalize_scope(row)
        stat = get(scope)
        stat.predictions += 1
        # Prefer hermes_session_id (rotates on /new) for accurate per-session
        # grouping; fall back to session_id (the stable chat key) for older
        # rows written before the UUID was captured.
        session_id = str(row.get("hermes_session_id") or row.get("session_id") or "").strip()
        if session_id:
            stat.sessions.add(session_id)
        policy_source = str(row.get("policy_source") or "").strip().lower() or "preset"
        stat.policy_sources[policy_source] += 1
        if session_id:
            stat.cohort_sessions[policy_source].add(session_id)
        prediction_id = str(row.get("prediction_id") or "").strip()
        if prediction_id:
            prediction_scope[prediction_id] = scope
            prediction_policy_source[prediction_id] = policy_source
            preview = str(row.get("message_preview") or "").strip()
            if preview:
                prediction_preview[prediction_id] = preview
            groups = trigger_tools(row)
            prediction_triggers[prediction_id] = groups
            for group, tools in groups.items():
                tool_groups_by_scope[scope][group].update(tools)
        stat.total_tokens_saved += safe_int(row.get("tokens_saved"))
        stat.total_ceiling_tokens += safe_int(row.get("ceiling_tokens"))
        stat.total_narrowed_tokens += safe_int(row.get("narrowed_tokens"))
        for tool in string_list(row.get("always_on_tools")):
            stat.always_on_carry_turns[tool] += 1
        for tool in string_list(row.get("cut_tools")):
            stat.cut_turns[tool] += 1
        for trigger in string_list(row.get("triggers_fired")):
            stat.trigger_fires[trigger] += 1
        learned_mode = str(row.get("learned_mode") or "").strip().lower()
        if learned_mode:
            stat.learned_modes[learned_mode] += 1
        if policy_source == "harvest":
            stat.harvest_predictions += 1
            preview = str(row.get("message_preview") or "").strip()
            if preview and len(stat.harvest_all_previews) < 2000:
                stat.harvest_all_previews.append(preview)

    for row in tool_calls:
        prediction_id = str(row.get("prediction_id") or "").strip()
        scope = normalize_scope(row)
        if scope == "unknown" and prediction_id in prediction_scope:
            scope = prediction_scope[prediction_id]
        stat = get(scope)
        tool = str(row.get("tool_name") or "").strip()
        cohort = prediction_policy_source.get(prediction_id, "")
        if tool:
            stat.tool_calls[tool] += 1
            if cohort:
                stat.cohort_tool_calls[cohort] += 1
            if prediction_id:
                prediction_tools_called[prediction_id].add(tool)
            # Harvest-source signal: count would-have-been cuts per tool
            # and capture the prediction's message preview so the trigger-
            # keyword suggester can mine candidates.
            if cohort == "harvest" and row.get("was_cut") is True:
                stat.harvest_was_cut[tool] += 1
                preview = prediction_preview.get(prediction_id, "")
                if preview and len(stat.harvest_cut_previews[tool]) < 200:
                    stat.harvest_cut_previews[tool].append(preview)
            if cohort == "harvest":
                stat.harvest_tool_calls_total += 1

        if tool == "expand_tools":
            category = category_from_expand_row(row)
            event_key = expansion_event_key(row, category)
            stat.expansion_events += 1
            stat.expansions_by_category[category] += 1
            stat.expansion_ids_by_category[category].add(event_key)
            if cohort:
                stat.cohort_expansions[cohort] += 1
            for resolved in _resolved_tools_from_expand_row(row):
                stat.tools_per_category[category].add(resolved)
            continue

        # Filter out the "tool was already in the initial allowed set"
        # case at read time so historical telemetry (written before the
        # writer-side fix) doesn't inflate expansion-success metrics.
        # The corrected writer no longer credits these, but old rows
        # still carry the inflated flag. ``was_initially_available`` is
        # present on every row written since the multi-profile telemetry
        # commit, so it's safe to consult here.
        if row.get("expand_tools_used") is True and not row.get("was_initially_available", False):
            category = str(row.get("expand_category") or "unknown").strip() or "unknown"
            stat.used_expansion_rows += 1
            expansion_prediction_id = str(row.get("expansion_prediction_id") or "").strip()
            event_prediction_id = expansion_prediction_id or (prediction_id if row.get("after_expand_tools") else "")
            if event_prediction_id:
                stat.used_expansion_event_ids_by_category[category].add(f"{event_prediction_id}:{category}")
            stat.expanded_uses_by_category[category] += 1
            expanded_tool = str(row.get("expanded_tool") or tool or "unknown").strip() or "unknown"
            stat.expanded_uses_by_tool[expanded_tool] += 1
            stat.turns_until_used.append(max(0, safe_int(row.get("turns_until_used"))))

    # Build a session-bounded lookahead window so that sticky-carried
    # tool calls in subsequent turns count as TPs for the trigger that
    # fired on the earlier turn. Without this, a trigger that correctly
    # predicted intent but where the model called the tool 1-2 turns
    # later (sticky residency carried the expansion) is miscounted as
    # an FP — inflating dampener-candidate FP counts by ~30% on file_write
    # and ~50% on shell in observed telemetry.
    #
    # Rows without a session identifier fall back to same-prediction behavior
    # so scope-ordered timeline lookups cannot leak across sessions.
    WINDOW = 3  # matches the default sticky_ttl_turns
    session_ordered_preds: dict[str, list[str]] = defaultdict(list)
    pred_session: dict[str, str] = {}
    for row in predictions:
        pid = str(row.get("prediction_id") or "").strip()
        sid = str(row.get("hermes_session_id") or row.get("session_id") or "").strip()
        if pid and sid:
            session_ordered_preds[sid].append(pid)
            pred_session[pid] = sid
    pred_index_in_session: dict[str, int] = {
        pid: i for plist in session_ordered_preds.values() for i, pid in enumerate(plist)
    }

    def _window_called(prediction_id: str) -> set[str]:
        """Tools called in this prediction + next WINDOW predictions
        within the same session. Falls back to same-prediction when
        session_id is unavailable."""
        same = prediction_tools_called.get(prediction_id, set())
        sid = pred_session.get(prediction_id, "")
        if not sid:
            return same
        plist = session_ordered_preds.get(sid, [])
        i = pred_index_in_session.get(prediction_id, -1)
        if i < 0:
            return same
        ahead: set[str] = set()
        for q in plist[i + 1 : i + 1 + WINDOW]:
            ahead |= prediction_tools_called.get(q, set())
        return same | ahead

    for prediction_id, groups in prediction_triggers.items():
        scope = prediction_scope.get(prediction_id, "unknown")
        stat = get(scope)
        called = _window_called(prediction_id)
        fired_groups = set(groups)
        preview = prediction_preview.get(prediction_id, "")
        for group, tools in groups.items():
            if called.intersection(tools):
                stat.trigger_hits[group] += 1
                if preview and len(stat.trigger_tp_previews[group]) < 500:
                    stat.trigger_tp_previews[group].append(preview)
            else:
                stat.trigger_false_positives[group] += 1
                if preview and len(stat.trigger_fp_previews[group]) < 500:
                    stat.trigger_fp_previews[group].append(preview)
        for group, tools in tool_groups_by_scope.get(scope, {}).items():
            if called.intersection(tools):
                stat.trigger_needs[group] += 1
                if group not in fired_groups:
                    stat.trigger_misses[group] += 1

    # Tool-list stability pass. Iterate session-ordered predictions and
    # compare each turn's tool_list_hash to the previous turn in the same
    # session. A change = the wire-level tool schema bytes diverged from
    # last turn = provider prefix-cache on tool schemas + everything after
    # them (including the conversation history) is invalidated.
    #
    # Turn 1 is recorded separately — it's always a cold start and would
    # bias any "did narrowing bust the cache?" comparison. Rows missing
    # tool_list_hash are counted so measurement coverage remains visible.
    pred_payload: dict[str, dict[str, Any]] = {}
    for row in predictions:
        pid = str(row.get("prediction_id") or "").strip()
        if pid:
            pred_payload[pid] = row

    for sid, plist in session_ordered_preds.items():
        prev_hash: str | None = None
        for i, pid in enumerate(plist):
            row = pred_payload.get(pid)
            if row is None:
                continue
            scope = prediction_scope.get(pid, "unknown")
            stat = get(scope)
            current_hash = str(row.get("tool_list_hash") or "").strip()
            provider = str(row.get("provider") or "").strip()
            model = str(row.get("model") or "").strip()
            if i == 0:
                stat.first_turns += 1
                prev_hash = current_hash if current_hash else None
                continue
            if not current_hash:
                stat.hash_missing_turns += 1
                # Keep the last available hash as the baseline so a gap in
                # hash coverage cannot manufacture a mutation.
                continue
            if prev_hash is None:
                # First scorable turn in this session — no baseline to
                # compare against (turn 1 was either missing a hash or
                # entirely absent). Treat as first_turns to be honest.
                stat.first_turns += 1
                prev_hash = current_hash
                continue
            if current_hash != prev_hash:
                stat.mutation_turns += 1
                stat.sessions_with_mutation.add(sid)
                stat.mutation_by_provider[provider] += 1
                stat.mutation_by_model[model] += 1
            else:
                stat.stable_turns += 1
                stat.stable_by_provider[provider] += 1
                stat.stable_by_model[model] += 1
            prev_hash = current_hash

    return dict(sorted(stats.items()))


def pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def median_int(values: list[int]) -> int:
    if not values:
        return 0
    return int(statistics.median(values))


def avg(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def carrying_cost(carried_turns: int, per_tool_tokens: int) -> int:
    return carried_turns * per_tool_tokens


import re

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize_preview(preview: str) -> list[str]:
    """Lowercase + word-tokenize. Punctuation drops; ellipsis truncations and
    other noise become non-tokens. Keep contractions intact so dampeners can
    target phrasing like ``don't``.
    """
    if not preview:
        return []
    return _TOKEN_RE.findall(preview.lower())


def _ngrams(tokens: list[str], n: int) -> list[str]:
    if n <= 0 or len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _load_preset_excludes(
    plugin_dir: Path,
) -> tuple[dict[str, list[re.Pattern[str]]], str]:
    """Load compiled exclude_keywords per trigger group from policy.yaml.

    Returns ``(excludes_by_trigger, status)`` where ``status`` is one of:

      ``"ok"``           — policy.yaml parsed and excludes compiled
      ``"no_yaml"``      — PyYAML not importable in this Python runtime
      ``"no_policy"``    — policy.yaml file missing under ``plugin_dir``
      ``"parse_error"``  — YAML present but failed to parse

    The status lets the caller surface a clear degraded-mode message
    instead of silently reporting "existing_exclude_keyword_count: 0",
    which previously made an empty result look like a real policy state
    on a runtime that simply lacked PyYAML.
    """
    path = plugin_dir / "policy.yaml"
    if not path.exists():
        return {}, "no_policy"
    try:
        import yaml  # type: ignore[import-untyped]
    except Exception:
        return {}, "no_yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}, "parse_error"
    out: dict[str, list[re.Pattern[str]]] = {}
    for entry in data.get("triggers", []) or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        excludes = entry.get("exclude_keywords") or []
        patterns: list[re.Pattern[str]] = []
        for raw in excludes:
            if not isinstance(raw, str):
                continue
            try:
                patterns.append(re.compile(raw, flags=re.IGNORECASE))
            except re.error:
                continue
        if name and patterns:
            out[name] = patterns
    return out, "ok"


_EXCLUDES_STATUS_MESSAGE = {
    "no_yaml": (
        "PyYAML is not installed in this Python runtime — policy.yaml "
        "exclude_keywords are NOT being applied to dampener candidates. "
        "Counts shown as 'existing_exclude_keyword_count: 0' reflect the "
        "missing loader, not an empty policy. Install pyyaml or run the "
        "analyzer under the Hermes venv to restore filtering."
    ),
    "no_policy": (
        "policy.yaml not found alongside analyze.py — dampener candidates "
        "will not be deduplicated against existing exclude_keywords."
    ),
    "parse_error": (
        "policy.yaml failed to parse — dampener candidates will not be "
        "deduplicated against existing exclude_keywords this run."
    ),
}


def _load_preset_always_on(plugin_dir: Path) -> tuple[set[str], str]:
    """Load the preset always_on tool names from policy.yaml.

    Returns ``(tools, status)`` using the same status enum as
    :func:`_load_preset_excludes`. When PyYAML is unavailable, fall back
    to a tiny line-oriented parser for the top-level ``always_on`` list
    so the analyzer still honors explicit resident tools.
    """
    path = plugin_dir / "policy.yaml"
    if not path.exists():
        return set(), "no_policy"

    def fallback_parse() -> set[str]:
        tools: set[str] = set()
        in_always_on = False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not in_always_on:
                if stripped == "always_on:":
                    in_always_on = True
                continue
            if raw_line and not raw_line.startswith((" ", "	")):
                break
            if stripped.startswith("- "):
                item = stripped[2:].split("#", 1)[0].strip()
                if item:
                    tools.add(item)
        return tools

    try:
        import yaml  # type: ignore[import-untyped]
    except Exception:
        tools = fallback_parse()
        return tools, "ok" if tools else "no_yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        tools = fallback_parse()
        return tools, "ok" if tools else "parse_error"
    tools = {
        str(tool).strip()
        for tool in (data.get("always_on") or [])
        if str(tool).strip()
    }
    return tools, "ok"


def effective_protected_always_on(plugin_dir: Path | None = None) -> set[str]:
    """Return the analyzer's effective do-not-suggest always-on set.

    ``policy.yaml`` is the source of truth. The hard-coded base set is a
    safe fallback when YAML is unavailable so we still avoid noisy
    suggestions for core meta tools.
    """
    protected = set(BASE_PROTECTED_ALWAYS_ON)
    if plugin_dir is None:
        plugin_dir = Path(__file__).resolve().parent
    preset_tools, _status = _load_preset_always_on(plugin_dir)
    protected.update(preset_tools)
    return protected


def _load_preset_triggers(
    plugin_dir: Path,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Load the full trigger structure from policy.yaml.

    Returns ``({group_name: {tools: [...], keyword_patterns: [...]}}, status)``
    with the same status enum as :func:`_load_preset_excludes`. Used by
    the trigger-keyword suggester to:
      · Decide which group a candidate keyword should join (the one whose
        ``tools`` already lists the cut tool).
      · Avoid re-suggesting keywords that an existing trigger pattern
        already matches.
    """
    path = plugin_dir / "policy.yaml"
    if not path.exists():
        return {}, "no_policy"
    try:
        import yaml  # type: ignore[import-untyped]
    except Exception:
        return {}, "no_yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}, "parse_error"
    out: dict[str, dict[str, Any]] = {}
    for entry in data.get("triggers", []) or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        tools = [str(t) for t in (entry.get("tools") or []) if isinstance(t, str)]
        keywords = entry.get("keywords") or []
        patterns: list[re.Pattern[str]] = []
        for raw in keywords:
            if not isinstance(raw, str):
                continue
            try:
                patterns.append(re.compile(raw, flags=re.IGNORECASE))
            except re.error:
                continue
        out[name] = {"tools": tools, "keyword_patterns": patterns}
    return out, "ok"


def _trigger_group_for_tool(
    tool: str, triggers: dict[str, dict[str, Any]],
) -> str | None:
    """Find the existing trigger group whose ``tools`` list includes
    ``tool``. Returns the group name, or None when no group claims it."""
    for name, spec in triggers.items():
        if tool in spec.get("tools", []):
            return name
    return None


def _suggested_new_trigger_name(tool: str) -> str:
    """Heuristic group name when no existing group claims a cut tool.

    Common pattern: ``browser_navigate`` / ``browser_click`` → use the
    underscore prefix (``browser``). Falls back to the bare tool name.
    """
    if "_" in tool:
        return tool.split("_", 1)[0]
    return tool


def _is_already_covered(ngram: str, existing_patterns: list[re.Pattern[str]]) -> bool:
    """True if any existing exclude pattern already matches this n-gram."""
    if not existing_patterns:
        return False
    for pat in existing_patterns:
        if pat.search(ngram):
            return True
    return False


def _ngram_to_regex(ngram: str) -> str:
    """Convert a lowercase n-gram to a case-insensitive word-boundary regex.

    The plugin compiles exclude_keywords with re.IGNORECASE, so the
    suggested regex doesn't need an inline flag. Word boundaries prevent
    "fix" inside "prefix" from matching.
    """
    return rf"\b{re.escape(ngram)}\b"


def _suggest_dampeners_for_trigger(
    fp_previews: list[str],
    tp_previews: list[str],
    existing_patterns: list[re.Pattern[str]],
    *,
    min_n: int,
    max_n: int,
    min_support: int,
    min_precision: float,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Mine candidate exclude-keyword n-grams from FP vs TP message previews.

    Algorithm:
      1. Tokenize each preview to words.
      2. Count per-n-gram occurrences across FP and TP sets.
      3. Filter: support (FP count) ≥ min_support; precision
         (fp / (fp + tp)) ≥ min_precision.
      4. Drop n-grams already covered by an existing exclude pattern.
      5. Substring-dominance dedupe: prefer the longest covering n-gram
         when a shorter one is strictly contained in it with similar
         precision (keeps "should we save" over "should we" only when the
         longer form is meaningfully more precise; otherwise keeps the
         shorter, more general one).
      6. Rank by (precision desc, support desc, length desc) and trim.
    """
    if not fp_previews:
        return []

    fp_counts: Counter[str] = Counter()
    tp_counts: Counter[str] = Counter()
    for preview in fp_previews:
        tokens = _tokenize_preview(preview)
        seen: set[str] = set()
        for n in range(min_n, max_n + 1):
            for gram in _ngrams(tokens, n):
                if gram in seen:
                    continue
                seen.add(gram)
                fp_counts[gram] += 1
    for preview in tp_previews:
        tokens = _tokenize_preview(preview)
        seen = set()
        for n in range(min_n, max_n + 1):
            for gram in _ngrams(tokens, n):
                if gram in seen:
                    continue
                seen.add(gram)
                tp_counts[gram] += 1

    candidates: list[dict[str, Any]] = []
    for ngram, fp_count in fp_counts.items():
        if fp_count < min_support:
            continue
        tp_count = tp_counts.get(ngram, 0)
        precision = fp_count / (fp_count + tp_count)
        if precision < min_precision:
            continue
        if _is_already_covered(ngram, existing_patterns):
            continue
        candidates.append({
            "ngram": ngram,
            "fp_count": fp_count,
            "tp_count": tp_count,
            "precision": round(precision, 4),
            "length": len(ngram.split()),
        })

    # Substring-dominance dedupe: if a shorter n-gram is strictly contained
    # in a longer one AND the longer one has at least 80% of the shorter's
    # support, prefer the longer (more specific) form. Otherwise keep the
    # shorter (more general) — it catches more cases.
    candidates.sort(key=lambda c: (-c["length"], -c["fp_count"]))
    kept: list[dict[str, Any]] = []
    dropped: set[str] = set()
    for cand in candidates:
        gram = cand["ngram"]
        if gram in dropped:
            continue
        for shorter in candidates:
            sg = shorter["ngram"]
            if sg == gram or sg in dropped or shorter["length"] >= cand["length"]:
                continue
            if f" {sg} " in f" {gram} " or gram.startswith(sg + " ") or gram.endswith(" " + sg):
                # Longer subsumes shorter. Keep longer iff it covers >=80%
                # of the shorter's FP support; otherwise drop longer and
                # keep the more general shorter form.
                if cand["fp_count"] >= 0.8 * shorter["fp_count"]:
                    dropped.add(sg)
                else:
                    dropped.add(gram)
                    break
        if gram not in dropped:
            kept.append(cand)

    kept = [c for c in kept if c["ngram"] not in dropped]
    kept.sort(key=lambda c: (-c["precision"], -c["fp_count"], -c["length"]))

    out: list[dict[str, Any]] = []
    for cand in kept[:max_candidates]:
        ngram = cand["ngram"]
        # Pull up to 3 sample FP previews containing this n-gram for human eyes.
        samples = [p for p in fp_previews if ngram in p.lower()][:3]
        out.append({
            "pattern": ngram,
            "suggested_regex": _ngram_to_regex(ngram),
            "fp_count": cand["fp_count"],
            "tp_count": cand["tp_count"],
            "precision": cand["precision"],
            "sample_previews": samples,
        })
    return out


def _stability_payload(stat: "ScopeStats") -> dict[str, Any]:
    """Per-scope tool-list stability summary.

    Mutation = the wire-level tool schemas changed from one turn to the
    next in the same session. Provider prefix-caches fingerprint the
    actual schema bytes, so a mutation invalidates the cache for the
    tool block AND everything after it (notably the conversation history,
    which accumulates and is the dominant cost on long sessions).

    A high mutation rate in a cache-on environment means narrowing's
    schema savings are partially or fully eaten by re-billing the
    history at full input rate. This payload reports the raw frequency;
    the cache-cost analysis uses per-call ``cache_read_tokens`` captured
    in ``api_calls.jsonl`` and computed by ``scripts/cache-freeze-replay.py``,
    surfaced in the Cache-Aware Savings section of this report.

    Counts exclude turn 1 (always a cold cache by definition) and rows
    missing tool_list_hash. hash_missing_turns is surfaced as a coverage
    signal — high values mean the stability figures reflect a narrower
    slice than the full telemetry window.
    """
    comparable = stat.mutation_turns + stat.stable_turns
    return {
        "first_turns": stat.first_turns,
        "comparable_turns": comparable,
        "mutation_turns": stat.mutation_turns,
        "stable_turns": stat.stable_turns,
        "hash_missing_turns": stat.hash_missing_turns,
        "mutation_rate": round(stat.mutation_turns / comparable, 4) if comparable else 0.0,
        "sessions_with_mutation": len(stat.sessions_with_mutation),
        "sessions_observed": len(stat.sessions),
        "sessions_with_mutation_rate": (
            round(len(stat.sessions_with_mutation) / len(stat.sessions), 4)
            if stat.sessions else 0.0
        ),
        "mutation_by_provider": dict(stat.mutation_by_provider),
        "stable_by_provider": dict(stat.stable_by_provider),
        "mutation_by_model": dict(stat.mutation_by_model),
        "stable_by_model": dict(stat.stable_by_model),
    }


def _cohort_payload(stat: "ScopeStats") -> dict[str, Any]:
    """Per-policy_source cohort summary for A/B baseline comparison.

    Reports prediction count, session count, tool-call count, and
    expand_tools count per cohort. Useful sanity check on bypass behavior:
    a bypass cohort should have notably more tool calls per prediction (no
    gating) but no expand_tools events (no narrowing to recover from).
    """
    out: dict[str, Any] = {}
    sources = set(stat.policy_sources) | set(stat.cohort_sessions) | set(stat.cohort_tool_calls)
    for source in sorted(sources):
        predictions = stat.policy_sources.get(source, 0)
        tool_calls = stat.cohort_tool_calls.get(source, 0)
        expansions = stat.cohort_expansions.get(source, 0)
        out[source] = {
            "predictions": predictions,
            "sessions": len(stat.cohort_sessions.get(source, set())),
            "tool_calls": tool_calls,
            "tool_calls_per_prediction": round(tool_calls / predictions, 4) if predictions else 0.0,
            "expand_tools_events": expansions,
            "expand_rate": round(expansions / predictions, 4) if predictions else 0.0,
        }
    return out


def category_net_value(
    expansions: int,
    predictions: int,
    tools_in_category: int,
    expand_round_trip_tokens: int,
    per_tool_tokens: int,
) -> dict[str, int]:
    """Estimate the net token value of promoting a category to always-on.

    Saved by promotion: each expansion event avoids one round-trip
    (``expansions × expand_round_trip_tokens``). Costs of promotion: every
    prediction now carries the category's tools in the request schema
    (``predictions × tools_in_category × per_tool_tokens``). Net value is
    saved minus cost; positive means promotion is likely beneficial under
    these defaults, negative means keep gated.
    """
    saved = max(0, expansions) * max(0, expand_round_trip_tokens)
    cost = max(0, predictions) * max(0, tools_in_category) * max(0, per_tool_tokens)
    return {
        "saved_round_trip_tokens": saved,
        "added_carry_tokens": cost,
        "net_value_tokens": saved - cost,
    }


# Stop words to filter out as candidate keywords. An n-gram composed
# entirely of these is too generic to be a useful trigger keyword — even
# at precision=1.0 it would just be a coincidence of the noise sample.
_KEYWORD_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "for", "in",
    "on", "at", "by", "with", "from", "as", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "this",
    "that", "these", "those", "it", "its", "what", "when", "where",
    "who", "why", "how", "yes", "no", "ok", "okay", "sure", "thanks",
    "thank", "you", "your", "yours", "i", "we", "us", "our", "me", "my",
    "him", "her", "his", "hers", "them", "they", "their", "so", "just",
})


def _has_content_word(ngram: str) -> bool:
    """True if at least one token is not in the stop-word set.
    Filters out pure-stopword n-grams like ``for a`` / ``and the``.
    Single short tokens (<3 chars) are also treated as stop-ish."""
    for tok in ngram.split():
        t = tok.strip().lower()
        if len(t) >= 3 and t not in _KEYWORD_STOPWORDS:
            return True
    return False


def _suggest_keywords_for_cut_tool(
    cut_previews: list[str],
    noise_previews: list[str],
    existing_patterns: list[re.Pattern[str]],
    *,
    min_n: int,
    max_n: int,
    min_support: int,
    min_precision: float,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Mine n-grams that recur in cut-tool messages but rarely elsewhere.

    Mirrors :func:`_suggest_dampeners_for_trigger` with inverted semantics:
    dampener mining wants n-grams that ONLY appear in FP messages
    (suppress over-triggering); keyword mining wants n-grams that
    appear in cut-tool messages but NOT in unrelated noise (surface
    under-triggering).

    Returns candidate keyword dicts ranked by precision then frequency.
    """
    if not cut_previews or len(cut_previews) < min_support:
        return []

    raw_candidates: list[dict[str, Any]] = []
    cut_lower = [p.lower() for p in cut_previews]
    noise_lower = [p.lower() for p in noise_previews]

    seen: set[str] = set()
    for preview in cut_lower:
        tokens = _tokenize_preview(preview)
        for n in range(min_n, max_n + 1):
            for ngram in _ngrams(tokens, n):
                if ngram in seen:
                    continue
                seen.add(ngram)
                if not _has_content_word(ngram):
                    continue
                cut_count = sum(1 for p in cut_lower if ngram in p)
                if cut_count < min_support:
                    continue
                noise_count = sum(1 for p in noise_lower if ngram in p)
                precision = cut_count / max(1, cut_count + noise_count)
                if precision < min_precision:
                    continue
                raw_candidates.append({
                    "ngram": ngram,
                    "cut_count": cut_count,
                    "noise_count": noise_count,
                    "precision": precision,
                    "length": n,
                })

    # Dedupe substrings: if "open the page" survives and "open the" also
    # survives with similar precision, prefer the longer (more specific)
    # one. Identical-stats shorter candidates get dropped.
    raw_candidates.sort(key=lambda c: (-c["precision"], -c["cut_count"], -c["length"]))
    dropped: set[str] = set()
    kept: list[dict[str, Any]] = []
    for cand in raw_candidates:
        if cand["ngram"] in dropped:
            continue
        if _is_already_covered(cand["ngram"], existing_patterns):
            continue
        kept.append(cand)
        for other in raw_candidates:
            if (other["ngram"] != cand["ngram"]
                and other["ngram"] in cand["ngram"]
                and other["cut_count"] <= cand["cut_count"]):
                dropped.add(other["ngram"])

    out: list[dict[str, Any]] = []
    for cand in kept[:max_candidates]:
        ngram = cand["ngram"]
        samples = [p for p in cut_previews if ngram in p.lower()][:3]
        out.append({
            "pattern": ngram,
            "suggested_regex": _ngram_to_regex(ngram),
            "cut_count": cand["cut_count"],
            "noise_count": cand["noise_count"],
            "precision": round(cand["precision"], 4),
            "sample_previews": samples,
        })
    return out


def trigger_keyword_candidates(
    stats: dict[str, ScopeStats], args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Per-(scope, tool) keyword candidates for adding to triggers.

    The symmetric inversion of :func:`dampener_candidates`. Where the
    dampener flow mines FP messages to suppress over-triggering, this
    flow mines was_cut messages to surface under-triggering — keywords
    that would have fired a trigger to cover a tool the historical
    model called but current narrowing would have cut.

    Auto-detects: returns empty when ``--suggest-trigger-keywords`` is
    off OR no scope has harvest data with sufficient cut volume per tool.

    For each candidate cut tool, decides whether the suggestion should
    augment an existing trigger group (the one whose ``tools`` already
    lists this tool) or propose a new group.
    """
    if not getattr(args, "suggest_trigger_keywords", False):
        return []
    plugin_dir = Path(__file__).resolve().parent
    triggers, status = _load_preset_triggers(plugin_dir)
    if status != "ok":
        # Reuse the dampener status messages — same failure modes.
        warning = _EXCLUDES_STATUS_MESSAGE.get(
            status, f"preset triggers unavailable: {status}"
        )
        print(f"warning: tool-belt analyzer (triggers): {warning}",
              file=sys.stderr)

    # Noise corpus: ALL harvest prediction previews per scope (capped
    # at 2000 during collect_stats). This is the correct denominator —
    # without it, common-everywhere phrases like "for a" appear in
    # cut_previews but score precision=1.0 because the narrow noise
    # pool (other-tools' cut_previews only) doesn't happen to contain
    # them. Using the full prediction corpus puts genuine common-noise
    # phrases out of the running.
    noise_corpus: dict[str, list[str]] = {
        scope: list(stat.harvest_all_previews) for scope, stat in stats.items()
    }

    rows: list[dict[str, Any]] = []
    min_cuts = max(1, args.harvest_min_cuts)
    for scope, stat in sorted(stats.items()):
        if stat.harvest_predictions == 0:
            continue
        for tool, cut_count in stat.harvest_was_cut.most_common():
            if cut_count < min_cuts:
                continue
            cut_previews = stat.harvest_cut_previews.get(tool, [])
            if not cut_previews:
                continue
            # Noise = harvest previews where THIS tool was NOT cut.
            other_previews = [
                p for p in noise_corpus.get(scope, [])
                if p not in cut_previews
            ]
            # Determine target trigger
            target_group = _trigger_group_for_tool(tool, triggers)
            if target_group:
                existing_patterns = triggers.get(target_group, {}).get("keyword_patterns", [])
                action = "add_keywords_to_trigger"
            else:
                target_group = _suggested_new_trigger_name(tool)
                existing_patterns = []
                action = "create_new_trigger"

            candidates = _suggest_keywords_for_cut_tool(
                cut_previews=cut_previews,
                noise_previews=other_previews,
                existing_patterns=existing_patterns,
                min_n=args.dampener_min_n,
                max_n=args.dampener_max_n,
                min_support=args.dampener_min_support,
                min_precision=args.dampener_min_precision,
                max_candidates=args.dampener_max_candidates,
            )
            if not candidates:
                continue
            rows.append({
                "scope": scope,
                "tool": tool,
                "cut_count": cut_count,
                "target_trigger": target_group,
                "action": action,
                "existing_keyword_pattern_count": len(existing_patterns),
                "candidates": candidates,
                "preset_triggers_status": status,
            })
    return rows


def dampener_candidates(stats: dict[str, ScopeStats], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Per-(scope, trigger) dampener candidate sets.

    Returns one row per scope/trigger combination that has any surviving
    candidates after filtering. Empty list when ``--suggest-dampeners``
    isn't set or no scope has enough FP volume to mine.
    """
    if not getattr(args, "suggest_dampeners", False):
        return []
    plugin_dir = Path(__file__).resolve().parent
    existing_by_trigger, excludes_status = _load_preset_excludes(plugin_dir)
    if excludes_status != "ok":
        warning = _EXCLUDES_STATUS_MESSAGE.get(
            excludes_status, f"preset excludes unavailable: {excludes_status}"
        )
        print(f"warning: tool-belt analyzer: {warning}", file=sys.stderr)
    rows: list[dict[str, Any]] = []
    for scope, stat in stats.items():
        for trigger, fp_previews in stat.trigger_fp_previews.items():
            if len(fp_previews) < args.dampener_min_support:
                continue
            tp_previews = stat.trigger_tp_previews.get(trigger, [])
            existing = existing_by_trigger.get(trigger, [])
            candidates = _suggest_dampeners_for_trigger(
                fp_previews=fp_previews,
                tp_previews=tp_previews,
                existing_patterns=existing,
                min_n=args.dampener_min_n,
                max_n=args.dampener_max_n,
                min_support=args.dampener_min_support,
                min_precision=args.dampener_min_precision,
                max_candidates=args.dampener_max_candidates,
            )
            if not candidates:
                continue
            rows.append({
                "scope": scope,
                "trigger": trigger,
                "fp_previews_mined": len(fp_previews),
                "tp_previews_mined": len(tp_previews),
                "existing_exclude_keyword_count": len(existing),
                "preset_excludes_status": excludes_status,
                "candidates": candidates,
            })
    return rows


def harvest_recommendation_rows(
    stats: dict[str, ScopeStats],
    args: argparse.Namespace,
    protected_always_on: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Per-tool promotion candidates derived from harvest telemetry.

    Harvest rows carry ``was_cut: True`` whenever the historical model
    called a tool that current narrowing would have gated. Each such
    call represents a counterfactual ``expand_tools`` round-trip. The
    direct auto-apply signal is:

        net_savings = cuts × expand_round_trip_tokens
                    − predictions × per_tool_tokens

    Positive net favors always-on promotion; negative favors keeping
    the tool trigger-gated (or improving trigger recall to cover its
    use cases). Auto-detected — emits nothing when no harvest rows are
    present, so the function is safe to call unconditionally.

    Note this is a per-tool recommendation alongside the existing
    per-category recommendations from ``recommendation_rows`` — the
    two views answer different questions. Per-tool ("promote terminal
    to always_on?") is what harvest data can answer directly. Per-
    category ("is the browser expand round-trip net-positive?") needs
    live expand_tools events with success-rate signal, which harvest
    can't produce.
    """
    rows: list[dict[str, Any]] = []
    protected = set(protected_always_on or BASE_PROTECTED_ALWAYS_ON)
    min_cuts = max(1, args.harvest_min_cuts)
    for scope, stat in sorted(stats.items()):
        if stat.harvest_predictions == 0:
            continue
        denom = max(stat.harvest_predictions, 1)
        for tool, cuts in stat.harvest_was_cut.most_common():
            if tool in protected:
                continue
            if cuts < min_cuts:
                continue
            round_trip_cost = cuts * args.expand_round_trip_tokens
            carry_cost = denom * args.per_tool_tokens
            net = round_trip_cost - carry_cost
            cut_rate = cuts / denom
            if net > 0:
                action = "promote_always_on"
                status = "review"
                confidence = "medium" if cuts >= min_cuts * 3 else "low"
            elif cut_rate >= 0.10:
                # Frequently cut but carry cost outweighs round-trip.
                # The right move is usually trigger broadening, not
                # promotion. Flag it for human review.
                action = "broaden_trigger_recall"
                status = "review"
                confidence = "medium"
            else:
                action = "keep_gated"
                status = "watch"
                confidence = "medium"
            rows.append({
                "scope": scope,
                "kind": "harvest_tool_promotion",
                "item": tool,
                "action": action,
                "status": status,
                "confidence": confidence,
                "metrics": {
                    "harvest_predictions": stat.harvest_predictions,
                    "harvest_was_cut": cuts,
                    "cut_rate": round(cut_rate, 4),
                    "round_trip_cost_tokens": round_trip_cost,
                    "carry_cost_tokens": carry_cost,
                    "net_savings_tokens": net,
                },
                "defaults": {
                    "expand_round_trip_tokens": args.expand_round_trip_tokens,
                    "per_tool_tokens": args.per_tool_tokens,
                    "harvest_min_cuts": min_cuts,
                },
                "proposed_learned_patch": {
                    "scopes": {
                        scope: {
                            "always_on": [tool] if action == "promote_always_on" else [],
                        }
                    }
                },
                "reason": (
                    f"historical sessions called {tool} {cuts} time(s) "
                    f"({cut_rate:.1%} of predictions) where current policy "
                    f"would have cut it; round-trip cost {round_trip_cost:,} tok "
                    f"vs carry cost {carry_cost:,} tok → net {net:+,} tok"
                ),
            })
    return rows


def recommendation_rows(
    stats: dict[str, ScopeStats],
    args: argparse.Namespace,
    protected_always_on: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    protected = set(protected_always_on or BASE_PROTECTED_ALWAYS_ON)
    for scope, stat in stats.items():
        denominator = max(stat.predictions, 1)
        for category, count in stat.expansions_by_category.most_common():
            used_rows = stat.expanded_uses_by_category.get(category, 0)
            used_events = len(stat.used_expansion_event_ids_by_category.get(category, set()))
            expand_rate = count / denominator
            success_rate = used_events / count if count else 0.0
            tools_in_category = len(stat.tools_per_category.get(category, set()))
            cost_value = category_net_value(
                expansions=count,
                predictions=stat.predictions,
                tools_in_category=tools_in_category,
                expand_round_trip_tokens=args.expand_round_trip_tokens,
                per_tool_tokens=args.per_tool_tokens,
            )
            if count < args.min_expansions:
                action = "watch"
                confidence = "low"
                status = "insufficient_data"
            elif expand_rate >= args.promote_expand_rate and success_rate >= args.promote_use_rate:
                # Heuristic guard: even if expand_rate/use_rate clear the
                # threshold, a negative net-value estimate means carry cost
                # outweighs round-trip savings under the current defaults.
                if cost_value["net_value_tokens"] < 0:
                    action = "review_net_negative"
                    confidence = "low"
                else:
                    action = "promote_candidate"
                    confidence = "medium"
                status = "review"
            elif success_rate >= args.promote_use_rate:
                action = "watch_high_precision"
                confidence = "medium"
                status = "watch"
            else:
                action = "keep_gated"
                confidence = "medium"
                status = "review"
            rows.append({
                "scope": scope,
                "kind": "expanded_category",
                "item": category,
                "action": action,
                "status": status,
                "confidence": confidence,
                "metrics": {
                    "predictions": stat.predictions,
                    "expansions": count,
                    "expansions_with_any_use": used_events,
                    "expanded_tool_use_rows": used_rows,
                    "expand_rate": round(expand_rate, 4),
                    "success_rate": round(success_rate, 4),
                    "rows_per_expansion": round(used_rows / count, 4) if count else 0.0,
                    "turns_until_used_avg": round(avg(stat.turns_until_used), 2),
                    "turns_until_used_median": median_int(stat.turns_until_used),
                    "tools_in_category": tools_in_category,
                    **cost_value,
                },
                "defaults": {
                    "min_expansions": args.min_expansions,
                    "promote_expand_rate": args.promote_expand_rate,
                    "promote_use_rate": args.promote_use_rate,
                    "expand_round_trip_tokens": args.expand_round_trip_tokens,
                    "per_tool_tokens": args.per_tool_tokens,
                },
                "proposed_learned_patch": {
                    "scopes": {
                        scope: {
                            "always_on": [category] if action == "promote_candidate" else [],
                        }
                    }
                },
                "reason": (
                    f"expanded {count} time(s), had downstream use on {used_events} expansion event(s), "
                    f"produced {used_rows} expanded-tool call row(s); "
                    f"est. round-trip savings={cost_value['saved_round_trip_tokens']:,} tok, "
                    f"est. carry cost={cost_value['added_carry_tokens']:,} tok, "
                    f"net={cost_value['net_value_tokens']:+,} tok under "
                    f"per_tool={args.per_tool_tokens}, round_trip={args.expand_round_trip_tokens}"
                ),
            })

        for tool, carried in stat.always_on_carry_turns.most_common():
            if tool in protected:
                continue
            calls = stat.tool_calls.get(tool, 0)
            if carried >= args.unused_carry_turns and calls == 0:
                rows.append({
                    "scope": scope,
                    "kind": "always_on_tool",
                    "item": tool,
                    "action": "demote_candidate",
                    "status": "review",
                    "confidence": "low",
                    "metrics": {
                        "carried_turns": carried,
                        "calls": calls,
                        "estimated_carrying_cost_tokens": carrying_cost(carried, args.per_tool_tokens),
                    },
                    "defaults": {
                        "unused_carry_turns": args.unused_carry_turns,
                        "per_tool_tokens": args.per_tool_tokens,
                    },
                    "proposed_learned_patch": {
                        "scopes": {
                            scope: {
                                "always_off": [tool],
                            }
                        }
                    },
                    "reason": (
                        f"carried as always-on for {carried} prediction turn(s) with no logged calls; "
                        f"default review floor is {args.unused_carry_turns} turns"
                    ),
                })

        for trigger, fires in stat.trigger_fires.most_common():
            if fires < args.trigger_min_fires:
                continue
            hits = stat.trigger_hits.get(trigger, 0)
            needs = stat.trigger_needs.get(trigger, 0)
            false_positives = stat.trigger_false_positives.get(trigger, 0)
            misses = stat.trigger_misses.get(trigger, 0)
            precision = hits / fires if fires else 0.0
            recall = hits / needs if needs else 0.0
            rows.append({
                "scope": scope,
                "kind": "trigger_group",
                "item": trigger,
                "action": "review_trigger_precision" if precision < 0.5 else "keep_trigger",
                "status": "review",
                "confidence": "low",
                "metrics": {
                    "fires": fires,
                    "hits_same_prediction": hits,
                    "needs_same_scope_tool_map": needs,
                    "misses_same_scope_tool_map": misses,
                    "false_positive_same_prediction": false_positives,
                    "precision_same_prediction": round(precision, 4),
                    "recall_same_scope_tool_map": round(recall, 4),
                },
                "defaults": {
                    "trigger_min_fires": args.trigger_min_fires,
                },
                "proposed_learned_patch": {"scopes": {scope: {"trigger_adjustments": {}}}},
                "reason": (
                    f"trigger fired {fires} time(s); same-prediction precision estimate {precision:.2f}; "
                    f"same-scope recall estimate {recall:.2f}. "
                    "These are conservative proxies because older rows may lack trigger_tools_by_group."
                ),
            })
    return rows


def summary_payload(
    stats: dict[str, ScopeStats],
    recs: list[dict[str, Any]],
    args: argparse.Namespace,
    dampeners: list[dict[str, Any]] | None = None,
    trigger_keywords: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    total_predictions = sum(stat.predictions for stat in stats.values())
    total_tool_calls = sum(sum(stat.tool_calls.values()) for stat in stats.values())
    total_expansions = sum(stat.expansion_events for stat in stats.values())
    total_used_rows = sum(stat.used_expansion_rows for stat in stats.values())
    total_used_events = sum(
        sum(len(event_ids) for event_ids in stat.used_expansion_event_ids_by_category.values())
        for stat in stats.values()
    )
    total_saved = sum(stat.total_tokens_saved for stat in stats.values())
    total_expand_cost = total_expansions * args.expand_round_trip_tokens
    total_first_turns = sum(stat.first_turns for stat in stats.values())
    total_mutation_turns = sum(stat.mutation_turns for stat in stats.values())
    total_stable_turns = sum(stat.stable_turns for stat in stats.values())
    total_hash_missing = sum(stat.hash_missing_turns for stat in stats.values())
    total_comparable = total_mutation_turns + total_stable_turns
    total_sessions_with_mutation = sum(
        len(stat.sessions_with_mutation) for stat in stats.values()
    )
    total_sessions = sum(len(stat.sessions) for stat in stats.values())
    return {
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "state_dir": str(args.state_dir),
        "include_archives": args.include_archives,
        "totals": {
            "scopes": len(stats),
            "prediction_rows": total_predictions,
            "tool_call_rows": total_tool_calls,
            "expand_tools_events": total_expansions,
            "expanded_tool_use_rows": total_used_rows,
            "expanded_tool_use_rows_per_expansion": round(total_used_rows / total_expansions, 4) if total_expansions else 0.0,
            "expansions_with_any_use": total_used_events,
            "expansion_success_rate": round(total_used_events / total_expansions, 4) if total_expansions else 0.0,
            "logged_first_call_tokens_saved": total_saved,
            "estimated_expand_round_trip_cost_tokens": total_expand_cost,
            "estimated_net_savings_tokens": total_saved - total_expand_cost,
            "recommendation_candidates": len(recs),
            "first_turns": total_first_turns,
            "comparable_turns": total_comparable,
            "mutation_turns": total_mutation_turns,
            "stable_turns": total_stable_turns,
            "hash_missing_turns": total_hash_missing,
            "mutation_rate": round(total_mutation_turns / total_comparable, 4) if total_comparable else 0.0,
            "sessions_with_mutation": total_sessions_with_mutation,
            "sessions_observed": total_sessions,
            "sessions_with_mutation_rate": round(total_sessions_with_mutation / total_sessions, 4) if total_sessions else 0.0,
        },
        "thresholds": threshold_payload(args),
        "scopes": {scope: scope_payload(stat, args) for scope, stat in stats.items()},
        "recommendations": recs,
        "dampener_candidates": list(dampeners or []),
        "trigger_keyword_candidates": list(trigger_keywords or []),
    }


def threshold_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "per_tool_tokens": args.per_tool_tokens,
        "min_expansions": args.min_expansions,
        "promote_expand_rate": args.promote_expand_rate,
        "promote_use_rate": args.promote_use_rate,
        "unused_carry_turns": args.unused_carry_turns,
        "trigger_min_fires": args.trigger_min_fires,
        "expand_round_trip_tokens": args.expand_round_trip_tokens,
    }


def scope_payload(stat: ScopeStats, args: argparse.Namespace) -> dict[str, Any]:
    protected = effective_protected_always_on(Path(__file__).resolve().parent)
    used_events = sum(len(event_ids) for event_ids in stat.used_expansion_event_ids_by_category.values())
    expand_round_trip_total = stat.expansion_events * args.expand_round_trip_tokens
    net_savings = stat.total_tokens_saved - expand_round_trip_total
    return {
        "predictions": stat.predictions,
        "sessions": len(stat.sessions),
        "logged_first_call_tokens_saved": stat.total_tokens_saved,
        "estimated_expand_round_trip_cost_tokens": expand_round_trip_total,
        "estimated_net_savings_tokens": net_savings,
        "ceiling_tokens": stat.total_ceiling_tokens,
        "narrowed_tokens": stat.total_narrowed_tokens,
        "expand_tools_events": stat.expansion_events,
        "expanded_tool_use_rows": stat.used_expansion_rows,
        "expanded_tool_use_rows_per_expansion": round(stat.used_expansion_rows / stat.expansion_events, 4) if stat.expansion_events else 0.0,
        "expansions_with_any_use": used_events,
        "expansion_success_rate": round(used_events / stat.expansion_events, 4) if stat.expansion_events else 0.0,
        "turns_until_used": {
            "avg": round(avg(stat.turns_until_used), 2),
            "median": median_int(stat.turns_until_used),
            "samples": len(stat.turns_until_used),
        },
        "expansions_by_category": dict(stat.expansions_by_category),
        "expanded_uses_by_category": dict(stat.expanded_uses_by_category),
        "expansions_with_any_use_by_category": {
            category: len(event_ids)
            for category, event_ids in stat.used_expansion_event_ids_by_category.items()
        },
        "top_tool_calls": dict(stat.tool_calls.most_common(20)),
        "estimated_always_on_carrying_cost_tokens": {
            tool: carrying_cost(turns, args.per_tool_tokens)
            for tool, turns in stat.always_on_carry_turns.items()
            if tool not in protected
        },
        "cohorts": _cohort_payload(stat),
        "tool_list_stability": _stability_payload(stat),
        "trigger_precision_recall": {
            trigger: {
                "fires": fires,
                "hits_same_prediction": stat.trigger_hits.get(trigger, 0),
                "needs_same_scope_tool_map": stat.trigger_needs.get(trigger, 0),
                "misses_same_scope_tool_map": stat.trigger_misses.get(trigger, 0),
                "precision_same_prediction": round(stat.trigger_hits.get(trigger, 0) / fires, 4) if fires else 0.0,
                "recall_same_scope_tool_map": round(stat.trigger_hits.get(trigger, 0) / stat.trigger_needs.get(trigger, 0), 4) if stat.trigger_needs.get(trigger, 0) else 0.0,
            }
            for trigger, fires in stat.trigger_fires.items()
        },
        "learned_modes": dict(stat.learned_modes),
    }


_CACHE_FREEZE_MOD = None


def _load_cache_freeze_module():
    """Load scripts/cache-freeze-replay.py via importlib (hyphenated filename)."""
    global _CACHE_FREEZE_MOD
    if _CACHE_FREEZE_MOD is not None:
        return _CACHE_FREEZE_MOD
    script = Path(__file__).resolve().parent / "scripts" / "cache-freeze-replay.py"
    if not script.exists():
        return None
    spec = importlib.util.spec_from_file_location("cache_freeze_replay", script)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CACHE_FREEZE_MOD = mod
    return mod


def compute_cache_cost(
    predictions: list[dict[str, Any]],
    api_calls: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    scopes: list[str],
) -> dict[str, dict[str, Any]]:
    """Run the cache-freeze replay per scope and return a scope→result map.

    Each value has:
      - freeze: freeze_simulation() output
      - counterfactual: matched_counterfactual() output
    Returns an empty dict if the module or api_calls are unavailable.
    """
    mod = _load_cache_freeze_module()
    if mod is None or not api_calls:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for scope in scopes:
        freeze = mod.freeze_simulation(predictions, api_calls, tool_calls=tool_calls, scope_filter=scope)
        cf = mod.matched_counterfactual(api_calls, scope_filter=scope)
        result[scope] = {"freeze": freeze, "counterfactual": cf}
    return result


def _cache_cost_total_usd(cache_cost: dict[str, dict[str, Any]]) -> float:
    """Sum est_usd_lost_upper_bound across all scopes and models."""
    total = 0.0
    for scope_data in cache_cost.values():
        for model_data in scope_data.get("counterfactual", {}).get("per_model", {}).values():
            total += model_data.get("est_usd_lost_upper_bound", 0.0)
    return round(total, 4)


def _cache_cost_total_tokens_lost(cache_cost: dict[str, dict[str, Any]]) -> int:
    """Sum cache_read_lost_upper_bound across all scopes and models."""
    total = 0
    for scope_data in cache_cost.values():
        for model_data in scope_data.get("counterfactual", {}).get("per_model", {}).values():
            total += model_data.get("cache_read_lost_upper_bound", 0)
    return total


def format_summary(
    stats: dict[str, ScopeStats],
    recs: list[dict[str, Any]],
    args: argparse.Namespace,
    dampeners: list[dict[str, Any]] | None = None,
) -> str:
    payload = summary_payload(stats, recs, args, dampeners)
    totals = payload["totals"]
    protected = effective_protected_always_on(Path(__file__).resolve().parent)
    lines = [
        "tool-belt analyzer",
        f"scopes: {totals['scopes']}",
        f"prediction rows: {totals['prediction_rows']}",
        f"tool-call rows: {totals['tool_call_rows']}",
        f"expand_tools events: {totals['expand_tools_events']}",
        f"expansions with any downstream use: {totals['expansions_with_any_use']} ({pct(totals['expansions_with_any_use'], totals['expand_tools_events']):.1f}% success rate)",
        f"expanded-tool use rows: {totals['expanded_tool_use_rows']} ({totals['expanded_tool_use_rows_per_expansion']:.2f} rows per expansion)",
        f"logged first-call tokens saved: {totals['logged_first_call_tokens_saved']:,}",
        f"est. expand_tools round-trip cost: {totals['estimated_expand_round_trip_cost_tokens']:,} tok "
        f"(@ {args.expand_round_trip_tokens}/event)",
        f"est. net savings (saved − round-trip cost): {totals['estimated_net_savings_tokens']:+,} tok",
        f"recommendation candidates: {totals['recommendation_candidates']}",
        (
            f"tool-list stability: {totals['mutation_turns']:,}/{totals['comparable_turns']:,} comparable turns mutated "
            f"({totals['mutation_rate'] * 100:.1f}%); "
            f"{totals['sessions_with_mutation']}/{totals['sessions_observed']} sessions affected "
            f"({totals['sessions_with_mutation_rate'] * 100:.1f}%); "
            f"first turns excluded={totals['first_turns']:,}; "
            f"hash-missing={totals['hash_missing_turns']:,}"
        ),
    ]
    if args.include_archives:
        lines.append("archives: included by explicit flag")
    for scope, stat in stats.items():
        top_expansions = ", ".join(f"{k}={v}" for k, v in stat.expansions_by_category.most_common(3)) or "none"
        insufficient = sum(1 for rec in recs if rec.get("scope") == scope and rec.get("status") == "insufficient_data")
        stability = _stability_payload(stat)
        lines.append(
            f"- {scope}: predictions={stat.predictions}, sessions={len(stat.sessions)}, "
            f"expansions={stat.expansion_events}, successful_expansions={sum(len(event_ids) for event_ids in stat.used_expansion_event_ids_by_category.values())}, "
            f"expanded_use_rows={stat.used_expansion_rows}, top_expansions={top_expansions}, insufficient_data={insufficient}, "
            f"tool_list_mutation_rate={stability['mutation_rate'] * 100:.1f}% "
            f"({stability['mutation_turns']}/{stability['comparable_turns']} comparable, "
            f"{stability['sessions_with_mutation']} sessions affected)"
        )
    return "\n".join(lines)


def markdown_report(
    stats: dict[str, ScopeStats],
    recs: list[dict[str, Any]],
    args: argparse.Namespace,
    dampeners: list[dict[str, Any]] | None = None,
    trigger_keywords: list[dict[str, Any]] | None = None,
    cache_cost: dict[str, dict[str, Any]] | None = None,
) -> str:
    generated = time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())
    payload = summary_payload(stats, recs, args, dampeners, trigger_keywords)
    totals = payload["totals"]
    protected = effective_protected_always_on(Path(__file__).resolve().parent)

    lines = [
        "# tool-belt Analyzer Report",
        "",
        f"Generated: {generated}",
        f"State dir: `{args.state_dir}`",
        f"Archives included: `{args.include_archives}`",
        "",
        "## Summary",
        "",
        f"- Scopes: **{totals['scopes']}**",
        f"- Prediction rows: **{totals['prediction_rows']}**",
        f"- Tool-call rows: **{totals['tool_call_rows']}**",
        f"- expand_tools events: **{totals['expand_tools_events']}**",
        f"- Expansions with any downstream use: **{totals['expansions_with_any_use']}** ({pct(totals['expansions_with_any_use'], totals['expand_tools_events']):.1f}% success rate)",
        f"- Expanded-tool use rows: **{totals['expanded_tool_use_rows']}** ({totals['expanded_tool_use_rows_per_expansion']:.2f} rows per expansion)",
        f"- Logged first-call tokens saved: **{totals['logged_first_call_tokens_saved']:,}**",
        f"- Estimated expand_tools round-trip cost: **{totals['estimated_expand_round_trip_cost_tokens']:,}** tokens",
        f"- Estimated net savings (saved − round-trip cost): **{totals['estimated_net_savings_tokens']:+,}** tokens",
        "",
        "## Tool-list Stability (prefix-cache impact)",
        "",
        "A *mutation* is a turn where the wire-level tool schemas differed from the previous turn in the same session. "
        "Provider prefix-caches fingerprint the actual schema bytes, so a mutation invalidates the cache for the tool block AND the conversation history that follows it. "
        "The savings model above (`logged_first_call_tokens_saved`) does **not** subtract that loss — see the Cache-Aware Savings section below for the corrected net. "
        "Turn 1 is excluded from comparable-turn counts (always a cold cache by definition). "
        "`hash_missing_turns` flags rows where the hash is unavailable — when high, the stability figures reflect a narrower slice than the full telemetry window.",
        "",
        f"- Comparable turns (turn > 1, hash present): **{totals['comparable_turns']:,}**",
        f"- Mutation turns: **{totals['mutation_turns']:,}** ({totals['mutation_rate'] * 100:.1f}% of comparable)",
        f"- Stable turns: **{totals['stable_turns']:,}**",
        f"- First turns (excluded): {totals['first_turns']:,}",
        f"- Hash-missing turns (excluded): {totals['hash_missing_turns']:,}",
        f"- Sessions with at least one mutation: **{totals['sessions_with_mutation']:,}** of {totals['sessions_observed']:,} ({totals['sessions_with_mutation_rate'] * 100:.1f}%)",
        "",
        "Per-scope mutation rates appear in the Per-Scope Metrics section below.",
        "",
    ]

    # ─── Cache-Aware Savings section ──────────────────────────────────
    if cache_cost:
        total_usd_lost = _cache_cost_total_usd(cache_cost)
        total_tokens_lost = _cache_cost_total_tokens_lost(cache_cost)
        net_savings = totals["estimated_net_savings_tokens"]
        lines.extend([
            "## Cache-Aware Savings (matched counterfactual)",
            "",
            "Per-call `cache_read_tokens` are captured in `api_calls.jsonl` on every API response. "
            "The cache-freeze replay matches each mutated call against the stable cohort at the same "
            "`api_call_idx` position within the session, computing the counterfactual cache reads "
            "that were lost to tool-list mutations. Dollar estimates use per-model list prices.",
            "",
            f"- Cache read tokens lost to mutations (upper bound): **{total_tokens_lost:,}**",
            f"- Estimated USD cost of those lost cache reads: **${total_usd_lost:.4f}**",
            f"- Schema-savings net (from Summary above): **{net_savings:+,}** tokens",
            f"- Cache-corrected net = schema savings − cache-miss penalty (in tokens, not directly comparable due to different price rates)",
            "",
        ])
        for scope, scope_data in cache_cost.items():
            freeze = scope_data["freeze"]
            cf = scope_data["counterfactual"]
            lines.extend([
                f"### {scope}",
                "",
                f"- Comparable calls: **{freeze['comparable_calls']:,}**",
                f"- Matches frozen hash: **{freeze['matches_freeze']}** (avg cache_read: {freeze['avg_cache_read_when_matches']:,.0f})",
                f"- Expand-driven mutations: **{freeze['expand_driven_mutations']}** (accepted — model requested new tools)",
                f"- Would-break mutations: **{freeze['would_break_mutations']}** (avg cache_read: {freeze['avg_cache_read_when_would_break']:,.0f})",
                f"- Freeze eliminates **{freeze['freeze_eliminates_pct_of_mutations'] * 100:.1f}%** of observed mutations",
                "",
            ])
            per_model = cf.get("per_model", {})
            if per_model:
                lines.extend([
                    "Per-model cache cost:",
                    "",
                    "| model | calls | mut | stable | cache_read_lost | est_usd_lost |",
                    "|---|---:|---:|---:|---:|---:|",
                ])
                for model_name, m in sorted(per_model.items(), key=lambda kv: -int(kv[1]["calls"])):
                    lines.append(
                        f"| `{model_name}` | {m['calls']} | {m['mutated']} | {m['stable']} "
                        f"| {m['cache_read_lost_upper_bound']:,} | ${m['est_usd_lost_upper_bound']:.4f} |"
                    )
                lines.extend([
                    "",
                    "Numbers are upper bounds (signed). Negative values mean the mutation coincided with a legitimate cache refresh.",
                    "",
                ])
            else:
                lines.extend(["No mutated calls with position-matched cohort data for this scope.", ""])
    else:
        lines.extend([
            "## Cache-Aware Savings (matched counterfactual)",
            "",
            "> ⚠ No `api_calls.jsonl` data available — cache cost not computed. "
            "The freeze-replay script (`scripts/cache-freeze-replay.py`) can be run manually.",
            "",
        ])

    lines.extend([
        "",
        "## Adjustable Defaults Used",
        "",
        f"- Per-tool token estimate: `{args.per_tool_tokens}` tokens/API call",
        f"- Expand round-trip token estimate: `{args.expand_round_trip_tokens}` tokens/event",
        f"- Minimum expansions before promotion review: `{args.min_expansions}`",
        f"- Promotion expand-rate default: `{args.promote_expand_rate:.2f}`",
        f"- Promotion use-rate default: `{args.promote_use_rate:.2f}`",
        f"- Always-on unused review floor: `{args.unused_carry_turns}` carried turns",
        f"- Trigger precision review floor: `{args.trigger_min_fires}` fires",
        "",
        "These defaults are levers. Live telemetry should move them if re-invocation rates, latency, or false-positive costs point elsewhere.",
        "",
        "## Per-Scope Metrics",
        "",
    ])

    for scope, stat in stats.items():
        used_events = sum(len(event_ids) for event_ids in stat.used_expansion_event_ids_by_category.values())
        used_rows_per_expansion = (stat.used_expansion_rows / stat.expansion_events) if stat.expansion_events else 0.0
        lines.extend([
            f"### {scope}",
            "",
            f"- Predictions: {stat.predictions}",
            f"- Sessions observed: {len(stat.sessions)}",
            f"- Logged first-call tokens saved: {stat.total_tokens_saved:,}",
            f"- expand_tools events: {stat.expansion_events}",
            f"- Expansions with any downstream use: {used_events} ({pct(used_events, stat.expansion_events):.1f}% success rate)",
            f"- Expanded-tool use rows: {stat.used_expansion_rows} ({used_rows_per_expansion:.2f} rows per expansion)",
            f"- turns_until_used: avg {avg(stat.turns_until_used):.1f}, median {median_int(stat.turns_until_used)}, samples {len(stat.turns_until_used)}",
            "",
            "Expansion counts:",
        ])
        if stat.expansions_by_category:
            for category, count in stat.expansions_by_category.most_common():
                used_rows = stat.expanded_uses_by_category.get(category, 0)
                used_event_count = len(stat.used_expansion_event_ids_by_category.get(category, set()))
                data_note = " — insufficient data" if count < args.min_expansions else ""
                lines.append(
                    f"- `{category}`: expanded {count}, successful expansions {used_event_count}, "
                    f"success rate {pct(used_event_count, count):.1f}%, expanded-tool rows {used_rows} ({used_rows / count:.2f} rows per expansion){data_note}"
                )
        else:
            lines.append("- none")
        lines.extend(["", "Top actual tool calls:"])
        if stat.tool_calls:
            for tool, count in stat.tool_calls.most_common(10):
                lines.append(f"- `{tool}`: {count}")
        else:
            lines.append("- none")
        stability = _stability_payload(stat)
        lines.extend([
            "",
            "Tool-list stability (prefix-cache impact):",
            f"- Comparable turns: {stability['comparable_turns']}",
            f"- Mutations: {stability['mutation_turns']} ({stability['mutation_rate'] * 100:.1f}% of comparable)",
            f"- Stable: {stability['stable_turns']}",
            f"- Sessions with at least one mutation: {stability['sessions_with_mutation']} of {stability['sessions_observed']} ({stability['sessions_with_mutation_rate'] * 100:.1f}%)",
            f"- First turns excluded: {stability['first_turns']}; hash missing: {stability['hash_missing_turns']}",
        ])
        if stability["mutation_by_provider"] or stability["stable_by_provider"]:
            providers = sorted(set(stability["mutation_by_provider"]) | set(stability["stable_by_provider"]))
            lines.append("- By provider:")
            for p in providers:
                m = stability["mutation_by_provider"].get(p, 0)
                s = stability["stable_by_provider"].get(p, 0)
                label = p if p else "(blank)"
                lines.append(
                    f"  - `{label}`: mutations={m}, stable={s}, "
                    f"mutation_rate={pct(m, m + s):.1f}%"
                )
        if stability["mutation_by_model"] or stability["stable_by_model"]:
            models = sorted(set(stability["mutation_by_model"]) | set(stability["stable_by_model"]))
            lines.append("- By model:")
            for m_name in models:
                m = stability["mutation_by_model"].get(m_name, 0)
                s = stability["stable_by_model"].get(m_name, 0)
                label = m_name if m_name else "(blank)"
                lines.append(
                    f"  - `{label}`: mutations={m}, stable={s}, "
                    f"mutation_rate={pct(m, m + s):.1f}%"
                )

        lines.extend(["", "Cohorts by policy_source:"])
        cohorts = _cohort_payload(stat)
        if cohorts:
            for source, data in cohorts.items():
                lines.append(
                    f"- `{source}`: predictions={data['predictions']}, sessions={data['sessions']}, "
                    f"tool_calls={data['tool_calls']} "
                    f"({data['tool_calls_per_prediction']:.2f}/pred), "
                    f"expand_events={data['expand_tools_events']} "
                    f"(rate {data['expand_rate']:.3f})"
                )
        else:
            lines.append("- none observed")
        lines.extend(["", "Trigger precision/recall proxy:"])
        if stat.trigger_fires:
            for trigger, fires in stat.trigger_fires.most_common():
                hits = stat.trigger_hits.get(trigger, 0)
                needs = stat.trigger_needs.get(trigger, 0)
                lines.append(
                    f"- `{trigger}`: fired {fires}, same-prediction hits {hits}, "
                    f"precision {pct(hits, fires):.1f}%, recall {pct(hits, needs):.1f}%"
                )
        else:
            lines.append("- none supported by current rows")
        lines.extend(["", "Always-on carrying cost / unused candidates:"])
        unused = [
            (tool, carried)
            for tool, carried in stat.always_on_carry_turns.most_common()
            if tool not in protected and stat.tool_calls.get(tool, 0) == 0 and carried >= args.unused_carry_turns
        ]
        if unused:
            for tool, carried in unused[:10]:
                cost = carrying_cost(carried, args.per_tool_tokens)
                lines.append(f"- `{tool}`: carried {carried} turns, no calls, estimated carry cost {cost:,} tokens")
        else:
            lines.append("- none supported by current data/default floor")
        lines.append("")

    lines.extend(["## Recommendation Candidates", ""])
    if recs:
        for rec in recs:
            label = f"{rec['scope']} / {rec['kind']} / `{rec['item']}`"
            lines.extend([
                f"### {label}",
                "",
                f"- Action: **{rec['action']}**",
                f"- Status: {rec['status']}",
                f"- Confidence: {rec['confidence']}",
                f"- Metrics: `{json.dumps(rec.get('metrics', {}), sort_keys=True)}`",
                f"- Reason: {rec['reason']}",
                "",
            ])
    else:
        lines.append("No candidates met the current default floors.")

    if dampeners:
        lines.extend(["", "## Suggested Dampener Candidates", ""])
        lines.append(
            "Mined from messages where a trigger fired but no matching tool was "
            "called. Review and paste high-precision candidates into the "
            "preset's `exclude_keywords` list."
        )
        statuses = {row.get("preset_excludes_status", "ok") for row in dampeners}
        degraded = sorted(s for s in statuses if s and s != "ok")
        if degraded:
            for status in degraded:
                note = _EXCLUDES_STATUS_MESSAGE.get(
                    status, f"preset excludes unavailable: {status}"
                )
                lines.append("")
                lines.append(f"> ⚠ Degraded mode (`{status}`): {note}")
        lines.append("")
        for row in dampeners:
            status = row.get("preset_excludes_status", "ok")
            status_suffix = "" if status == "ok" else f", excludes_status={status}"
            lines.append(
                f"### {row['scope']} / `{row['trigger']}` "
                f"(FP={row['fp_previews_mined']}, TP={row['tp_previews_mined']}, "
                f"existing excludes={row['existing_exclude_keyword_count']}{status_suffix})"
            )
            lines.append("")
            for cand in row.get("candidates", []):
                lines.append(
                    f"- pattern `{cand['pattern']}` → regex `{cand['suggested_regex']}`  "
                    f"(precision {cand['precision']:.2f}, fp={cand['fp_count']}, tp={cand['tp_count']})"
                )
                for sample in cand.get("sample_previews", []):
                    lines.append(f"    - {sample}")
                lines.append("")
    elif getattr(args, "suggest_dampeners", False):
        lines.extend([
            "",
            "## Suggested Dampener Candidates",
            "",
            "No candidates met the current floors. Try lowering "
            "`--dampener-min-support` or `--dampener-min-precision`, or wait "
            "for more telemetry to accumulate.",
        ])

    # ─── Harvest-driven sections (only present when harvest data is loaded) ───
    harvest_recs = [r for r in (recs or []) if r.get("kind") == "harvest_tool_promotion"]
    if harvest_recs:
        lines.extend(["", "## Harvest-Driven Per-Tool Promotion Candidates", ""])
        lines.append(
            "Derived from `policy_source: harvest` rows where the historical "
            "model called a tool that current narrowing would have cut. Each "
            "such call is a counterfactual `expand_tools` round-trip. Action "
            "values: `promote_always_on` (net token savings positive), "
            "`broaden_trigger_recall` (frequent demand but per-tool carry "
            "would lose net — fix the trigger instead), `keep_gated` (sparse)."
        )
        lines.append("")
        for r in harvest_recs:
            m = r["metrics"]
            lines.append(
                f"### {r['scope']} / `{r['item']}` — **{r['action']}** "
                f"(cuts={m['harvest_was_cut']}, cut_rate={m['cut_rate']:.1%}, "
                f"net={m['net_savings_tokens']:+,} tok)"
            )
            lines.append("")
            lines.append(f"- {r['reason']}")
            lines.append("")

    if trigger_keywords:
        lines.extend(["", "## Suggested Trigger-Keyword Candidates", ""])
        lines.append(
            "Mined from harvest messages where the model called a tool that "
            "would have been cut by current narrowing. Symmetric inverse of "
            "the dampener flow — these add coverage, dampeners remove false "
            "fires. Review high-precision candidates before pasting into "
            "the target trigger group's `keywords` list."
        )
        lines.append("")
        for row in trigger_keywords:
            lines.append(
                f"### {row['scope']} / `{row['tool']}` → **{row['action']}** "
                f"`{row['target_trigger']}` (cuts={row['cut_count']}, "
                f"existing patterns={row['existing_keyword_pattern_count']})"
            )
            lines.append("")
            for cand in row.get("candidates", []):
                lines.append(
                    f"- pattern `{cand['pattern']}` → regex `{cand['suggested_regex']}`  "
                    f"(precision {cand['precision']:.2f}, "
                    f"cut={cand['cut_count']}, noise={cand['noise_count']})"
                )
                for sample in cand.get("sample_previews", []):
                    lines.append(f"    - {sample}")
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_recommendations(
    path: Path,
    recs: list[dict[str, Any]],
    args: argparse.Namespace,
    dampeners: list[dict[str, Any]] | None = None,
) -> None:
    payload = {
        "version": 2,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "recommend",
        "safety": {
            "writes_learned_json": False,
            "applies_policy": False,
            "default_behavior": "safe/off unless learned_mode is explicitly changed elsewhere",
        },
        "thresholds": threshold_payload(args),
        "recommendations": recs,
        "dampener_candidates": list(dampeners or []),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze tool-belt JSONL telemetry")
    parser.add_argument("--state-dir", type=Path, default=state_dir_from_env(), help="Directory containing predictions.jsonl and tool_calls.jsonl")
    parser.add_argument("--reports-dir", type=Path, default=None, help="Directory for markdown reports; default: plugin reports/")
    parser.add_argument("--no-report", action="store_true", help="Print summary only; do not write markdown report")
    parser.add_argument("--write-recommendations", action="store_true", help="Write learned_recommendations.json next to telemetry")
    parser.add_argument("--include-archives", action="store_true", help="Also read matching JSONL files under state-dir/archive/; off by default to avoid surprise live-history scope changes")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Stdout format")
    parser.add_argument("--per-tool-tokens", type=int, default=DEFAULT_PER_TOOL_TOKENS)
    parser.add_argument("--min-expansions", type=int, default=DEFAULT_MIN_EXPANSIONS)
    parser.add_argument("--promote-expand-rate", type=float, default=DEFAULT_PROMOTE_EXPAND_RATE)
    parser.add_argument("--promote-use-rate", type=float, default=DEFAULT_PROMOTE_USE_RATE)
    parser.add_argument("--unused-carry-turns", type=int, default=DEFAULT_UNUSED_CARRY_TURNS)
    parser.add_argument("--trigger-min-fires", type=int, default=DEFAULT_TRIGGER_MIN_FIRES)
    parser.add_argument(
        "--expand-round-trip-tokens",
        type=int,
        default=DEFAULT_EXPAND_ROUND_TRIP_TOKENS,
        help=(
            "Estimated total token cost of one expand_tools round-trip "
            "(model output + result + extra API request). Used by net-value "
            "promotion math. Tune to your measured baseline."
        ),
    )
    parser.add_argument(
        "--suggest-dampeners",
        action="store_true",
        help=(
            "Mine candidate exclude_keywords from telemetry: n-grams that "
            "appear in messages where a trigger fired but no matching tool "
            "was called. Output appears in JSON, markdown report, and (with "
            "--write-recommendations) the recommendations JSON."
        ),
    )
    parser.add_argument("--dampener-min-support", type=int, default=DEFAULT_DAMPENER_MIN_SUPPORT,
                        help="Minimum false-positive occurrences for a candidate n-gram")
    parser.add_argument("--dampener-min-precision", type=float, default=DEFAULT_DAMPENER_MIN_PRECISION,
                        help="Minimum FP / (FP+TP) ratio for a candidate to survive")
    parser.add_argument("--dampener-min-n", type=int, default=DEFAULT_DAMPENER_MIN_N,
                        help="Shortest n-gram length to consider (default 2)")
    parser.add_argument("--dampener-max-n", type=int, default=DEFAULT_DAMPENER_MAX_N,
                        help="Longest n-gram length to consider (default 4)")
    parser.add_argument("--harvest-min-cuts", type=int, default=3,
        help="Minimum was_cut count before a tool surfaces as a harvest-driven "
             "promotion candidate. Filters thin-signal noise. (default: 3)")
    parser.add_argument("--suggest-trigger-keywords", action="store_true",
        help="Mine cut-tool message previews for candidate trigger keywords. "
             "Symmetric inverse of --suggest-dampeners. Requires harvest data.")
    parser.add_argument("--dampener-max-candidates", type=int, default=DEFAULT_DAMPENER_MAX_CANDIDATES,
                        help="Max candidates surfaced per (scope, trigger)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions_paths = telemetry_paths(args.state_dir, "predictions.jsonl", args.include_archives)
    tool_calls_paths = telemetry_paths(args.state_dir, "tool_calls.jsonl", args.include_archives)
    predictions = load_rows(predictions_paths)
    tool_calls = load_rows(tool_calls_paths)
    stats = collect_stats(predictions, tool_calls)
    protected_always_on = effective_protected_always_on(Path(__file__).resolve().parent)
    recs = recommendation_rows(stats, args, protected_always_on) + harvest_recommendation_rows(stats, args, protected_always_on)
    dampeners = dampener_candidates(stats, args)
    trigger_keywords = trigger_keyword_candidates(stats, args)

    # Load api_calls.jsonl for cache-cost analysis (cache_read_tokens per call)
    api_calls_paths = telemetry_paths(args.state_dir, "api_calls.jsonl", args.include_archives)
    api_calls = load_rows(api_calls_paths)
    cache_cost = compute_cache_cost(predictions, api_calls, tool_calls, list(stats.keys())) if api_calls else None

    # Break down tool_calls by source so the headline counts don't conflate
    # narrowed gateway calls with cron / subagent calls that bypass narrowing.
    source_counts = {"gateway": 0, "cron": 0, "subagent": 0}
    for row in tool_calls:
        src = row.get("source")
        if src not in source_counts:
            sid = str(row.get("session_id") or "")
            pid = str(row.get("prediction_id") or "")
            src = "cron" if sid.startswith("cron_") else ("gateway" if pid else "subagent")
        source_counts[src] += 1

    if args.format == "json":
        payload = summary_payload(stats, recs, args, dampeners, trigger_keywords)
        payload["totals"]["tool_call_source_counts"] = source_counts
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_summary(stats, recs, args, dampeners))
        excluded = source_counts["cron"] + source_counts["subagent"]
        if excluded:
            print(
                f"  └─ {source_counts['gateway']} narrowed (gateway), "
                f"{source_counts['cron']} excluded (cron), "
                f"{source_counts['subagent']} excluded (subagent)"
            )

    status_stream = sys.stderr if args.format == "json" else sys.stdout

    if args.write_recommendations:
        rec_path = args.state_dir / "learned_recommendations.json"
        write_recommendations(rec_path, recs, args, dampeners)
        print(f"recommendations: {rec_path}", file=status_stream)

    if not args.no_report:
        reports_dir = args.reports_dir or Path(__file__).resolve().parent / "reports"
        stamp = time.strftime("%Y-%m-%d-%H%M%S", time.gmtime())
        report_path = reports_dir / f"{stamp}-analysis.md"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            markdown_report(stats, recs, args, dampeners, trigger_keywords, cache_cost=cache_cost),
            encoding="utf-8",
        )
        print(f"report: {report_path}", file=status_stream)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
