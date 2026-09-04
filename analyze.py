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
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Canonical telemetry adapter. Every prediction/tool-call row is read through
# ``logger_io.normalize_prediction_row`` / ``normalize_tool_call_row`` so v1, v2,
# and mixed streams all present the same carrying-model fields. This module
# consumes ``carry_tools``, ``expand_only_tools``, and ``was_expand_only``.
# Works both as a script (sibling on sys.path) and as the
# ``tool_belt_plugin.analyze`` module.
try:  # package / test import
    from . import logger_io  # type: ignore[import-not-found]
    from . import savings as _savings  # type: ignore[import-not-found]
    from .yaml_required import require_yaml  # type: ignore[import-not-found]
except ImportError:  # script mode
    import logger_io  # type: ignore[no-redef]
    import savings as _savings  # type: ignore[no-redef]
    from yaml_required import require_yaml  # type: ignore[no-redef]

DEFAULT_PER_TOOL_TOKENS = logger_io.DEFAULT_PER_TOOL_TOKENS  # canonical home: logger_io
DEFAULT_MIN_EXPANSIONS = 2
DEFAULT_PROMOTE_EXPAND_RATE = 0.50
DEFAULT_PROMOTE_USE_RATE = 0.80
DEFAULT_UNUSED_CARRY_TURNS = 10
DEFAULT_TRIGGER_MIN_FIRES = 3
# Per-event cost of one expand_tools round-trip. ``main`` MEASURES this per
# scope from the same telemetry (``savings.measure_expand_overhead``); this
# constant is the FALLBACK charged when a scope's telemetry is too thin, and
# --expand-round-trip-tokens overrides only that fallback. Single-sourced in
# the canonical savings engine so the analyzer, the savings report, the
# projection, and the shaper all fall back to the same figure.
DEFAULT_EXPAND_ROUND_TRIP_TOKENS = _savings.EXPAND_ROUND_TRIP_TOKENS
# Dampener suggestion defaults. These are conservative — tuning down min
# support or precision quickly produces noise; tuning up shrinks the
# candidate set to obviously-good patterns only.
DEFAULT_DAMPENER_MIN_SUPPORT = 3
DEFAULT_DAMPENER_MIN_PRECISION = 0.8
DEFAULT_DAMPENER_MIN_N = 2
DEFAULT_DAMPENER_MAX_N = 4
DEFAULT_DAMPENER_MAX_CANDIDATES = 10
# Minimum was_expand_only call count before a tool surfaces as a harvest-driven
# promote-to-carry candidate.
DEFAULT_HARVEST_MIN_EXPAND_ONLY_CALLS = 3
# Session-bounded lookahead window for trigger true-positive attribution. This
# is an analysis-time approximation of the runtime ``sticky_ttl_turns`` (default
# 3), which is config-overridable — the two can diverge on a tuned install.
DEFAULT_STICKY_LOOKAHEAD_TURNS = 3
# Schema version stamped on every artifact this module emits — the `--format
# json` summary payload and `learned_recommendations.json`. It tracks the
# analyzer's *output* layout (key names and nesting), not the telemetry schema
# and not the learned.json schema, and both artifacts carry the same number
# because one run produces them from the same structures.
ANALYZER_PAYLOAD_VERSION = 2

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
    # Tools resolved by each category (from expand_tools result payload).
    # Used by expand-cost accounting to estimate carry cost if we promote.
    tools_per_category: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    used_expansion_rows: int = 0
    used_expansion_event_ids_by_category: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    expanded_uses_by_category: Counter[str] = field(default_factory=Counter)
    turns_until_used: list[int] = field(default_factory=list)
    tool_calls: Counter[str] = field(default_factory=Counter)
    # Per-tool count of turns a `carry`-class resident was carried. Immutable
    # `always_carry` residents are excluded (they can never be demoted). Feeds
    # demote-to-expand_only candidate detection.
    carry_turns: Counter[str] = field(default_factory=Counter)
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
    # Harvest-specific counterfactual signal: per-tool count of historical
    # calls the current partition would have left in the derived `expand_only`
    # class (canonical ``was_expand_only`` on the tool_call row). Populated only
    # when harvest-source rows are present. The corresponding
    # `harvest_predictions` is the denominator for carry-cost math when
    # proposing promote-to-carry.
    harvest_was_expand_only: Counter[str] = field(default_factory=Counter)
    # Distinct harvest predictions that made at least one was_expand_only call
    # for the tool. `harvest_was_expand_only` counts *calls* (several can land
    # in one turn), so only this set can form a true per-prediction rate.
    harvest_expand_only_predictions: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set))
    harvest_predictions: int = 0
    # Per-tool sample of message previews that led to a was_expand_only call.
    # Used by the trigger-keyword suggester to mine candidate keywords for
    # broadening the trigger groups that would activate those expand_only tools.
    harvest_expand_only_previews: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
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
    return str(row.get("scope") or "").strip().lower() or "unknown"


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
    on the source side and ``result.resolved_tools`` on the response side;
    a row may carry only one of them.
    """
    for key in ("expand_event", "result"):
        source = row.get(key) if isinstance(row.get(key), dict) else {}
        tools = source.get("resolved_tools") if isinstance(source, dict) else None
        if isinstance(tools, list) and tools:
            return [str(t).strip() for t in tools if str(t).strip()]
    return []


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
    # Read every row through the central adapter so v1, v2, and mixed streams
    # present the same canonical carrying-model fields. Idempotent, so callers
    # that already normalized (e.g. ``main``) pay only a cheap re-pass.
    predictions = [logger_io.normalize_prediction_row(r) for r in predictions]
    tool_calls = [logger_io.normalize_tool_call_row(r) for r in tool_calls]

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
        # grouping; fall back to session_id (the stable chat key).
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
        # Only `carry`-class residents feed demote candidacy; immutable
        # always_carry residents can never be demoted (filtered later against
        # the preset always_carry set, which also protects rows whose
        # residents all normalize into `carry_tools`).
        for tool in string_list(row.get("carry_tools")):
            stat.carry_turns[tool] += 1
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
            # Harvest-source counterfactual: count calls the current partition
            # would have left expand_only (canonical ``was_expand_only``) per
            # tool, and capture the prediction's message preview so the trigger-
            # keyword suggester can mine candidates.
            if cohort == "harvest" and row.get("was_expand_only") is True:
                stat.harvest_was_expand_only[tool] += 1
                if prediction_id:
                    stat.harvest_expand_only_predictions[tool].add(prediction_id)
                preview = prediction_preview.get(prediction_id, "")
                if preview and len(stat.harvest_expand_only_previews[tool]) < 200:
                    stat.harvest_expand_only_previews[tool].append(preview)

        if tool == "expand_tools":
            category = category_from_expand_row(row)
            stat.expansion_events += 1
            stat.expansions_by_category[category] += 1
            if cohort:
                stat.cohort_expansions[cohort] += 1
            for resolved in _resolved_tools_from_expand_row(row):
                stat.tools_per_category[category].add(resolved)
            continue

        # Skip calls where the tool was already active before expansion —
        # those are not expansion successes.
        if row.get("expansion_provided_access") is True and not row.get("was_initially_active", False):
            category = str(row.get("expand_category") or "unknown").strip() or "unknown"
            stat.used_expansion_rows += 1
            expansion_prediction_id = str(row.get("expansion_prediction_id") or "").strip()
            event_prediction_id = expansion_prediction_id or (prediction_id if row.get("after_expand_tools") else "")
            if event_prediction_id:
                stat.used_expansion_event_ids_by_category[category].add(f"{event_prediction_id}:{category}")
            stat.expanded_uses_by_category[category] += 1
            stat.turns_until_used.append(max(0, safe_int(row.get("turns_until_used"))))

    # Build a session-bounded lookahead window so that sticky-carried
    # tool calls in subsequent turns count as TPs for the trigger that
    # fired on the earlier turn. Without this, a trigger that correctly
    # predicted intent but where the model called the tool 1-2 turns
    # later (sticky residency carried the expansion) would be miscounted as
    # an FP, inflating dampener-candidate FP counts.
    #
    # Rows without a session identifier fall back to same-prediction behavior
    # so scope-ordered timeline lookups cannot leak across sessions.
    WINDOW = DEFAULT_STICKY_LOOKAHEAD_TURNS
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
      ``"no_policy"``    — policy.yaml file missing under ``plugin_dir``
      ``"parse_error"``  — YAML present but failed to parse

    A missing PyYAML is not one of them: :func:`require_yaml` exits the
    process instead, because a run without the parser silently reports
    "existing_exclude_keyword_count: 0" against a policy it never read.
    """
    path = plugin_dir / "policy.yaml"
    if not path.exists():
        return {}, "no_policy"
    yaml = require_yaml()
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
    "no_policy": (
        "policy.yaml not found alongside analyze.py — dampener candidates "
        "will not be deduplicated against existing exclude_keywords."
    ),
    "parse_error": (
        "policy.yaml failed to parse — dampener candidates will not be "
        "deduplicated against existing exclude_keywords this run."
    ),
}


def _load_preset_always_carry(plugin_dir: Path) -> tuple[set[str], str]:
    """Load the preset's immutable ``always_carry`` tool names from policy.yaml.

    These are the class-A residents that win every conflict and can never be
    demoted; the analyzer treats them as the do-not-suggest set (an always_carry
    tool is never a demote candidate, and a row whose residents all normalize
    into ``carry`` is protected against spurious demotion by this set).

    Returns ``(tools, status)`` using the same status enum as
    :func:`_load_preset_excludes`. PyYAML is required — a line-oriented
    fallback would honor ``always_carry`` but silently drop
    ``exclude_keywords`` and ``triggers``, producing a half-read policy that
    looked like a complete one.
    """
    path = plugin_dir / "policy.yaml"
    if not path.exists():
        return set(), "no_policy"

    yaml = require_yaml()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return set(), "parse_error"
    tools = {
        str(tool).strip()
        for tool in (data.get("always_carry") or [])
        if str(tool).strip()
    }
    return tools, "ok"


def preset_always_carry(plugin_dir: Path | None = None) -> set[str]:
    """Return the immutable ``always_carry`` set from the resolved preset.

    ``policy.yaml`` is the single source of truth — no hard-coded protected
    list. always_carry residents are the tools the adaptive layer must never
    demote, so the analyzer excludes them from demote-to-expand_only candidacy
    and from carry-cost accounting.
    """
    if plugin_dir is None:
        plugin_dir = Path(__file__).resolve().parent
    tools, _status = _load_preset_always_carry(plugin_dir)
    return tools


def _load_preset_triggers(
    plugin_dir: Path,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Load the full trigger structure from policy.yaml.

    Returns ``({group_name: {tools: [...], keyword_patterns: [...]}}, status)``
    with the same status enum as :func:`_load_preset_excludes`. Used by
    the trigger-keyword suggester to:
      · Decide which group a candidate keyword should join (the one whose
        ``tools`` already lists the expand_only tool).
      · Avoid re-suggesting keywords that an existing trigger pattern
        already matches.
    """
    path = plugin_dir / "policy.yaml"
    if not path.exists():
        return {}, "no_policy"
    yaml = require_yaml()
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
    """Heuristic group name when no existing group claims an expand_only tool.

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
    in ``api_calls.jsonl`` and computed by ``scripts/cache-stability-replay.py``,
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


def expand_overhead_for_scope(
    args: argparse.Namespace, scope: str
) -> "_savings.MeasuredExpandOverhead | None":
    """The scope's measured expansion overhead, if ``main`` computed one.

    ``main`` stores ``{scope: MeasuredExpandOverhead}`` on
    ``args.expand_overhead_by_scope``; callers that build ``args`` directly
    (tests, ad-hoc scripts) have none and fall back to the flag.
    """
    table = getattr(args, "expand_overhead_by_scope", None)
    if isinstance(table, dict):
        measured = table.get(scope)
        if measured is not None:
            return measured
    return None


def expand_round_trip_tokens_for(args: argparse.Namespace, scope: str) -> int:
    """Per-event expand_tools cost for ``scope``: the MEASURED event-weighted
    figure across its cache cohorts, else ``--expand-round-trip-tokens``."""
    measured = expand_overhead_for_scope(args, scope)
    if measured is not None:
        return int(measured.blended_per_event)
    return int(args.expand_round_trip_tokens)


def expand_round_trip_basis_for(args: argparse.Namespace, scope: str) -> str:
    """``measured`` | ``fallback`` | ``mixed`` for the figure above."""
    measured = expand_overhead_for_scope(args, scope)
    return measured.blended_basis if measured is not None else "fallback"


def category_net_value(
    expansions: int,
    predictions: int,
    tools_in_category: int,
    expand_round_trip_tokens: int,
    per_tool_tokens: int,
) -> dict[str, int]:
    """Estimate the net token value of promoting a category into the carry class.

    Saved by promotion: each expansion event avoids one round-trip
    (``expansions × expand_round_trip_tokens`` — the caller passes the scope's
    measured per-event cost via :func:`expand_round_trip_tokens_for`). Costs
    of promotion: every prediction now carries the category's tools in the
    request schema (``predictions × tools_in_category × per_tool_tokens``).
    Net value is saved minus cost; positive means promotion is likely
    beneficial under these defaults, negative means keep gated.
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


def _ngram_contains(longer: str, shorter: str) -> bool:
    """True when ``shorter`` sits inside ``longer`` on whole-word boundaries.

    Raw substring containment would call ``"pen"`` contained in ``"open the"``
    and let an unrelated candidate suppress a real one. Padding both sides with
    spaces makes the test word-wise, matching the dampener miner's dedupe.
    """
    if shorter == longer or not shorter:
        return False
    return f" {shorter} " in f" {longer} "


def _suggest_keywords_for_expand_only_tool(
    expand_only_previews: list[str],
    noise_previews: list[str],
    existing_patterns: list[re.Pattern[str]],
    *,
    min_n: int,
    max_n: int,
    min_support: int,
    min_precision: float,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Mine n-grams that recur in expand_only-tool messages but rarely elsewhere.

    Mirrors :func:`_suggest_dampeners_for_trigger` with inverted semantics:
    dampener mining wants n-grams that ONLY appear in FP messages (suppress
    over-triggering); keyword mining wants n-grams that appear in messages where
    the model reached for a tool the current partition would have left
    expand_only, but NOT in unrelated noise (surface under-triggering).

    Returns candidate keyword dicts ranked by precision then frequency.
    """
    if not expand_only_previews or len(expand_only_previews) < min_support:
        return []

    raw_candidates: list[dict[str, Any]] = []
    expand_only_lower = [p.lower() for p in expand_only_previews]
    noise_lower = [p.lower() for p in noise_previews]

    seen: set[str] = set()
    for preview in expand_only_lower:
        tokens = _tokenize_preview(preview)
        for n in range(min_n, max_n + 1):
            for ngram in _ngrams(tokens, n):
                if ngram in seen:
                    continue
                seen.add(ngram)
                if not _has_content_word(ngram):
                    continue
                support_count = sum(1 for p in expand_only_lower if ngram in p)
                if support_count < min_support:
                    continue
                noise_count = sum(1 for p in noise_lower if ngram in p)
                precision = support_count / max(1, support_count + noise_count)
                if precision < min_precision:
                    continue
                raw_candidates.append({
                    "ngram": ngram,
                    "support_count": support_count,
                    "noise_count": noise_count,
                    "precision": precision,
                    "length": n,
                })

    # Dedupe substrings: if "open the page" survives and "open the" also
    # survives with similar precision, prefer the longer (more specific)
    # one. Identical-stats shorter candidates get dropped.
    raw_candidates.sort(key=lambda c: (-c["precision"], -c["support_count"], -c["length"]))
    dropped: set[str] = set()
    kept: list[dict[str, Any]] = []
    for cand in raw_candidates:
        if cand["ngram"] in dropped:
            continue
        if _is_already_covered(cand["ngram"], existing_patterns):
            continue
        kept.append(cand)
        for other in raw_candidates:
            if (_ngram_contains(cand["ngram"], other["ngram"])
                    and other["support_count"] <= cand["support_count"]):
                dropped.add(other["ngram"])
    # Final re-filter, mirroring _suggest_dampeners_for_trigger: an entry
    # appended before a later-ranked, longer n-gram subsumed it is still in
    # `kept`, so drop it here rather than emitting a candidate we discarded.
    kept = [c for c in kept if c["ngram"] not in dropped]

    out: list[dict[str, Any]] = []
    for cand in kept[:max_candidates]:
        ngram = cand["ngram"]
        samples = [p for p in expand_only_previews if ngram in p.lower()][:3]
        out.append({
            "pattern": ngram,
            "suggested_regex": _ngram_to_regex(ngram),
            "support_count": cand["support_count"],
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
    flow mines expand_only messages to surface under-triggering — keywords
    that would have fired a trigger to activate a tool the historical
    model called but the current partition would have left expand_only.

    Auto-detects: returns empty when ``--suggest-trigger-keywords`` is
    off OR no scope has harvest data with sufficient expand_only volume
    per tool.

    For each candidate expand_only tool, decides whether the suggestion
    should augment an existing trigger group (the one whose ``tools``
    already lists this tool) or propose a new group.
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
    # expand_only previews but score precision=1.0 because the narrow noise
    # pool (other-tools' expand_only previews only) doesn't happen to contain
    # them. Using the full prediction corpus puts genuine common-noise
    # phrases out of the running.
    noise_corpus: dict[str, list[str]] = {
        scope: list(stat.harvest_all_previews) for scope, stat in stats.items()
    }

    rows: list[dict[str, Any]] = []
    min_calls = max(1, args.harvest_min_expand_only_calls)
    for scope, stat in sorted(stats.items()):
        if stat.harvest_predictions == 0:
            continue
        for tool, expand_only_count in stat.harvest_was_expand_only.most_common():
            if expand_only_count < min_calls:
                continue
            expand_only_previews = stat.harvest_expand_only_previews.get(tool, [])
            if not expand_only_previews:
                continue
            # Noise = harvest previews where THIS tool was NOT expand_only.
            other_previews = [
                p for p in noise_corpus.get(scope, [])
                if p not in expand_only_previews
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

            candidates = _suggest_keywords_for_expand_only_tool(
                expand_only_previews=expand_only_previews,
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
                "expand_only_count": expand_only_count,
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
    immutable_always_carry: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Per-tool promote-to-carry candidates derived from harvest telemetry.

    Harvest rows carry ``was_expand_only: True`` whenever the historical model
    called a tool the current partition would have left expand_only. Each such
    call represents a counterfactual ``expand_tools`` round-trip. The direct
    auto-apply signal is:

        net_savings = expand_only_calls × expand_round_trip_tokens
                    − predictions × per_tool_tokens

    where ``expand_round_trip_tokens`` is the scope's measured per-event cost
    (:func:`expand_round_trip_tokens_for`; the flag is only the fallback).
    Positive net favors promote-to-carry; negative favors keeping the tool
    expand_only (relying on trigger activation / explicit expansion to cover its
    use cases). Auto-detected — emits nothing when no harvest rows are present,
    so the function is safe to call unconditionally.

    Note this is a per-tool recommendation alongside the existing per-category
    recommendations from ``recommendation_rows`` — the two views answer
    different questions. Per-tool ("promote terminal to carry?") is what harvest
    data can answer directly. Per-category ("is the browser expand round-trip
    net-positive?") needs live expand_tools events with success-rate signal,
    which harvest can't produce.
    """
    rows: list[dict[str, Any]] = []
    immutable = set(immutable_always_carry if immutable_always_carry is not None
                    else preset_always_carry())
    min_calls = max(1, args.harvest_min_expand_only_calls)
    for scope, stat in sorted(stats.items()):
        if stat.harvest_predictions == 0:
            continue
        denom = max(stat.harvest_predictions, 1)
        per_event = expand_round_trip_tokens_for(args, scope)
        per_event_basis = expand_round_trip_basis_for(args, scope)
        for tool, expand_only_calls in stat.harvest_was_expand_only.most_common():
            if tool in immutable:
                continue
            if expand_only_calls < min_calls:
                continue
            round_trip_cost = expand_only_calls * per_event
            carry_cost = denom * args.per_tool_tokens
            net = round_trip_cost - carry_cost
            # A true rate: the share of harvest *predictions* in which this
            # tool was reached for at least once. Dividing the call count by
            # the prediction count is not a rate — one turn can make several
            # was_expand_only calls, which would push the figure over 100%.
            expand_only_predictions = len(
                stat.harvest_expand_only_predictions.get(tool, ()))
            if expand_only_predictions:
                expand_only_rate = expand_only_predictions / denom
            else:
                # Rows without a prediction_id can't be counted distinctly;
                # fall back to the call-based ratio, capped at a real rate.
                expand_only_rate = min(1.0, expand_only_calls / denom)
            calls_per_prediction = expand_only_calls / denom
            if net > 0:
                action = "promote_to_carry"
                status = "review"
                confidence = "medium" if expand_only_calls >= min_calls * 3 else "low"
            elif expand_only_rate >= 0.10:
                # Frequently reached for but carry cost outweighs the round-trip.
                # The right move is usually broadening the trigger that would
                # activate it, not making it resident. Keep it expand_only and
                # flag for human review.
                action = "keep_expand_only"
                status = "review"
                confidence = "medium"
            else:
                action = "keep_expand_only"
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
                    "harvest_was_expand_only": expand_only_calls,
                    "harvest_expand_only_predictions": expand_only_predictions,
                    "expand_only_rate": round(expand_only_rate, 4),
                    "expand_only_calls_per_prediction": round(calls_per_prediction, 4),
                    "round_trip_cost_tokens": round_trip_cost,
                    "carry_cost_tokens": carry_cost,
                    "net_savings_tokens": net,
                },
                "defaults": {
                    "expand_round_trip_tokens": per_event,
                    "expand_round_trip_basis": per_event_basis,
                    "per_tool_tokens": args.per_tool_tokens,
                    "harvest_min_expand_only_calls": min_calls,
                },
                # Advisory-only rows carry an empty patch, matching the
                # category/trigger rows: a `keep_expand_only` row proposes no
                # state change, so it must not name a scope with an empty
                # carry list (a syntactically valid patch that means nothing).
                "proposed_learned_patch": (
                    {"scopes": {scope: {"carry": [tool]}}}
                    if action == "promote_to_carry" else {"scopes": {}}
                ),
                "reason": (
                    f"historical sessions called {tool} {expand_only_calls} time(s) "
                    f"(in {expand_only_rate:.1%} of predictions) where the current "
                    f"partition would have left it expand_only; round-trip cost "
                    f"{round_trip_cost:,} tok vs carry cost {carry_cost:,} tok → "
                    f"net {net:+,} tok"
                ),
            })
    return rows


def recommendation_rows(
    stats: dict[str, ScopeStats],
    args: argparse.Namespace,
    immutable_always_carry: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    immutable = set(immutable_always_carry if immutable_always_carry is not None
                    else preset_always_carry())
    for scope, stat in stats.items():
        denominator = max(stat.predictions, 1)
        per_event = expand_round_trip_tokens_for(args, scope)
        per_event_basis = expand_round_trip_basis_for(args, scope)
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
                expand_round_trip_tokens=per_event,
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
                    action = "promote_to_carry"
                    confidence = "medium"
                status = "review"
            elif success_rate >= args.promote_use_rate:
                action = "watch_high_precision"
                confidence = "medium"
                status = "watch"
            else:
                action = "keep_expand_only"
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
                    "expand_round_trip_tokens": per_event,
                    "expand_round_trip_basis": per_event_basis,
                    "per_tool_tokens": args.per_tool_tokens,
                },
                # Category rows are advisory: ``item`` is a toolset name, not a
                # tool name, so no per-tool ``carry`` patch can be emitted. The
                # analyzer also has no reliable category→enabled-member map here
                # (``tools_per_category`` is telemetry-observed usage, not the
                # preset's membership). The promote signal is carried by
                # ``action``/``metrics``/``reason`` instead.
                "proposed_learned_patch": {"scopes": {}},
                "reason": (
                    f"expanded {count} time(s), had downstream use on {used_events} expansion event(s), "
                    f"produced {used_rows} expanded-tool call row(s); "
                    f"est. round-trip savings={cost_value['saved_round_trip_tokens']:,} tok, "
                    f"est. carry cost={cost_value['added_carry_tokens']:,} tok, "
                    f"net={cost_value['net_value_tokens']:+,} tok under "
                    f"per_tool={args.per_tool_tokens}, round_trip={per_event} ({per_event_basis})"
                ),
            })

        for tool, carried in stat.carry_turns.most_common():
            if tool in immutable:
                continue
            calls = stat.tool_calls.get(tool, 0)
            if carried >= args.unused_carry_turns and calls == 0:
                rows.append({
                    "scope": scope,
                    "kind": "carry_tool",
                    "item": tool,
                    "action": "demote_to_expand_only",
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
                                "expand_only": [tool],
                            }
                        }
                    },
                    "reason": (
                        f"carried as a resident for {carried} prediction turn(s) with no logged calls; "
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
                # Trigger quality is advisory only. The analyzer never edits
                # trigger groups or writes trigger adjustments into learned
                # state (trigger definitions are immutable under the carrying
                # model), so this recommendation proposes no learned patch.
                "proposed_learned_patch": {"scopes": {}},
                "reason": (
                    f"trigger fired {fires} time(s); same-prediction precision estimate {precision:.2f}; "
                    f"same-scope recall estimate {recall:.2f}. "
                    "These are conservative proxies: a row may lack trigger_tools_by_group."
                ),
            })
    return rows


def summary_payload(
    stats: dict[str, ScopeStats],
    recs: list[dict[str, Any]],
    args: argparse.Namespace,
    dampeners: list[dict[str, Any]] | None = None,
    trigger_keywords: list[dict[str, Any]] | None = None,
    cache_cost: dict[str, dict[str, Any]] | None = None,
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
    # Each scope's events at that scope's measured per-event cost.
    total_expand_cost = sum(
        stat.expansion_events * expand_round_trip_tokens_for(args, scope)
        for scope, stat in stats.items()
    )
    total_first_turns = sum(stat.first_turns for stat in stats.values())
    total_mutation_turns = sum(stat.mutation_turns for stat in stats.values())
    total_stable_turns = sum(stat.stable_turns for stat in stats.values())
    total_hash_missing = sum(stat.hash_missing_turns for stat in stats.values())
    total_comparable = total_mutation_turns + total_stable_turns
    # Union before counting: one session can appear under more than one scope
    # (a cron turn and a gateway turn share a session id), and summing the
    # per-scope set sizes would count it once per scope.
    all_sessions: set[str] = set()
    mutated_sessions: set[str] = set()
    for stat in stats.values():
        all_sessions |= stat.sessions
        mutated_sessions |= stat.sessions_with_mutation
    total_sessions_with_mutation = len(mutated_sessions)
    total_sessions = len(all_sessions)
    payload: dict[str, Any] = {
        "version": ANALYZER_PAYLOAD_VERSION,
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
    # Parity with the markdown report's "Cache-Aware Savings" section: machine
    # consumers get the same corrected net the human report shows.
    if cache_cost:
        payload["cache_cost"] = {
            "totals": {
                "cache_read_lost_upper_bound": _cache_cost_total(
                    cache_cost, "cache_read_lost_upper_bound", 0),
                "est_usd_lost_upper_bound": round(_cache_cost_total(
                    cache_cost, "est_usd_lost_upper_bound", 0.0), 4),
            },
            "scopes": cache_cost,
        }
    return payload


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
    immutable = preset_always_carry(Path(__file__).resolve().parent)
    used_events = sum(len(event_ids) for event_ids in stat.used_expansion_event_ids_by_category.values())
    per_event = expand_round_trip_tokens_for(args, stat.scope)
    measured = expand_overhead_for_scope(args, stat.scope)
    expand_round_trip_total = stat.expansion_events * per_event
    net_savings = stat.total_tokens_saved - expand_round_trip_total
    return {
        "predictions": stat.predictions,
        "sessions": len(stat.sessions),
        "logged_first_call_tokens_saved": stat.total_tokens_saved,
        "expand_round_trip_tokens": per_event,
        "expand_round_trip_basis": expand_round_trip_basis_for(args, stat.scope),
        "expand_overhead": measured.to_json() if measured is not None else None,
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
        "estimated_carry_cost_tokens": {
            tool: carrying_cost(turns, args.per_tool_tokens)
            for tool, turns in stat.carry_turns.items()
            if tool not in immutable
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


_CACHE_STABILITY_MOD = None


def _load_cache_stability_module():
    """Load scripts/cache-stability-replay.py via importlib (hyphenated filename)."""
    global _CACHE_STABILITY_MOD
    if _CACHE_STABILITY_MOD is not None:
        return _CACHE_STABILITY_MOD
    script = Path(__file__).resolve().parent / "scripts" / "cache-stability-replay.py"
    if not script.exists():
        return None
    spec = importlib.util.spec_from_file_location("cache_stability_replay", script)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CACHE_STABILITY_MOD = mod
    return mod


def compute_cache_cost(
    predictions: list[dict[str, Any]],
    api_calls: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    scopes: list[str],
) -> dict[str, dict[str, Any]]:
    """Run the cache-stability replay per scope and return a scope→result map.

    Each value has:
      - stability: stability_simulation() output
      - counterfactual: matched_counterfactual() output
    Returns an empty dict if the module or api_calls are unavailable.
    """
    mod = _load_cache_stability_module()
    if mod is None or not api_calls:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for scope in scopes:
        stability = mod.stability_simulation(predictions, api_calls, tool_calls=tool_calls, scope_filter=scope)
        cf = mod.matched_counterfactual(api_calls, scope_filter=scope)
        result[scope] = {"stability": stability, "counterfactual": cf}
    return result


def _cache_cost_total(cache_cost: dict[str, dict[str, Any]], key: str, default: Any = 0) -> Any:
    """Sum one per-model counterfactual field across all scopes and models."""
    total = default
    for scope_data in cache_cost.values():
        for model_data in scope_data.get("counterfactual", {}).get("per_model", {}).values():
            total += model_data.get(key, default)
    return total


def _per_event_label(stats: dict[str, ScopeStats], args: argparse.Namespace) -> str:
    """Human label for the per-event cost basis across the reported scopes."""
    parts = []
    for scope in stats:
        parts.append(
            f"{scope} @ {expand_round_trip_tokens_for(args, scope):,}/event"
            f" {expand_round_trip_basis_for(args, scope)}"
        )
    if not parts:
        return f"@ {args.expand_round_trip_tokens:,}/event fallback"
    return "; ".join(parts)


def format_summary(
    stats: dict[str, ScopeStats],
    recs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    # The text summary renders only ``payload["totals"]``, which dampener
    # candidates do not affect, so none are passed through.
    payload = summary_payload(stats, recs, args)
    totals = payload["totals"]
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
        f"({_per_event_label(stats, args)})",
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
    immutable = preset_always_carry(Path(__file__).resolve().parent)

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
        total_usd_lost = round(_cache_cost_total(cache_cost, "est_usd_lost_upper_bound", 0.0), 4)
        total_tokens_lost = _cache_cost_total(cache_cost, "cache_read_lost_upper_bound", 0)
        net_savings = totals["estimated_net_savings_tokens"]
        lines.extend([
            "## Cache-Aware Savings (matched counterfactual)",
            "",
            "Per-call `cache_read_tokens` are captured in `api_calls.jsonl` on every API response. "
            "The cache-stability replay matches each mutated call against the stable cohort at the same "
            "`api_call_idx` position within the session, computing the counterfactual cache reads "
            "that were lost to tool-list mutations. Dollar estimates use per-model list prices.",
            "",
            f"- Cache read tokens lost to mutations (upper bound): **{total_tokens_lost:,}**",
            f"- Estimated USD cost of those lost cache reads: **${total_usd_lost:.4f}**",
            f"- Schema-savings net (from Summary above): **{net_savings:+,}** tokens",
            "- Cache-corrected net = schema savings − cache-miss penalty; the two are in tokens but "
            "priced at different rates, so they are not directly comparable.",
            "",
        ])
        for scope, scope_data in cache_cost.items():
            stability = scope_data["stability"]
            cf = scope_data["counterfactual"]
            lines.extend([
                f"### {scope}",
                "",
                f"- Comparable calls: **{stability['comparable_calls']:,}**",
                f"- Matches first-call hash: **{stability['matches_stable']}** (avg cache_read: {stability['avg_cache_read_when_matches']:,.0f})",
                f"- Expand-driven mutations: **{stability['expand_driven_mutations']}** (accepted — model requested new tools)",
                # Printed so matches + the three mutation classes reconcile to
                # comparable_calls, as the standalone replay report does.
                f"- Trigger-driven mutations: **{stability.get('trigger_driven_mutations', 0)}** (accepted — trigger activation)",
                f"- Would-break mutations: **{stability['would_break_mutations']}** (avg cache_read: {stability['avg_cache_read_when_would_break']:,.0f})",
                f"- Unexplained share of mutations: **{stability['unexplained_mutation_share'] * 100:.1f}%**",
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
            "> ⚠ No `api_calls.jsonl` data available — cache cost not computed.",
            "",
        ])

    lines.extend([
        "",
        "## Adjustable Defaults Used",
        "",
        f"- Per-tool token estimate: `{args.per_tool_tokens}` tokens/API call",
        f"- Expand round-trip fallback: `{args.expand_round_trip_tokens}` tokens/event "
        f"(charged only where a scope's telemetry is too thin to measure; "
        f"per scope: {_per_event_label(stats, args)})",
        f"- Minimum expansions before promotion review: `{args.min_expansions}`",
        f"- Promotion expand-rate default: `{args.promote_expand_rate:.2f}`",
        f"- Promotion use-rate default: `{args.promote_use_rate:.2f}`",
        f"- Unused-carry demotion floor: `{args.unused_carry_turns}` carried turns",
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
        lines.extend(["", "Carry-cost / unused demote-to-expand_only candidates:"])
        unused = [
            (tool, carried)
            for tool, carried in stat.carry_turns.most_common()
            if tool not in immutable and stat.tool_calls.get(tool, 0) == 0 and carried >= args.unused_carry_turns
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
        lines.extend(["", "## Harvest-Driven Per-Tool Promote-to-Carry Candidates", ""])
        lines.append(
            "Derived from `policy_source: harvest` rows where the historical "
            "model called a tool the current partition would have left "
            "expand_only. Each such call is a counterfactual `expand_tools` "
            "round-trip. Action values: `promote_to_carry` (net token savings "
            "positive), `keep_expand_only` (either frequently reached for but "
            "per-tool carry would lose net — broaden the activating trigger "
            "instead — or sparse)."
        )
        lines.append("")
        for r in harvest_recs:
            m = r["metrics"]
            lines.append(
                f"### {r['scope']} / `{r['item']}` — **{r['action']}** "
                f"(expand_only_calls={m['harvest_was_expand_only']}, "
                f"expand_only_rate={m['expand_only_rate']:.1%}, "
                f"net={m['net_savings_tokens']:+,} tok)"
            )
            lines.append("")
            lines.append(f"- {r['reason']}")
            lines.append("")

    if trigger_keywords:
        lines.extend(["", "## Suggested Trigger-Keyword Candidates", ""])
        lines.append(
            "Mined from harvest messages where the model called a tool the "
            "current partition would have left expand_only. Symmetric inverse of "
            "the dampener flow — these add coverage, dampeners remove false "
            "fires. Review high-precision candidates before pasting into "
            "the target trigger group's `keywords` list."
        )
        # Same degraded-mode callout the dampener section renders: without a
        # YAML loader the existing-pattern count is 0 because nothing could be
        # read, not because no patterns exist.
        statuses = {row.get("preset_triggers_status", "ok") for row in trigger_keywords}
        for status in sorted(s for s in statuses if s and s != "ok"):
            note = _EXCLUDES_STATUS_MESSAGE.get(
                status, f"preset triggers unavailable: {status}"
            )
            lines.append("")
            lines.append(f"> ⚠ Degraded mode (`{status}`): {note}")
        lines.append("")
        for row in trigger_keywords:
            status = row.get("preset_triggers_status", "ok")
            status_suffix = "" if status == "ok" else f", triggers_status={status}"
            lines.append(
                f"### {row['scope']} / `{row['tool']}` → **{row['action']}** "
                f"`{row['target_trigger']}` (expand_only_calls={row['expand_only_count']}, "
                f"existing patterns={row['existing_keyword_pattern_count']}{status_suffix})"
            )
            lines.append("")
            for cand in row.get("candidates", []):
                lines.append(
                    f"- pattern `{cand['pattern']}` → regex `{cand['suggested_regex']}`  "
                    f"(precision {cand['precision']:.2f}, "
                    f"support={cand['support_count']}, noise={cand['noise_count']})"
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
        "version": ANALYZER_PAYLOAD_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "recommend",
        "safety": {
            "writes_learned_json": False,
            "applies_policy": False,
            "default_behavior": "safe: recommend mode does not merge learned.json; only learned_mode: apply does",
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
            "FALLBACK per-event cost of one expand_tools round-trip, charged "
            "only for scopes whose telemetry is too thin to measure the real "
            "cost (a prefix-cache break on caching providers; one extra "
            "full-price call otherwise). Measured figures always win."
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
    parser.add_argument("--dampener-max-candidates", type=int, default=DEFAULT_DAMPENER_MAX_CANDIDATES,
                        help="Max candidates surfaced per (scope, trigger)")
    parser.add_argument("--harvest-min-expand-only-calls", type=int,
        default=DEFAULT_HARVEST_MIN_EXPAND_ONLY_CALLS,
        help="Minimum was_expand_only call count before a tool surfaces as a "
             "harvest-driven promote-to-carry candidate. Filters thin-signal "
             "noise. (default: 3)")
    parser.add_argument("--suggest-trigger-keywords", action="store_true",
        help="Mine expand_only-tool message previews for candidate trigger "
             "keywords. Symmetric inverse of --suggest-dampeners. Requires "
             "harvest data.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions_paths = telemetry_paths(args.state_dir, "predictions.jsonl", args.include_archives)
    tool_calls_paths = telemetry_paths(args.state_dir, "tool_calls.jsonl", args.include_archives)
    # Read every row through the central adapter so all downstream consumers
    # (collect_stats, cache-cost, source counts) see canonical carrying-model
    # fields regardless of the on-disk schema version.
    predictions = [logger_io.normalize_prediction_row(r) for r in load_rows(predictions_paths)]
    tool_calls = [logger_io.normalize_tool_call_row(r) for r in load_rows(tool_calls_paths)]
    stats = collect_stats(predictions, tool_calls)

    # Load api_calls.jsonl for cache-cost analysis (cache_read_tokens per call)
    # and for the per-scope MEASURED expand_tools cost every net-value figure
    # below charges (the --expand-round-trip-tokens flag is only its fallback).
    api_calls_paths = telemetry_paths(args.state_dir, "api_calls.jsonl", args.include_archives)
    api_calls = load_rows(api_calls_paths)
    args.expand_overhead_by_scope = {
        scope: _savings.measure_expand_overhead(
            predictions, api_calls, tool_calls, scope=scope,
            fallback_per_event=args.expand_round_trip_tokens,
        )
        for scope in stats
    }

    immutable_always_carry = preset_always_carry(Path(__file__).resolve().parent)
    recs = recommendation_rows(stats, args, immutable_always_carry) + harvest_recommendation_rows(stats, args, immutable_always_carry)
    dampeners = dampener_candidates(stats, args)
    trigger_keywords = trigger_keyword_candidates(stats, args)

    cache_cost = compute_cache_cost(predictions, api_calls, tool_calls, list(stats.keys())) if api_calls else None

    # Break down tool_calls by source so the headline counts don't conflate
    # narrowed gateway calls with cron / subagent calls that bypass narrowing.
    # A row that carries an unrecognized non-empty `source` (a future writer
    # value) is bucketed as "other" rather than re-derived — re-deriving would
    # silently relabel a value the writer actually recorded. Only rows with no
    # source at all fall through to the session/prediction heuristic.
    source_counts = {"gateway": 0, "cron": 0, "subagent": 0, "other": 0}
    for row in tool_calls:
        src = str(row.get("source") or "").strip()
        if not src:
            sid = str(row.get("session_id") or "")
            pid = str(row.get("prediction_id") or "")
            src = "cron" if sid.startswith("cron_") else ("gateway" if pid else "subagent")
        elif src not in source_counts:
            src = "other"
        source_counts[src] += 1

    if args.format == "json":
        payload = summary_payload(stats, recs, args, dampeners, trigger_keywords,
                                  cache_cost=cache_cost)
        payload["totals"]["tool_call_source_counts"] = source_counts
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_summary(stats, recs, args))
        excluded = source_counts["cron"] + source_counts["subagent"] + source_counts["other"]
        if excluded:
            parts = [
                f"{source_counts['gateway']} narrowed (gateway)",
                f"{source_counts['cron']} excluded (cron)",
                f"{source_counts['subagent']} excluded (subagent)",
            ]
            if source_counts["other"]:
                parts.append(f"{source_counts['other']} unclassified (other)")
            print("  └─ " + ", ".join(parts))

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
