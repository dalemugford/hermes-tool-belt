#!/usr/bin/env python3
"""Deterministic telemetry analyzer for the dynamic-tools plugin.

Reads append-only JSONL telemetry and emits a human-readable summary plus a
markdown report. The analyzer is intentionally conservative: it tolerates older
rows with missing fields, treats recommendations as suggestions, and keeps all
thresholds as adjustable defaults rather than permanent policy.
"""

from __future__ import annotations

import argparse
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

PROTECTED_ALWAYS_ON = {
    "memory",
    "session_search",
    "clarify",
    "skill_view",
    "skills_list",
    "todo",
    "send_message",
    "expand_tools",
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


def state_dir_from_env() -> Path:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(home) / "state" / "dynamic-tools"


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
        session_id = str(row.get("session_id") or "").strip()
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

        if row.get("expand_tools_used") is True:
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

    for prediction_id, groups in prediction_triggers.items():
        scope = prediction_scope.get(prediction_id, "unknown")
        stat = get(scope)
        called = prediction_tools_called.get(prediction_id, set())
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
    try:
        import yaml  # type: ignore[import-untyped]
    except Exception:
        return {}, "no_yaml"
    path = plugin_dir / "policy.yaml"
    if not path.exists():
        return {}, "no_policy"
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
        print(f"warning: dynamic-tools analyzer: {warning}", file=sys.stderr)
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


def recommendation_rows(stats: dict[str, ScopeStats], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
            if tool in PROTECTED_ALWAYS_ON:
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
        },
        "thresholds": threshold_payload(args),
        "scopes": {scope: scope_payload(stat, args) for scope, stat in stats.items()},
        "recommendations": recs,
        "dampener_candidates": list(dampeners or []),
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
            if tool not in PROTECTED_ALWAYS_ON
        },
        "cohorts": _cohort_payload(stat),
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


def format_summary(
    stats: dict[str, ScopeStats],
    recs: list[dict[str, Any]],
    args: argparse.Namespace,
    dampeners: list[dict[str, Any]] | None = None,
) -> str:
    payload = summary_payload(stats, recs, args, dampeners)
    totals = payload["totals"]
    lines = [
        "dynamic-tools analyzer",
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
    ]
    if args.include_archives:
        lines.append("archives: included by explicit flag")
    for scope, stat in stats.items():
        top_expansions = ", ".join(f"{k}={v}" for k, v in stat.expansions_by_category.most_common(3)) or "none"
        insufficient = sum(1 for rec in recs if rec.get("scope") == scope and rec.get("status") == "insufficient_data")
        lines.append(
            f"- {scope}: predictions={stat.predictions}, sessions={len(stat.sessions)}, "
            f"expansions={stat.expansion_events}, successful_expansions={sum(len(event_ids) for event_ids in stat.used_expansion_event_ids_by_category.values())}, "
            f"expanded_use_rows={stat.used_expansion_rows}, top_expansions={top_expansions}, insufficient_data={insufficient}"
        )
    return "\n".join(lines)


def markdown_report(
    stats: dict[str, ScopeStats],
    recs: list[dict[str, Any]],
    args: argparse.Namespace,
    dampeners: list[dict[str, Any]] | None = None,
) -> str:
    generated = time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())
    payload = summary_payload(stats, recs, args, dampeners)
    totals = payload["totals"]

    lines = [
        "# dynamic-tools Analyzer Report",
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
    ]

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
            if tool not in PROTECTED_ALWAYS_ON and stat.tool_calls.get(tool, 0) == 0 and carried >= args.unused_carry_turns
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
    parser = argparse.ArgumentParser(description="Analyze dynamic-tools JSONL telemetry")
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
    recs = recommendation_rows(stats, args)
    dampeners = dampener_candidates(stats, args)

    if args.format == "json":
        print(json.dumps(summary_payload(stats, recs, args, dampeners), indent=2, sort_keys=True))
    else:
        print(format_summary(stats, recs, args, dampeners))

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
        report_path.write_text(markdown_report(stats, recs, args, dampeners), encoding="utf-8")
        print(f"report: {report_path}", file=status_stream)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
