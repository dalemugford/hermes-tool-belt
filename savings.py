"""Canonical read-only savings engine — Tool Belt 1.0.

One engine, two separately-labeled cohorts that are **never** summed together:

  * **Observed** — realized savings computed from organic Tool Belt telemetry
    (``predictions.jsonl`` / ``api_calls.jsonl`` / ``tool_calls.jsonl``).
    Provider-returned usage is authoritative. This is what actually happened.

  * **Projected** — a counterfactual replay of historical Hermes sessions
    through either the *current effective* carrying assignments or a
    *caller-supplied proposed* per-scope assignment (used by ``tool-belt
    configure`` before anything is applied). Every projected figure is labeled
    counterfactual until matched by organic post-apply telemetry.

This module is the single home for:

  * the per-model USD price table + cost classifier (``PRICE_TABLE`` /
    ``price_for`` / ``classify_cost``) — ``scripts/cache-freeze-replay.py``
    imports the table from here rather than defining its own;
  * the explicit-expansion overhead constant (``EXPAND_ROUND_TRIP_TOKENS``);
  * full-definition schema tokenization (``schema_tokens``) built on
    ``logger_io.estimate_tokens`` — the one token estimator in the codebase;
  * agent/scope discovery for the public ``tool-belt savings`` command;
  * the observed-cohort math that ``scripts/savings-report.py`` re-exports as a
    thin backward-compatibility wrapper.

The engine performs **no writes** — not to config, learned state, telemetry, or
sessions. It is safe to run against a live Hermes home.

Import styles
-------------
Imported as ``tool_belt_plugin.savings`` (the normal path, via the package)
*and* standalone as ``savings`` (how ``scripts/cache-freeze-replay.py`` pulls in
the price table after inserting the plugin dir on ``sys.path``). The lightweight
pricing/estimator surface loads in both contexts; the heavier projection path
lazily imports ``presets`` / ``predictor`` / ``learned`` only when a projection
is actually requested, so a standalone ``from savings import price_for`` never
drags the package-relative modules in.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:  # pragma: no cover - typing only; runtime import stays lazy
    from . import presets

# The token estimator is single-sourced in logger_io; import it in a way that
# works whether we were loaded as a package submodule or standalone.
try:  # package context
    from . import logger_io as logger_io
except ImportError:  # pragma: no cover - standalone load (plugin dir on path)
    import logger_io  # type: ignore[no-redefine]


# ─── Canonical constants ──────────────────────────────────────────────────────

#: Estimated total token cost of one explicit ``expand_tools`` round-trip: the
#: model's tool-call output, the tool result, and the extra API request carrying
#: the widened schemas. The single canonical estimator for expansion overhead;
#: ``analyze.py`` exposes it as the ``--expand-round-trip-tokens`` default and
#: ``scripts/savings-report.py`` reads it from here. Trigger *activation* is not
#: an ``expand_tools`` round trip and is never charged this cost.
EXPAND_ROUND_TRIP_TOKENS = 1500

#: Single source of truth for per-model token economics. Tokens-per-million in
#: USD; ``miss_premium`` is the input/cache_read ratio the cache correction
#: uses. Unknown models fall back to ``generic``. A price row is necessary but
#: NOT sufficient to show dollars — see :func:`classify_cost`: a public list
#: price does not turn an OAuth/subscription route into known variable costing.
PRICE_TABLE: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-sonnet-4-6": {"input": 3.00, "cache_read": 0.30, "cache_write": 3.75, "output": 15.00, "miss_premium": 10.0},
    "claude-haiku-4-5-20251001": {"input": 1.00, "cache_read": 0.10, "cache_write": 1.25, "output": 5.00, "miss_premium": 10.0},
    # OpenAI Codex (list prices for ratio computation)
    "gpt-5.4": {"input": 1.25, "cache_read": 0.125, "cache_write": 1.25, "output": 10.00, "miss_premium": 10.0},
    "gpt-5.4-mini": {"input": 0.15, "cache_read": 0.075, "cache_write": 0.15, "output": 0.60, "miss_premium": 2.0},
    "gpt-5.5": {"input": 2.50, "cache_read": 0.25, "cache_write": 2.50, "output": 10.00, "miss_premium": 10.0},
    # Kimi (Ollama Cloud) — flat cloud route, no provider-side prefix caching
    "kimi-k2.6:cloud": {"input": 0.0, "cache_read": 0.0, "cache_write": 0.0, "output": 0.0, "miss_premium": 1.0},
    "generic": {"input": 1.0, "cache_read": 0.1, "cache_write": 1.0, "output": 5.0, "miss_premium": 10.0},
}

#: Dated tag identifying the rate table above, surfaced in results so a
#: dollar figure always carries the rate basis it was computed against.
PRICE_TABLE_RATE_BASIS = "list-2026-05"


def price_for(model: str) -> dict[str, float]:
    """Return the price row for ``model``, falling back to ``generic``."""
    return PRICE_TABLE.get(model, PRICE_TABLE["generic"])


# ─── Cost classification ──────────────────────────────────────────────────────
#
# Three routes, per the 1.0 methodology:
#   known        — variable/metered pricing is deterministically known; show
#                  estimated USD against the dated rate table and name the rate.
#   subscription — a flat/subscription route is known; never show dollars.
#   unknown      — the billing route or pricing is not provable; never dollars.
#
# The distinction is the *route*, not the existence of a list price. An API-key
# metered path is `known`; the same model over an OAuth/subscription route is
# `subscription`. When the telemetry doesn't prove the route we stay `unknown`.

_METERED_API_MODES = frozenset({"api", "api_key", "apikey", "metered", "pay_per_token"})
_SUBSCRIPTION_API_MODES = frozenset(
    {"oauth", "subscription", "chatgpt", "claude_max", "claude_pro", "claude_code", "flat"}
)
#: Models whose only route is a known flat/free plan (dollars never apply).
_SUBSCRIPTION_MODELS = frozenset({"kimi-k2.6:cloud"})


@dataclass
class CostClass:
    """How a (model, provider, route) row may be costed."""

    cost_class: str          # "known" | "subscription" | "unknown"
    model: str
    provider: str = ""
    api_mode: str = ""
    rate_basis: str = ""     # dated rate tag when known
    reason: str = ""

    @property
    def dollars_allowed(self) -> bool:
        return self.cost_class == "known"


def classify_cost(model: str, provider: str = "", api_mode: str = "") -> CostClass:
    """Classify a model/provider/route row into known/subscription/unknown.

    ``known`` requires an explicitly metered API route *and* a non-generic price
    row. A public list price alone is never enough — an OAuth/subscription route
    stays ``subscription`` even for a model that appears in ``PRICE_TABLE``.

    ``api_mode`` must name a *billing* route. Producers that record a transport
    label there instead (``chat_completions``, ``codex_responses``, …) prove
    nothing about billing and are deliberately left ``unknown``: guessing a
    route from a transport name is how a report starts inventing dollars.
    """
    model = str(model or "")
    mode = str(api_mode or "").strip().lower()

    if mode in _METERED_API_MODES and model in PRICE_TABLE and model != "generic":
        return CostClass(
            cost_class="known", model=model, provider=provider, api_mode=mode,
            rate_basis=PRICE_TABLE_RATE_BASIS,
            reason=f"metered API route ({mode}); dated rate for {model}",
        )
    if mode in _SUBSCRIPTION_API_MODES or model in _SUBSCRIPTION_MODELS:
        return CostClass(
            cost_class="subscription", model=model, provider=provider, api_mode=mode,
            reason="flat/subscription route known; list price is not variable cost",
        )
    return CostClass(
        cost_class="unknown", model=model, provider=provider, api_mode=mode,
        reason="billing route not provable from telemetry; dollars suppressed",
    )


# ─── Token helpers ────────────────────────────────────────────────────────────


def schema_tokens(defs: Iterable[Any]) -> int:
    """Tokenize a list of *complete* tool definitions.

    ``defs`` is an iterable of full provider tool-definition objects — the exact
    ``session_meta.tools`` entries (OpenAI ``{type, function:{name, description,
    parameters}}`` or Anthropic ``{name, description, input_schema}``). We
    serialize and tokenize the whole definition, not a ``{"name": ...}``
    placeholder: descriptions and JSON-Schema parameter blocks dominate real
    tool-schema token cost, so names-only counts understate savings by an order
    of magnitude and are never a valid substitute.
    """
    return logger_io.estimate_tokens(list(defs))


def token_estimator_name() -> str:
    return logger_io.token_estimator_name()


# ─── Discovery ────────────────────────────────────────────────────────────────


def default_hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes"))


def default_state_dir(hermes_home: Path | None = None) -> Path:
    home = hermes_home or default_hermes_home()
    return Path(home) / "state" / "tool-belt"


@dataclass
class AgentLocation:
    """A discovered, currently-present Hermes agent profile."""

    agent: str
    profile_home: Path       # the profile root (== hermes_home for "default")
    state_dir: Path          # <profile_home>/state/tool-belt
    sessions_dir: Path       # <profile_home>/sessions


def _tool_belt_explicitly_disabled(profile_home: Path) -> bool:
    """Return True only when profile config explicitly disables Tool Belt.

    Missing or unreadable config remains discoverable: directory presence is the
    compatibility fallback. An explicit ``plugins.enabled`` exclusion or
    ``plugins.tool-belt.enabled: false`` prevents stale telemetry from reviving a
    disabled profile.
    """
    config_path = profile_home / "config.yaml"
    if not config_path.is_file():
        return False
    try:
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return False
    plugins = raw.get("plugins") if isinstance(raw, dict) else None
    if not isinstance(plugins, dict):
        return False
    enabled_plugins = plugins.get("enabled")
    if isinstance(enabled_plugins, list) and "tool-belt" not in {
        str(name) for name in enabled_plugins
    }:
        return True
    plugin_config = plugins.get("tool-belt")
    return isinstance(plugin_config, dict) and plugin_config.get("enabled") is False


def discover_agents(
    hermes_home: Path | None = None, agent_filter: str | None = None
) -> list[AgentLocation]:
    """Discover currently enabled/present agent profiles.

    An agent is included only when its profile is present on disk now and its
    config does not explicitly disable Tool Belt. Missing/unreadable config falls
    back to directory discovery for compatibility; an explicit disable or
    ``plugins.enabled`` exclusion prevents stale telemetry from reviving it.
    """
    home = Path(hermes_home or default_hermes_home())
    out: list[AgentLocation] = []

    def _present(profile_home: Path) -> bool:
        return (profile_home / "sessions").is_dir() or (
            profile_home / "state" / "tool-belt"
        ).is_dir()

    if not agent_filter or agent_filter == "default":
        if _present(home) and not _tool_belt_explicitly_disabled(home):
            out.append(
                AgentLocation(
                    agent="default",
                    profile_home=home,
                    state_dir=home / "state" / "tool-belt",
                    sessions_dir=home / "sessions",
                )
            )
    profiles_dir = home / "profiles"
    if profiles_dir.is_dir():
        for child in sorted(profiles_dir.iterdir()):
            if not child.is_dir() or child.name == "default":
                continue  # "default" under profiles/ is reserved by Hermes
            if agent_filter and child.name != agent_filter:
                continue
            if _present(child) and not _tool_belt_explicitly_disabled(child):
                out.append(
                    AgentLocation(
                        agent=child.name,
                        profile_home=child,
                        state_dir=child / "state" / "tool-belt",
                        sessions_dir=child / "sessions",
                    )
                )
    return out


class UnknownAgentError(ValueError):
    """Raised when ``--agent=NAME`` names an agent that isn't enabled/present."""


# ─── JSONL loading ────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


class InvalidSinceError(ValueError):
    """Raised when a ``--since`` value was supplied but could not be parsed.

    Silently degrading a malformed cutoff to "no filter" is the worst possible
    outcome: the caller asked for a window and would be shown the entire
    history as if it were that window. A supplied value must parse or fail.
    """


def parse_since(s: str | None) -> float:
    """Parse a ``--since`` cutoff into an epoch timestamp.

    An empty/absent value means "no cutoff" and returns ``0.0``. A supplied
    value that is neither an ISO-8601 datetime nor ``YYYY-MM-DD`` raises
    :class:`InvalidSinceError` — it is never downgraded to "no cutoff".
    """
    if not s:
        return 0.0
    try:
        return _dt.datetime.fromisoformat(s).timestamp()
    except Exception:
        pass
    try:
        return _dt.datetime.strptime(s, "%Y-%m-%d").timestamp()
    except Exception:
        raise InvalidSinceError(
            f"could not parse --since value {s!r}; expected YYYY-MM-DD or an "
            "ISO-8601 datetime"
        ) from None


def _session_key(row: dict[str, Any]) -> str:
    return str(row.get("hermes_session_id") or row.get("session_id") or "")


# ══════════════════════════════════════════════════════════════════════════════
#  OBSERVED cohort
# ══════════════════════════════════════════════════════════════════════════════


def last_api_call_by_prediction(
    api_calls: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """For each prediction_id, the last (highest api_call_idx) API call."""
    by_pred: dict[str, dict[str, Any]] = {}
    for row in api_calls:
        pid = str(row.get("prediction_id") or "")
        if not pid:
            continue
        idx = int(row.get("api_call_idx") or 0)
        cur = by_pred.get(pid)
        if cur is None or idx > int(cur.get("api_call_idx") or 0):
            by_pred[pid] = row
    return by_pred


def classify_prediction_mode(
    p: dict[str, Any], api_last: dict[str, dict[str, Any]]
) -> str:
    """Classify one prediction as bypass, cache-on, cache-off, or pending.

    Bypass (the A/B control) takes precedence so intentionally-unnarrowed rows
    never drag the savings figures down; then the last api_call's most-evolved
    ``cache_mode``; then a ``frozen_reuse`` implies cache-on.
    """
    if str(p.get("policy_source") or "") == "bypass":
        return "bypass"
    pid = str(p.get("prediction_id") or "")
    last = api_last.get(pid, {})
    mode = str(last.get("cache_mode") or "")
    if mode in ("on", "off", "pending"):
        return mode
    return "on" if p.get("frozen_reuse") else "pending"


def aggregate_api_call_totals(
    api_calls: list[dict[str, Any]], pred_ids: set[str]
) -> dict[str, int]:
    totals = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0, "n_calls": 0}
    for row in api_calls:
        pid = str(row.get("prediction_id") or "")
        if pid not in pred_ids:
            continue
        totals["input"] += int(row.get("input_tokens") or 0)
        totals["cache_read"] += int(row.get("cache_read_tokens") or 0)
        totals["cache_write"] += int(row.get("cache_write_tokens") or 0)
        totals["output"] += int(row.get("output_tokens") or 0)
        totals["n_calls"] += 1
    return totals


def cohort_stats(
    predictions: list[dict[str, Any]],
    api_last: dict[str, dict[str, Any]],
    api_calls: list[dict[str, Any]],
    mode_filter: str,
) -> dict[str, Any]:
    """Compute realized savings for one cache-mode cohort within a scope.

    ``mode_filter`` is ``"on"``, ``"off"``, ``"pending"``, or ``"bypass"``.
    This is the observed-savings math ``scripts/savings-report.py`` re-exports.
    """
    rows = [p for p in predictions if classify_prediction_mode(p, api_last) == mode_filter]
    if not rows:
        return {"n_predictions": 0, "n_sessions": 0}

    sessions = {k for k in (_session_key(p) for p in rows) if k}
    n_predictions = len(rows)
    n_sessions = len(sessions)
    ceiling_total = sum(int(p.get("ceiling_tokens") or 0) for p in rows)
    narrowed_total = sum(int(p.get("narrowed_tokens") or 0) for p in rows)
    saved_total = ceiling_total - narrowed_total
    # ``n_predictions`` is provably > 0 here: the empty-``rows`` early return above.
    ceiling_count_avg = sum(int(p.get("ceiling_count") or 0) for p in rows) / n_predictions
    narrowed_count_avg = sum(int(p.get("narrowed_count") or 0) for p in rows) / n_predictions
    reduction_pct = (saved_total / ceiling_total * 100) if ceiling_total else 0.0

    pred_ids = {str(p.get("prediction_id") or "") for p in rows if p.get("prediction_id")}
    api_totals = aggregate_api_call_totals(api_calls, pred_ids)

    out = {
        "n_predictions": n_predictions,
        "n_sessions": n_sessions,
        "ceiling_count_avg": ceiling_count_avg,
        "narrowed_count_avg": narrowed_count_avg,
        "ceiling_tokens_total": ceiling_total,
        "narrowed_tokens_total": narrowed_total,
        "saved_tokens_total": saved_total,
        "saved_tokens_per_turn_avg": (saved_total / n_predictions) if n_predictions else 0,
        "reduction_pct": reduction_pct,
        "api_input_tokens": api_totals["input"],
        "api_cache_read_tokens": api_totals["cache_read"],
        "api_cache_write_tokens": api_totals["cache_write"],
        "api_n_calls": api_totals["n_calls"],
    }
    denom = api_totals["input"] + api_totals["cache_read"] + api_totals["cache_write"]
    out["cache_hit_rate"] = (api_totals["cache_read"] / denom * 100) if denom else 0.0
    return out


def _count_expand_events(tool_calls: list[dict[str, Any]]) -> int:
    return sum(1 for t in tool_calls if t.get("tool_name") == "expand_tools")


@dataclass
class ObservedCohort:
    """Realized savings from organic telemetry (never summed with projected)."""

    label: str = "observed"
    n_predictions: int = 0
    n_sessions: int = 0
    # Realized schema-token reduction (Σ ceiling − narrowed across cache-on and
    # cache-off cohorts; bypass/pending excluded from the headline).
    realized_schema_token_reduction: int = 0
    # Observed explicit-expansion overhead: expand_tools events × the canonical
    # per-event estimator. Trigger activations are not counted.
    expansion_events: int = 0
    expansion_overhead: int = 0
    net_token_reduction: int = 0
    cache_on: dict[str, Any] = field(default_factory=dict)
    cache_off: dict[str, Any] = field(default_factory=dict)
    pending: dict[str, Any] = field(default_factory=dict)
    bypass: dict[str, Any] = field(default_factory=dict)
    token_estimator: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_predictions": self.n_predictions,
            "n_sessions": self.n_sessions,
            "realized_schema_token_reduction": self.realized_schema_token_reduction,
            "expansion_events": self.expansion_events,
            "expansion_overhead": self.expansion_overhead,
            "net_token_reduction": self.net_token_reduction,
            "cache_on": self.cache_on,
            "cache_off": self.cache_off,
            "pending": self.pending,
            "bypass": self.bypass,
            "token_estimator": self.token_estimator,
        }


def compute_observed(
    state_dir: Path,
    scopes: Iterable[str] | None = None,
    since_ts: float = 0.0,
) -> ObservedCohort:
    """Compute the observed cohort from a state dir's live telemetry.

    ``scopes`` restricts to a set of ``agent:platform`` scopes (None = all).
    Pure read: loads the three JSONL files and derives realized reduction.
    """
    scope_set = set(scopes) if scopes is not None else None
    predictions = [
        logger_io.normalize_prediction_row(p)
        for p in load_jsonl(state_dir / "predictions.jsonl")
        if float(p.get("ts") or 0) >= since_ts
    ]
    api_calls = [
        a for a in load_jsonl(state_dir / "api_calls.jsonl")
        if float(a.get("ts") or 0) >= since_ts
    ]
    tool_calls = [
        logger_io.normalize_tool_call_row(t)
        for t in load_jsonl(state_dir / "tool_calls.jsonl")
        if float(t.get("ts") or 0) >= since_ts
    ]

    if scope_set is not None:
        predictions = [p for p in predictions if str(p.get("scope") or "") in scope_set]
        tool_calls = [t for t in tool_calls if str(t.get("scope") or "") in scope_set]

    api_last = last_api_call_by_prediction(api_calls)
    on = cohort_stats(predictions, api_last, api_calls, "on")
    off = cohort_stats(predictions, api_last, api_calls, "off")
    pending = cohort_stats(predictions, api_last, api_calls, "pending")
    bypass = cohort_stats(predictions, api_last, api_calls, "bypass")

    # Realized reduction: the narrowed cohorts only (bypass never narrows;
    # pending is not yet classified). expand_tools overhead is observed, not
    # estimated per-turn — it applies to the cache-off net figure.
    realized = on.get("saved_tokens_total", 0) + off.get("saved_tokens_total", 0)
    expand_events = _count_expand_events(tool_calls)
    overhead = expand_events * EXPAND_ROUND_TRIP_TOKENS
    net = realized - overhead

    n_pred = on.get("n_predictions", 0) + off.get("n_predictions", 0)
    n_sess = len({
        k for k in (_session_key(p) for p in predictions
                    if classify_prediction_mode(p, api_last) in ("on", "off")) if k
    })

    return ObservedCohort(
        n_predictions=n_pred,
        n_sessions=n_sess,
        realized_schema_token_reduction=realized,
        expansion_events=expand_events,
        expansion_overhead=overhead,
        net_token_reduction=net,
        cache_on=on,
        cache_off=off,
        pending=pending,
        bypass=bypass,
        token_estimator=token_estimator_name(),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PROJECTED cohort — historical replay
# ══════════════════════════════════════════════════════════════════════════════


def _import_sibling(name: str):
    """Import a sibling plugin module in either package or standalone context."""
    pkg = __package__ or ""
    if pkg:
        return importlib.import_module(f"{pkg}.{name}")
    return importlib.import_module(name)


@dataclass
class HistoricalSession:
    """A parsed historical session, with **complete** tool definitions kept."""

    session_file: Path
    agent: str
    platform: str
    model: str
    provider: str
    api_mode: str
    tool_defs: dict[str, Any]          # name -> full provider tool definition
    tool_order: list[str]              # ceiling order as recorded
    turns: list[dict[str, Any]]        # user/assistant rows in order
    schemas_complete: bool = True      # every recorded entry has description + schema

    @property
    def scope(self) -> str:
        return f"{self.agent}:{self.platform}"


def _def_name(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return None
    if "function" in entry and isinstance(entry["function"], dict):
        name = entry["function"].get("name")
    else:
        name = entry.get("name")
    return name if isinstance(name, str) and name else None


def _definition_is_complete(entry: Any) -> bool:
    """Whether an entry carries a full provider tool definition."""
    if not isinstance(entry, dict):
        return False
    body = entry.get("function") if isinstance(entry.get("function"), dict) else entry
    name = body.get("name")
    description = body.get("description")
    schema = body.get("parameters")
    if not isinstance(schema, dict):
        schema = body.get("input_schema")
    return (
        isinstance(name, str) and bool(name)
        and isinstance(description, str)
        and isinstance(schema, dict)
    )


def parse_session_full(session_file: Path, agent: str) -> HistoricalSession | None:
    """Parse one session JSONL, preserving the **complete** tool definitions.

    Unlike the names-only harvest parse, this keeps every ``session_meta.tools``
    entry verbatim (both OpenAI ``{type, function}`` and Anthropic ``{name,
    input_schema}`` shapes) so the projection tokenizes real schema bytes.
    Returns None on an unusable session (no meta, no user turns, bad JSON).
    """
    try:
        lines = [json.loads(l)
                 for l in session_file.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    except Exception:
        return None
    meta = next((row for row in lines if row.get("role") == "session_meta"), None)
    if not meta:
        return None

    recorded_defs = meta.get("tools") or []
    if not isinstance(recorded_defs, list):
        recorded_defs = []
    schemas_complete = bool(recorded_defs) and all(
        _definition_is_complete(entry) for entry in recorded_defs
    )
    tool_defs: dict[str, Any] = {}
    tool_order: list[str] = []
    for entry in recorded_defs:
        name = _def_name(entry)
        if name is None:
            continue
        if name not in tool_defs:
            tool_defs[name] = entry
            tool_order.append(name)

    turns = [row for row in lines if row.get("role") in ("user", "assistant")]
    if not any(t.get("role") == "user" for t in turns):
        return None

    return HistoricalSession(
        session_file=session_file,
        agent=agent,
        platform=str(meta.get("platform") or "unknown"),
        model=str(meta.get("model") or ""),
        provider=str(meta.get("provider") or ""),
        api_mode=str(meta.get("api_mode") or meta.get("billing") or ""),
        tool_defs=tool_defs,
        tool_order=tool_order,
        turns=turns,
        schemas_complete=schemas_complete,
    )


def iter_session_files(sessions_dir: Path, since_ts: float = 0.0) -> Iterable[Path]:
    if not sessions_dir.is_dir():
        return []
    out: list[Path] = []
    for path in sorted(sessions_dir.glob("*.jsonl")):
        if since_ts and path.stat().st_mtime < since_ts:
            continue
        out.append(path)
    return out


def _message_text(row: dict[str, Any]) -> str:
    content = row.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("text") or block.get("content")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return ""


def _called_tools_for_turn(turns: list[dict[str, Any]], user_idx: int) -> list[str]:
    """Tool names the model called responding to the user turn at ``user_idx``."""
    called: list[str] = []
    for j in range(user_idx + 1, len(turns)):
        row = turns[j]
        if row.get("role") == "user":
            break
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


def _is_mcp(name: str) -> bool:
    return name.startswith("mcp__") or name.startswith("mcp_")


@dataclass
class ProjectedModelRow:
    """Per-(model, route) costing row within a projected cohort."""

    model: str
    provider: str
    cost_class: str
    reason: str
    rate_basis: str
    gross_schema_token_reduction: int
    expansion_overhead: int
    net_token_reduction: int
    estimated_usd_savings: float | None  # only for known variable cost

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "cost_class": self.cost_class,
            "reason": self.reason,
            "rate_basis": self.rate_basis,
            "gross_schema_token_reduction": self.gross_schema_token_reduction,
            "expansion_overhead": self.expansion_overhead,
            "net_token_reduction": self.net_token_reduction,
            "estimated_usd_savings": self.estimated_usd_savings,
        }


@dataclass
class ProjectedCohort:
    """Counterfactual replay savings (never summed with observed)."""

    label: str = "projected"
    counterfactual: bool = True
    cache_mode: str = "on"
    sessions_analyzed: int = 0
    user_turns_analyzed: int = 0
    gross_schema_token_reduction: int = 0
    expansion_events: int = 0
    estimated_expansion_overhead: int = 0
    net_token_reduction: int = 0
    ceiling_tokens_total: int = 0
    # Schema-only reduction pct — explicitly NOT the session-input percentage.
    schema_reduction_pct: float | None = None
    # Session-input denominator.
    input_token_denominator: int = 0
    denominator_source: str = "reconstructed"  # provider_reported | partial | reconstructed | none
    net_input_reduction_pct: float | None = None
    # Costing.
    models: list[ProjectedModelRow] = field(default_factory=list)
    estimated_usd_savings: float | None = None  # sum over known rows only
    usd_coverage: str = "none"  # none | partial | full
    confidence: str = "insufficient"  # high | medium | low | insufficient
    reasons: list[str] = field(default_factory=list)
    assignment_source: str = "current_effective"  # or "proposed"
    token_estimator: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "counterfactual": self.counterfactual,
            "cache_mode": self.cache_mode,
            "assignment_source": self.assignment_source,
            "sessions_analyzed": self.sessions_analyzed,
            "user_turns_analyzed": self.user_turns_analyzed,
            "gross_schema_token_reduction": self.gross_schema_token_reduction,
            "expansion_events": self.expansion_events,
            "estimated_expansion_overhead": self.estimated_expansion_overhead,
            "net_token_reduction": self.net_token_reduction,
            "ceiling_tokens_total": self.ceiling_tokens_total,
            "schema_reduction_pct": self.schema_reduction_pct,
            "input_token_denominator": self.input_token_denominator,
            "denominator_source": self.denominator_source,
            "net_input_reduction_pct": self.net_input_reduction_pct,
            "models": [m.to_json() for m in self.models],
            "estimated_usd_savings": self.estimated_usd_savings,
            "usd_coverage": self.usd_coverage,
            "confidence": self.confidence,
            "reasons": sorted(set(self.reasons)),
            "token_estimator": self.token_estimator,
        }


def _resolve_effective_preset(
    plugin_config: dict[str, Any],
    scope: str,
    proposed: dict[str, Any] | None,
):
    """Resolve the carrying preset for a scope, read-only.

    ``proposed`` (supplied by the configure flow) is a ``{carry: [...],
    expand_only: [...]}`` mapping to project *before* it is applied. Without
    it, the *current effective* assignment is used.

    **Both branches resolve through the same pipeline**
    (:func:`presets.resolve_preset`, which honors the applied learned overlay
    and per-channel config, read-only). The proposed deltas are layered on top
    of that resolved preset, so a ``current`` and a ``proposed`` projection of
    the same scope differ *only* by the proposed assignment — which is exactly
    the comparison the onboarding flow presents. Resolving the proposed branch
    against the raw base policy instead would silently drop the learned overlay
    and channel config and make the two projections non-comparable.
    """
    presets = _import_sibling("presets")
    base = presets.resolve_preset(plugin_config, scope)
    if proposed is None:
        return base

    if base.is_wildcard:
        return base
    always_carry = list(base.always_carry)
    always_carry_set = set(always_carry)
    policy_carry = list(base.carry)
    prop_carry = [str(t) for t in (proposed.get("carry") or [])]
    prop_expand = set(str(t) for t in (proposed.get("expand_only") or []))
    effective = [t for t in policy_carry if t not in prop_expand]
    for t in prop_carry:
        if t not in effective:
            effective.append(t)
    effective = [t for t in effective if t not in always_carry_set]
    return presets.Preset(
        name=f"{base.name}+proposed[{scope}]",
        always_carry=always_carry,
        carry=effective,
        triggers=base.triggers,
    )


def _replay_active_names(
    session: HistoricalSession,
    preset: presets.Preset,
    cache_mode: str,
) -> tuple[list[list[str]], list[int], int]:
    """Replay the predictor over each user turn.

    Returns ``(active_per_turn, user_turn_indices, expansion_events)`` where
    ``active_per_turn[k]`` is the resolved active tool-name list for the k-th
    user turn.

      * cache-off: the active set is resolved fresh every turn.
      * cache-on:  the active set is frozen at the first turn, then grows
        monotonically as triggers activate expand_only tools and as explicit
        expansions admit called-but-unloaded tools. A trigger activation is not
        an ``expand_tools`` round trip and carries no overhead; each explicit
        expansion event is charged once.

    MCP/passthrough tools are never narrowed — they stay active every turn.
    """
    predictor = _import_sibling("predictor")
    ceiling_names = list(session.tool_defs.keys())
    ceiling_set = set(ceiling_names)
    mcp_names = {n for n in ceiling_names if _is_mcp(n)}

    active_per_turn: list[list[str]] = []
    user_indices: list[int] = []
    expansion_events = 0
    frozen: set[str] | None = None

    for i, row in enumerate(session.turns):
        if row.get("role") != "user":
            continue
        message = _message_text(row)
        if not message.strip():
            continue
        prediction = predictor.predict(message, None, preset)

        if prediction.is_wildcard:
            per_turn = set(ceiling_names)
        else:
            resolved = set(prediction.active_tool_names) & ceiling_set
            per_turn = resolved | mcp_names  # passthrough never narrowed

        called = _called_tools_for_turn(session.turns, i)
        # Which triggers fired this turn -> the tools they activate (T).
        triggered_tools: set[str] = set()
        for group in getattr(preset, "triggers", []) or []:
            if group.name in set(prediction.triggers_fired):
                triggered_tools |= {t for t in group.tools if t in ceiling_set}

        if cache_mode == "on":
            if frozen is None:
                frozen = set(per_turn)
            # Monotonic trigger activation: a fired trigger's expand_only tool
            # joins the frozen set with no round-trip charge.
            frozen |= (triggered_tools & ceiling_set)
            # Explicit expansion: a called tool that is neither resident/active
            # nor trigger-activated must have been expanded. Charge once per
            # such expansion event; the tool then persists in the frozen set.
            for name in called:
                if name in ceiling_set and name not in frozen and name not in mcp_names:
                    expansion_events += 1
                    frozen.add(name)
            active = set(frozen)
        else:  # cache-off: per-turn resolution
            active = set(per_turn)
            for name in called:
                if (name in ceiling_set and name not in active
                        and name not in triggered_tools and not _is_mcp(name)):
                    expansion_events += 1
                    active.add(name)

        active_per_turn.append(sorted(active))
        user_indices.append(i)

    return active_per_turn, user_indices, expansion_events


def _reconstructed_input_denominator(
    session: HistoricalSession, user_indices: list[int]
) -> int:
    """Reconstruct cumulative request input tokens across a session's turns.

    Each user/API turn re-sends the accumulated context: the full tool
    definitions plus every prior message. We sum, per user turn, the tokens of
    the complete tool block and the conversation up to and including that turn.
    This models accumulated context growth — it is NOT one-pass transcript size,
    and it is labeled ``reconstructed`` with a downgraded confidence.
    """
    tools_tokens = schema_tokens(list(session.tool_defs.values()))
    total = 0
    for idx in user_indices:
        convo = session.turns[: idx + 1]
        convo_tokens = logger_io.estimate_tokens(
            [{"role": r.get("role"), "content": _message_text(r)} for r in convo]
        )
        total += tools_tokens + convo_tokens
    return total


def project_sessions(
    sessions: list[HistoricalSession],
    plugin_config: dict[str, Any],
    *,
    cache_mode: str = "on",
    proposed_by_scope: dict[str, dict[str, Any]] | None = None,
    provider_input_by_session: dict[str, int] | None = None,
    route_by_session: dict[str, SessionRoute] | None = None,
) -> ProjectedCohort:
    """Replay a set of historical sessions into one projected cohort.

    ``proposed_by_scope`` (supplied by the configure flow) carries
    not-yet-applied assignments per scope; when absent the current effective
    assignment is used.
    ``provider_input_by_session`` maps a session key to provider-reported
    ``input_tokens`` — when present for a session it is the authoritative
    denominator and wins over reconstruction.
    ``route_by_session`` maps a session key to the billing route its API calls
    were actually made on (see :func:`_api_call_evidence`). It is the
    production source of ``provider``/``api_mode`` for costing, because live
    Hermes ``session_meta`` records carry neither field. A value recorded in
    ``session_meta`` still wins as an explicit override.
    """
    proposed_by_scope = proposed_by_scope or {}
    provider_input_by_session = provider_input_by_session or {}
    route_by_session = route_by_session or {}

    gross = 0
    ceiling_total = 0
    expansion_events = 0
    turns_analyzed = 0
    sessions_analyzed = 0
    denom_provider = 0
    denom_reconstructed = 0
    have_provider = False
    have_reconstructed = False
    incomplete_schema = False
    route_from_api_calls = False
    route_conflicts = False

    # Per-model accumulation of gross/net for costing.
    per_model_gross: dict[tuple[str, str, str], int] = defaultdict(int)
    per_model_overhead: dict[tuple[str, str, str], int] = defaultdict(int)
    assignment_source = "proposed" if proposed_by_scope else "current_effective"

    for session in sessions:
        if not session.tool_defs or not session.schemas_complete:
            incomplete_schema = True
            continue
        scope = session.scope
        proposed = proposed_by_scope.get(scope)
        try:
            preset = _resolve_effective_preset(plugin_config, scope, proposed)
        except Exception:
            incomplete_schema = True
            continue

        active_per_turn, user_indices, exp_events = _replay_active_names(
            session, preset, cache_mode
        )
        if not user_indices:
            continue

        full_defs = list(session.tool_defs.values())
        ceiling_tok = schema_tokens(full_defs)
        session_gross = 0
        for active in active_per_turn:
            active_defs = [session.tool_defs[n] for n in active if n in session.tool_defs]
            narrowed_tok = schema_tokens(active_defs)
            session_gross += max(0, ceiling_tok - narrowed_tok)

        overhead = exp_events * EXPAND_ROUND_TRIP_TOKENS
        gross += session_gross
        ceiling_total += ceiling_tok * len(user_indices)
        expansion_events += exp_events
        turns_analyzed += len(user_indices)
        sessions_analyzed += 1

        skey = session.session_file.stem

        # Costing route. session_meta is an explicit override when it carries
        # the fields; otherwise the evidence comes from this session's api_calls
        # rows (live session_meta records never carry provider/api_mode). No
        # evidence at all leaves both fields empty -> classify_cost -> unknown
        # -> no dollars.
        route = route_by_session.get(skey)
        provider = session.provider or (route.provider if route else "")
        api_mode = session.api_mode or (route.api_mode if route else "")
        if route is not None:
            if route.conflicting:
                route_conflicts = True
            if (not session.provider and route.provider) or (
                not session.api_mode and route.api_mode
            ):
                route_from_api_calls = True

        key = (session.model or "generic", provider, api_mode)
        per_model_gross[key] += session_gross
        per_model_overhead[key] += overhead

        # Denominator: provider-reported wins per session.
        prov = provider_input_by_session.get(skey)
        if prov is not None:
            denom_provider += int(prov)
            have_provider = True
        else:
            denom_reconstructed += _reconstructed_input_denominator(session, user_indices)
            have_reconstructed = True

    cohort = ProjectedCohort(cache_mode=cache_mode, assignment_source=assignment_source)
    cohort.token_estimator = token_estimator_name()
    cohort.sessions_analyzed = sessions_analyzed
    cohort.user_turns_analyzed = turns_analyzed
    cohort.gross_schema_token_reduction = gross
    cohort.expansion_events = expansion_events
    cohort.estimated_expansion_overhead = expansion_events * EXPAND_ROUND_TRIP_TOKENS
    cohort.net_token_reduction = gross - cohort.estimated_expansion_overhead
    cohort.ceiling_tokens_total = ceiling_total
    cohort.schema_reduction_pct = (
        round(gross / ceiling_total * 100, 2) if ceiling_total else None
    )

    # Denominator resolution + confidence.
    reasons: list[str] = []
    if sessions_analyzed == 0:
        cohort.denominator_source = "none"
        cohort.confidence = "insufficient"
        if incomplete_schema:
            reasons.append("incomplete tool schemas in historical sessions")
        else:
            reasons.append("no replayable historical sessions found")
        cohort.reasons = reasons
        cohort.net_input_reduction_pct = None
        return cohort

    # Denominator discipline: the reconstruction models only tool schemas +
    # conversation text per user turn. It omits the system prompt, context
    # injections, tool results, and per-API-call accumulation, so against
    # provider billing it can undercount input by orders of magnitude. It is
    # therefore NEVER used for the session-input percentage.
    #   * provider_reported — the only basis for net_input_reduction_pct.
    #   * partial          — provider data exists but does not cover every
    #                        replayed session; percentage suppressed (mixing
    #                        units would understate the denominator and
    #                        overstate the reduction).
    #   * none             — pre-install sessions only: the schema-only
    #                        percentage (schema_reduction_pct) is shown instead,
    #                        explicitly labeled.
    if have_provider and not have_reconstructed:
        cohort.denominator_source = "provider_reported"
        cohort.input_token_denominator = denom_provider
        cohort.confidence = "high"
        cohort.net_input_reduction_pct = round(
            cohort.net_token_reduction / denom_provider * 100, 2
        ) if denom_provider > 0 else None
        reasons.append("provider-reported input_tokens joined via predictions bridge")
        reasons.append("session-input percentage is provider-basis; numerator is per-user-turn (conservative)")
    elif have_provider and have_reconstructed:
        cohort.denominator_source = "partial"
        cohort.input_token_denominator = denom_provider
        cohort.confidence = "low"
        cohort.net_input_reduction_pct = None
        reasons.append(
            "provider denominator covers only part of the replay; "
            "session-input percentage suppressed (reconstruction omits tool "
            "results, system prompt, and per-call accumulation)"
        )
    else:
        cohort.denominator_source = "reconstructed"
        cohort.input_token_denominator = denom_reconstructed
        cohort.confidence = "low"
        cohort.net_input_reduction_pct = None
        reasons.append(
            "no provider-reported usage for these sessions; reconstructed "
            "input omits tool results/system prompt/per-call accumulation, so "
            "the session-input percentage is suppressed"
        )
        reasons.append("schema-only reduction shown as schema_reduction_pct; not the session-input %")

    if incomplete_schema:
        if cohort.confidence != "insufficient":
            cohort.confidence = "low"
            cohort.net_input_reduction_pct = None
        reasons.append("some sessions had incomplete schemas and were skipped")
        reasons.append("session-input percentage and USD suppressed for incomplete evidence")

    # Costing rows.
    if route_from_api_calls:
        reasons.append(
            "billing route (provider/api_mode) sourced from api_calls.jsonl; "
            "session_meta does not record it"
        )
    if route_conflicts:
        reasons.append(
            "some sessions' API calls disagreed on provider/api_mode; that "
            "route evidence was discarded (conservative) and those rows stay unknown"
        )
    known_usd_total = 0.0
    any_known = False
    any_non_known = False
    for key in sorted(per_model_gross.keys()):
        model, provider, api_mode = key
        cc = classify_cost(model, provider, api_mode)
        g = per_model_gross[key]
        oh = per_model_overhead[key]
        net = g - oh
        usd: float | None = None
        if cc.dollars_allowed:
            any_known = True
            if not incomplete_schema:
                prices = price_for(model)
                usd = round(max(0, net) * prices["input"] / 1_000_000.0, 4)
                known_usd_total += usd
        else:
            any_non_known = True
        cohort.models.append(ProjectedModelRow(
            model=model, provider=provider, cost_class=cc.cost_class,
            reason=cc.reason, rate_basis=cc.rate_basis,
            gross_schema_token_reduction=g, expansion_overhead=oh,
            net_token_reduction=net, estimated_usd_savings=usd,
        ))

    if incomplete_schema:
        cohort.usd_coverage = "none"
        cohort.estimated_usd_savings = None
    elif any_known and not any_non_known:
        cohort.usd_coverage = "full"
        cohort.estimated_usd_savings = round(known_usd_total, 4)
    elif any_known and any_non_known:
        cohort.usd_coverage = "partial"
        cohort.estimated_usd_savings = round(known_usd_total, 4)
        reasons.append("USD shown only for known-cost rows; token totals are complete")
    else:
        cohort.usd_coverage = "none"
        cohort.estimated_usd_savings = None
        reasons.append("no known variable-cost route; dollars suppressed")

    cohort.reasons = reasons
    return cohort


def compute_projected(
    location: AgentLocation,
    plugin_config: dict[str, Any],
    *,
    scopes: Iterable[str] | None = None,
    cache_mode: str = "on",
    since_ts: float = 0.0,
    proposed_by_scope: dict[str, dict[str, Any]] | None = None,
) -> ProjectedCohort:
    """Load an agent's historical sessions and project savings over them."""
    scope_set = set(scopes) if scopes is not None else None
    sessions: list[HistoricalSession] = []
    for path in iter_session_files(location.sessions_dir, since_ts):
        parsed = parse_session_full(path, location.agent)
        if parsed is None:
            continue
        if scope_set is not None and parsed.scope not in scope_set:
            continue
        sessions.append(parsed)

    # api_calls join: provider-reported input_tokens (denominator) and the
    # billing route each session was actually served on (costing evidence).
    provider_input, routes = _api_call_evidence(location.state_dir, since_ts)

    return project_sessions(
        sessions, plugin_config,
        cache_mode=cache_mode,
        proposed_by_scope=proposed_by_scope,
        provider_input_by_session=provider_input,
        route_by_session=routes,
    )


@dataclass(frozen=True)
class SessionRoute:
    """Billing-route evidence for one session, sourced from ``api_calls.jsonl``.

    Empty strings mean "no evidence" — never "not metered". A field is emptied
    when the session's api-call rows disagree about it (see
    :func:`_api_call_evidence`), which keeps :func:`classify_cost` at
    ``unknown`` and suppresses dollars.
    """

    provider: str = ""
    api_mode: str = ""
    conflicting: bool = False


def _api_call_session_keys(
    row: dict[str, Any], pid_to_hermes: dict[str, str]
) -> list[str]:
    """Session keys an ``api_calls`` row can be attributed to.

    ``api_calls`` rows carry the chat-level ``session_id`` (constant per chat),
    while historical session files are named by the rotating Hermes session
    UUID. The bridge is ``predictions.jsonl``: its rows carry both
    ``prediction_id`` and ``hermes_session_id``, so each api_call is joined
    prediction -> Hermes session under that file-stem key. ``session_file`` is
    a supported external-producer key. The raw chat ``session_id`` is the
    last-resort fallback when no bridge is available.
    """
    keys: list[str] = []
    hermes = pid_to_hermes.get(str(row.get("prediction_id") or ""))
    if hermes:
        keys.append(hermes)
    session_file = str(row.get("session_file") or "")
    if session_file:
        keys.append(session_file)
    if not keys:
        # No bridge available: fall back to the chat key so the denominator
        # still counts (it may aggregate across /new within one long-lived
        # chat; confidence labeling covers the imprecision).
        chat = str(row.get("session_id") or "")
        if chat:
            keys.append(chat)
    return keys


def _api_call_evidence(
    state_dir: Path, since_ts: float = 0.0
) -> tuple[dict[str, int], dict[str, SessionRoute]]:
    """Join ``api_calls.jsonl`` to sessions for denominator **and** route evidence.

    Returns ``(provider_input_by_session, route_by_session)``:

    * ``provider_input_by_session`` — summed provider-reported ``input_tokens``
      per session key (rows reporting zero input contribute nothing).
    * ``route_by_session`` — the ``provider`` / ``api_mode`` the calls for that
      session were actually billed through. These are the only fields that can
      prove a metered route, and live Hermes ``session_meta`` records never
      carry them; ``api_calls`` rows do (written by the ``post_api_request``
      hook).

    **Conservative conflict rule.** Each field is resolved independently and
    only when the session's rows agree: if a session's calls report more than
    one distinct non-empty ``provider`` (or ``api_mode``) — a mid-session
    failover or route change — that field is emptied and the route is flagged
    ``conflicting``. An empty ``api_mode`` cannot classify as ``known``, so a
    mixed-route session falls back to ``unknown`` and earns no dollars. We
    never pick the cheaper, the more common, or the first-seen value.
    """
    pid_to_hermes: dict[str, str] = {}
    for p in load_jsonl(state_dir / "predictions.jsonl"):
        pid = str(p.get("prediction_id") or "")
        hermes = str(p.get("hermes_session_id") or "")
        if pid and hermes:
            pid_to_hermes[pid] = hermes

    tokens_by_session: dict[str, int] = defaultdict(int)
    providers: dict[str, set[str]] = defaultdict(set)
    modes: dict[str, set[str]] = defaultdict(set)
    for a in load_jsonl(state_dir / "api_calls.jsonl"):
        if float(a.get("ts") or 0) < since_ts:
            continue
        keys = _api_call_session_keys(a, pid_to_hermes)
        if not keys:
            continue
        tokens = int(a.get("input_tokens") or 0)
        provider = str(a.get("provider") or "").strip()
        api_mode = str(a.get("api_mode") or "").strip().lower()
        for key in keys:
            if tokens:
                tokens_by_session[key] += tokens
            if provider:
                providers[key].add(provider)
            if api_mode:
                modes[key].add(api_mode)

    routes: dict[str, SessionRoute] = {}
    for key in set(providers) | set(modes):
        seen_providers = providers.get(key) or set()
        seen_modes = modes.get(key) or set()
        routes[key] = SessionRoute(
            provider=next(iter(seen_providers)) if len(seen_providers) == 1 else "",
            api_mode=next(iter(seen_modes)) if len(seen_modes) == 1 else "",
            conflicting=len(seen_providers) > 1 or len(seen_modes) > 1,
        )
    return dict(tokens_by_session), routes


def _provider_input_by_session(state_dir: Path, since_ts: float = 0.0) -> dict[str, int]:
    """Summed provider-reported ``input_tokens`` per session (denominator only)."""
    return _api_call_evidence(state_dir, since_ts)[0]


# ══════════════════════════════════════════════════════════════════════════════
#  Top-level report assembly
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class AgentSavings:
    agent: str
    platforms: list[str]
    observed: ObservedCohort
    projected: ProjectedCohort

    def to_json(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "platforms": sorted(self.platforms),
            "observed": self.observed.to_json(),
            "projected": self.projected.to_json(),
        }


@dataclass
class SavingsReport:
    generated_for: str            # "all" or an agent name
    cache_mode: str
    agents: list[AgentSavings]
    hermes_home: str
    token_estimator: str

    def to_json(self) -> dict[str, Any]:
        # Aggregate token totals are complete across every agent; USD sums only
        # over known-cost projected rows.
        agg_observed_realized = sum(a.observed.realized_schema_token_reduction for a in self.agents)
        agg_observed_net = sum(a.observed.net_token_reduction for a in self.agents)
        agg_proj_gross = sum(a.projected.gross_schema_token_reduction for a in self.agents)
        agg_proj_net = sum(a.projected.net_token_reduction for a in self.agents)
        known_usd = 0.0
        any_known = False
        any_non_known = False
        for a in self.agents:
            for m in a.projected.models:
                if m.cost_class == "known" and m.estimated_usd_savings is not None:
                    known_usd += m.estimated_usd_savings
                    any_known = True
                else:
                    any_non_known = True
        coverage = "none"
        usd: float | None = None
        if any_known and not any_non_known:
            coverage, usd = "full", round(known_usd, 4)
        elif any_known and any_non_known:
            coverage, usd = "partial", round(known_usd, 4)
        return {
            "schema": "tool-belt/savings/v1",
            "generated_for": self.generated_for,
            "cache_mode": self.cache_mode,
            "hermes_home": self.hermes_home,
            "token_estimator": self.token_estimator,
            "agents": [a.to_json() for a in self.agents],
            "aggregate": {
                "observed": {
                    "label": "observed",
                    "realized_schema_token_reduction": agg_observed_realized,
                    "net_token_reduction": agg_observed_net,
                },
                "projected": {
                    "label": "projected",
                    "counterfactual": True,
                    "gross_schema_token_reduction": agg_proj_gross,
                    "net_token_reduction": agg_proj_net,
                    "estimated_usd_savings": usd,
                    "usd_coverage": coverage,
                },
            },
        }


def _scopes_for_agent(location: AgentLocation, since_ts: float = 0.0) -> list[str]:
    """All ``agent:platform`` scopes for an agent, from observed telemetry AND
    historical sessions, so ``--agent`` reflects every platform it touched."""
    scopes: set[str] = set()
    for p in load_jsonl(location.state_dir / "predictions.jsonl"):
        if float(p.get("ts") or 0) < since_ts:
            continue
        scope = str(p.get("scope") or "")
        if scope:
            scopes.add(scope)
    for path in iter_session_files(location.sessions_dir, since_ts):
        parsed = parse_session_full(path, location.agent)
        if parsed is not None:
            scopes.add(parsed.scope)
    return sorted(scopes)


def compute(
    *,
    agent: str | None = None,
    hermes_home: Path | None = None,
    since: str | None = None,
    cache_mode: str = "on",
    plugin_config: dict[str, Any] | None = None,
    proposed_by_scope: dict[str, dict[str, Any]] | None = None,
) -> SavingsReport:
    """Build the full savings report — the one entry point the CLI + onboarding
    both call. Strictly read-only.

    ``agent`` restricts to a single enabled agent (raising
    :class:`UnknownAgentError` if it isn't present/enabled). ``proposed_by_scope``
    lets the configure flow project a not-yet-applied shape without writing any
    state.
    """
    home = Path(hermes_home or default_hermes_home())
    since_ts = parse_since(since)
    plugin_config = plugin_config if plugin_config is not None else {"enabled": True}

    locations = discover_agents(home, agent_filter=agent)
    if agent and not locations:
        raise UnknownAgentError(
            f"agent {agent!r} is not an enabled/discovered Hermes profile"
        )

    agents: list[AgentSavings] = []
    for loc in locations:
        scopes = _scopes_for_agent(loc, since_ts)
        observed = compute_observed(loc.state_dir, scopes=None, since_ts=since_ts)
        projected = compute_projected(
            loc, plugin_config, cache_mode=cache_mode, since_ts=since_ts,
            proposed_by_scope=proposed_by_scope,
        )
        agents.append(AgentSavings(
            agent=loc.agent,
            platforms=scopes,
            observed=observed,
            projected=projected,
        ))

    return SavingsReport(
        generated_for=agent or "all",
        cache_mode=cache_mode,
        agents=agents,
        hermes_home=str(home),
        token_estimator=token_estimator_name(),
    )
