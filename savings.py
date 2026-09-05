"""Canonical read-only savings engine — Tool Belt 1.0.

One engine, two separately-labeled cohorts that are **never** summed together:

  * **Observed** — realized savings computed from organic Tool Belt telemetry
    (``predictions.jsonl`` / ``api_calls.jsonl`` / ``tool_calls.jsonl``).
    Provider-returned usage is authoritative. This is what actually happened.

  * **Projected** — a counterfactual replay of historical Hermes sessions
    through the *current effective* carrying assignments. Every projected
    figure is labeled counterfactual until matched by organic post-apply
    telemetry.

This module is the single home for:

  * the per-model USD price table + cost classifier (``PRICE_TABLE`` /
    ``price_for`` / ``classify_cost``) — ``cache_replay.py`` imports the
    table from here rather than defining its own;
  * cache-aware accounting: the priced-gross factor (``price_factor_for``),
    the measured per-cohort expansion cost (``measure_expand_overhead``) and
    its thin-data fallback constant (``EXPAND_ROUND_TRIP_TOKENS``);
  * full-definition schema tokenization (``schema_tokens``) built on
    ``logger_io.estimate_tokens`` — the one token estimator in the codebase;
  * agent/scope discovery for the public ``tool-belt savings`` command.

The engine performs **no writes** — not to config, learned state, telemetry, or
sessions. It is safe to run against a live Hermes home.

Import styles
-------------
Imported as ``tool_belt_plugin.savings`` (the normal path, via the package)
*and* standalone as ``savings`` (how ``cache_replay.py`` pulls in the price
table after inserting the plugin dir on ``sys.path``). The lightweight
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

#: FALLBACK per-event cost of one explicit ``expand_tools`` round-trip, used
#: only when telemetry is too thin to measure the real cost (see
#: :func:`measure_expand_overhead`). It is NOT the model: the measured cost is
#: a prefix-cache BREAK on caching providers (the whole history re-billed at
#: the input rate — tens of thousands of tokens per event) and one extra
#: full-price API call on non-caching providers. This constant is the
#: pre-measurement estimate (tool-call output + result + the widened schemas)
#: and is kept so a fresh install can still charge *something*. ``analyze.py``
#: exposes it as the ``--expand-round-trip-tokens`` fallback override and
#: ``shaping.py`` re-imports it as its thin-data penalty. Trigger *activation*
#: is not an ``expand_tools`` round trip and is never charged this cost.
EXPAND_ROUND_TRIP_TOKENS = 1500

#: Single source of truth for per-model token economics. Tokens-per-million in
#: USD; ``miss_premium`` is the input/cache_read ratio: ``1 / miss_premium`` is
#: the input-token-equivalent worth of a schema token that would have been a
#: cache read (see :func:`price_factor_for`). Unknown models fall back to
#: ``generic``. A price row is necessary but NOT sufficient to show dollars —
#: see :func:`classify_cost`: a public list price does not turn an
#: OAuth/subscription route into known variable costing.
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


#: Provider-level prompt-caching hints, consulted for api_calls rows without
#: a per-call ``provider_caches`` field. Keyed by the row's
#: ``provider`` string (lower-cased). ``False`` = the route never serves a
#: prefix-cache hit, whatever ``cache_mode`` the session was tagged with (the
#: session posture reflects the provider that locked it, so a fallback route's
#: calls can carry a misleading ``on``). Overridable: callers may pass their own
#: table to :func:`provider_caches_for_call`. A provider absent here has no
#: hint and falls through to the cohort posture.
PROVIDER_CACHE_HINTS: dict[str, bool] = {
    # Ollama Cloud: flat cloud route, 0% observed cache_read hits.
    "ollama-cloud": False,
    "ollama": False,
}


def price_factor_for(model: str, provider_caches: bool | None) -> float:
    """Input-token-equivalent worth of one saved schema token.

    On a caching provider the unsent schema tokens would have lived in the
    cached prefix and billed at the cache_read rate, so each one is worth
    ``1 / miss_premium`` input tokens (~0.1). On a non-caching provider — or
    when the cache status is unknown — every saved token was a full-price
    input token (factor 1.0).
    """
    if provider_caches is not True:
        return 1.0
    premium = float(price_for(model).get("miss_premium") or 0.0)
    return (1.0 / premium) if premium > 0 else 1.0


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


def _settings_block(raw):
    """``plugins.entries.tool-belt.settings`` out of a parsed config.yaml."""
    node = raw
    for part in ("plugins", "entries", "tool-belt", "settings"):
        node = node.get(part) if isinstance(node, dict) else None
    return node if isinstance(node, dict) else {}


def agent_display_name(profile_home: Path, fallback: str) -> str:
    """Human name for a profile in reports.

    Prefers the plugin config's own ``agent`` setting (the scope
    agent name, e.g. ``bernard`` on a root profile whose directory identity is
    ``default``); falls back to the profile name. Display-only — JSON output
    and telemetry keep the canonical profile name.
    """
    config_path = profile_home / "config.yaml"
    try:
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        name = _settings_block(raw).get("agent")
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception:
        pass
    return fallback


def _tool_belt_explicitly_disabled(profile_home: Path) -> bool:
    """Return True only when profile config explicitly disables Tool Belt.

    Missing or unreadable config remains discoverable: directory presence is the
    compatibility fallback. An explicit ``plugins.enabled`` exclusion or
    ``enabled: false`` prevents stale telemetry from reviving a
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
    return _settings_block(raw).get("enabled") is False


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


def index_api_calls_by_prediction(
    api_calls: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """All api_calls rows per prediction_id, in ``api_call_idx`` order."""
    by_pred: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in api_calls:
        pid = str(row.get("prediction_id") or "")
        if pid:
            by_pred[pid].append(row)
    for rows in by_pred.values():
        rows.sort(key=lambda r: int(r.get("api_call_idx") or 0))
    return dict(by_pred)


def provider_caches_for_call(
    row: dict[str, Any],
    hints: dict[str, bool] | None = None,
) -> bool | None:
    """Whether this one API call was served by a prompt-caching route.

    The per-call ``provider_caches`` field (True/False = that call's
    scope|provider detection bucket locked on/off; None while pending) is
    authoritative. Rows without it fall back to the
    provider hint table (:data:`PROVIDER_CACHE_HINTS`, or ``hints``); a
    provider with no hint returns None — "unknown", for the caller to resolve
    from the cohort posture. Never guessed from ``cache_mode``: that is the
    SESSION posture, and a fallback provider's calls inherit it wrongly.
    """
    explicit = row.get("provider_caches")
    if explicit is True or explicit is False:
        return explicit
    table = PROVIDER_CACHE_HINTS if hints is None else hints
    provider = str(row.get("provider") or "").strip().lower()
    if provider in table:
        return bool(table[provider])
    return None


def prediction_provider_caches(
    calls: list[dict[str, Any]] | None,
    cohort_mode: str | None = None,
    hints: dict[str, bool] | None = None,
) -> bool | None:
    """Resolve one prediction's cache status from its API calls.

    The last call's explicit/hinted status wins (it is the most-evolved
    route); otherwise any call with a known status; otherwise the cohort the
    prediction was classified into (``"on"`` → caching, ``"off"`` → not).
    Returns None only when nothing resolves (pending/bypass rows with no
    per-call facts) — such a prediction is priced at full input rate and
    excluded from overhead measurement.
    """
    calls = calls or []
    if calls:
        last = max(calls, key=lambda r: int(r.get("api_call_idx") or 0))
        status = provider_caches_for_call(last, hints)
        if status is not None:
            return status
        for row in calls:
            status = provider_caches_for_call(row, hints)
            if status is not None:
                return status
    if cohort_mode == "on":
        return True
    if cohort_mode == "off":
        return False
    return None


def cohort_stats(
    predictions: list[dict[str, Any]],
    api_last: dict[str, dict[str, Any]],
    api_calls: list[dict[str, Any]],
    mode_filter: str,
    calls_by_pred: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Compute realized savings for one cache-mode cohort within a scope.

    ``mode_filter`` is ``"on"``, ``"off"``, ``"pending"``, or ``"bypass"``.

    Two gross figures are emitted. ``saved_tokens_total`` is the RAW schema
    token reduction (Σ ceiling − narrowed), kept for continuity and as the
    basis of ``reduction_pct`` (a schema-size stat, not a cost stat).
    ``saved_input_equiv_total`` is the PRICED reduction: each prediction's
    saving × :func:`price_factor_for` (``1 / miss_premium`` on a caching
    route, 1.0 otherwise) — what the unsent tokens were actually worth.
    ``calls_by_pred`` is the :func:`index_api_calls_by_prediction` index;
    built from ``api_calls`` when not supplied.
    """
    rows = [p for p in predictions if classify_prediction_mode(p, api_last) == mode_filter]
    if not rows:
        return {"n_predictions": 0, "n_sessions": 0}
    if calls_by_pred is None:
        calls_by_pred = index_api_calls_by_prediction(api_calls)
    # Cohort fallback for rows with no per-call cache fact: only the locked
    # postures resolve; pending/bypass stay unknown (priced at full rate).
    cohort_mode = mode_filter if mode_filter in ("on", "off") else None

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

    saved_input_equiv = 0.0
    saved_input_equiv_caching = 0.0
    saved_input_equiv_noncaching = 0.0
    n_caching = 0
    n_noncaching = 0
    for p in rows:
        pid = str(p.get("prediction_id") or "")
        calls = calls_by_pred.get(pid, [])
        caches = prediction_provider_caches(calls, cohort_mode)
        model = str((api_last.get(pid) or {}).get("model") or "")
        saved = int(p.get("ceiling_tokens") or 0) - int(p.get("narrowed_tokens") or 0)
        priced = saved * price_factor_for(model, caches)
        saved_input_equiv += priced
        if caches is True:
            n_caching += 1
            saved_input_equiv_caching += priced
        elif caches is False:
            n_noncaching += 1
            saved_input_equiv_noncaching += priced
        else:
            # Unknown provider-cache status (pending/bypass) — priced at full
            # rate and treated as ongoing (non-caching) for the forward view.
            saved_input_equiv_noncaching += priced

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
        "saved_input_equiv_total": int(round(saved_input_equiv)),
        "saved_input_equiv_caching": int(round(saved_input_equiv_caching)),
        "saved_input_equiv_noncaching": int(round(saved_input_equiv_noncaching)),
        "n_predictions_caching": n_caching,
        "n_predictions_noncaching": n_noncaching,
        "reduction_pct": reduction_pct,
        "api_input_tokens": api_totals["input"],
        "api_cache_read_tokens": api_totals["cache_read"],
        "api_cache_write_tokens": api_totals["cache_write"],
        "api_n_calls": api_totals["n_calls"],
    }
    denom = api_totals["input"] + api_totals["cache_read"] + api_totals["cache_write"]
    out["cache_hit_rate"] = (api_totals["cache_read"] / denom * 100) if denom else 0.0
    return out


# ─── Measured expand_tools overhead ───────────────────────────────────────────
#
# What one explicit expansion actually costs, measured from the same telemetry
# the savings are computed from, per cache cohort:
#
#   caching     — an expand_tools call mutates the tool list mid-session and
#                 breaks the provider prefix cache; the next call re-bills the
#                 whole history at the input rate instead of cache_read. Cost
#                 per event = (excess cache-miss rate caused by expanding) ×
#                 (median re-billed premium on a missed expand turn).
#   non-caching — nothing to break; the cost is one extra full-price API call
#                 (the expand meta-call), i.e. its input_tokens.
#
# Thin data falls back to EXPAND_ROUND_TRIP_TOKENS and says so.

#: Calls with a prompt this small are cold-start/system-only and cannot tell a
#: cache break from an ordinary miss; they never count as a miss.
CACHE_MISS_MIN_PROMPT_TOKENS = 2000
#: Below these sample sizes the measured figures are noise; fall back.
MEASURED_OVERHEAD_MIN_EXPAND_EVENTS = 5
MEASURED_OVERHEAD_MIN_BASELINE_PREDICTIONS = 10


def _call_prompt_tokens(row: dict[str, Any]) -> int:
    """Provider-reported prompt size (cached + uncached); reconstructed from
    ``input_tokens + cache_read_tokens`` for rows without the field."""
    prompt = row.get("prompt_tokens")
    if prompt is not None:
        return int(prompt or 0)
    return int(row.get("input_tokens") or 0) + int(row.get("cache_read_tokens") or 0)


def _call_is_cache_miss(row: dict[str, Any]) -> bool:
    return (
        int(row.get("cache_read_tokens") or 0) == 0
        and _call_prompt_tokens(row) > CACHE_MISS_MIN_PROMPT_TOKENS
    )


def _rebill_premium(calls: list[dict[str, Any]]) -> float:
    """Tokens re-billed at the input rate by this prediction's cache misses.

    Σ over the prediction's MISSED calls of ``prompt_tokens × (input −
    cache_read) / input`` at the call model's rates — the part of the prompt
    that would have been a cache read had the prefix survived. Hit calls are
    not summed: their prompt was billed at cache_read as intended.
    """
    total = 0.0
    for row in calls:
        if not _call_is_cache_miss(row):
            continue
        prices = price_for(str(row.get("model") or ""))
        inp = float(prices.get("input") or 0.0)
        ratio = ((inp - float(prices.get("cache_read") or 0.0)) / inp) if inp > 0 else 0.0
        total += _call_prompt_tokens(row) * max(0.0, ratio)
    return total


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


@dataclass
class MeasuredExpandOverhead:
    """Per-cohort per-event cost of an explicit expansion, from telemetry.

    ``*_basis`` is ``"measured"`` when the figure came from the data and
    ``"fallback"`` when it is :data:`EXPAND_ROUND_TRIP_TOKENS` (or the
    caller's override) because the cohort was too thin. The measurement
    inputs are kept so a report can show its work.
    """

    caching_per_event: int = EXPAND_ROUND_TRIP_TOKENS
    caching_basis: str = "fallback"
    noncaching_per_event: int = EXPAND_ROUND_TRIP_TOKENS
    noncaching_basis: str = "fallback"
    fallback_per_event: int = EXPAND_ROUND_TRIP_TOKENS
    # Caching-cohort inputs.
    n_expand_events_caching: int = 0
    n_expand_predictions_caching: int = 0
    n_baseline_predictions: int = 0
    baseline_miss_rate: float = 0.0
    expand_miss_rate: float = 0.0
    median_rebill_premium: int = 0
    # Non-caching-cohort inputs.
    n_expand_events_noncaching: int = 0
    n_expand_predictions_noncaching: int = 0
    median_expand_call_input: int = 0

    def per_event(self, provider_caches: bool | None) -> int:
        """Per-event cost for a prediction with the given cache status.
        Unknown status is charged as non-caching (the full-price call)."""
        return self.caching_per_event if provider_caches is True else self.noncaching_per_event

    def basis_for(self, provider_caches: bool | None) -> str:
        return self.caching_basis if provider_caches is True else self.noncaching_basis

    @property
    def blended_per_event(self) -> int:
        """Event-weighted per-event cost across both cohorts (for callers
        that only have an event count, e.g. the analyzer's scope totals)."""
        n = self.n_expand_events_caching + self.n_expand_events_noncaching
        if n <= 0:
            return self.noncaching_per_event
        total = (self.n_expand_events_caching * self.caching_per_event
                 + self.n_expand_events_noncaching * self.noncaching_per_event)
        return int(round(total / n))

    @property
    def blended_basis(self) -> str:
        bases = set()
        if self.n_expand_events_caching:
            bases.add(self.caching_basis)
        if self.n_expand_events_noncaching:
            bases.add(self.noncaching_basis)
        if not bases:
            # No events: the figure blended_per_event reports is the
            # non-caching one, so label it as that figure's basis.
            return self.noncaching_basis
        return bases.pop() if len(bases) == 1 else "mixed"

    def to_json(self) -> dict[str, Any]:
        return {
            "caching_per_event": self.caching_per_event,
            "caching_basis": self.caching_basis,
            "noncaching_per_event": self.noncaching_per_event,
            "noncaching_basis": self.noncaching_basis,
            "fallback_per_event": self.fallback_per_event,
            "n_expand_events_caching": self.n_expand_events_caching,
            "n_expand_predictions_caching": self.n_expand_predictions_caching,
            "n_baseline_predictions": self.n_baseline_predictions,
            "baseline_miss_rate": round(self.baseline_miss_rate, 4),
            "expand_miss_rate": round(self.expand_miss_rate, 4),
            "median_rebill_premium": self.median_rebill_premium,
            "n_expand_events_noncaching": self.n_expand_events_noncaching,
            "n_expand_predictions_noncaching": self.n_expand_predictions_noncaching,
            "median_expand_call_input": self.median_expand_call_input,
        }


def measure_expand_overhead(
    predictions: list[dict[str, Any]],
    api_calls: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    *,
    scope: str | None = None,
    api_last: dict[str, dict[str, Any]] | None = None,
    calls_by_pred: dict[str, list[dict[str, Any]]] | None = None,
    fallback_per_event: int = EXPAND_ROUND_TRIP_TOKENS,
    min_expand_events: int = MEASURED_OVERHEAD_MIN_EXPAND_EVENTS,
    min_baseline_predictions: int = MEASURED_OVERHEAD_MIN_BASELINE_PREDICTIONS,
    hints: dict[str, bool] | None = None,
) -> MeasuredExpandOverhead:
    """Measure the per-event cost of an explicit expansion, per cache cohort.

    Pure read over the three telemetry lists (``scope`` restricts to one
    ``agent:platform``). Each prediction is placed in the caching or
    non-caching cohort by :func:`prediction_provider_caches` (per-call fact,
    provider hint, then its cache-mode cohort); unresolved rows are skipped.

    Caching cohort — ``per_event = max(0, expand_miss_rate −
    baseline_miss_rate) × median_rebill_premium``:
      * a prediction *missed* iff any of its calls has ``cache_read_tokens ==
        0`` with a prompt above :data:`CACHE_MISS_MIN_PROMPT_TOKENS`;
      * ``baseline_miss_rate`` is over caching predictions that made no
        ``expand_tools`` call and were not the session's cold first turn
        (``frozen_reuse``, or a carry-all row, which never expands);
      * ``expand_miss_rate`` is over caching predictions that expanded;
      * ``median_rebill_premium`` is the median :func:`_rebill_premium` over
        the missed expand predictions.
    Non-caching cohort — ``per_event`` = median over expand predictions of
    the expand meta-call's ``input_tokens`` (the prediction's call with the
    smallest ``output_tokens``: the model's "expand_tools" turn is its
    shortest output).

    Fallback: a cohort with fewer than ``min_expand_events`` events, or
    (caching) fewer than ``min_baseline_predictions`` baseline rows, is
    charged ``fallback_per_event`` and marked ``"fallback"``.
    """
    if scope is not None:
        want = scope.strip().lower()  # analyzer scope keys are normalized this way
        predictions = [p for p in predictions
                       if str(p.get("scope") or "").strip().lower() == want]
        tool_calls = [t for t in tool_calls
                      if str(t.get("scope") or "").strip().lower() == want]
    if api_last is None:
        api_last = last_api_call_by_prediction(api_calls)
    if calls_by_pred is None:
        calls_by_pred = index_api_calls_by_prediction(api_calls)

    expand_events: dict[str, int] = defaultdict(int)
    for t in tool_calls:
        if t.get("tool_name") == "expand_tools":
            pid = str(t.get("prediction_id") or "")
            if pid:
                expand_events[pid] += 1

    out = MeasuredExpandOverhead(
        caching_per_event=int(fallback_per_event),
        noncaching_per_event=int(fallback_per_event),
        fallback_per_event=int(fallback_per_event),
    )

    baseline_missed = 0
    n_baseline = 0
    expand_missed = 0
    n_expand_caching = 0
    rebill: list[float] = []
    expand_inputs: list[float] = []
    for p in predictions:
        pid = str(p.get("prediction_id") or "")
        if not pid:
            continue
        mode = classify_prediction_mode(p, api_last)
        cohort_mode = mode if mode in ("on", "off") else None
        calls = calls_by_pred.get(pid, [])
        caches = prediction_provider_caches(calls, cohort_mode, hints)
        if caches is None:
            continue
        n_events = expand_events.get(pid, 0)
        if caches:
            if n_events:
                n_expand_caching += 1
                out.n_expand_events_caching += n_events
                if calls and any(_call_is_cache_miss(c) for c in calls):
                    expand_missed += 1
                    rebill.append(_rebill_premium(calls))
            elif calls and (
                p.get("frozen_reuse")
                or str(p.get("policy_source") or "") == "cache_on_carry_all"
            ):
                n_baseline += 1
                if any(_call_is_cache_miss(c) for c in calls):
                    baseline_missed += 1
        elif n_events:
            out.n_expand_events_noncaching += n_events
            out.n_expand_predictions_noncaching += 1
            if calls:
                meta = min(calls, key=lambda r: int(r.get("output_tokens") or 0))
                expand_inputs.append(float(meta.get("input_tokens") or 0))

    out.n_expand_predictions_caching = n_expand_caching
    out.n_baseline_predictions = n_baseline
    out.baseline_miss_rate = (baseline_missed / n_baseline) if n_baseline else 0.0
    out.expand_miss_rate = (expand_missed / n_expand_caching) if n_expand_caching else 0.0
    out.median_rebill_premium = int(round(_median(rebill)))
    out.median_expand_call_input = int(round(_median(expand_inputs)))

    if (out.n_expand_events_caching >= min_expand_events
            and n_baseline >= min_baseline_predictions):
        excess = max(0.0, out.expand_miss_rate - out.baseline_miss_rate)
        out.caching_per_event = int(round(excess * out.median_rebill_premium))
        out.caching_basis = "measured"
    if out.n_expand_events_noncaching >= min_expand_events and expand_inputs:
        out.noncaching_per_event = out.median_expand_call_input
        out.noncaching_basis = "measured"
    return out


@dataclass
class ObservedCohort:
    """Realized savings from organic telemetry (never summed with projected)."""

    label: str = "observed"
    n_predictions: int = 0
    n_sessions: int = 0
    # RAW realized schema-token reduction (Σ ceiling − narrowed across cache-on
    # and cache-off cohorts; bypass/pending excluded from the headline). A
    # schema-size figure, kept as the detail line.
    realized_schema_token_reduction: int = 0
    # PRICED reduction: the same tokens weighed by what they were worth —
    # 1/miss_premium on a caching route (they would have been cache reads),
    # 1.0 on a non-caching one. The gross the headline nets from.
    saved_input_equiv_total: int = 0
    # Observed explicit-expansion overhead: each expand_tools event × the
    # MEASURED per-event cost of its prediction's cache cohort (see
    # measure_expand_overhead), or EXPAND_ROUND_TRIP_TOKENS when thin.
    # Trigger activations are not counted.
    expansion_events: int = 0
    expansion_overhead: int = 0
    overhead_per_event_caching: int = EXPAND_ROUND_TRIP_TOKENS
    overhead_per_event_noncaching: int = EXPAND_ROUND_TRIP_TOKENS
    overhead_basis: str = "fallback"  # measured | fallback | mixed
    # Historical (blended) net: saved_input_equiv_total − expansion_overhead
    # (input-token equivalent). What actually happened over the window.
    net_token_reduction: int = 0
    # Forward (ongoing) net: the non-caching cohort only — a carry-all caching
    # provider narrows nothing and ships no expand_tools. This is the number
    # to quote and to annualize.
    net_forward: int = 0
    saved_input_equiv_noncaching: int = 0
    overhead_noncaching: int = 0
    # The caching cohort's blended net, reported for transparency.
    net_caching_historical: int = 0
    # Wall-clock span of the measured traffic (epoch seconds; 0 when empty) —
    # lets reports annualize the observed pace.
    first_ts: float = 0.0
    last_ts: float = 0.0
    # Uncached-only slice of the above, for the forward view. Tool Belt is an
    # uncached-API tool, so the per-session average and the annualized pace
    # scope to the sessions/span that actually accrued ``net_forward`` — the
    # non-caching predictions (``caches is not True``, the same rows priced
    # into ``saved_input_equiv_noncaching``). Counting caching sessions in
    # those denominators dilutes both. 0 when there is no uncached traffic.
    n_sessions_noncaching: int = 0
    first_ts_noncaching: float = 0.0
    last_ts_noncaching: float = 0.0
    cache_on: dict[str, Any] = field(default_factory=dict)
    cache_off: dict[str, Any] = field(default_factory=dict)
    pending: dict[str, Any] = field(default_factory=dict)
    bypass: dict[str, Any] = field(default_factory=dict)
    # The measurement behind the per-event figures (shows its work).
    overhead_measurement: dict[str, Any] = field(default_factory=dict)
    token_estimator: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_predictions": self.n_predictions,
            "n_sessions": self.n_sessions,
            "realized_schema_token_reduction": self.realized_schema_token_reduction,
            "saved_input_equiv_total": self.saved_input_equiv_total,
            "expansion_events": self.expansion_events,
            "expansion_overhead": self.expansion_overhead,
            "overhead_per_event_caching": self.overhead_per_event_caching,
            "overhead_per_event_noncaching": self.overhead_per_event_noncaching,
            "overhead_basis": self.overhead_basis,
            "net_token_reduction": self.net_token_reduction,
            "net_forward": self.net_forward,
            "saved_input_equiv_noncaching": self.saved_input_equiv_noncaching,
            "overhead_noncaching": self.overhead_noncaching,
            "net_caching_historical": self.net_caching_historical,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "n_sessions_noncaching": self.n_sessions_noncaching,
            "first_ts_noncaching": self.first_ts_noncaching,
            "last_ts_noncaching": self.last_ts_noncaching,
            "cache_on": self.cache_on,
            "cache_off": self.cache_off,
            "pending": self.pending,
            "bypass": self.bypass,
            "overhead_measurement": self.overhead_measurement,
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
    calls_by_pred = index_api_calls_by_prediction(api_calls)
    on = cohort_stats(predictions, api_last, api_calls, "on", calls_by_pred)
    off = cohort_stats(predictions, api_last, api_calls, "off", calls_by_pred)
    pending = cohort_stats(predictions, api_last, api_calls, "pending", calls_by_pred)
    bypass = cohort_stats(predictions, api_last, api_calls, "bypass", calls_by_pred)

    # Realized reduction: the narrowed cohorts only (bypass never narrows;
    # pending is not yet classified). expand_tools overhead is observed, not
    # estimated per-turn — and it is charged over the SAME cohorts whose
    # savings are summed: an expansion in a bypass or pending session has no
    # counted saving to net against. Each event is charged its own cohort's
    # MEASURED per-event cost (a caching prediction's expansion broke the
    # prefix cache; a non-caching one paid an extra full-price call).
    realized = on.get("saved_tokens_total", 0) + off.get("saved_tokens_total", 0)
    saved_input_equiv = (on.get("saved_input_equiv_total", 0)
                         + off.get("saved_input_equiv_total", 0))
    counted_mode = {
        str(p.get("prediction_id") or ""): mode for p in predictions
        for mode in (classify_prediction_mode(p, api_last),)
        if mode in ("on", "off")
    }
    # Uncached-only denominators for the forward view. Walk the SAME rows that
    # price into ``saved_input_equiv_noncaching`` (caches is not True) and
    # record their distinct sessions and wall-clock span, so the per-session
    # average and pace annualize over uncached traffic only — not the caching
    # sessions that contribute nothing to ``net_forward``.
    noncaching_sessions: set[str] = set()
    noncaching_ts: list[float] = []
    for p in predictions:
        pid = str(p.get("prediction_id") or "")
        mode = counted_mode.get(pid)
        if mode is None:
            continue
        if prediction_provider_caches(calls_by_pred.get(pid), mode) is True:
            continue
        sk = _session_key(p)
        if sk:
            noncaching_sessions.add(sk)
        ts = float(p.get("ts") or 0)
        if ts > 0:
            noncaching_ts.append(ts)
    n_sessions_noncaching = len(noncaching_sessions)
    first_ts_noncaching = min(noncaching_ts) if noncaching_ts else 0.0
    last_ts_noncaching = max(noncaching_ts) if noncaching_ts else 0.0

    measured = measure_expand_overhead(
        predictions, api_calls, tool_calls, api_last=api_last,
        calls_by_pred=calls_by_pred,
    )
    expand_events = 0
    overhead = 0
    overhead_caching = 0
    overhead_noncaching = 0
    bases: set[str] = set()
    for t in tool_calls:
        if t.get("tool_name") != "expand_tools":
            continue
        pid = str(t.get("prediction_id") or "")
        if pid not in counted_mode:
            continue
        caches = prediction_provider_caches(calls_by_pred.get(pid), counted_mode[pid])
        expand_events += 1
        per_event = measured.per_event(caches)
        overhead += per_event
        if caches is True:
            overhead_caching += per_event
        else:
            overhead_noncaching += per_event
        bases.add(measured.basis_for(caches))
    if not bases:
        overhead_basis = "measured"  # no events: nothing charged, nothing estimated
    elif len(bases) == 1:
        overhead_basis = bases.pop()
    else:
        overhead_basis = "mixed"
    net = saved_input_equiv - overhead

    # Forward (ongoing) view: only the non-caching cohort keeps saving, so
    # ``net_token_reduction`` is the blended history and ``net_forward`` is
    # what to project.
    saved_input_equiv_noncaching = (on.get("saved_input_equiv_noncaching", 0)
                                    + off.get("saved_input_equiv_noncaching", 0))
    saved_input_equiv_caching = (on.get("saved_input_equiv_caching", 0)
                                 + off.get("saved_input_equiv_caching", 0))
    net_forward = saved_input_equiv_noncaching - overhead_noncaching
    # The caching cohort's blended net — negative when expand breaks outweigh
    # the cache-read-priced schema savings.
    net_caching_historical = saved_input_equiv_caching - overhead_caching

    n_pred = on.get("n_predictions", 0) + off.get("n_predictions", 0)
    n_sess = len({
        k for k in (_session_key(p) for p in predictions
                    if classify_prediction_mode(p, api_last) in ("on", "off")) if k
    })

    measured_ts = [
        float(p.get("ts") or 0) for p in predictions
        if classify_prediction_mode(p, api_last) in ("on", "off")
        and float(p.get("ts") or 0) > 0
    ]

    return ObservedCohort(
        n_predictions=n_pred,
        n_sessions=n_sess,
        first_ts=min(measured_ts) if measured_ts else 0.0,
        last_ts=max(measured_ts) if measured_ts else 0.0,
        n_sessions_noncaching=n_sessions_noncaching,
        first_ts_noncaching=first_ts_noncaching,
        last_ts_noncaching=last_ts_noncaching,
        realized_schema_token_reduction=realized,
        saved_input_equiv_total=saved_input_equiv,
        expansion_events=expand_events,
        expansion_overhead=overhead,
        overhead_per_event_caching=measured.caching_per_event,
        overhead_per_event_noncaching=measured.noncaching_per_event,
        overhead_basis=overhead_basis,
        net_token_reduction=net,
        net_forward=net_forward,
        saved_input_equiv_noncaching=saved_input_equiv_noncaching,
        overhead_noncaching=overhead_noncaching,
        net_caching_historical=net_caching_historical,
        cache_on=on,
        cache_off=off,
        pending=pending,
        bypass=bypass,
        overhead_measurement=measured.to_json(),
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
    for entry in recorded_defs:
        name = _def_name(entry)
        if name is not None and name not in tool_defs:
            tool_defs[name] = entry

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
    # Priced gross (input-token equivalent) the net is taken from.
    gross_input_equiv_reduction: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "cost_class": self.cost_class,
            "reason": self.reason,
            "rate_basis": self.rate_basis,
            "gross_schema_token_reduction": self.gross_schema_token_reduction,
            "gross_input_equiv_reduction": self.gross_input_equiv_reduction,
            "expansion_overhead": self.expansion_overhead,
            "net_token_reduction": self.net_token_reduction,
            "estimated_usd_savings": self.estimated_usd_savings,
        }


@dataclass
class ProjectedCohort:
    """Counterfactual replay savings (never summed with observed)."""

    label: str = "projected"
    counterfactual: bool = True
    sessions_analyzed: int = 0
    user_turns_analyzed: int = 0
    gross_schema_token_reduction: int = 0
    # Priced gross: the input-token-equivalent worth of the saved schema
    # tokens (see price_factor_for). The net is taken from this.
    gross_input_equiv_reduction: int = 0
    expansion_events: int = 0
    estimated_expansion_overhead: int = 0
    # Per-event cost charged: the agent's MEASURED figure
    # (measure_expand_overhead over its observed telemetry), or
    # EXPAND_ROUND_TRIP_TOKENS when that telemetry is too thin.
    overhead_per_event: int = EXPAND_ROUND_TRIP_TOKENS
    overhead_basis: str = "fallback"  # measured | fallback
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
    estimated_usd_savings: float | None = None  # sum over known-cost rows only
    usd_coverage: str = "none"
    confidence: str = "insufficient"  # high | medium | low | insufficient
    reasons: list[str] = field(default_factory=list)
    token_estimator: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "counterfactual": self.counterfactual,
            "sessions_analyzed": self.sessions_analyzed,
            "user_turns_analyzed": self.user_turns_analyzed,
            "gross_schema_token_reduction": self.gross_schema_token_reduction,
            "gross_input_equiv_reduction": self.gross_input_equiv_reduction,
            "expansion_events": self.expansion_events,
            "estimated_expansion_overhead": self.estimated_expansion_overhead,
            "overhead_per_event": self.overhead_per_event,
            "overhead_basis": self.overhead_basis,
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


def _resolve_effective_preset(plugin_config: dict[str, Any], scope: str):
    """Resolve the current effective carrying preset for a scope, read-only.

    :func:`presets.resolve_preset` honors the applied learned overlay and the
    per-channel config, so the projection replays exactly the assignment the
    runtime would use.
    """
    presets = _import_sibling("presets")
    return presets.resolve_preset(plugin_config, scope)


def _replay_active_names(
    session: HistoricalSession,
    preset: presets.Preset,
) -> tuple[list[list[str]], list[int], int]:
    """Replay the predictor over each user turn.

    Returns ``(active_per_turn, user_turn_indices, expansion_events)`` where
    ``active_per_turn[k]`` is the resolved active tool-name list for the k-th
    user turn: the active set is resolved fresh every turn, and a called tool
    that is neither resident nor trigger-activated is charged as one explicit
    expansion event. A trigger activation is not an ``expand_tools`` round
    trip and carries no overhead.

    MCP/passthrough tools are never narrowed — they stay active every turn.
    """
    predictor = _import_sibling("predictor")
    ceiling_names = list(session.tool_defs.keys())
    ceiling_set = set(ceiling_names)
    mcp_names = {n for n in ceiling_names if _is_mcp(n)}

    active_per_turn: list[list[str]] = []
    user_indices: list[int] = []
    expansion_events = 0

    for i, row in enumerate(session.turns):
        if row.get("role") != "user":
            continue
        message = _message_text(row)
        if not message.strip():
            continue
        prediction = predictor.predict(message, None, preset)

        if prediction.no_narrowing:
            per_turn = set(ceiling_names)
        else:
            # Full-start: residency is the ceiling minus demotions (plus any
            # explicit carry), mirroring carrying.resolve; the predictor's
            # candidate set contributes trigger activations on top.
            demoted = set(getattr(preset, "demoted", []) or [])
            residents = (ceiling_set - mcp_names - demoted) | (
                (set(preset.always_carry) | set(preset.carry)) & ceiling_set
            )
            resolved = residents | (set(prediction.active_tool_names) & ceiling_set)
            per_turn = resolved | mcp_names  # passthrough never narrowed

        called = _called_tools_for_turn(session.turns, i)
        # Which triggers fired this turn -> the tools they activate (T).
        triggered_tools: set[str] = set()
        for group in getattr(preset, "triggers", []) or []:
            if group.name in set(prediction.triggers_fired):
                triggered_tools |= {t for t in group.tools if t in ceiling_set}

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
    provider_input_by_session: dict[str, int] | None = None,
    route_by_session: dict[str, SessionRoute] | None = None,
    expand_overhead: MeasuredExpandOverhead | None = None,
) -> ProjectedCohort:
    """Replay a set of historical sessions into one projected cohort.

    ``provider_input_by_session`` maps a session key to provider-reported
    ``input_tokens`` — when present for a session it is the authoritative
    denominator and wins over reconstruction.
    ``route_by_session`` maps a session key to the billing route its API calls
    were actually made on (see :func:`_api_call_evidence`). It is the
    production source of ``provider``/``api_mode`` for costing, because live
    Hermes ``session_meta`` records carry neither field. A value recorded in
    ``session_meta`` still wins as an explicit override.
    ``expand_overhead`` is the agent's measured per-event expansion cost
    (:func:`measure_expand_overhead` over its observed telemetry). Absent,
    every event is charged the :data:`EXPAND_ROUND_TRIP_TOKENS` fallback.

    The replay models cache-off economics throughout: on a caching provider
    the runtime carries everything and ships no ``expand_tools``, so there is
    no narrowing to project there.
    """
    provider_input_by_session = provider_input_by_session or {}
    route_by_session = route_by_session or {}
    if expand_overhead is None:
        expand_overhead = MeasuredExpandOverhead()
    per_event = expand_overhead.per_event(provider_caches=False)
    overhead_basis = expand_overhead.basis_for(provider_caches=False)

    gross = 0
    gross_priced = 0.0
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
    per_model_gross_priced: dict[tuple[str, str, str], float] = defaultdict(float)
    per_model_overhead: dict[tuple[str, str, str], int] = defaultdict(int)

    for session in sessions:
        if not session.tool_defs or not session.schemas_complete:
            incomplete_schema = True
            continue
        try:
            preset = _resolve_effective_preset(plugin_config, session.scope)
        except Exception:
            incomplete_schema = True
            continue

        active_per_turn, user_indices, exp_events = _replay_active_names(session, preset)
        if not user_indices:
            continue

        full_defs = list(session.tool_defs.values())
        ceiling_tok = schema_tokens(full_defs)
        session_gross = 0
        for active in active_per_turn:
            active_defs = [session.tool_defs[n] for n in active if n in session.tool_defs]
            narrowed_tok = schema_tokens(active_defs)
            session_gross += max(0, ceiling_tok - narrowed_tok)

        overhead = exp_events * per_event
        session_priced = session_gross * price_factor_for(
            session.model, provider_caches=False
        )
        gross += session_gross
        gross_priced += session_priced
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
        per_model_gross_priced[key] += session_priced
        per_model_overhead[key] += overhead

        # Denominator: provider-reported wins per session.
        prov = provider_input_by_session.get(skey)
        if prov is not None:
            denom_provider += int(prov)
            have_provider = True
        else:
            denom_reconstructed += _reconstructed_input_denominator(session, user_indices)
            have_reconstructed = True

    cohort = ProjectedCohort()
    cohort.token_estimator = token_estimator_name()
    cohort.sessions_analyzed = sessions_analyzed
    cohort.user_turns_analyzed = turns_analyzed
    cohort.gross_schema_token_reduction = gross
    cohort.gross_input_equiv_reduction = int(round(gross_priced))
    cohort.expansion_events = expansion_events
    cohort.estimated_expansion_overhead = expansion_events * per_event
    cohort.overhead_per_event = per_event
    cohort.overhead_basis = overhead_basis
    cohort.net_token_reduction = (
        cohort.gross_input_equiv_reduction - cohort.estimated_expansion_overhead
    )
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
    for key in sorted(per_model_gross.keys()):
        model, provider, api_mode = key
        cc = classify_cost(model, provider, api_mode)
        g = per_model_gross[key]
        g_priced = int(round(per_model_gross_priced[key]))
        oh = per_model_overhead[key]
        net = g_priced - oh
        usd: float | None = None
        if cc.dollars_allowed and not incomplete_schema:
            prices = price_for(model)
            usd = round(max(0, net) * prices["input"] / 1_000_000.0, 4)
        cohort.models.append(ProjectedModelRow(
            model=model, provider=provider, cost_class=cc.cost_class,
            reason=cc.reason, rate_basis=cc.rate_basis,
            gross_schema_token_reduction=g, expansion_overhead=oh,
            net_token_reduction=net, estimated_usd_savings=usd,
            gross_input_equiv_reduction=g_priced,
        ))

    # Hermes producers record a *transport* label in ``api_mode``, never a
    # billing route, so :func:`classify_cost` holds every row at
    # subscription/unknown and the cohort carries no dollars.
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
    since_ts: float = 0.0,
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
    # The per-event expansion cost is measured from the agent's own observed
    # telemetry (whole history — a thin --since window would only push it
    # back to the fallback); thin data falls back inside the helper.
    expand_overhead = measure_expand_overhead(
        [logger_io.normalize_prediction_row(p)
         for p in load_jsonl(location.state_dir / "predictions.jsonl")],
        load_jsonl(location.state_dir / "api_calls.jsonl"),
        [logger_io.normalize_tool_call_row(t)
         for t in load_jsonl(location.state_dir / "tool_calls.jsonl")],
    )

    return project_sessions(
        sessions, plugin_config,
        provider_input_by_session=provider_input,
        route_by_session=routes,
        expand_overhead=expand_overhead,
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
    prediction -> Hermes session under that file-stem key. The raw chat
    ``session_id`` is the last-resort fallback when no bridge is available.
    """
    keys: list[str] = []
    hermes = pid_to_hermes.get(str(row.get("prediction_id") or ""))
    if hermes:
        keys.append(hermes)
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


# ══════════════════════════════════════════════════════════════════════════════
#  Top-level report assembly
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class AgentSavings:
    agent: str
    platforms: list[str]
    observed: ObservedCohort
    projected: ProjectedCohort
    #: Human name for text rendering (config's scope agent name when set);
    #: JSON keeps the canonical profile name in ``agent``.
    display_name: str = ""

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
    agents: list[AgentSavings]
    hermes_home: str
    token_estimator: str
    #: The raw --since value this report honored; None = all recorded history.
    since: str | None = None

    def to_json(self) -> dict[str, Any]:
        # Aggregate token totals are complete across every agent; dollars stay
        # suppressed for the same reason the per-agent cohorts suppress them
        # (no provable billing route — see classify_cost).
        agg_observed_realized = sum(a.observed.realized_schema_token_reduction for a in self.agents)
        agg_observed_priced = sum(a.observed.saved_input_equiv_total for a in self.agents)
        agg_observed_net = sum(a.observed.net_token_reduction for a in self.agents)
        agg_proj_gross = sum(a.projected.gross_schema_token_reduction for a in self.agents)
        agg_proj_priced = sum(a.projected.gross_input_equiv_reduction for a in self.agents)
        agg_proj_net = sum(a.projected.net_token_reduction for a in self.agents)
        return {
            "schema": "tool-belt/savings/v1",
            "generated_for": self.generated_for,
            "hermes_home": self.hermes_home,
            "token_estimator": self.token_estimator,
            "since": self.since,
            "agents": [a.to_json() for a in self.agents],
            "aggregate": {
                "observed": {
                    "label": "observed",
                    "realized_schema_token_reduction": agg_observed_realized,
                    "saved_input_equiv_total": agg_observed_priced,
                    "net_token_reduction": agg_observed_net,
                },
                "projected": {
                    "label": "projected",
                    "counterfactual": True,
                    "gross_schema_token_reduction": agg_proj_gross,
                    "gross_input_equiv_reduction": agg_proj_priced,
                    "net_token_reduction": agg_proj_net,
                    "estimated_usd_savings": None,
                    "usd_coverage": "none",
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
    plugin_config: dict[str, Any] | None = None,
) -> SavingsReport:
    """Build the full savings report — the one entry point the CLI + onboarding
    both call. Strictly read-only.

    ``agent`` restricts to a single enabled agent (raising
    :class:`UnknownAgentError` if it isn't present/enabled).
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
        projected = compute_projected(loc, plugin_config, since_ts=since_ts)
        agents.append(AgentSavings(
            agent=loc.agent,
            platforms=scopes,
            observed=observed,
            projected=projected,
            display_name=agent_display_name(loc.profile_home, loc.agent),
        ))

    return SavingsReport(
        generated_for=agent or "all",
        agents=agents,
        hermes_home=str(home),
        token_estimator=token_estimator_name(),
        since=since,
    )
