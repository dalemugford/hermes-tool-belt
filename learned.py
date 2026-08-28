"""Persistent learned policy state for tool-belt (Tool Belt 1.0 carrying model).

The learned layer is deliberately small and inspectable: a JSON file under
``$HERMES_HOME/state/tool-belt/learned.json``. Prediction only reads from it;
the analyzer or an explicit user action writes it.

Schema (v2)::

    {
      "version": 2,
      "updated_at": "...",
      "scopes": {
        "default:telegram": {
          "carry": [],         # tools to promote into adaptive residency
          "expand_only": [],   # tools to demote out of residency
          "shaping": {}        # cache-aware shaping metadata (opaque here)
        }
      }
    }

Reading normalizes an older v1 document into this shape **in memory only** — it
never rewrites the file. The v1 → v2 field mapping is: ``always_on → carry``,
``always_off → expand_only``, ``cache_aware → shaping``. v1
``trigger_adjustments`` are dropped with a warning: trigger definitions are
immutable under the 1.0 model. The next explicit/confirmed writer
(:func:`write_state`) may persist the normalized v2 shape atomically.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .presets import Preset, TriggerGroup

logger = logging.getLogger(__name__)

LEARNED_VERSION = 2
# Two effective learned modes:
#   recommend (default) — runtime does NOT merge learned.json into the preset.
#                         Recommendations flow through analyze.py
#                         (--write-recommendations) and the shaper for human review.
#   apply               — runtime merges learned.json into the preset.
_ALLOWED_MODES = {"recommend", "apply"}
# Legacy config values are aliased at load time (clean migration, no shim layer):
# old "off" behaved like recommend at runtime; old "auto"/"audit" both merged.
# This is the unrelated learned_mode setting — its aliases are preserved.
_LEGACY_ALIASES = {"off": "recommend", "auto": "apply", "audit": "apply"}
_CACHE: dict[str, Any] = {"path": None, "mtime_ns": None, "state": None, "hash": ""}


@dataclass
class LearnedMergeResult:
    """Result metadata from applying learned state to a preset.

    ``mode`` is the resolved learned_mode (``recommend`` or ``apply``). Only
    ``apply`` merges learned state; ``recommend`` returns the preset unchanged.
    """

    preset: Preset
    mode: str = "recommend"
    policy_source: str = "preset"
    policy_version: str = ""
    learned_changes: list[str] = field(default_factory=list)
    learned_scope: str = ""


def state_dir() -> Path:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(home) / "state" / "tool-belt"


def learned_path() -> Path:
    return state_dir() / "learned.json"


def normalize_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    mode = _LEGACY_ALIASES.get(mode, mode)  # migrate off/auto/audit → recommend/apply
    return mode if mode in _ALLOWED_MODES else "recommend"


def scope_candidates(scope: str) -> list[str]:
    """Return lookup order for a scope.

    ``assistant-a:telegram`` should win over ``telegram``. The platform fallback
    keeps existing channel-style configs useful while the learned layer
    moves toward agent/platform scopes.
    """
    out: list[str] = []
    clean = str(scope or "").strip().lower()
    if clean:
        out.append(clean)
        if ":" in clean:
            platform = clean.rsplit(":", 1)[-1]
            if platform and platform not in out:
                out.append(platform)
    return out


def learned_mode(plugin_config: dict[str, Any], scope: str) -> str:
    """Resolve learned_mode with per-scope/per-platform override support."""
    mode = normalize_mode(plugin_config.get("learned_mode", "recommend"))
    channels = plugin_config.get("channels") or {}
    if isinstance(channels, dict):
        for key in scope_candidates(scope):
            cfg = channels.get(key)
            if isinstance(cfg, dict) and "learned_mode" in cfg:
                return normalize_mode(cfg.get("learned_mode"))
    return mode


# ─── v1 → v2 normalization (in-memory; no read-time write) ──────────────────

def _empty_v2() -> dict[str, Any]:
    return {"version": LEARNED_VERSION, "updated_at": "", "scopes": {}}


#: Scope-entry keys the normalizer owns (renames the v1 spellings, keeps the v2
#: ones). Every *other* key in a scope dict is unrecognized metadata and passes
#: through untouched, so a writer never drops a field it doesn't understand.
_KNOWN_SCOPE_KEYS = frozenset(
    {
        "always_on", "always_off", "cache_aware", "trigger_adjustments",  # v1
        "carry", "expand_only", "shaping",                                # v2
    }
)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _reconcile_overlap(
    carry: list[str], expand_only: list[str], *, scope: str | None = None
) -> list[str]:
    """Resolve a ``carry`` ∩ ``expand_only`` overlap toward carrying.

    A tool named in both lists is contradictory; carrying wins (fail safe
    toward keeping capability resident) and the tool is dropped from
    ``expand_only``. Returns the reconciled ``expand_only`` list and warns once,
    naming the offenders. This is the single home for the carry-wins precedence
    decision — both the read-time normalizer (:func:`_normalize_scope_entry`)
    and :func:`apply_to_preset` route through it.
    """
    carry_set = set(carry)
    overlap = sorted({t for t in expand_only if t in carry_set})
    if not overlap:
        return expand_only
    if scope:
        logger.warning(
            "tool-belt: learned scope %r names %d tool(s) in both carry and "
            "expand_only; resolving toward carry: %s",
            scope, len(overlap), ", ".join(overlap),
        )
    else:
        logger.warning(
            "tool-belt: learned scope names %d tool(s) in both carry and "
            "expand_only; resolving toward carry: %s",
            len(overlap), ", ".join(overlap),
        )
    return [t for t in expand_only if t not in carry_set]


def _normalize_scope_entry(entry: Any, *, warn_trigger_adjustments: bool = True) -> dict[str, Any]:
    """Map one scope's learned entry to the v2 ``{carry, expand_only, shaping}``.

    Accepts both v1 keys (``always_on`` / ``always_off`` / ``cache_aware`` /
    ``trigger_adjustments``) and native v2 keys (``carry`` / ``expand_only`` /
    ``shaping``); native v2 keys win when both are present. A malformed
    carry/expand_only overlap is resolved toward carrying, with a warning.

    Only the known v1/v2 keys are renamed/normalized; any *other* key in the
    scope dict is unrecognized metadata and is copied through untouched, so a
    normalize → write round-trip never silently drops a field.
    """
    if not isinstance(entry, dict):
        return {"carry": [], "expand_only": [], "shaping": {}}

    # Preserve every unrecognized key verbatim; the three v2 keys below overwrite.
    out: dict[str, Any] = {k: v for k, v in entry.items() if k not in _KNOWN_SCOPE_KEYS}

    # v1 field mapping.
    carry = _string_list(entry.get("always_on"))
    expand_only = _string_list(entry.get("always_off"))
    shaping_raw = entry.get("cache_aware")
    shaping = dict(shaping_raw) if isinstance(shaping_raw, dict) else {}

    # Native v2 keys take precedence when present.
    if "carry" in entry:
        carry = _string_list(entry.get("carry"))
    if "expand_only" in entry:
        expand_only = _string_list(entry.get("expand_only"))
    if isinstance(entry.get("shaping"), dict):
        shaping = dict(entry["shaping"])

    if warn_trigger_adjustments and entry.get("trigger_adjustments"):
        adjustments = entry["trigger_adjustments"]
        count = len(adjustments) if hasattr(adjustments, "__len__") else 1
        logger.warning(
            "tool-belt: ignoring v1 learned trigger_adjustments during v2 "
            "normalization (%d) — trigger definitions are immutable under the "
            "1.0 carrying model",
            count,
        )

    out["carry"] = carry
    out["expand_only"] = _reconcile_overlap(carry, expand_only)
    out["shaping"] = shaping
    return out


def normalize_state(doc: Any) -> dict[str, Any]:
    """Normalize a learned-state document into the v2 shape, in memory.

    v1 → v2 per scope: ``always_on → carry``, ``always_off → expand_only``,
    ``cache_aware → shaping``. v1 ``trigger_adjustments`` are dropped with a
    warning. A malformed carry/expand_only overlap resolves toward carrying
    (warn). No write happens here — the caller decides whether to persist.
    """
    if not isinstance(doc, dict):
        return _empty_v2()

    # Preserve every unrecognized top-level key verbatim; the known keys below
    # overwrite, and ``global`` is handled specially just after.
    out: dict[str, Any] = {
        k: v for k, v in doc.items()
        if k not in ("version", "updated_at", "scopes", "global")
    }
    out["version"] = LEARNED_VERSION
    out["updated_at"] = str(doc.get("updated_at") or "")
    out["scopes"] = {}
    scopes = doc.get("scopes")
    if isinstance(scopes, dict):
        for scope, entry in scopes.items():
            out["scopes"][str(scope)] = _normalize_scope_entry(entry)

    # Preserve a normalized global block only if it carries something. NOTE:
    # this block is inert today — no consumer reads it (scope_state /
    # apply_to_preset look at ``scopes`` only). Retained pending the global
    # fallback decision; do not wire it in this pass.
    global_raw = doc.get("global")
    if isinstance(global_raw, dict):
        g = _normalize_scope_entry(global_raw, warn_trigger_adjustments=False)
        if g["carry"] or g["expand_only"] or g["shaping"]:
            out["global"] = g

    return out


def load_state(force: bool = False) -> dict[str, Any]:
    """Load learned.json with mtime caching, normalized to v2 in memory.

    Missing/invalid means empty state. Reading a v1 document normalizes it to
    the v2 shape in memory and never rewrites the file (no read-time write).
    """
    path = learned_path()
    try:
        stat = path.stat()
    except FileNotFoundError:
        _CACHE.update({"path": str(path), "mtime_ns": None, "state": {}, "hash": ""})
        return {}
    except Exception as exc:
        logger.debug("tool-belt: learned state stat failed: %s", exc)
        return {}

    if (
        not force
        and _CACHE.get("path") == str(path)
        and _CACHE.get("mtime_ns") == stat.st_mtime_ns
        and isinstance(_CACHE.get("state"), dict)
    ):
        return deepcopy(_CACHE["state"])

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("learned state root is not an object")
        try:
            version = int(data.get("version", 1))
        except (TypeError, ValueError):
            version = 1
        if version > LEARNED_VERSION:
            # A newer schema than we understand — fail safe to empty (no merge)
            # rather than guess. The file is left untouched.
            logger.warning(
                "tool-belt: learned state version %r is newer than supported v%d — ignoring",
                data.get("version"), LEARNED_VERSION,
            )
            state: dict[str, Any] = {}
        else:
            # Normalize v1 → v2 in memory. The next explicit writer may persist v2.
            state = normalize_state(data)
        digest = hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
        _CACHE.update({"path": str(path), "mtime_ns": stat.st_mtime_ns, "state": state, "hash": digest})
        return deepcopy(state)
    except Exception as exc:
        logger.warning("tool-belt: failed to load learned state %s: %s", path, exc)
        _CACHE.update({"path": str(path), "mtime_ns": stat.st_mtime_ns, "state": {}, "hash": ""})
        return {}


def state_hash() -> str:
    load_state()
    return str(_CACHE.get("hash") or "")


def write_state(state: dict[str, Any], path: Path | None = None) -> None:
    """Atomically write learned state as v2. Intended for analyzer/manual commands.

    The write reconciles opposite assignments atomically: normalization ensures
    no tool remains in both ``carry`` and ``expand_only`` for a scope (a
    malformed overlap resolves toward carrying). The version is stamped to v2.
    """
    target = path or learned_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = normalize_state(dict(state or {}))
    payload["version"] = LEARNED_VERSION
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)
    load_state(force=True)


def scope_state(state: dict[str, Any], scope: str) -> tuple[str, dict[str, Any]]:
    scopes = state.get("scopes") or {}
    if not isinstance(scopes, dict):
        return "", {}
    for key in scope_candidates(scope):
        value = scopes.get(key)
        if isinstance(value, dict):
            return key, value
    return "", {}


def apply_to_preset(preset: Preset, plugin_config: dict[str, Any], scope: str) -> LearnedMergeResult:
    """Merge the learned overlay into a resolved preset when learned_mode is
    ``apply``.

    Implements the centralized Tool Belt 1.0 precedence::

        effective_always_carry = policy always_carry
        effective_carry = ((policy carry − learned expand_only)
                            ∪ learned carry) − always_carry

    ``always_carry`` wins every conflict and is immutable. Trigger definitions
    are copied byte-for-byte — promotion/demotion move a tool across the
    ``carry`` ⇄ ``expand_only`` boundary only, never touching a trigger group.
    ``recommend`` (the default) never merges; the recommendation path lives in
    the analyzer and shaper instead.
    """
    mode = learned_mode(plugin_config, scope)
    version = f"{preset.name}+learned:{state_hash() or 'none'}"
    if mode != "apply" or preset.no_narrowing:
        return LearnedMergeResult(preset=preset, mode=mode, policy_version=version)

    state = load_state()
    matched_scope, scoped = scope_state(state, scope)

    always_carry = list(preset.always_carry)
    always_carry_set = set(always_carry)
    policy_carry = list(preset.carry)

    learned_carry = _string_list(scoped.get("carry"))
    learned_expand = _string_list(scoped.get("expand_only"))

    # Malformed read overlap (a tool in both learned lists): resolve toward
    # carrying and warn. (normalize_state already does this at load; this is a
    # belt-and-braces guard for a hand-built scope dict.)
    learned_expand = _reconcile_overlap(
        learned_carry, learned_expand, scope=matched_scope or scope
    )

    # always_carry is immutable — a demotion signal naming one is ignored + warned.
    demoted_always_carry = always_carry_set & set(learned_expand)
    if demoted_always_carry:
        logger.warning(
            "tool-belt: learned expand_only names always_carry tool(s) %s — "
            "ignored (always_carry is immutable and wins every conflict)",
            ", ".join(sorted(demoted_always_carry)),
        )

    # Centralized precedence.
    expand_set = set(learned_expand)
    effective_carry = [t for t in policy_carry if t not in expand_set]
    for tool in learned_carry:
        if tool not in effective_carry:
            effective_carry.append(tool)
    effective_carry = [t for t in effective_carry if t not in always_carry_set]

    # Trigger definitions are immutable across promotion/demotion — copy them
    # byte-for-byte. Never strip a demoted tool out of its trigger group; an
    # expand_only tool must stay trigger-activatable.
    triggers = [
        TriggerGroup(
            name=group.name,
            tools=list(group.tools),
            keyword_patterns=list(group.keyword_patterns),
            exclude_patterns=list(group.exclude_patterns),
            has_attachment=group.has_attachment,
        )
        for group in preset.triggers
    ]

    # Change log (telemetry): what moved relative to the policy baseline.
    policy_carry_set = set(policy_carry)
    effective_set = set(effective_carry)
    changes: list[str] = []
    for tool in effective_carry:
        if tool not in policy_carry_set:
            changes.append(f"{tool}:carry")
    for tool in policy_carry:
        if tool not in effective_set:
            changes.append(f"{tool}:expand_only")

    if not changes:
        return LearnedMergeResult(
            preset=preset, mode=mode, policy_version=version, learned_scope=matched_scope,
        )

    merged = Preset(
        name=f"{preset.name}+learned[{matched_scope or scope}]",
        always_carry=always_carry,
        carry=effective_carry,
        triggers=triggers,
    )
    return LearnedMergeResult(
        preset=merged,
        mode=mode,
        policy_source="learned",
        policy_version=version,
        learned_changes=changes,
        learned_scope=matched_scope,
    )
