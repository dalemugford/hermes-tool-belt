"""Persistent learned policy state for tool-belt (Tool Belt 1.0 carrying model).

The learned layer is deliberately small and inspectable: a JSON file under
``$HERMES_HOME/state/tool-belt/learned.json``. Prediction only reads from it;
the shaper (manual or in-process auto-shape), the configure flows, and
inventory reconciliation write it.

Schema (v2)::

    {
      "version": 2,
      "updated_at": "...",
      "scopes": {
        "assistant-a:telegram": {
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
immutable under the 1.0 model.

This module is the sole owner of learned-state persistence: every writer
(the shaper — manual and auto-shape — configure flows, and inventory
reconciliation) routes through :func:`write_state`
(atomic, normalize-on-write, version-stamped) and every scope reset through
:func:`reset_scope`. All v1 spelling lives in this module's adapter — no
script writes or mirrors the v1 keys.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .presets import Preset, TriggerGroup

logger = logging.getLogger(__name__)

LEARNED_VERSION = 2
# Two effective learned modes:
#   apply (default)     — runtime merges learned.json into the preset.
#                         Evidence-driven shaping applies automatically; this
#                         is the zero-config product contract.
#   recommend           — opt-in observe/trial mode: runtime does NOT merge
#                         learned.json. Recommendations flow through analyze.py
#                         (--write-recommendations) and the shaper for human
#                         review.
_ALLOWED_MODES = {"recommend", "apply"}
#: The zero-config default (full-start contract).
DEFAULT_MODE = "apply"
# ``learned_mode`` accepts the pre-1.0 spellings "off"/"auto"/"audit" as aliases
# for "recommend"/"apply" ("off" behaved like recommend at runtime; "auto" and
# "audit" both merged). Unrelated to the v1→v2 learned-state field migration
# below.
_LEGACY_ALIASES = {"off": "recommend", "auto": "apply", "audit": "apply"}
_CACHE: dict[str, Any] = {"path": None, "mtime_ns": None, "state": None, "hash": ""}


@dataclass
class LearnedMergeResult:
    """Result metadata from applying learned state to a preset.

    ``mode`` is the resolved learned_mode (``recommend`` or ``apply``). Only
    ``apply`` merges learned state; ``recommend`` returns the preset unchanged.
    """

    preset: Preset
    mode: str = "apply"
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
    return mode if mode in _ALLOWED_MODES else DEFAULT_MODE


#: Where Tool Belt's settings live in Hermes' config.yaml — the plugin
#: settings subtree Hermes reserves for every plugin (``ctx.get_config``,
#: ``config_schema`` validation and ``hermes plugins`` all key off it).
CONFIG_PREFIX = "plugins.entries.tool-belt.settings"

#: The pre-2026.9 location. Never read any more; a block found there is
#: reported once at load so an operator knows to move it.
LEGACY_CONFIG_PREFIX = "plugins.tool-belt"


def flatten_channels(raw: Any) -> dict[str, Any]:
    """On-disk ``channels`` → the internal ``{"agent:platform": {...}}`` map.

    On disk, channels nest as ``channels.<agent>.<platform>`` because
    Hermes' plugin-settings API forbids ``:`` in a key segment. Internally
    every consumer keys by the scope string (which is also how telemetry
    and learned.json name scopes), so the loader flattens once here. A
    flat ``"agent:platform"`` key is passed through unchanged (in-memory
    configs built by tests and scripts already use that form).
    """
    out: dict[str, Any] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        name = str(key)
        if ":" in name or not isinstance(value, dict):
            out[name] = value
            continue
        nested = {str(p): v for p, v in value.items() if isinstance(v, dict)}
        if nested and all(isinstance(v, dict) for v in value.values()):
            for platform, cfg in nested.items():
                out[f"{name}:{platform}"] = cfg
        else:
            out[name] = value  # a bare-platform entry (older configs)
    return out


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
    mode = normalize_mode(plugin_config.get("learned_mode", DEFAULT_MODE))
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
        "triggers",                                                       # v2 overlay
    }
)


def _normalize_overlay_triggers(value: Any) -> list[dict[str, Any]]:
    """Validate a scope's learned trigger-overlay list.

    Each overlay entry is an *additive* trigger group learned from evidence —
    it can only ever ACTIVATE an ``expand_only`` tool (``T = triggered ∩ X``
    in ``carrying.resolve``), never demote, disable, or touch ``always_carry``.
    Shipped ``policy.yaml`` triggers are never edited; the overlay unions with
    them at preset resolution (``apply_to_preset``).

    Accepted entry shape (unknown keys pass through untouched)::

        {"name": str, "tools": [str, ...] non-empty,
         "keywords": [regex-str, ...] non-empty,
         "exclude_keywords": [regex-str, ...],       # optional dampeners
         "source": "mined" | "name_tokens", "created_at": iso}

    Entries without at least one tool and one keyword are dropped (warned) —
    an overlay group that can't fire or can't activate anything is noise.
    """
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        entry = dict(raw)
        tools = _string_list(entry.get("tools"))
        keywords = _string_list(entry.get("keywords"))
        if not tools or not keywords:
            logger.warning(
                "tool-belt: dropping malformed learned trigger overlay entry "
                "%r — needs at least one tool and one keyword",
                entry.get("name") or "<unnamed>",
            )
            continue
        entry["name"] = str(entry.get("name") or f"auto:{tools[0]}")
        entry["tools"] = tools
        entry["keywords"] = keywords
        entry["exclude_keywords"] = _string_list(entry.get("exclude_keywords"))
        out.append(entry)
    return out


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
    where = f" {scope!r}" if scope else ""
    logger.warning(
        "tool-belt: learned scope%s names %d tool(s) in both carry and "
        "expand_only; resolving toward carry: %s",
        where, len(overlap), ", ".join(overlap),
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
    # Learned trigger overlay — only kept when non-empty so
    # untouched scopes stay byte-identical to their pre-overlay shape.
    overlay = _normalize_overlay_triggers(entry.get("triggers"))
    if overlay:
        out["triggers"] = overlay
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

    # Preserve a normalized global block only if it carries something.
    # NOTE: inert today — no consumer reads it (scope_state /
    # apply_to_preset look at ``scopes`` only).
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
    """Atomically persist learned state as schema v2 — the single writer.

    Every production write of ``learned.json`` (shaper merge — manual and
    auto-shape — configure flows, inventory reconciliation) routes through
    here. The document is normalized
    on write: v1 spellings are renamed to their v2 fields, a ``carry`` ∩
    ``expand_only`` overlap resolves toward carrying, unrelated scopes and
    metadata are preserved, and the version is stamped to v2.

    The write is atomic and durable: a uniquely named temp file in the target
    directory (two concurrent writers can never clobber each other's temp),
    flushed and fsynced, then renamed over the target. The temp file is
    removed if anything fails before the rename.
    """
    target = path or learned_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = normalize_state(dict(state or {}))
    payload["version"] = LEARNED_VERSION
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    load_state(force=True)


#: Adaptive shaping keys a reset clears from a scope — the v2 assignments and
#: evidence plus their v1 spellings. Everything else in a scope dict (and
#: every other scope) is unrelated metadata a reset must preserve.
_ADAPTIVE_SCOPE_KEYS = frozenset(
    {"carry", "expand_only", "shaping", "always_on", "always_off", "cache_aware",
     # The learned trigger overlay is adaptive state too: a scope reset clears
     # its auto-learned triggers along with the carrying assignments. Shipped
     # policy.yaml triggers live elsewhere and are untouched by a reset.
     "triggers"}
)


def reset_scope(state: dict[str, Any], scope: str) -> tuple[dict[str, Any], bool]:
    """Drop one scope's adaptive shaping assignments/evidence, in memory.

    The single reset semantic for learned state — configure's reset flow and
    any other reset path route through here rather than hand-rolling their own.
    Removes only the adaptive carrying keys (``carry`` / ``expand_only`` /
    ``shaping`` and their v1 spellings) from the named scope. Any other key in that
    scope's dict is unrelated metadata and is preserved; if the scope becomes
    empty it is dropped entirely. Every *other* scope, the top-level metadata,
    and the always-carry policy / trigger definitions (which live in policy.yaml,
    never here) are untouched. Returns ``(new_state, changed)``; no write happens.
    """
    if not isinstance(state, dict):
        return {}, False
    new_state = dict(state)
    scopes_in = new_state.get("scopes")
    if not isinstance(scopes_in, dict) or scope not in scopes_in:
        return new_state, False

    scopes = dict(scopes_in)
    entry = scopes.get(scope)
    if not isinstance(entry, dict):
        scopes.pop(scope, None)
        new_state["scopes"] = scopes
        return new_state, True

    remaining = {k: v for k, v in entry.items() if k not in _ADAPTIVE_SCOPE_KEYS}
    changed = len(remaining) != len(entry)
    if remaining:
        scopes[scope] = remaining
    else:
        scopes.pop(scope, None)
        changed = changed or scope in scopes_in
    new_state["scopes"] = scopes
    return new_state, changed


def prune_tool(state: dict[str, Any], tool: str) -> tuple[dict[str, Any], bool]:
    """Remove every learned reference to ``tool`` across all scopes, in memory.

    Inventory reconciliation: a tool that has vanished from the
    install's registry is cleaned out of learned state after the grace period.
    Removes the tool from each scope's ``carry`` / ``expand_only`` lists, from
    the recorded ``shaping`` promote/demote evidence rows, and from learned
    trigger-overlay entries (an overlay entry whose ``tools`` list empties is
    dropped entirely). Other tools, scopes, and metadata are untouched.
    Returns ``(new_state, changed)``; no write happens here.
    """
    if not isinstance(state, dict):
        return {}, False
    tool = str(tool or "").strip()
    scopes_in = state.get("scopes")
    if not tool or not isinstance(scopes_in, dict):
        return state, False

    changed = False
    new_scopes: dict[str, Any] = {}
    for scope, entry in scopes_in.items():
        if not isinstance(entry, dict):
            new_scopes[scope] = entry
            continue
        new_entry = dict(entry)
        for key in ("carry", "expand_only", "always_on", "always_off"):
            values = new_entry.get(key)
            if isinstance(values, list) and tool in values:
                new_entry[key] = [t for t in values if t != tool]
                changed = True
        shaping = new_entry.get("shaping")
        if isinstance(shaping, dict):
            new_shaping = dict(shaping)
            for key in ("promote", "demote"):
                rows = new_shaping.get(key)
                if isinstance(rows, list):
                    kept = [r for r in rows
                            if not (isinstance(r, dict) and r.get("tool") == tool)]
                    if len(kept) != len(rows):
                        new_shaping[key] = kept
                        changed = True
            new_entry["shaping"] = new_shaping
        overlay = new_entry.get("triggers")
        if isinstance(overlay, list):
            kept_overlay: list[Any] = []
            for group in overlay:
                if isinstance(group, dict) and isinstance(group.get("tools"), list) \
                        and tool in group["tools"]:
                    remaining = [t for t in group["tools"] if t != tool]
                    changed = True
                    if remaining:
                        g = dict(group)
                        g["tools"] = remaining
                        kept_overlay.append(g)
                    # else: the overlay entry only activated the vanished tool
                    # — drop it entirely.
                else:
                    kept_overlay.append(group)
            new_entry["triggers"] = kept_overlay
        new_scopes[scope] = new_entry

    if not changed:
        return state, False
    new_state = dict(state)
    new_state["scopes"] = new_scopes
    return new_state, True


def scope_state(state: dict[str, Any], scope: str) -> tuple[str, dict[str, Any]]:
    scopes = state.get("scopes") or {}
    if not isinstance(scopes, dict):
        return "", {}
    for key in scope_candidates(scope):
        value = scopes.get(key)
        if isinstance(value, dict):
            return key, value
    return "", {}


def config_always_carry(plugin_config: dict[str, Any], scope: str) -> list[str]:
    """Per-agent always_carry config pins for a scope, in stable order.

    The union of the global ``always_carry`` setting and the
    scope's ``channels.<scope>.always_carry`` list (platform fallback via
    :func:`scope_candidates`; the first matching channel entry contributes).
    Union semantics only — a scope entry can add pins, never remove the
    global ones. Names are not validated here: an unknown/disabled name is
    inert at resolution (``∩ E`` in ``carrying.resolve``) and warned about at
    request time when the live ceiling is known.
    """
    pins = _string_list(plugin_config.get("always_carry"))
    channels = plugin_config.get("channels") or {}
    if isinstance(channels, dict):
        for key in scope_candidates(scope):
            cfg = channels.get(key)
            if isinstance(cfg, dict) and "always_carry" in cfg:
                for tool in _string_list(cfg.get("always_carry")):
                    if tool not in pins:
                        pins.append(tool)
                break
    # De-dupe, order-preserving.
    return list(dict.fromkeys(pins))


def _compile_overlay_groups(
    scoped: dict[str, Any], always_carry_set: set[str], scope: str
) -> list[TriggerGroup]:
    """Compile a scope's learned trigger overlay into runtime TriggerGroups.

    Additive only: the compiled groups are appended to the shipped policy
    triggers at resolution time — never replacing or editing them. The
    overlay is structurally activation-only: a trigger group can only put a
    tool into ``T = triggered ∩ X`` (see ``carrying.resolve``), so it cannot
    demote, disable, or resurrect a tool outside the enabled ceiling ``E``;
    an ``always_carry`` tool named here is filtered out (it is already
    resident — activating it is meaningless). Exclude keywords (dampeners)
    compile exactly like policy ones.
    """
    from .presets import _compile_keywords

    groups: list[TriggerGroup] = []
    for entry in _normalize_overlay_triggers(scoped.get("triggers")):
        tools = [t for t in entry["tools"] if t not in always_carry_set]
        if not tools:
            continue
        patterns = _compile_keywords(entry["keywords"])
        if not patterns:
            logger.warning(
                "tool-belt: learned trigger overlay entry %r for scope %r has "
                "no compilable keyword — skipped",
                entry["name"], scope,
            )
            continue
        groups.append(TriggerGroup(
            name=entry["name"],
            tools=tools,
            keyword_patterns=patterns,
            exclude_patterns=_compile_keywords(entry["exclude_keywords"]),
            has_attachment=None,
        ))
    return groups


def apply_to_preset(preset: Preset, plugin_config: dict[str, Any], scope: str) -> LearnedMergeResult:
    """Merge config pins + the learned overlay into a resolved preset.

    The single home of carrying precedence (full-start contract)::

        effective_always_carry = policy always_carry ∪ config always_carry
        demoted  = learned expand_only − learned carry − effective_always_carry
        carry    = learned carry − effective_always_carry

    The bulk of adaptive residency is *implicit*: ``carrying.resolve``
    computes class C as ``((E − A) − demoted) ∪ carry`` against the live
    ceiling, so with no learned state everything enabled is carried.
    Config pins apply in every mode; the learned demotion/promotion overlay
    merges only when ``learned_mode`` resolves to ``apply`` (the default) —
    ``recommend`` is the opt-in observe/trial mode and never narrows.

    ``always_carry`` (policy ∪ config) wins every conflict and is immutable —
    undemotable by construction. Trigger definitions are copied byte-for-byte;
    promotion/demotion never touch a trigger group.
    """
    mode = learned_mode(plugin_config, scope)
    version = f"{preset.name}+learned:{state_hash() or 'none'}"

    # Config always_carry pins join the structural baseline in EVERY mode.
    pins = config_always_carry(plugin_config, scope)
    new_pins = [t for t in pins if t not in set(preset.always_carry)]
    if new_pins and not preset.no_narrowing:
        preset = Preset(
            name=preset.name,
            always_carry=[*preset.always_carry, *new_pins],
            carry=[t for t in preset.carry if t not in set(new_pins)],
            triggers=preset.triggers,
            demoted=[t for t in getattr(preset, "demoted", []) if t not in set(new_pins)],
        )
    setattr(preset, "config_always_carry", pins)

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

    # always_carry (policy ∪ config pins) is immutable — a demotion signal
    # naming one is ignored + warned. This is the undemotable-by-construction
    # guarantee for config-pinned tools.
    demoted_always_carry = always_carry_set & set(learned_expand)
    if demoted_always_carry:
        logger.warning(
            "tool-belt: learned expand_only names always_carry tool(s) %s — "
            "ignored (always_carry is immutable and wins every conflict)",
            ", ".join(sorted(demoted_always_carry)),
        )
    learned_expand = [t for t in learned_expand if t not in always_carry_set]

    # Centralized precedence (full-start): the demotion list is the ONLY thing
    # that moves an enabled tool out of residency; explicit carry entries
    # (learned promotions, plus any legacy policy carry) only ever add.
    expand_set = set(learned_expand)
    effective_carry = [t for t in policy_carry if t not in expand_set]
    for tool in learned_carry:
        if tool not in effective_carry:
            effective_carry.append(tool)
    effective_carry = [t for t in effective_carry if t not in always_carry_set]
    effective_demoted = [t for t in learned_expand if t not in set(effective_carry)]

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

    # Learned trigger overlay: auto-learned groups UNION with the
    # shipped policy triggers — additive, activation-only, per scope. The
    # shipped policy.yaml is never edited.
    overlay_groups = _compile_overlay_groups(scoped, always_carry_set, matched_scope or scope)
    triggers.extend(overlay_groups)

    # Change log (telemetry): what moved relative to the full-start baseline
    # (everything enabled carried). Demotions are the shaping motion; explicit
    # carry entries record promotions/un-demotions.
    policy_carry_set = set(policy_carry)
    changes: list[str] = []
    for tool in effective_carry:
        if tool not in policy_carry_set:
            changes.append(f"{tool}:carry")
    for tool in effective_demoted:
        changes.append(f"{tool}:expand_only")

    if not changes and not overlay_groups:
        return LearnedMergeResult(
            preset=preset, mode=mode, policy_version=version, learned_scope=matched_scope,
        )

    merged = Preset(
        name=f"{preset.name}+learned[{matched_scope or scope}]",
        always_carry=always_carry,
        carry=effective_carry,
        triggers=triggers,
        demoted=effective_demoted,
    )
    setattr(merged, "config_always_carry", pins)
    return LearnedMergeResult(
        preset=merged,
        mode=mode,
        policy_source="learned",
        policy_version=version,
        learned_changes=changes,
        learned_scope=matched_scope,
    )
