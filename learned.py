"""Persistent learned policy state for tool-belt.

The learned layer is deliberately small and inspectable: a JSON file under
``$HERMES_HOME/state/tool-belt/learned.json``. Prediction only reads from
it. Analyzer or explicit user action writes it.
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

LEARNED_VERSION = 1
_ALLOWED_MODES = {"off", "recommend", "auto", "audit"}
_APPLY_MODES = {"auto", "audit"}
_CACHE: dict[str, Any] = {"path": None, "mtime_ns": None, "state": None, "hash": ""}


@dataclass
class LearnedMergeResult:
    """Result metadata from applying learned state to a preset."""

    preset: Preset
    mode: str = "off"
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
    mode = str(value or "off").strip().lower()
    return mode if mode in _ALLOWED_MODES else "off"


def scope_candidates(scope: str) -> list[str]:
    """Return lookup order for a scope.

    ``bernard:telegram`` should win over ``telegram``. The platform fallback
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
    mode = normalize_mode(plugin_config.get("learned_mode", "off"))
    channels = plugin_config.get("channels") or {}
    if isinstance(channels, dict):
        for key in scope_candidates(scope):
            cfg = channels.get(key)
            if isinstance(cfg, dict) and "learned_mode" in cfg:
                return normalize_mode(cfg.get("learned_mode"))
    return mode


def load_state(force: bool = False) -> dict[str, Any]:
    """Load learned.json with mtime caching. Missing/invalid means empty state."""
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
        if int(data.get("version", LEARNED_VERSION)) != LEARNED_VERSION:
            logger.warning("tool-belt: unsupported learned state version %r", data.get("version"))
            data = {}
        digest = hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
        _CACHE.update({"path": str(path), "mtime_ns": stat.st_mtime_ns, "state": data, "hash": digest})
        return deepcopy(data)
    except Exception as exc:
        logger.warning("tool-belt: failed to load learned state %s: %s", path, exc)
        _CACHE.update({"path": str(path), "mtime_ns": stat.st_mtime_ns, "state": {}, "hash": ""})
        return {}


def state_hash() -> str:
    load_state()
    return str(_CACHE.get("hash") or "")


def write_state(state: dict[str, Any], path: Path | None = None) -> None:
    """Atomically write learned state. Intended for analyzer/manual commands."""
    target = path or learned_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state or {})
    payload.setdefault("version", LEARNED_VERSION)
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
    """Merge learned state into a resolved preset when learned_mode applies."""
    mode = learned_mode(plugin_config, scope)
    version = f"{preset.name}+learned:{state_hash() or 'none'}"
    if mode not in _APPLY_MODES or preset.is_wildcard:
        return LearnedMergeResult(preset=preset, mode=mode, policy_version=version)

    state = load_state()
    matched_scope, scoped = scope_state(state, scope)
    global_cfg = state.get("global") if isinstance(state.get("global"), dict) else {}

    always_on = list(preset.always_on) if isinstance(preset.always_on, list) else []
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
    changes: list[str] = []

    learned_on = _string_list(scoped.get("always_on"))
    learned_off = set(_string_list(global_cfg.get("always_off"))) | set(_string_list(scoped.get("always_off")))

    for tool in learned_on:
        if tool not in always_on and tool not in learned_off:
            always_on.append(tool)
            changes.append(f"{tool}:always_on")

    if learned_off:
        before = set(always_on)
        always_on = [tool for tool in always_on if tool not in learned_off]
        for tool in sorted(before - set(always_on)):
            changes.append(f"{tool}:always_off")
        for group in triggers:
            original = list(group.tools)
            group.tools = [tool for tool in group.tools if tool not in learned_off]
            for tool in sorted(set(original) - set(group.tools)):
                changes.append(f"{tool}:removed_from_trigger:{group.name}")

    adjustments = scoped.get("trigger_adjustments") or {}
    if isinstance(adjustments, dict):
        disabled: set[str] = set()
        for name, adjustment in adjustments.items():
            action = ""
            if isinstance(adjustment, dict):
                action = str(adjustment.get("action") or "").strip().lower()
            elif isinstance(adjustment, str):
                action = adjustment.strip().lower()
            if action in {"demote", "disable"}:
                disabled.add(str(name))
        if disabled:
            before_names = {group.name for group in triggers}
            triggers = [group for group in triggers if group.name not in disabled]
            for name in sorted(before_names - {group.name for group in triggers}):
                changes.append(f"trigger:{name}:disabled")

    if not changes:
        return LearnedMergeResult(preset=preset, mode=mode, policy_version=version, learned_scope=matched_scope)

    merged_off = list(getattr(preset, "always_off", []) or [])
    for tool in sorted(learned_off):
        if tool not in merged_off:
            merged_off.append(tool)
    merged = Preset(
        name=f"{preset.name}+learned[{matched_scope or scope}]",
        always_on=always_on,
        triggers=triggers,
        always_off=merged_off,
    )
    return LearnedMergeResult(
        preset=merged,
        mode=mode,
        policy_source="learned",
        policy_version=version,
        learned_changes=changes,
        learned_scope=matched_scope,
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
