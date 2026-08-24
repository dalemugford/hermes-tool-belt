"""Tool-policy loading and per-scope override resolution.

There's one policy YAML (``policy.yaml`` at the plugin root) that defines:
  - ``always_on``: list of tool names always loaded (or the literal "*" for
    "everything in user's ceiling" — used internally by the bypass path).
  - ``triggers``: list of trigger groups, each with a ``name``, ``tools`` to
    add when fired, and signals (``keywords``, ``exclude_keywords``,
    ``has_attachment``).

Per-scope overrides live in ~/.hermes/config.yaml under
``plugins.tool-belt.channels.<scope>``. Override schema mirrors the
policy schema. Scope-level keys win over global. Missing keys fall through
to the base policy.

The "kill switch" — disable narrowing entirely for a scope — is
``bypass_rate: 1.0`` (global or per-scope). No more named-preset modes.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Single shipped policy at the plugin root. Hand-curated.
_POLICY_FILE = Path(__file__).parent / "policy.yaml"

# Sentinel value for always_on meaning "load everything in the user's ceiling".
# Kept because the bypass cohort path stamps this directly onto state without
# going through the policy resolver.
WILDCARD_ALWAYS_ON = "*"


@dataclass
class TriggerGroup:
    """A single trigger group within a preset."""
    name: str
    tools: list[str] = field(default_factory=list)
    keyword_patterns: list[re.Pattern[str]] = field(default_factory=list)
    exclude_patterns: list[re.Pattern[str]] = field(default_factory=list)
    has_attachment: str | None = None  # "image" or None

    def is_excluded(self, message: str) -> bool:
        """Return True if any exclude pattern vetoes this trigger."""
        for pat in self.exclude_patterns:
            if pat.search(message):
                return True
        return False

    def would_fire_positive(self, message: str, attachments: Iterable[str] | None = None) -> bool:
        """Return True if positive signals (attachment or keyword) would fire, ignoring excludes."""
        if self.has_attachment and attachments:
            if self.has_attachment in attachments:
                return True
        for pat in self.keyword_patterns:
            if pat.search(message):
                return True
        return False

    def matches(self, message: str, attachments: Iterable[str] | None = None) -> bool:
        """Return True if this trigger fires for the given message + attachments.

        Exclude patterns are checked first; any match vetoes the trigger even
        if positive keyword patterns would otherwise match.
        """
        if self.exclude_patterns and self.is_excluded(message):
            return False
        return self.would_fire_positive(message, attachments)


@dataclass
class Preset:
    """A resolved preset — what the predictor consumes."""
    name: str
    always_on: list[str] | str  # list of tool names, or WILDCARD_ALWAYS_ON
    triggers: list[TriggerGroup] = field(default_factory=list)
    # Tools explicitly forced off (deprecated / superseded). These are added
    # to the "known" set so the unknown-tool safe-default cuts them instead
    # of silently keeping them on. They are never added to always_on or any
    # trigger, so they can only re-enter the ceiling via expand_tools.
    always_off: list[str] = field(default_factory=list)

    @property
    def is_wildcard(self) -> bool:
        return self.always_on == WILDCARD_ALWAYS_ON


def _compile_keywords(raw: list[str] | None) -> list[re.Pattern[str]]:
    """Compile a list of raw regex strings, ignoring case. Skips bad ones."""
    out: list[re.Pattern[str]] = []
    if not raw:
        return out
    for kw in raw:
        if not isinstance(kw, str):
            continue
        try:
            out.append(re.compile(kw, flags=re.IGNORECASE))
        except re.error as exc:
            logger.warning("tool-belt: bad regex %r: %s", kw, exc)
    return out


def _parse_triggers(raw: Any) -> list[TriggerGroup]:
    if not isinstance(raw, list):
        return []
    out: list[TriggerGroup] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or "unnamed"
        tools = entry.get("tools") or []
        if not isinstance(tools, list):
            continue
        out.append(TriggerGroup(
            name=str(name),
            tools=[str(t) for t in tools if isinstance(t, str)],
            keyword_patterns=_compile_keywords(entry.get("keywords")),
            exclude_patterns=_compile_keywords(entry.get("exclude_keywords")),
            has_attachment=entry.get("has_attachment"),
        ))
    return out


def load_preset_file(path: Path) -> Preset:
    """Read and parse a preset YAML. Raises on missing file or bad shape."""
    import yaml  # type: ignore[import-untyped]
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"preset {path} is not a YAML mapping")
    name = str(data.get("name") or path.stem)
    always_on_raw = data.get("always_on", [])
    if always_on_raw == WILDCARD_ALWAYS_ON:
        always_on: list[str] | str = WILDCARD_ALWAYS_ON
    elif isinstance(always_on_raw, list):
        always_on = [str(t) for t in always_on_raw if isinstance(t, str)]
    else:
        always_on = []
    always_off_raw = data.get("always_off", [])
    always_off = (
        [str(t) for t in always_off_raw if isinstance(t, str)]
        if isinstance(always_off_raw, list)
        else []
    )
    return Preset(
        name=name,
        always_on=always_on,
        triggers=_parse_triggers(data.get("triggers")),
        always_off=always_off,
    )


def load_base_policy() -> Preset:
    """Load the single shipped policy from ``policy.yaml``.

    If the file is missing or malformed, return a wildcard preset — i.e.
    behave as if narrowing were disabled. Failing safe matters more than
    failing loud here; a broken policy file shouldn't break the gateway.
    """
    try:
        return load_preset_file(_POLICY_FILE)
    except Exception as exc:
        logger.warning(
            "tool-belt: failed to load policy.yaml (%s) — falling back to wildcard (no narrowing)",
            exc,
        )
        return Preset(name="wildcard-fallback", always_on=WILDCARD_ALWAYS_ON, triggers=[])


def resolve_preset(
    plugin_config: dict[str, Any],
    channel: str,
) -> Preset:
    """Resolve the effective policy for a given scope.

    Resolution order (later overrides earlier):
      1. Base policy from ``policy.yaml``
      2. Top-level overrides: ``plugin_config["always_on_extra"]``,
         ``plugin_config["always_off"]``
      3. Per-scope: ``plugin_config["channels"][scope]`` may set its own
         ``always_on_extra``, ``always_off``.
      4. Learned state, when ``learned_mode`` is ``auto`` or ``audit``.

    Returns a fully-resolved :class:`Preset`. Never raises; falls back to
    a wildcard (no-narrowing) preset on any failure so sessions don't
    break. To disable narrowing intentionally on a scope, set
    ``bypass_rate: 1.0`` — that path is handled in ``__init__.py``.
    """
    try:
        return _resolve_preset_inner(plugin_config, channel)
    except Exception as exc:
        logger.warning(
            "tool-belt: policy resolution failed for scope=%r, falling back to wildcard: %s",
            channel, exc,
        )
        return Preset(name="wildcard-fallback", always_on=WILDCARD_ALWAYS_ON, triggers=[])


def _resolve_preset_inner(plugin_config: dict[str, Any], channel: str) -> Preset:
    channels_cfg = plugin_config.get("channels") or {}
    channel_cfg = _channel_config(channels_cfg, channel)

    preset = load_base_policy()

    if preset.is_wildcard:
        # Wildcard fallback (e.g. policy.yaml missing) — no overrides to layer.
        return preset

    # Apply additive overrides — global first, then scope-specific.
    extra_global: list[str] = list(plugin_config.get("always_on_extra") or [])
    off_global: list[str] = list(plugin_config.get("always_off") or [])
    extra_channel: list[str] = list(channel_cfg.get("always_on_extra") or [])
    off_channel: list[str] = list(channel_cfg.get("always_off") or [])

    always_on = list(preset.always_on) if isinstance(preset.always_on, list) else []
    for extra in (extra_global, extra_channel):
        for t in extra:
            if isinstance(t, str) and t not in always_on:
                always_on.append(t)
    # Config-level always_off removes a tool from always_on AND names it in
    # the preset's always_off set so it lands in the "known" bucket and is
    # cut (not kept as an unknown). Merge with the policy.yaml always_off.
    always_off = list(preset.always_off)
    for off in (off_global, off_channel):
        for t in off:
            if not isinstance(t, str):
                continue
            if t in always_on:
                always_on.remove(t)
            if t not in always_off:
                always_off.append(t)

    resolved = Preset(
        name=f"{preset.name}+overrides[{channel}]",
        always_on=always_on,
        triggers=preset.triggers,
        always_off=always_off,
    )

    # Learned state is imported lazily to avoid a module import cycle.
    try:
        from . import learned as learned_mod
        result = learned_mod.apply_to_preset(resolved, plugin_config, channel)
        setattr(result.preset, "learned_mode", result.mode)
        setattr(result.preset, "policy_source", result.policy_source)
        setattr(result.preset, "policy_version", result.policy_version)
        setattr(result.preset, "learned_changes", result.learned_changes)
        setattr(result.preset, "learned_scope", result.learned_scope)
        return result.preset
    except Exception as exc:
        logger.warning("tool-belt: learned state merge failed for %r: %s", channel, exc)
        return resolved


def _channel_config(channels_cfg: Any, channel: str) -> dict[str, Any]:
    """Return config for a scope, falling back to its platform segment.

    Existing configs often use ``telegram``; the agent/platform scope is
    ``bernard:telegram``. Supporting both keeps rollout non-disruptive.
    """
    if not isinstance(channels_cfg, dict):
        return {}
    keys = [str(channel or "").strip().lower()]
    if keys[0] and ":" in keys[0]:
        keys.append(keys[0].rsplit(":", 1)[-1])
    for key in keys:
        value = channels_cfg.get(key)
        if isinstance(value, dict):
            return value
    return {}
