"""Tool-policy loading and per-scope override resolution (Tool Belt 1.0).

There's one policy YAML (``policy.yaml`` at the plugin root). Under the 1.0
carrying model it declares a two-way resident partition plus trigger groups:

  - ``always_carry``: immutable residents. They win every conflict and are
    never shaped by the learned/adaptive layer (partition class ``A`` once
    intersected with Hermes's enabled built-in ceiling ``E``).
  - ``carry``: adaptive residents. Promotion/demotion move tools across the
    ``carry`` ⇄ ``expand_only`` boundary (partition class ``C``). Trigger
    definitions never change during either transition.
  - ``triggers``: trigger groups that *activate* an enabled-but-not-resident
    (``expand_only``) tool for a single message without changing its residency.

``expand_only`` (class ``X``) is *derived* — the enabled remainder
``E − (A ∪ C)`` — and is only knowable when Hermes supplies ``E`` at request
time. Tool Belt has no disabling semantics: absent/disabled tools are owned by
Hermes's ceiling, not by policy.

Legacy compatibility: ``Preset`` also accepts ``always_on=`` / ``always_off=``
at construction and exposes ``.always_on`` / ``.always_off`` / ``.is_wildcard``
as read-only views. The runtime filter (``predictor.py``, ``__init__.py``)
still consumes ``.always_on`` / ``.is_wildcard``; these views are derived from
the canonical fields, never authoritative. ``.always_on`` is the resident union
``always_carry ∪ carry`` — precisely "what loads on every message".

Per-scope narrowing is driven by the learned overlay (``learned.py``). The old
config-level ``always_on_extra`` / ``always_off`` promote-and-disable knobs are
gone: a detected legacy value is warned about, never silently applied.

The "kill switch" — disable narrowing entirely for a scope — is
``bypass_rate: 1.0`` (handled in ``__init__.py``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Single shipped policy at the plugin root. Hand-curated.
_POLICY_FILE = Path(__file__).parent / "policy.yaml"

# Runtime "allow everything in the user's ceiling" sentinel. This is the value
# a *Prediction* (``predictor.py``) stamps onto ``allowed_tool_names`` on the
# fail-open / bypass path, and ``__init__.py`` compares against it. It is NOT
# part of the Preset domain model — a no-narrowing Preset is expressed by the
# ``no_narrowing`` flag below.
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


class Preset:
    """A resolved preset — the canonical Tool Belt 1.0 carrying model.

    Canonical fields:
      · ``always_carry`` — immutable residents (source of partition class A).
        Win every conflict; never shaped by the learned/adaptive layer.
      · ``carry``        — adaptive residents (source of class C). Promotion/
        demotion mutate this loadout; trigger definitions never change.
      · ``triggers``     — trigger groups (``expand_only`` activation).
      · ``no_narrowing`` — neutral "load the whole ceiling" flag. The model's
        replacement for the old ``always_on == "*"`` wildcard sentinel.

    Legacy compatibility (transitional; see the module docstring): the
    constructor still accepts ``always_on=`` / ``always_off=`` and the read
    views ``.always_on`` / ``.always_off`` / ``.is_wildcard`` are derived from
    the canonical fields for un-rewired runtime filtering.
    """

    def __init__(
        self,
        name: str,
        always_carry: list[str] | None = None,
        carry: list[str] | None = None,
        triggers: list[TriggerGroup] | None = None,
        no_narrowing: bool = False,
        *,
        always_on: list[str] | str | None = None,
        always_off: list[str] | None = None,
    ) -> None:
        self.name = str(name)
        self.triggers: list[TriggerGroup] = list(triggers) if triggers else []
        self.no_narrowing = bool(no_narrowing)

        legacy_off = [str(t) for t in (always_off or []) if isinstance(t, str)]

        if always_on is not None and always_carry is None and carry is None:
            # Legacy construction: a single "load on every message" list (or the
            # wildcard sentinel). Fold it into the adaptive ``carry`` loadout —
            # the immutable split isn't expressible in the pre-1.0 shape.
            if always_on == WILDCARD_ALWAYS_ON:
                self.no_narrowing = True
                self.always_carry: list[str] = []
                self.carry: list[str] = []
            elif isinstance(always_on, list):
                self.always_carry = []
                self.carry = [str(t) for t in always_on if isinstance(t, str)]
            else:
                self.always_carry = []
                self.carry = []
        else:
            self.always_carry = [str(t) for t in (always_carry or []) if isinstance(t, str)]
            self.carry = [str(t) for t in (carry or []) if isinstance(t, str)]

        # Legacy disable list. Empty under the 1.0 model; a handful of pre-1.0
        # tests still construct a preset with an explicit ``always_off=``.
        self._legacy_always_off = legacy_off

    # ─── Legacy read views (transitional) ──────────────────────────────────
    @property
    def always_on(self) -> list[str] | str:
        """Residents loaded on every message: ``always_carry ∪ carry``.

        Returns the wildcard sentinel for a no-narrowing preset, preserving the
        exact shape the pre-1.0 runtime branched on.
        """
        if self.no_narrowing:
            return WILDCARD_ALWAYS_ON
        out = list(self.always_carry)
        for tool in self.carry:
            if tool not in out:
                out.append(tool)
        return out

    @property
    def always_off(self) -> list[str]:
        return list(self._legacy_always_off)

    @property
    def is_wildcard(self) -> bool:
        return self.no_narrowing

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"Preset(name={self.name!r}, always_carry={self.always_carry!r}, "
            f"carry={self.carry!r}, triggers={len(self.triggers)}, "
            f"no_narrowing={self.no_narrowing!r})"
        )


def _tool_list(raw: Any) -> list[str]:
    """Coerce a YAML list to a list of tool-name strings."""
    if not isinstance(raw, list):
        return []
    return [str(t) for t in raw if isinstance(t, str)]


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
    """Read and parse a policy YAML into the 1.0 carrying model.

    Parses the v2 schema (``always_carry`` + ``carry``). A pre-1.0 policy that
    only has ``always_on`` is folded into ``carry`` so an un-migrated file still
    loads. Raises on missing file or bad shape.
    """
    import yaml  # type: ignore[import-untyped]
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"preset {path} is not a YAML mapping")
    name = str(data.get("name") or path.stem)

    no_narrowing = False
    always_carry = _tool_list(data.get("always_carry"))
    carry = _tool_list(data.get("carry"))

    if data.get("always_carry") == WILDCARD_ALWAYS_ON or data.get("carry") == WILDCARD_ALWAYS_ON:
        # A wildcard carrying baseline means "no narrowing".
        no_narrowing = True
        always_carry, carry = [], []
    elif not always_carry and not carry and "always_on" in data:
        # Pre-1.0 policy: a single always_on list (or wildcard) → adaptive carry.
        legacy = data.get("always_on")
        if legacy == WILDCARD_ALWAYS_ON:
            no_narrowing = True
        else:
            carry = _tool_list(legacy)

    # always_carry wins every conflict — a tool named in both is a resident of A.
    carry = [t for t in carry if t not in set(always_carry)]

    return Preset(
        name=name,
        always_carry=always_carry,
        carry=carry,
        triggers=_parse_triggers(data.get("triggers")),
        no_narrowing=no_narrowing,
    )


def load_base_policy() -> Preset:
    """Load the single shipped policy from ``policy.yaml``.

    If the file is missing or malformed, return a no-narrowing preset — i.e.
    behave as if narrowing were disabled. Failing safe matters more than
    failing loud here; a broken policy file shouldn't break the gateway.
    """
    try:
        return load_preset_file(_POLICY_FILE)
    except Exception as exc:
        logger.warning(
            "tool-belt: failed to load policy.yaml (%s) — falling back to no-narrowing",
            exc,
        )
        return Preset(name="no-narrowing-fallback", no_narrowing=True)


def resolve_preset(plugin_config: dict[str, Any], channel: str) -> Preset:
    """Resolve the effective policy for a given scope.

    Resolution order (later overrides earlier):
      1. Base policy from ``policy.yaml`` (``always_carry`` + ``carry``).
      2. Learned overlay, when ``learned_mode`` is ``apply`` — the centralized
         precedence lives in :func:`learned.apply_to_preset`.

    The pre-1.0 config-level ``always_on_extra`` / ``always_off`` knobs are no
    longer inputs; a stale value is warned about (see
    :func:`_warn_legacy_disable_inputs`), never applied.

    Returns a fully-resolved :class:`Preset`. Never raises; falls back to a
    no-narrowing preset on any failure so sessions don't break. To disable
    narrowing intentionally on a scope, set ``bypass_rate: 1.0`` — that path is
    handled in ``__init__.py``.
    """
    try:
        return _resolve_preset_inner(plugin_config, channel)
    except Exception as exc:
        logger.warning(
            "tool-belt: policy resolution failed for scope=%r, falling back to no-narrowing: %s",
            channel, exc,
        )
        return Preset(name="no-narrowing-fallback", no_narrowing=True)


def _resolve_preset_inner(plugin_config: dict[str, Any], channel: str) -> Preset:
    channels_cfg = plugin_config.get("channels") or {}
    channel_cfg = _channel_config(channels_cfg, channel)

    preset = load_base_policy()

    if preset.no_narrowing:
        # No-narrowing fallback (e.g. policy.yaml missing) — nothing to layer.
        return preset

    # Surface (never apply) removed pre-1.0 disable/promote config knobs.
    _warn_legacy_disable_inputs(plugin_config, channel_cfg, channel)

    # Learned overlay is imported lazily to avoid a module import cycle.
    try:
        from . import learned as learned_mod
        result = learned_mod.apply_to_preset(preset, plugin_config, channel)
        setattr(result.preset, "learned_mode", result.mode)
        setattr(result.preset, "policy_source", result.policy_source)
        setattr(result.preset, "policy_version", result.policy_version)
        setattr(result.preset, "learned_changes", result.learned_changes)
        setattr(result.preset, "learned_scope", result.learned_scope)
        return result.preset
    except Exception as exc:
        logger.warning("tool-belt: learned state merge failed for %r: %s", channel, exc)
        return preset


def _warn_legacy_disable_inputs(
    plugin_config: dict[str, Any],
    channel_cfg: dict[str, Any],
    channel: str,
) -> None:
    """Warn (never silently apply) when removed pre-1.0 config knobs are set.

    ``always_on_extra`` (promote-extra) and config-level ``always_off``
    (disable) are no longer runtime inputs under the 1.0 carrying model: the
    resident partition comes from policy ``always_carry`` / ``carry`` plus the
    learned overlay, and disabling is owned by Hermes's ceiling. A stale value
    in config is surfaced, not consumed.
    """
    detected: list[str] = []
    for label, cfg in (("global", plugin_config), (f"scope {channel!r}", channel_cfg)):
        if not isinstance(cfg, dict):
            continue
        if cfg.get("always_on_extra"):
            detected.append(f"always_on_extra ({label})")
        if cfg.get("always_off"):
            detected.append(f"always_off ({label})")
    if detected:
        logger.warning(
            "tool-belt: ignoring removed pre-1.0 config knob(s) %s — Tool Belt 1.0 "
            "resident policy is always_carry/carry plus the learned overlay; "
            "disabling is owned by Hermes's ceiling",
            ", ".join(detected),
        )


def _channel_config(channels_cfg: Any, channel: str) -> dict[str, Any]:
    """Return config for a scope, falling back to its platform segment.

    A scope key may be given either fully qualified (``agent:platform``) or as
    a bare platform segment; the qualified key wins.
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
