"""Predictor: regex/keyword classifier with image-attachment signal.

Given a message, returns the set of tool names the model is likely to need
this turn. The result is the union of:

  · The preset's residents (``always_carry`` ∪ ``carry``)
  · Tools from every trigger group whose signals match the message

Always returns a result. Never raises — on any error, falls back to
WILDCARD_ALWAYS_ON so the gateway behaves as if the plugin weren't installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

from .presets import Preset, WILDCARD_ALWAYS_ON

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """The output of a predictor run."""

    # Tool names the model is likely to need this turn — the candidate active
    # set (residents ∪ trigger-activated tools). Either a list, or the
    # WILDCARD_ALWAYS_ON sentinel meaning "no narrowing — load the whole
    # enabled ceiling". The final partition against the live enabled ceiling
    # ``E`` is computed in ``carrying.resolve`` at request-build time.
    active_tool_names: list[str] | str

    # Names of trigger groups that fired (for logging / future tightening)
    triggers_fired: list[str] = field(default_factory=list)

    # Names of trigger groups suppressed by exclude_keywords (positive signals matched
    # but an exclude pattern vetoed the trigger — logged for dampener auditing)
    triggers_suppressed: list[str] = field(default_factory=list)

    # Echo of the resolved preset name (for logging)
    preset_name: str = ""

    # Carrying residency counts, split by class (Tool Belt 1.0):
    #   · always_carry_count — immutable residents (class A source)
    #   · carry_count        — adaptive residents  (class C source)
    always_carry_count: int = 0
    carry_count: int = 0

    @property
    def is_wildcard(self) -> bool:
        return self.active_tool_names == WILDCARD_ALWAYS_ON


def predict(
    message: str,
    attachments: Iterable[str] | None,
    preset: Preset,
) -> Prediction:
    """Run the classifier. Always returns a Prediction.

    ``attachments`` is an iterable of attachment kind strings (e.g. ``"image"``,
    ``"audio"``). Empty / None means no attachments. Used by triggers that key
    on attachment presence (e.g. vision_analyze).
    """
    try:
        return _predict_inner(message, attachments, preset)
    except Exception as exc:
        logger.warning("tool-belt: predictor failed (%s) — falling back to wildcard", exc)
        return Prediction(
            active_tool_names=WILDCARD_ALWAYS_ON,
            triggers_fired=[],
            preset_name=preset.name,
            always_carry_count=0,
            carry_count=0,
        )


def _predict_inner(
    message: str,
    attachments: Iterable[str] | None,
    preset: Preset,
) -> Prediction:
    if preset.is_wildcard:
        return Prediction(
            active_tool_names=WILDCARD_ALWAYS_ON,
            triggers_fired=[],
            preset_name=preset.name,
            always_carry_count=len(preset.always_carry),
            carry_count=len(preset.carry),
        )

    msg = message or ""
    # Normalize attachments to a set of strings — be defensive against
    # callers that pass raw dicts or other shapes.
    atts: set[str] = set()
    for a in (attachments or []):
        if isinstance(a, str):
            atts.add(a)
        elif isinstance(a, dict):
            kind = a.get("type") or a.get("kind") or ""
            if isinstance(kind, str) and kind:
                atts.add(kind.lower())

    # Start from the residents (always_carry ∪ carry). The final A/C/X split
    # against the live enabled ceiling happens later in carrying.resolve; here
    # we build only the per-turn candidate active set.
    residents = preset.always_on
    active: list[str] = list(residents) if isinstance(residents, list) else []
    triggers_fired: list[str] = []
    triggers_suppressed: list[str] = []

    # Walk triggers; each that fires contributes its tools to the active set.
    # Track suppressions separately so dampener auditing can distinguish
    # "excluded by dampener" from "simply no positive match".
    for group in preset.triggers:
        if group.matches(msg, atts):
            triggers_fired.append(group.name)
            for tool in group.tools:
                if tool not in active:
                    active.append(tool)
        elif (
            group.exclude_patterns
            and group.is_excluded(msg)
            and group.would_fire_positive(msg, atts)
        ):
            triggers_suppressed.append(group.name)

    return Prediction(
        active_tool_names=active,
        triggers_fired=triggers_fired,
        triggers_suppressed=triggers_suppressed,
        preset_name=preset.name,
        always_carry_count=len(preset.always_carry),
        carry_count=len(preset.carry),
    )
