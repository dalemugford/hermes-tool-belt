"""Shell-bypass detection table — CLI shadows of latent native tools.

When the model runs a shell command via the ``terminal`` tool whose content
maps to a native Tool Belt tool that was *latent* this turn (in the ceiling
but not made active), it has bypassed the tool the promotion machinery is
supposed to learn from. The expand-only harvest path only sees tools the
model *recovered* via ``expand_tools``; a tool bypassed via shell — because
the model never thought to expand it — is invisible to that path. This module
supplies the detector that makes such bypasses observable.

The table is **data, not code**: it lives in ``policy.yaml`` under the
top-level ``bypass_patterns`` key, one entry per latent tool with a CLI
shadow, so operators can extend it without touching plugin logic. Each entry
is ``{id, tool, pattern}`` where ``pattern`` is a regex matched against the
shell command string.

Privacy (hard contract, see ``docs/PRIVACY.md``): this module only ever
returns the matched ``pattern_id`` and ``tool`` name. The command text it
inspects is never returned, stored, or previewed by the detector or its
callers. The bypass-event row reuses the prediction's ``message_preview``
(an already-accepted disclosure), never the command.

Fail-open by construction: a missing file, a malformed section, or a bad
regex yields an empty table, so detection silently produces nothing rather
than raising into the tool-call hot path.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Same shipped policy file the presets loader reads. Single source of truth.
_POLICY_FILE = Path(__file__).parent / "policy.yaml"


@dataclass(frozen=True)
class BypassPattern:
    """One compiled shell-bypass rule: ``pattern`` → native ``tool``."""
    id: str
    tool: str
    pattern: re.Pattern[str]


#: Parsed-table cache keyed by path, invalidated by (mtime_ns, size) — the
#: same discipline presets uses. The policy file changes rarely; the stat
#: check keeps live edits honest without re-compiling on every tool call.
_TABLE_CACHE: dict[str, tuple[int, int, list[BypassPattern]]] = {}


def _compile_table(raw: Any) -> list[BypassPattern]:
    """Compile the ``bypass_patterns`` YAML section into rules.

    Skips malformed entries and uncompilable regexes individually rather than
    discarding the whole table — one bad operator edit can't disable detection
    of every other tool.
    """
    if not isinstance(raw, list):
        return []
    out: list[BypassPattern] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        pattern_id = str(entry.get("id") or "").strip()
        tool = str(entry.get("tool") or "").strip()
        pattern_src = entry.get("pattern")
        if not (pattern_id and tool and isinstance(pattern_src, str) and pattern_src):
            continue
        try:
            compiled = re.compile(pattern_src, re.IGNORECASE)
        except re.error as exc:
            logger.warning(
                "tool-belt: skipping bad bypass pattern %r (%s)", pattern_id, exc
            )
            continue
        out.append(BypassPattern(id=pattern_id, tool=tool, pattern=compiled))
    return out


def load_bypass_patterns(path: Path | None = None) -> list[BypassPattern]:
    """Load and compile the bypass-pattern table from ``policy.yaml``.

    Returns an empty list on any failure (missing file, non-mapping YAML,
    absent section) — detection is best-effort and must never raise into the
    hot path. Cached per path by mtime/size.
    """
    policy_path = path or _POLICY_FILE
    key = str(policy_path)
    try:
        st = policy_path.stat()
        cached = _TABLE_CACHE.get(key)
        if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
            return cached[2]
    except OSError:
        return []
    try:
        import yaml  # type: ignore[import-untyped]

        with policy_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        logger.warning("tool-belt: failed to read bypass patterns (%s)", exc)
        return []
    if not isinstance(data, dict):
        return []
    table = _compile_table(data.get("bypass_patterns"))
    _TABLE_CACHE[key] = (st.st_mtime_ns, st.st_size, table)
    return table


def detect_bypass(command: str, path: Path | None = None) -> tuple[str, str] | None:
    """Return ``(pattern_id, tool)`` for the first rule matching ``command``.

    ``command`` is the raw shell command string. Returns ``None`` when no rule
    matches (the common case) or when ``command`` is empty/non-string. The
    command text is never retained.
    """
    if not command or not isinstance(command, str):
        return None
    for rule in load_bypass_patterns(path):
        if rule.pattern.search(command):
            return rule.id, rule.tool
    return None
