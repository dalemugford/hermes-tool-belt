#!/usr/bin/env python3
"""Tool Belt onboarding — one command from install to a working configuration.

Run this after ``hermes plugins install dalemugford/hermes-tool-belt``::

    python3 scripts/configure.py

The command is a conversation, not a form. Every invocation detects the
current state of each agent/platform scope and offers the right next step:

  fresh      — nothing configured yet; pick a path
  observing  — the scope is collecting telemetry in recommend mode
  ready      — enough sessions collected; shaping can be reviewed and applied
  shaped     — the learned overlay is live for this scope

Two paths are offered on a fresh scope:

  Shape now         Read the history you already have, show exactly what would
                    change per agent, and — only after you confirm — write the
                    learned overlay and switch that scope to ``learned_mode:
                    apply``.

  Recommend first   Leave the tool set untouched while telemetry accumulates
                    (observation mode). Re-run this command later; it reports
                    how many more sessions each scope needs and offers the
                    same review/confirm/apply step once the data is there.

Config is written **only** through ``hermes config set`` / ``hermes config
unset``. Hermes owns ``config.yaml``; this script never edits it directly.
Every write is preceded by a ``before → after`` line — the config keys, the
``learned.json`` overlay, and the configure-state sidecar alike — and nothing
is written without explicit confirmation (or ``--yes``).

Usage
=====

  python3 scripts/configure.py                     # interactive
  python3 scripts/configure.py --status            # read-only state report
  python3 scripts/configure.py --agent default --path recommend --yes
  python3 scripts/configure.py --agent default --path shape --dry-run
  python3 scripts/configure.py --reset default     # back to recommend mode
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

HERE = Path(__file__).resolve().parent
PLUGIN_DIR = HERE.parent

#: Root of the plugin's live config block inside ``~/.hermes/config.yaml``.
CONFIG_PREFIX = "plugins.tool-belt"

#: Sidecar remembering the per-scope bypass value that observation mode
#: replaced, so a reset can restore it. Deliberately *not* ``learned.json`` —
#: that file's shape belongs to the shaper.
CONFIGURE_STATE_FILE = "configure-state.json"

STATE_FRESH = "fresh"
STATE_OBSERVING = "observing"
STATE_READY = "ready"
STATE_SHAPED = "shaped"

#: Per-scope bypass value that puts a scope into full observation — every
#: session ships the untouched ceiling while telemetry still records what the
#: predictor *would* have done.
OBSERVATION_BYPASS = 1.0
#: The shipped default: narrow immediately, no observation cohort.
NARROW_BYPASS = 0.0


class Abort(Exception):
    """Raised when the user ends the conversation (EOF / Ctrl-C)."""


# ─────────────────────────── plugin module loading ───────────────────────────


def _load_script_module(module_name: str, filename: str):
    """Load a sibling script by path (filenames contain hyphens)."""
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, HERE / filename)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_plugin_package():
    """Register the hyphenated plugin directory as an importable package.

    Mirrors ``tests/conftest.py`` so the same module objects are reused when
    the test suite has already registered them.
    """
    name = "tool_belt_plugin"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError("cannot load tool-belt package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_shaper():
    """The between-session shaper — the single source of shaping math."""
    return _load_script_module("tool_belt_shape_ceiling", "shape-ceiling.py")


def load_learned():
    """``learned.py`` (mode normalization, learned-state helpers), or None."""
    try:
        _load_plugin_package()
        return importlib.import_module("tool_belt_plugin.learned")
    except Exception:
        return None


def load_base_preset():
    """The shipped policy preset, or None if it cannot be read."""
    try:
        _load_plugin_package()
        presets = importlib.import_module("tool_belt_plugin.presets")
        return presets.load_base_policy()
    except Exception:
        return None


def load_savings_engine():
    """The canonical savings engine — the single source of projection math."""
    try:
        _load_plugin_package()
        return importlib.import_module("tool_belt_plugin.savings")
    except Exception:
        return None


def require_yaml():
    """PyYAML, or a loud exit — the shared operator-script policy.

    Delegates to the root ``yaml_required.require_yaml`` so this script
    degrades exactly like ``shape-ceiling.py`` / ``analyze.py``: a missing
    parser means the wrong interpreter, and stopping beats reading a partial
    policy. Raises ``SystemExit(2)``; never returns ``None``.
    """
    _load_plugin_package()
    guard = importlib.import_module("tool_belt_plugin.yaml_required")
    return guard.require_yaml()


def load_savings_cli():
    """The ``tool-belt savings`` CLI module (launcher helper lives there)."""
    try:
        _load_plugin_package()
        return importlib.import_module("tool_belt_plugin.savings_cli")
    except Exception:
        return None


def normalize_mode(value: Any) -> str:
    """Resolve a config value to ``recommend`` / ``apply``.

    Delegates to ``learned.normalize_mode`` so legacy aliases stay in one
    place; falls back to a minimal equivalent when the package is unavailable.
    """
    learned = load_learned()
    if learned is not None:
        try:
            return learned.normalize_mode(value)
        except Exception:
            pass
    mode = str(value or "").strip().lower()
    mode = {"off": "recommend", "auto": "apply", "audit": "apply"}.get(mode, mode)
    # Default flipped to "apply" (Promise #2): matches learned.DEFAULT_MODE.
    return mode if mode in {"recommend", "apply"} else "apply"


# ──────────────────────────────── discovery ──────────────────────────────────


@dataclass
class ScopeInfo:
    """One agent/platform scope, with the telemetry we have for it."""

    scope: str
    agent: str
    platform: str
    state_dir: Path
    sessions: int = 0
    inferred: bool = False  # no telemetry yet; platform came from the user

    @property
    def config_prefix(self) -> str:
        return f"{CONFIG_PREFIX}.channels.{self.scope}"


def default_hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes"))


def configured_agent_name(profile_home: Path) -> str:
    """The profile's ``plugins.tool-belt.agent`` config value, or ''.

    This is the agent name the tool's own output (status rows, savings
    report) shows for the profile — e.g. ``bernard`` on a root profile whose
    directory identity is ``default``. ``--agent`` must accept it too:
    rejecting the exact string the tool itself prints is a first-run trap.
    """
    try:
        yaml = require_yaml()
        raw = yaml.safe_load((profile_home / "config.yaml").read_text(
            encoding="utf-8")) or {}
        name = ((raw.get("plugins") or {}).get("tool-belt") or {}).get("agent")
        if isinstance(name, str) and name.strip():
            return name.strip()
    except SystemExit:
        raise
    except Exception:
        pass
    return ""


def _filter_hits(profile_filter: str | None, dir_name: str,
                 profile_home: Path) -> bool:
    """``--agent``/``--reset`` match by profile directory name OR by the
    configured agent name the UI displays for that profile."""
    if not profile_filter:
        return True
    if dir_name == profile_filter:
        return True
    return configured_agent_name(profile_home) == profile_filter


def discover_state_dirs(
    hermes_home: Path, profile_filter: str | None = None
) -> list[tuple[str, Path]]:
    """Return ``[(agent_label, state_dir), ...]`` for the root and named profiles.

    Same shape and ordering as ``bootstrap._discover_state_dirs``, with one
    difference: a profile that has ``sessions/`` but no ``state/tool-belt``
    yet is still surfaced, because onboarding runs *before* the plugin has
    written any state.
    """
    out: list[tuple[str, Path]] = []
    root_state = hermes_home / "state" / "tool-belt"
    if _filter_hits(profile_filter, "default", hermes_home):
        if root_state.is_dir() or (hermes_home / "sessions").is_dir():
            out.append(("default", root_state))
    profiles_dir = hermes_home / "profiles"
    if profiles_dir.is_dir():
        for child in sorted(profiles_dir.iterdir()):
            if not child.is_dir() or child.name == "default":
                continue  # "default" under profiles/ is reserved by Hermes
            if not _filter_hits(profile_filter, child.name, child):
                continue
            p_state = child / "state" / "tool-belt"
            if p_state.is_dir() or (child / "sessions").is_dir():
                out.append((child.name, p_state))
    return out


def _session_counts_by_scope(state_dir: Path) -> dict[str, int]:
    """Distinct sessions per scope from ``predictions.jsonl``.

    Grouping is delegated to the shaper so "a session" means exactly what it
    means everywhere else in the plugin.
    """
    shaper = load_shaper()
    preds = shaper.load_jsonl(state_dir / "predictions.jsonl")
    if not preds:
        return {}
    grouped = shaper.group_predictions_by_scope_session(preds)
    return {scope: len(sessions) for scope, sessions in grouped.items()}


def discover_scopes(
    hermes_home: Path,
    profile_filter: str | None = None,
    platform_hint: Sequence[str] | None = None,
) -> list[ScopeInfo]:
    """Discover every ``agent:platform`` scope this install knows about.

    Platforms come from the scopes actually recorded in each profile's
    ``predictions.jsonl``. A profile with no telemetry yet yields one inferred
    scope per ``platform_hint`` entry (the caller asks the user), or none.
    """
    out: list[ScopeInfo] = []
    for label, state_dir in discover_state_dirs(hermes_home, profile_filter):
        counts = _session_counts_by_scope(state_dir) if state_dir.is_dir() else {}
        seen_any = False
        for scope, sessions in sorted(counts.items()):
            agent, _, platform = scope.rpartition(":")
            if not agent:  # bare-platform scope row (older telemetry)
                agent, platform = label, scope
                scope = f"{label}:{platform}"
            out.append(
                ScopeInfo(
                    scope=scope,
                    agent=agent,
                    platform=platform,
                    state_dir=state_dir,
                    sessions=sessions,
                )
            )
            seen_any = True
        if not seen_any:
            for platform in platform_hint or ():
                platform = str(platform).strip().lower()
                if not platform:
                    continue
                out.append(
                    ScopeInfo(
                        scope=f"{label}:{platform}",
                        agent=label,
                        platform=platform,
                        state_dir=state_dir,
                        sessions=0,
                        inferred=True,
                    )
                )
    return out


# ────────────────────────────── hermes config I/O ────────────────────────────

Runner = Callable[..., Any]

_NOT_SET_PREFIX = "Config key not set"


# These indirections exist so the module attribute is resolved at call time.
# Binding ``subprocess.run`` / ``print`` / ``input`` directly as a default
# argument would capture the original object, and a test that patches the
# module attribute would still reach the real one — which means real writes
# to a real config.
def _default_runner(argv, capture_output=False, text=False, check=False):
    return subprocess.run(argv, capture_output=capture_output, text=text, check=check)


def _default_which(name: str) -> str | None:
    return shutil.which(name)


def _default_out(message: str) -> None:
    print(message)


def _default_reader(message: str) -> str:
    return input(message)


def hermes_available(which: Callable[[str], str | None] = _default_which) -> bool:
    return bool(which("hermes"))


def _run(runner: Runner, argv: list[str]):
    return runner(argv, capture_output=True, text=True, check=False)


def hermes_config_get(key: str, runner: Runner = _default_runner) -> str | None:
    """Read a resolved config value. ``None`` means "not set"."""
    try:
        result = _run(runner, ["hermes", "config", "get", key])
    except Exception:
        return None
    out = (getattr(result, "stdout", "") or "").strip()
    if getattr(result, "returncode", 0) != 0:
        return None
    if not out or out.startswith(_NOT_SET_PREFIX):
        return None
    return out


def read_plugin_config(runner: Runner = _default_runner) -> dict[str, Any]:
    """Read the whole ``plugins.tool-belt`` block as a dict.

    Returns ``{}`` when the block is unset or unparseable. A *missing PyYAML*
    is not a degraded mode: it means the wrong interpreter, and
    :func:`require_yaml` exits loudly (status 2) rather than silently reporting
    an empty config block — which would read as "nothing is configured" and
    offer to configure a scope that already is.
    """
    raw = hermes_config_get(CONFIG_PREFIX, runner=runner)
    if not raw:
        return {}
    yaml = require_yaml()
    try:
        data = yaml.safe_load(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except Exception:
        return None


def scope_settings(
    scope: str,
    plugin_config: dict[str, Any] | None,
    runner: Runner = _default_runner,
) -> dict[str, Any]:
    """Current per-scope ``learned_mode`` / ``bypass_rate`` (raw, may be None).

    Prefers the parsed config block; falls back to targeted ``hermes config
    get`` reads when the block is unset or could not be parsed (a *missing*
    PyYAML never reaches here — ``read_plugin_config`` exits first).
    """
    if plugin_config:
        channels = plugin_config.get("channels")
        entry = channels.get(scope) if isinstance(channels, dict) else None
        entry = entry if isinstance(entry, dict) else {}
        return {
            "learned_mode": entry.get("learned_mode", plugin_config.get("learned_mode")),
            "bypass_rate": _to_float(entry.get("bypass_rate", plugin_config.get("bypass_rate"))),
            "scope_learned_mode": entry.get("learned_mode"),
            "scope_bypass_rate": _to_float(entry.get("bypass_rate")),
            "configured": bool(entry) or bool(plugin_config),
        }
    prefix = f"{CONFIG_PREFIX}.channels.{scope}"
    scope_mode = hermes_config_get(f"{prefix}.learned_mode", runner=runner)
    scope_bypass = _to_float(hermes_config_get(f"{prefix}.bypass_rate", runner=runner))
    global_mode = hermes_config_get(f"{CONFIG_PREFIX}.learned_mode", runner=runner)
    global_bypass = _to_float(hermes_config_get(f"{CONFIG_PREFIX}.bypass_rate", runner=runner))
    return {
        "learned_mode": scope_mode if scope_mode is not None else global_mode,
        "bypass_rate": scope_bypass if scope_bypass is not None else global_bypass,
        "scope_learned_mode": scope_mode,
        "scope_bypass_rate": scope_bypass,
        "configured": any(
            v is not None for v in (scope_mode, scope_bypass, global_mode, global_bypass)
        ),
    }


# ──────────────────────────────── state machine ──────────────────────────────


def shape_thresholds() -> dict[str, int]:
    """Shaper minima, straight from ``policy.yaml`` ``learning.shape_ceiling``."""
    return load_shaper().load_shape_ceiling_defaults()


def required_sessions(thresholds: dict[str, int] | None = None) -> int:
    """Sessions needed before a scope has a complete shaping picture.

    Derived, never hardcoded: promotion needs ``promote_min_sessions`` and
    demotion needs ``demote_min_sessions_no_use``; the larger of the two is
    the point at which both halves of the recommendation are available.
    """
    thresholds = thresholds or shape_thresholds()
    return max(
        int(thresholds.get("promote_min_sessions", 2)),
        int(thresholds.get("demote_min_sessions_no_use", 20)),
    )


def _has_learned_assignment(info: ScopeInfo) -> bool:
    """True when the scope has an adaptive carrying assignment in learned.json.

    Read-only and fail-open (a read problem means "no assignment"). Checks
    the v2 keys and their v1 spellings.
    """
    try:
        doc = json.loads((info.state_dir / "learned.json").read_text(encoding="utf-8"))
        entry = (doc.get("scopes") or {}).get(info.scope) or {}
        return bool(
            entry.get("carry") or entry.get("expand_only")
            or entry.get("always_on") or entry.get("always_off")
        )
    except Exception:
        return False


def classify_scope(
    info: ScopeInfo,
    settings: dict[str, Any],
    thresholds: dict[str, int] | None = None,
) -> str:
    """Which of the four states this scope is in.

    ``learned_mode`` now DEFAULTS to ``apply`` (full-start contract), so an
    unset mode no longer implies "shaped": fresh means everything is active
    and telemetry is accumulating. A scope counts as shaped when the operator
    set ``apply`` explicitly, or when a learned carrying assignment exists on
    disk (evidence-driven shaping has landed, hand-run or auto).
    """
    needed = required_sessions(thresholds)
    explicit_mode = settings.get("learned_mode")
    if explicit_mode is not None and normalize_mode(explicit_mode) == "apply":
        return STATE_SHAPED
    if _has_learned_assignment(info):
        return STATE_SHAPED
    bypass = settings.get("scope_bypass_rate")
    if bypass is None:
        bypass = settings.get("bypass_rate")
    observing = bypass is not None and float(bypass) >= OBSERVATION_BYPASS
    if observing:
        return STATE_READY if info.sessions >= needed else STATE_OBSERVING
    if not settings.get("configured") and info.sessions < needed:
        return STATE_FRESH
    return STATE_READY if info.sessions >= needed else STATE_FRESH


def remaining_sessions(info: ScopeInfo, thresholds: dict[str, int] | None = None) -> int:
    return max(0, required_sessions(thresholds) - info.sessions)


# ────────────────────────── configure-state sidecar ──────────────────────────


def _configure_state_path(state_dir: Path) -> Path:
    return state_dir / CONFIGURE_STATE_FILE


def read_configure_state(state_dir: Path) -> dict[str, Any]:
    try:
        data = json.loads(_configure_state_path(state_dir).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _atomic_write_json(target: Path, payload: dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)


def remember_previous_full_ceiling_rate(
    state_dir: Path, scope: str, value: float | None
) -> None:
    """Record the full-ceiling rate observation mode is about to replace."""
    state = read_configure_state(state_dir)
    scopes = dict(state.get("scopes") or {})
    entry = dict(scopes.get(scope) or {})
    entry["previous_full_ceiling_rate"] = NARROW_BYPASS if value is None else float(value)
    scopes[scope] = entry
    state["scopes"] = scopes
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_write_json(_configure_state_path(state_dir), state)


def previous_full_ceiling_rate(state_dir: Path, scope: str) -> float:
    """The full-ceiling rate to restore on reset. Defaults to shipped ``0.0``.

    Reads the pre-rename ``previous_bypass_rate`` key once for sidecars written
    by an older onboarding; new writes use the canonical name only.
    """
    entry = (read_configure_state(state_dir).get("scopes") or {}).get(scope) or {}
    value = _to_float(entry.get("previous_full_ceiling_rate"))
    if value is None:
        value = _to_float(entry.get("previous_bypass_rate"))
    return NARROW_BYPASS if value is None else value


# ──────────────────────────────── write planning ─────────────────────────────


@dataclass
class ConfigWrite:
    """One pending ``hermes config`` mutation, with its before/after."""

    key: str
    after: str | None
    before: str | None = None
    action: str = "set"  # "set" | "unset"

    def argv(self) -> list[str]:
        if self.action == "unset":
            return ["hermes", "config", "unset", self.key]
        return ["hermes", "config", "set", self.key, str(self.after), "--force"]


def format_value(value: Any) -> str:
    return "(not set)" if value is None else str(value)


#: Friendly labels for the config keys this tool writes, so the disclosure
#: reads as plain changes instead of dotted config paths.
_FRIENDLY_KEYS = {"learned_mode": "Shaping mode", "bypass_rate": "Observation rate"}


def build_diff(write: ConfigWrite) -> str:
    """One human-readable ``before → after`` line for a pending write."""
    after = "(removed)" if write.action == "unset" else format_value(write.after)
    leaf = write.key.rsplit(".", 1)[-1]
    label = _FRIENDLY_KEYS.get(leaf, write.key)
    return f"    {label}: {format_value(write.before)} → {after}"


def build_overlay_diff(
    info: ScopeInfo, before: dict[str, list[str]], after: dict[str, list[str]]
) -> list[str]:
    """``before → after`` lines for the ``learned.json`` overlay write.

    The config diff alone under-discloses: the substantive change a confirmed
    shaping makes is to the learned overlay, not to the two config scalars.
    Every list that changes is shown with its per-tool moves, so nothing is
    written that did not appear in the diff.
    """
    labels = {"carry": "Tools carried",
              "expand_only": "Tools available by expansion"}
    lines: list[str] = []
    for key in ("carry", "expand_only"):
        old = sorted({str(t) for t in (before.get(key) or [])})
        new = sorted({str(t) for t in (after.get(key) or [])})
        if old == new:
            lines.append(f"    :: {labels[key]}: {len(old)} → unchanged")
            continue
        old_names = f" ({', '.join(old)})" if 0 < len(old) <= 3 else ""
        lines.append(f"    :: {labels[key]}: {len(old)}{old_names} → {len(new)}")
        if len(new) <= 10:
            # Small result: list the full new membership.
            for tool in new:
                lines.append(f"       - {tool}")
        else:
            # Large result: list just the moves, capped, so the screen stays
            # readable even on a first shape that moves everything at once.
            added = [t for t in new if t not in set(old)]
            removed = [t for t in old if t not in set(new)]
            shown = 0
            for tool in added:
                if shown >= 10:
                    break
                lines.append(f"       + {tool}")
                shown += 1
            for tool in removed:
                if shown >= 10:
                    break
                lines.append(f"       - {tool}")
                shown += 1
            hidden = len(added) + len(removed) - shown
            if hidden > 0:
                lines.append(f"       … and {hidden} more")
    return lines


def _fmt_rate(value: float) -> str:
    """Render a bypass rate without rounding it away.

    ``0.05`` is a real A/B cohort value a user may already have set; it must
    survive a round-trip through observation mode and back.
    """
    text = f"{float(value):g}"
    return f"{text}.0" if "." not in text and "e" not in text else text


def plan_shape_writes(info: ScopeInfo, settings: dict[str, Any]) -> list[ConfigWrite]:
    """Turn shaping on for a scope: apply the overlay, stop observing."""
    writes = [
        ConfigWrite(
            key=f"{info.config_prefix}.learned_mode",
            after="apply",
            before=settings.get("scope_learned_mode") or settings.get("learned_mode"),
        )
    ]
    current = settings.get("scope_bypass_rate")
    if current is not None and float(current) != NARROW_BYPASS:
        writes.append(
            ConfigWrite(
                key=f"{info.config_prefix}.bypass_rate",
                after=_fmt_rate(NARROW_BYPASS),
                before=_fmt_rate(current),
            )
        )
    return writes


def plan_recommend_writes(info: ScopeInfo, settings: dict[str, Any]) -> list[ConfigWrite]:
    """Put a scope into observation mode: watch, don't narrow."""
    return [
        ConfigWrite(
            key=f"{info.config_prefix}.learned_mode",
            after="recommend",
            before=settings.get("scope_learned_mode") or settings.get("learned_mode"),
        ),
        ConfigWrite(
            key=f"{info.config_prefix}.bypass_rate",
            after=_fmt_rate(OBSERVATION_BYPASS),
            before=(
                None
                if settings.get("scope_bypass_rate") is None
                else _fmt_rate(settings["scope_bypass_rate"])
            ),
        ),
    ]


def plan_reset_writes(
    info: ScopeInfo, settings: dict[str, Any], restore_bypass: float
) -> list[ConfigWrite]:
    """Return a shaped scope to recommend mode with its old bypass value."""
    return [
        ConfigWrite(
            key=f"{info.config_prefix}.learned_mode",
            after="recommend",
            before=settings.get("scope_learned_mode") or settings.get("learned_mode"),
        ),
        ConfigWrite(
            key=f"{info.config_prefix}.bypass_rate",
            after=_fmt_rate(restore_bypass),
            before=(
                None
                if settings.get("scope_bypass_rate") is None
                else _fmt_rate(settings["scope_bypass_rate"])
            ),
        ),
    ]


def apply_writes(
    writes: Sequence[ConfigWrite],
    runner: Runner = _default_runner,
    dry_run: bool = False,
    out: Callable[[str], None] = _default_out,
) -> list[str]:
    """Execute pending writes. ``dry_run`` runs no subprocess at all.

    Under ``dry_run`` the returned commands are the ones that *would* run; the
    epilogue titles them "Would apply:", so they carry no per-line marker.
    """
    applied: list[str] = []
    for write in writes:
        if dry_run:
            applied.append(" ".join(write.argv()))
            continue
        try:
            result = _run(runner, write.argv())
        except Exception as exc:
            out(f"  ! {write.key}: {exc}")
            continue
        if getattr(result, "returncode", 0) != 0:
            stderr = (getattr(result, "stderr", "") or "").strip()
            out(f"  ! {write.key}: hermes config failed — {stderr[:200]}")
            continue
        applied.append(" ".join(write.argv()))
    return applied


# ───────────────────────────── learned overlay I/O ───────────────────────────


def compute_recommendations(info: ScopeInfo, thresholds: dict[str, int]) -> dict[str, Any] | None:
    """Run the shaper's own analysis for one scope. No math is reimplemented."""
    shaper = load_shaper()
    preds = shaper.load_jsonl(info.state_dir / "predictions.jsonl")
    if not preds:
        return None
    grouped = shaper.group_predictions_by_scope_session(preds)
    sessions = grouped.get(info.scope)
    if not sessions:
        return None
    calls_by_pred = shaper.index_tool_calls_by_prediction(
        shaper.load_jsonl(info.state_dir / "tool_calls.jsonl")
    )
    return shaper.compute_scope_recommendations(
        scope=info.scope,
        sessions=sessions,
        calls_by_pred=calls_by_pred,
        window=int(thresholds.get("session_window", 100)),
        promote_min_sessions=int(thresholds.get("promote_min_sessions", 2)),
        promote_min_calls=int(thresholds.get("promote_min_calls", 3)),
        demote_min_sessions_no_use=int(thresholds.get("demote_min_sessions_no_use", 20)),
        demote_k=float(thresholds.get("demote_k", 1.5)),
        schema_sizes=shaper.load_schema_sizes(info.state_dir),
        cache_mode=shaper.read_cache_mode(info.state_dir, info.scope),
        api_call_counts=shaper.index_api_call_counts(
            shaper.load_jsonl(info.state_dir / "api_calls.jsonl")),
    )


def write_learned_overlay(
    info: ScopeInfo, recs: dict[str, Any], dry_run: bool = False
) -> bool:
    """Persist the shaper's recommendations via the shaper's merge, which
    itself persists through ``learned.write_state`` (the single writer)."""
    shaper = load_shaper()
    _state, changed = shaper.merge_into_learned(
        info.state_dir, {info.scope: recs}, dry_run
    )
    return changed


def _assignment_of(state: dict[str, Any], scope: str) -> dict[str, list[str]]:
    """Read one scope's carrying assignment out of a learned-state document.

    The document is put through ``learned.normalize_state`` — the central
    v1→v2 adapter — so no v1 key spelling is read (or known) here.
    """
    learned = load_learned()
    if learned is not None:
        try:
            state = learned.normalize_state(state)
        except Exception:
            pass
    entry = (state.get("scopes") or {}).get(scope) or {}

    def _lst(key: str) -> list[str]:
        value = entry.get(key)
        if not isinstance(value, list):
            return []
        return [str(t) for t in value if str(t).strip()]

    return {"carry": _lst("carry"), "expand_only": _lst("expand_only")}


def current_assignment(info: ScopeInfo) -> dict[str, list[str]]:
    """The scope's carrying assignment as ``learned.json`` holds it today.

    The "before" half of the overlay disclosure. A missing/unreadable file is
    an empty assignment — the same thing the shaper's merge starts from.
    """
    path = info.state_dir / "learned.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"carry": [], "expand_only": []}
    if not isinstance(state, dict):
        return {"carry": [], "expand_only": []}
    return _assignment_of(state, info.scope)


def proposed_assignment(info: ScopeInfo, recs: dict[str, Any]) -> dict[str, list[str]]:
    """The carrying assignment shaping would write, without writing it.

    Delegates to the shaper's own ``merge_into_learned`` in dry-run mode and
    reads the proposed scope entry back from the merged state — the same moves
    (promote/demote across the adaptive carry ⇄ expand_only boundary) the real
    apply would make, applied to the assignment the scope *already* has. No
    math is reimplemented here, so a re-shape previews what it will write.
    """
    shaper = load_shaper()
    merged, _changed = shaper.merge_into_learned(
        info.state_dir, {info.scope: recs}, dry_run=True
    )
    return _assignment_of(merged, info.scope)


def remove_learned_scope(info: ScopeInfo, dry_run: bool = False) -> bool:
    """Clear one scope's adaptive assignments from ``learned.json``.

    Routes through ``learned.reset_scope`` — the single reset semantic: only
    the scope's adaptive ``carry`` / ``expand_only`` assignments and ``shaping``
    evidence are cleared; unrelated per-scope metadata, every other scope, and
    top-level metadata are preserved (always_carry policy and trigger
    definitions live in policy.yaml, never here). Persistence goes through
    ``learned.write_state`` (atomic, normalize-on-write, v2-stamped).
    """
    path = info.state_dir / "learned.json"
    if not path.exists():
        return False
    learned = load_learned()
    if learned is None:
        return False
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(state, dict):
        return False
    new_state, changed = learned.reset_scope(state, info.scope)
    if not changed:
        return False
    if dry_run:
        return True
    learned.write_state(new_state, path)
    return True


# ────────────────────────────────── rendering ────────────────────────────────


def render_projection(info: "ScopeInfo", projection) -> list[str]:
    """Render the projected-savings preview for one scope's proposal.

    ``projection`` is a canonical ``ProjectedCohort`` from the savings engine —
    its math is never duplicated here. Presentation follows the engine's
    confidence discipline: unsupported figures are suppressed, never guessed.
    """
    lines = [
        f"  Projected savings for {info.scope} — counterfactual, "
        f"{projection.sessions_analyzed} session(s) analyzed, "
        f"confidence: {projection.confidence}"
    ]
    lines.append(
        f"    gross schema reduction: {projection.gross_schema_token_reduction:,} tok"
    )
    if projection.expansion_events:
        lines.append(
            f"    est. expansion overhead: −{projection.estimated_expansion_overhead:,} tok "
            f"({projection.expansion_events} event(s))"
        )
    lines.append(f"    net projected reduction: {projection.net_token_reduction:,} tok")
    if projection.net_input_reduction_pct is not None:
        lines.append(
            f"    net input reduction: {projection.net_input_reduction_pct:.1f}% "
            f"(denominator: {projection.denominator_source})"
        )
    elif projection.schema_reduction_pct is not None:
        lines.append(
            f"    schema-only reduction: {projection.schema_reduction_pct:.1f}% "
            "(not the session-input %)"
        )
    if projection.estimated_usd_savings is not None:
        rate = next(
            (m.rate_basis for m in projection.models if m.cost_class == "known"),
            "n/a",
        )
        lines.append(
            f"    estimated USD savings: ${projection.estimated_usd_savings:.4f} "
            f"({projection.usd_coverage} coverage, rate {rate})"
        )
    else:
        lines.append("    estimated USD savings: n/a (no known variable-cost route)")
    return lines


def _project_scope(info: "ScopeInfo", proposal: dict[str, list[str]]):
    """Run the canonical savings engine over this scope with the proposal.

    Returns ``None`` when the engine or the scope's session history is
    unavailable — the preview is then simply omitted, never approximated.
    """
    engine = load_savings_engine()
    if engine is None:
        return None
    try:
        report = engine.compute(
            hermes_home=default_hermes_home(),
            cache_mode="on",
            proposed_by_scope={info.scope: {
                "carry": list(proposal.get("carry") or []),
                "expand_only": list(proposal.get("expand_only") or []),
            }},
        )
    except Exception:
        return None
    for a in report.agents:
        if a.projected.sessions_analyzed and info.scope in (
            getattr(a.projected, "scopes", None) or ()
        ):
            return a
    # Fallback: the proposal's scope governs — take the agent whose projection
    # consumed sessions under the proposed key (the engine keys by session
    # scope, not profile name, so the proposal reaches the right cohort).
    for a in report.agents:
        if a.projected.sessions_analyzed:
            return a
    return None


def _always_carried_line(plugin_config: dict[str, Any] | None) -> str:
    """One-line always-carried summary, showing config pins distinctly.

    e.g. ``Always carried: 6 policy + 2 pinned (web_search, terminal)``.
    Config pins come from ``plugins.tool-belt.always_carry`` (global; scope
    entries add more per scope). Never raises.
    """
    policy_count = 0
    try:
        preset = load_base_preset()
        policy_count = len(list(getattr(preset, "always_carry", []) or []))
    except Exception:
        pass
    pins: list[str] = []
    try:
        learned = load_learned()
        if learned is not None:
            pins = list(learned.config_always_carry(plugin_config or {}, ""))
    except Exception:
        pins = []
    if pins:
        return (f"Always carried: {policy_count} policy + {len(pins)} pinned "
                f"({', '.join(sorted(pins))})")
    return f"Always carried: {policy_count} policy (no config pins)"


def _preset_baselines(preset: Any) -> tuple[list[str], list[str]]:
    """``(always_carry, carry)`` from the shipped policy preset, defensively."""
    always_carry: list[str] = []
    carry: list[str] = []
    if preset is not None:
        raw = getattr(preset, "always_carry", None)
        if isinstance(raw, list):
            always_carry = [str(t) for t in raw]
        raw = getattr(preset, "carry", None)
        if isinstance(raw, list):
            carry = [str(t) for t in raw]
    return always_carry, carry


def _trigger_lines(preset: Any) -> list[str]:
    """The "trigger groups unchanged" line, or nothing when there are none."""
    triggers = list(getattr(preset, "triggers", []) or []) if preset is not None else []
    rows = [
        f"{getattr(group, 'name', '?')} ({len(list(getattr(group, 'tools', []) or []))} tools)"
        for group in triggers
    ]
    if not rows:
        return []
    return [
        f"  Trigger groups unchanged by these transitions ({len(rows)}): "
        + ", ".join(rows)
    ]


def _effective_carried(
    always_carry: Sequence[str], policy_carry: Sequence[str], proposal: dict[str, list[str]]
) -> list[str]:
    """Adaptive residents once ``proposal`` is the live overlay.

    Mirrors ``learned.apply_to_preset``'s precedence — ``(policy carry −
    learned expand_only) ∪ learned carry`` minus the immutable
    ``always_carry`` — over the overlay the shaper's merge would actually
    write. The promote/demote move algebra itself is never recomputed here;
    it arrives already resolved in ``proposal``.
    """
    expand_set = {str(t) for t in (proposal.get("expand_only") or [])}
    always_set = {str(t) for t in always_carry}
    carried = [t for t in policy_carry if t not in expand_set]
    for tool in proposal.get("carry") or []:
        if tool not in carried:
            carried.append(str(tool))
    return [t for t in carried if t not in always_set]


def render_shaping_summary(
    info: "ScopeInfo",
    recs: dict[str, Any] | None,
    preset: Any = None,
    thresholds: dict[str, int] | None = None,
    proposal: dict[str, list[str]] | None = None,
) -> list[str]:
    """Plain-language description of what shaping would do to one scope.

    Presents the Tool Belt 1.0 carrying model in its own vocabulary:

      * Always carried — the immutable policy baseline; never shaped.
      * Carried        — the effective adaptive residents after this shaping.
      * Proposed promotions into carry and demotions into expand-only — the
        demoted tools stay recoverable via triggers or ``expand_tools``, and
        are never disabled.
      * Trigger groups, which those transitions never touch.

    ``proposal`` is the assignment the apply would write —
    ``proposed_assignment()``, i.e. the shaper's own dry-run merge. It is
    computed here when the caller does not supply it. Because it starts from
    the scope's *existing* overlay, a re-shape previews the same result the
    apply will write; nothing about the move algebra is recomputed locally.

    Never raises on thin or empty input — an unshapeable scope simply says so.
    """
    thresholds = thresholds or {}
    recs = recs or {}
    promote = [p for p in (recs.get("promote") or []) if isinstance(p, dict)]
    demote = [d for d in (recs.get("demote") or []) if isinstance(d, dict)]
    considered = int(recs.get("sessions_considered") or 0)

    lines = [f"{info.scope} — from {considered} recorded session(s)"]

    base_always_carry, base_carry = _preset_baselines(preset)

    demoted_names = sorted(str(d.get("tool") or "") for d in demote)
    demoted_names = [n for n in demoted_names if n]
    promoted_names = sorted(str(p.get("tool") or "") for p in promote)
    promoted_names = [name for name in promoted_names if name]

    if proposal is None:
        try:
            proposal = proposed_assignment(info, recs)
        except Exception:
            proposal = {"carry": [], "expand_only": []}
    carried = _effective_carried(base_always_carry, base_carry, proposal)

    lines.append(f"  Always carried — permanent baseline ({len(base_always_carry)}): "
                 + (", ".join(sorted(base_always_carry)) if base_always_carry else "none"))
    if carried:
        lines.append(f"  Carried — adaptive residents after shaping ({len(carried)}): "
                     + ", ".join(sorted(carried)))
    else:
        lines.append("  Carried — adaptive residents: unchanged from the shipped policy")
    if promoted_names:
        detail = ", ".join(
            f"{p['tool']} (asked for in {p.get('sessions', 0)} session(s))"
            for p in promote
            if p.get("tool")
        )
        lines.append(f"  Proposed promotions into carry: {detail}")
    if demoted_names:
        uneconomic = sorted(
            str(d.get("tool") or "") for d in demote
            if d.get("evidence") == "carry_uneconomic"
        )
        lines.append(
            f"  Proposed demotions into expand-only ({len(demoted_names)}): "
            + ", ".join(demoted_names)
        )
        if uneconomic:
            lines.append(
                "    Of these, used but uneconomic to carry (cheaper to fetch "
                "on demand): " + ", ".join(n for n in uneconomic if n)
            )
        lines.append(
            "    Still fully available — the agent recovers any of these mid-session "
            "via triggers or expand_tools."
        )
    else:
        min_demote = int(thresholds.get("demote_min_sessions_no_use", 0) or 0)
        if min_demote and considered < min_demote:
            lines.append(
                f"  Proposed demotions: none yet — needs {min_demote} sessions, "
                f"has {considered}."
            )
        else:
            lines.append(
                "  Proposed demotions: none — every adaptive carry resident "
                "earns its slot (used enough that carrying costs less than "
                "fetching on demand)."
            )

    lines.extend(_trigger_lines(preset))
    return lines


def _shaped_detail(info: ScopeInfo) -> str:
    """Status detail for a shaped scope, naming an auto-apply when recorded.

    The in-process auto-shape engine stamps ``source: "auto"`` and
    ``applied_at`` into the scope's learned ``shaping`` block on each
    automatic apply; surface that as ``shaping applied (auto, <date>)`` so an
    operator can tell an automatic apply from a hand-run one. Read-only and
    fail-open: any read problem falls back to the plain detail.
    """
    try:
        doc = json.loads((info.state_dir / "learned.json").read_text(encoding="utf-8"))
        entry = (doc.get("scopes") or {}).get(info.scope) or {}
        shaping = entry.get("shaping") or entry.get("cache_aware") or {}
        if isinstance(shaping, dict) and shaping.get("source") == "auto":
            applied = str(shaping.get("applied_at") or "")[:10]
            if applied:
                return f"shaping applied (auto, {applied})"
            return "shaping applied (auto)"
    except Exception:
        pass
    return "shaping applied"


def _overlay_trigger_count(info: ScopeInfo) -> int:
    """Number of auto-learned trigger-overlay entries for a scope.

    Read-only and fail-open — a read problem simply reports zero. The overlay
    lives in the scope's learned ``triggers`` list (never in policy.yaml).
    """
    try:
        doc = json.loads((info.state_dir / "learned.json").read_text(encoding="utf-8"))
        entry = (doc.get("scopes") or {}).get(info.scope) or {}
        overlay = entry.get("triggers")
        if isinstance(overlay, list):
            return sum(1 for g in overlay if isinstance(g, dict))
    except Exception:
        pass
    return 0


def render_status_row(
    info: ScopeInfo, state: str, thresholds: dict[str, int]
) -> str:
    needed = required_sessions(thresholds)
    if state == STATE_SHAPED:
        detail = _shaped_detail(info)
    elif state == STATE_READY:
        detail = f"ready to shape ({info.sessions}/{needed} sessions)"
    elif state == STATE_OBSERVING:
        detail = (
            f"observing — {info.sessions}/{needed} sessions, "
            f"{remaining_sessions(info, thresholds)} to go"
        )
    else:
        detail = f"not configured ({info.sessions}/{needed} sessions)"
    overlay_count = _overlay_trigger_count(info)
    if overlay_count:
        detail += f", auto-learned triggers: {overlay_count}"
    return f"  {info.scope:<28} {state:<10} {detail}"


# ────────────────────────────────── prompting ────────────────────────────────


def prompt(message: str, reader: Callable[[str], str] = _default_reader) -> str:
    try:
        return reader(message).strip()
    except (EOFError, KeyboardInterrupt):
        raise Abort()


def prompt_choice(
    message: str, choices: Sequence[str], reader: Callable[[str], str] = _default_reader
) -> str:
    """Re-prompt until one of ``choices`` is entered."""
    valid = {c.lower() for c in choices}
    while True:
        answer = prompt(f"{message} [{'/'.join(choices)}]: ", reader).lower()
        if answer in valid:
            return answer
        print(f"  Please answer one of: {', '.join(choices)}")


def confirm(message: str, reader: Callable[[str], str] = _default_reader) -> bool:
    return prompt_choice(message, ("y", "n"), reader) == "y"


def prompt_multi_select(
    infos: Sequence[ScopeInfo], reader: Callable[[str], str] = _default_reader
) -> list[ScopeInfo]:
    """Numbered multi-select. ``all`` selects everything; blank re-prompts."""
    if len(infos) == 1:
        return list(infos)
    print("\n  Which agents should this cover?")
    for idx, info in enumerate(infos, 1):
        print(f"    {idx}. {info.scope}  ({info.sessions} session(s) recorded)")
    while True:
        answer = prompt("  Numbers (comma-separated) or 'all': ", reader).lower()
        if answer in {"all", "*"}:
            return list(infos)
        picked: list[ScopeInfo] = []
        ok = True
        for part in answer.replace(" ", "").split(","):
            if not part:
                continue
            try:
                index = int(part)
            except ValueError:
                ok = False
                break
            if not 1 <= index <= len(infos):
                ok = False
                break
            candidate = infos[index - 1]
            if candidate not in picked:
                picked.append(candidate)
        if ok and picked:
            return picked
        print("  Please enter numbers from the list, or 'all'.")


# ───────────────────────────────── degraded mode ─────────────────────────────


def print_manual_commands(writes: Sequence[ConfigWrite], out: Callable[[str], None] = _default_out) -> None:
    out("\n  `hermes` is not on PATH, so nothing was written.")
    out("  Run these yourself once the Hermes CLI is available:\n")
    for write in writes:
        out(f"    {' '.join(write.argv())}")
    out("")


# ─────────────────────────────────── flows ───────────────────────────────────


@dataclass
class RunContext:
    hermes_home: Path
    runner: Runner = _default_runner
    dry_run: bool = False
    assume_yes: bool = False
    reader: Callable[[str], str] = _default_reader
    out: Callable[[str], None] = _default_out
    thresholds: dict[str, int] = field(default_factory=dict)
    plugin_config: dict[str, Any] = field(default_factory=dict)
    have_hermes: bool = True
    applied: list[str] = field(default_factory=list)

    def settings(self, scope: str) -> dict[str, Any]:
        if not self.have_hermes:
            # Nothing to read from and nothing to read with — don't spawn a
            # process that cannot exist.
            return {
                "learned_mode": None,
                "bypass_rate": None,
                "scope_learned_mode": None,
                "scope_bypass_rate": None,
                "configured": False,
            }
        return scope_settings(scope, self.plugin_config, runner=self.runner)


def _confirm_writes(
    ctx: RunContext,
    title: str,
    writes: Sequence[ConfigWrite],
    extra: Sequence[str] = (),
) -> bool:
    """Show the diff, then ask. ``--yes`` skips the ask, never the diff.

    ``extra`` carries the non-config half of the disclosure (the learned
    overlay, the configure-state sidecar). It is printed with the config diff,
    before the question — an answer of ``y`` must never write something the
    user was not shown.
    """
    ctx.out(f"\n  {title}")
    for write in writes:
        ctx.out(build_diff(write))
    for line in extra:
        ctx.out(line)
    if not ctx.have_hermes:
        print_manual_commands(writes, ctx.out)
        return False
    if ctx.dry_run:
        ctx.out("  [dry-run] nothing will be written.")
        return True
    if ctx.assume_yes:
        return True
    return confirm("  Apply these changes?", ctx.reader)


def flow_shape(ctx: RunContext, infos: Sequence[ScopeInfo]) -> int:
    """Path 1 — read the history, show the shaping, apply on confirmation."""
    preset = load_base_preset()
    shaped_any = False
    for info in infos:
        recs = compute_recommendations(info, ctx.thresholds)
        if not recs:
            ctx.out(f"\n  {info.scope}: no telemetry recorded yet — nothing to shape.")
            ctx.out("    Choose the 'recommend' path for this agent instead.")
            continue
        # The compact diff (below) is the disclosure; the manual-era shaping
        # summary and projection block are gone — the savings report answers
        # "what did it do".
        proposal = proposed_assignment(info, recs)
        writes = plan_shape_writes(info, ctx.settings(info.scope))
        overlay = build_overlay_diff(info, current_assignment(info), proposal)
        if not _confirm_writes(ctx, f"Changes for {info.scope}:", writes, overlay):
            ctx.out(f"  Skipped {info.scope}. Nothing written.")
            continue

        changed = write_learned_overlay(info, recs, dry_run=ctx.dry_run)
        target = info.state_dir / "learned.json"
        if ctx.dry_run:
            ctx.out(f"  [dry-run] would write shaping overlay to {target}")
        elif changed:
            ctx.out(f"  Wrote shaping overlay to {target}")
        else:
            ctx.out(f"  Shaping overlay already current at {target}")
        ctx.applied.extend(apply_writes(writes, ctx.runner, ctx.dry_run, ctx.out))
        shaped_any = True

    if shaped_any and not ctx.dry_run:
        _offer_launcher(ctx)
        ctx.out("\n  What happens next")
        ctx.out("    · Restart the Hermes gateway to pick up the new configuration.")
        ctx.out("    · Shaped agents load a tighter tool set from their next session on.")
        ctx.out("    · Anything moved to expand-only comes back instantly via expand_tools.")
        ctx.out("    · Re-run this command any time to review or reset an agent.")
    return 0


def _offer_launcher(ctx: RunContext) -> None:
    """Offer the ``tool-belt`` command launcher after a confirmed apply.

    Only after a confirmed apply, only via the CLI module's confirmed helper —
    never as a side effect of a report or a dry run.
    """
    if ctx.dry_run:
        return
    cli = load_savings_cli()
    if cli is None:
        return

    def ask(message: str) -> bool:
        if ctx.assume_yes:
            return True
        try:
            return confirm(message, ctx.reader)
        except Abort:
            return False

    cli.ensure_launcher(
        ctx.hermes_home,
        PLUGIN_DIR / "tool-belt",
        confirm=ask,
        out=ctx.out,
    )


def flow_recommend(ctx: RunContext, infos: Sequence[ScopeInfo]) -> int:
    """Path 2 — observe first, shape later."""
    needed = required_sessions(ctx.thresholds)
    for info in infos:
        settings = ctx.settings(info.scope)
        writes = plan_recommend_writes(info, settings)
        sidecar = [
            f"  {_configure_state_path(info.state_dir)} — records the current "
            f"bypass_rate for {info.scope} so a later reset can restore it.",
            "    The learned overlay is not touched on this path.",
        ]
        if not _confirm_writes(ctx, f"Changes for {info.scope}:", writes, sidecar):
            ctx.out(f"  Skipped {info.scope}. Nothing written.")
            continue
        if not ctx.dry_run:
            remember_previous_full_ceiling_rate(
                info.state_dir, info.scope, settings.get("scope_bypass_rate")
            )
        ctx.applied.extend(apply_writes(writes, ctx.runner, ctx.dry_run, ctx.out))
        remaining = remaining_sessions(info, ctx.thresholds)
        ctx.out(
            f"  {info.scope}: observation mode on — {info.sessions}/{needed} sessions "
            f"recorded, {remaining} more needed."
        )

    if not ctx.dry_run:
        ctx.out("\n  What happens next")
        ctx.out("    · Restart the Hermes gateway; tool loading is unchanged for now.")
        ctx.out("    · Keep using your agents normally — every message is recorded.")
        ctx.out(
            f"    · Once a scope reaches {needed} sessions, re-run this command; it will "
            "offer the shaping review."
        )
        ctx.out("    · `python3 scripts/configure.py --status` shows progress at any time.")
    return 0


def flow_reset(ctx: RunContext, infos: Sequence[ScopeInfo]) -> int:
    """Undo shaping for a scope and return it to recommend mode."""
    for info in infos:
        settings = ctx.settings(info.scope)
        restore = previous_full_ceiling_rate(info.state_dir, info.scope)
        writes = plan_reset_writes(info, settings, restore)
        ctx.out(f"\n  Reset {info.scope}:")
        ctx.out(
            f"    clear this scope's learned carry / expand-only assignments and "
            f"shaping evidence from {info.state_dir / 'learned.json'}"
        )
        ctx.out(
            "    always-carried tools, trigger groups, and every other scope stay "
            "exactly as they are"
        )
        ctx.out("    the scope returns to recommend-mode observation")
        if not _confirm_writes(ctx, f"Config changes for {info.scope}:", writes):
            ctx.out(f"  Skipped {info.scope}. Nothing written.")
            continue
        removed = remove_learned_scope(info, dry_run=ctx.dry_run)
        if ctx.dry_run:
            ctx.out("  [dry-run] learned overlay left untouched.")
        elif removed:
            ctx.out("  Removed the learned overlay entry.")
        else:
            ctx.out("  No learned overlay entry to remove.")
        ctx.applied.extend(apply_writes(writes, ctx.runner, ctx.dry_run, ctx.out))
    return 0


def flow_status(ctx: RunContext, infos: Sequence[ScopeInfo]) -> int:
    """Read-only. Never writes anything, ever."""
    needed = required_sessions(ctx.thresholds)
    ctx.out("Tool Belt — configuration status")
    ctx.out(f"  Hermes home: {ctx.hermes_home}")
    ctx.out(f"  Sessions needed before shaping: {needed}")
    ctx.out("  " + _always_carried_line(ctx.plugin_config))
    if not ctx.have_hermes:
        ctx.out("  `hermes` not on PATH — config values could not be read.")
    if not infos:
        ctx.out("\n  No agent scopes found yet. Send a message through a gateway, then re-run.")
        return 0
    ctx.out("")
    for info in infos:
        state = classify_scope(info, ctx.settings(info.scope), ctx.thresholds)
        ctx.out(render_status_row(info, state, ctx.thresholds))
    return 0


def _pick_one(ctx: RunContext, items: Sequence, render, prompt_label: str):
    """Numbered single-pick. Returns the item, or None to cancel."""
    if len(items) == 1:
        return items[0]
    ctx.out("")
    for idx, item in enumerate(items, 1):
        ctx.out(f"    {idx}. {render(item)}")
    while True:
        answer = prompt(f"  {prompt_label} (number, or blank to cancel): ",
                        ctx.reader)
        if not answer:
            return None
        try:
            idx = int(answer)
        except ValueError:
            ctx.out(f"    Enter a number 1–{len(items)}.")
            continue
        if 1 <= idx <= len(items):
            return items[idx - 1]
        ctx.out(f"    Number must be 1–{len(items)}.")


def _pick_scopes(ctx: RunContext, infos: Sequence[ScopeInfo]) -> list[ScopeInfo]:
    """Pick one, several, or all channels for the chosen agent."""
    if len(infos) == 1:
        return list(infos)
    ctx.out("\n  Channels:")
    for idx, info in enumerate(infos, 1):
        ctx.out(f"    {idx}. {info.platform}  ({info.sessions} session(s))")
    while True:
        answer = prompt("  Numbers (comma-separated) or 'all', blank to cancel: ",
                        ctx.reader).lower()
        if not answer:
            return []
        if answer in {"all", "*"}:
            return list(infos)
        picked, ok = [], True
        for part in answer.replace(" ", "").split(","):
            if not part:
                continue
            try:
                i = int(part)
            except ValueError:
                ok = False
                break
            if not 1 <= i <= len(infos):
                ok = False
                break
            if infos[i - 1] not in picked:
                picked.append(infos[i - 1])
        if ok and picked:
            return picked
        ctx.out("    Enter numbers from the list, or 'all'.")


def _plan_mode_write(info: ScopeInfo, settings: dict[str, Any],
                     mode_value: str) -> ConfigWrite:
    """A single learned_mode write for a scope (apply | recommend)."""
    return ConfigWrite(
        key=f"{info.config_prefix}.learned_mode",
        after=mode_value,
        before=settings.get("scope_learned_mode") or settings.get("learned_mode"),
    )


def _apply_mode(ctx: RunContext, infos: Sequence[ScopeInfo], mode: str) -> int:
    """Set the shaping mode for the chosen scopes.

    mode == "history"  → learned_mode: apply, and shape from existing sessions
                          now (delegates to flow_shape).
    mode == "learning" → learned_mode: apply; no history run — auto-shaping at
                          session end handles it as usage accumulates.
    mode == "off"      → learned_mode: recommend; the learned overlay is not
                          applied (full-start carries everything) and
                          auto-shaping skips the scope.
    """
    if mode == "history":
        return flow_shape(ctx, infos)

    after = "apply" if mode == "learning" else "recommend"
    for info in infos:
        settings = ctx.settings(info.scope)
        write = _plan_mode_write(info, settings, after)
        if mode == "learning":
            extra = [
                "    Shaping is ON (learning). Tool Belt will tighten this",
                "    scope's loadout automatically as sessions accumulate;",
                "    nothing changes until there's enough evidence.",
            ]
        else:
            extra = [
                "    Shaping is OFF. This scope carries the full tool set;",
                "    the learned overlay is kept but not applied, and",
                "    automatic shaping is paused for it.",
            ]
        if not _confirm_writes(ctx, f"Changes for {info.scope}:", [write], extra):
            ctx.out(f"  Skipped {info.scope}. Nothing written.")
            continue
        ctx.applied.extend(apply_writes([write], ctx.runner, ctx.dry_run, ctx.out))
    return 0


def read_config_pins(profile_home: Path) -> list[str]:
    """The profile's ``plugins.tool-belt.always_carry`` config pins.

    Read directly from config.yaml (read-only; every WRITE still goes through
    ``hermes config set`` — Hermes owns the file).
    """
    try:
        yaml = require_yaml()
        raw = yaml.safe_load((profile_home / "config.yaml").read_text(
            encoding="utf-8")) or {}
        pins = ((raw.get("plugins") or {}).get("tool-belt") or {}).get("always_carry")
        if isinstance(pins, list):
            return [str(p) for p in pins if str(p).strip()]
    except SystemExit:
        raise
    except Exception:
        pass
    return []


def agent_tool_inventory(infos: Sequence[ScopeInfo]) -> list[str]:
    """Every tool observed in the agent's recorded ceilings, sorted.

    Union of ``ceiling_tools`` across recent prediction rows for the agent's
    scopes — a single row can be a tiny internal session, so no one row is
    trusted alone.
    """
    shaper = load_shaper()
    tools: set[str] = set()
    seen_dirs: set[Path] = set()
    scopes = {i.scope for i in infos}
    for info in infos:
        if info.state_dir in seen_dirs:
            continue
        seen_dirs.add(info.state_dir)
        rows = shaper.load_jsonl(info.state_dir / "predictions.jsonl")
        for row in rows[-400:]:
            if str(row.get("scope") or "") in scopes:
                # ceiling_tools on v2 rows; resident/active fields cover
                # sparse v1 telemetry so old installs still get a list.
                for field in ("ceiling_tools", "active_tools", "always_on_tools"):
                    for t in (row.get(field) or []):
                        tools.add(str(t))
    return sorted(tools)


def checkbox_picker(
    items: Sequence[str],
    checked: set[str],
    reader: Callable[[str], str],
    out: Callable[[str], None],
) -> set[str] | None:
    """Toggle a set of items; returns the final set, or None on cancel.

    On a real TTY: an in-place list — arrow keys / j/k to move, SPACE to
    toggle, Enter to accept, q to cancel. Anywhere else (pipes, tests,
    --yes automation): a numbered toggle loop with the same semantics.
    """
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _checkbox_picker_tty(items, set(checked))
        except Exception:
            pass  # raw-mode failed (odd terminal) — fall through
    state = set(checked)
    while True:
        out("")
        for idx, item in enumerate(items, 1):
            mark = "x" if item in state else " "
            out(f"    [{mark}] {idx:>2}. {item}")
        answer = prompt(
            "  Toggle (numbers, comma-separated) — blank when done, q to cancel: ",
            reader).lower()
        if answer == "q":
            return None
        if not answer:
            return state
        for part in answer.replace(" ", ",").split(","):
            if not part:
                continue
            try:
                i = int(part)
            except ValueError:
                out(f"    Not a number: {part}")
                continue
            if 1 <= i <= len(items):
                item = items[i - 1]
                state.symmetric_difference_update({item})
            else:
                out(f"    Out of range: {i}")


def _checkbox_picker_tty(items: Sequence[str], state: set[str]) -> set[str] | None:
    """Raw-terminal spacebar picker with a scrolling viewport.

    TTY-only; caller handles fallback. Renders a window sized to the
    terminal (a 50-tool inventory must work in a 30-row window — redrawing
    the full list would scroll off-screen and clamp the cursor-up, which
    made navigation look dead).
    """
    import os
    import shutil as _shutil
    import termios
    import tty

    fd = sys.stdin.fileno()
    n = len(items)
    rows = _shutil.get_terminal_size().lines
    height = max(3, min(n, rows - 4))
    cursor, top = 0, 0
    write = sys.stdout.write
    drawn = False

    def render() -> None:
        nonlocal drawn, top
        if cursor < top:
            top = cursor
        elif cursor >= top + height:
            top = cursor - height + 1
        if drawn:
            write(f"\x1b[{height + 1}A")
        drawn = True
        for i in range(top, top + height):
            item = items[i]
            mark = "x" if item in state else " "
            pointer = ">" if i == cursor else " "
            write(f"\x1b[2K  {pointer} [{mark}] {item}\n")
        above = f" ↑{top}" if top > 0 else ""
        below = f" ↓{n - top - height}" if top + height < n else ""
        write(f"\x1b[2K    space toggle · ↑/↓ move · enter done · "
              f"q cancel  ({cursor + 1}/{n}{above}{below})\n")
        sys.stdout.flush()

    render()
    old = termios.tcgetattr(fd)
    try:
        while True:
            tty.setraw(fd)
            try:
                ch = os.read(fd, 1).decode(errors="ignore")
                if ch == "\x1b":  # arrow keys: ESC [ A/B
                    seq = os.read(fd, 2).decode(errors="ignore")
                    ch = {"[A": "k", "[B": "j"}.get(seq, "")
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            if ch in ("\r", "\n"):
                return state
            if ch in ("q", "\x03", "\x04"):  # q / Ctrl-C / Ctrl-D
                return None
            if ch == " ":
                state.symmetric_difference_update({items[cursor]})
            elif ch == "j":
                cursor = min(cursor + 1, n - 1)
            elif ch == "k":
                cursor = max(cursor - 1, 0)
            render()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def flow_protected(ctx: RunContext, agent: str,
                   infos: Sequence[ScopeInfo]) -> int:
    """Manage the agent's protected (always-carried) tools.

    Policy pins are shown as a note and are not toggleable (union-only:
    users add pins, never remove shipped ones). The toggle list is the
    agent's observed tool inventory; checked = current config pins. The
    result is written to ``plugins.tool-belt.always_carry`` via
    ``hermes config set`` after the usual disclosed confirm.
    """
    # Profile home from the scope's state dir (<home>/state/tool-belt) —
    # never from the agent NAME, which for a root profile is the configured
    # display name (e.g. 'bernard'), not a profiles/ directory.
    profile_home = (infos[0].state_dir.parent.parent if infos
                    else ctx.hermes_home)
    preset = load_base_preset()
    policy_pins = sorted(preset.always_carry) if preset else []
    if policy_pins:
        ctx.out("\n  Note: Tool Belt will always carry: "
                + ", ".join(policy_pins))
    current = read_config_pins(profile_home)

    inventory = [t for t in agent_tool_inventory(infos)
                 if t not in set(policy_pins)]
    # Pins for tools not currently in the ceiling still count (inert-but-kept).
    for pin in current:
        if pin not in inventory and pin not in set(policy_pins):
            inventory.append(pin)
    inventory.sort()
    if not inventory:
        ctx.out("\n  No tool inventory recorded for this agent yet — run a "
                "session first.")
        return 0

    ctx.out(f"\n  Protected tools for {agent} — always carried, never shaped.")
    result = checkbox_picker(inventory, set(current), ctx.reader, ctx.out)
    if result is None:
        ctx.out("\n  Nothing changed.")
        return 0
    new_pins = sorted(result)
    if new_pins == sorted(current):
        ctx.out("\n  No changes.")
        return 0

    added = sorted(set(new_pins) - set(current))
    removed = sorted(set(current) - set(new_pins))
    write = ConfigWrite(
        key=f"{CONFIG_PREFIX}.always_carry",
        after=json.dumps(new_pins),
        before=json.dumps(sorted(current)) if current else None,
    )
    extra = []
    if added:
        extra.append("    Now protected: " + ", ".join(added))
    if removed:
        extra.append("    No longer protected (back to shaping): "
                     + ", ".join(removed))
    if not _confirm_writes(ctx, f"Changes for {agent}:", [write], extra):
        ctx.out("  Skipped. Nothing written.")
        return 0
    ctx.applied.extend(apply_writes([write], ctx.runner, ctx.dry_run, ctx.out))
    return 0


def _menu(ctx: RunContext, infos: Sequence[ScopeInfo]) -> int:
    """configure = turn shaping on/off per agent, per channel.

    Three steps: choose the agent, choose the channels, choose the mode
    (on/learning · on/use-history · off). No dashboard — the savings report
    answers "what did it do"; this only sets the mode. Every write is
    disclosed and confirmed at the funnel bottom.
    """
    agents = sorted({i.agent for i in infos})
    agent = _pick_one(ctx, agents, lambda a: a, "Agent")
    if agent is None:
        ctx.out("\n  Nothing selected.")
        return 0
    agent_scopes = [i for i in infos if i.agent == agent]

    ctx.out("\n    1. Protected tools")
    ctx.out("    2. Tool shaping options")
    while True:
        answer = prompt("  Option? [1/2, blank to cancel]: ", ctx.reader).strip()
        if not answer:
            ctx.out("\n  Nothing selected.")
            return 0
        if answer == "1":
            return flow_protected(ctx, agent, agent_scopes)
        if answer == "2":
            break
        ctx.out("    Enter 1 or 2.")

    selected = _pick_scopes(ctx, agent_scopes)
    if not selected:
        ctx.out("\n  Nothing selected.")
        return 0

    ctx.out(f"\n  Shaping mode for {agent} "
            f"({', '.join(i.platform for i in selected)}):")
    ctx.out("    1. On — learning    shape automatically from future usage")
    ctx.out("    2. On — use history shape now from recorded sessions")
    ctx.out("    3. Off              carry everything; don't shape")
    modes = {"1": "learning", "2": "history", "3": "off"}
    while True:
        answer = prompt("  Mode? [1/2/3, blank to cancel]: ", ctx.reader).strip()
        if not answer:
            ctx.out("\n  Nothing selected.")
            return 0
        if answer in modes:
            return _apply_mode(ctx, selected, modes[answer])
        ctx.out("    Enter 1, 2, or 3.")


def _ask_platforms(ctx: RunContext) -> list[str]:
    answer = prompt("  Which platforms do you use? (e.g. telegram, slack, cli): ", ctx.reader)
    return [p.strip().lower() for p in answer.replace(" ", ",").split(",") if p.strip()]


def split_platform_args(values: Sequence[str] | None) -> list[str] | None:
    """Comma/space-split --platform values exactly like the interactive
    prompt that teaches users to type "telegram, slack" (B2: a raw comma
    value used to silently become one garbage scope named
    ``agent:telegram,slack`` and could then receive real config writes)."""
    if not values:
        return None
    return [p.strip().lower()
            for v in values
            for p in str(v).replace(" ", ",").split(",")
            if p.strip()] or None


def _print_no_profiles(ctx: RunContext) -> None:
    """Said only when ``discover_state_dirs`` genuinely found nothing."""
    ctx.out(f"\n  No Hermes profiles found under {ctx.hermes_home}.")
    ctx.out("  Point this at the right home with --hermes-home, or install and")
    ctx.out("  start a Hermes gateway first, then re-run this command.")


def _print_fresh_install_guidance(ctx: RunContext) -> None:
    """The brand-new-install front door: profiles exist, telemetry does not.

    Never says the profiles are absent — the caller has just listed them by
    name. What is missing is telemetry, which only real gateway sessions
    produce.
    """
    needed = required_sessions(ctx.thresholds)
    ctx.out("\n  What to expect")
    ctx.out("    · Tool Belt records one row per gateway session; a brand-new")
    ctx.out("      install has none until you use your agents.")
    ctx.out(f"    · Once a scope reaches {needed} recorded session(s), re-running")
    ctx.out("      this command offers the shaping review.")
    ctx.out("    · `python3 scripts/configure.py --status` shows the count at any time.")
    ctx.out("    · To start observation mode now — before any telemetry exists —")
    ctx.out("      name the platforms you run:")
    ctx.out("        python3 scripts/configure.py --platform telegram --platform slack")


def _recover_fresh_install(ctx: RunContext, profile_filter: str | None, args) -> list[ScopeInfo]:
    """No scopes: distinguish "no profiles" from "no telemetry yet", then help.

    Returns any scopes recovered by asking which platforms the user runs; an
    empty list means the caller should stop after the guidance printed here.
    """
    found = discover_state_dirs(ctx.hermes_home, profile_filter)
    if not found:
        _print_no_profiles(ctx)
        return []

    labels = [label for label, _ in found]
    ctx.out(f"\n  Hermes profile(s) found: {', '.join(labels)}.")
    ctx.out("  No Tool Belt telemetry has been recorded for them yet, so the")
    ctx.out("  platforms each one runs aren't known.")
    # --platform was already honored by discovery; --yes must not prompt.
    if args.platform or args.yes:
        _print_fresh_install_guidance(ctx)
        return []

    platforms = _ask_platforms(ctx)
    infos = discover_scopes(ctx.hermes_home, profile_filter, platforms)
    if not infos:
        _print_fresh_install_guidance(ctx)
    return infos


# ──────────────────────────────────── CLI ────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="configure.py",
        description="Tool Belt onboarding — configure adaptive tool loadouts per agent.",
    )
    parser.add_argument("--status", action="store_true", help="print per-scope state and telemetry counts, write nothing")
    parser.add_argument("--agent", default=None, help="restrict to one agent/profile (skips selection)")
    parser.add_argument("--path", choices=("shape", "recommend"), default=None, help="skip the path question")
    parser.add_argument("--reset", metavar="AGENT", default=None, help="return an agent to recommend mode")
    parser.add_argument("--yes", action="store_true", help="apply without the interactive confirmation")
    parser.add_argument("--dry-run", action="store_true", help="show every diff, write nothing")
    parser.add_argument("--platform", action="append", default=None, help="platform to assume when a profile has no telemetry (repeatable)")
    parser.add_argument("--hermes-home", type=Path, default=None, help="override HERMES_HOME discovery")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    hermes_home = args.hermes_home or default_hermes_home()

    # Both of the PyYAML-dependent reads (the shaper's policy thresholds, then
    # the plugin config block) happen here, before a single line of the
    # conversation is printed — so a wrong-interpreter run exits 2 with the
    # guard's message and nothing else, never half a rendered flow.
    ctx = RunContext(
        hermes_home=hermes_home,
        dry_run=bool(args.dry_run),
        assume_yes=bool(args.yes),
        thresholds=shape_thresholds(),
        have_hermes=hermes_available(),
    )
    if ctx.have_hermes:
        ctx.plugin_config = read_plugin_config(ctx.runner)

    profile_filter = args.reset or args.agent
    args.platform = split_platform_args(args.platform)
    infos = discover_scopes(hermes_home, profile_filter, args.platform)

    # A filter that matches nothing on a populated install is a wrong NAME,
    # not an empty install — say so, and name what exists (M3/P3: the generic
    # "no profiles found" text told users with a typo to go install Hermes,
    # and a --dry-run silently no-op'd with exit 0).
    if profile_filter and not infos:
        available = discover_state_dirs(hermes_home, None)
        if available:
            ctx.out(f"\n  No profile matching {profile_filter!r} under {hermes_home}.")
            names = []
            for label, state_dir in available:
                phome = hermes_home if label == "default" else (
                    hermes_home / "profiles" / label)
                shown = configured_agent_name(phome)
                names.append(f"{label} (agent: {shown})" if shown and shown != label
                             else label)
            ctx.out("  Profiles found: " + ", ".join(names))
            ctx.out("  --agent accepts either the profile name or its configured "
                    "agent name.")
            return 2

    if args.status:
        return flow_status(ctx, infos)

    ctx.out("=" * 64)
    ctx.out("  Tool Belt — configure")
    ctx.out("=" * 64)

    if not ctx.have_hermes:
        ctx.out("\n  `hermes` is not on PATH. Running in preview mode: every change")
        ctx.out("  below is printed as a command for you to run by hand.")

    try:
        if not infos:
            infos = _recover_fresh_install(ctx, profile_filter, args)
            if not infos:
                return 0

        if args.reset:
            rc = flow_reset(ctx, infos)
        elif args.path == "shape":
            rc = flow_shape(ctx, infos)
        elif args.path == "recommend":
            rc = flow_recommend(ctx, infos)
        else:
            rc = _menu(ctx, infos)
    except Abort:
        ctx.out("\n\n  Stopped. No changes were written.")
        return 0

    if ctx.applied:
        ctx.out("\n  Would apply:" if ctx.dry_run else "\n  Applied:")
        for line in ctx.applied:
            ctx.out(f"    {line}")
    elif not ctx.dry_run:
        ctx.out("\n  No configuration changes were written.")
    return rc


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
