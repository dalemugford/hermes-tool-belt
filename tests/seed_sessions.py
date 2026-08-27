#!/usr/bin/env python3
"""Scripted-session seeder — synthetic telemetry for onboarding simulation.

Design: ``docs/TEST_HARNESS.md``.

``scripts/configure.py`` discovers ``agent:platform`` scopes from the
telemetry the runtime hooks write during real gateway sessions. Producing
that telemetry for real needs a live provider, a running gateway, *and* a
messaging-platform account (``docs/TEST_HARNESS.md`` §1). This module
replaces only the transport: it runs the **real** policy resolver and the
**real** predictor over scripted messages — the same two calls
``_on_pre_gateway_dispatch`` makes (``__init__.py:1513-1514``) — and writes
the resulting rows straight into ``predictions.jsonl`` /
``tool_calls.jsonl`` under a throwaway ``HERMES_HOME``.

Because the rows are built by constructing a real
``logger_io.PredictionRecord`` and calling ``.to_dict()``, the harness
inherits the production schema by construction: a renamed or dropped field
becomes a ``TypeError`` here rather than a silent test blind spot.

Deliberate deviations from production
=====================================

These are the only places the seeded rows differ from what the runtime
would write, and each is a determinism trade made knowingly:

* **Token counts are fixed synthetic integers** (``SEEDED_CEILING_TOKENS`` /
  ``SEEDED_NARROWED_TOKENS``) with ``tokens_estimator="seeded"``, not
  ``logger_io.estimate_tokens()``. The real estimator reports
  ``tiktoken-cl100k`` when tiktoken happens to be installed and
  ``chars-div-4`` otherwise, so counts differ between machines and between
  CI and a laptop. Neither ``configure.py`` nor ``shape-ceiling.py`` reads
  these fields, so the constants cost nothing and the estimator name keeps
  the provenance honest. **Seeded telemetry is therefore not a valid
  fixture source for ``scripts/savings-report.py``** — a savings assertion
  over these rows would be measuring the seed constant.
* **``ts`` is a seeded monotonic constant** (``BASE_TS + session*1000 +
  turn``), never ``time.time()``, so the shaper's recency window is stable
  across runs.
* **``hermes_session_id`` is a synthetic UUID-shaped string** per session
  (``20260827_120000_<8hex>``, the hex derived from the script name and
  session index) rather than a real Hermes session UUID.
* **``provider`` / ``model`` are left blank** — no API call happens, so
  there is nothing honest to record.

Everything else — ``triggers_fired``, ``triggers_suppressed``,
``always_on_tools``, the narrowed tool set — comes from real policy
evaluation against ``policy.yaml``, so a policy regression fails the seed
instead of being papered over by hardcoded fixture values.

Usage
=====

As a library (what ``tests/test_onboarding_e2e.py`` does)::

    from seed_sessions import load_script, seed, seed_all

    result = seed(load_script(SCRIPTS_DIR / "terminal-heavy.yaml"), home)

As a command, to populate a scratch home for a manual ``configure.py`` demo::

    python3 tests/seed_sessions.py --home /tmp/demo-home
    HERMES_HOME=/tmp/demo-home python3 scripts/configure.py --status
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

TESTS_DIR = Path(__file__).resolve().parent
PLUGIN_DIR = TESTS_DIR.parent
SCRIPTS_DIR = TESTS_DIR / "scripts"

sys.path.insert(0, str(TESTS_DIR))
import conftest  # noqa: F401,E402 — registers tool_belt_plugin

logger_io = importlib.import_module("tool_belt_plugin.logger_io")
presets = importlib.import_module("tool_belt_plugin.presets")
predictor = importlib.import_module("tool_belt_plugin.predictor")
learned = importlib.import_module("tool_belt_plugin.learned")

#: Fixed epoch for every seeded row. 2026-01-01T00:00:00Z — far enough from
#: "now" that a seeded row is obviously seeded when read by hand.
BASE_TS = 1767225600.0

#: Session-id prefix, shaped like Hermes' own ``YYYYMMDD_HHMMSS_<hex>``.
SESSION_ID_PREFIX = "20260827_120000"

#: Synthetic token counts. See the module docstring for why these are not
#: computed — the estimator is machine-dependent and no consumer reads them.
SEEDED_CEILING_TOKENS = 12000
SEEDED_NARROWED_TOKENS = 3500
SEEDED_TOKENS_ESTIMATOR = "seeded"


class ScriptMismatch(AssertionError):
    """A script's ``expect_triggers`` disagreed with the real predictor.

    Raised at seed time, deliberately: a policy change that breaks trigger
    matching should fail loudly here rather than produce quietly-wrong
    telemetry that the tests then assert against.
    """


@dataclass
class SeedResult:
    """What one script run put on disk."""

    name: str
    hermes_home: Path
    state_dir: Path
    scope: str
    sessions: int
    predictions: int
    tool_calls: int


# ────────────────────────────── script loading ───────────────────────────────


def load_script(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a scripted-conversation YAML file."""
    import yaml  # type: ignore[import-untyped]

    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"script {path} is not a YAML mapping")
    for key in ("name", "agent", "platform", "sessions", "turns"):
        if key not in data:
            raise ValueError(f"script {path} is missing required key {key!r}")
    if not isinstance(data.get("turns"), list) or not data["turns"]:
        raise ValueError(f"script {path} has no turns")
    return data


def _state_dir_for(hermes_home: Path, profile: Any) -> Path:
    """Root state dir for the default profile, nested for a named one.

    Mirrors ``configure.discover_state_dirs`` (``scripts/configure.py:190``),
    which scans from the Hermes root so it can report on every profile at
    once, while the runtime's ``learned.state_dir()`` is the flat root path.
    """
    name = str(profile or "").strip().lower()
    if not name or name == "default":
        return hermes_home / "state" / "tool-belt"
    return hermes_home / "profiles" / name / "state" / "tool-belt"


@contextlib.contextmanager
def _hermes_home(home: Path) -> Iterator[None]:
    """Point ``HERMES_HOME`` at ``home`` for the duration of the block.

    Non-negotiable for isolation: ``learned.state_dir()``
    (``learned.py:66-67``) falls back to ``~/.hermes``, so a
    ``resolve_preset`` call without this reads the developer's real learned
    overlay (``docs/TEST_HARNESS.md`` §F.3).
    """
    previous = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = str(home)
    # learned.py caches learned.json by mtime; drop it so a prior home's
    # overlay can never leak into this one.
    learned._CACHE.update({"path": None, "mtime_ns": None, "state": None, "hash": ""})
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = previous
        learned._CACHE.update({"path": None, "mtime_ns": None, "state": None, "hash": ""})


# ─────────────────────────────── row building ────────────────────────────────


def _ceiling_tools(preset: Any) -> list[str]:
    """A plausible user ceiling: always-on ∪ every trigger's tools ∪ always-off."""
    out: list[str] = []
    for tool in list(getattr(preset, "always_on", []) or []):
        if tool not in out:
            out.append(str(tool))
    for group in getattr(preset, "triggers", []) or []:
        for tool in getattr(group, "tools", []) or []:
            if tool not in out:
                out.append(str(tool))
    for tool in getattr(preset, "always_off", []) or []:
        if tool not in out:
            out.append(str(tool))
    return out


def _session_uid(script_name: str, index: int) -> str:
    digest = hashlib.sha1(f"{script_name}-{index}".encode("utf-8")).hexdigest()[:8]
    return f"{SESSION_ID_PREFIX}_{digest}"


def _normalize_calls(raw: Any) -> list[dict[str, Any]]:
    """Accept both the bare-string and mapping forms of a ``calls`` entry."""
    out: list[dict[str, Any]] = []
    for entry in raw or []:
        if isinstance(entry, str):
            out.append({"tool": entry, "expanded": False, "count": 1})
        elif isinstance(entry, dict):
            tool = str(entry.get("tool") or "").strip()
            if not tool:
                continue
            out.append(
                {
                    "tool": tool,
                    "expanded": bool(entry.get("expanded")),
                    "count": max(1, int(entry.get("count", 1))),
                }
            )
    return out


def build_prediction_row(
    *,
    script_name: str,
    scope: str,
    agent: str,
    platform: str,
    session_index: int,
    turn_index: int,
    text: str,
    preset: Any,
    prediction: Any,
    ceiling: Sequence[str],
    unknown_kept: Sequence[str],
    expanded_tools: Sequence[str],
) -> dict[str, Any]:
    """Build one ``predictions.jsonl`` row via ``logger_io.PredictionRecord``.

    Going through the production dataclass (rather than a dict literal) is
    what makes the harness inherit the production schema — see the module
    docstring.
    """
    allowed = prediction.allowed_tool_names
    narrowed = [] if allowed == presets.WILDCARD_ALWAYS_ON else [str(t) for t in allowed]
    always_on = list(preset.always_on) if isinstance(preset.always_on, list) else []
    fired = set(prediction.triggers_fired)
    trigger_tools = {
        str(g.name): [str(t) for t in (g.tools or [])]
        for g in (getattr(preset, "triggers", []) or [])
        if str(g.name) in fired
    }
    session_uid = _session_uid(script_name, session_index)

    record = logger_io.PredictionRecord(
        ts=BASE_TS + session_index * 1000 + turn_index,
        prediction_id=f"{script_name}-s{session_index}-t{turn_index}",
        session_id=f"{agent}:main:{platform}",
        hermes_session_id=session_uid,
        channel=scope,  # legacy alias the production writer still emits
        agent=agent,
        platform=platform,
        scope=scope,
        message_hash=logger_io.hash_message(text),
        message_preview=logger_io.message_preview(text),
        preset=prediction.preset_name,
        triggers_fired=list(prediction.triggers_fired),
        triggers_suppressed=list(prediction.triggers_suppressed),
        always_on_count=prediction.always_on_count,
        always_on_tools=always_on,
        ceiling_count=len(ceiling),
        narrowed_count=len(narrowed),
        ceiling_tokens=SEEDED_CEILING_TOKENS,
        narrowed_tokens=SEEDED_NARROWED_TOKENS,
        tokens_estimator=SEEDED_TOKENS_ESTIMATOR,
        ceiling_tools=list(ceiling),
        allowed_tools=narrowed,
        cut_tools=[t for t in ceiling if t not in set(narrowed)],
        unknown_kept_tools=list(unknown_kept),
        mcp_passthrough_tools=[],
        trigger_tools_by_group=trigger_tools,
        expanded_tools=list(expanded_tools),
        policy_source=str(getattr(preset, "policy_source", "preset")),
        policy_version=str(getattr(preset, "policy_version", "")),
        learned_mode=str(getattr(preset, "learned_mode", "recommend")),
        learned_scope=str(getattr(preset, "learned_scope", "")),
        learned_changes=list(getattr(preset, "learned_changes", []) or []),
        tool_list_hash=hashlib.sha256(
            json.dumps(narrowed, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16],
    )
    return record.to_dict()


def build_tool_call_row(
    *,
    prediction_id: str,
    session_id: str,
    ts: float,
    scope: str,
    agent: str,
    platform: str,
    tool_name: str,
    expanded: bool,
) -> dict[str, Any]:
    """Build one ``tool_calls.jsonl`` row shaped like ``_on_post_tool_call``.

    ``was_expanded`` / ``expand_tools_used`` are the two flags the shaper
    accepts as promote evidence (``shape-ceiling.py:286-291``).
    """
    row: dict[str, Any] = {
        "ts": ts,
        "prediction_id": prediction_id,
        "session_id": session_id,
        "tool_name": tool_name,
        "agent": agent,
        "platform": platform,
        "scope": scope,
        "source": "gateway",
        "was_initially_available": not expanded,
        "was_cut": expanded,
        "was_expanded": expanded,
    }
    if expanded:
        row["expand_tools_used"] = True
        row["expand_category"] = ""
        row["expanded_tool"] = tool_name
        row["turns_until_used"] = 0
    return row


# ──────────────────────────────── the seeder ─────────────────────────────────


def _append_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """One ``write_text`` per file, byte-identical to append-mode output.

    Existing content is preserved so several scripts can share a state dir.
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(existing + payload, encoding="utf-8")


def seed(
    script: dict[str, Any],
    hermes_home: str | Path,
    *,
    verify_triggers: bool = True,
    sessions_override: int | None = None,
    profile_override: str | None = None,
) -> SeedResult:
    """Run ``script`` against the real predictor and write its telemetry.

    ``sessions_override`` and ``profile_override`` let one script cover more
    than one fixture shape (e.g. the same conversation seeded into a named
    profile for the profile-scoping test) without duplicating the YAML.
    """
    hermes_home = Path(hermes_home)
    name = str(script["name"])
    agent = str(script["agent"])
    platform = str(script["platform"])
    scope = f"{agent}:{platform}"
    profile = profile_override if profile_override is not None else script.get("profile")
    sessions = int(sessions_override if sessions_override is not None else script["sessions"])
    plugin_config = dict(script.get("config") or {})
    unknown_kept = [str(t) for t in (script.get("unknown_kept") or [])]
    turns = list(script["turns"])

    state_dir = _state_dir_for(hermes_home, profile)
    state_dir.mkdir(parents=True, exist_ok=True)

    pred_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []

    with _hermes_home(hermes_home):
        # Load the base policy once so a broken policy.yaml surfaces here and
        # not as a confusing per-turn wildcard fallback.
        base = presets.load_base_policy()
        if base.is_wildcard:
            raise ScriptMismatch(
                "policy.yaml did not load — the predictor would fall back to wildcard"
            )
        # resolve_preset is constant for a (plugin_config, scope) pair; hoisting
        # it out of the turn loop removes most of the per-turn cost.
        preset = presets.resolve_preset(plugin_config, scope)
        ceiling = _ceiling_tools(preset)

        for i in range(sessions):
            session_uid = _session_uid(name, i)
            for t, turn in enumerate(turns):
                text = str(turn.get("user") or "")
                prediction = predictor.predict(text, None, preset)

                expected = turn.get("expect_triggers")
                if verify_triggers and expected is not None:
                    actual = sorted(prediction.triggers_fired)
                    if actual != sorted(str(e) for e in expected):
                        raise ScriptMismatch(
                            f"{name} turn {t} ({text!r}): expected triggers "
                            f"{sorted(str(e) for e in expected)}, predictor fired {actual}"
                        )

                calls = _normalize_calls(turn.get("calls"))
                pred_rows.append(
                    build_prediction_row(
                        script_name=name,
                        scope=scope,
                        agent=agent,
                        platform=platform,
                        session_index=i,
                        turn_index=t,
                        text=text,
                        preset=preset,
                        prediction=prediction,
                        ceiling=ceiling,
                        unknown_kept=unknown_kept,
                        expanded_tools=[c["tool"] for c in calls if c["expanded"]],
                    )
                )
                prediction_id = f"{name}-s{i}-t{t}"
                for call in calls:
                    for _ in range(call["count"]):
                        call_rows.append(
                            build_tool_call_row(
                                prediction_id=prediction_id,
                                session_id=f"{agent}:main:{platform}",
                                ts=BASE_TS + i * 1000 + t,
                                scope=scope,
                                agent=agent,
                                platform=platform,
                                tool_name=call["tool"],
                                expanded=call["expanded"],
                            )
                        )

    _append_jsonl(state_dir / "predictions.jsonl", pred_rows)
    _append_jsonl(state_dir / "tool_calls.jsonl", call_rows)

    return SeedResult(
        name=name,
        hermes_home=hermes_home,
        state_dir=state_dir,
        scope=scope,
        sessions=sessions,
        predictions=len(pred_rows),
        tool_calls=len(call_rows),
    )


def seed_all(
    scripts_dir: str | Path,
    hermes_home: str | Path,
    *,
    verify_triggers: bool = True,
) -> list[SeedResult]:
    """Seed every ``*.yaml`` in ``scripts_dir``, in filename order."""
    directory = Path(scripts_dir)
    return [
        seed(load_script(path), hermes_home, verify_triggers=verify_triggers)
        for path in sorted(directory.glob("*.yaml"))
    ]


# ──────────────────────────────────── CLI ────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seed_sessions.py",
        description=(
            "Populate a throwaway Hermes home with scripted telemetry, so "
            "`scripts/configure.py` can be driven end to end by hand."
        ),
    )
    parser.add_argument("--home", type=Path, required=True, help="target HERMES_HOME (created if absent)")
    parser.add_argument(
        "--script",
        action="append",
        default=None,
        metavar="PATH",
        help=f"script to seed (repeatable; default: every *.yaml in {SCRIPTS_DIR})",
    )
    parser.add_argument("--sessions", type=int, default=None, help="override each script's session count")
    parser.add_argument("--profile", default=None, help="override each script's target profile")
    parser.add_argument(
        "--no-verify-triggers",
        action="store_true",
        help="do not fail when a script's expect_triggers disagrees with the predictor",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    paths = [Path(p) for p in args.script] if args.script else sorted(SCRIPTS_DIR.glob("*.yaml"))
    if not paths:
        print(f"No scripts found under {SCRIPTS_DIR}.", file=sys.stderr)
        return 1

    home = args.home.expanduser()
    home.mkdir(parents=True, exist_ok=True)
    for path in paths:
        result = seed(
            load_script(path),
            home,
            verify_triggers=not args.no_verify_triggers,
            sessions_override=args.sessions,
            profile_override=args.profile,
        )
        print(
            f"  {result.name:<16} {result.scope:<24} "
            f"{result.sessions} session(s), {result.predictions} prediction(s), "
            f"{result.tool_calls} tool call(s) → {result.state_dir}"
        )
    print(f"\nSeeded {len(paths)} script(s) into {home}. Try:")
    print(f"  HERMES_HOME={home} python3 {PLUGIN_DIR / 'scripts' / 'configure.py'} --status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
