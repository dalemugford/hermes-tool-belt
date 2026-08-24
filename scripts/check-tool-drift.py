#!/usr/bin/env python3
"""Tool-name drift detector for the tool-belt preset.

The plugin's "unknown tools" safe-default keeps any ceiling tool that the
preset doesn't name switched ON, every turn. That's the right failure mode
(never silently drop a capability), but it means new tools added upstream
silently defeat the narrowing until someone updates ``policy.yaml``. This
script makes that drift visible and easy to close.

What it does
============

  1. Builds the preset's KNOWN set = ``always_on`` ∪ every trigger group's
     ``tools`` ∪ ``always_off`` (from ``policy.yaml``).
  2. Reads the live ceiling tool names — preferring the most recent
     ``predictions.jsonl`` row's ``ceiling_tools`` (the real per-scope
     ceiling, already at tool-name granularity), and falling back to
     Hermes' toolset table (``toolsets``) resolved down to tool names
     when no telemetry is present.
  3. Prints the diff: tools in the ceiling but NOT named anywhere in the
     preset (these are the ones kept on as "unknown").

Exit status
===========

  0  no drift — every ceiling tool is named in the preset.
  1  drift found — one or more ceiling tools are unnamed.
  2  could not determine the ceiling (no toolsets import, no telemetry).

Usage
=====

  python3 scripts/check-tool-drift.py
  python3 scripts/check-tool-drift.py --json
  python3 scripts/check-tool-drift.py --update   # append drifted tools to
                                                 # a review block in policy.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

_PLUGIN_DIR = Path(__file__).resolve().parent.parent
_POLICY_PATH = _PLUGIN_DIR / "policy.yaml"

_AUTO_HEADER = "# AUTO-DISCOVERED TOOLS (review and assign to trigger groups)"


def default_state_dir() -> Path:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(home) / "state" / "tool-belt"


def _base_tool_name(name: str) -> str:
    """Strip Hermes' ``mcp_`` send-time prefix so names match the preset."""
    if isinstance(name, str) and name.startswith("mcp_"):
        return name[len("mcp_"):]
    return name


# ─── Preset known-set ──────────────────────────────────────────────────────

def load_known_set(policy_path: Path = _POLICY_PATH) -> set[str]:
    """Return always_on ∪ all trigger tools ∪ always_off from policy.yaml."""
    import yaml  # type: ignore[import-untyped]

    with policy_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return set()

    known: set[str] = set()
    ao = data.get("always_on")
    if isinstance(ao, list):
        known.update(str(t) for t in ao if isinstance(t, str))
    off = data.get("always_off")
    if isinstance(off, list):
        known.update(str(t) for t in off if isinstance(t, str))
    triggers = data.get("triggers")
    if isinstance(triggers, list):
        for group in triggers:
            if not isinstance(group, dict):
                continue
            tools = group.get("tools")
            if isinstance(tools, list):
                known.update(str(t) for t in tools if isinstance(t, str))
    return known


# ─── Live ceiling ──────────────────────────────────────────────────────────

def ceiling_from_toolsets() -> list[str] | None:
    """Fall back to Hermes' toolset table, resolved to tool names.

    ``get_toolset_names()`` returns CATEGORY names (``browser``, ``coding``
    …), not tool names, so we resolve each category through
    ``resolve_toolset`` to get the tool-name granularity the preset and the
    known set are expressed in. Returns None when ``toolsets`` isn't
    importable. This is a coarse fallback: it is the full installed
    universe, not a single scope's ceiling.
    """
    try:
        import toolsets  # type: ignore[import-not-found]
    except Exception:
        return None
    get_names = getattr(toolsets, "get_toolset_names", None)
    resolve = getattr(toolsets, "resolve_toolset", None)
    if not callable(get_names):
        return None
    try:
        categories = list(get_names())
    except Exception:
        return None

    names: set[str] = set()
    for cat in categories:
        tools: Any = None
        if callable(resolve):
            try:
                tools = resolve(cat)
            except Exception:
                tools = None
        if not tools:
            # Category name itself is not a tool — skip if it won't resolve.
            continue
        for t in tools:
            if isinstance(t, str):
                names.add(t)
            else:
                n = getattr(t, "name", None)
                if isinstance(n, str):
                    names.add(n)
    return sorted(names) if names else None


def ceiling_from_telemetry(state_dir: Path) -> list[str] | None:
    """Most recent predictions.jsonl row that carries a non-empty ceiling."""
    path = state_dir / "predictions.jsonl"
    if not path.exists():
        return None
    latest: list[str] | None = None
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                ceil = row.get("ceiling_tools")
                if isinstance(ceil, list) and ceil:
                    latest = [str(t) for t in ceil if isinstance(t, str)]
    except Exception:
        return None
    return latest


def resolve_ceiling(state_dir: Path) -> tuple[list[str] | None, str]:
    """Return (ceiling names, source label).

    Telemetry wins: it's the real, per-scope ceiling at tool-name
    granularity. The toolsets table is a coarse fallback (whole universe).
    """
    tele = ceiling_from_telemetry(state_dir)
    if tele is not None:
        return tele, str(state_dir / "predictions.jsonl")
    ts = ceiling_from_toolsets()
    if ts is not None:
        return ts, "toolsets (resolved to tool names)"
    return None, "(none)"


def compute_drift(ceiling: Iterable[str], known: set[str]) -> list[str]:
    """Ceiling tools whose base name is not named anywhere in the preset."""
    drift: list[str] = []
    for name in ceiling:
        if _base_tool_name(name) not in known:
            drift.append(name)
    # Stable, de-duplicated ordering.
    seen: set[str] = set()
    out: list[str] = []
    for n in sorted(drift):
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


# ─── --update: append a review block to policy.yaml ────────────────────────

def apply_update(policy_path: Path, drifted: list[str]) -> int:
    """Append drifted tools to the AUTO-DISCOVERED review block.

    This is a REVIEW AID only — the tools land in a commented block at the
    end of the file for a human to place into a trigger group. They are NOT
    added to always_on and NOT auto-gated.
    """
    text = policy_path.read_text(encoding="utf-8")
    existing_block: set[str] = set()
    if _AUTO_HEADER in text:
        # Collect names already listed so re-runs are idempotent.
        after = text.split(_AUTO_HEADER, 1)[1]
        for raw in after.splitlines():
            s = raw.strip()
            if s.startswith("#   - "):
                existing_block.add(s[len("#   - "):].strip())

    fresh = [d for d in drifted if d not in existing_block]
    if not fresh:
        print("check-tool-drift: nothing new to append (block already current).")
        return 0

    lines = [] if text.endswith("\n") else ["\n"]
    if _AUTO_HEADER not in text:
        lines.append("\n")
        lines.append(_AUTO_HEADER + "\n")
        lines.append("# These tools appeared in the live ceiling but are not named in any\n")
        lines.append("# trigger group or always_on/always_off. Review each and either add it\n")
        lines.append("# to an existing trigger group, a new group, or always_off. This block\n")
        lines.append("# is a review aid — it is NOT parsed or applied automatically.\n")
    for name in fresh:
        lines.append(f"#   - {name}\n")

    policy_path.write_text(text + "".join(lines), encoding="utf-8")
    print(f"check-tool-drift: appended {len(fresh)} tool(s) to the review block in {policy_path}")
    return 0


# ─── main ──────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect tool-name drift vs. policy.yaml preset.")
    parser.add_argument("--policy", type=Path, default=_POLICY_PATH, help="path to policy.yaml")
    parser.add_argument("--state-dir", type=Path, default=None, help="tool-belt state dir (telemetry)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--update", action="store_true",
        help="append drifted tools to a review block at the end of policy.yaml",
    )
    args = parser.parse_args(argv)

    state_dir = args.state_dir or default_state_dir()
    known = load_known_set(args.policy)
    ceiling, source = resolve_ceiling(state_dir)

    if ceiling is None:
        msg = "check-tool-drift: could not determine live ceiling (no toolsets import, no telemetry)."
        if args.json:
            print(json.dumps({"error": "no_ceiling", "known_count": len(known)}))
        else:
            print(msg, file=sys.stderr)
        return 2

    drifted = compute_drift(ceiling, known)

    if args.json:
        print(json.dumps({
            "source": source,
            "ceiling_count": len(ceiling),
            "known_count": len(known),
            "drift_count": len(drifted),
            "drift": drifted,
        }, indent=2))
    else:
        print(f"check-tool-drift: ceiling={len(ceiling)} known={len(known)} source={source}")
        if drifted:
            print(f"check-tool-drift: {len(drifted)} tool(s) in ceiling not named in preset:")
            for name in drifted:
                print(f"    {name}")
            print("\nRun with --update to append these to a review block in policy.yaml,")
            print("then assign each to a trigger group / always_off by hand.")
        else:
            print("check-tool-drift: no drift — every ceiling tool is named in the preset. ✓")

    if args.update and drifted:
        apply_update(args.policy, drifted)

    return 1 if drifted else 0


if __name__ == "__main__":
    raise SystemExit(main())
