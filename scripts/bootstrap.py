#!/usr/bin/env python3
"""Day-one warm start for the tool-belt plugin — mode-aware.

The plugin runs in two modes per scope (cache-on default for Anthropic /
OpenAI auto-cache; cache-off fallback for kimi / gpt-5.4-mini). This script
runs the right warm-start for each:

  · Cache-on scopes → ``shape-ceiling.py`` reports per-tool promote /
    demote candidates from real ``expand_tools`` evidence in live
    telemetry. Conservative thresholds; nothing surfaces until ≥2
    sessions of data per scope.

  · Cache-off scopes → ``harvest-replay.py`` replays your existing
    Hermes session JSONLs through the per-turn predictor, then the
    analyzer mines trigger-keyword candidates and tool promotions.
    Useful on day one because it leverages history you already have.

Both paths produce a unified ranked **TOP ACTIONS** summary. If neither
finds anything, that's reported honestly — usually means thin sample
size, not that the plugin has nothing to do.

Idempotent: re-runs refresh the harvest and re-read live telemetry
without disrupting either.

Usage:
  python3 scripts/bootstrap.py
  python3 scripts/bootstrap.py --profile assistant-a  # one profile only
  python3 scripts/bootstrap.py --window-days 60   # harvest window
  python3 scripts/bootstrap.py --skip-harvest     # cache-on path only
  python3 scripts/bootstrap.py --skip-shape       # cache-off path only
  python3 scripts/bootstrap.py --quiet            # only print top actions
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUGIN_DIR = HERE.parent


def _default_python() -> str:
    """Interpreter for child scripts, with an explicit scheduler override."""
    return os.environ.get("HERMES_PYTHON") or sys.executable


def _discover_state_dirs(hermes_home: Path, profile_filter: str | None) -> list[tuple[str, Path]]:
    """Return ``[(label, state_dir), ...]`` for every profile's live state
    directory. Used by the cache-on path (shape-ceiling reads live
    telemetry, not harvest)."""
    out: list[tuple[str, Path]] = []
    root_state = hermes_home / "state" / "tool-belt"
    if (not profile_filter or profile_filter == "default") and root_state.is_dir():
        out.append(("default", root_state))
    profiles_dir = hermes_home / "profiles"
    if profiles_dir.is_dir():
        for child in sorted(profiles_dir.iterdir()):
            if not child.is_dir():
                continue
            if child.name == "default":
                continue  # reserved by Hermes for the root profile
            if profile_filter and child.name != profile_filter:
                continue
            p_state = child / "state" / "tool-belt"
            if p_state.is_dir():
                out.append((child.name, p_state))
    return out


def _discover_harvest_dirs(hermes_home: Path, profile_filter: str | None) -> list[tuple[str, Path]]:
    """Return ``[(label, harvest_dir), ...]`` for every profile that produced
    harvest output (used by the cache-off path)."""
    out: list[tuple[str, Path]] = []
    for label, state_dir in _discover_state_dirs(hermes_home, profile_filter):
        harvest = state_dir / "harvest"
        if harvest.is_dir() and (harvest / "predictions.jsonl").exists():
            out.append((label, harvest))
    return out


def _shape_ceiling_actions(state_dirs: list[tuple[str, Path]], python: str) -> list[dict]:
    """Run shape-ceiling.py --dry-run against each profile's live state.
    Surfaces cache-on promote/demote candidates."""
    actions: list[dict] = []
    for label, sdir in state_dirs:
        try:
            result = subprocess.run(
                [python, str(PLUGIN_DIR / "scripts" / "shape-ceiling.py"),
                 "--state-dir", str(sdir),
                 "--dry-run"],
                capture_output=True, text=True, check=False,
            )
            # Re-parse: shape-ceiling.py emits human-readable text, but we
            # can re-invoke its inner machinery via JSON-mode if available.
            # For now: parse the stdout structure. Format we emit:
            #   === scope:platform  (sessions_considered=N) ===
            #     Promote: ...
            #     Demote: ...
            current_scope = ""
            in_promote = False
            in_demote = False
            for line in result.stdout.splitlines():
                line = line.rstrip()
                if line.startswith("=== "):
                    current_scope = line.lstrip("= ").split(" (")[0].strip()
                    in_promote = in_demote = False
                elif line.lstrip().startswith("Promote:"):
                    in_promote = True
                    in_demote = False
                elif line.lstrip().startswith("Demote:"):
                    in_demote = True
                    in_promote = False
                elif (in_promote or in_demote) and line.lstrip().startswith("+ "):
                    # Promote row: "    + tool_name      sessions=N  calls=M  evidence=..."
                    parts = line.lstrip("+ ").split()
                    if parts:
                        actions.append({
                            "kind": "shape_promote" if in_promote else "shape_demote",
                            "label": label,
                            "scope": current_scope,
                            "tool": parts[0],
                            "raw": line.strip(),
                        })
                elif (in_promote or in_demote) and line.lstrip().startswith("- "):
                    parts = line.lstrip("- ").split()
                    if parts:
                        actions.append({
                            "kind": "shape_demote",
                            "label": label,
                            "scope": current_scope,
                            "tool": parts[0],
                            "raw": line.strip(),
                        })
        except Exception as exc:
            print(f"  warning: shape-ceiling failed on {label}: {exc}", file=sys.stderr)
    return actions


def _harvest_actions(harvest_dirs: list[tuple[str, Path]], python: str) -> list[dict]:
    """Run analyzer in JSON mode against each harvest dir; collect tool-
    promotion + trigger-keyword candidates ranked by impact."""
    actions: list[dict] = []
    for label, hdir in harvest_dirs:
        try:
            result = subprocess.run(
                [python, str(PLUGIN_DIR / "analyze.py"),
                 "--state-dir", str(hdir),
                 "--format", "json",
                 "--no-report",
                 "--suggest-trigger-keywords",
                 "--suggest-dampeners",
                 "--dampener-min-support", "3",
                 "--dampener-min-precision", "0.7"],
                capture_output=True, text=True, check=True,
            )
            payload = json.loads(result.stdout)
            for rec in payload.get("recommendations", []):
                if rec.get("kind") == "harvest_tool_promotion" and rec.get("action") in (
                    "promote_always_on", "broaden_trigger_recall"
                ):
                    actions.append({
                        "kind": "harvest_promotion",
                        "label": label,
                        "scope": rec["scope"],
                        "action": rec["action"],
                        "item": rec["item"],
                        "net": rec["metrics"]["net_savings_tokens"],
                        "cuts": rec["metrics"]["harvest_was_cut"],
                    })
            for row in payload.get("trigger_keyword_candidates", []):
                if not row.get("candidates"):
                    continue
                top = row["candidates"][0]
                actions.append({
                    "kind": "harvest_keyword",
                    "label": label,
                    "scope": row["scope"],
                    "tool": row["tool"],
                    "target_trigger": row["target_trigger"],
                    "action": row["action"],
                    "pattern": top["pattern"],
                    "cuts": row["cut_count"],
                    "precision": top["precision"],
                })
        except subprocess.CalledProcessError as exc:
            print(f"  warning: analyzer failed on {label}: {exc.stderr[:200]}",
                  file=sys.stderr)
        except json.JSONDecodeError as exc:
            print(f"  warning: analyzer output not parseable for {label}: {exc}",
                  file=sys.stderr)
    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="restrict to one profile")
    parser.add_argument("--window-days", type=int, default=None,
        help="harvest window for cache-off path (sessions modified within last N days)")
    parser.add_argument("--skip-shape", action="store_true",
        help="skip the cache-on shape-ceiling path")
    parser.add_argument("--skip-harvest", action="store_true",
        help="skip the cache-off harvest path")
    parser.add_argument("--quiet", action="store_true",
        help="only print top actions, suppress per-phase status output")
    parser.add_argument("--hermes-home", type=Path,
        default=Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes"))
    parser.add_argument("--python",
        default=_default_python(),
        help="Python interpreter for child scripts (default: HERMES_PYTHON or this interpreter)")
    args = parser.parse_args()

    def info(msg: str) -> None:
        if not args.quiet:
            print(msg)

    info("=" * 64)
    info("  tool-belt bootstrap — mode-aware warm start")
    info("=" * 64)

    state_dirs = _discover_state_dirs(args.hermes_home, args.profile)
    if not state_dirs:
        print("\nNo tool-belt state directories found under", args.hermes_home, file=sys.stderr)
        return 1

    # Cache-on path — shape-ceiling against live telemetry
    shape_actions: list[dict] = []
    if not args.skip_shape:
        info(f"\n[cache-on] Running shape-ceiling across {len(state_dirs)} profile(s)...")
        shape_actions = _shape_ceiling_actions(state_dirs, args.python)
        if not shape_actions:
            info("  No cache-on candidates yet (need ≥2 sessions per scope with expand_tools evidence).")

    # Cache-off path — harvest + analyzer
    harvest_actions: list[dict] = []
    if not args.skip_harvest:
        info("\n[cache-off] Harvesting session history...")
        harvest_cmd = [args.python, str(PLUGIN_DIR / "scripts" / "harvest-replay.py")]
        if args.profile:
            harvest_cmd += ["--profile", args.profile]
        if args.window_days is not None:
            harvest_cmd += ["--window-days", str(args.window_days)]
        if args.hermes_home:
            harvest_cmd += ["--hermes-home", str(args.hermes_home)]
        try:
            subprocess.run(harvest_cmd, capture_output=args.quiet, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"\n  warning: harvest-replay failed (rc={exc.returncode})", file=sys.stderr)
            if exc.stderr:
                print(exc.stderr[-500:], file=sys.stderr)

        harvest_dirs = _discover_harvest_dirs(args.hermes_home, args.profile)
        if harvest_dirs:
            info(f"  Analyzing harvest across {len(harvest_dirs)} profile(s)...")
            harvest_actions = _harvest_actions(harvest_dirs, args.python)
        else:
            info("  No harvest output (no sessions/*.jsonl, or window too narrow).")

    # ─── Unified TOP ACTIONS ────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  TOP ACTIONS")
    print("=" * 64)
    if not shape_actions and not harvest_actions:
        print("\n  No actionable recommendations surfaced today. Reasons:")
        print("    · cache-on scopes need ≥2 sessions of real expand_tools evidence")
        print("    · cache-off scopes need sessions/*.jsonl content to harvest")
        print("    · or your current policy already covers your usage")
        print("\n  Re-run after a few days of organic use for a fuller picture.")
        return 0

    # Shape-ceiling output (cache-on, organic) first — it's stronger signal
    shape_promotes = [a for a in shape_actions if a["kind"] == "shape_promote"]
    shape_demotes = [a for a in shape_actions if a["kind"] == "shape_demote"]
    if shape_promotes:
        print("\n  [cache-on] Promote into the frozen ceiling (from real expand_tools evidence):")
        for i, a in enumerate(shape_promotes, 1):
            print(f"    {i}. {a['scope']:<22} + {a['tool']}    ({a['raw'].split(a['tool'], 1)[-1].strip()})")
    if shape_demotes:
        print("\n  [cache-on] Demote from always-on (unused across recent sessions):")
        for i, a in enumerate(shape_demotes, 1):
            print(f"    {i}. {a['scope']:<22} − {a['tool']}    ({a['raw'].split(a['tool'], 1)[-1].strip()})")

    # Harvest output (cache-off, historical) second
    harvest_promotions = sorted([a for a in harvest_actions if a["kind"] == "harvest_promotion"],
                                key=lambda a: -a.get("net", 0))
    harvest_keywords = sorted([a for a in harvest_actions if a["kind"] == "harvest_keyword"],
                              key=lambda a: -a["cuts"])
    if harvest_promotions:
        print("\n  [cache-off / harvest] Tool promotions (edit policy.yaml or channels.<scope>.always_on_extra):")
        for i, p in enumerate(harvest_promotions, 1):
            tag = "PROMOTE" if p["action"] == "promote_always_on" else "BROADEN"
            print(f"    {i}. [{tag}] {p['scope']:<22} {p['item']:<20} "
                  f"cuts={p['cuts']:>4}  net={p['net']:+,} tok")
    if harvest_keywords:
        print("\n  [cache-off / harvest] Trigger keyword candidates (add to the named trigger's `keywords`):")
        for i, k in enumerate(harvest_keywords[:10], 1):
            print(f"    {i}. {k['scope']:<22} {k['target_trigger']:<14} ← \"{k['pattern']}\"")
            print(f"       (cuts {k['cuts']}, precision {k['precision']:.2f} — "
                  f"would have fired for {k['tool']})")

    print()
    print("  To activate cache-on recommendations: set `learned_mode: apply` for the scope in config.yaml,")
    print("  then re-run `shape-ceiling.py` (without --dry-run) to write learned.json.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
