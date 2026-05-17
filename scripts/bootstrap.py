#!/usr/bin/env python3
"""First-install warm-start for the dynamic-tools plugin.

The plugin's value proposition is "install, activate, done — saves
tokens on tool overhead." That promise has a cold-start problem: on
day one, there's no telemetry, so the analyzer has nothing to
recommend, and the user sees no immediate value beyond the static
narrowing.

This script closes that gap. It runs harvest-replay against the
user's existing Hermes sessions, then runs the analyzer in
harvest-aware mode with both suggesters (dampener + trigger-keyword)
enabled. Output is a single concise summary of the top actions the
user should consider — meaningful recommendations from day one,
backed by their own real session history.

Designed to be invoked once after `hermes plugins install
dalemugford/dynamic-tools`. Idempotent: re-running just refreshes
the harvest from current session data without disrupting live
telemetry collection (harvest lives in a separate `harvest/` subdir).

Usage:
  python3 scripts/bootstrap.py
  python3 scripts/bootstrap.py --window-days 60  # restrict harvest window
  python3 scripts/bootstrap.py --profile sue     # one profile only
  python3 scripts/bootstrap.py --quiet           # only print top actions
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


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kwargs)


def _discover_harvest_dirs(hermes_home: Path, profile_filter: str | None) -> list[tuple[str, Path]]:
    """Return ``[(label, state_dir/harvest), ...]`` for every profile
    that successfully produced harvest output."""
    out: list[tuple[str, Path]] = []
    root_harvest = hermes_home / "state" / "dynamic-tools" / "harvest"
    if root_harvest.is_dir() and not profile_filter or profile_filter == "bernard":
        if (root_harvest / "predictions.jsonl").exists():
            out.append(("bernard", root_harvest))
    profiles_dir = hermes_home / "profiles"
    if profiles_dir.is_dir():
        for child in sorted(profiles_dir.iterdir()):
            if not child.is_dir():
                continue
            if profile_filter and child.name != profile_filter:
                continue
            p_harvest = child / "state" / "dynamic-tools" / "harvest"
            if p_harvest.is_dir() and (p_harvest / "predictions.jsonl").exists():
                out.append((child.name, p_harvest))
    return out


def _top_actions(harvest_dirs: list[tuple[str, Path]], python: str) -> list[dict]:
    """Run analyzer in JSON mode against each harvest dir; collect the
    most actionable items across all scopes, ranked by impact."""
    all_recs: list[dict] = []
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
                    all_recs.append({
                        "kind": "promotion",
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
                all_recs.append({
                    "kind": "keyword",
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
    return all_recs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="restrict to one profile")
    parser.add_argument("--window-days", type=int, default=None,
        help="only process sessions modified within the last N days")
    parser.add_argument("--quiet", action="store_true",
        help="only print top actions, suppress per-phase status output")
    parser.add_argument("--hermes-home", type=Path,
        default=Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes"))
    parser.add_argument("--python",
        default=os.environ.get("HERMES_PYTHON")
                or str(Path.home() / ".hermes/hermes-agent/venv/bin/python3")
                if (Path.home() / ".hermes/hermes-agent/venv/bin/python3").exists()
                else sys.executable)
    args = parser.parse_args()

    def info(msg: str) -> None:
        if not args.quiet:
            print(msg)

    info("=" * 64)
    info("  dynamic-tools bootstrap — first-install warm start")
    info("=" * 64)

    # Phase 1: harvest
    harvest_cmd = [args.python, str(PLUGIN_DIR / "scripts" / "harvest-replay.py")]
    if args.profile:
        harvest_cmd += ["--profile", args.profile]
    if args.window_days is not None:
        harvest_cmd += ["--window-days", str(args.window_days)]
    if args.hermes_home:
        harvest_cmd += ["--hermes-home", str(args.hermes_home)]
    info("\n[1/2] Replaying session history into synthetic telemetry...")
    try:
        result = subprocess.run(harvest_cmd, capture_output=args.quiet,
                                text=True, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"\nerror: harvest-replay failed (rc={exc.returncode})", file=sys.stderr)
        if exc.stderr:
            print(exc.stderr[-500:], file=sys.stderr)
        return 1

    # Phase 2: analyze + collect top actions
    harvest_dirs = _discover_harvest_dirs(args.hermes_home, args.profile)
    if not harvest_dirs:
        print("\nNo harvest output found. Likely causes:", file=sys.stderr)
        print("  - No sessions/*.jsonl files exist for this profile yet.", file=sys.stderr)
        print("  - --profile filter didn't match any profile directory.", file=sys.stderr)
        return 1

    info(f"\n[2/2] Analyzing harvest output across {len(harvest_dirs)} profile(s)...")
    actions = _top_actions(harvest_dirs, args.python)

    # Final summary — the user-facing deliverable
    print("\n" + "=" * 64)
    print("  TOP ACTIONS")
    print("=" * 64)
    if not actions:
        print("\n  No actionable recommendations surfaced. Either:")
        print("    - Your existing policy already covers everything your usage needs.")
        print("    - Sample size is too thin for confident recommendations yet.")
        print("\n  The plugin is now narrowing tool sets per the default policy.")
        print("  Re-run this script after a week of live use for an updated audit.")
        return 0

    # Rank: promotions first (sorted by net), then keyword suggestions
    promotions = sorted([a for a in actions if a["kind"] == "promotion"],
                        key=lambda a: -a.get("net", 0))
    keywords = sorted([a for a in actions if a["kind"] == "keyword"],
                      key=lambda a: -a["cuts"])

    if promotions:
        print("\n  Tool promotions (edit policy.yaml or channels.<scope>.always_on_extra):")
        for i, p in enumerate(promotions, 1):
            tag = "PROMOTE  " if p["action"] == "promote_always_on" else "BROADEN  "
            print(f"    {i}. [{tag}] {p['scope']:<22} {p['item']:<20} "
                  f"cuts={p['cuts']:>4}  net={p['net']:+,} tok")

    if keywords:
        print("\n  Trigger keyword candidates (add to the named trigger's `keywords` list):")
        for i, k in enumerate(keywords[:10], 1):
            print(f"    {i}. {k['scope']:<22} {k['target_trigger']:<14} ← \"{k['pattern']}\"")
            print(f"       (cuts {k['cuts']}, precision {k['precision']:.2f} — "
                  f"would have fired for {k['tool']})")

    print("\n  Full details:")
    for label, hdir in harvest_dirs:
        print(f"    {label}: {hdir}/")
    print("    (run `analyze.py --state-dir <harvest-dir>` for the full markdown report)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
