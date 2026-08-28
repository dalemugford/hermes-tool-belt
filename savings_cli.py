"""``tool-belt savings`` — the public, read-only savings command.

Hermes native plugin manifests can't register shell subcommands, so the plugin
ships a repository-root ``tool-belt`` executable that dispatches here. This
module is import-only logic (no side effects at import) so tests can drive it
with a temporary ``HERMES_HOME`` and captured stdout.

Contract::

    tool-belt savings                 # every enabled agent + aggregate totals
    tool-belt savings --agent=default # one agent, all its platforms
    tool-belt savings --agent=alice   # unknown/disabled -> non-zero error
    tool-belt savings --json          # stable machine-readable schema, no prose
    tool-belt savings --since 2026-05-15

The command never writes. The Phase 8 launcher helper (:func:`ensure_launcher`)
is the only writing surface here and it writes exactly one file — the
``$HERMES_HOME/bin/tool-belt`` shim — and only when the caller confirms.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Callable

try:  # package context
    from . import savings as _savings
except ImportError:  # pragma: no cover - standalone load (direct exec)
    # Standalone execution (root `tool-belt` launcher) has no package parent.
    # Register the namespace so savings.py — and the sibling modules it pulls
    # in (predictor, presets, learned) — resolve their relative imports.
    import importlib
    import types

    _PLUGIN_DIR = Path(__file__).resolve().parent
    if "tool_belt_plugin" not in sys.modules:
        _pkg = types.ModuleType("tool_belt_plugin")
        _pkg.__path__ = [str(_PLUGIN_DIR)]
        sys.modules["tool_belt_plugin"] = _pkg
    _savings = importlib.import_module("tool_belt_plugin.savings")


# ─── Human rendering ──────────────────────────────────────────────────────────


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def render_text(report: _savings.SavingsReport) -> str:
    width = 74
    line = "─" * width
    out: list[str] = []
    out.append("═" * width)
    out.append("  Hermes Tool Belt — Savings")
    out.append("═" * width)
    scope_label = "all enabled agents" if report.generated_for == "all" else f"agent {report.generated_for!r}"
    out.append(f"  Reporting: {scope_label}")
    out.append(f"  Cache mode (projection): {report.cache_mode}")
    out.append(f"  Token estimator: {report.token_estimator}")
    out.append(f"  Hermes home: {report.hermes_home}")
    out.append("")

    if not report.agents:
        out.append("  No enabled/discovered agent profiles with telemetry or sessions.")
        out.append("")
        return "\n".join(out)

    for a in report.agents:
        obs = a.observed
        proj = a.projected
        out.append(f"┌{line}┐")
        out.append(f"│  AGENT: {a.agent}".ljust(width + 1) + "│")
        platforms = ", ".join(a.platforms) if a.platforms else "(none discovered)"
        out.append(f"│    platforms: {platforms}".ljust(width + 1) + "│")
        out.append(f"├{line}┤")
        # OBSERVED
        out.append(f"│  OBSERVED (realized — provider usage authoritative)".ljust(width + 1) + "│")
        out.append(
            f"│    schema tokens saved: {_fmt_int(obs.realized_schema_token_reduction)}"
            f"  across {obs.n_predictions} turn(s), {obs.n_sessions} session(s)".ljust(width + 1) + "│"
        )
        out.append(
            f"│    expand_tools overhead: −{_fmt_int(obs.expansion_overhead)}"
            f"  ({obs.expansion_events} event(s))".ljust(width + 1) + "│"
        )
        out.append(f"│    net realized savings: {_fmt_int(obs.net_token_reduction)} tok".ljust(width + 1) + "│")
        out.append(f"│".ljust(width + 1) + "│")
        # PROJECTED
        out.append(f"│  PROJECTED (counterfactual — not yet organic)".ljust(width + 1) + "│")
        out.append(
            f"│    sessions/turns analyzed: {proj.sessions_analyzed}/{proj.user_turns_analyzed}"
            f"  · confidence: {proj.confidence}".ljust(width + 1) + "│"
        )
        out.append(f"│    gross schema reduction: {_fmt_int(proj.gross_schema_token_reduction)} tok".ljust(width + 1) + "│")
        out.append(
            f"│    est. expansion overhead: −{_fmt_int(proj.estimated_expansion_overhead)}"
            f"  ({proj.expansion_events} event(s))".ljust(width + 1) + "│"
        )
        out.append(f"│    net projected reduction: {_fmt_int(proj.net_token_reduction)} tok".ljust(width + 1) + "│")
        if proj.net_input_reduction_pct is not None:
            out.append(
                f"│    net input reduction: {proj.net_input_reduction_pct:.1f}%"
                f"  (denominator: {proj.denominator_source})".ljust(width + 1) + "│"
            )
        elif proj.schema_reduction_pct is not None:
            out.append(
                f"│    schema-only reduction: {proj.schema_reduction_pct:.1f}%"
                f"  (not the session-input %)".ljust(width + 1) + "│"
            )
        if proj.estimated_usd_savings is not None:
            out.append(
                f"│    est. USD savings: ${proj.estimated_usd_savings:.4f}"
                f"  ({proj.usd_coverage} coverage, rate {report_rate(proj)})".ljust(width + 1) + "│"
            )
        else:
            out.append(f"│    est. USD savings: n/a (no known variable-cost route)".ljust(width + 1) + "│")
        out.append(f"└{line}┘")
        out.append("")

    # Aggregate
    agg = report.to_json()["aggregate"]
    out.append("  AGGREGATE (cohorts labeled separately — never summed together)")
    out.append(
        f"    observed net:  {_fmt_int(agg['observed']['net_token_reduction'])} tok"
    )
    proj_usd = agg["projected"]["estimated_usd_savings"]
    usd_txt = f"${proj_usd:.4f} ({agg['projected']['usd_coverage']})" if proj_usd is not None else "n/a"
    out.append(
        f"    projected net: {_fmt_int(agg['projected']['net_token_reduction'])} tok"
        f"  · est. USD: {usd_txt}  · counterfactual"
    )
    out.append("")
    return "\n".join(out)


def report_rate(proj: _savings.ProjectedCohort) -> str:
    for m in proj.models:
        if m.cost_class == "known":
            return m.rate_basis or "n/a"
    return "n/a"


# ─── Phase 8 launcher helper ──────────────────────────────────────────────────

_LAUNCHER_TEMPLATE = """#!/usr/bin/env sh
# Hermes Tool Belt launcher — created by onboarding. Delegates to the plugin's
# repository-root `tool-belt` executable so `tool-belt savings` works from PATH.
export HERMES_PYTHON={python}
exec {executable} "$@"
"""


def user_local_bin() -> Path:
    """The user-level bin dir the Hermes installer guarantees is on PATH."""
    return Path.home() / ".local" / "bin"


def launcher_path(hermes_home: Path, *, user_home: Path | None = None) -> Path:
    """Where the ``tool-belt`` command launcher should live.

    Prefers ``~/.local/bin`` — the directory the Hermes installer itself
    guarantees is on PATH (it links ``hermes`` there and appends it to the
    shell profile). Falls back to ``$HERMES_HOME/bin`` only when
    ``~/.local/bin`` does not exist (headless/CI homes, custom installs);
    that directory is deliberately NOT on PATH by Hermes convention.
    """
    home = user_home or Path.home()
    local_bin = home / ".local" / "bin"
    if local_bin.is_dir():
        return local_bin / "tool-belt"
    return Path(hermes_home) / "bin" / "tool-belt"


def path_guidance(hermes_home: Path) -> str:
    target = launcher_path(hermes_home)
    return (
        f"`tool-belt` is not on your PATH yet. Open a new terminal, or add its"
        f" directory now:\n"
        f'  export PATH="{target.parent}:$PATH"\n'
        f"Until then, run it by full path:\n"
        f"  {target} savings"
    )


def ensure_launcher(
    hermes_home: Path,
    repo_executable: Path,
    *,
    confirm: Callable[[str], bool],
    python: str | None = None,
    out: Callable[[str], None] = print,
    user_home: Path | None = None,
) -> bool:
    """Idempotently offer to install the ``tool-belt`` command launcher.

    Prefers ``~/.local/bin`` (on PATH for standard Hermes installs); falls
    back to ``$HERMES_HOME/bin`` when ``~/.local/bin`` does not exist. Never
    writes silently: ``confirm`` must return True. Returns True if the
    launcher exists (already present, or created now). Prints PATH guidance
    when the launcher's directory is not on PATH. Intended for onboarding —
    this function is the *only* writing surface in the CLI module.
    """
    target = launcher_path(hermes_home, user_home=user_home)
    if target.exists():
        out(f"Launcher already present: {target}")
        _maybe_path_note(hermes_home, out)
        return True
    if not confirm(f"Create launcher at {target}? [y/N] "):
        out("Skipped launcher creation.")
        out(path_guidance(hermes_home))
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    content = _LAUNCHER_TEMPLATE.format(
        python=shlex.quote(python or sys.executable),
        executable=shlex.quote(str(repo_executable)),
    )
    target.write_text(content, encoding="utf-8")
    target.chmod(0o755)
    out(f"Created launcher: {target}")
    _maybe_path_note(hermes_home, out)
    return True


def _maybe_path_note(hermes_home: Path, out: Callable[[str], None]) -> None:
    bin_dir = str(launcher_path(hermes_home).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir not in path_entries:
        out(path_guidance(hermes_home))


# ─── argparse entry ───────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tool-belt savings",
        description="Read-only token-savings report for enabled Hermes agents.",
    )
    parser.add_argument(
        "--agent", default=None,
        help="report a single enabled agent across all its platforms",
    )
    parser.add_argument(
        "--since", default=None,
        help="only count telemetry/sessions since this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--cache-mode", default="on", choices=["on", "off"],
        help="projection cache model: cache-on freeze (default) or cache-off per-turn",
    )
    parser.add_argument("--json", action="store_true", help="emit stable machine-readable JSON")
    parser.add_argument(
        "--hermes-home", type=Path, default=None,
        help="override HERMES_HOME (defaults to $HERMES_HOME or ~/.hermes)",
    )
    return parser


def run(argv: list[str] | None = None, *, out: Callable[[str], None] = None) -> int:
    """Execute the ``savings`` subcommand. Returns a process exit code."""
    args = build_parser().parse_args(argv or [])
    emit = out or (lambda s: print(s))
    try:
        report = _savings.compute(
            agent=args.agent,
            hermes_home=args.hermes_home,
            since=args.since,
            cache_mode=args.cache_mode,
        )
    except _savings.UnknownAgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        emit(json.dumps(report.to_json(), indent=2, sort_keys=True))
    else:
        emit(render_text(report))
    return 0


def _run_configure(argv: list[str]) -> int:
    """Delegate to scripts/configure.py's main() without importing it eagerly."""
    import importlib.util

    script = Path(__file__).resolve().parent / "scripts" / "configure.py"
    spec = importlib.util.spec_from_file_location("tool_belt_configure", script)
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules *before* exec: dataclasses resolves field types
    # via sys.modules[cls.__module__], which is None until registration.
    sys.modules["tool_belt_configure"] = module
    spec.loader.exec_module(module)
    return module.main(argv)


def main(argv: list[str] | None = None) -> int:
    """Top-level ``tool-belt`` dispatch: currently only ``savings``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(
            "usage: tool-belt <command> [options]\n"
            "\n"
            "commands:\n"
            "  savings     read-only token-savings report\n"
            "  configure   interactive onboarding (shape/recommend tool loadouts)"
        )
        return 0 if argv else 1
    command, rest = argv[0], argv[1:]
    if command == "savings":
        return run(rest)
    if command == "configure":
        return _run_configure(rest)
    print(f"error: unknown command {command!r} (known: savings, configure)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
