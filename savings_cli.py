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
    tool-belt configure               # interactive onboarding (separate module)

The ``savings`` command never writes. The launcher helper
(:func:`ensure_launcher`, used by ``tool-belt configure``) is the only writing
surface here and it writes exactly one file — the ``$HERMES_HOME/bin/tool-belt``
shim — and only when the caller confirms.
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


def _rate_basis_label(proj: _savings.ProjectedCohort) -> str:
    for m in proj.models:
        if m.cost_class == "known":
            return m.rate_basis or "n/a"
    return "n/a"


def _render_projection(out: list[str], proj: _savings.ProjectedCohort, indent: str) -> None:
    """The projection lines for one agent — shown only while measured data
    is insufficient (an estimate from replayed session history)."""
    out.append(
        f"{indent}estimated net savings: {_fmt_int(proj.net_token_reduction)} tok"
        f"  (from {proj.sessions_analyzed} past session(s), confidence: {proj.confidence})"
    )
    if proj.net_input_reduction_pct is not None:
        out.append(
            f"{indent}net input reduction: {proj.net_input_reduction_pct:.1f}%"
            f"  (denominator: {proj.denominator_source})"
        )
    elif proj.schema_reduction_pct is not None:
        out.append(
            f"{indent}schema-only reduction: {proj.schema_reduction_pct:.1f}%"
            "  (not the session-input %)"
        )
    if proj.estimated_usd_savings is not None:
        out.append(
            f"{indent}est. USD savings: ${proj.estimated_usd_savings:.4f}"
            f"  ({proj.usd_coverage} coverage, rate {_rate_basis_label(proj)})"
        )


def render_text(report: _savings.SavingsReport) -> str:
    """Lead with the one number users care about: net tokens actually saved.

    Measured savings (from real traffic) are the headline. An agent's
    projection appears only while it has no measured sessions yet — the two
    are never summed, and the projection block disappears once real data
    exists.
    """
    out: list[str] = []
    out.append("")
    out.append("  Hermes Tool Belt — Savings")
    out.append("  " + "─" * 40)

    if not report.agents:
        out.append("  No enabled/discovered agent profiles with telemetry or sessions.")
        out.append("")
        return "\n".join(out)

    measured = [a for a in report.agents if a.observed.n_sessions > 0]
    unmeasured = [a for a in report.agents if a.observed.n_sessions == 0]

    total_net = sum(a.observed.net_token_reduction for a in measured)
    total_sessions = sum(a.observed.n_sessions for a in measured)
    if measured:
        out.append("")
        out.append(f"  NET TOKENS SAVED: {_fmt_int(total_net)}")
        out.append(f"  measured across {total_sessions} session(s) of real traffic")
        out.append("")
        for a in measured:
            obs = a.observed
            out.append(
                f"    {a.agent:<12} {_fmt_int(obs.net_token_reduction):>12} tok"
                f"   {obs.n_sessions} session(s)"
            )
        out.append("")
        out.append(
            "  Calculation: Unsent tool-definition tokens (vs carrying all), "
            "minus expand_tools fetch overhead."
        )

    for a in unmeasured:
        out.append("")
        out.append(
            f"  {a.agent}: no measured traffic yet — estimate from session history:"
        )
        _render_projection(out, a.projected, indent="    ")

    scope_label = "all enabled agents" if report.generated_for == "all" else f"agent {report.generated_for!r}"
    out.append("")
    out.append(f"  ({scope_label} · estimator {report.token_estimator} · {report.hermes_home})")
    out.append("")
    return "\n".join(out)


# ─── Launcher helper (used by `tool-belt configure`) ──────────────────────────

_LAUNCHER_TEMPLATE = """#!/usr/bin/env sh
# Hermes Tool Belt launcher — created by onboarding. Delegates to the plugin's
# repository-root `tool-belt` executable so `tool-belt savings` works from PATH.
export HERMES_PYTHON={python}
exec {executable} "$@"
"""

#: Ownership marker — present in every launcher this module has ever written.
#: A file at the target without it belongs to someone else and is never touched.
_LAUNCHER_MARKER = "Hermes Tool Belt launcher"


def _launcher_exec_target(content: str) -> str | None:
    """The path a generated launcher ``exec``s, or None if unreadable."""
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped.startswith("exec "):
            continue
        try:
            parts = shlex.split(stripped[len("exec "):])
        except ValueError:
            return None
        return parts[0] if parts else None
    return None


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
    writes silently: ``confirm`` must return True. Returns True if a working
    launcher for *this* plugin exists (already present, or created now).
    Prints PATH guidance when the launcher's directory is not on PATH.
    Intended for onboarding — this function is the *only* writing surface in
    the CLI module.

    An existing file at the target is not taken on faith. The launcher bakes
    in an absolute ``exec`` path, so a shim left behind by a moved or replaced
    plugin directory keeps dispatching to a path that no longer exists (or to
    a different checkout). Three cases:

    * ours and current (``exec`` target exists and is ``repo_executable``) —
      nothing to do, still idempotent;
    * ours and stale — offer to refresh it through the same ``confirm`` gate;
    * not ours (no Tool Belt marker) — warn, leave it alone, claim nothing.
    """
    target = launcher_path(hermes_home, user_home=user_home)
    desired_exec = str(repo_executable)
    if target.exists():
        try:
            content = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            content = ""
        if _LAUNCHER_MARKER not in content:
            out(
                f"A file at {target} was not created by Tool Belt; leaving it "
                "untouched. Remove or rename it, then re-run configure."
            )
            out(path_guidance(hermes_home))
            return False
        existing = _launcher_exec_target(content)
        if existing == desired_exec and Path(desired_exec).exists():
            out(f"Launcher already present: {target}")
            _maybe_path_note(hermes_home, out)
            return True
        if not existing:
            detail = "it has no readable exec target"
        elif not Path(existing).exists():
            detail = f"its exec target no longer exists: {existing}"
        else:
            detail = f"it points at a different plugin: {existing}"
        out(f"Launcher at {target} is stale — {detail}")
        if not confirm(f"Refresh launcher at {target} to {desired_exec}? [y/N] "):
            out("Skipped launcher refresh.")
            out(path_guidance(hermes_home))
            return False
    elif not confirm(f"Create launcher at {target}? [y/N] "):
        out("Skipped launcher creation.")
        out(path_guidance(hermes_home))
        return False
    was_present = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    content = _LAUNCHER_TEMPLATE.format(
        python=shlex.quote(python or sys.executable),
        executable=shlex.quote(str(repo_executable)),
    )
    target.write_text(content, encoding="utf-8")
    target.chmod(0o755)
    out(f"{'Refreshed' if was_present else 'Created'} launcher: {target}")
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


def run(argv: list[str] | None = None, *, out: Callable[[str], None] | None = None) -> int:
    """Execute the ``savings`` subcommand. Returns a process exit code.

    ``out`` is the **stdout** sink only (the report body). Errors are always
    written straight to ``sys.stderr`` and are not routed through ``out``, so a
    caller capturing ``out`` must capture stderr separately to see them.
    """
    args = build_parser().parse_args(argv or [])
    emit = out or (lambda s: print(s))
    try:
        report = _savings.compute(
            agent=args.agent,
            hermes_home=args.hermes_home,
            since=args.since,
            cache_mode=args.cache_mode,
        )
    except (_savings.UnknownAgentError, _savings.InvalidSinceError) as exc:
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
    if spec is None or spec.loader is None:
        print(f"error: configure script not found at {script}", file=sys.stderr)
        return 2
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules *before* exec: dataclasses resolves field types
    # via sys.modules[cls.__module__], which is None until registration.
    sys.modules["tool_belt_configure"] = module
    spec.loader.exec_module(module)
    return module.main(argv)


def main(argv: list[str] | None = None) -> int:
    """Top-level ``tool-belt`` dispatch: dispatches ``savings`` and ``configure``."""
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
