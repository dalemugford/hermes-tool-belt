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
    if report.since:
        out.append(f"  Window: since {report.since}")

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
        out.append(
            f"  NET TOKENS SAVED: {_fmt_int(total_net)}"
            f"  measured across {total_sessions} session(s) of real traffic."
        )
        per_session = total_net // total_sessions if total_sessions else 0
        if per_session > 0:
            out.append(
                f"  Agent conversations use ≈{_fmt_int(per_session)} fewer"
                " tokens with Tool Belt."
            )
        # Illustrative dollar equivalents at public input list prices. These
        # are what the saved tokens would have billed on a metered API route —
        # rates from PRICE_TABLE so the figure always carries its rate basis.
        examples = [
            ("OpenAI API", "gpt-5.5"),
            ("Anthropic API", "claude-sonnet-4-6"),
        ]
        priced = [
            (label, model, _savings.PRICE_TABLE[model]["input"])
            for label, model in examples
            if model in _savings.PRICE_TABLE
        ]
        if total_net > 0 and priced:
            out.append("")
            out.append(
                f"  Estimated value at API list prices"
                f" ({_savings.PRICE_TABLE_RATE_BASIS}, input rate):"
            )
            for label, model, rate in priced:
                usd = total_net / 1_000_000 * rate
                out.append(f"    {label:<16} ≈ ${usd:,.2f}  ({model})")
        # Annualized pace: net saved / measured wall-clock span, projected to
        # 12 months. Needs a week of history to say anything defensible.
        first = min((a.observed.first_ts for a in measured
                     if a.observed.first_ts > 0), default=0.0)
        last = max((a.observed.last_ts for a in measured), default=0.0)
        span_days = (last - first) / 86400.0
        if total_net > 0 and span_days >= 7.0:
            yearly = int(total_net * 365.0 / span_days)
            usd_txt = ""
            if priced:
                lo = min(yearly / 1_000_000 * rate for _, _, rate in priced)
                hi = max(yearly / 1_000_000 * rate for _, _, rate in priced)
                usd_txt = f" (≈${lo:,.0f}–${hi:,.0f} at the rates above)"
            out.append("")
            out.append(
                f"  At this pace ({span_days:.0f} days measured), 12 months of"
                f" Tool Belt saves"
            )
            out.append(f"  ≈{_fmt_int(yearly)} tokens{usd_txt}.")
        out.append("")
        out.append("  PER AGENT")
        for a in measured:
            obs = a.observed
            name = a.display_name or a.agent
            out.append(
                f"    {name:<12} {_fmt_int(obs.net_token_reduction):>12} tok"
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
            f"  {a.display_name or a.agent}: no measured traffic yet — "
            "estimate from session history:"
        )
        _render_projection(out, a.projected, indent="    ")

    if report.token_estimator == "chars-div-4":
        out.append("")
        out.append(
            "  Note: counts are ~4-chars-per-token estimates. Install tiktoken"
            " into the"
        )
        out.append(
            "  Hermes environment (pip install tiktoken) for exact token counts."
        )
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
    shell profile). Falls back to ``$HERMES_HOME/bin`` when ``~/.local/bin``
    does not exist (headless/CI homes); that directory is deliberately NOT
    on PATH by Hermes convention.

    Sandbox containment: when ``hermes_home`` is NOT the user's default
    install home (``~/.hermes``) — a test fixture, a staging copy, a custom
    ``HERMES_HOME`` — every write stays inside it. The interactivity audit
    found that a ``HERMES_HOME``-scoped run could still read (and offer to
    overwrite) the real operator's ``~/.local/bin/tool-belt``; a command
    whose every other write is scoped to the Hermes home must not be the
    one exception.
    """
    home = user_home or Path.home()
    try:
        is_default_home = Path(hermes_home).resolve() == (home / ".hermes").resolve()
    except OSError:
        is_default_home = False
    if is_default_home:
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
        if not confirm(f"Refresh launcher at {target} to {desired_exec}?"):
            out("Skipped launcher refresh.")
            out(path_guidance(hermes_home))
            return False
    elif not confirm(f"Create launcher at {target}?"):
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


def _run_configure(argv: list[str], prog: str | None = None) -> int:
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
    return module.main(argv, prog=prog)


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    """Top-level ``tool-belt`` dispatch: dispatches ``savings`` and ``configure``.

    ``prog`` is the command form the user actually typed (e.g. ``tool-belt``
    or ``hermes tool-belt``) so downstream guidance echoes a runnable command
    instead of a repo-relative script path. When None it is inferred from
    ``sys.argv[0]`` (the launcher installs under the name ``tool-belt``).
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if prog is None and Path(sys.argv[0] or "").name == "tool-belt":
        prog = "tool-belt"
    usage = (
        "usage: tool-belt <command> [options]\n"
        "\n"
        "commands:\n"
        "  savings     read-only token-savings report\n"
        "  configure   interactive onboarding (shape/recommend tool loadouts)"
    )
    if argv and argv[0] in ("-h", "--help"):
        print(usage)
        return 0
    if not argv:
        # Help is not an error; a missing command is — and must not print
        # byte-identical text under a different exit code.
        print(f"error: no command given\n{usage}", file=sys.stderr)
        return 2
    command, rest = argv[0], argv[1:]
    if command == "savings":
        return run(rest)
    if command == "configure":
        return _run_configure(rest, prog=prog)
    print(f"error: unknown command {command!r} (known: savings, configure)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
