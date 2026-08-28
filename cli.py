"""``hermes tool-belt ...`` — the official Hermes plugin CLI registration.

Hermes lets a plugin contribute a top-level terminal subcommand by calling
``ctx.register_cli_command(name=..., help=..., setup_fn=..., handler_fn=...)``
from its ``register(ctx)`` (see ``hermes_cli/plugins.py`` and the argparse
wiring in ``hermes_cli/main.py``). ``setup_fn`` receives the plugin's own
argparse subparser; ``handler_fn`` is installed via ``set_defaults(func=...)``
and is called with the parsed ``Namespace``. Its ``int`` return value becomes
the process exit code (non-zero ⇒ ``sys.exit(rc)``).

Two properties matter here:

* **Zero drift.** This module defines *no* flags of its own. It captures the
  words after ``savings`` / ``configure`` verbatim and hands them to
  :func:`savings_cli.main` — the exact entry point the repository-root
  ``tool-belt`` launcher execs. ``hermes tool-belt savings --json`` and
  ``tool-belt savings --json`` therefore run the same parser and the same code.
  Pass-through is achieved by building the two verb subparsers with a
  non-printable ``prefix_chars`` and ``add_help=False``, so argparse treats
  ``--json`` / ``--help`` as ordinary words instead of trying (and failing) to
  interpret them here.
* **Cheap setup.** :func:`register_cli` runs during argparse construction for
  every ``hermes`` invocation that reaches plugin discovery, so it imports
  nothing beyond ``argparse``. ``savings_cli`` (and the yaml/telemetry stack
  behind it) is imported only inside :func:`tool_belt_command`.
"""

from __future__ import annotations

import argparse
import sys

#: Pass-through subparsers disable option parsing by declaring a prefix char no
#: shell argument can contain, so every following token — ``--json``, ``-h``,
#: ``--`` — is classified as a plain positional and forwarded untouched.
_PASSTHROUGH_PREFIX = "\x00"

_VERBS: tuple[tuple[str, str], ...] = (
    ("savings", "read-only token-savings report (see `hermes tool-belt savings --help`)"),
    ("configure", "interactive onboarding (see `hermes tool-belt configure --help`)"),
)


def register_cli(parser: argparse.ArgumentParser) -> None:
    """Build the ``hermes tool-belt`` argparse tree. Must stay import-cheap."""
    subs = parser.add_subparsers(
        dest="tool_belt_action",
        metavar="{savings,configure}",
    )
    for verb, help_text in _VERBS:
        sub = subs.add_parser(
            verb,
            help=help_text,
            add_help=False,
            prefix_chars=_PASSTHROUGH_PREFIX,
        )
        sub.add_argument(
            "tool_belt_args",
            nargs="*",
            metavar="ARGS",
            help="passed through verbatim to the `tool-belt` CLI",
        )


def _load_savings_cli():
    """Import the shared CLI module. Lazy — never at argparse-setup time."""
    try:
        from . import savings_cli  # type: ignore[no-any-return]

        return savings_cli
    except ImportError:
        # Standalone/unpackaged load: fall back to a path import. savings_cli
        # itself handles a missing package parent (it registers the
        # ``tool_belt_plugin`` namespace before importing its siblings).
        import importlib.util
        from pathlib import Path

        script = Path(__file__).resolve().parent / "savings_cli.py"
        spec = importlib.util.spec_from_file_location("tool_belt_savings_cli", script)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules["tool_belt_savings_cli"] = module
        spec.loader.exec_module(module)
        return module


def tool_belt_command(args: argparse.Namespace) -> int:
    """Dispatch ``hermes tool-belt ...`` through ``savings_cli.main``."""
    action = getattr(args, "tool_belt_action", None)
    rest = list(getattr(args, "tool_belt_args", None) or [])
    argv = ([action, *rest] if action else [])

    try:
        savings_cli = _load_savings_cli()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"error: tool-belt CLI is unavailable: {exc}", file=sys.stderr)
        return 1

    try:
        rc = savings_cli.main(argv)
    except KeyboardInterrupt:
        print("", file=sys.stderr)
        return 130
    except SystemExit as exc:  # a nested parser may exit rather than return
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(str(code), file=sys.stderr)
        return 1
    return rc if isinstance(rc, int) else 0
