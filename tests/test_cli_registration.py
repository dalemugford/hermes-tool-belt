"""Regression tests for the ``hermes tool-belt ...`` CLI registration.

The plugin registers a top-level Hermes subcommand from ``register(ctx)`` via
``ctx.register_cli_command(name=..., setup_fn=..., handler_fn=...)``. These
tests pin the contract Hermes actually implements (``hermes_cli/plugins.py``
``PluginContext.register_cli_command`` + the argparse wiring in
``hermes_cli/main.py``: ``setup_fn(subparser)`` then
``subparser.set_defaults(func=handler_fn)``, with the handler's int return
value used as the process exit code) and the two properties that keep it
honest: argparse-time cheapness, and zero flag drift from ``savings_cli``.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from unittest import mock

# Reuse conftest's plugin-loader bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401,E402
import unittest  # noqa: E402

plugin = sys.modules["tool_belt_plugin"]
cli = importlib.import_module("tool_belt_plugin.cli")

_SAVINGS_CLI_MODULE = "tool_belt_plugin.savings_cli"


class _RecordingCtx:
    """ctx double recording tool/hook/CLI registrations."""

    def __init__(self) -> None:
        self.tools: list[str] = []
        self.hooks: list[str] = []
        self.cli_commands: list[dict] = []

    def register_tool(self, *, name, toolset=None, schema=None, handler=None,
                      description=None):
        self.tools.append(name)

    def register_hook(self, hook_name, handler):
        self.hooks.append(hook_name)

    def register_cli_command(self, *, name, help, setup_fn, handler_fn=None,
                             description=""):
        self.cli_commands.append({
            "name": name,
            "help": help,
            "setup_fn": setup_fn,
            "handler_fn": handler_fn,
            "description": description,
        })


class _LegacyCtx(_RecordingCtx):
    """A Hermes too old to support plugin CLI commands."""

    def __getattribute__(self, item):
        if item == "register_cli_command":
            raise AttributeError(item)
        return super().__getattribute__(item)


def _register(ctx) -> None:
    """Run register() without touching config, caches or monkey-patches."""
    with mock.patch.object(plugin, "_load_user_config", lambda: None), \
            mock.patch.object(plugin, "_load_detection_cache", lambda: None), \
            mock.patch.object(plugin, "_install_patches", lambda: True):
        plugin.register(ctx)


def _hermes_parser() -> argparse.ArgumentParser:
    """Rebuild the exact wiring hermes_cli/main.py performs for a plugin."""
    ctx = _RecordingCtx()
    _register(ctx)
    info = ctx.cli_commands[0]
    top = argparse.ArgumentParser(prog="hermes")
    subs = top.add_subparsers(dest="command")
    plugin_parser = subs.add_parser(
        info["name"],
        help=info["help"],
        description=info.get("description", ""),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    info["setup_fn"](plugin_parser)
    if info.get("handler_fn") is not None:
        plugin_parser.set_defaults(func=info["handler_fn"])
    return top


class RegistrationTest(unittest.TestCase):
    def test_registers_tool_belt_cli_command(self) -> None:
        ctx = _RecordingCtx()
        _register(ctx)

        self.assertEqual(1, len(ctx.cli_commands))
        info = ctx.cli_commands[0]
        self.assertEqual("tool-belt", info["name"])
        self.assertTrue(info["help"])
        self.assertIs(cli.register_cli, info["setup_fn"])
        self.assertIs(cli.tool_belt_command, info["handler_fn"])
        # The tool and hooks still register alongside it.
        self.assertIn("expand_tools", ctx.tools)
        self.assertEqual(5, len(ctx.hooks))

    def test_register_survives_ctx_without_cli_support(self) -> None:
        ctx = _LegacyCtx()
        _register(ctx)  # must not raise

        self.assertEqual([], ctx.cli_commands)
        self.assertIn("expand_tools", ctx.tools)
        self.assertEqual(5, len(ctx.hooks))

    def test_register_survives_failing_cli_registration(self) -> None:
        class _BoomCtx(_RecordingCtx):
            def register_cli_command(self, **kwargs):
                raise RuntimeError("nope")

        ctx = _BoomCtx()
        _register(ctx)  # fail-open

        self.assertIn("expand_tools", ctx.tools)
        self.assertEqual(5, len(ctx.hooks))


class SetupIsCheapTest(unittest.TestCase):
    def test_setup_fn_does_not_import_savings_cli(self) -> None:
        # setup_fn runs at argparse-construction time for every hermes
        # invocation that reaches plugin discovery — it must not drag in the
        # savings/yaml/telemetry stack.
        with mock.patch.dict(sys.modules):
            sys.modules.pop(_SAVINGS_CLI_MODULE, None)
            parser = argparse.ArgumentParser(prog="hermes tool-belt")
            cli.register_cli(parser)
            self.assertNotIn(_SAVINGS_CLI_MODULE, sys.modules)


class PassThroughTest(unittest.TestCase):
    """The whole flag surface is savings_cli's; nothing is redefined here."""

    def _dispatch(self, argv: list[str]) -> tuple[list[str] | None, int]:
        top = _hermes_parser()
        args = top.parse_args(argv)
        seen: dict[str, list[str]] = {}

        def _fake_main(argv_in):
            seen["argv"] = list(argv_in)
            return 0

        sc = importlib.import_module(_SAVINGS_CLI_MODULE)
        with mock.patch.object(sc, "main", _fake_main):
            rc = args.func(args)
        return seen.get("argv"), rc

    def test_savings_flags_pass_through(self) -> None:
        forwarded, rc = self._dispatch(
            ["tool-belt", "savings", "--json", "--agent", "default"]
        )
        self.assertEqual(["savings", "--json", "--agent", "default"], forwarded)
        self.assertEqual(0, rc)

    def test_configure_flags_pass_through(self) -> None:
        forwarded, _ = self._dispatch(["tool-belt", "configure", "--status"])
        self.assertEqual(["configure", "--status"], forwarded)

    def test_subcommand_help_is_forwarded_not_intercepted(self) -> None:
        forwarded, _ = self._dispatch(["tool-belt", "configure", "--help"])
        self.assertEqual(["configure", "--help"], forwarded)

    def test_bare_invocation_forwards_empty_argv(self) -> None:
        forwarded, _ = self._dispatch(["tool-belt"])
        self.assertEqual([], forwarded)

    def test_top_level_help_names_both_subcommands(self) -> None:
        top = _hermes_parser()
        # The `tool-belt` subparser action holds the plugin parser.
        sub_action = next(
            a for a in top._actions if isinstance(a, argparse._SubParsersAction)
        )
        text = sub_action.choices["tool-belt"].format_help()
        self.assertIn("savings", text)
        self.assertIn("configure", text)


class ExitStatusTest(unittest.TestCase):
    def _handler(self, main_impl):
        args = argparse.Namespace(tool_belt_action="savings", tool_belt_args=[])
        sc = importlib.import_module(_SAVINGS_CLI_MODULE)
        with mock.patch.object(sc, "main", main_impl):
            return cli.tool_belt_command(args)

    def test_return_code_propagates(self) -> None:
        self.assertEqual(2, self._handler(lambda argv: 2))

    def test_system_exit_code_propagates(self) -> None:
        def _boom(argv):
            raise SystemExit(3)

        self.assertEqual(3, self._handler(_boom))

    def test_system_exit_message_becomes_exit_1(self) -> None:
        def _boom(argv):
            raise SystemExit("bad news")

        self.assertEqual(1, self._handler(_boom))

    def test_system_exit_none_is_success(self) -> None:
        def _clean(argv):
            raise SystemExit(None)

        self.assertEqual(0, self._handler(_clean))

    def test_non_int_return_is_success(self) -> None:
        self.assertEqual(0, self._handler(lambda argv: None))


if __name__ == "__main__":
    unittest.main()
