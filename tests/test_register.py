"""Regression tests for plugin registration.

register() must declare its tool and all five hooks unconditionally so
Plugin Doctor can validate the declarations in plugin.yaml against what
the code actually registers. Functional disabling is handled by the
per-hook ``_CONFIG["enabled"]`` guards, not by skipping registration.

Guards a regression where ``register()`` returned early when
``_CONFIG["enabled"]`` was False, leaving Doctor to report the tool and
every hook as declared-but-not-registered.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

# Reuse conftest's plugin-loader bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401
import unittest

plugin = sys.modules["tool_belt_plugin"]

EXPECTED_HOOKS = {
    "pre_gateway_dispatch",
    "post_tool_call",
    "post_api_request",
    "on_session_end",
    "on_session_reset",
}


class _RecordingCtx:
    """Minimal ctx double that records tool and hook registrations."""

    def __init__(self) -> None:
        self.tools: list[str] = []
        self.hooks: list[str] = []

    def register_tool(self, *, name, toolset=None, schema=None, handler=None,
                      description=None):
        self.tools.append(name)

    def register_hook(self, hook_name, handler):
        self.hooks.append(hook_name)


class RegisterUnconditionalTest(unittest.TestCase):
    def test_registers_everything_when_disabled(self) -> None:
        ctx = _RecordingCtx()
        # enabled=False must NOT block registration. Patch config loading so
        # the test doesn't depend on the host's ~/.hermes/config.yaml, and
        # avoid touching the real detection cache / monkey-patches.
        with mock.patch.object(plugin, "_load_user_config", lambda: None), \
                mock.patch.object(plugin, "_load_detection_cache", lambda: None), \
                mock.patch.object(plugin, "_install_patches", lambda: True), \
                mock.patch.dict(plugin._CONFIG, {"enabled": False}, clear=False):
            plugin.register(ctx)

        self.assertIn("expand_tools", ctx.tools)
        self.assertEqual(EXPECTED_HOOKS, set(ctx.hooks))
        # No duplicate hook registrations.
        self.assertEqual(len(ctx.hooks), len(EXPECTED_HOOKS))


if __name__ == "__main__":
    unittest.main()
