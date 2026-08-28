"""Portability coverage for root/named profiles and custom Hermes homes."""
from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PLUGIN_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = PLUGIN_DIR / "tests"
sys.path.insert(0, str(TESTS_DIR))
import conftest  # noqa: F401,E402 — registers tool_belt_plugin

plugin = sys.modules["tool_belt_plugin"]
analyze = importlib.import_module("tool_belt_plugin.analyze")
learned = importlib.import_module("tool_belt_plugin.learned")
logger_io = importlib.import_module("tool_belt_plugin.logger_io")


def _load_script(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        module_name, PLUGIN_DIR / "scripts" / filename
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


bootstrap = _load_script("tool_belt_bootstrap_portability", "bootstrap.py")
harvest = _load_script("tool_belt_harvest_portability", "harvest-replay.py")
cache_replay = _load_script("tool_belt_cache_replay_portability", "cache-freeze-replay.py")
savings = _load_script("tool_belt_savings_portability", "savings-report.py")
shape = _load_script("tool_belt_shape_portability", "shape-ceiling.py")


class ProfileDiscoveryPortabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.hermes_home = Path(self.tmp.name) / "custom hermes home"
        self.hermes_home.mkdir()

    def test_bootstrap_discovers_default_and_named_profile_with_spaces(self) -> None:
        root_state = self.hermes_home / "state" / "tool-belt"
        named_state = (
            self.hermes_home / "profiles" / "assistant-a" / "state" / "tool-belt"
        )
        reserved_state = (
            self.hermes_home / "profiles" / "default" / "state" / "tool-belt"
        )
        root_state.mkdir(parents=True)
        named_state.mkdir(parents=True)
        reserved_state.mkdir(parents=True)

        self.assertEqual(
            bootstrap._discover_state_dirs(self.hermes_home, None),
            [("default", root_state), ("assistant-a", named_state)],
        )
        self.assertEqual(
            bootstrap._discover_state_dirs(self.hermes_home, "default"),
            [("default", root_state)],
        )
        self.assertEqual(
            bootstrap._discover_state_dirs(self.hermes_home, "assistant-a"),
            [("assistant-a", named_state)],
        )

    def test_harvest_discovers_default_and_named_sessions(self) -> None:
        root_sessions = self.hermes_home / "sessions"
        named_home = self.hermes_home / "profiles" / "assistant-a"
        reserved_home = self.hermes_home / "profiles" / "default"
        root_sessions.mkdir(parents=True)
        (named_home / "sessions").mkdir(parents=True)
        (reserved_home / "sessions").mkdir(parents=True)

        self.assertEqual(
            harvest.discover_profiles(self.hermes_home),
            [("default", self.hermes_home), ("assistant-a", named_home)],
        )
        self.assertEqual(
            harvest._profile_agent_from_path(root_sessions / "root.jsonl"),
            "default",
        )
        self.assertEqual(
            harvest._profile_agent_from_path(named_home / "sessions" / "named.jsonl"),
            "assistant-a",
        )

    def test_absent_state_and_session_dirs_are_cleanly_ignored(self) -> None:
        self.assertEqual(bootstrap._discover_state_dirs(self.hermes_home, None), [])
        self.assertEqual(harvest.discover_profiles(self.hermes_home), [])

    def test_bootstrap_honors_explicit_python_override(self) -> None:
        with mock.patch.dict(
            os.environ, {"HERMES_PYTHON": "/opt/custom python/bin/python"}
        ):
            self.assertEqual(
                bootstrap._default_python(), "/opt/custom python/bin/python"
            )

    def test_bootstrap_defaults_to_invoking_interpreter(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(bootstrap._default_python(), sys.executable)


class StatePathPortabilityTests(unittest.TestCase):
    def test_all_state_helpers_honor_custom_hermes_home_with_spaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hermes_home = Path(tmp) / "custom hermes home"
            expected = hermes_home / "state" / "tool-belt"
            with mock.patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
                self.assertEqual(analyze.state_dir_from_env(), expected)
                self.assertEqual(learned.state_dir(), expected)
                self.assertEqual(logger_io._state_dir(), expected)
                self.assertEqual(cache_replay.default_state_dir(), expected)
                self.assertEqual(savings.default_state_dir(), expected)
                self.assertEqual(shape.default_state_dir(), expected)

    def test_daily_analysis_handles_spaced_home_and_missing_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hermes_home = Path(tmp) / "custom hermes home"
            (hermes_home / "profiles" / "default" / "state" / "tool-belt").mkdir(
                parents=True
            )
            env = {
                **os.environ,
                "HERMES_HOME": str(hermes_home),
                "HERMES_PYTHON": sys.executable,
            }
            result = subprocess.run(
                ["bash", str(PLUGIN_DIR / "scripts" / "daily-analysis.sh")],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = (
                hermes_home
                / "state"
                / "tool-belt"
                / "cron-logs"
                / "daily-summary.log"
            ).read_text()
            self.assertEqual(summary.count("[default]  no_telemetry"), 1)


if __name__ == "__main__":
    unittest.main()
