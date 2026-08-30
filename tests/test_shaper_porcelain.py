"""Contract tests for the shaper's machine-readable output and the two
consumers that used to screen-scrape its prose.

What is pinned here:

  1. ``shape-ceiling.py --json`` emits a versioned porcelain document whose
     top-level ``wrote_learned_state`` is the authoritative answer to "did
     this run rewrite learned.json".
  2. ``bootstrap.py`` consumes that document, never the human report — so
     rewording the report cannot change what bootstrap surfaces.
  3. Neither script ever tells an operator to hand-edit a config file.
  4. The operator scripts exit loudly when PyYAML is missing instead of
     degrading to a partial policy read.

Everything runs against a temporary HERMES_HOME / state dir; the live
installed plugin state is never read or written, and the shaper is never
run un-dry against it.
"""
from __future__ import annotations

import importlib
import importlib.util
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
PLUGIN_DIR = HERE.parent
sys.path.insert(0, str(PLUGIN_DIR.parent))
sys.path.insert(0, str(PLUGIN_DIR))
sys.path.insert(0, str(HERE))
import conftest  # noqa: F401,E402 — registers tool_belt_plugin

analyze = importlib.import_module("tool_belt_plugin.analyze")

SHAPER_PATH = PLUGIN_DIR / "scripts" / "shape-ceiling.py"


def _load_script(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        module_name, PLUGIN_DIR / "scripts" / filename
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


shape_ceiling = _load_script("tool_belt_shape_ceiling_test", "shape-ceiling.py")
bootstrap = _load_script("tool_belt_bootstrap_test", "bootstrap.py")
harvest_replay = _load_script("tool_belt_harvest_test", "harvest-replay.py")


def seed_state_dir(state_dir: Path, *, scope: str = "agent-a:telegram", sessions: int = 3) -> None:
    """Write telemetry that yields exactly one promote candidate for ``scope``."""
    state_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    preds, calls = [], []
    for i in range(sessions):
        pid = f"pred-{i}"
        preds.append({
            "ts": now + i,
            "schema_version": 2,
            "scope": scope,
            "session_id": "key",
            "hermes_session_id": f"sess-{i}",
            "prediction_id": pid,
            "always_carry_tools": ["read_file"],
            "carry_tools": ["read_file"],
            "expand_only_tools": ["grep_files"],
            "active_tools": ["read_file"],
            "ceiling_tools": ["read_file", "grep_files"],
        })
        calls.append({
            "ts": now + i,
            "schema_version": 2,
            "prediction_id": pid,
            "tool_name": "grep_files",
            "activated_by_expansion": True,
        })
    (state_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8")
    (state_dir / "tool_calls.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in calls), encoding="utf-8")


def run_shaper(state_dir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SHAPER_PATH), "--state-dir", str(state_dir), *args],
        capture_output=True, text=True, check=False,
    )


class PorcelainDocumentTests(unittest.TestCase):
    """The JSON schema itself — the contract every consumer branches on."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.state_dir = self.tmp / "state" / "tool-belt"
        seed_state_dir(self.state_dir)

    def test_dry_run_document_shape(self):
        result = run_shaper(self.state_dir, "--dry-run", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        doc = json.loads(result.stdout)

        self.assertEqual(doc["schema"], "tool-belt/shape-ceiling")
        self.assertEqual(doc["version"], shape_ceiling.PORCELAIN_VERSION)
        for key in ("generated_at", "state_dir", "learned_path", "dry_run",
                    "changed", "wrote_learned_state", "thresholds", "scopes"):
            self.assertIn(key, doc)
        self.assertTrue(doc["dry_run"])
        self.assertTrue(doc["changed"])
        # The stable marker: a dry run never claims to have written.
        self.assertIs(doc["wrote_learned_state"], False)
        self.assertFalse((self.state_dir / "learned.json").exists())

        self.assertEqual(len(doc["scopes"]), 1)
        scope_doc = doc["scopes"][0]
        self.assertEqual(scope_doc["scope"], "agent-a:telegram")
        self.assertEqual(scope_doc["sessions_considered"], 3)
        self.assertEqual(
            scope_doc["promote"],
            [{"tool": "grep_files", "sessions": 3, "calls": 3, "evidence": "expansion"}],
        )
        self.assertEqual(scope_doc["demote"], [])
        self.assertTrue(scope_doc["demote_skipped_insufficient_sessions"])
        for key in ("session_window", "promote_min_sessions", "promote_min_calls",
                    "demote_min_sessions_no_use"):
            self.assertIsInstance(doc["thresholds"][key], int)

    def test_wrote_learned_state_true_only_when_written(self):
        first = json.loads(run_shaper(self.state_dir, "--json").stdout)
        self.assertIs(first["wrote_learned_state"], True)
        self.assertTrue(first["changed"])
        self.assertTrue((self.state_dir / "learned.json").exists())

        # Second identical run: nothing changed, so nothing was written.
        second = json.loads(run_shaper(self.state_dir, "--json").stdout)
        self.assertIs(second["wrote_learned_state"], False)
        self.assertIs(second["changed"], False)

    def test_json_file_writes_the_same_document(self):
        out = self.tmp / "nested" / "shape.json"
        result = run_shaper(self.state_dir, "--dry-run", "--json-file", str(out))
        self.assertEqual(result.returncode, 0, result.stderr)
        doc = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(doc["schema"], "tool-belt/shape-ceiling")
        self.assertIs(doc["wrote_learned_state"], False)
        # Prose still goes to stdout when only --json-file was requested.
        self.assertIn("=== agent-a:telegram", result.stdout)

    def test_json_stdout_is_only_the_document(self):
        result = run_shaper(self.state_dir, "--dry-run", "--json")
        json.loads(result.stdout)  # would raise if prose leaked into stdout
        self.assertIn("=== agent-a:telegram", result.stderr)

    def test_empty_state_still_emits_a_parseable_document(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        result = run_shaper(empty, "--dry-run", "--json")
        self.assertEqual(result.returncode, 1)
        doc = json.loads(result.stdout)
        self.assertEqual(doc["error"], "no_predictions")
        self.assertIs(doc["wrote_learned_state"], False)
        self.assertEqual(doc["scopes"], [])


class BootstrapConsumesPorcelainTests(unittest.TestCase):
    """bootstrap reads the JSON document, never the human report."""

    PORCELAIN = {
        "schema": "tool-belt/shape-ceiling",
        "version": 1,
        "dry_run": True,
        "changed": True,
        "wrote_learned_state": False,
        "scopes": [{
            "scope": "agent-a:telegram",
            "sessions_considered": 22,
            "promote": [{"tool": "grep_files", "sessions": 4, "calls": 9,
                         "evidence": "expansion"}],
            "demote": [{"tool": "stale_tool", "sessions_without_use": 22,
                        "evidence": "carry_unused"}],
            "demote_skipped_insufficient_sessions": False,
        }],
    }

    def _run_with_stdout(self, stdout: str):
        captured: list[list[str]] = []

        def fake_run(argv, **kwargs):
            captured.append(list(argv))
            return mock.Mock(returncode=0, stdout=stdout, stderr="")

        with mock.patch.object(bootstrap.subprocess, "run", fake_run):
            actions = bootstrap._shape_ceiling_actions(
                [("default", Path("/nonexistent/state"))], sys.executable)
        return actions, captured

    def test_consumes_json_and_ignores_prose_wording(self):
        # The document is the whole stdout — exactly what --json produces.
        actions, captured = self._run_with_stdout(json.dumps(self.PORCELAIN))
        self.assertIn("--json", captured[0])
        self.assertIn("--dry-run", captured[0])  # bootstrap only reports

        kinds = {a["kind"] for a in actions}
        self.assertEqual(kinds, {"shape_promote", "shape_demote"})
        promote = next(a for a in actions if a["kind"] == "shape_promote")
        self.assertEqual(promote["tool"], "grep_files")
        self.assertEqual(promote["scope"], "agent-a:telegram")
        self.assertIn("sessions=4", promote["detail"])
        self.assertIn("calls=9", promote["detail"])
        demote = next(a for a in actions if a["kind"] == "shape_demote")
        self.assertEqual(demote["tool"], "stale_tool")
        self.assertIn("sessions_without_use=22", demote["detail"])

    def test_reworded_human_report_alone_yields_nothing(self):
        """The old contract, restated as a negative.

        A run whose stdout is only prose — in any wording — must produce no
        actions, because prose is not a contract. The previous screen-scraper
        would have mined this text (or silently lost candidates the moment
        the wording changed).
        """
        reworded = textwrap.dedent("""
            ### agent-a:telegram  (22 sessions examined)
              Recommended for carrying:
                * grep_files    sessions=4  calls=9
        """)
        actions, _ = self._run_with_stdout(reworded)
        self.assertEqual(actions, [])

    def test_old_human_layout_is_no_longer_parsed(self):
        """The exact pre-porcelain layout must now yield nothing.

        Proof that the screen-scraper is gone: this is byte-for-byte what the
        old parser mined, and bootstrap no longer sees a single action in it.
        """
        legacy = (
            "\n=== agent-a:telegram  (sessions_considered=22) ===\n"
            "  Promote:\n"
            "    + grep_files                     sessions= 4  calls=  9  evidence=expansion\n"
            "  Demote:\n"
            "    - stale_tool                     sessions_without_use=22  evidence=carry_unused\n"
        )
        actions, _ = self._run_with_stdout(legacy)
        self.assertEqual(actions, [])

    def test_unparseable_stdout_warns_and_yields_nothing(self):
        actions, _ = self._run_with_stdout("{not json")
        self.assertEqual(actions, [])


class ActivationGuidanceTests(unittest.TestCase):
    """Printed guidance must never point at hand-editing a config file."""

    FORBIDDEN = "config.yaml"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_shaper_write_guidance(self):
        state_dir = self.tmp / "state" / "tool-belt"
        seed_state_dir(state_dir)
        result = run_shaper(state_dir)  # writes into the temp dir only
        combined = result.stdout + result.stderr
        self.assertNotIn(self.FORBIDDEN, combined)
        self.assertNotIn("``", combined)  # no raw RST rendered to a terminal
        self.assertIn("configure.py", combined)
        self.assertIn(
            "hermes config set plugins.tool-belt.channels.", combined)

    def test_bootstrap_top_actions_guidance(self):
        hermes_home = self.tmp / "home"
        seed_state_dir(hermes_home / "state" / "tool-belt")
        porcelain = json.dumps(BootstrapConsumesPorcelainTests.PORCELAIN)

        def fake_run(argv, **kwargs):
            return mock.Mock(returncode=0, stdout=porcelain, stderr="")

        argv = ["bootstrap.py", "--hermes-home", str(hermes_home), "--skip-harvest"]
        buf = io.StringIO()
        with mock.patch.object(bootstrap.subprocess, "run", fake_run):
            with mock.patch.object(sys, "argv", argv):
                with redirect_stdout(buf):
                    rc = bootstrap.main()
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("TOP ACTIONS", out)
        self.assertIn("grep_files", out)
        self.assertNotIn(self.FORBIDDEN, out)
        self.assertIn("configure.py", out)
        self.assertIn(
            "hermes config set plugins.tool-belt.channels.", out)


class MissingPyYAMLTests(unittest.TestCase):
    """Operator scripts exit loudly rather than reading a partial policy."""

    def _no_yaml(self):
        return mock.patch.dict(sys.modules, {"yaml": None})

    def _assert_loud_exit(self, fn, *args, **kwargs):
        with self._no_yaml():
            with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
                with self.assertRaises(SystemExit) as ctx:
                    fn(*args, **kwargs)
        self.assertEqual(ctx.exception.code, 2)
        message = err.getvalue()
        self.assertIn("PyYAML is required", message)
        self.assertIn("pip install pyyaml", message)
        return message

    def test_shaper_defaults_exit(self):
        self._assert_loud_exit(shape_ceiling.load_shape_ceiling_defaults)

    def test_analyzer_always_carry_exits(self):
        self._assert_loud_exit(analyze._load_preset_always_carry, PLUGIN_DIR)

    def test_analyzer_excludes_exit(self):
        self._assert_loud_exit(analyze._load_preset_excludes, PLUGIN_DIR)

    def test_analyzer_triggers_exit(self):
        self._assert_loud_exit(analyze._load_preset_triggers, PLUGIN_DIR)

    def test_harvest_config_load_exits(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "config.yaml").write_text("plugins:\n  tool-belt:\n    enabled: true\n",
                                         encoding="utf-8")
        self._assert_loud_exit(harvest_replay._load_plugin_config, tmp)

    def test_runtime_presets_still_fail_open(self):
        """Deliberate asymmetry: the gateway must not die on a missing parser."""
        presets = importlib.import_module("tool_belt_plugin.presets")
        with self._no_yaml():
            preset = presets.load_base_policy()
        self.assertTrue(preset.no_narrowing)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
