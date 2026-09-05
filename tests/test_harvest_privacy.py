"""Privacy regression tests for scripts/harvest-replay.py.

The harvester reads real Hermes session JSONLs containing private user
messages. Its output must contain only the same redacted shape the live
writer produces: ``message_hash`` (sha1) + ``message_preview`` (80 char
truncation). Anything more is a privacy regression.

These tests construct a synthetic session with a known full-text message,
run the harvester's replay logic on it, and assert that no substring of
the original message longer than the preview window appears in any
derived row. They also assert that tool-call argument payloads are never
written to disk.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUGIN_DIR = HERE.parent
sys.path.insert(0, str(PLUGIN_DIR.parent))
sys.path.insert(0, str(PLUGIN_DIR))
sys.path.insert(0, str(PLUGIN_DIR / "tests"))
import conftest  # noqa: F401

presets_mod = importlib.import_module("tool_belt_plugin.presets")

# Load the harvest-replay script as a module so we can call its
# functions in-process. Lives under scripts/, not in the package.
# Register in sys.modules BEFORE exec_module — dataclass field resolution
# under Python 3.14 looks up the module by __module__ name during decorator
# execution, and a missing entry raises AttributeError.
import importlib.util
_harvest_spec = importlib.util.spec_from_file_location(
    "harvest_replay", PLUGIN_DIR / "scripts" / "harvest-replay.py"
)
harvest_replay = importlib.util.module_from_spec(_harvest_spec)
sys.modules["harvest_replay"] = harvest_replay
_harvest_spec.loader.exec_module(harvest_replay)


# A long, distinctive message we can later grep for. Anything beyond the
# preview window (80 chars) appearing in output is a leak.
SECRET_MESSAGE = (
    "PRIVACY_CANARY_BEGIN this is an extremely distinctive sentence the "
    "harvester must never write to its output files in full because it "
    "contains sensitive personal information about a user's medical history "
    "and a password ABC123XYZ789 PRIVACY_CANARY_END"
)

SECRET_TOOL_ARG = "SECRET_ARG_PAYLOAD_should_not_leak_into_output_files_at_all"


class HarvestPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__import__("tempfile").mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

        # Construct a minimal session JSONL with the canary message and a
        # tool call whose arguments contain another canary.
        session_file = self.tmp / "session.jsonl"
        rows = [
            {
                "role": "session_meta",
                "platform": "telegram",
                "model": "test-model",
                "tools": [
                    {"type": "function", "function": {
                        "name": "write_file", "description": "x",
                        "parameters": {"type": "object", "properties": {}},
                    }},
                    {"type": "function", "function": {
                        "name": "send_message", "description": "x",
                        "parameters": {"type": "object", "properties": {}},
                    }},
                ],
            },
            {"role": "user", "content": SECRET_MESSAGE, "timestamp": "2026-05-17T12:00:00"},
            {"role": "assistant", "content": "", "timestamp": "2026-05-17T12:00:01",
             "tool_calls": [
                 {"id": "call_1", "type": "function",
                  "function": {"name": "write_file",
                               "arguments": json.dumps({"path": "/tmp/x.txt",
                                                       "content": SECRET_TOOL_ARG})}},
             ]},
        ]
        session_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        # Parse + replay this single session through the harvester
        self.session = harvest_replay.parse_session(session_file)
        self.assertIsNotNone(self.session, "test setup: session must parse")

        plugin_config = {"enabled": True}
        preset = presets_mod.resolve_preset(plugin_config, channel="default:telegram")

        self.predictions: list[dict] = []
        self.tool_calls: list[dict] = []
        harvest_replay.replay_session(
            self.session, preset, self.predictions, self.tool_calls,
            harvest_run_ts=time.time(),
        )

    def _serialized_output(self) -> str:
        """All harvest-derived rows as a single string for substring search."""
        return (
            "\n".join(json.dumps(r) for r in self.predictions)
            + "\n"
            + "\n".join(json.dumps(r) for r in self.tool_calls)
        )

    def test_full_message_text_never_appears_in_output(self):
        """No 81+ char substring of the input message can appear in output.

        The documented redaction window is 80 chars, so the window here is
        exactly one char wider than the widest legitimate preview. A wider
        window would leave 81-99 char leaks invisible.
        """
        output = self._serialized_output()
        WINDOW = 81
        leaks = []
        for i in range(0, len(SECRET_MESSAGE) - WINDOW):
            chunk = SECRET_MESSAGE[i:i + WINDOW]
            if chunk in output:
                leaks.append(chunk[:60])
        self.assertFalse(leaks,
            f"input message leaked to output (sample: {leaks[:3]})")

    def test_canary_password_never_leaks(self):
        """A distinctive secret token from the middle of the message must
        not appear anywhere in derived output."""
        output = self._serialized_output()
        # ABC123XYZ789 lives at message-position ~165, beyond the 80-char
        # preview cutoff. If it shows up, redaction broke.
        self.assertNotIn("ABC123XYZ789", output,
            "distinctive password token leaked to output")

    def test_tool_arguments_never_leak(self):
        """Tool call argument payloads from the historical session must
        never be written to harvest output."""
        output = self._serialized_output()
        self.assertNotIn(SECRET_TOOL_ARG, output,
            "tool argument payload leaked to output")
        self.assertNotIn("/tmp/x.txt", output,
            "tool argument path leaked to output")

    def test_preview_truncated_to_80_chars(self):
        """Every derived row's preview is within the documented 80-char
        window (+1 for the ellipsis the truncator appends)."""
        self.assertTrue(self.predictions, "test setup: replay produced no predictions")
        checked = 0
        for row in list(self.predictions) + list(self.tool_calls):
            if "message_preview" not in row:
                continue
            preview = row["message_preview"] or ""
            checked += 1
            self.assertLessEqual(len(preview), 81,
                f"preview exceeded 80 chars: {len(preview)} chars in {row!r}")
        self.assertTrue(checked, "test setup: no derived row carried a preview")

    def test_message_hash_is_deterministic_but_not_reversible(self):
        """The hash field is exactly the sha1-16 digest of the message.

        Computed here from the algorithm independently of logger_io, so a
        change that lets raw text through the hash field fails this test.
        """
        self.assertTrue(self.predictions)
        hashed = self.predictions[0].get("message_hash", "")
        expected = hashlib.sha1(
            SECRET_MESSAGE.encode("utf-8", errors="replace")
        ).hexdigest()[:16]
        self.assertEqual(hashed, expected,
            "message_hash is not the deterministic sha1-16 digest")
        # Belt and braces: no recognisable fragment of the secret survives.
        for i in range(0, len(SECRET_MESSAGE) - 8):
            self.assertNotIn(SECRET_MESSAGE[i:i + 8], hashed,
                "hash field contained raw message text")


if __name__ == "__main__":
    unittest.main()
