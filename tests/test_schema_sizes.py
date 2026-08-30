"""Per-tool schema size sidecar (economic-demotion Phase 1).

The shaper's token-denominated demotion test needs each tool's real schema
size. Live defs only exist in-process, so the hot path snapshots them to
``schema_sizes.json`` (debounced, atomic, merge-preserving); the offline
shaper reads the sidecar and falls back to ``DEFAULT_PER_TOOL_TOKENS``.
These tests fail on pre-fix code (the sidecar API does not exist there).
"""

import json
import tempfile
import unittest
from pathlib import Path

from tests.test_carrying_model import shaper  # noqa: F401  (package bootstrap)
import tool_belt_plugin
from tool_belt_plugin import logger_io


def _anthropic_def(name, desc="d"):
    return {"name": name, "description": desc,
            "input_schema": {"type": "object", "properties": {}}}


def _openai_def(name):
    return {"type": "function", "function": {
        "name": name, "description": "d",
        "parameters": {"type": "object", "properties": {}}}}


class SchemaSizesSidecar(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "schema_sizes.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_measured_per_tool_sizes_both_schema_shapes(self):
        tools = [_anthropic_def("terminal", "run shell commands " * 20),
                 _openai_def("clarify")]
        wrote = logger_io.update_schema_sizes(tools, path=self.path)
        self.assertTrue(wrote)
        doc = json.loads(self.path.read_text())
        self.assertEqual(doc["schema"], "tool-belt/schema-sizes")
        sizes = doc["tools"]
        self.assertIn("terminal", sizes)
        self.assertIn("clarify", sizes)
        self.assertGreater(sizes["terminal"], sizes["clarify"],
                           "a fatter schema measures larger — per-tool, not average")

    def test_merge_preserves_tools_absent_from_this_ceiling(self):
        logger_io.update_schema_sizes([_anthropic_def("terminal")], path=self.path, now=0.0)
        logger_io.update_schema_sizes([_anthropic_def("web_search")], path=self.path,
                                      now=logger_io.SCHEMA_SIZES_REFRESH_SECONDS + 1)
        sizes = logger_io.load_schema_sizes(self.path.parent)
        self.assertIn("terminal", sizes, "another scope's tools keep their last size")
        self.assertIn("web_search", sizes)

    def test_debounce_skips_within_refresh_window(self):
        self.assertTrue(logger_io.update_schema_sizes(
            [_anthropic_def("terminal")], path=self.path))
        self.assertFalse(logger_io.update_schema_sizes(
            [_anthropic_def("web_search")], path=self.path),
            "second write inside the refresh window is skipped")

    def test_load_missing_or_invalid_is_empty(self):
        self.assertEqual(logger_io.load_schema_sizes(self.path.parent), {})
        self.path.write_text("not json")
        self.assertEqual(logger_io.load_schema_sizes(self.path.parent), {})

    def test_fallback_constant_canonical_in_logger_io(self):
        self.assertEqual(logger_io.DEFAULT_PER_TOOL_TOKENS, 388)

    def test_hot_path_logs_snapshot_from_prediction(self):
        # _maybe_log_prediction must attempt the snapshot with the ceiling defs.
        calls = []
        orig = logger_io.update_schema_sizes
        orig_log = logger_io.log_prediction
        logger_io.update_schema_sizes = lambda tools, **kw: calls.append(list(tools)) or True
        logger_io.log_prediction = lambda record: None  # keep live telemetry untouched
        orig_cfg = dict(tool_belt_plugin._CONFIG)
        tool_belt_plugin._CONFIG["log"] = True  # other tests mutate module config
        try:
            tool_belt_plugin._maybe_log_prediction(
                {"logged": False, "prediction_id": "p1", "scope": "a:t"},
                [_anthropic_def("terminal")], [],
            )
        finally:
            logger_io.update_schema_sizes = orig
            logger_io.log_prediction = orig_log
            tool_belt_plugin._CONFIG.clear()
            tool_belt_plugin._CONFIG.update(orig_cfg)
        self.assertEqual(len(calls), 1, "prediction logging snapshots schema sizes")
        self.assertEqual(logger_io._sidecar_tool_name(calls[0][0]), "terminal")


if __name__ == "__main__":
    unittest.main()
