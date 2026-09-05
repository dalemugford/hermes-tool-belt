"""Unit tests for the central telemetry normalizer (``logger_io``).

The centralized normalizer is the one surface every downstream consumer reads
through, and it emits ONLY the canonical field set. These tests pin it against
the stream shapes it must survive: malformed, sparse, and complete rows — for
both prediction rows and tool-call rows.
"""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401,E402

logger_io = importlib.import_module("tool_belt_plugin.logger_io")


class PredictionNormalizationTests(unittest.TestCase):
    def test_malformed_rows_never_raise(self):
        for bad in (None, 123, "nope", [], object()):
            out = logger_io.normalize_prediction_row(bad)  # type: ignore[arg-type]
            self.assertEqual(out.get("schema_version"), logger_io.SCHEMA_VERSION)
            self.assertFalse(out.get("residency_inferred"))

    def test_sparse_row_does_not_infer_residency(self):
        out = logger_io.normalize_prediction_row(
            {"prediction_id": "p", "active_tools": ["clarify"]}
        )
        self.assertFalse(out["residency_inferred"])
        self.assertIsNone(out["residency"])
        # Canonical fields are always present, even on a sparse row.
        self.assertEqual(out["active_tools"], ["clarify"])
        self.assertEqual(out["schema_version"], logger_io.SCHEMA_VERSION)
        for field in ("always_carry_tools", "carry_tools", "expand_only_tools",
                      "always_carry_count", "carry_count",
                      "trigger_activated_tools"):
            self.assertIn(field, out)

    def test_complete_row_infers_the_authoritative_partition(self):
        row = {
            "schema_version": 2,
            "ceiling_tools": ["clarify", "read_file", "web_extract"],
            "always_carry_tools": ["clarify"],
            "carry_tools": ["read_file"],
            "active_tools": ["clarify", "read_file"],
            "expand_only_tools": ["web_extract"],
        }
        out = logger_io.normalize_prediction_row(row)
        self.assertEqual(out["always_carry_tools"], ["clarify"])
        self.assertEqual(out["carry_tools"], ["read_file"])
        self.assertEqual(out["active_tools"], ["clarify", "read_file"])
        self.assertEqual(out["expand_only_tools"], ["web_extract"])
        self.assertTrue(out["residency_inferred"])
        self.assertEqual(out["residency"]["always_carry"], ["clarify"])
        self.assertEqual(out["residency"]["carry"], ["read_file"])
        self.assertEqual(out["residency"]["expand_only"], ["web_extract"])
        # Non-mutating: the input dict gains nothing.
        self.assertNotIn("residency", row)

    def test_a_tool_outside_the_ceiling_never_enters_the_partition(self):
        out = logger_io.normalize_prediction_row({
            "schema_version": 2,
            "ceiling_tools": ["clarify", "web_extract"],
            "always_carry_tools": ["clarify"],
            "carry_tools": ["mcp__github__create_issue"],
            "active_tools": ["clarify", "mcp__github__create_issue"],
            "expand_only_tools": ["web_extract"],
        })
        self.assertEqual(out["residency"]["carry"], [])
        self.assertEqual(out["residency"]["expand_only"], ["web_extract"])

    def test_counts_default_from_membership_but_explicit_counts_win(self):
        out = logger_io.normalize_prediction_row({
            "schema_version": 2,
            "always_carry_tools": ["clarify", "skills_list"],
            "carry_tools": ["read_file"],
        })
        self.assertEqual(out["always_carry_count"], 2)
        self.assertEqual(out["carry_count"], 1)
        out = logger_io.normalize_prediction_row({
            "schema_version": 2,
            "always_carry_tools": ["clarify"],
            "carry_tools": [],
            "always_carry_count": 7,
            "carry_count": 9,
        })
        self.assertEqual(out["always_carry_count"], 7)
        self.assertEqual(out["carry_count"], 9)


class ToolCallNormalizationTests(unittest.TestCase):
    def test_malformed_rows_never_raise(self):
        for bad in (None, 42, "x", []):
            out = logger_io.normalize_tool_call_row(bad)  # type: ignore[arg-type]
            self.assertEqual(out.get("schema_version"), logger_io.SCHEMA_VERSION)

    def test_flags_are_coerced_to_bool(self):
        out = logger_io.normalize_tool_call_row({
            "tool_name": "terminal",
            "was_initially_active": 0,
            "was_expand_only": 1,
            "activated_by_expansion": False,
            "expansion_provided_access": True,
            "activation_source": "expansion",
        })
        self.assertIs(out["was_initially_active"], False)
        self.assertIs(out["was_expand_only"], True)
        self.assertIs(out["activated_by_expansion"], False)
        self.assertIs(out["expansion_provided_access"], True)
        self.assertEqual(out["activation_source"], "expansion")

    def test_missing_activation_source_becomes_blank(self):
        out = logger_io.normalize_tool_call_row({
            "tool_name": "web_extract",
            "was_initially_active": True,
            "was_expand_only": False,
        })
        self.assertEqual(out["activation_source"], "")

    def test_counterfactual_row_gains_no_spurious_expansion_flag(self):
        # A harvest row deliberately omits expansion flags — normalization must
        # not fabricate a False that the promote filter would misread.
        out = logger_io.normalize_tool_call_row({
            "tool_name": "terminal",
            "was_initially_active": True,
            "was_expand_only": False,
        })
        self.assertNotIn("expansion_provided_access", out)
        self.assertNotIn("activated_by_expansion", out)


if __name__ == "__main__":
    unittest.main()
