"""Unit tests for the central v1→v2 telemetry normalizer (``logger_io``).

The centralized normalizer maps historical (v1) rows forward so every
downstream consumer reads the same canonical v2 row shape — and emits ONLY
that canonical surface. These tests pin the normalizer against
the stream shapes it must survive: malformed, sparse, complete-v1, already-v2,
and mixed-version streams — for both prediction rows and tool-call rows.
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

    def test_sparse_v1_row_does_not_infer_residency(self):
        out = logger_io.normalize_prediction_row(
            {"prediction_id": "p", "allowed_tools": ["clarify"]}
        )
        self.assertFalse(out["residency_inferred"])
        self.assertIsNone(out["residency"])
        # Canonical fields are always present, even on a sparse row.
        self.assertEqual(out["active_tools"], ["clarify"])
        self.assertEqual(out["schema_version"], logger_io.SCHEMA_VERSION)

    def test_complete_v1_row_maps_to_v2_and_infers_residency(self):
        out = logger_io.normalize_prediction_row({
            "prediction_id": "p",
            "ceiling_tools": ["clarify", "read_file", "web_extract"],
            "always_on_tools": ["clarify", "read_file"],
            "allowed_tools": ["clarify", "read_file"],
            "cut_tools": ["web_extract"],
        })
        self.assertTrue(out["residency_inferred"])
        # clarify is in the permanent always_carry baseline; read_file is carry.
        self.assertIn("clarify", out["always_carry_tools"])
        self.assertIn("read_file", out["carry_tools"])
        self.assertEqual(out["expand_only_tools"], ["web_extract"])
        self.assertEqual(out["active_tools"], ["clarify", "read_file"])
        # v1 residency mapping keeps class A empty (no immutable split existed).
        self.assertEqual(out["residency"]["always_carry"], [])
        # v1 spellings are consumed, not re-emitted: canonical fields only.
        self.assertNotIn("cut_tools", out)
        self.assertNotIn("always_on_tools", out)
        self.assertNotIn("allowed_tools", out)

    def test_v1_unknown_kept_folds_into_residents_but_mcp_never_does(self):
        out = logger_io.normalize_prediction_row({
            "ceiling_tools": ["clarify", "novel_tool"],
            "always_on_tools": ["clarify"],
            "allowed_tools": ["clarify"],
            "cut_tools": [],
            "unknown_kept_tools": ["novel_tool", "mcp__github__create_issue"],
        })
        self.assertIn("novel_tool", out["carry_tools"])
        # MCP pass-through can never become residency/shaping evidence.
        self.assertNotIn("mcp__github__create_issue", out["carry_tools"])
        self.assertNotIn("mcp__github__create_issue", out["always_carry_tools"])
        # The retired field is dropped from the canonical row.
        self.assertNotIn("unknown_kept_tools", out)

    def test_already_v2_row_passes_through_canonically(self):
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
        self.assertTrue(out["residency_inferred"])
        # v2 residency reflects the authoritative A/C/X.
        self.assertEqual(out["residency"]["always_carry"], ["clarify"])
        self.assertEqual(out["residency"]["carry"], ["read_file"])
        self.assertEqual(out["residency"]["expand_only"], ["web_extract"])
        # No v1 alias is synthesized onto a v2 row.
        self.assertNotIn("always_on_tools", out)
        self.assertNotIn("allowed_tools", out)
        self.assertNotIn("cut_tools", out)
        # Non-mutating: the input dict is untouched.
        self.assertNotIn("always_on_tools", row)

    def test_mixed_version_stream_normalizes_uniformly(self):
        stream = [
            {"prediction_id": "v1", "ceiling_tools": ["clarify", "web_extract"],
             "always_on_tools": ["clarify"], "allowed_tools": ["clarify"],
             "cut_tools": ["web_extract"]},
            {"prediction_id": "v2", "schema_version": 2,
             "ceiling_tools": ["clarify", "web_extract"],
             "always_carry_tools": ["clarify"], "carry_tools": [],
             "active_tools": ["clarify"], "expand_only_tools": ["web_extract"]},
            {"prediction_id": "sparse", "tokens_saved": 5},
        ]
        rows = [logger_io.normalize_prediction_row(r) for r in stream]
        for r in rows:
            self.assertEqual(r["schema_version"], logger_io.SCHEMA_VERSION)
            for field in ("always_carry_tools", "carry_tools", "active_tools",
                          "expand_only_tools", "always_carry_count", "carry_count"):
                self.assertIn(field, r)
        self.assertTrue(rows[0]["residency_inferred"])
        self.assertTrue(rows[1]["residency_inferred"])
        self.assertFalse(rows[2]["residency_inferred"])


class ToolCallNormalizationTests(unittest.TestCase):
    def test_malformed_rows_never_raise(self):
        for bad in (None, 42, "x", []):
            out = logger_io.normalize_tool_call_row(bad)  # type: ignore[arg-type]
            self.assertEqual(out.get("schema_version"), logger_io.SCHEMA_VERSION)

    def test_v1_flags_map_to_v2_without_aliases(self):
        out = logger_io.normalize_tool_call_row({
            "tool_name": "terminal",
            "was_initially_available": False,
            "was_cut": True,
            "was_expanded": False,
            "expand_tools_used": True,
        })
        self.assertFalse(out["was_initially_active"])
        self.assertTrue(out["was_expand_only"])
        self.assertFalse(out["activated_by_expansion"])
        self.assertTrue(out["expansion_provided_access"])
        # v1 spellings are consumed, not re-emitted: canonical flags only.
        self.assertNotIn("was_initially_available", out)
        self.assertNotIn("was_cut", out)
        self.assertNotIn("was_expanded", out)
        self.assertNotIn("expand_tools_used", out)

    def test_already_v2_row_gains_no_legacy_aliases(self):
        out = logger_io.normalize_tool_call_row({
            "tool_name": "web_extract",
            "was_initially_active": True,
            "was_expand_only": False,
            "activation_source": "carry",
        })
        self.assertTrue(out["was_initially_active"])
        self.assertFalse(out["was_expand_only"])
        self.assertEqual(out["activation_source"], "carry")
        self.assertNotIn("was_initially_available", out)
        self.assertNotIn("was_cut", out)

    def test_counterfactual_row_gains_no_spurious_expansion_flag(self):
        # A harvest row deliberately omits expansion flags — normalization must
        # not fabricate a False that the promote filter would misread.
        out = logger_io.normalize_tool_call_row({
            "tool_name": "terminal",
            "was_initially_available": True,
            "was_cut": False,
        })
        self.assertNotIn("expansion_provided_access", out)
        self.assertNotIn("expand_tools_used", out)


class FallbackBaselineDriftTests(unittest.TestCase):
    def test_fallback_always_carry_matches_shipped_policy(self):
        """The normalizer's hardcoded fallback baseline must equal the
        shipped policy.yaml ``always_carry`` set — a silent divergence would
        mis-split every historical v1 row."""
        presets = importlib.import_module("tool_belt_plugin.presets")
        base = presets.load_preset_file(presets._POLICY_FILE)
        self.assertEqual(
            logger_io._FALLBACK_ALWAYS_CARRY,
            frozenset(base.always_carry),
        )


if __name__ == "__main__":
    unittest.main()
