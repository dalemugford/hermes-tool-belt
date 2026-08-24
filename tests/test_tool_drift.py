"""Tests for tool-name drift detection.

Two surfaces are covered:

  1. The runtime drift WARNING emitted from the wrapped ``_build_api_kwargs``
     when the live ceiling carries tools the preset doesn't name (kept on as
     "unknown"). The safe-default behaviour is unchanged — the tools stay ON —
     but the drift must now be loud instead of silent.

  2. The ``scripts/check-tool-drift.py`` diagnostic: it detects drift (exit 1)
     vs. a clean state (exit 0), and its ``--update`` flag appends the drifted
     tools to a review block at the end of policy.yaml.
"""

from __future__ import annotations

import contextvars
import importlib
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

if "tool_belt_plugin" not in sys.modules:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent.parent))
    sys.path.insert(0, str(here.parent))
    from tests import conftest  # noqa: F401 — side-effect: register package

plugin = importlib.import_module("tool_belt_plugin")

_SCRIPT_PATH = Path(plugin.__file__).resolve().parent / "scripts" / "check-tool-drift.py"


def _load_drift_script():
    spec = importlib.util.spec_from_file_location("check_tool_drift", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeAgent:
    provider = "anthropic"


def _run_wrapped(tools, *, allowed, known):
    """Invoke the wrapped _build_api_kwargs with a stubbed original.

    Returns (kept_tool_names, log_records) where log_records is the list of
    WARNING+ records emitted on the plugin logger.
    """
    def original(self, api_messages):
        return {"tools": list(tools), "model": "claude-test"}

    wrapped = plugin._wrap_build_api_kwargs(original)

    state = {
        "allowed_tool_names": list(allowed),
        "known_tool_names": set(known),
        "expansions": set(),
        "logged": False,
        "session_id": "",
    }
    cv_token = plugin._PREDICTION_CV.set(state)
    try:
        with mock.patch.dict(plugin._CONFIG, {"enabled": True}), \
                mock.patch.object(plugin, "_maybe_log_prediction", lambda *a, **k: None):
            result = wrapped(_FakeAgent(), api_messages=[])
    finally:
        plugin._PREDICTION_CV.reset(cv_token)
    kept = [plugin._tool_name(t) for t in result["tools"]]
    return kept, state


class DriftWarningTests(unittest.TestCase):
    def test_unknown_tool_triggers_warning_and_is_kept(self):
        tools = [{"name": "read_file"}, {"name": "brand_new_tool"}]
        with self.assertLogs("tool_belt_plugin", level="WARNING") as cm:
            kept, state = _run_wrapped(
                tools,
                allowed=["read_file"],
                known={"read_file"},  # brand_new_tool is NOT known → unknown
            )
        # Safe-default preserved: the unknown tool is still ON.
        self.assertIn("brand_new_tool", kept)
        self.assertIn("brand_new_tool", state["unknown_kept_tools"])
        # Drift surfaced loudly.
        joined = "\n".join(cm.output)
        self.assertIn("not named in preset", joined)
        self.assertIn("check-tool-drift.py", joined)

    def test_no_unknown_tools_emits_no_warning(self):
        tools = [{"name": "read_file"}, {"name": "write_file"}]
        with mock.patch.object(plugin.logger, "warning") as warn:
            kept, state = _run_wrapped(
                tools,
                allowed=["read_file"],
                known={"read_file", "write_file"},  # write_file cut, none unknown
            )
        self.assertEqual(state["unknown_kept_tools"], [])
        # write_file is known-but-not-allowed → cut, not kept.
        self.assertNotIn("write_file", kept)
        self.assertEqual(warn.call_count, 0)


class CheckToolDriftScriptTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load_drift_script()

    def _write_policy(self, tmp: Path, *, known_tools) -> Path:
        lines = ["name: test", "always_on:"]
        lines += [f"  - {t}" for t in known_tools]
        lines += ["always_off:", "  - memory", "triggers: []", ""]
        p = tmp / "policy.yaml"
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def _write_state(self, tmp: Path, ceiling) -> Path:
        import json
        sd = tmp / "state"
        sd.mkdir()
        (sd / "predictions.jsonl").write_text(
            json.dumps({"ceiling_tools": list(ceiling)}) + "\n", encoding="utf-8"
        )
        return sd

    def test_detects_drift_exit_1(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            policy = self._write_policy(tmp, known_tools=["read_file"])
            sd = self._write_state(tmp, ["read_file", "brand_new_tool"])
            code = self.mod.main(["--policy", str(policy), "--state-dir", str(sd)])
        self.assertEqual(code, 1)

    def test_clean_state_exit_0(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            policy = self._write_policy(tmp, known_tools=["read_file", "write_file"])
            sd = self._write_state(tmp, ["read_file", "write_file"])
            code = self.mod.main(["--policy", str(policy), "--state-dir", str(sd)])
        self.assertEqual(code, 0)

    def test_strips_mcp_prefix_when_matching(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            policy = self._write_policy(tmp, known_tools=["read_file"])
            # ceiling carries the mcp_-prefixed send-time name — should match.
            sd = self._write_state(tmp, ["mcp_read_file"])
            code = self.mod.main(["--policy", str(policy), "--state-dir", str(sd)])
        self.assertEqual(code, 0)

    def test_update_appends_review_block(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            policy = self._write_policy(tmp, known_tools=["read_file"])
            sd = self._write_state(tmp, ["read_file", "brand_new_tool"])
            code = self.mod.main(["--policy", str(policy), "--state-dir", str(sd), "--update"])
            self.assertEqual(code, 1)  # drift still reported
            text = policy.read_text(encoding="utf-8")
        self.assertIn(self.mod._AUTO_HEADER, text)
        self.assertIn("#   - brand_new_tool", text)

    def test_update_is_idempotent(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            policy = self._write_policy(tmp, known_tools=["read_file"])
            sd = self._write_state(tmp, ["read_file", "brand_new_tool"])
            self.mod.main(["--policy", str(policy), "--state-dir", str(sd), "--update"])
            self.mod.main(["--policy", str(policy), "--state-dir", str(sd), "--update"])
            text = policy.read_text(encoding="utf-8")
        # The drifted tool appears exactly once in the review block.
        self.assertEqual(text.count("#   - brand_new_tool"), 1)


class KnownSetIncludesAlwaysOffTests(unittest.TestCase):
    def test_always_off_tool_is_known_and_cut(self):
        presets = importlib.import_module("tool_belt_plugin.presets")
        preset = presets.Preset(
            name="t", always_on=["read_file"], triggers=[], always_off=["memory"],
        )
        known = plugin._all_known_tool_names(preset)
        self.assertIn("memory", known)
        # A known-but-not-allowed tool is cut (kept off), not kept as unknown.
        tools = [{"name": "read_file"}, {"name": "memory"}]
        kept, state = _run_wrapped(tools, allowed=["read_file"], known=known)
        self.assertNotIn("memory", kept)
        self.assertNotIn("memory", state["unknown_kept_tools"])


if __name__ == "__main__":
    unittest.main()
