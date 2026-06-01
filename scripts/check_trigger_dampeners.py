#!/usr/bin/env python3
"""Focused smoke checks for tool-belt trigger dampeners.

Loads policy.yaml and verifies that the high-noise trigger groups still
fire for direct action requests while discussion/meta phrasing is vetoed by
exclude_keywords.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]

# The plugin directory is named ``tool-belt`` (hyphen), which is not a
# valid Python module name. We construct a synthetic package so the modules
# inside can use their normal ``from .presets import …`` relative imports.
_PKG_NAME = "_tool_belt_smoke"
_pkg = types.ModuleType(_PKG_NAME)
_pkg.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
sys.modules[_PKG_NAME] = _pkg


def _load(submodule: str):
    full_name = f"{_PKG_NAME}.{submodule}"
    spec = importlib.util.spec_from_file_location(full_name, PLUGIN_DIR / f"{submodule}.py")
    assert spec and spec.loader, f"failed to build spec for {submodule}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


presets_mod = _load("presets")
learned_mod = _load("learned")

TriggerGroup = presets_mod.TriggerGroup
load_preset_file = presets_mod.load_preset_file
Preset = presets_mod.Preset


def trigger(name: str) -> TriggerGroup:
    preset = load_preset_file(PLUGIN_DIR / "policy.yaml")
    for group in preset.triggers:
        if group.name == name:
            return group
    raise AssertionError(f"trigger not found: {name}")


def assert_fires(group_name: str, message: str) -> None:
    group = trigger(group_name)
    assert group.matches(message, []), f"expected {group_name} to fire: {message!r}"


def assert_suppressed(group_name: str, message: str) -> None:
    group = trigger(group_name)
    assert not group.matches(message, []), f"expected {group_name} not to fire: {message!r}"
    assert group.is_excluded(message), f"expected {group_name} exclude to match: {message!r}"
    assert group.would_fire_positive(message, []), f"expected {group_name} positive signal to match: {message!r}"


def assert_not_fires(group_name: str, message: str) -> None:
    group = trigger(group_name)
    assert not group.matches(message, []), f"expected {group_name} not to fire: {message!r}"


def main() -> int:
    # file_write: direct edits still load write tools; planning/meta talk does not.
    assert_fires("file_write", "Save notes on this to a markdown file.")
    assert_fires("file_write", "Patch `presets.py` with that change.")
    assert_fires("file_write", "Update `/Users/macmini/.hermes/plugins/tool-belt/PLAN.md`.")
    assert_suppressed("file_write", "Should we save this as a file later?")
    assert_suppressed("file_write", "What do you think about the file-write trigger?")
    assert_suppressed("file_write", "Talk through the approach before editing anything.")
    assert_suppressed("file_write", "Don't patch it yet.")

    # delegation: explicit worker/spawn intent still fires; conceptual discussion does not.
    assert_fires("delegation", "Delegate this to Claude Code using your artifact wrapper.")
    assert_fires("delegation", "Spin up CC to review this plugin.")
    assert_fires("delegation", "Use subagents to research these in parallel.")
    assert_suppressed("delegation", "Should we use subagents for this eventually?")
    assert_not_fires("delegation", "We had ideas about delegation.")
    assert_not_fires("delegation", "This is a big project, but let's talk first.")

    # cronjob: schedule/reminder actions still fire; habits/explanations do not.
    assert_fires("cronjob", "Remind me every Friday at 9am to check this.")
    assert_fires("cronjob", "Schedule a weekly report.")
    assert_fires("cronjob", "Create a cron job to run this every morning.")
    assert_not_fires("cronjob", "I check this every day.")
    assert_not_fires("cronjob", "How does our cron setup work?")
    assert_suppressed("cronjob", "Should we schedule this reminder later?")

    # Learned-merge path: dampeners must survive learned state application.
    # Regression guard for the bug where apply_to_preset rebuilt TriggerGroup
    # without exclude_patterns, silently disabling every dampener in
    # learned_mode: auto / audit.
    _check_learned_merge_preserves_dampeners()

    print("trigger dampener checks passed")
    return 0


def _check_learned_merge_preserves_dampeners() -> None:
    """Apply a non-empty learned state and verify dampeners still veto."""
    preset = load_preset_file(PLUGIN_DIR / "policy.yaml")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Point HERMES_HOME at a temp dir so we don't touch real state.
        original_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = tmpdir
        try:
            state_dir = Path(tmpdir) / "state" / "tool-belt"
            state_dir.mkdir(parents=True, exist_ok=True)
            # Write a minimal learned.json that triggers the merge path (a
            # promotion forces apply_to_preset to take the "rebuild triggers"
            # branch).
            learned_payload = {
                "version": 1,
                "updated_at": "2026-05-11T00:00:00Z",
                "scopes": {
                    "bernard:telegram": {
                        "always_on": ["terminal"],
                    }
                },
                "global": {},
            }
            (state_dir / "learned.json").write_text(json.dumps(learned_payload), encoding="utf-8")
            learned_mod.load_state(force=True)

            plugin_config = {"learned_mode": "auto"}
            result = learned_mod.apply_to_preset(preset, plugin_config, "bernard:telegram")
            assert result.policy_source == "learned", (
                f"expected policy_source=learned, got {result.policy_source!r}"
            )
            merged = result.preset
            file_write = next((g for g in merged.triggers if g.name == "file_write"), None)
            assert file_write is not None, "file_write trigger missing after learned merge"
            assert file_write.exclude_patterns, (
                "regression: exclude_patterns dropped by apply_to_preset; "
                "dampeners would be silently disabled in learned_mode=auto/audit"
            )
            # And confirm the actual veto still works end-to-end.
            assert not file_write.matches("Should we save this as a file later?", []), (
                "file_write dampener stopped vetoing after learned merge"
            )
        finally:
            if original_home is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = original_home
            learned_mod.load_state(force=True)


if __name__ == "__main__":
    raise SystemExit(main())
