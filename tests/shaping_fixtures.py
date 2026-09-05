"""Shared fixtures for the between-session shaper tests.

Canonical v2 telemetry rows built by hand, plus a ``_compute`` wrapper over
``shaping.compute_scope_recommendations`` with explicit thresholds. Kept in
one module so trimming any single test file cannot pull the fixtures out
from under its siblings.

Consumers: test_day_window, test_economic_demotion, test_schema_sizes,
test_shaper_merge.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent

if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
import conftest  # noqa: F401,E402 — registers the tool_belt_plugin package

shaper = importlib.import_module("tool_belt_plugin.shaping")


def _pred_row(scope, sid, pid, *, ceiling, always_carry, carry, active, ts=0.0):
    """A complete canonical v2 prediction row (residency reconstructible)."""
    expand_only = [t for t in ceiling if t not in set(always_carry) | set(carry)]
    return {
        "schema_version": 2,
        "scope": scope,
        "hermes_session_id": sid,
        "prediction_id": pid,
        "ts": ts,
        "ceiling_tools": list(ceiling),
        "always_carry_tools": list(always_carry),
        "carry_tools": list(carry),
        "active_tools": list(active),
        "expand_only_tools": expand_only,
    }


def _expansion_call(pid, tool):
    """A tool-call row that is direct expansion evidence (expand_only → carry)."""
    return {
        "schema_version": 2,
        "prediction_id": pid,
        "tool_name": tool,
        "was_initially_active": False,
        "was_expand_only": True,
        "activated_by_expansion": True,
        "expansion_provided_access": True,
        "activation_source": "expansion",
    }


def _trigger_call(pid, tool):
    """A tool-call row activated by a trigger — NOT expansion evidence."""
    return {
        "schema_version": 2,
        "prediction_id": pid,
        "tool_name": tool,
        "was_initially_active": True,
        "was_expand_only": True,
        "activation_source": "trigger",
    }


def _compute(scope, pred_rows, call_rows, **overrides):
    grouped = shaper.group_predictions_by_scope_session(pred_rows)
    calls = shaper.index_tool_calls_by_prediction(call_rows)
    kwargs = dict(
        window_days=7, promote_min_sessions=2, promote_min_calls=3,
        demote_min_sessions_no_use=20,
    )
    kwargs.update(overrides)
    return shaper.compute_scope_recommendations(
        scope=scope, sessions=grouped.get(scope, {}), calls_by_pred=calls, **kwargs
    )
