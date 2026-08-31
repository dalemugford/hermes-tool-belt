"""PyYAML availability guard for the operator scripts.

Every Hermes install ships PyYAML in its virtualenv, so a missing ``yaml``
module never means "this host lacks a dependency" — it means the script was
started with the wrong interpreter (a bare system ``python3`` instead of the
Hermes one). Silently degrading in that situation is worse than stopping:
the operator scripts read ``policy.yaml``, and a partial parse produces
recommendations computed against a policy that isn't the one in force.

So the operator scripts (``analyze.py``, ``scripts/shape-ceiling.py``,
``scripts/harvest-replay.py``) call :func:`require_yaml` and exit loudly.

This policy is deliberately **not** applied to the runtime plugin: the
gateway must never be taken down by a missing parser, so ``presets.py``
keeps its fail-open behavior (a policy it cannot load becomes a
no-narrowing preset and the model sees every tool).
"""

from __future__ import annotations

import sys
from types import ModuleType

#: Exit status used when PyYAML is missing. Distinct from the ``1`` the
#: scripts use for "no data / nothing to do" so a scheduler can tell an
#: environment fault from an empty run.
YAML_MISSING_EXIT_CODE = 2

MESSAGE = (
    "PyYAML is required but is not importable by this interpreter.\n"
    "  interpreter: {executable}\n"
    "Run this script through the Hermes interpreter — the `tool-belt` launcher\n"
    "resolves it automatically, or set HERMES_PYTHON — or install it with\n"
    "`pip install pyyaml`."
)


def require_yaml() -> ModuleType:
    """Return the ``yaml`` module, or exit with a clear message.

    Raises ``SystemExit(YAML_MISSING_EXIT_CODE)`` — never returns ``None``
    and never falls back to a hand-rolled parser.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised via injection
        print(
            "tool-belt: " + MESSAGE.format(executable=sys.executable),
            file=sys.stderr,
        )
        raise SystemExit(YAML_MISSING_EXIT_CODE) from exc
    return yaml
