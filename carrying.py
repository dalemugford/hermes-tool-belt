"""Carrying-model partition resolver — Tool Belt 1.0.

Given Hermes' *enabled built-in ceiling* ``E`` and the plugin's carrying
loadout, this module finalizes the three-way residency partition and the
per-message active set. It is the single authority for the locked algebra::

    A = always_carry ∩ E                     # immutable residents (class A)
    C = ((E − A) − demoted) ∪ ((carry ∩ E) − A)   # adaptive residents (class C)
    X = E − (A ∪ C)                          # expand_only         (class X)
    T = triggered ∩ X                        # trigger-activated expand_only
    R = expanded  ∩ X                        # explicitly expanded expand_only
    active = A ∪ C ∪ T ∪ R

This is the *full-start* contract: with no learned
state (``demoted`` empty), every enabled tool is carried — ``active == E`` —
and only evidence-driven demotion moves a tool into ``expand_only``. Unknown
enabled built-ins are CARRIED until demoted. ``carry`` is the explicit
promotion loadout (learned promotions / un-demotions); it wins over
``demoted`` for a tool named in both (carry-wins, matching ``learned.py``'s
reconciliation).

Invariants:

  * A, C, X are a genuine three-way partition of E (precedence
    always_carry > carry > demoted). The partition itself enforces the
    always_carry precedence: C is computed minus A.
  * X ⊆ demoted: a tool only becomes expand_only through a demotion signal
    (or by being outside E entirely, in which case it is simply absent).
  * No selection source (policy, learning, triggers, or explicit expansion)
    can add a tool absent from ``E``. Disabled/absent tools never re-enter.
  * Any internal failure fails **open** — the returned model's active set is
    the whole enabled ceiling (no narrowing).

Demotion/promotion *reconciliation* still lives upstream, in
``learned.apply_to_preset`` — the always_carry-immunity and carried∧demoted
conflict rules (fail safe toward carrying, with a warning) have their single
home there. What reaches this module is the already-reconciled ``demoted``
loadout (evidence-driven expand_only assignments) plus the explicit ``carry``
promotions; this module only applies the set algebra against the live
ceiling ``E``, which is where the full-start default materializes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CarryingModel:
    """The resolved partition and per-message active set.

    Field names are chosen so the contract's tolerant accessors read them
    directly (``always_carry``/``carry``/``expand_only``/``active``).
    """

    always_carry: set[str] = field(default_factory=set)
    carry: set[str] = field(default_factory=set)
    expand_only: set[str] = field(default_factory=set)
    active: set[str] = field(default_factory=set)


def _coerce(value) -> set[str]:
    """Coerce a name-collection to a ``set[str]``.

    A lone string is treated as a single name. ``None`` becomes the empty set.
    Iteration errors propagate to the caller's fail-open handler.
    """
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    out: set[str] = set()
    for item in value:  # may raise — caller's try/except handles fail-open
        if isinstance(item, str) and item:
            out.add(item)
    return out


def resolve(
    *,
    enabled,
    always_carry,
    carry,
    demoted=(),
    triggered=(),
    expanded=(),
) -> CarryingModel:
    """Finalize the carrying partition over the live enabled ceiling ``E``.

    Every argument is a collection of tool names. ``enabled`` is authoritative:
    no other source can introduce a name it doesn't contain. Returns a
    :class:`CarryingModel`. Never raises — an internal failure fails open,
    returning ``active == enabled`` (no narrowing).
    """
    E: set[str] = set()
    try:
        E = _coerce(enabled)
        return _resolve_inner(E, always_carry, carry, demoted, triggered, expanded)
    except Exception as exc:  # fail OPEN — never narrow on an internal error
        logger.warning("tool-belt: carrying.resolve failed (%s) — failing open", exc)
        return CarryingModel(
            always_carry=set(),
            carry=set(),
            expand_only=set(E),
            active=set(E),
        )


def _resolve_inner(
    E: set[str],
    always_carry,
    carry,
    demoted,
    triggered,
    expanded,
) -> CarryingModel:
    ac = _coerce(always_carry)
    ca = _coerce(carry)
    dm = _coerce(demoted)
    tr = _coerce(triggered)
    ex = _coerce(expanded)

    # ── Class A: immutable residents (always_carry ∩ E). ──────────────────
    A = ac & E

    # ── Class C: adaptive residents — full-start minus demotions. ─────────
    # Everything enabled and not always_carry is carried unless an
    # evidence-driven demotion names it; an explicit ``carry`` entry
    # (learned promotion) wins over a demotion for a tool named in both.
    # Subtracting A enforces the always_carry precedence inside the
    # partition itself; always_carry-immunity to demotion was already
    # reconciled upstream in ``learned.apply_to_preset``.
    C = ((E - A) - dm) | ((ca & E) - A)

    # ── Class X: expand_only — demoted-only (unknowns are carried). ────────
    X = E - (A | C)

    # ── Activation over X only (residency never changes). ─────────────────
    T = tr & X
    R = ex & X

    return CarryingModel(
        always_carry=A,
        carry=C,
        expand_only=X,
        active=A | C | T | R,
    )
