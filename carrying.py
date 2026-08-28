"""Carrying-model partition resolver — Tool Belt 1.0.

Given Hermes' *enabled built-in ceiling* ``E`` and the plugin's carrying
loadout, this module finalizes the three-way residency partition and the
per-message active set. It is the single authority for the locked algebra::

    A = always_carry ∩ E                     # immutable residents (class A)
    C = (carry ∩ E) − A                      # adaptive residents  (class C)
    X = E − (A ∪ C)                          # expand_only         (class X)
    T = triggered ∩ X                        # trigger-activated expand_only
    R = expanded  ∩ X                        # explicitly expanded expand_only
    active = A ∪ C ∪ T ∪ R ∪ passthrough ∪ (prior_active ∩ (E ∪ passthrough))

Invariants:

  * A, C, X are a genuine three-way partition of E (precedence
    always_carry > carry > expand_only).
  * No selection source (policy, learning, triggers, sticky/prior-active
    carry-forward, or explicit expansion) can add a tool absent from ``E``.
    Disabled/absent tools never re-enter.
  * ``always_carry`` is immune to a learned ``demoted`` (always_off) signal —
    the demotion is ignored and a warning is emitted.
  * A contradictory overlap (a tool both carried and demoted) fails safe
    *toward carrying*: the tool stays resident and a warning is emitted.
  * MCP/plugin ``passthrough`` tools live outside the built-in partition; they
    are never narrowed and never counted in A/C/X.
  * Any internal failure fails **open** — the returned model's active set is
    the whole enabled ceiling (no narrowing).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CarryingModel:
    """The resolved partition and per-message active set.

    Field names are chosen so the contract's tolerant accessors read them
    directly (``always_carry``/``carry``/``expand_only``/``active``/
    ``passthrough``/``warnings``). ``A``/``C``/``X`` aliases are provided for
    call sites that prefer the algebra's letters.
    """

    always_carry: set[str] = field(default_factory=set)
    carry: set[str] = field(default_factory=set)
    expand_only: set[str] = field(default_factory=set)
    active: set[str] = field(default_factory=set)
    passthrough: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)

    # Algebra-letter aliases.
    @property
    def A(self) -> set[str]:
        return self.always_carry

    @property
    def C(self) -> set[str]:
        return self.carry

    @property
    def X(self) -> set[str]:
        return self.expand_only


def _coerce(value) -> set[str]:
    """Coerce a name-collection to a ``set[str]``.

    A lone string is treated as a single name. ``None`` becomes the empty set.
    Any element that raises or isn't string-able is skipped defensively.
    """
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    out: set[str] = set()
    for item in value:  # may raise — caller's try/except handles fail-open
        if isinstance(item, str):
            if item:
                out.add(item)
        elif item is not None:
            out.add(str(item))
    return out


def resolve(
    *,
    enabled,
    always_carry,
    carry,
    triggered=(),
    expanded=(),
    passthrough=(),
    demoted=(),
    prior_active=(),
) -> CarryingModel:
    """Finalize the carrying partition over the live enabled ceiling ``E``.

    Every argument is a collection of tool names. ``enabled`` is authoritative:
    no other source can introduce a name it doesn't contain. Returns a
    :class:`CarryingModel`. Never raises — an internal failure fails open,
    returning ``active == enabled`` (no narrowing).
    """
    try:
        E = _coerce(enabled)
    except Exception:
        E = set()
    try:
        return _resolve_inner(
            E, always_carry, carry, triggered, expanded, passthrough, demoted, prior_active
        )
    except Exception as exc:  # fail OPEN — never narrow on an internal error
        logger.warning("tool-belt: carrying.resolve failed (%s) — failing open", exc)
        return CarryingModel(
            always_carry=set(),
            carry=set(),
            expand_only=set(E),
            active=set(E),
            passthrough=set(),
            warnings=[f"carrying resolve failed open: {type(exc).__name__}: {exc}"],
        )


def _resolve_inner(
    E: set[str],
    always_carry,
    carry,
    triggered,
    expanded,
    passthrough,
    demoted,
    prior_active,
) -> CarryingModel:
    ac = _coerce(always_carry)
    ca = _coerce(carry)
    tr = _coerce(triggered)
    ex = _coerce(expanded)
    pt = _coerce(passthrough)
    dm = _coerce(demoted)
    pa = _coerce(prior_active)

    warnings: list[str] = []

    # Passthrough (MCP/plugin) tools sit outside the built-in partition. Never
    # let a passthrough name leak into the built-in ceiling accounting.
    pt_active = set(pt)
    E_builtin = set(E) - pt_active

    # ── Class A: immutable residents (always_carry ∩ E). ──────────────────
    A = ac & E_builtin

    # A learned demotion naming an always_carry tool is impossible-by-policy;
    # ignore it and warn (immunity).
    for name in sorted(dm & A):
        warnings.append(
            f"demotion signal names always_carry tool {name!r}; ignored (immune)"
        )

    # ── Class C: adaptive residents (carry ∩ E) − A. ──────────────────────
    C = (ca & E_builtin) - A

    # Contradictory overlap: a tool that is both a carry resident and named by
    # a demotion signal. Fail safe *toward carrying* — keep it resident — and
    # warn so the conflict is auditable rather than silently narrowed away.
    for name in sorted(dm & C):
        warnings.append(
            f"tool {name!r} is both carried and demoted; failing safe toward "
            "carrying (stays resident)"
        )

    # ── Class X: expand_only — the enabled remainder (incl. unknowns). ─────
    X = E_builtin - (A | C)

    # ── Activation over X only (residency never changes). ─────────────────
    T = tr & X
    R = ex & X

    # prior_active carries forward the session's frozen active set (cache-on).
    # It is intersected with the live ceiling (plus passthrough) so a stale
    # entry for a now-disabled tool can never re-enter.
    prior = pa & (E_builtin | pt_active)

    active = A | C | T | R | pt_active | prior

    return CarryingModel(
        always_carry=A,
        carry=C,
        expand_only=X,
        active=active,
        passthrough=pt_active,
        warnings=warnings,
    )
