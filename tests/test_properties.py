"""Property invariants: the algebra that must hold for ANY input, not fixtures.

Example-based tests prove points; these prove the space. Seeded random
generation (deterministic, no hypothesis dependency), ~200 cases per property.

Justifications:
  · partition/containment — the carrying model's algebra (A/C/X disjoint,
    active ⊆ E ∪ passthrough, full-start supremum) for arbitrary inputs
    including names outside E; a fixture can't sweep this space.
  · pin immunity — always_carry ∩ E ⊆ active for ANY learned/trigger/expansion
    state; the only test proving pins are undemotable by construction rather
    than by the specific states other tests happen to build.
  · never empty — active == ∅ is the one forbidden fail state; adversarial
    inputs (raising iterables, unstringables) must fail OPEN, not closed.
  · wrapper fail-open — an injected exception at each internal seam of the
    real wrapped _build_api_kwargs must return the original kwargs unchanged;
    the only test sweeping every seam instead of one.
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401

plugin = sys.modules["tool_belt_plugin"]
from tool_belt_plugin import carrying  # noqa: E402

POOL = [f"tool_{i}" for i in range(24)] + ["clarify", "expand_tools", "terminal"]
FOREIGN = [f"ghost_{i}" for i in range(6)]  # names outside E — must never enter


def _cases(seed, n=200):
    rng = random.Random(seed)
    for _ in range(n):
        e = set(rng.sample(POOL, rng.randint(1, len(POOL))))
        pick = lambda: set(rng.sample(POOL + FOREIGN,
                                      rng.randint(0, 10)))
        yield e, pick(), pick(), pick(), pick(), pick(), pick()


class PartitionAndContainment(unittest.TestCase):
    def test_partition_algebra_holds_for_random_inputs(self):
        for e, ac, carry, demoted, trig, exp, prior in _cases(101):
            m = carrying.resolve(enabled=e, always_carry=ac, carry=carry,
                                 demoted=demoted, triggered=trig, expanded=exp,
                                 passthrough=(), prior_active=prior)
            a, c, x = set(m.always_carry), set(m.carry), set(m.expand_only)
            self.assertEqual(a & c, set(), "A and C disjoint")
            self.assertEqual(a & x, set(), "A and X disjoint")
            self.assertEqual(c & x, set(), "C and X disjoint")
            self.assertEqual(a | c | x, e, "A∪C∪X covers exactly E")
            self.assertLessEqual(set(m.active), e,
                                 "active never exceeds the ceiling")
            self.assertEqual(set(m.active) & set(FOREIGN), set(),
                             "a name outside E can never activate")
            if not (set(demoted) & e):
                self.assertEqual(set(m.active), e,
                                 "no demotions ⇒ full start (active == E)")


class PinImmunity(unittest.TestCase):
    def test_pins_active_for_any_learned_state(self):
        for e, ac, carry, demoted, trig, exp, prior in _cases(202):
            m = carrying.resolve(enabled=e, always_carry=ac, carry=carry,
                                 demoted=demoted | ac,  # hostile: demote the pins too
                                 triggered=trig, expanded=exp,
                                 passthrough=(), prior_active=prior)
            self.assertLessEqual(ac & e, set(m.active),
                                 "every enabled pin is active regardless of "
                                 "learned demotions naming it")


class NeverEmpty(unittest.TestCase):
    def test_adversarial_inputs_fail_open_never_closed(self):
        class _RaisingIter:
            def __iter__(self):
                raise RuntimeError("boom")

        class _Unstringable:
            def __str__(self):
                raise RuntimeError("boom")

        e = {"clarify", "terminal", "read_file"}
        hostile = [
            dict(always_carry=_RaisingIter()),
            dict(carry=_RaisingIter()),
            dict(demoted=_RaisingIter()),
            dict(triggered=_RaisingIter()),
            dict(expanded=_RaisingIter()),
            dict(prior_active=_RaisingIter()),
            dict(enabled=list(e) + [_Unstringable()]),
        ]
        for kw in hostile:
            args = dict(enabled=e, always_carry=set(), carry=set(),
                        demoted=set(), triggered=set(), expanded=set(),
                        passthrough=(), prior_active=set())
            args.update(kw)
            m = carrying.resolve(**args)
            self.assertTrue(set(m.active) >= e or set(m.active) >= e - {""},
                            f"hostile {list(kw)} must fail OPEN to the "
                            f"readable ceiling, got {sorted(m.active)}")
            self.assertNotEqual(set(m.active), set(),
                                "active == ∅ is the forbidden fail state")


class HostSignatureForwarding(unittest.TestCase):
    """The deployed gateway calls _build_api_kwargs(api_messages,
    tools_for_api=...) on cache-plan branches; a fixed two-argument wrapper
    raises TypeError at call BINDING — before fail-open can engage — so the
    wrapper must accept and forward arbitrary host args."""

    def test_extra_host_kwargs_are_forwarded(self):
        seen: dict = {}

        def original(_self, _msgs, tools_for_api=None):
            seen["tools_for_api"] = tools_for_api
            return {"tools": [], "model": "m"}

        wrapped = plugin._wrap_build_api_kwargs(original)
        with mock.patch.dict(plugin._CONFIG, {"enabled": True}):
            out = wrapped(object(), [], tools_for_api=["decorated-copy"])
        self.assertEqual(seen["tools_for_api"], ["decorated-copy"])
        self.assertIn("tools", out)


class WrapperFailOpen(unittest.TestCase):
    """An exception at ANY internal seam of the wrapped _build_api_kwargs
    returns the original kwargs unchanged — the gateway never sees the error."""

    SEAMS = [
        "_resolve_cache_mode_for_session",
        "_maybe_log_prediction",
        "_tool_list_hash",
    ]

    def _run(self):
        defs = [{"name": n, "description": "d", "input_schema": {}}
                for n in ("clarify", "expand_tools", "read_file", "terminal")]

        def original(_self, _msgs):
            return {"tools": list(defs), "model": "m"}

        state = {
            "active_tool_names": ["clarify"],
            "resolved_always_carry": ["clarify", "expand_tools"],
            "resolved_carry": ["read_file", "terminal"],
            "resolved_demoted": [], "triggered_tools": [],
            "expansions": set(), "logged": False, "session_id": "",
        }
        token = plugin._PREDICTION_CV.set(state)
        try:
            wrapped = plugin._wrap_build_api_kwargs(original)
            with mock.patch.dict(plugin._CONFIG, {"enabled": True}):
                return wrapped(object(), [])
        finally:
            plugin._PREDICTION_CV.reset(token)

    def test_injected_exception_at_each_seam_fails_open(self):
        for seam in self.SEAMS:
            if not hasattr(plugin, seam):
                self.fail(f"seam {seam} vanished — update the fail-open sweep")
            with self.subTest(seam=seam), \
                    mock.patch.object(plugin, seam,
                                      side_effect=RuntimeError("boom")):
                result = self._run()
                names = {plugin._tool_name(t) for t in result["tools"]}
                self.assertEqual(
                    names,
                    {"clarify", "expand_tools", "read_file", "terminal"},
                    f"exception in {seam} must ship the original toolset")


if __name__ == "__main__":
    unittest.main()
