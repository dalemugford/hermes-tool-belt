"""Regression locks for the per-provider cache axis (D3) and cache-aware
savings (D4), landed 2026-09-02.

The bug these guard against: cache-mode was keyed per SCOPE, but whether
narrowing helps or hurts is a property of the PROVIDER, per call. A scope
that fails over between a caching primary and a non-caching fallback was
mis-locked in both directions. Each test here fails on the pre-fix tree
(`_detection_key`, the ``buckets`` sub-structure, ``provider_caches`` and
``measure_expand_overhead`` did not exist).
"""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401

plugin = sys.modules["tool_belt_plugin"]
savings = importlib.import_module("tool_belt_plugin.savings")
shaping = importlib.import_module("tool_belt_plugin.shaping")


def _seed_config() -> None:
    plugin._CONFIG.clear()
    plugin._CONFIG.update({
        "enabled": True, "log": False, "agent": "assistant-a",
        "bypass_rate": 0.0, "cache_mode": "auto", "channels": {},
        "cache_auto": {
            "detect_calls": 5, "detect_min_input_tokens": 5000,
            "on_threshold": 0.40,
            "providers_off_models": ["kimi-k2.6:cloud", "ollama-cloud"],
        },
    })


def _reset() -> None:
    plugin._CACHE_MODE_BY_SESSION.clear()
    plugin._CACHE_DECISION_BY_SESSION.clear()
    plugin._DETECTION_CACHE.clear()
    plugin._DETECTION_CACHE_LOADED = False
    plugin._HOST_MODEL.update(provider="", model="")


class ProviderBucketIsolationTests(unittest.TestCase):
    """The core fix: a non-caching provider's calls never drag a caching
    provider's bucket within one scope."""

    SCOPE = "assistant-a:telegram"

    def setUp(self):
        _seed_config()
        _reset()

    def test_failover_provider_does_not_flip_primary_bucket(self):
        # Primary (openai-codex): 5 warm calls → locks "on".
        for _ in range(5):
            plugin._update_cache_mode_detection(
                session_key="sid", model="gpt-5.4",
                cache_read=8000, cache_write=0, input_tokens=2000,
                scope=self.SCOPE, provider="openai-codex",
            )
        # Fallback (ollama-cloud): many cold calls in the SAME session.
        for _ in range(8):
            plugin._update_cache_mode_detection(
                session_key="sid", model="glm-5.3",
                cache_read=0, cache_write=0, input_tokens=9000,
                scope=self.SCOPE, provider="ollama-cloud",
            )
        buckets = plugin._CACHE_MODE_BY_SESSION["sid"]["buckets"]
        self.assertEqual(buckets["openai-codex"]["mode"], "on",
                         "the caching primary stays locked on")
        self.assertEqual(buckets["ollama-cloud"]["mode"], "off",
                         "the non-caching fallback locks off in its own bucket")

    def test_provider_caches_reports_per_bucket(self):
        for _ in range(5):
            plugin._update_cache_mode_detection(
                session_key="sid", model="gpt-5.4",
                cache_read=8000, cache_write=0, input_tokens=2000,
                scope=self.SCOPE, provider="openai-codex",
            )
        self.assertIs(plugin._provider_caches(self.SCOPE, "openai-codex", "sid"), True)
        # A provider we've never observed is unknown, not False.
        self.assertIsNone(plugin._provider_caches(self.SCOPE, "openrouter", "sid"))

    def test_providers_off_models_matches_provider_name(self):
        # ollama-cloud is a provider NAME in the blocklist (not a model id).
        mode = plugin._update_cache_mode_detection(
            session_key="sid", model="glm-5.3",
            cache_read=0, cache_write=0, input_tokens=100,
            scope=self.SCOPE, provider="ollama-cloud",
        )
        self.assertEqual(mode, "off")
        self.assertEqual(
            plugin._CACHE_MODE_BY_SESSION["sid"]["buckets"]["ollama-cloud"]["lock_reason"],
            "provider_blocklist",
        )


class ProviderChangeEvictionTests(unittest.TestCase):
    """A mid-session provider change that flips the posture evicts the pin."""

    SCOPE = "assistant-a:telegram"

    def setUp(self):
        _seed_config()
        _reset()

    def test_posture_flip_evicts_pin_and_keeps_new_provider(self):
        # Pinned "on" against the caching primary.
        plugin._CACHE_DECISION_BY_SESSION["sid"] = {
            "mode": "on", "provider": "openai-codex"}
        # The fallback bucket is known-off.
        plugin._DETECTION_CACHE[plugin._detection_key(self.SCOPE, "ollama-cloud")] = {
            "mode": "off"}
        plugin._maybe_evict_on_provider_change("sid", self.SCOPE, "ollama-cloud")
        self.assertNotIn("sid", plugin._CACHE_DECISION_BY_SESSION,
                         "posture flip on provider change evicts the stale pin")

    def test_same_posture_change_updates_provider_silently(self):
        plugin._CACHE_DECISION_BY_SESSION["sid"] = {
            "mode": "on", "provider": "openai-codex"}
        # Another caching provider — same posture, no eviction, pin updated.
        plugin._DETECTION_CACHE[plugin._detection_key(self.SCOPE, "openrouter")] = {
            "mode": "on"}
        plugin._maybe_evict_on_provider_change("sid", self.SCOPE, "openrouter")
        self.assertEqual(
            plugin._CACHE_DECISION_BY_SESSION["sid"]["provider"], "openrouter",
            "same-posture change keeps the pin but records the new provider")


class CacheAwareSavingsTests(unittest.TestCase):
    """D4: gross priced by provider caching; overhead measured with a
    thin-data fallback."""

    def test_cache_status_comes_off_the_row_never_from_the_provider(self):
        # The same provider name, two rows. The explicit flag is honoured;
        # its absence is unknown — for the caller to resolve from the cohort
        # posture — and never guessed from the provider name.
        self.assertIs(savings.provider_caches_for_call(
            {"provider": "ollama-cloud", "provider_caches": False}), False)
        self.assertIsNone(savings.provider_caches_for_call(
            {"provider": "ollama-cloud"}))
        # Both price saved schema tokens at full rate, not the cache_read
        # discount.
        self.assertEqual(savings.price_factor_for("gpt-5.4", False), 1.0)
        self.assertEqual(savings.price_factor_for("gpt-5.4", None), 1.0)

    def test_caching_gross_is_cache_read_discounted(self):
        factor = savings.price_factor_for("gpt-5.4", True)
        self.assertLess(factor, 1.0,
                        "on a caching provider saved schema tokens are worth "
                        "the cache_read rate, not full input")


class NarrowedTrafficShapingTests(unittest.TestCase):
    """Shaping keys on the TRAFFIC, not on the configured primary provider.

    A carry-all turn shipped the full ceiling and offered no ``expand_tools``,
    so it is evidence for neither arm. These fail on a tree that short-circuits
    the whole scope on its primary provider's cache posture: there, a mixed
    scope either learns from carry-all sessions it should ignore, or learns
    from nothing at all.
    """

    SCOPE = "assistant-a:telegram"
    E = ["clarify", "browser_exec", "terminal"]

    def _sessions(self, n, *, carry_all_ids=(), start=0):
        """n sessions, one prediction each, ``browser_exec`` carried and never
        used. Sessions named in ``carry_all_ids`` are stamped carry-all."""
        preds, calls = [], []
        for i in range(start, start + n):
            pid = f"p{i}"
            row = {
                "schema_version": 2,
                "scope": self.SCOPE,
                "hermes_session_id": f"s{i}",
                "prediction_id": pid,
                "ts": float(i),
                "ceiling_tools": list(self.E),
                "always_carry_tools": ["clarify"],
                "carry_tools": ["browser_exec", "terminal"],
                "active_tools": list(self.E),
                "expand_only_tools": [],
            }
            if i in carry_all_ids:
                row["policy_source"] = "cache_on_carry_all"
            preds.append(row)
            calls.append({
                "schema_version": 2, "prediction_id": pid,
                "tool_name": "terminal", "was_initially_active": True,
                "activation_source": "carry",
            })
        return preds, calls

    def _compute(self, preds, calls, **overrides):
        grouped = shaping.group_predictions_by_scope_session(preds)
        kwargs = dict(
            window_days=7, promote_min_sessions=2, promote_min_calls=3,
            demote_min_sessions_no_use=20,
            schema_sizes={"browser_exec": 2000},
        )
        kwargs.update(overrides)
        return shaping.compute_scope_recommendations(
            scope=self.SCOPE, sessions=grouped.get(self.SCOPE, {}),
            calls_by_pred=shaping.index_tool_calls_by_prediction(calls),
            **kwargs,
        )

    def test_mixed_traffic_shapes_from_the_narrowed_sessions_only(self):
        # 40 sessions, 20 of them carry-all: only the 20 narrowed ones are
        # evidence, and the demote saving is priced over those alone.
        preds, calls = self._sessions(40, carry_all_ids=set(range(20)))
        recs = self._compute(preds, calls)
        self.assertEqual(recs.get("sessions_considered"), 20)
        entry = {d["tool"]: d for d in recs["demote"]}
        self.assertIn("browser_exec", entry)
        self.assertEqual(entry["browser_exec"]["sessions_without_use"], 20)
        self.assertEqual(entry["browser_exec"]["carry_tokens"], 2000 * 20)

    def test_carry_all_sessions_do_not_reach_the_session_floor(self):
        # 25 sessions, 20 of them carry-all: 5 narrowed sessions is under
        # demote_min_sessions_no_use=20, so the demote arm must not run.
        preds, calls = self._sessions(25, carry_all_ids=set(range(20)))
        recs = self._compute(preds, calls)
        self.assertEqual(recs.get("sessions_considered"), 5)
        self.assertEqual(recs["demote"], [],
                         "carry-all sessions never help clear the session floor")

    def test_scope_with_only_carry_all_sessions_is_a_no_op(self):
        preds, calls = self._sessions(30, carry_all_ids=set(range(30)))
        recs = self._compute(preds, calls)
        self.assertEqual(recs.get("reason"), "no_narrowed_sessions")
        self.assertEqual(recs.get("sessions_considered"), 0)
        self.assertEqual(recs.get("demote"), [])
        self.assertEqual(recs.get("promote"), [])


class BlocklistFirstDispatchTests(unittest.TestCase):
    """A provider/model in ``providers_off_models`` must resolve "off" at the
    session's FIRST dispatch, before any API call has produced a lock.

    The two ``*_resolves_off_*`` tests are the regression locks: they fail on
    the pre-fix tree, where ``_resolve_posture_for_provider`` returned "on" for
    any unlocked bucket, so a known-uncached route shipped one full carry-all
    session while the post-call lock landed. ``test_unlisted_route_still_defaults_on``
    and ``test_explicit_locked_bucket_wins_over_blocklist_default`` guard
    behavior the fix leaves unchanged (they pass on both trees).
    ``test_blocklisted_primary_model_does_not_narrow_caching_failover`` locks
    the failover safeguard — it fails on a naive fix that attributes the
    primary model to every provider.
    """

    SCOPE = "assistant-a:telegram"

    def setUp(self):
        _seed_config()
        _reset()

    def test_blocklisted_provider_name_resolves_off_before_any_call(self):
        # ollama-cloud is blocklisted by name; no _update_cache_mode_detection
        # call has run, so the bucket is unlocked.
        posture = plugin._resolve_posture_for_provider(self.SCOPE, "ollama-cloud")
        self.assertEqual(posture, "off",
                         "a blocklisted provider narrows from the first dispatch")

    def test_blocklisted_model_resolves_off_via_session_resolver(self):
        # Host provider is NOT blocklisted, but the configured model IS.
        plugin._HOST_MODEL.update(provider="openrouter", model="kimi-k2.6:cloud")
        posture = plugin._resolve_cache_mode_for_session("sid-new", scope=self.SCOPE)
        self.assertEqual(posture, "off",
                         "a blocklisted model narrows the first session even when "
                         "the provider name is not itself blocklisted")

    def test_unlisted_route_still_defaults_on(self):
        # The safe default is unchanged for routes we know nothing about.
        posture = plugin._resolve_posture_for_provider(self.SCOPE, "openrouter")
        self.assertEqual(posture, "on",
                         "an unlisted, unlocked route still defaults to carry-all")

    def test_blocklisted_primary_model_does_not_narrow_caching_failover(self):
        # Finding 1 regression: the primary model is blocklisted, but the
        # session has failed over to a caching provider that is NOT itself
        # blocklisted. The stale primary model must not be attributed to the
        # failover provider — otherwise the caching provider narrows per-turn
        # and its prefix cache is busted.
        plugin._HOST_MODEL.update(provider="openai-codex", model="kimi-k2.6:cloud")
        plugin._CACHE_MODE_BY_SESSION["sid-fo"] = {"observed_provider": "openrouter"}
        posture = plugin._resolve_cache_mode_for_session("sid-fo", scope=self.SCOPE)
        self.assertEqual(posture, "on",
                         "a blocklisted PRIMARY model must not narrow an "
                         "unrelated caching failover provider")

    def test_explicit_locked_bucket_wins_over_blocklist_default(self):
        # A bucket already locked "on" by observation is honored; the blocklist
        # only supplies the UNLOCKED default, it doesn't override a real lock.
        for _ in range(5):
            plugin._update_cache_mode_detection(
                session_key="obs", model="gpt-5.4",
                cache_read=8000, cache_write=0, input_tokens=2000,
                scope=self.SCOPE, provider="openrouter",
            )
        self.assertEqual(
            plugin._resolve_posture_for_provider(self.SCOPE, "openrouter", "obs"),
            "on", "an observed lock is not second-guessed by the blocklist path")


if __name__ == "__main__":
    unittest.main()
