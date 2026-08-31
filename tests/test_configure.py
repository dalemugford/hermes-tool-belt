"""Coverage for the onboarding command (``scripts/configure.py``).

Everything here runs against a temporary ``HERMES_HOME`` and a fake
``subprocess.run``. No live Hermes state is read and no real ``hermes``
process is ever spawned.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PLUGIN_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = PLUGIN_DIR / "tests"
sys.path.insert(0, str(TESTS_DIR))
import conftest  # noqa: F401,E402 — registers tool_belt_plugin


def _load_script(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        module_name, PLUGIN_DIR / "scripts" / filename
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


configure = _load_script("tool_belt_configure", "configure.py")


class FakeRunner:
    """Stands in for ``subprocess.run``; records argv, replays canned reads."""

    def __init__(self, get_values: dict[str, str] | None = None, returncode: int = 0):
        self.calls: list[list[str]] = []
        self.get_values = get_values or {}
        self.returncode = returncode

    def __call__(self, argv, capture_output=False, text=False, check=False):
        self.calls.append(list(argv))
        stdout = ""
        if len(argv) >= 4 and argv[1:3] == ["config", "get"]:
            key = argv[3]
            stdout = self.get_values.get(key, f"Config key not set: {key}")
        return mock.Mock(returncode=self.returncode, stdout=stdout, stderr="")

    @property
    def writes(self) -> list[list[str]]:
        return [c for c in self.calls if c[1:3] in (["config", "set"], ["config", "unset"])]


def isolate(runner: FakeRunner, which: str | None = "/usr/bin/hermes", sink: list | None = None):
    """Every path out of the module that could touch the real machine.

    Both ``subprocess.run`` and ``_default_runner`` are patched: the module
    resolves the former at call time, but pinning both means a future default
    argument that binds early still cannot reach a real ``hermes``.
    """
    patches = [
        mock.patch.object(configure.subprocess, "run", runner),
        mock.patch.object(configure, "_default_runner", runner),
        mock.patch.object(configure.shutil, "which", return_value=which),
    ]
    if sink is not None:
        patches.append(
            mock.patch.object(
                configure, "_default_out", lambda m, _s=sink: _s.append(str(m))
            )
        )
        patches.append(mock.patch("builtins.print", lambda *a, **k: sink.append(" ".join(str(x) for x in a))))
    return patches


def _write_jsonl(path: Path, rows: list[dict], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r) + "\n" for r in rows)
    with path.open("a" if append else "w", encoding="utf-8") as fh:
        fh.write(body)


def seed_telemetry(
    state_dir: Path,
    scope: str,
    sessions: int,
    always_on: list[str] | None = None,
    expanded_tool: str | None = None,
    expanded_sessions: int = 0,
    expanded_calls_each: int = 0,
    append: bool = False,
) -> None:
    """Write synthetic predictions/tool_calls for ``sessions`` distinct sessions."""
    preds: list[dict] = []
    calls: list[dict] = []
    for i in range(sessions):
        pid = f"{scope}-p{i}"
        preds.append(
            {
                "ts": 1000 + i,
                "scope": scope,
                "hermes_session_id": f"sess-{i}",
                "prediction_id": pid,
                "always_on_tools": list(always_on or []),
            }
        )
        if expanded_tool and i < expanded_sessions:
            for _ in range(expanded_calls_each):
                calls.append(
                    {
                        "prediction_id": pid,
                        "tool_name": expanded_tool,
                        "was_expanded": True,
                    }
                )
    _write_jsonl(state_dir / "predictions.jsonl", preds, append=append)
    _write_jsonl(state_dir / "tool_calls.jsonl", calls, append=append)


def make_ctx(hermes_home: Path, runner: FakeRunner, **kwargs) -> "configure.RunContext":
    kwargs.setdefault("thresholds", configure.shape_thresholds())
    kwargs.setdefault("out", lambda _msg: None)
    return configure.RunContext(hermes_home=hermes_home, runner=runner, **kwargs)


class TempHomeTestCase(unittest.TestCase):
    """Every test gets a throwaway Hermes home whose path contains a space."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / "custom hermes home"
        self.home.mkdir()
        self.root_state = self.home / "state" / "tool-belt"
        self.root_state.mkdir(parents=True)
        self.thresholds = configure.shape_thresholds()
        self.needed = configure.required_sessions(self.thresholds)


class ScopeDiscoveryTests(TempHomeTestCase):
    def test_discovers_default_and_named_profiles(self) -> None:
        named_state = self.home / "profiles" / "assistant-a" / "state" / "tool-belt"
        named_state.mkdir(parents=True)
        (self.home / "profiles" / "default" / "state" / "tool-belt").mkdir(parents=True)

        labels = [label for label, _ in configure.discover_state_dirs(self.home)]
        self.assertEqual(labels, ["default", "assistant-a"])
        self.assertEqual(
            configure.discover_state_dirs(self.home, "assistant-a"),
            [("assistant-a", named_state)],
        )

    def test_scopes_come_from_predictions_with_session_counts(self) -> None:
        seed_telemetry(self.root_state, "default:telegram", 3)
        named_state = self.home / "profiles" / "assistant-a" / "state" / "tool-belt"
        seed_telemetry(named_state, "assistant-a:slack", 5)

        infos = configure.discover_scopes(self.home)
        by_scope = {i.scope: i for i in infos}
        self.assertEqual(set(by_scope), {"default:telegram", "assistant-a:slack"})
        self.assertEqual(by_scope["default:telegram"].sessions, 3)
        self.assertEqual(by_scope["default:telegram"].agent, "default")
        self.assertEqual(by_scope["default:telegram"].platform, "telegram")
        self.assertEqual(by_scope["assistant-a:slack"].sessions, 5)
        self.assertEqual(by_scope["assistant-a:slack"].state_dir, named_state)

    def test_platform_hint_used_when_no_telemetry(self) -> None:
        infos = configure.discover_scopes(self.home, platform_hint=["telegram", "cli"])
        self.assertEqual([i.scope for i in infos], ["default:telegram", "default:cli"])
        self.assertTrue(all(i.inferred for i in infos))
        # ...and no hint means no invented scopes.
        self.assertEqual(configure.discover_scopes(self.home), [])

    def test_absent_home_is_cleanly_ignored(self) -> None:
        missing = self.home / "does not exist"
        self.assertEqual(configure.discover_state_dirs(missing), [])
        self.assertEqual(configure.discover_scopes(missing), [])


class AgentNameFilterTests(TempHomeTestCase):
    """B1/M3/P3 locks: --agent must accept the agent name the tool itself
    displays (plugins.tool-belt.agent), not only the profile directory name;
    a filter miss on a populated install must name what exists and exit 2,
    never claim the install is empty (or silently no-op a --dry-run). B2:
    --platform comma-splits like the interactive prompt. No other tests
    exercise the display-name filter or the flag normalization."""

    def _root_with_configured_name(self, name="bernard"):
        (self.home / "config.yaml").write_text(
            f"plugins:\n  tool-belt:\n    agent: {name}\n", encoding="utf-8")
        seed_telemetry(self.root_state, f"{name}:telegram", 3)

    def test_filter_accepts_configured_agent_name(self) -> None:
        self._root_with_configured_name("bernard")
        hits = configure.discover_state_dirs(self.home, "bernard")
        self.assertEqual([label for label, _ in hits], ["default"],
                         "the displayed agent name selects its profile")
        self.assertTrue(configure.discover_scopes(self.home, "bernard"))
        self.assertEqual(configure.discover_state_dirs(self.home, "nosuch"), [])

    def test_filter_miss_names_found_profiles_and_exits_2(self) -> None:
        self._root_with_configured_name("bernard")
        lines: list[str] = []
        with mock.patch.object(configure.shutil, "which", return_value=None), \
                mock.patch.object(configure, "prompt",
                                  side_effect=configure.Abort), \
                contextlib.redirect_stdout(io.StringIO()) as buf:
            rc = configure.main(["--agent", "nosuch",
                                 "--hermes-home", str(self.home)])
        out = buf.getvalue()
        self.assertEqual(rc, 2, "wrong --agent on a populated install is an "
                                "error, not an empty install / silent no-op")
        self.assertIn("No profile matching 'nosuch'", out)
        self.assertIn("bernard", out, "the error names what DOES exist, "
                                      "including the displayed agent name")
        self.assertNotIn("install and", out,
                         "must not tell a populated install to go install Hermes")

    def test_platform_flag_comma_splits_like_the_prompt(self) -> None:
        self.assertEqual(configure.split_platform_args(["telegram, slack"]),
                         ["telegram", "slack"])
        self.assertEqual(configure.split_platform_args(["cli"]), ["cli"])
        self.assertIsNone(configure.split_platform_args(None))
        self.assertIsNone(configure.split_platform_args([" , "]))


class ConfigureModeFlowTests(TempHomeTestCase):
    """configure = agent → channels → mode (learning/history/off). Locks the
    three-step shape and each mode's learned_mode mapping. No other test
    drives the top-level flow; these are its contract."""

    def _seed_two_channels(self, agent="default"):
        # always_on tools that never get called → real demotion evidence, so
        # the history path produces a non-empty overlay to write.
        seed_telemetry(self.root_state, f"{agent}:telegram", self.needed,
                       always_on=["web_search", "read_file"])
        seed_telemetry(self.root_state, f"{agent}:slack", self.needed,
                       always_on=["web_search", "read_file"], append=True)

    def _run(self, keys: str):
        infos = configure.discover_scopes(self.home)
        lines: list[str] = []
        it = iter(keys.splitlines())
        runner = FakeRunner()
        ctx = make_ctx(self.home, runner, assume_yes=False,
                       reader=lambda _p: next(it), out=lines.append,
                       plugin_config={})
        rc = configure._menu(ctx, infos)
        return rc, "\n".join(lines), runner

    def test_off_writes_recommend_mode(self):
        self._seed_two_channels()
        # one agent (skipped), channels 'all', mode 3 (off), confirm y ×2
        rc, out, runner = self._run("all\n3\ny\ny\n")
        self.assertEqual(rc, 0)
        keys = {c[3]: c[4] for c in runner.writes}
        self.assertEqual(
            keys.get("plugins.tool-belt.channels.default:slack.learned_mode"),
            "recommend", "OFF sets learned_mode=recommend (overlay not applied)")
        self.assertIn("Shaping is OFF", out)

    def test_learning_writes_apply_without_history_run(self):
        self._seed_two_channels()
        # channels 'all', mode 1 (learning), confirm y ×2
        rc, out, runner = self._run("all\n1\ny\ny\n")
        self.assertEqual(rc, 0)
        keys = {c[3]: c[4] for c in runner.writes}
        self.assertEqual(
            keys.get("plugins.tool-belt.channels.default:telegram.learned_mode"),
            "apply", "learning sets learned_mode=apply")
        # No learned overlay written on the learning path (that's history's job).
        self.assertFalse((self.root_state / "learned.json").exists(),
                         "learning mode does NOT shape from history now")
        self.assertIn("learning", out)

    def test_history_mode_delegates_to_shape(self):
        self._seed_two_channels()
        # channel 1 only, mode 2 (history), confirm ('y'), decline launcher.
        rc, out, runner = self._run("1\n2\ny\nn\n")
        self.assertEqual(rc, 0)
        # flow_shape's epilogue is emitted ONLY on the history path — the
        # faithful signal that mode 2 shapes from recorded history now,
        # unlike learning (mode 1), which shows the learning message and
        # runs no shaper. (A synthetic fixture can't drive the shaper's
        # residency-demotion path; delegation is what this locks.)
        self.assertIn("load a tighter tool set", out)
        keys = {c[3]: c[4] for c in runner.writes}
        self.assertEqual(
            keys.get("plugins.tool-belt.channels.default:slack.learned_mode"),
            "apply", "history also sets learned_mode=apply")

    def test_blank_at_any_step_cancels_without_writing(self):
        self._seed_two_channels()
        for keys in ("\n", "all\n\n"):  # cancel at channels, cancel at mode
            rc, out, runner = self._run(keys)
            self.assertEqual(rc, 0)
            self.assertEqual(runner.writes, [], "cancelling writes nothing")
            self.assertIn("Nothing selected", out)


class StateMachineTests(TempHomeTestCase):
    def _info(self, sessions: int) -> "configure.ScopeInfo":
        return configure.ScopeInfo(
            scope="default:telegram",
            agent="default",
            platform="telegram",
            state_dir=self.root_state,
            sessions=sessions,
        )

    def test_fresh_when_nothing_configured(self) -> None:
        settings = configure.scope_settings("default:telegram", {}, runner=FakeRunner())
        self.assertEqual(
            configure.classify_scope(self._info(0), settings, self.thresholds),
            configure.STATE_FRESH,
        )

    def test_observing_while_under_the_session_minimum(self) -> None:
        cfg = {"channels": {"default:telegram": {"bypass_rate": 1.0, "learned_mode": "recommend"}}}
        settings = configure.scope_settings("default:telegram", cfg)
        self.assertEqual(
            configure.classify_scope(self._info(self.needed - 1), settings, self.thresholds),
            configure.STATE_OBSERVING,
        )

    def test_ready_once_the_session_minimum_is_met(self) -> None:
        cfg = {"channels": {"default:telegram": {"bypass_rate": 1.0, "learned_mode": "recommend"}}}
        settings = configure.scope_settings("default:telegram", cfg)
        self.assertEqual(
            configure.classify_scope(self._info(self.needed), settings, self.thresholds),
            configure.STATE_READY,
        )

    def test_shaped_when_learned_mode_applies(self) -> None:
        cfg = {"channels": {"default:telegram": {"learned_mode": "apply"}}}
        settings = configure.scope_settings("default:telegram", cfg)
        self.assertEqual(
            configure.classify_scope(self._info(2), settings, self.thresholds),
            configure.STATE_SHAPED,
        )
        # Legacy alias normalizes to apply rather than reading as recommend.
        legacy = configure.scope_settings(
            "default:telegram", {"channels": {"default:telegram": {"learned_mode": "auto"}}}
        )
        self.assertEqual(
            configure.classify_scope(self._info(2), legacy, self.thresholds),
            configure.STATE_SHAPED,
        )

    def test_remaining_sessions_counts_down_and_floors_at_zero(self) -> None:
        self.assertEqual(configure.remaining_sessions(self._info(0), self.thresholds), self.needed)
        self.assertEqual(configure.remaining_sessions(self._info(self.needed + 5), self.thresholds), 0)

    def test_required_sessions_comes_from_policy_not_a_literal(self) -> None:
        self.assertEqual(
            configure.required_sessions({"promote_min_sessions": 4, "demote_min_sessions_no_use": 11}),
            11,
        )
        self.assertEqual(
            configure.required_sessions({"promote_min_sessions": 30, "demote_min_sessions_no_use": 11}),
            30,
        )


class ReShapePreviewTests(TempHomeTestCase):
    """The preview of a re-shape must equal what the apply then writes.

    Regression: the summary used to re-implement the move algebra as
    ``policy carry − demoted + promoted``, ignoring the scope's *existing*
    learned assignment — so re-shaping an already-shaped scope previewed a
    different loadout than ``merge_into_learned`` wrote.
    """

    SCOPE = "default:telegram"

    def _seed_already_shaped(self) -> "configure.ScopeInfo":
        seed_telemetry(
            self.root_state,
            self.SCOPE,
            sessions=self.needed,
            always_on=["web_search", "read_file"],
            expanded_tool="terminal",
            # Expanded in most sessions so the economic test decisively
            # favors carrying (penalty 1500/session-with-use far exceeds
            # the schema cost of the few unused sessions).
            expanded_sessions=12,
            expanded_calls_each=2,
        )
        # An overlay that a previous shaping left behind: one adaptive resident
        # the current recommendations never mention, and one policy carry tool
        # already demoted out.
        (self.root_state / "learned.json").write_text(
            json.dumps(
                {
                    "version": 2,
                    "scopes": {
                        self.SCOPE: {
                            "carry": ["mnemosyne_recall"],
                            "expand_only": ["read_file"],
                            "shaping": {},
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return configure.discover_scopes(self.home)[0]

class WriteDisclosureTests(TempHomeTestCase):
    """Nothing is written that did not appear in the pre-prompt diff."""

    def _shapeable(self) -> "configure.ScopeInfo":
        seed_telemetry(
            self.root_state,
            "default:telegram",
            sessions=self.needed,
            always_on=["web_search", "read_file"],
            expanded_tool="terminal",
            # Expanded in most sessions so the economic test decisively
            # favors carrying (penalty 1500/session-with-use far exceeds
            # the schema cost of the few unused sessions).
            expanded_sessions=12,
            expanded_calls_each=2,
        )
        return configure.discover_scopes(self.home)[0]

    def _run_until_prompt(self, flow, info) -> list[str]:
        """Drive ``flow`` with a reader that declines, marking the ask."""
        sink: list[str] = []

        def reader(message: str) -> str:
            sink.append(f"<<PROMPT>> {message}")
            return "n"

        ctx = make_ctx(
            self.home,
            FakeRunner(),
            thresholds=self.thresholds,
            out=sink.append,
            reader=reader,
        )
        flow(ctx, [info])
        return sink

    def test_learned_overlay_change_is_shown_before_the_question(self) -> None:
        info = self._shapeable()
        sink = self._run_until_prompt(configure.flow_shape, info)
        blob = "\n".join(sink)
        # Compact disclosure (Dale): counts, not a per-tool wall. The overlay
        # change is still shown, and still before the ask.
        self.assertIn("Tools carried:", blob)
        self.assertIn("Tools available by expansion:", blob)

        overlay_at = next(
            i for i, l in enumerate(sink) if "Tools carried:" in l
        )
        prompt_at = next(i for i, l in enumerate(sink) if l.startswith("<<PROMPT>>"))
        self.assertLess(overlay_at, prompt_at, "the overlay diff must precede the ask")

        # Declining still writes nothing at all.
        self.assertFalse((self.root_state / "learned.json").exists())

    def test_sidecar_write_is_disclosed_on_the_recommend_path(self) -> None:
        info = self._shapeable()
        sink = self._run_until_prompt(configure.flow_recommend, info)
        blob = "\n".join(sink)
        self.assertIn(configure.CONFIGURE_STATE_FILE, blob)
        sidecar_at = next(
            i for i, l in enumerate(sink) if configure.CONFIGURE_STATE_FILE in l
        )
        prompt_at = next(i for i, l in enumerate(sink) if l.startswith("<<PROMPT>>"))
        self.assertLess(sidecar_at, prompt_at)

    def test_disclosed_overlay_equals_the_overlay_written_on_yes(self) -> None:
        info = self._shapeable()
        sink: list[str] = []
        ctx = make_ctx(
            self.home, FakeRunner(), assume_yes=True, thresholds=self.thresholds,
            out=sink.append,
        )
        configure.flow_shape(ctx, [info])
        entry = json.loads((self.root_state / "learned.json").read_text())["scopes"][
            info.scope
        ]
        # The disclosed COUNT equals what was written (compact-diff contract).
        carry_line = next(l for l in sink if "Tools carried:" in l)
        n = len(entry["carry"])
        self.assertTrue(carry_line.rstrip().endswith(f"→ {n}")
                        or carry_line.rstrip().endswith("→ unchanged"),
                        f"disclosed carry count must equal what was written "
                        f"({n}); got {carry_line!r}")


class ApplyFlowTests(TempHomeTestCase):
    def _shapeable_scope(self) -> "configure.ScopeInfo":
        seed_telemetry(
            self.root_state,
            "default:telegram",
            sessions=self.needed,
            always_on=["web_search", "read_file"],
            expanded_tool="terminal",
            # Expanded in most sessions so the economic test decisively
            # favors carrying (penalty 1500/session-with-use far exceeds
            # the schema cost of the few unused sessions).
            expanded_sessions=12,
            expanded_calls_each=2,
        )
        infos = configure.discover_scopes(self.home)
        self.assertEqual(len(infos), 1)
        return infos[0]

    def test_shape_path_writes_config_keys_and_learned_overlay(self) -> None:
        info = self._shapeable_scope()
        runner = FakeRunner()
        ctx = make_ctx(
            self.home,
            runner,
            assume_yes=True,
            thresholds=self.thresholds,
            plugin_config={"channels": {"default:telegram": {"bypass_rate": 1.0, "learned_mode": "recommend"}}},
        )
        self.assertEqual(configure.flow_shape(ctx, [info]), 0)

        emitted = {c[3]: c[4] for c in runner.writes}
        self.assertEqual(
            emitted,
            {
                "plugins.tool-belt.channels.default:telegram.learned_mode": "apply",
                "plugins.tool-belt.channels.default:telegram.bypass_rate": "0.0",
            },
        )
        self.assertTrue(all(c[-1] == "--force" for c in runner.writes))

        learned = json.loads((self.root_state / "learned.json").read_text())
        entry = learned["scopes"]["default:telegram"]
        self.assertIn("terminal", entry["carry"])
        self.assertEqual(entry["shaping"]["scope"], "default:telegram")
        # Canonical v2 keys only — the transitional v1 mirror is gone.
        for stale in ("always_on", "always_off", "cache_aware"):
            self.assertNotIn(stale, entry)
        # Atomic write leaves no temp file behind.
        self.assertEqual(list(self.root_state.glob("learned*.tmp")), [])

    def test_shape_path_targets_the_named_profile_state_dir(self) -> None:
        named_state = self.home / "profiles" / "assistant-a" / "state" / "tool-belt"
        seed_telemetry(
            named_state,
            "assistant-a:slack",
            sessions=self.needed,
            always_on=["web_search"],
            expanded_tool="terminal",
            # Expanded in most sessions so the economic test decisively
            # favors carrying (penalty 1500/session-with-use far exceeds
            # the schema cost of the few unused sessions).
            expanded_sessions=12,
            expanded_calls_each=2,
        )
        info = next(i for i in configure.discover_scopes(self.home) if i.agent == "assistant-a")
        runner = FakeRunner()
        ctx = make_ctx(self.home, runner, assume_yes=True, thresholds=self.thresholds)

        configure.flow_shape(ctx, [info])

        self.assertTrue((named_state / "learned.json").exists())
        self.assertFalse((self.root_state / "learned.json").exists())
        self.assertIn(
            "plugins.tool-belt.channels.assistant-a:slack.learned_mode",
            [c[3] for c in runner.writes],
        )

    def test_recommend_path_sets_observation_mode_and_remembers_prior_bypass(self) -> None:
        seed_telemetry(self.root_state, "default:telegram", sessions=2)
        info = configure.discover_scopes(self.home)[0]
        runner = FakeRunner()
        ctx = make_ctx(
            self.home,
            runner,
            assume_yes=True,
            thresholds=self.thresholds,
            plugin_config={"channels": {"default:telegram": {"bypass_rate": 0.05}}},
        )
        self.assertEqual(configure.flow_recommend(ctx, [info]), 0)

        emitted = {c[3]: c[4] for c in runner.writes}
        self.assertEqual(
            emitted,
            {
                "plugins.tool-belt.channels.default:telegram.learned_mode": "recommend",
                "plugins.tool-belt.channels.default:telegram.bypass_rate": "1.0",
            },
        )
        self.assertEqual(configure.previous_full_ceiling_rate(self.root_state, "default:telegram"), 0.05)

    def test_previous_bypass_defaults_to_narrow_immediately(self) -> None:
        self.assertEqual(
            configure.previous_full_ceiling_rate(self.root_state, "never:seen"), configure.NARROW_BYPASS
        )

    def test_writes_are_skipped_when_confirmation_is_declined(self) -> None:
        info = self._shapeable_scope()
        runner = FakeRunner()
        ctx = make_ctx(
            self.home, runner, thresholds=self.thresholds, reader=lambda _p: "n"
        )
        configure.flow_shape(ctx, [info])
        self.assertEqual(runner.writes, [])
        self.assertFalse((self.root_state / "learned.json").exists())


class ResetFlowTests(TempHomeTestCase):
    def _shaped_scope(self) -> "configure.ScopeInfo":
        seed_telemetry(
            self.root_state,
            "default:telegram",
            sessions=self.needed,
            always_on=["web_search"],
            expanded_tool="terminal",
            # Expanded in most sessions so the economic test decisively
            # favors carrying (penalty 1500/session-with-use far exceeds
            # the schema cost of the few unused sessions).
            expanded_sessions=12,
            expanded_calls_each=2,
        )
        info = configure.discover_scopes(self.home)[0]
        ctx = make_ctx(self.home, FakeRunner(), assume_yes=True, thresholds=self.thresholds)
        configure.flow_shape(ctx, [info])
        return info

    def test_reset_removes_overlay_restores_mode_and_bypass(self) -> None:
        info = self._shaped_scope()
        configure.remember_previous_full_ceiling_rate(self.root_state, info.scope, 0.05)
        runner = FakeRunner()
        ctx = make_ctx(
            self.home,
            runner,
            assume_yes=True,
            thresholds=self.thresholds,
            plugin_config={"channels": {"default:telegram": {"learned_mode": "apply", "bypass_rate": 0.0}}},
        )
        self.assertEqual(configure.flow_reset(ctx, [info]), 0)

        emitted = {c[3]: c[4] for c in runner.writes}
        self.assertEqual(
            emitted,
            {
                "plugins.tool-belt.channels.default:telegram.learned_mode": "recommend",
                "plugins.tool-belt.channels.default:telegram.bypass_rate": "0.05",
            },
        )
        learned = json.loads((self.root_state / "learned.json").read_text())
        self.assertNotIn("default:telegram", learned.get("scopes", {}))
        # The file itself survives; only the scope entry is dropped.
        self.assertTrue((self.root_state / "learned.json").exists())

    def test_reset_without_remembered_bypass_restores_zero(self) -> None:
        info = self._shaped_scope()
        runner = FakeRunner()
        ctx = make_ctx(self.home, runner, assume_yes=True, thresholds=self.thresholds)
        configure.flow_reset(ctx, [info])
        emitted = {c[3]: c[4] for c in runner.writes}
        self.assertEqual(emitted["plugins.tool-belt.channels.default:telegram.bypass_rate"], "0.0")

    def test_reset_leaves_other_scopes_alone(self) -> None:
        info = self._shaped_scope()
        path = self.root_state / "learned.json"
        state = json.loads(path.read_text())
        state["scopes"]["other:cli"] = {"always_on": ["keepme"]}
        path.write_text(json.dumps(state))

        ctx = make_ctx(self.home, FakeRunner(), assume_yes=True, thresholds=self.thresholds)
        configure.flow_reset(ctx, [info])

        learned = json.loads(path.read_text())
        # The untouched scope's assignment survives — normalized to the v2
        # shape on write (learned.write_state normalizes every persist).
        other = learned["scopes"]["other:cli"]
        self.assertEqual(other["carry"], ["keepme"])
        self.assertNotIn("always_on", other)

    def test_reset_clears_only_adaptive_keys_and_preserves_scope_metadata(self) -> None:
        # Regression: reset used to pop the whole scope entry, destroying
        # unrelated per-scope metadata. The single reset semantic
        # (learned.reset_scope) clears ONLY the adaptive carry/expand_only
        # assignments and shaping evidence.
        info = self._shaped_scope()
        path = self.root_state / "learned.json"
        state = json.loads(path.read_text())
        state["scopes"][info.scope]["notes"] = "hand-edited, keep me"
        path.write_text(json.dumps(state))

        ctx = make_ctx(self.home, FakeRunner(), assume_yes=True, thresholds=self.thresholds)
        configure.flow_reset(ctx, [info])

        learned = json.loads(path.read_text())
        entry = learned["scopes"][info.scope]
        # Adaptive assignments/evidence cleared (normalize-on-write may render
        # a cleared field as empty rather than absent — both mean cleared).
        self.assertEqual(entry.get("carry", []), [])
        self.assertEqual(entry.get("expand_only", []), [])
        self.assertEqual(entry.get("shaping", {}), {})
        for stale in ("always_on", "always_off", "cache_aware"):
            self.assertNotIn(stale, entry)
        self.assertEqual(entry["notes"], "hand-edited, keep me",
                         "reset must preserve unrelated per-scope metadata")


class DryRunTests(TempHomeTestCase):
    def _info(self) -> "configure.ScopeInfo":
        seed_telemetry(
            self.root_state,
            "default:telegram",
            sessions=self.needed,
            always_on=["web_search"],
            expanded_tool="terminal",
            # Expanded in most sessions so the economic test decisively
            # favors carrying (penalty 1500/session-with-use far exceeds
            # the schema cost of the few unused sessions).
            expanded_sessions=12,
            expanded_calls_each=2,
        )
        return configure.discover_scopes(self.home)[0]

    def _fs_snapshot(self) -> set[str]:
        return {str(p.relative_to(self.home)) for p in self.home.rglob("*")}

    def test_dry_run_shape_writes_nothing_to_disk_or_subprocess(self) -> None:
        info = self._info()
        before = self._fs_snapshot()
        runner = FakeRunner()
        ctx = make_ctx(self.home, runner, dry_run=True, assume_yes=True, thresholds=self.thresholds)
        configure.flow_shape(ctx, [info])
        self.assertEqual(runner.writes, [])
        self.assertEqual(self._fs_snapshot(), before)
        self.assertFalse((self.root_state / "learned.json").exists())

    def test_dry_run_recommend_writes_nothing(self) -> None:
        info = self._info()
        before = self._fs_snapshot()
        runner = FakeRunner()
        ctx = make_ctx(self.home, runner, dry_run=True, assume_yes=True, thresholds=self.thresholds)
        configure.flow_recommend(ctx, [info])
        self.assertEqual(runner.writes, [])
        self.assertEqual(self._fs_snapshot(), before)

    def test_dry_run_reset_leaves_the_overlay_intact(self) -> None:
        info = self._info()
        configure.flow_shape(
            make_ctx(self.home, FakeRunner(), assume_yes=True, thresholds=self.thresholds), [info]
        )
        overlay = (self.root_state / "learned.json").read_text()
        runner = FakeRunner()
        ctx = make_ctx(self.home, runner, dry_run=True, assume_yes=True, thresholds=self.thresholds)
        configure.flow_reset(ctx, [info])
        self.assertEqual(runner.writes, [])
        self.assertEqual((self.root_state / "learned.json").read_text(), overlay)

    def test_status_never_writes(self) -> None:
        self._info()
        before = self._fs_snapshot()
        runner = FakeRunner()
        ctx = make_ctx(self.home, runner, thresholds=self.thresholds)
        infos = configure.discover_scopes(self.home)
        self.assertEqual(configure.flow_status(ctx, infos), 0)
        self.assertEqual(runner.writes, [])
        self.assertEqual(self._fs_snapshot(), before)


class FreshInstallFrontDoorTests(TempHomeTestCase):
    """The very first command a new user runs, on a home with no telemetry.

    Regression: profiles plainly present, ``discover_scopes`` empty, and the
    user was told "No Hermes profiles found" and given nothing.
    """

    ABSENT = "No Hermes profiles found"

    def _run(self, argv: list[str], answers: list[str] | None = None):
        runner = FakeRunner()
        lines: list[str] = []
        replies = iter(answers or [])
        with contextlib.ExitStack() as stack:
            for patch in isolate(runner, "/usr/bin/hermes", lines):
                stack.enter_context(patch)
            # RunContext binds ``_default_reader`` as a dataclass default, so
            # the interception has to happen one level down, at ``input``.
            stack.enter_context(mock.patch("builtins.input", lambda _p="": next(replies)))
            rc = configure.main(argv + ["--hermes-home", str(self.home)])
        return rc, "\n".join(lines), runner

    def test_profiles_without_telemetry_are_never_reported_absent(self) -> None:
        # self.home has state/tool-belt (so a profile is discoverable) but no
        # predictions.jsonl at all — a genuinely fresh install.
        self.assertEqual(configure.discover_scopes(self.home), [])
        self.assertTrue(configure.discover_state_dirs(self.home))

        rc, output, runner = self._run(["--yes"])
        self.assertEqual(rc, 0)
        self.assertNotIn(self.ABSENT, output)
        self.assertIn("Hermes profile(s) found: default", output)
        self.assertIn("No Tool Belt telemetry has been recorded", output)
        self.assertEqual(runner.writes, [])

    def test_the_fresh_install_is_told_what_to_expect(self) -> None:
        _rc, output, _runner = self._run(["--yes"])
        self.assertIn("What to expect", output)
        self.assertIn("one row per gateway session", output)
        self.assertIn(f"{self.needed} recorded session(s)", output)
        self.assertIn("--status", output)
        self.assertIn("--platform", output)

    def test_the_platform_prompt_is_reachable_without_agent(self) -> None:
        # No --agent, no --platform: the recovery must still be offered, and
        # naming a platform must produce a configurable scope.
        rc, output, runner = self._run(["--path", "recommend"], answers=["telegram", "y"])
        self.assertEqual(rc, 0)
        self.assertNotIn(self.ABSENT, output)
        self.assertIn("default:telegram", output)
        self.assertEqual(
            {c[3]: c[4] for c in runner.writes},
            {
                "plugins.tool-belt.channels.default:telegram.learned_mode": "recommend",
                "plugins.tool-belt.channels.default:telegram.bypass_rate": "1.0",
            },
        )

    def test_an_empty_answer_falls_through_to_the_same_guidance(self) -> None:
        rc, output, runner = self._run(["--path", "recommend"], answers=[""])
        self.assertEqual(rc, 0)
        self.assertNotIn(self.ABSENT, output)
        self.assertIn("What to expect", output)
        self.assertEqual(runner.writes, [])

    def test_a_home_with_no_profiles_still_says_so(self) -> None:
        empty = Path(self.tmp.name) / "empty home"
        empty.mkdir()
        runner = FakeRunner()
        lines: list[str] = []
        with contextlib.ExitStack() as stack:
            for patch in isolate(runner, "/usr/bin/hermes", lines):
                stack.enter_context(patch)
            rc = configure.main(["--yes", "--hermes-home", str(empty)])
        output = "\n".join(lines)
        self.assertEqual(rc, 0)
        self.assertIn(self.ABSENT, output)
        self.assertEqual(runner.writes, [])


class DegradedModeTests(TempHomeTestCase):
    def setUp(self) -> None:
        super().setUp()
        seed_telemetry(
            self.root_state,
            "default:telegram",
            sessions=self.needed,
            always_on=["web_search"],
            expanded_tool="terminal",
            # Expanded in most sessions so the economic test decisively
            # favors carrying (penalty 1500/session-with-use far exceeds
            # the schema cost of the few unused sessions).
            expanded_sessions=12,
            expanded_calls_each=2,
        )

    def test_missing_hermes_prints_manual_commands_and_writes_nothing(self) -> None:
        runner = FakeRunner()
        lines: list[str] = []
        before = {str(p.relative_to(self.home)) for p in self.home.rglob("*")}
        with contextlib.ExitStack() as stack:
            for patch in isolate(runner, None, lines):
                stack.enter_context(patch)
            rc = configure.main(
                ["--agent", "default", "--path", "recommend", "--yes", "--hermes-home", str(self.home)]
            )
        output = "\n".join(lines)
        self.assertEqual(rc, 0)
        self.assertEqual(runner.calls, [], "no subprocess should be spawned without hermes")
        self.assertIn("hermes config set", output)
        self.assertIn("not on PATH", output)
        self.assertEqual({str(p.relative_to(self.home)) for p in self.home.rglob("*")}, before)

    def test_hermes_available_uses_the_injected_lookup(self) -> None:
        self.assertTrue(configure.hermes_available(lambda _n: "/usr/bin/hermes"))
        self.assertFalse(configure.hermes_available(lambda _n: None))


class ConfigReadTests(unittest.TestCase):
    def test_not_set_sentinel_reads_as_none(self) -> None:
        runner = FakeRunner()
        self.assertIsNone(configure.hermes_config_get("plugins.tool-belt.nope", runner=runner))

    def test_scalar_value_is_returned_verbatim(self) -> None:
        runner = FakeRunner({"plugins.tool-belt.learned_mode": "apply"})
        self.assertEqual(
            configure.hermes_config_get("plugins.tool-belt.learned_mode", runner=runner), "apply"
        )

    def test_nonzero_returncode_reads_as_none(self) -> None:
        runner = FakeRunner({"plugins.tool-belt.learned_mode": "apply"}, returncode=2)
        self.assertIsNone(
            configure.hermes_config_get("plugins.tool-belt.learned_mode", runner=runner)
        )

    def test_block_is_parsed_into_a_dict(self) -> None:
        runner = FakeRunner(
            {
                "plugins.tool-belt": (
                    "enabled: true\nlearned_mode: recommend\n"
                    "channels:\n  default:telegram:\n    bypass_rate: 1.0\n"
                )
            }
        )
        cfg = configure.read_plugin_config(runner)
        self.assertEqual(cfg.get("learned_mode"), "recommend")
        self.assertEqual(cfg["channels"]["default:telegram"]["bypass_rate"], 1.0)

    def test_scope_settings_falls_back_to_scalar_reads(self) -> None:
        runner = FakeRunner(
            {
                "plugins.tool-belt.channels.default:telegram.bypass_rate": "1.0",
                "plugins.tool-belt.learned_mode": "recommend",
            }
        )
        settings = configure.scope_settings("default:telegram", {}, runner=runner)
        self.assertEqual(settings["scope_bypass_rate"], 1.0)
        self.assertEqual(settings["learned_mode"], "recommend")
        self.assertIsNone(settings["scope_learned_mode"])


class PromptTests(unittest.TestCase):
    def test_eof_aborts_cleanly(self) -> None:
        def reader(_prompt: str) -> str:
            raise EOFError()

        with self.assertRaises(configure.Abort):
            configure.prompt("? ", reader)

    def test_keyboard_interrupt_aborts_cleanly(self) -> None:
        def reader(_prompt: str) -> str:
            raise KeyboardInterrupt()

        with self.assertRaises(configure.Abort):
            configure.prompt("? ", reader)

    def test_invalid_choice_reprompts(self) -> None:
        answers = iter(["maybe", "", "y"])
        with mock.patch("builtins.print"):
            self.assertTrue(configure.confirm("ok?", lambda _p: next(answers)))

    def test_multi_select_accepts_numbers_and_all(self) -> None:
        infos = [
            configure.ScopeInfo(scope=f"a{i}:cli", agent=f"a{i}", platform="cli", state_dir=Path("/tmp"))
            for i in range(3)
        ]
        with mock.patch("builtins.print"):
            self.assertEqual(
                [i.scope for i in configure.prompt_multi_select(infos, lambda _p: "1,3")],
                ["a0:cli", "a2:cli"],
            )
            self.assertEqual(len(configure.prompt_multi_select(infos, lambda _p: "all")), 3)

    def test_multi_select_reprompts_on_garbage(self) -> None:
        infos = [
            configure.ScopeInfo(scope=f"a{i}:cli", agent=f"a{i}", platform="cli", state_dir=Path("/tmp"))
            for i in range(2)
        ]
        answers = iter(["nope", "9", "2"])
        with mock.patch("builtins.print"):
            self.assertEqual(
                [i.scope for i in configure.prompt_multi_select(infos, lambda _p: next(answers))],
                ["a1:cli"],
            )

    def test_single_scope_needs_no_selection(self) -> None:
        infos = [configure.ScopeInfo(scope="a:cli", agent="a", platform="cli", state_dir=Path("/tmp"))]

        def reader(_prompt: str) -> str:
            raise AssertionError("should not prompt for a single scope")

        self.assertEqual(configure.prompt_multi_select(infos, reader), infos)


if __name__ == "__main__":
    unittest.main()
