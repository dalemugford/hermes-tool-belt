# Releasing

The checklist a maintainer follows to cut a Tool Belt release. Versions use
CalVer — `YYYY.M.D`, with an optional prerelease suffix (for example
`2026.5.17-beta`). The authoritative version is the `version` field in
[`plugin.yaml`](../plugin.yaml); the release tag is the same string.

Verification conventions are set in [CONTRIBUTING.md](../CONTRIBUTING.md#verification)
and [AGENTS.md](../AGENTS.md#verification) — this document sequences them for a
release rather than restating them. Release discipline itself (develop on
`main`, tag stable snapshots, keep experimental behavior disabled by config)
comes from [AGENTS.md](../AGENTS.md#branch-and-release-discipline).

Read the output of every command. A zero exit status is not by itself a pass —
two of the checks below report "nothing to validate" and exit 0.

---

## 1. Pre-release verification

Run this on a **clean clone**, not your working tree, so uncommitted files and
stale bytecode can't mask a failure.

```bash
git clone https://github.com/dalemugford/hermes-tool-belt.git /tmp/tool-belt-release
cd /tmp/tool-belt-release
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

Full quality gate:

```bash
.venv/bin/python tests/run_tests.py
.venv/bin/python scripts/smoke-test.py
.venv/bin/python -m compileall -q .
bash -n scripts/rotate-telemetry.sh
```

Expected:

| Command | Expected result |
| --- | --- |
| `tests/run_tests.py` | `OK` — 0 failures, 0 errors. Several skips are expected on a clean clone outside the Hermes venv: every case gated on `hermes_cli` or `tools.tool_search` not being importable (about a dozen — the real-bridge tests and `configure.py`'s curses-contract tests), plus a `tiktoken`-gated case when that package isn't installed. Any other `skipped` line means something changed and needs explaining. |
| `scripts/smoke-test.py` | Two blocks: `9/9 checks passed` (cache-off / narrowing & attribution) and `16/16 checks passed` (cache-on / carry-all contract). |
| `compileall -q .` | No output, exit 0. |
| `bash -n scripts/rotate-telemetry.sh` | No output, exit 0. |

The smoke test prints repeated `tool-belt: cannot import run_agent` lines from
its synthetic fixtures. That is expected fixture noise, not a failure — only
the `N/N checks passed` lines decide the result.

Then the runtime contract check, which needs the full Hermes runtime:

```bash
hermes plugins doctor --ci .
```

Expected — the manifest version, no warnings, and exactly the declared surface:

```
  OK: runtime discovery, manifest parsing, import, and registration passed
  registrations: 1 tool(s), 5 hook(s)
```

`1 tool` is `expand_tools`; the `5 hooks` are the five entries under
`provides_hooks` in `plugin.yaml`. A count mismatch means a registration
regressed against the manifest.

---

## 2. Clean-profile install test

> **Hard rule: never run the install test against your live profile.** Point
> `HERMES_HOME` at a throwaway directory for every command in this section and
> the next. The Hermes CLI honors `HERMES_HOME` and scaffolds a fresh profile
> there; installing into your real profile overwrites the plugin you are
> testing and pollutes the telemetry the drift check reads.

```bash
export HERMES_HOME=$(mktemp -d)
echo "$HERMES_HOME"
```

`hermes plugins install` accepts a Git URL, an `owner/repo` shorthand, or an
index name — **not a local filesystem path**. Install the pushed commit:

```bash
hermes plugins install dalemugford/hermes-tool-belt
```

To test an exact commit before it is tagged, pin it:

```bash
hermes plugins install dalemugford/hermes-tool-belt --ref <40-char-commit-sha>
```

To test a local clone that has not been pushed at all, copy it into the temp
profile instead of using the installer:

```bash
cp -R /tmp/tool-belt-release "$HERMES_HOME/plugins/tool-belt"
```

Confirm registration and a fresh read-only state report:

```bash
hermes plugins list
hermes plugins doctor --ci "$HERMES_HOME/plugins/tool-belt"
python3 "$HERMES_HOME/plugins/tool-belt/scripts/configure.py" --status
```

Expected: `tool-belt` appears in the plugin list at the release version;
`--status` prints `Tool Belt: enabled`, names the discovered profiles, and
reports that no telemetry has been recorded yet — and writes nothing. Re-running `--status` must produce identical output — it is
read-only by contract.

Optionally send one test message through the temp profile and confirm a row
lands in telemetry:

```bash
tail -1 "$HERMES_HOME/state/tool-belt/predictions.jsonl" | python3 -m json.tool
```

---

## 3. Behavioral spot-checks

Still under the temp `HERMES_HOME` from step 2. Each check reads
`predictions.jsonl`; the field reference is in
[CONFIGURATION.md](CONFIGURATION.md#telemetry-outputs).

A convenient view of the rows as you go:

```bash
tail -f "$HERMES_HOME/state/tool-belt/predictions.jsonl" | \
  jq -c '{scope, policy_source, ceiling_count, narrowed_count, tokens_saved, tool_list_hash, triggers: .triggers_fired}'
```

1. **Cache-on: carry-all loadout is stable across turns.** On a prefix-caching
   model, send several turns in one session. Every dispatch writes
   `policy_source: "cache_on_carry_all"` with `ceiling_count == narrowed_count`
   and `tokens_saved: 0`, and `tool_list_hash` is byte-identical across every
   turn of the session (`expand_tools` is absent from the shipped tool list
   throughout). Note: the older `frozen_reuse` / `frozen_reuse_count` fields
   still exist in the schema but are vestigial — carry-all rows never set
   them, so they stay `false` / `0`; don't use them to judge this check.

2. **Cache-off: per-turn narrowing and sticky residency.** On a non-caching
   model, `tool_list_hash` changes as intent changes. After an `expand_tools`
   call, the expanded tools persist for the sticky-residency window, then drop.

3. **`/new` or `/reset` starts a fresh carry-all session.** Issue `/new`, then
   send a message. Expect a new `tool_list_hash` for the new session (still
   stable turn-to-turn within it) and `ceiling_count == narrowed_count` again
   from the first dispatch.

4. **`expand_tools` recovery, both forms.** Confirm the model can recover by
   category and by individual tool name, and that the recovered tools appear in
   the next dispatch's tool list:

   ```
   expand_tools(category="browser")
   expand_tools(tool="browser_navigate")
   ```

5. **`bypass_rate: 1.0` ships the full ceiling.** Set it on one scope, restart
   the gateway, send a message. Expect the full configured ceiling in the
   payload and `policy_source: bypass` in the row. Telemetry keeps flowing.

6. **`enabled: false` turns the plugin fully off.** Restart, send a message,
   confirm no narrowing and no new telemetry rows.

7. **`log: false` stops telemetry but not narrowing.** Restart, send a message,
   confirm the tool list is still narrowed and no new row is appended.

8. **Analyzer runs and writes a report.**

   ```bash
   python3 "$HERMES_HOME/plugins/tool-belt/analyze.py"
   ```

   Expect a summary on stdout and a new `reports/YYYY-MM-DD-HHMMSS-analysis.md`
   under the plugin directory.

9. **Shaper dry-run is clean.**

   ```bash
   python3 "$HERMES_HOME/plugins/tool-belt/scripts/shape-ceiling.py" --dry-run
   ```

   Expect per-scope promote/demote candidates and a closing
   `[dry-run] …` line — either `Recommendations differ from learned.json;
   nothing was written.` or `No changes — recommendations match current
   learned.json content.` Either is a pass; anything else is not.

10. **Disable and uninstall leave nothing behind.** Set `enabled: false`, then
    remove the directory and confirm Hermes starts clean without it:

    ```bash
    rm -rf "$HERMES_HOME/plugins/tool-belt"
    hermes plugins list
    ```

Tear the temp profile down when finished, and make sure the variable is gone
before you touch your real profile again:

```bash
rm -rf "$HERMES_HOME"
unset HERMES_HOME
```

---

## 4. Documentation reconciliation

- **Install commands.** The commands in [README.md](../README.md#install-and-configure)
  still match current Hermes Agent behavior — check them against the
  [Hermes Agent docs](https://hermes-agent.nousresearch.com/docs) and against
  `hermes plugins install --help`.
- **Changelog completeness.** Every user-facing change since the last tag has an
  `Unreleased` entry. `git log --oneline <last-tag>..HEAD` is the cross-check;
  CONTRIBUTING.md requires an entry per user-facing change, so an unlisted
  behavioral commit is a gap.
- **Companion docs resolve.** Every relative link in the repo's markdown points
  at a file that exists:

  ```bash
  python3 - <<'PY'
  import re, pathlib
  missing = []
  for md in pathlib.Path('.').glob('**/*.md'):
      if '.venv' in md.parts or '.git' in md.parts:
          continue
      for target in re.findall(r'\]\(([^)]+)\)', md.read_text()):
          if target.startswith(('http://', 'https://', 'mailto:')):
              continue
          path = target.split('#')[0]
          if path and not (md.parent / path).exists():
              missing.append(f"{md}: {target}")
  print('\n'.join(missing) or 'all relative links resolve')
  PY
  ```

- **Configuration defaults.** Defaults in
  [CONFIGURATION.md](CONFIGURATION.md) match [`policy.yaml`](../policy.yaml) and
  the constants in the code. `learned_mode` in particular defaults to
  `apply` (full-start): shaping is on unless a scope opts out to
  `recommend`.
- **Known issues are still true.** Re-read [KNOWN_ISSUES.md](KNOWN_ISSUES.md)
  against the models and providers you actually run. Remove observations that no
  longer reproduce — a stale entry blamed on a provider that has since fixed it
  is worse than no entry.

---

## 5. Version and changelog finalization

1. Bump `version` in [`plugin.yaml`](../plugin.yaml) to the release CalVer —
   the date you are cutting on, `YYYY.M.D`, with no zero padding
   (`2026.8.27`, not `2026.08.27`). Add a suffix for a prerelease
   (`2026.8.27-beta`).

2. In [CHANGELOG.md](../CHANGELOG.md), retitle the `## [Unreleased]` heading to
   `## [YYYY.M.D] - YYYY-MM-DD` (ISO date, zero-padded) and open a fresh empty
   `## [Unreleased]` above it:

   ```markdown
   ## [Unreleased]

   ## [2026.8.27] - 2026-08-27
   ```

3. Confirm the two agree before committing:

   ```bash
   grep '^version:' plugin.yaml
   grep -m1 '^## \[' CHANGELOG.md
   ```

4. Commit:

   ```bash
   git commit -am "chore: release 2026.8.27"
   ```

---

## 6. Tag and publish

The tag string is the `plugin.yaml` version **verbatim — no `v` prefix**. The
existing `2026.5.17-beta` tag sets this convention, and README's Releases
section promises a tag matching `plugin.yaml`.

```bash
git push origin main                       # land the release commit first
git tag -a 2026.8.27 -m "Release 2026.8.27"
git push origin 2026.8.27                   # triggers .github/workflows/release.yml
```

Pushing the tag is the publish step. [`release.yml`](../.github/workflows/release.yml)
fires on any CalVer tag, re-runs the gate (`tests/run_tests.py` +
`scripts/smoke-test.py` on 3.11 and 3.12) against the tagged commit, **hard-fails
if the tag string does not equal `plugin.yaml`'s `version`**, extracts that
version's `CHANGELOG.md` section as the notes, and publishes the GitHub Release
(marked pre-release when the tag carries a suffix). No manual `gh release create`
is needed on the happy path.

Watch it land:

```bash
gh run watch "$(gh run list --workflow release.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
```

If the workflow is unavailable (or you need to re-cut notes by hand), publish
manually — the same version guard applies, so confirm the tag matches
`plugin.yaml` first:

```bash
awk '/^## \[2026\.8\.27\]/{f=1; next} f && /^## \[/{exit} f' CHANGELOG.md > /tmp/release-notes.md
gh release create 2026.8.27 --title "2026.8.27" --notes-file /tmp/release-notes.md
```

Read `/tmp/release-notes.md` before publishing. The `awk` stops at the next
`## [` heading and handles the case where the new section is the last one in
the file; a naive `sed` range does not.

Announce per project conventions, if any apply.

---

## 7. Post-release

- **Tag matches the manifest.** The version recorded at the tag is the released
  version:

  ```bash
  git show 2026.8.27:plugin.yaml | grep '^version:'
  ```

- **CI is green on the tagged commit.** [`ci.yml`](../.github/workflows/ci.yml)
  runs `tests/run_tests.py` and `scripts/smoke-test.py` on Python 3.11 and 3.12
  for pushes to `main` (and [`release.yml`](../.github/workflows/release.yml)
  re-runs both on the tag). Neither runs `hermes plugins doctor` — that check
  has no CI coverage and only happened because you ran it in step 1.

  ```bash
  gh run list --branch main --limit 3
  ```

- **Install the tag from scratch.** Repeat the step 2 clean-profile install
  against the published tag, confirming the version reported by
  `hermes plugins list` is the one you just cut.

- **Reformat the published release note.** `release.yml` publishes the raw
  `## [x.y.z]` CHANGELOG section as the note; edit it into the standing
  format — a 2–3 sentence summary paragraph, then `## ✨ Highlights`, then
  `## 🔧 Core changes` (`### Added` / `### Changed` / `### Fixed`). Draft
  the body and apply it with `gh release edit <tag> --notes-file <file>`.

- **Bump the community plugin index `ref`.** The entry in
  `NousResearch/hermes-agent`'s `hermes_cli/data/plugin_index.json` pins a
  specific commit SHA, not a branch — so a release the index should serve
  needs a follow-up PR (or a push to the open one) setting `ref` to the new
  release commit:

  ```bash
  git rev-list -n1 <tag>   # the 40-char SHA to pin
  ```

  Update the entry's `ref`, and the PR body if it cites the SHA. Until this
  lands, `hermes plugins install tool-belt` (by index name) resolves to the
  previously pinned commit; installing `dalemugford/hermes-tool-belt` directly
  still clones `main`.

- **Update any other external pointers.** Marketplace listing or docs-site
  references, where those are maintained.
