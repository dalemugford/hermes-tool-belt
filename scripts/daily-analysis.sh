#!/usr/bin/env bash
# Scheduled twice-daily run of the dynamic-tools analyzer.
#
# Runs `analyze.py` with full recommendations + dampener mining, drops a
# timestamped markdown report under the plugin's reports/ dir, and
# appends a single-line summary to a running log so a week of runs is
# scannable with one `cat`.
#
# Designed to be invoked by launchd (see com.dalemugford.hermes.dynamic-tools-analyzer.plist)
# but safe to run by hand any time.
set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
STATE_DIR="${HERMES_STATE_DIR:-${HERMES_HOME}/state/dynamic-tools}"
PYTHON="${HERMES_PYTHON:-${HERMES_HOME}/hermes-agent/venv/bin/python3}"
LOG_DIR="${STATE_DIR}/cron-logs"
SUMMARY_LOG="${LOG_DIR}/daily-summary.log"

mkdir -p "${LOG_DIR}"

TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TS_LOCAL="$(date +%Y-%m-%d-%H%M%S)"
STDERR_LOG="${LOG_DIR}/${TS_LOCAL}.stderr"

# Skip cleanly if there's no telemetry to analyze yet. Avoids noise in the
# log during the first few days after a rotate when the gateway hasn't
# accumulated predictions in this window.
if [[ ! -s "${STATE_DIR}/predictions.jsonl" ]]; then
    echo "${TS_UTC}  no_telemetry  predictions.jsonl is empty or missing — skipping" \
        | tee -a "${SUMMARY_LOG}" >&2
    exit 0
fi

if [[ ! -x "${PYTHON}" ]]; then
    echo "${TS_UTC}  error  python not executable at ${PYTHON}" | tee -a "${SUMMARY_LOG}" >&2
    exit 1
fi

# Run analyzer with full feature set. JSON to stdout for capture; markdown
# report + recommendations JSON written to disk as side effects.
JSON_OUT="${LOG_DIR}/${TS_LOCAL}.json"

if ! "${PYTHON}" "${PLUGIN_DIR}/analyze.py" \
        --state-dir "${STATE_DIR}" \
        --format json \
        --suggest-dampeners \
        --write-recommendations \
        > "${JSON_OUT}" \
        2> "${STDERR_LOG}"; then
    rc=$?
    echo "${TS_UTC}  error  analyzer exited rc=${rc}; see ${STDERR_LOG}" | tee -a "${SUMMARY_LOG}" >&2
    exit "${rc}"
fi

# Pull one-line summary fields out of the JSON for the running log.
# Use jq if available; fall back to python.
if command -v jq >/dev/null 2>&1; then
    line=$(jq -r --arg ts "${TS_UTC}" '
        "\($ts)  ok  " +
        "predictions=\(.totals.prediction_rows) " +
        "scopes=\(.totals.scopes) " +
        "expand_events=\(.totals.expand_tools_events) " +
        "expand_success_rate=\(.totals.expansion_success_rate) " +
        "net_savings=\(.totals.estimated_net_savings_tokens) " +
        "recs=\(.totals.recommendation_candidates) " +
        "dampener_candidates=\(.dampener_candidates | length)"
    ' "${JSON_OUT}")
else
    line=$("${PYTHON}" - "${JSON_OUT}" "${TS_UTC}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
ts = sys.argv[2]
t = data["totals"]
print(f"{ts}  ok  predictions={t['prediction_rows']} scopes={t['scopes']} "
      f"expand_events={t['expand_tools_events']} "
      f"expand_success_rate={t['expansion_success_rate']} "
      f"net_savings={t['estimated_net_savings_tokens']} "
      f"recs={t['recommendation_candidates']} "
      f"dampener_candidates={len(data.get('dampener_candidates', []))}")
PY
    )
fi

echo "${line}" | tee -a "${SUMMARY_LOG}"

# On success the stderr log only holds the analyzer's "report:" /
# "recommendations:" info lines (analyzer routes those to stderr when
# --format json so stdout stays parseable). Drop it to keep the
# directory readable — a real failure exits non-zero above and stderr
# is preserved.
rm -f "${STDERR_LOG}"
