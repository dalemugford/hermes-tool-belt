#!/usr/bin/env bash
# Scheduled twice-daily run of the tool-belt analyzer.
#
# Walks every tool-belt telemetry source on this host — the root
# state dir plus each profile-scoped state dir under HERMES_HOME/profiles/*/
# — and runs analyze.py against each one that has data. Each source gets
# its own dated markdown report (under reports/<label>/) and its own
# line in the consolidated daily-summary.log, prefixed with the source
# label so a week of runs is still scannable with one `cat`.
#
# Safe to run by hand or from the scheduler appropriate to the host.
set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PYTHON="${HERMES_PYTHON:-python3}"

# Logs are always written to the root state dir, regardless of which
# source the run is analyzing — keeps the summary log consolidated.
ROOT_LOG_DIR="${HERMES_HOME}/state/tool-belt/cron-logs"
SUMMARY_LOG="${ROOT_LOG_DIR}/daily-summary.log"
mkdir -p "${ROOT_LOG_DIR}"

TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TS_LOCAL="$(date +%Y-%m-%d-%H%M%S)"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "${TS_UTC}  error  python interpreter not found: ${PYTHON}; set HERMES_PYTHON" \
        | tee -a "${SUMMARY_LOG}" >&2
    exit 1
fi

# Build the list of (label, state_dir) pairs to process.
SOURCES=()
SOURCES+=("default:${HERMES_HOME}/state/tool-belt")
shopt -s nullglob
for profile_state in "${HERMES_HOME}"/profiles/*/state/tool-belt; do
    # Extract <name> from <home>/profiles/<name>/state/tool-belt
    profile_name="$(basename "$(dirname "$(dirname "${profile_state}")")")"
    if [[ "${profile_name}" == "default" ]]; then
        continue  # reserved by Hermes for the root profile
    fi
    SOURCES+=("${profile_name}:${profile_state}")
done
shopt -u nullglob

overall_rc=0
ran_any=0

for entry in "${SOURCES[@]}"; do
    label="${entry%%:*}"
    state_dir="${entry#*:}"

    # Skip cleanly if there's no telemetry here yet. Avoids noise in the
    # log during the first few days after a rotate, or for profiles that
    # haven't received any messages.
    if [[ ! -s "${state_dir}/predictions.jsonl" ]]; then
        echo "${TS_UTC}  [${label}]  no_telemetry  predictions.jsonl is empty or missing — skipping" \
            | tee -a "${SUMMARY_LOG}" >&2
        continue
    fi

    ran_any=1

    JSON_OUT="${ROOT_LOG_DIR}/${TS_LOCAL}-${label}.json"
    STDERR_LOG="${ROOT_LOG_DIR}/${TS_LOCAL}-${label}.stderr"
    REPORTS_DIR="${PLUGIN_DIR}/reports/${label}"

    if "${PYTHON}" "${PLUGIN_DIR}/analyze.py" \
            --state-dir "${state_dir}" \
            --reports-dir "${REPORTS_DIR}" \
            --format json \
            --suggest-dampeners \
            --write-recommendations \
            > "${JSON_OUT}" \
            2> "${STDERR_LOG}"; then
        :
    else
        rc=$?
        echo "${TS_UTC}  [${label}]  error  analyzer exited rc=${rc}; see ${STDERR_LOG}" \
            | tee -a "${SUMMARY_LOG}" >&2
        overall_rc=$rc
        continue
    fi

    # Pull one-line summary fields out of the JSON for the running log.
    if command -v jq >/dev/null 2>&1; then
        line=$(jq -r --arg ts "${TS_UTC}" --arg label "${label}" '
            "\($ts)  [\($label)]  ok  " +
            "predictions=\(.totals.prediction_rows) " +
            "scopes=\(.totals.scopes) " +
            "expand_events=\(.totals.expand_tools_events) " +
            "expand_success_rate=\(.totals.expansion_success_rate) " +
            "net_savings=\(.totals.estimated_net_savings_tokens) " +
            "recs=\(.totals.recommendation_candidates) " +
            "dampener_candidates=\(.dampener_candidates | length)"
        ' "${JSON_OUT}")
    else
        line=$("${PYTHON}" - "${JSON_OUT}" "${TS_UTC}" "${label}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
ts, label = sys.argv[2], sys.argv[3]
t = data["totals"]
print(f"{ts}  [{label}]  ok  predictions={t['prediction_rows']} scopes={t['scopes']} "
      f"expand_events={t['expand_tools_events']} "
      f"expand_success_rate={t['expansion_success_rate']} "
      f"net_savings={t['estimated_net_savings_tokens']} "
      f"recs={t['recommendation_candidates']} "
      f"dampener_candidates={len(data.get('dampener_candidates', []))}")
PY
        )
    fi

    echo "${line}" | tee -a "${SUMMARY_LOG}"

    SHAPE_LOG="${ROOT_LOG_DIR}/${TS_LOCAL}-${label}.shape.log"
    if "${PYTHON}" "${PLUGIN_DIR}/scripts/shape-ceiling.py" \
            --state-dir "${state_dir}" \
            > "${SHAPE_LOG}" \
            2>&1; then
        :
    else
        rc=$?
        echo "${TS_UTC}  [${label}]  shape_error  shaper exited rc=${rc}; see ${SHAPE_LOG}" \
            | tee -a "${SUMMARY_LOG}" >&2
        overall_rc=$rc
        continue
    fi

    if grep -q "Wrote updated recommendations" "${SHAPE_LOG}"; then
        echo "${TS_UTC}  [${label}]  shape_ok  learned.json updated" \
            | tee -a "${SUMMARY_LOG}"
    else
        echo "${TS_UTC}  [${label}]  shape_ok  learned.json unchanged" \
            | tee -a "${SUMMARY_LOG}"
    fi

    # On success the stderr log only holds the analyzer's "report:" /
    # "recommendations:" info lines (analyzer routes those to stderr when
    # --format json so stdout stays parseable). Drop it to keep the
    # directory readable — a real failure exits non-zero above and stderr
    # is preserved. Keep the shaper log only when it actually wrote or
    # failed, so normal no-op runs stay tidy while first writes are auditable.
    rm -f "${STDERR_LOG}"
    if ! grep -q "Wrote updated recommendations" "${SHAPE_LOG}"; then
        rm -f "${SHAPE_LOG}"
    fi
done

if [[ "${ran_any}" -eq 0 ]]; then
    # Every source was empty. Don't pretend this was a run.
    exit 0
fi

exit "${overall_rc}"
