#!/usr/bin/env bash
# Rotate dynamic-tools telemetry to start a clean measurement window.
#
# Moves current predictions.jsonl + tool_calls.jsonl into
# state-dir/archive/reset-<ts>/. Safe to run while the Hermes gateway
# is up: logger_io opens append-mode on every write, so the next event
# transparently creates a fresh file at the original path.
#
# Optional argument: a tag to suffix the archive folder
# (e.g. ./rotate-telemetry.sh pre-simplification).
set -euo pipefail

STATE_DIR="${HERMES_STATE_DIR:-${HERMES_HOME:-$HOME/.hermes}/state/dynamic-tools}"
TAG="${1:-}"
TS="$(date -u +%Y-%m-%d-%H%M%S)"
ARCHIVE_NAME="reset-${TS}${TAG:+-${TAG}}"
ARCHIVE_DIR="${STATE_DIR}/archive/${ARCHIVE_NAME}"

if [[ ! -d "${STATE_DIR}" ]]; then
    echo "rotate-telemetry: state dir not found: ${STATE_DIR}" >&2
    exit 1
fi

mkdir -p "${ARCHIVE_DIR}"

moved=0
for name in predictions.jsonl tool_calls.jsonl; do
    src="${STATE_DIR}/${name}"
    if [[ -f "${src}" ]]; then
        mv "${src}" "${ARCHIVE_DIR}/${name}"
        echo "  archived: ${name} -> ${ARCHIVE_DIR}/${name}"
        moved=$((moved + 1))
    fi
done

if [[ "${moved}" -eq 0 ]]; then
    echo "rotate-telemetry: nothing to rotate (no live JSONL files in ${STATE_DIR})"
    # Clean up empty archive dir
    rmdir "${ARCHIVE_DIR}" 2>/dev/null || true
    exit 0
fi

echo "rotate-telemetry: ${moved} file(s) archived under ${ARCHIVE_DIR}"
echo "rotate-telemetry: live JSONL files will be recreated on next plugin write"
