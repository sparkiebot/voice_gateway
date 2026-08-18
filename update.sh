#!/usr/bin/env bash
set -Eeuo pipefail

readonly project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly remote="${VOICE_GATEWAY_UPDATE_REMOTE:-origin}"
readonly branch="${VOICE_GATEWAY_UPDATE_BRANCH:-main}"
readonly service="${VOICE_GATEWAY_SYSTEMD_SERVICE:-voice-gateway.service}"
readonly secret_file="${project_dir}/runtime/secret.env"

fail() {
    echo "[voice-gateway-update] ERROR: $*" >&2
    exit 1
}

cd "${project_dir}"
[[ -d .git ]] || fail "${project_dir} is not a Git checkout"
[[ -x runtime/venv/bin/python ]] || fail "runtime/venv/bin/python is missing"
[[ -f "${secret_file}" ]] || fail "runtime/secret.env is missing"
[[ -f /etc/sparkie/voice-gateway.json ]] || fail "/etc/sparkie/voice-gateway.json is missing"

if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    fail "the source checkout has local changes; commit or remove them first"
fi

current_branch="$(git symbolic-ref --quiet --short HEAD)" || \
    fail "the source checkout is detached"
[[ "${current_branch}" == "${branch}" ]] || \
    fail "expected branch ${branch}, found ${current_branch}"

before="$(git rev-parse --short=12 HEAD)"
echo "[voice-gateway-update] Current source: ${before}"
echo "[voice-gateway-update] Fetching ${remote}/${branch}"
git fetch --prune "${remote}" "${branch}"
git merge-base --is-ancestor HEAD "${remote}/${branch}" || \
    fail "${remote}/${branch} is not a fast-forward of the deployed source"
git merge --ff-only "${remote}/${branch}"
after="$(git rev-parse --short=12 HEAD)"
expected_version="$(git describe --always)"

echo "[voice-gateway-update] Validating source ${after}"
python3 -m py_compile voice_gateway/*.py tests/*.py
bash -n bin/install-service update.sh
python3 -m unittest discover -s tests -v

sudo install -m 0644 systemd/voice-gateway.service \
    /etc/systemd/system/voice-gateway.service
sudo systemctl daemon-reload
sudo systemctl restart "${service}"

set -a
# shellcheck disable=SC1090
source "${secret_file}"
set +a
[[ -n "${SPARKIE_VOICE_GATEWAY_SECRET:-}" ]] || \
    fail "SPARKIE_VOICE_GATEWAY_SECRET is missing"

health="$(curl --fail --silent --show-error \
    --retry 20 --retry-all-errors --retry-delay 1 \
    -H "Authorization: Bearer ${SPARKIE_VOICE_GATEWAY_SECRET}" \
    http://127.0.0.1:8090/health)" || fail "gateway readiness check failed"

runtime_version="$(python3 -c \
    'import json,sys; print(json.load(sys.stdin).get("version", ""))' \
    <<<"${health}")"
[[ "${runtime_version}" == "${expected_version}" ]] || \
    fail "running version ${runtime_version:-unknown} does not match source ${expected_version}"

if [[ "${before}" == "${after}" ]]; then
    echo "[voice-gateway-update] Already current: ${after}"
else
    echo "[voice-gateway-update] Updated ${before} -> ${after}"
fi
echo "[voice-gateway-update] Gateway ready at source version ${runtime_version}"
