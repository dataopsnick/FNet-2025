#!/bin/bash
# =================================================================
#  IMMUTABLE BOOTSTRAPPER (The "Key Turner")
# =================================================================
# This script is BAKED into the Docker image as its main ENTRYPOINT.
# Its ONLY job is to clone the application repository and then
# hand off control to the runtime script from that repository.
# =================================================================

set -eu

# --- Configuration ---
APP_DIR="/app"

# --- Parse Command Line Arguments ---
declare -a remaining_args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --git_repo_url)
            GIT_REPO_URL="$2"
            shift 2
            ;;
        --git_branch)
            GIT_BRANCH="$2"
            shift 2
            ;;
        *)
            remaining_args+=("$1")
            shift
            ;;
    esac
done

# --- Validate arguments ---
if [ -z "${GIT_REPO_URL:-}" ] || [ -z "${GIT_BRANCH:-}" ]; then
  echo "FATAL: --git_repo_url and --git_branch must be provided." >&2
  exit 1
fi

echo "✅ Bootstrapper initiated."
echo "   - Git Repo URL: ${GIT_REPO_URL}"
echo "   - Git Branch:   ${GIT_BRANCH}"

cd /
# --- Clone the Repository ---
echo "Cloning application code..."
rm -rf "${APP_DIR}"
git clone --single-branch --branch "${GIT_BRANCH}" --depth 1 "${GIT_REPO_URL}" "${APP_DIR}"

# --- Show the resulting file structure---
echo "Performing recursive list of the cloned directory at ${APP_DIR}..."
# Use 'ls -laR' for a recursive, detailed, long-format listing.
# This will show us everything: subdirectories, file permissions, owners, and sizes.
ls -laR "${APP_DIR}"

# --- Hand off Control ---
RUNTIME_SCRIPT="${APP_DIR}/aws/scripts/runtime_entrypoint.sh"
echo "Handing off execution to the runtime script: ${RUNTIME_SCRIPT}"

if [ ! -f "${RUNTIME_SCRIPT}" ]; then
    echo "FATAL: Runtime script not found at ${RUNTIME_SCRIPT} in the cloned repository." >&2
    exit 1
fi

# Use 'exec' to replace this process, ensuring proper signal handling.
exec bash "${RUNTIME_SCRIPT}" "${remaining_args[@]}"