#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# --- Configuration ---
APP_DIR="/app" # Directory to clone the repo into

# --- Parse Command Line Arguments ---
# This loop parses arguments specific to this script (--git_repo_url, --git_branch)
# and collects all other arguments to be passed on to the training script.
declare -a train_args
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
            train_args+=("$1")
            shift
            ;;
    esac
done

# --- Validate arguments ---
if [ -z "${GIT_REPO_URL}" ] || [ -z "${GIT_BRANCH}" ]; then
  echo "Error: --git_repo_url and --git_branch must be provided."
  exit 1
fi

echo "Git Repo URL: ${GIT_REPO_URL}"
echo "Git Branch:   ${GIT_BRANCH}"
echo "App Directory:  ${APP_DIR}"
echo "Training args:  ${train_args[@]}"

# --- Clone the repository ---
echo "Cloning repository..."
# Clean up the directory in case of a retry on the same instance
rm -rf ${APP_DIR}
git clone --single-branch --branch "${GIT_BRANCH}" "${GIT_REPO_URL}" "${APP_DIR}"

# --- Install/Update Dependencies ---
cd ${APP_DIR}
if [ -f "requirements.txt" ]; then
    echo "Found requirements.txt in the repo, installing/updating dependencies..."
    pip install -r requirements.txt --upgrade
else
    echo "No requirements.txt found in the repository, using base dependencies."
fi

# --- Execute the Python Training Script ---
echo "Starting training script..."
python -m src.train "${train_args[@]}"