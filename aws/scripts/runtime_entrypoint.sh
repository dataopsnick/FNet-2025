#!/bin/bash
# =================================================================
#  DYNAMIC RUNTIME SCRIPT (The "Chef")
# =================================================================
# This script is part of the Git repository. It is responsible for
# all application-specific setup and for launching the training.
# =================================================================

set -eu

# --- Environment and Path Setup ---
APP_ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "${APP_ROOT_DIR}"
echo "✅ Runtime script initiated in $(pwd)"
echo "   - Forwarded arguments: $@"

# --- Install Dependencies ---
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies from requirements.txt..."
    pip install --no-cache-dir -r requirements.txt --upgrade
    echo "✅ Dependencies installed."
else
    echo "⚠️ WARNING: No requirements.txt found. Using base image dependencies."
fi

# --- Execute the Python Training Script ---
echo "🚀 Launching Python training script (src/train.py)..."
python -u src/train.py "$@"

echo "🎉 Training script finished."