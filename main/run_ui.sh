#!/usr/bin/env bash
# macOS/Linux shell script to run the Streamlit dashboard

# Enable strict error handling
set -euo pipefail

# Determine the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change directory to the project root so main/src/ui/dashboard.py path is resolved correctly
cd "$PROJECT_ROOT"

# Set environment variables
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH="$PROJECT_ROOT/main${PYTHONPATH:+:$PYTHONPATH}"

echo "Checking Python environment..."

# Determine the correct Python interpreter
PYTHON_EXEC=""

# On macOS, try to use miniforge base python first
if [[ "$(uname)" == "Darwin" ]] && [[ -x "/opt/homebrew/Caskroom/miniforge/base/bin/python" ]]; then
    PYTHON_EXEC="/opt/homebrew/Caskroom/miniforge/base/bin/python"
    echo "Found Miniforge Python at macOS default location: $PYTHON_EXEC"
elif [[ -n "${CONDA_PREFIX:-}" ]] && [[ -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON_EXEC="$CONDA_PREFIX/bin/python"
    echo "Using active Conda environment Python: $PYTHON_EXEC"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_EXEC="python3"
    echo "Using system python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_EXEC="python"
    echo "Using system python"
else
    echo "Error: Python is not installed or not found on PATH." >&2
    exit 1
fi

# Verify Python version
$PYTHON_EXEC --version

# Run Streamlit dashboard
echo "Starting Streamlit dashboard..."
exec "$PYTHON_EXEC" -m streamlit run main/src/ui/dashboard.py "$@"
