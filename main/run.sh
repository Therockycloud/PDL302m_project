#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

export PYTHONPATH="$PROJECT_ROOT/main${PYTHONPATH:+:$PYTHONPATH}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

usage() {
  cat <<'EOF'
Usage:
  ./run.sh ui [streamlit-args...]
  ./run.sh api [uvicorn-args...]
  ./run.sh all

Defaults to `ui` when no subcommand is given.
EOF
}

pick_python() {
  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    echo "$CONDA_PREFIX/bin/python"
    return
  fi

  if command -v conda >/dev/null 2>&1; then
    echo "conda run -n base python"
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi

  echo "python"
}

read -r -a PYTHON_CMD <<<"$(pick_python)"

run_ui() {
  "${PYTHON_CMD[@]}" -m streamlit run "$PROJECT_ROOT/main/src/ui/dashboard.py" "$@"
}

run_api() {
  "${PYTHON_CMD[@]}" -m uvicorn main.src.api.app:app --reload --port 8000 "$@"
}

main() {
  local mode="${1:-ui}"
  shift || true

  case "$mode" in
    ui|dashboard)
      run_ui "$@"
      ;;
    api|backend)
      run_api "$@"
      ;;
    all)
      run_api &
      local api_pid=$!
      trap 'kill "$api_pid" >/dev/null 2>&1 || true' EXIT INT TERM
      sleep 3
      run_ui "$@"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "Unknown command: $mode" >&2
      usage >&2
      return 1
      ;;
  esac
}

main "$@"
