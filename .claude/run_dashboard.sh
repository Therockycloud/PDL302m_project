#!/usr/bin/env bash
# Wrapper so the preview tool launches Streamlit with the right interpreter,
# env (OpenMP guard), and PYTHONPATH. Honors $PORT if the preview tool sets it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH="$ROOT/main${PYTHONPATH:+:$PYTHONPATH}"
exec /opt/homebrew/Caskroom/miniforge/base/bin/python -m streamlit run \
  main/src/ui/dashboard.py \
  --server.port "${PORT:-8502}" --server.headless true \
  --browser.gatherUsageStats false
