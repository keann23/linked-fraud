#!/usr/bin/env bash
# Creates .venv and installs dependencies.
# On macOS without Homebrew's libomp, LightGBM cannot find OpenMP; we point it
# at the copy scikit-learn already bundles.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
if [[ "$(uname)" == "Darwin" ]] && ! .venv/bin/python -c "import lightgbm" 2>/dev/null; then
  SP=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
  install_name_tool -add_rpath "@loader_path/../../sklearn/.dylibs" "$SP/lightgbm/lib/lib_lightgbm.dylib"
  codesign --force -s - "$SP/lightgbm/lib/lib_lightgbm.dylib"
fi
.venv/bin/python -c "import duckdb, lightgbm, shap, networkx; print('environment ready')"
