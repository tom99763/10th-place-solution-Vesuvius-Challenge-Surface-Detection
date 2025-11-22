#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BETTI_DIR="external/Betti-Matching-3D"

# fresh configure so we don't reuse wrong Python from cache
rm -rf "$BETTI_DIR/build"
mkdir -p "$BETTI_DIR/build"
cd "$BETTI_DIR/build"

PYBIN="$(which python)"

# Ask THIS python for its include/lib paths
PY_INC="$("$PYBIN" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("INCLUDEPY") or sysconfig.get_paths().get("include",""))
PY
)"

PY_LIB="$("$PYBIN" - <<'PY'
import sysconfig, os
libdir = sysconfig.get_config_var("LIBDIR") or ""
ldver  = sysconfig.get_config_var("LDVERSION") or sysconfig.get_config_var("VERSION")
candidates = [
    os.path.join(libdir, f"libpython{ldver}.so"),
    os.path.join(libdir, f"libpython{ldver}.a"),
]
print(next((p for p in candidates if os.path.exists(p)), ""))
PY
)"

echo "Using:"
echo "  Python_EXECUTABLE=$PYBIN"
echo "  Python_INCLUDE_DIR=$PY_INC"
echo "  Python_LIBRARY=$PY_LIB"

# If PY_LIB was empty, you likely need the shared/static lib:
#   conda install -n topo-metrics-3d -c conda-forge libpython
# or adjust to .../libpython3.11.a
[ -n "$PY_LIB" ] || { echo "ERROR: Could not find libpython in this env."; exit 1; }

CMAKE_ARGS=(
  # >>> Modern FindPython variables <<<
  -DPython_EXECUTABLE="$PYBIN"
  -DPython_INCLUDE_DIR="$PY_INC"
  -DPython_LIBRARY="$PY_LIB"
  -DPython_ROOT_DIR="${CONDA_PREFIX:-}"

  # pybind11 prefers this on new CMake
  -DPYBIND11_FINDPYTHON=ON

  # Help CMake find conda packages first
  -DCMAKE_PREFIX_PATH="${CONDA_PREFIX:-}"
)

# Apple Silicon hint (ignored on Linux)
if [[ "$(uname -s)" == "Darwin" ]] && sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -q "Apple"; then
  CMAKE_ARGS+=(-DCMAKE_OSX_ARCHITECTURES=arm64)
fi

# Configure & build (generator-agnostic)
cmake -S .. -B . "${CMAKE_ARGS[@]}"
cmake --build . --parallel
