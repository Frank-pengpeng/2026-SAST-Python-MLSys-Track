#!/usr/bin/env bash
# Build src/simple_ml_ext.cpp into an importable extension on Windows/MinGW.
#
# The upstream Makefile targets Linux/macOS:
#     c++ -O3 -Wall -shared -std=c++11 -fPIC $(python -m pybind11 --includes) \
#         src/simple_ml_ext.cpp -o src/simple_ml_ext.so
#
# Two things break on Windows:
#
#   1. CPython on Windows only imports extensions named *.pyd, not *.so.
#   2. MinGW's ld cannot use CPython's MSVC-format libs/python313.lib, and the
#      resulting .pyd would otherwise depend on libwinpthread-1.dll, which is
#      not on Python's DLL search path -> "DLL load failed".
#
# So we (a) synthesise a MinGW import library from python313.dll with
# gendef/dlltool, and (b) statically link libgcc/libstdc++/winpthread.
#
# Usage:  PYTHON=/path/to/python.exe bash scripts/build_windows.sh
set -euo pipefail

# Work from the repo root and stay on *relative* paths: MinGW's g++/ld are native
# Windows binaries and do not understand MSYS-style "/c/Users/..." paths.
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null 2>&1 || {
    echo "error: '$PYTHON' not found; set PYTHON=/path/to/python.exe" >&2
    exit 1
}

# Locate the *base* CPython installation: inside a venv, sys.executable lives in
# <venv>/Scripts/ but python3xx.dll and Include/ live in the base prefix.
PY_INFO=$("$PYTHON" -c '
import glob, os, sys
for root in (sys.base_prefix, os.path.dirname(sys.executable), sys.prefix):
    found = [d for d in sorted(glob.glob(os.path.join(root, "python3*.dll")))
             if os.path.basename(d) != "python3.dll"]   # skip the ABI stub
    if found:
        print(root)
        print(found[-1])
        break
')

if [ -z "$PY_INFO" ]; then
    echo "error: could not locate python3xx.dll" >&2
    exit 1
fi

PY_ROOT=$(echo "$PY_INFO" | head -1)
PY_DLL=$(echo "$PY_INFO" | tail -1)
echo ">>> python root : $PY_ROOT"
echo ">>> python dll  : $PY_DLL"

DLL_NAME=$(basename "$PY_DLL")                       # e.g. python313.dll
LIB_NAME="lib${DLL_NAME%.dll}.a"                     # e.g. libpython313.a

IMPLIB_DIR="build/pylib"
mkdir -p "$IMPLIB_DIR"

if [ ! -f "$IMPLIB_DIR/$LIB_NAME" ]; then
    echo ">>> generating $LIB_NAME from $DLL_NAME (this can take a couple of minutes)"
    ( cd "$IMPLIB_DIR" && gendef "$PY_DLL" >/dev/null && \
      dlltool -d "${DLL_NAME%.dll}.def" -l "$LIB_NAME" -D "$DLL_NAME" )
fi

echo ">>> compiling src/simple_ml_ext.pyd with $(g++ --version | head -1)"
g++ -O3 -Wall -shared -std=c++11 -fPIC \
    -I"$PY_ROOT/Include" \
    $("$PYTHON" -m pybind11 --includes) \
    src/simple_ml_ext.cpp \
    -o src/simple_ml_ext.pyd \
    -L"$IMPLIB_DIR" -l"${DLL_NAME%.dll}" \
    -static-libgcc -static-libstdc++ -Wl,-Bstatic -lwinpthread -Wl,-Bdynamic

echo ">>> built: src/simple_ml_ext.pyd"
"$PYTHON" -c "import sys; sys.path.insert(0,'src'); import simple_ml_ext as m; print('    import check ->', m.softmax_regression_epoch_cpp.__name__)"
