#!/bin/bash
#
# Build DeepEP inside the container.
#
# Usage:  bash scripts/build.sh [nvshmem|nixl]
#   Default: nixl
#
# Builds in a local (non-shared) temp directory to avoid Lustre stale-handle
# issues when multiple nodes share the same source tree. The wheel is copied
# back to the shared dist/ for other nodes to install.

MODE=${1:-nixl}
DEEPEP_DIR=${DEEPEP_DIR:-/workspace/deepep}
BUILD_DIR=/tmp/deepep_build_$$

echo "=== Building DeepEP ($MODE) ==="
echo ""

cd "$DEEPEP_DIR"

# Source environment
source scripts/set_env.sh $MODE

# Copy source to a local (non-Lustre) temp directory
echo "--- Copying source to local build dir: $BUILD_DIR ---"
rm -rf /tmp/deepep_build_*
mkdir -p "$BUILD_DIR"
cp -a csrc deep_ep setup.py LICENSE "$BUILD_DIR/"
cd "$BUILD_DIR"

# Clean torch extension cache
rm -rf /root/.cache/torch_extensions/ /tmp/torch_extensions_*

echo ""
echo "--- Running: python3 setup.py bdist_wheel ---"
echo ""

python3 setup.py bdist_wheel 2>&1 | tee "$DEEPEP_DIR/build_${MODE}.log"
BUILD_STATUS=${PIPESTATUS[0]}

if [ $BUILD_STATUS -ne 0 ]; then
    echo ""
    echo "!!! BUILD FAILED (exit code $BUILD_STATUS) !!!"
    echo "Full log: $DEEPEP_DIR/build_${MODE}.log"
    echo ""
    echo "Last 30 lines of build log:"
    tail -30 "$DEEPEP_DIR/build_${MODE}.log"
    rm -rf "$BUILD_DIR"
    exit 1
fi

# Install from LOCAL build first (before touching shared filesystem)
# Use --no-cache to ensure the .so is actually replaced (same version, different binary)
if command -v uv &>/dev/null; then
    uv pip install dist/*.whl --force-reinstall --no-deps --no-cache
elif command -v pip &>/dev/null; then
    pip install dist/*.whl --force-reinstall --no-deps --no-cache-dir
else
    echo "WARNING: Neither uv nor pip found. Install manually."
fi

# Copy wheel to shared filesystem (for reference only)
mkdir -p "$DEEPEP_DIR/dist"
cp -f dist/*.whl "$DEEPEP_DIR/dist/"

# Cleanup
rm -rf "$BUILD_DIR"

echo ""
echo "=== Build complete ($MODE) ==="
echo "Wheel:  $(ls "$DEEPEP_DIR"/dist/*.whl 2>/dev/null)"
echo "Log:    $DEEPEP_DIR/build_${MODE}.log"
