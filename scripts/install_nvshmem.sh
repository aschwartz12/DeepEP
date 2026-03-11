#!/bin/bash
#
# Install NVSHMEM 3.5 SDK for the NVSHMEM backend.
# Tries apt first (full SDK with headers + static libs), falls back to pip (runtime only).
#
# Usage: bash scripts/install_nvshmem.sh
#
# After running this, build with: bash scripts/build.sh nvshmem

set -euo pipefail

# Detect CUDA major version
CUDA_MAJOR=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' || echo "13")
APT_PKG_1="nvshmem-cuda-${CUDA_MAJOR}"
APT_PKG_2="nvidia-nvshmem-cu${CUDA_MAJOR}"
PIP_PKG="nvidia-nvshmem-cu${CUDA_MAJOR}"

echo "=== Installing NVSHMEM (CUDA ${CUDA_MAJOR}) ==="
echo ""

# Prefer apt — gives full SDK (headers + shared + static libs)
echo "[1/2] Trying apt..."
apt-get update -qq 2>/dev/null
if apt-get -y install "$APT_PKG_1" 2>/dev/null; then
    echo "  Installed via apt ($APT_PKG_1)."
elif apt-get -y install "$APT_PKG_2" 2>/dev/null; then
    echo "  Installed via apt ($APT_PKG_2)."
else
    echo "  apt packages not available. Trying pip: $PIP_PKG ..."
    if command -v uv &>/dev/null; then
        uv pip install "$PIP_PKG"
    elif command -v pip &>/dev/null; then
        pip install "$PIP_PKG"
    else
        echo "ERROR: Neither apt, uv, nor pip could install NVSHMEM"
        exit 1
    fi
    echo "  WARNING: pip NVSHMEM is runtime-only (may lack libnvshmem_device.a for linking)."
    echo "  For a full build, install the apt package instead."
fi

# Also install CCCL (needed by NVSHMEM 3.5 headers)
echo ""
echo "[2/2] Ensuring CCCL headers are available..."
if ! python3 -c "import nvidia.cuda_cccl" 2>/dev/null; then
    if command -v uv &>/dev/null; then
        uv pip install nvidia-cuda-cccl
    elif command -v pip &>/dev/null; then
        pip install nvidia-cuda-cccl
    fi
fi

# Verify
echo ""
echo "=== Verification ==="
for d in /usr/lib/x86_64-linux-gnu/nvshmem/${CUDA_MAJOR} /usr/local/nvshmem; do
    if [ -f "$d/include/nvshmem.h" ]; then
        echo "  NVSHMEM SDK: $d"
        break
    fi
done
PIP_NVSHMEM=$(python3 -c "import nvidia.nvshmem; print(nvidia.nvshmem.__path__[0])" 2>/dev/null || true)
[ -n "$PIP_NVSHMEM" ] && echo "  pip NVSHMEM: $PIP_NVSHMEM"
PIP_CCCL=$(python3 -c "import nvidia.cuda_cccl; print(nvidia.cuda_cccl.__path__[0])" 2>/dev/null || true)
[ -n "$PIP_CCCL" ] && echo "  CCCL: $PIP_CCCL"

echo ""
echo "To build: bash scripts/build.sh nvshmem"
