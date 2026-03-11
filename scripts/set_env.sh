#!/bin/bash
#
# Set environment variables inside the NIXL container.
# Source this before building or running.
#
# Usage:  source scripts/set_env.sh [nvshmem|nixl]
#   Default: nixl

MODE=${1:-nixl}

# ── Base paths (adjust if your container layout differs) ──
export CUDA_HOME=/usr/local/cuda
export RDMA_HOME=/workspace/rdma-core/build
export DOCA_HOME=/workspace/doca/build/install
export UCX_BASE=/workspace/ucx
export GDR_BASE=/workspace/gdrcopy
export NIXL_HOME=/workspace/nixl
export NVSHMEM_HOME=/workspace/nvshmem_src/build
export ETCD_LIB=/workspace/etcd-cpp-apiv3/build/src

export GDR=$GDR_BASE/include
export GDR_LIB=$GDR_BASE/lib
export UCX_HOME=$UCX_BASE/rfs

# GDRCopy symlink
mkdir -p /opt/mellanox 2>/dev/null
ln -sf "$(dirname "$GDR")" /opt/mellanox/gdrcopy 2>/dev/null

# ── Include / library paths ──
export CUDA_INC=$CUDA_HOME/include
export CUDA_LIB64=$CUDA_HOME/lib64
export CPATH="$CUDA_INC:$DOCA_HOME/include:$RDMA_HOME/include:${CPATH:-}"
export INCLUDE="$NIXL_HOME/include:$DOCA_HOME/include:${INCLUDE:-}"
export LD_LIBRARY_PATH="$ETCD_LIB:$NIXL_HOME/build/lib:$NIXL_HOME/lib:$RDMA_HOME/lib:$DOCA_HOME/lib/x86_64-linux-gnu:$UCX_HOME/lib:$CUDA_LIB64:${LD_LIBRARY_PATH:-}"
export LDFLAGS="-L$UCX_HOME/lib -L$RDMA_HOME/lib -L$GDR_LIB -L$DOCA_HOME/lib/x86_64-linux-gnu -L$CUDA_LIB64"
export CFLAGS="-I$UCX_HOME/include -I$RDMA_HOME/include -I$GDR -I$DOCA_HOME/include -I$CUDA_INC -I/usr/local/include"
export CPPFLAGS="$CFLAGS -I/usr/local/cuda/include"
export CXXFLAGS="$CFLAGS"
export NVCC_FLAGS="-I$DOCA_HOME/include -I$RDMA_HOME/include -I$CUDA_INC"
export PKG_CONFIG_PATH="$RDMA_HOME/lib/pkgconfig:/usr/local/lib/pkgconfig:$UCX_BASE/build-release/lib/pkgconfig:$UCX_HOME/lib/pkgconfig:$DOCA_HOME/lib/x86_64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-}"
export PATH="$CUDA_HOME/bin:$NIXL_HOME/build/examples/cpp:$UCX_HOME/bin:$RDMA_HOME/bin:${PATH:-}"

# ── CUDA arch ──
unset TORCH_CUDA_ARCH_LIST
unset DISABLE_SM90_FEATURES
unset DISABLE_AGGRESSIVE_PTX_INSTRS

# ── Backend-specific configuration ──
if [ "$MODE" = "nvshmem" ]; then
    echo "[env] Configuring for NVSHMEM backend"
    unset NIXL_LIB_PATH
    unset NIXL_INCLUDE_PATHS

    # ── Auto-install NVSHMEM 3.5 SDK if not present ──
    CUDA_MAJOR=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' || echo "13")

    detect_apt_nvshmem() {
        APT_NVSHMEM_INC=""
        APT_NVSHMEM_LIB=""
        # Headers: /usr/include/nvshmem_<ver>/nvshmem.h (apt versioned layout)
        for inc in /usr/include/nvshmem_${CUDA_MAJOR} /usr/include/nvshmem /usr/local/nvshmem/include; do
            if [ -f "$inc/nvshmem.h" ]; then APT_NVSHMEM_INC="$inc"; break; fi
        done
        # Libs: /usr/lib/x86_64-linux-gnu/nvshmem/<ver>/ (files directly in dir, no lib/ subdir)
        for lib in /usr/lib/x86_64-linux-gnu/nvshmem/${CUDA_MAJOR} /usr/lib/x86_64-linux-gnu /usr/local/nvshmem/lib; do
            if [ -f "$lib/libnvshmem_device.a" ]; then APT_NVSHMEM_LIB="$lib"; break; fi
        done
        [ -n "$APT_NVSHMEM_INC" ] && [ -n "$APT_NVSHMEM_LIB" ]
    }

    if ! detect_apt_nvshmem; then
        echo "[env] NVSHMEM SDK not found. Trying apt install..."
        apt-get update -qq 2>/dev/null
        apt-get -y install "nvshmem-cuda-${CUDA_MAJOR}" 2>/dev/null \
            || apt-get -y install "nvidia-nvshmem-cu${CUDA_MAJOR}" 2>/dev/null \
            || echo "[env] WARNING: apt install failed"
        detect_apt_nvshmem
    fi

    # ── Find CCCL headers (needed by NVSHMEM 3.5 for cuda/std/tuple) ──
    CCCL_INCLUDE=""
    # Search common locations — note CCCL may nest headers under cccl/ subdirectory
    for candidate in \
        "$CUDA_HOME/include" \
        "$CUDA_HOME/targets/x86_64-linux/include" \
        "$CUDA_HOME/targets/x86_64-linux/include/cccl" \
        "/usr/local/cuda/targets/x86_64-linux/include/cccl" \
    ; do
        if [ -f "$candidate/cuda/std/tuple" ]; then
            CCCL_INCLUDE="$candidate"
            break
        fi
    done
    # Check pip packages (nvidia.cuda_cccl or nvidia.cu13)
    if [ -z "$CCCL_INCLUDE" ]; then
        for pypath in \
            $(python3 -c "import nvidia.cuda_cccl; print(nvidia.cuda_cccl.__path__[0])" 2>/dev/null) \
            $(python3 -c "import nvidia.cu13; print(nvidia.cu13.__path__[0])" 2>/dev/null) \
        ; do
            for sub in "include" "include/cccl"; do
                if [ -f "$pypath/$sub/cuda/std/tuple" ]; then
                    CCCL_INCLUDE="$pypath/$sub"
                    break 2
                fi
            done
        done
    fi
    # Brute-force search as last resort
    if [ -z "$CCCL_INCLUDE" ]; then
        FOUND=$(find /usr/local/cuda* /workspace/.venv -path "*/cccl/cuda/std/tuple" -type f 2>/dev/null | head -1)
        if [ -n "$FOUND" ]; then
            CCCL_INCLUDE=$(dirname "$(dirname "$(dirname "$(dirname "$FOUND")")")")
        fi
    fi
    if [ -n "$CCCL_INCLUDE" ]; then
        export CPATH="$CCCL_INCLUDE:${CPATH:-}"
        echo "[env] CCCL headers: $CCCL_INCLUDE"
    else
        echo "[env] WARNING: CCCL headers not found! NVSHMEM 3.5 build will fail."
        echo "[env]   Run: find / -path '*/cuda/std/tuple' 2>/dev/null"
    fi

    # ── Select NVSHMEM: prefer apt SDK (full), then Lustre, then pip ──
    NVSHMEM_FOUND=false

    # 1. apt-installed SDK — create a unified layout that setup.py expects ($HOME/include + $HOME/lib)
    if [ -n "${APT_NVSHMEM_INC:-}" ] && [ -n "${APT_NVSHMEM_LIB:-}" ]; then
        NVSHMEM_STAGING=/tmp/nvshmem_sdk
        mkdir -p "$NVSHMEM_STAGING"
        ln -sfn "$APT_NVSHMEM_INC" "$NVSHMEM_STAGING/include"
        ln -sfn "$APT_NVSHMEM_LIB" "$NVSHMEM_STAGING/lib"
        export NVSHMEM_HOME="$NVSHMEM_STAGING"
        export NVSHMEM_DIR="$NVSHMEM_STAGING"
        echo "[env] Using apt NVSHMEM: includes=$APT_NVSHMEM_INC libs=$APT_NVSHMEM_LIB"
        NVSHMEM_FOUND=true
    fi

    # 2. Lustre-mounted NVSHMEM (from-source build)
    if ! $NVSHMEM_FOUND; then
        for d in \
            /workspace/aschwartz/workspace/nvshmem_src \
            /workspace/nvshmem_src/build \
            /workspace/nvshmem/build \
        ; do
            if [ -d "$d/lib" ] && [ -d "$d/include" ]; then
                export NVSHMEM_HOME="$d"
                export NVSHMEM_DIR="$d"
                echo "[env] Using Lustre NVSHMEM at: $d"
                NVSHMEM_FOUND=true
                break
            fi
        done
    fi

    # 3. pip-installed (runtime only, may lack libnvshmem_host.so for linking)
    if ! $NVSHMEM_FOUND; then
        PIP_NVSHMEM=$(python3 -c "import nvidia.nvshmem; print(nvidia.nvshmem.__path__[0])" 2>/dev/null)
        if [ -n "$PIP_NVSHMEM" ]; then
            export NVSHMEM_HOME="$PIP_NVSHMEM"
            export NVSHMEM_DIR="$PIP_NVSHMEM"
            echo "[env] Using pip-installed NVSHMEM at: $PIP_NVSHMEM"
            echo "[env] WARNING: pip NVSHMEM may lack libnvshmem_host.so for linking."
            NVSHMEM_FOUND=true
        fi
    fi

    if ! $NVSHMEM_FOUND; then
        echo "ERROR: Cannot find NVSHMEM installation."
        echo "  Install with: apt-get -y install nvshmem-cuda-13"
        echo "  Or try: apt-get -y install nvidia-nvshmem-cu13"
        echo "  Or run: bash scripts/install_nvshmem.sh"
        echo "  Or set NVSHMEM_HOME manually before sourcing this script."
        return 1
    fi

    export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:${LD_LIBRARY_PATH:-}"
    export PATH=$NVSHMEM_HOME/bin:$PATH

    # Uninstall pip NVSHMEM if apt SDK is used — prevents "device/host version mismatch" at runtime.
    # Python's import nvidia.nvshmem loads its own libnvshmem_host.so which conflicts with the apt version.
    if [ -n "${APT_NVSHMEM_LIB:-}" ]; then
        if python3 -c "import nvidia.nvshmem" 2>/dev/null; then
            echo "[env] Removing pip NVSHMEM to avoid version mismatch with apt SDK..."
            if command -v uv &>/dev/null; then
                uv pip uninstall nvidia-nvshmem-cu12 nvidia-nvshmem-cu13 2>/dev/null || true
            elif command -v pip &>/dev/null; then
                pip uninstall -y nvidia-nvshmem-cu12 nvidia-nvshmem-cu13 2>/dev/null || true
            fi
        fi
    fi

    # Runtime NVSHMEM env vars are set by buffer.py (NVSHMEM_IB_ENABLE_IBGDA, etc.)
    # Do NOT export build-time cmake vars here -- they can interfere with runtime.

elif [ "$MODE" = "nixl" ]; then
    echo "[env] Configuring for NIXL backend"
    unset NVSHMEM_DIR

    # NIXL paths — try installed location first, fall back to build tree
    if [ -d "$NIXL_HOME/lib" ] && [ -f "$NIXL_HOME/lib/libnixl.so" ]; then
        export NIXL_LIB_PATH=$NIXL_HOME/lib
        export NIXL_INCLUDE_PATHS=$NIXL_HOME/include:$UCX_HOME/include:$DOCA_HOME/include
    elif [ -d "$NIXL_HOME/build/src" ]; then
        export NIXL_LIB_PATH=$NIXL_HOME/build/src
        export NIXL_INCLUDE_PATHS=$NIXL_HOME/src/api/gpu/ucx:$NIXL_HOME/src/api/cpp:$UCX_HOME/include:$DOCA_HOME/include
    else
        echo "WARNING: Cannot find NIXL libraries at $NIXL_HOME/lib or $NIXL_HOME/build/src"
    fi

    export LD_LIBRARY_PATH=$NIXL_LIB_PATH:${LD_LIBRARY_PATH:-}

else
    echo "Error: Unknown mode '$MODE'. Use 'nvshmem' or 'nixl'"
    return 1
fi

echo ""
echo "Environment configured for: $MODE"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  UCX_HOME=$UCX_HOME"
echo "  NIXL_HOME=$NIXL_HOME"
echo "  NVSHMEM_HOME=$NVSHMEM_HOME"
[ -n "${NIXL_LIB_PATH:-}" ]   && echo "  NIXL_LIB_PATH=$NIXL_LIB_PATH"
[ -n "${NIXL_INCLUDE_PATHS:-}" ] && echo "  NIXL_INCLUDE_PATHS=$NIXL_INCLUDE_PATHS"
[ -n "${NVSHMEM_DIR:-}" ]      && echo "  NVSHMEM_DIR=$NVSHMEM_DIR"
