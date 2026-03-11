# DeepEP on EOS — Full Setup Guide

Benchmarking NVSHMEM vs NIXL internode Expert Parallelism on EOS (H100 + ConnectX-7 IB).

## Prerequisites

- EOS login node access
- SLURM portfolio: `network_research_advdev` (alias: `network`)
- Container image: `latest.sqsh` on Lustre

---

## 0. Paths

```
Login node working dir:  /lustre/fsw/network_research_advdev/aschwartz/deepep_pr/DeepEP
Container mount:         /workspace/deepep        (same source tree, bind-mounted)
Container Lustre mount:  /workspace/aschwartz      (full Lustre home)
Container image:         /lustre/fsw/network_research_advdev/aschwartz/latest.sqsh
```

---

## 1. Import the Container Image (one-time)

The image is a NIXL dev container (`hybrid-ep:nixl-latest`) with CUDA 13, UCX, DOCA,
NIXL, and PyTorch pre-installed. Import it on a compute node (not the login node)
because it needs ~60 GB of temp space.

```bash
cd /lustre/fsw/network_research_advdev/aschwartz/deepep_pr/DeepEP

salloc -N 1 -p batch -A network_research_advdev -t 02:00:00 \
  -J network_research_advdev-deepep.import \
  --mem=0 --ntasks-per-node=1 \
  bash -c '
    rm -rf /home/$USER/.cache/enroot/ && \
    mkdir -p /lustre/fsw/network_research_advdev/aschwartz/tmp/enroot-cache && \
    export ENROOT_CACHE_PATH=/lustre/fsw/network_research_advdev/aschwartz/tmp/enroot-cache && \
    export ENROOT_TEMP_PATH=/tmp && \
    export TMPDIR=/tmp && \
    export ENROOT_SQUASH_OPTIONS="-processors 4 -mem 16384" && \
    enroot import -o /lustre/fsw/network_research_advdev/aschwartz/latest.sqsh \
      "docker://nvcr.io#nvidia/hybrid-ep:nixl-latest"
  '
```

**Key pitfalls:**

- `ENROOT_CACHE_PATH` must point to Lustre (home dir runs out of space).
- `ENROOT_TEMP_PATH` must be local `/tmp` (Lustre doesn't support overlayfs whiteouts).
- Use `--mem=0` to get full node memory for `mksquashfs`.
- Takes ~15-30 min. Wait for `salloc: Relinquishing job allocation` — that means it finished.
- Verify: `ls -lh /lustre/fsw/network_research_advdev/aschwartz/latest.sqsh` (should be ~20+ GB).

---

## 2. Allocate 2 Nodes

From the **login node** (not inside a container):

```bash
cd /lustre/fsw/network_research_advdev/aschwartz/deepep_pr/DeepEP

# Terminal 1 — allocate 2 nodes (this terminal becomes node 1)
bash scripts/submit.sh network 2
```

This runs `salloc` with the container image and mounts. You'll land inside the
container on node 1 once resources are granted.

---

## 3. Attach to Node 2

Open a **second terminal** on the login node:

```bash
cd /lustre/fsw/network_research_advdev/aschwartz/deepep_pr/DeepEP

# Terminal 2 — attach to node 2
bash scripts/submit.sh network attach 2
```

You now have two terminals, each inside the container on a different compute node.

---

## 4. One-Time Setup (inside container, BOTH nodes)

### 4a. Install NVSHMEM 3.5 SDK

```bash
bash scripts/install_nvshmem.sh
```

Or manually: `apt-get update && apt-get -y install nvshmem-cuda-13`

> This is also auto-triggered by `bash scripts/build.sh nvshmem` if NVSHMEM is missing.

---

## 5. Build

All builds happen inside the container. Run on **both nodes**.

### NVSHMEM backend

```bash
cd /workspace/deepep
bash scripts/build.sh nvshmem
```

### NIXL backend

```bash
cd /workspace/deepep
bash scripts/build.sh nixl
```

---

## 6. Run Tests

### 6a. NVSHMEM Internode

Get node 1's IP first:

```bash
# On node 1
hostname -i
# Example output: 10.52.51.205
```

**Node 1 (rank 0):**

```bash
cd /workspace/deepep/tests
export WORLD_SIZE=2 RANK=0 MASTER_ADDR=$(hostname -i) MASTER_PORT=8600
python3 test_internode.py --num-processes 8
```

**Node 2 (rank 1):** (replace `<NODE1_IP>` with the IP from above)

```bash
cd /workspace/deepep/tests
export WORLD_SIZE=2 RANK=1 MASTER_ADDR=<NODE1_IP> MASTER_PORT=8600
python3 test_internode.py --num-processes 8
```

Start both within **60 seconds** of each other (NCCL init timeout).

### 6b. NIXL Internode

**Node 1 (rank 0):**

```bash
cd /workspace/deepep/tests/nixl
export WORLD_SIZE=2 RANK=0 MASTER_ADDR=$(hostname -i) MASTER_PORT=8500
export PYTHONPATH=/workspace/deepep/tests/nixl
python3 test_internode.py --num-processes 8
```

**Node 2 (rank 1):**

```bash
cd /workspace/deepep/tests/nixl
export WORLD_SIZE=2 RANK=1 MASTER_PORT=8500 MASTER_ADDR=<NODE1_IP> 
export PYTHONPATH=/workspace/deepep/tests/nixl
python3 test_internode.py --num-processes 8 --tcp-server <NODE1_IP>
```

> Node 2 **must** pass `--tcp-server <NODE1_IP>` so it connects to node 1's
> TCPStore for NIXL metadata exchange. Without this flag, it starts its own
> independent store and the test hangs.

---

## 7. Compare Backends (automated)

The `compare_backends.sh` script runs NVSHMEM then NIXL back-to-back and
prints a side-by-side performance table on rank 0.

**Node 1:**

```bash
cd /workspace/deepep
bash scripts/compare_backends.sh $(hostname -i) 0
```

**Node 2:**

```bash
cd /workspace/deepep
bash scripts/compare_backends.sh <NODE1_IP> 1
```

Results are saved to `/workspace/deepep/results/comparison_<timestamp>.txt`.

---

## 8. Cleanup Between Runs

If a test hangs or you need to restart:

```bash
pkill -9 -f test_internode.py
pkill -9 -f python3
sleep 3
```

Always use a **different `MASTER_PORT`** if you get `EADDRINUSE`.

---

## Quick Reference


| What             | Command                                                  |
| ---------------- | -------------------------------------------------------- |
| Allocate 2 nodes | `bash scripts/submit.sh network 2`                       |
| Attach to node N | `bash scripts/submit.sh network attach N`                |
| Build NVSHMEM    | `bash scripts/build.sh nvshmem`                          |
| Build NIXL       | `bash scripts/build.sh nixl`                             |
| Run NVSHMEM test | `python3 tests/test_internode.py --num-processes 8`      |
| Run NIXL test    | `python3 tests/nixl/test_internode.py --num-processes 8` |
| Compare both     | `bash scripts/compare_backends.sh <ip> <rank>`           |
| Kill stuck procs | `pkill -9 -f python3`                                    |


## Container Layout

```
/workspace/
  deepep/          <-- DeepEP source (Lustre bind mount, shared across nodes)
  aschwartz/       <-- Full Lustre home
  nixl/            <-- NIXL library (from container image)
  ucx/             <-- UCX with GDAKI support (from container image)
  doca/            <-- DOCA SDK (from container image)
  rdma-core/       <-- rdma-core (from container image)
  gdrcopy/         <-- GDRCopy (from container image)
  .venv/           <-- Python virtualenv with PyTorch (from container image)
```

## Troubleshooting


| Symptom                                      | Fix                                                                                            |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `enroot import` fails with "No space left"   | Set `ENROOT_CACHE_PATH` to Lustre, `ENROOT_TEMP_PATH=/tmp`                                     |
| `enroot-aufs2ovlfs: Operation not permitted` | `ENROOT_TEMP_PATH` must be local `/tmp`, not Lustre                                            |
| NVSHMEM build: missing `cuda/std/tuple`      | `set_env.sh` auto-installs CCCL; or run `uv pip install nvidia-cuda-cccl`                      |
| NVSHMEM build: missing `libnvshmem_device.a` | `apt-get -y install nvshmem-cuda-13` or `bash scripts/install_nvshmem.sh`                     |
| NIXL build: `Broken function found`          | Clean build cache: `rm -rf /root/.cache/torch_extensions/ /tmp/torch_extensions_*` and rebuild |
| Test hangs at "initializing buffer"          | Node 2 missing `--tcp-server` flag (NIXL only)                                                 |
| `EADDRINUSE` on MASTER_PORT                  | `pkill -9 -f python3` and use a different port                                                 |
| All `recv_expert_counter` at -1              | NVSHMEM version mismatch. Ensure NVSHMEM 3.5 SDK (`apt` package)                               |
| `timeout (dispatch CPU)`                     | Nodes can't reach each other. Check `MASTER_ADDR` and port match on both sides                 |


