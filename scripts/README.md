# DeepEP Test Infrastructure

## Quick Reference

```
scripts/
  submit.sh      # Allocate SLURM nodes with the NIXL container
  set_env.sh     # Configure environment (nvshmem or nixl mode)
  build.sh       # Compile DeepEP for a specific backend
  run_test.sh    # Run tests (see modes below)
```

## Step-by-Step Workflow

### 1. Allocate Nodes

From the login node (outside the container):

```bash
cd /lustre/fsw/network_research_advdev/aschwartz/deepep_pr/DeepEP

# Allocate 2 nodes (needed for internode tests)
bash scripts/submit.sh network 2
```

This uses the container image configured in `submit.sh`
and mounts DeepEP at `/workspace/deepep` inside the container.

### 2. Attach to Nodes

Open separate terminals and attach to each node:

```bash
# Terminal 1 - attach to node 1
bash scripts/submit.sh network attach 1

# Terminal 2 - attach to node 2
bash scripts/submit.sh network attach 2
```

### 3. Build (inside the container, on BOTH nodes)

```bash
cd /workspace/deepep
bash scripts/build.sh nixl
```

### 4. Run Tests

Multi-node tests must be launched manually on each node since SLURM
commands don't work inside the container. Use `run_test.sh` for
single-node convenience, or follow the manual instructions below.

#### NIXL High-Throughput Internode (2 nodes, 16 GPUs)

**Node 1** (get IP first with `hostname -i`):
```bash
cd /workspace/deepep/tests/nixl
export WORLD_SIZE=2 RANK=0 MASTER_ADDR=$(hostname -i) MASTER_PORT=8371
export PYTHONPATH=/workspace/deepep/tests/nixl
python3 test_internode.py --num-processes 8
```

**Node 2** (use Node 1's IP for both MASTER_ADDR and --tcp-server):
```bash
cd /workspace/deepep/tests/nixl
export WORLD_SIZE=2 RANK=1 MASTER_ADDR=<node1_ip> MASTER_PORT=8371
export PYTHONPATH=/workspace/deepep/tests/nixl
python3 test_internode.py --num-processes 8 --tcp-server <node1_ip>
```

Important:
- Replace `<node1_ip>` with Node 1's actual IP (from `hostname -i`).
- Both nodes must start within ~60 seconds of each other (NCCL timeout).
- Node 2 **must** use `--tcp-server <node1_ip>` so it connects to Node 1's
  TCPStore for NIXL metadata exchange (without this, it creates its own
  independent store and hangs).
- If you get `EADDRINUSE`, a previous run left zombie processes. Run
  `pkill -f test_internode.py` and use a different `MASTER_PORT`.

#### NIXL Elastic Scaling (single or multi-node, needs etcd)

Start etcd on Node 1:
```bash
/workspace/aschwartz/tools/etcd/etcd \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://$(hostname -i):2379 &
export NIXL_ETCD_ENDPOINTS=http://$(hostname -i):2379
```

Then run the elastic test:
```bash
cd /workspace/deepep/tests/nixl/elastic
python3 elastic.py \
  --plan single_expansion.json \
  --num-processes 8 \
  --etcd-server http://$(hostname -i):2379
```

#### NVSHMEM Tests (requires NVSHMEM build + NVSHMEM container)

These use the original test scripts in `tests/` (not `tests/nixl/`):
```bash
# Build with NVSHMEM (needs container with NVSHMEM installed)
bash scripts/build.sh nvshmem

# Single-node intranode
bash scripts/run_test.sh nvshmem_intra
```

NVSHMEM internode tests also require manual 2-node launch
(same pattern as NIXL, but using `tests/test_internode.py`).

## Test Modes in run_test.sh

| Mode | Backend | What it does |
|------|---------|--------------|
| `nvshmem_intra` | NVSHMEM | NVLink intranode (single-node, `run_test.sh` works) |
| `nvshmem_ht` | NVSHMEM | RDMA+NVLink internode (multi-node, manual launch) |
| `nvshmem_ll` | NVSHMEM | IBGDA low-latency (multi-node, manual launch) |
| `nixl_ht` | NIXL | NIXL internode (multi-node, manual launch recommended) |
| `nixl_ll` | NIXL | NIXL low-latency (multi-node, manual launch recommended) |
| `nixl_elastic` | NIXL | NIXL elastic scaling (needs etcd) |

Note: `run_test.sh` tries to use `srun` for multi-node distribution, which
doesn't work inside the container. For multi-node tests, always use the
manual 2-terminal approach described above.

## Environment Variables

The `set_env.sh` script sets all paths automatically. Key variables:

| Variable | NVSHMEM mode | NIXL mode |
|----------|-------------|-----------|
| `NVSHMEM_DIR` | set | unset |
| `NIXL_LIB_PATH` | unset | set |
| `NIXL_INCLUDE_PATHS` | unset | set |

## Switching Backends

```bash
# Clean and rebuild for NIXL
bash scripts/build.sh nixl

# Clean and rebuild for NVSHMEM
bash scripts/build.sh nvshmem
```

## Troubleshooting

**`EADDRINUSE` on port 8361/8371**: Previous processes still running.
```bash
pkill -f test_internode.py; pkill -f elastic.py; sleep 2
```

**Test hangs at "initializing buffer"**: Metadata exchange issue. Make sure
Node 2 uses `--tcp-server <node1_ip>`.

**`accelerated IB support was not found`**: UCX warning, safe to ignore.
Uses standard verbs fallback. Performance may be slightly lower.

**`Buffer.__init__() got an unexpected keyword argument 'rank'`**:
You're running an NVSHMEM test with an NIXL build. NIXL tests are in
`tests/nixl/`, NVSHMEM tests are in `tests/`.
