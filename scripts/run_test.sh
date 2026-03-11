#!/bin/bash
#
# Run DeepEP tests across SLURM nodes.
#
# Usage:
#   bash scripts/run_test.sh <test_mode> [extra_args...]
#
# Test modes:
#   nvshmem_ht       - NVSHMEM high-throughput internode  (2+ nodes, 16+ ranks)
#   nvshmem_ll       - NVSHMEM low-latency               (2+ nodes, 16+ ranks)
#   nvshmem_intra    - NVSHMEM intranode only             (1 node,  8 ranks)
#   nixl_ht          - NIXL high-throughput internode     (2+ nodes, 16+ ranks)
#   nixl_ll          - NIXL low-latency                   (2+ nodes, 16+ ranks)
#   nixl_elastic     - NIXL elastic scaling test          (1-2 nodes)
#
# Examples:
#   bash scripts/run_test.sh nvshmem_ht
#   bash scripts/run_test.sh nixl_ll --num-tokens 2048
#   bash scripts/run_test.sh nixl_elastic --plan tests/elastic/single_expansion.json
#
# Prerequisites:
#   - Nodes allocated via scripts/submit.sh
#   - Inside the container on the head node
#   - DeepEP built via scripts/build.sh with the matching backend

set -euo pipefail

TEST_MODE=${1:?"Usage: run_test.sh <nvshmem_ht|nvshmem_ll|nvshmem_intra|nixl_ht|nixl_ll|nixl_elastic> [args...]"}
shift
EXTRA_ARGS="$*"

DEEPEP_DIR=${DEEPEP_DIR:-/workspace/deepep}
TESTS_DIR=$DEEPEP_DIR/tests
NUM_GPUS_PER_NODE=8

# ── DOCA preloads (needed for NIXL/UCX with GPUNetIO) ──
DOCA_HOME=${DOCA_HOME:-/workspace/doca/build/install}
DOCA_PRELOAD=""
if [ -d "$DOCA_HOME/lib/x86_64-linux-gnu" ]; then
    DOCA_PRELOAD="$DOCA_HOME/lib/x86_64-linux-gnu/libdoca_common.so"
    DOCA_PRELOAD+=":$DOCA_HOME/lib/x86_64-linux-gnu/libdoca_gpunetio.so"
    DOCA_PRELOAD+=":$DOCA_HOME/lib/x86_64-linux-gnu/libdoca_verbs.so"
fi

# ── Common environment ──
export UCX_LOG_LEVEL=error
export PYTHONPATH=$TESTS_DIR:${PYTHONPATH:-}

get_num_nodes() {
    if [ -n "${SLURM_JOB_NUM_NODES:-}" ]; then
        echo $SLURM_JOB_NUM_NODES
    else
        echo 1
    fi
}

NUM_NODES=$(get_num_nodes)

# ── Launcher helpers ──
run_torchrun() {
    local num_nodes=$1
    local script=$2
    shift 2
    local args="$*"

    if [ "$num_nodes" -gt 1 ]; then
        echo "[run] Multi-node torchrun: $num_nodes nodes x $NUM_GPUS_PER_NODE GPUs"
        LD_PRELOAD="${DOCA_PRELOAD}" \
        srun --ntasks-per-node=1 \
            torchrun \
                --nproc_per_node=$NUM_GPUS_PER_NODE \
                --nnodes=$num_nodes \
                --rdzv_backend=c10d \
                --rdzv_endpoint="${MASTER_ADDR:-$(hostname)}:${MASTER_PORT:-29500}" \
                $script $args
    else
        echo "[run] Single-node: $NUM_GPUS_PER_NODE GPUs"
        LD_PRELOAD="${DOCA_PRELOAD}" \
        python3 $script --num-processes $NUM_GPUS_PER_NODE $args
    fi
}

run_multiprocess() {
    local script=$1
    shift
    local args="$*"
    local total_ranks=$((NUM_NODES * NUM_GPUS_PER_NODE))

    echo "[run] Multiprocess spawn: $total_ranks ranks across $NUM_NODES node(s)"

    if [ "$NUM_NODES" -gt 1 ]; then
        export WORLD_SIZE=$NUM_NODES
        export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)}
        export MASTER_PORT=${MASTER_PORT:-8361}
        echo "[run] MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT, WORLD_SIZE=$WORLD_SIZE"

        srun --ntasks-per-node=1 \
            bash -c "
                export RANK=\$SLURM_NODEID
                export LD_PRELOAD='${DOCA_PRELOAD}'
                cd $TESTS_DIR
                python3 $script --num-processes $NUM_GPUS_PER_NODE $args
            "
    else
        export WORLD_SIZE=1
        export RANK=0
        LD_PRELOAD="${DOCA_PRELOAD}" \
        cd $TESTS_DIR && python3 $script --num-processes $NUM_GPUS_PER_NODE $args
    fi
}

# ── Test dispatch ──
case "$TEST_MODE" in

    # ─────────────────────────────────────────────
    # NVSHMEM High-Throughput (internode)
    # ─────────────────────────────────────────────
    nvshmem_ht)
        echo "=== NVSHMEM High-Throughput Internode Test ==="
        echo "    Requires: 2+ nodes, built with NVSHMEM backend"
        echo ""
        echo "Multi-node tests must be launched manually on each node."
        echo "See scripts/README.md for instructions."
        ;;

    # ─────────────────────────────────────────────
    # NVSHMEM Low-Latency
    # ─────────────────────────────────────────────
    nvshmem_ll)
        echo "=== NVSHMEM Low-Latency Test ==="
        echo "    Requires: 2+ nodes, built with NVSHMEM backend"
        echo ""
        echo "Multi-node tests must be launched manually on each node."
        echo "See scripts/README.md for instructions."
        ;;

    # ─────────────────────────────────────────────
    # NVSHMEM Intranode
    # ─────────────────────────────────────────────
    nvshmem_intra)
        echo "=== NVSHMEM Intranode Test ==="
        echo "    Requires: 1 node, 8 GPUs"
        echo ""
        cd $TESTS_DIR
        export WORLD_SIZE=1
        export RANK=0
        LD_PRELOAD="${DOCA_PRELOAD}" \
        python3 test_intranode.py --num-processes $NUM_GPUS_PER_NODE $EXTRA_ARGS
        ;;

    # ─────────────────────────────────────────────
    # NIXL High-Throughput (internode)
    # ─────────────────────────────────────────────
    nixl_ht)
        echo "=== NIXL High-Throughput Internode Test ==="
        echo ""
        echo "Multi-node NIXL tests must be launched manually on each node."
        echo "srun does not work inside the container."
        echo ""
        echo "Node 1:"
        echo "  cd /workspace/deepep/tests/nixl"
        echo "  export WORLD_SIZE=2 RANK=0 MASTER_ADDR=\$(hostname -i) MASTER_PORT=8371"
        echo "  export PYTHONPATH=/workspace/deepep/tests/nixl"
        echo "  python3 test_internode.py --num-processes 8"
        echo ""
        echo "Node 2 (replace <node1_ip> with Node 1's IP from 'hostname -i'):"
        echo "  cd /workspace/deepep/tests/nixl"
        echo "  export WORLD_SIZE=2 RANK=1 MASTER_ADDR=<node1_ip> MASTER_PORT=8371"
        echo "  export PYTHONPATH=/workspace/deepep/tests/nixl"
        echo "  python3 test_internode.py --num-processes 8 --tcp-server <node1_ip>"
        echo ""
        echo "Start both within 60 seconds of each other."
        ;;

    # ─────────────────────────────────────────────
    # NIXL Low-Latency
    # ─────────────────────────────────────────────
    nixl_ll)
        echo "=== NIXL Low-Latency Test ==="
        echo ""
        echo "Same as nixl_ht but add --test-ll-compatibility flag."
        echo "Follow the nixl_ht instructions above."
        ;;

    # ─────────────────────────────────────────────
    # NIXL Elastic Scaling
    # ─────────────────────────────────────────────
    nixl_elastic)
        echo "=== NIXL Elastic Scaling Test ==="
        echo "    Requires: NIXL backend, etcd running"
        echo ""

        NIXL_ELASTIC_DIR=$TESTS_DIR/nixl/elastic
        PLAN=${PLAN:-$NIXL_ELASTIC_DIR/single_expansion.json}
        ETCD_SERVER=${ETCD_SERVER:-http://127.0.0.1:2379}
        NUM_PROCS=${NUM_PROCS:-$NUM_GPUS_PER_NODE}

        if [ ! -d "$NIXL_ELASTIC_DIR" ]; then
            echo "ERROR: elastic test directory not found at $NIXL_ELASTIC_DIR"
            exit 1
        fi

        echo "Plan:         $PLAN"
        echo "ETCD server:  $ETCD_SERVER"
        echo "Processes:    $NUM_PROCS"
        echo ""

        export NIXL_ETCD_ENDPOINTS=$ETCD_SERVER

        LD_PRELOAD="${DOCA_PRELOAD}" \
        python3 $NIXL_ELASTIC_DIR/elastic.py \
            --plan $PLAN \
            --num-processes $NUM_PROCS \
            --etcd-server $ETCD_SERVER \
            $EXTRA_ARGS
        ;;

    *)
        echo "Unknown test mode: $TEST_MODE"
        echo ""
        echo "Available modes:"
        echo "  nvshmem_ht       NVSHMEM high-throughput internode"
        echo "  nvshmem_ll       NVSHMEM low-latency"
        echo "  nvshmem_intra    NVSHMEM intranode only"
        echo "  nixl_ht          NIXL high-throughput internode"
        echo "  nixl_ll          NIXL low-latency"
        echo "  nixl_elastic     NIXL elastic scaling"
        exit 1
        ;;
esac

echo ""
echo "=== Test complete ==="
