#!/bin/bash
#
# Compare NVSHMEM vs NIXL internode performance on a 2-node pair.
# Runs both High-Throughput and Low-Latency tests.
#
# Usage:
#   Node 0: bash scripts/compare_backends.sh <node0_ip> 0
#   Node 1: bash scripts/compare_backends.sh <node0_ip> 1
#
# Prerequisites:
#   - 2 nodes allocated via scripts/submit.sh, inside the container

set -uo pipefail

MASTER_IP=${1:?"Usage: compare_backends.sh <master_ip> <rank 0|1> [extra_test_args...]"}
NODE_RANK=${2:?"Need node rank (0 or 1)"}
shift 2
EXTRA_ARGS="${*:-}"

DEEPEP_DIR=${DEEPEP_DIR:-/workspace/deepep}
RESULTS_DIR=${RESULTS_DIR:-$DEEPEP_DIR/results}
mkdir -p "$RESULTS_DIR"

BASE_PORT=${MASTER_PORT:-8391}
NUM_PROCS=8
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

export WORLD_SIZE=2
export RANK=$NODE_RANK
export MASTER_ADDR=$MASTER_IP

# Log files for each phase
NS_HT_LOG="$RESULTS_DIR/nvshmem_ht_rank${NODE_RANK}_${TIMESTAMP}.log"
NS_LL_LOG="$RESULTS_DIR/nvshmem_ll_rank${NODE_RANK}_${TIMESTAMP}.log"
NX_HT_LOG="$RESULTS_DIR/nixl_ht_rank${NODE_RANK}_${TIMESTAMP}.log"
NX_LL_LOG="$RESULTS_DIR/nixl_ll_rank${NODE_RANK}_${TIMESTAMP}.log"

echo "============================================"
echo "  DeepEP Backend Comparison (HT + LL)"
echo "  Node rank: $NODE_RANK  Master: $MASTER_IP"
echo "============================================"
echo ""

PORT_OFFSET=0
next_port() {
    PORT_OFFSET=$((PORT_OFFSET + 1))
    export MASTER_PORT=$((BASE_PORT + PORT_OFFSET * 100))
}

run_phase() {
    local phase_name=$1 backend=$2 test_script=$3 test_dir=$4 log_file=$5
    shift 5
    local extra="$*"

    echo "▶ Phase: $phase_name"

    pkill -9 -f "test_internode.py" 2>/dev/null || true
    sleep 3

    cd "$DEEPEP_DIR"
    echo "  Building $backend backend..."
    if ! bash scripts/build.sh "$backend" 2>&1 | tee "$DEEPEP_DIR/build_${backend}.log" | tail -20; then
        echo "  ✗ $phase_name BUILD FAILED"
        echo "  Full build log: $DEEPEP_DIR/build_${backend}.log"
        echo "  Last 40 lines:"
        tail -40 "$DEEPEP_DIR/build_${backend}.log"
        return 1
    fi

    echo "  Running $phase_name test..."
    cd "$test_dir"
    export PYTHONPATH="${test_dir}:${PYTHONPATH:-}"

    if python3 "$test_script" --num-processes $NUM_PROCS $extra 2>&1 | tee "$log_file"; then
        echo "  ✓ $phase_name PASSED"
        return 0
    else
        echo "  ✗ $phase_name TEST FAILED"
        return 1
    fi
}

# ── Phase 1: NVSHMEM High-Throughput ─────────────────────────────
next_port
run_phase "NVSHMEM HT" nvshmem test_internode.py "$DEEPEP_DIR/tests" "$NS_HT_LOG" $EXTRA_ARGS || true

pkill -9 -f "test_internode.py" 2>/dev/null || true
sleep 5

# ── Phase 2: NVSHMEM High-Throughput + Low-Latency ───────────────
echo ""
next_port
run_phase "NVSHMEM LL" nvshmem test_internode.py "$DEEPEP_DIR/tests" "$NS_LL_LOG" --test-ll-compatibility $EXTRA_ARGS || true

pkill -9 -f "test_internode.py" 2>/dev/null || true
sleep 5

# ── Phase 3: NIXL High-Throughput ────────────────────────────────
echo ""
next_port

export NIXL_EP_NUM_CHANNELS=${NIXL_EP_NUM_CHANNELS:-10}
export UCX_RC_GDA_NUM_CHANNELS=${UCX_RC_GDA_NUM_CHANNELS:-10}

NIXL_EXTRA="$EXTRA_ARGS"
if [ "$NODE_RANK" = "1" ]; then
    NIXL_EXTRA="$NIXL_EXTRA --tcp-server $MASTER_IP"
fi

run_phase "NIXL HT" nixl test_internode.py "$DEEPEP_DIR/tests/nixl" "$NX_HT_LOG" $NIXL_EXTRA || true

pkill -9 -f "test_internode.py" 2>/dev/null || true
sleep 5

# ── Phase 4: NIXL Low-Latency ────────────────────────────────────
echo ""
next_port
run_phase "NIXL LL" nixl test_internode.py "$DEEPEP_DIR/tests/nixl" "$NX_LL_LOG" --test-ll-compatibility $NIXL_EXTRA || true

# ── Phase 5: Parse + Compare (rank 0 only) ───────────────────────
if [ "$NODE_RANK" != "0" ]; then
    echo ""
    echo "Logs: $NS_HT_LOG, $NS_LL_LOG, $NX_HT_LOG, $NX_LL_LOG"
    echo "Table printed on rank 0."
    exit 0
fi

echo ""
echo "Parsing results..."

extract() {
    local file=$1 pattern=$2 field=$3
    local line
    line=$(grep "$pattern" "$file" 2>/dev/null | head -1) || true
    [ -z "$line" ] && echo "—" && return

    case "$field" in
        transmit)
            local val
            val=$(echo "$line" | grep -oP 'transmit:\s*\K[0-9]+(\.[0-9]+)?' 2>/dev/null) || true
            if [ -z "$val" ]; then
                val=$(echo "$line" | grep -oP '\+\s*\K[0-9]+(\.[0-9]+)?' 2>/dev/null) || true
            fi
            echo "${val:-—}"
            ;;
        notify)
            local val
            val=$(echo "$line" | grep -oP 'notify:\s*\K[0-9]+(\.[0-9]+)?' 2>/dev/null) || true
            if [ -z "$val" ]; then
                val=$(echo "$line" | grep -oP '[0-9]+(\.[0-9]+)?(?=\s*\+\s*[0-9])' 2>/dev/null) || true
            fi
            echo "${val:-—}"
            ;;
        rdma_bw)
            local val
            val=$(echo "$line" | grep -oP '[0-9]+\.[0-9]+(?=\s*GB/s\s*\(RDMA\))' 2>/dev/null) || true
            echo "${val:-—}"
            ;;
        nvl_bw)
            local val
            val=$(echo "$line" | grep -oP '[0-9]+\.[0-9]+(?=\s*GB/s\s*\(NVL\))' 2>/dev/null) || true
            echo "${val:-—}"
            ;;
    esac
}

# LL output: "[rank 0] Dispatch bandwidth: X GB/s, avg_t=X us | Combine bandwidth: X GB/s, avg_t=X us"
extract_ll() {
    local file=$1 field=$2
    local line
    line=$(grep "\[rank 0\] Dispatch bandwidth:" "$file" 2>/dev/null | head -1) || true
    [ -z "$line" ] && echo "—" && return

    case "$field" in
        dispatch_bw)
            echo "$line" | grep -oP 'Dispatch bandwidth: \K[0-9]+\.[0-9]+' 2>/dev/null || echo "—"
            ;;
        dispatch_us)
            echo "$line" | grep -oP 'avg_t=\K[0-9]+\.[0-9]+' 2>/dev/null | head -1 || echo "—"
            ;;
        combine_bw)
            echo "$line" | grep -oP 'Combine bandwidth: \K[0-9]+\.[0-9]+' 2>/dev/null || echo "—"
            ;;
        combine_us)
            echo "$line" | grep -oP 'avg_t=\K[0-9]+\.[0-9]+' 2>/dev/null | tail -1 || echo "—"
            ;;
    esac
}

# ── HT metrics ──
NS_FP8_TX=$(extract "$NS_HT_LOG" "Best dispatch (FP8)" transmit)
NS_FP8_NT=$(extract "$NS_HT_LOG" "Best dispatch (FP8)" notify)
NS_FP8_RD=$(extract "$NS_HT_LOG" "Best dispatch (FP8)" rdma_bw)
NS_FP8_NV=$(extract "$NS_HT_LOG" "Best dispatch (FP8)" nvl_bw)

NX_FP8_TX=$(extract "$NX_HT_LOG" "Best dispatch (FP8)" transmit)
NX_FP8_NT=$(extract "$NX_HT_LOG" "Best dispatch (FP8)" notify)
NX_FP8_RD=$(extract "$NX_HT_LOG" "Best dispatch (FP8)" rdma_bw)
NX_FP8_NV=$(extract "$NX_HT_LOG" "Best dispatch (FP8)" nvl_bw)

NS_BF_TX=$(extract "$NS_HT_LOG" "Best dispatch (BF16)" transmit)
NS_BF_NT=$(extract "$NS_HT_LOG" "Best dispatch (BF16)" notify)
NS_BF_RD=$(extract "$NS_HT_LOG" "Best dispatch (BF16)" rdma_bw)
NS_BF_NV=$(extract "$NS_HT_LOG" "Best dispatch (BF16)" nvl_bw)

NX_BF_TX=$(extract "$NX_HT_LOG" "Best dispatch (BF16)" transmit)
NX_BF_NT=$(extract "$NX_HT_LOG" "Best dispatch (BF16)" notify)
NX_BF_RD=$(extract "$NX_HT_LOG" "Best dispatch (BF16)" rdma_bw)
NX_BF_NV=$(extract "$NX_HT_LOG" "Best dispatch (BF16)" nvl_bw)

NS_CO_TX=$(extract "$NS_HT_LOG" "Best combine" transmit)
NS_CO_NT=$(extract "$NS_HT_LOG" "Best combine" notify)
NS_CO_RD=$(extract "$NS_HT_LOG" "Best combine" rdma_bw)
NS_CO_NV=$(extract "$NS_HT_LOG" "Best combine" nvl_bw)

NX_CO_TX=$(extract "$NX_HT_LOG" "Best combine" transmit)
NX_CO_NT=$(extract "$NX_HT_LOG" "Best combine" notify)
NX_CO_RD=$(extract "$NX_HT_LOG" "Best combine" rdma_bw)
NX_CO_NV=$(extract "$NX_HT_LOG" "Best combine" nvl_bw)

# ── LL metrics ──
NS_LL_DISP_BW=$(extract_ll "$NS_LL_LOG" dispatch_bw)
NS_LL_DISP_US=$(extract_ll "$NS_LL_LOG" dispatch_us)
NS_LL_COMB_BW=$(extract_ll "$NS_LL_LOG" combine_bw)
NS_LL_COMB_US=$(extract_ll "$NS_LL_LOG" combine_us)

NX_LL_DISP_BW=$(extract_ll "$NX_LL_LOG" dispatch_bw)
NX_LL_DISP_US=$(extract_ll "$NX_LL_LOG" dispatch_us)
NX_LL_COMB_BW=$(extract_ll "$NX_LL_LOG" combine_bw)
NX_LL_COMB_US=$(extract_ll "$NX_LL_LOG" combine_us)

# ── Test pass counts ──
NS_HT_PASS=$(grep -c "passed" "$NS_HT_LOG" 2>/dev/null || echo 0)
NS_LL_PASS=$(grep -c "passed" "$NS_LL_LOG" 2>/dev/null || echo 0)
NX_HT_PASS=$(grep -c "passed" "$NX_HT_LOG" 2>/dev/null || echo 0)
NX_LL_PASS=$(grep -c "passed" "$NX_LL_LOG" 2>/dev/null || echo 0)

NS_LAYOUT=$(grep -oP '\[layout\] Kernel performance: \K[0-9]+\.[0-9]+' "$NS_HT_LOG" 2>/dev/null || echo "—")
NX_LAYOUT=$(grep -oP '\[layout\] Kernel performance: \K[0-9]+\.[0-9]+' "$NX_HT_LOG" 2>/dev/null || echo "—")

REPORT="$RESULTS_DIR/comparison_${TIMESTAMP}.txt"

{
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║          NVSHMEM vs NIXL — Internode Performance (HT + LL)                  ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Date:      $(date)"
    echo "  Host:      $(hostname)"
    echo "  Nodes:     2 × 8 GPUs"
    echo "  Channels:  NIXL_EP_NUM_CHANNELS=${NIXL_EP_NUM_CHANNELS:-?}, UCX_RC_GDA_NUM_CHANNELS=${UCX_RC_GDA_NUM_CHANNELS:-?}"
    echo ""
    echo "  ═══ HIGH-THROUGHPUT ═══"
    echo ""
    printf "  ┌──────────────────────────┬──────────────────────┬──────────────────────┐\n"
    printf "  │ %-24s │ %-20s │ %-20s │\n" "Metric" "NVSHMEM" "NIXL"
    printf "  ├──────────────────────────┼──────────────────────┼──────────────────────┤\n"
    printf "  │ %-24s │ %-20s │ %-20s │\n" "Tests passed" "$NS_HT_PASS" "$NX_HT_PASS"
    printf "  │ %-24s │ %-20s │ %-20s │\n" "Layout kernel (ms)" "$NS_LAYOUT" "$NX_LAYOUT"
    printf "  ├──────────────────────────┼──────────────────────┼──────────────────────┤\n"
    printf "  │ %-24s │ %20s │ %20s │\n" "DISPATCH FP8" "" ""
    printf "  │   %-22s │ %17s us │ %17s us │\n" "Transmit latency" "$NS_FP8_TX" "$NX_FP8_TX"
    printf "  │   %-22s │ %17s us │ %17s us │\n" "Notify latency" "$NS_FP8_NT" "$NX_FP8_NT"
    printf "  │   %-22s │ %14s GB/s │ %14s GB/s │\n" "RDMA bandwidth" "$NS_FP8_RD" "$NX_FP8_RD"
    printf "  │   %-22s │ %14s GB/s │ %14s GB/s │\n" "NVLink bandwidth" "$NS_FP8_NV" "$NX_FP8_NV"
    printf "  ├──────────────────────────┼──────────────────────┼──────────────────────┤\n"
    printf "  │ %-24s │ %20s │ %20s │\n" "DISPATCH BF16" "" ""
    printf "  │   %-22s │ %17s us │ %17s us │\n" "Transmit latency" "$NS_BF_TX" "$NX_BF_TX"
    printf "  │   %-22s │ %17s us │ %17s us │\n" "Notify latency" "$NS_BF_NT" "$NX_BF_NT"
    printf "  │   %-22s │ %14s GB/s │ %14s GB/s │\n" "RDMA bandwidth" "$NS_BF_RD" "$NX_BF_RD"
    printf "  │   %-22s │ %14s GB/s │ %14s GB/s │\n" "NVLink bandwidth" "$NS_BF_NV" "$NX_BF_NV"
    printf "  ├──────────────────────────┼──────────────────────┼──────────────────────┤\n"
    printf "  │ %-24s │ %20s │ %20s │\n" "COMBINE BF16" "" ""
    printf "  │   %-22s │ %17s us │ %17s us │\n" "Transmit latency" "$NS_CO_TX" "$NX_CO_TX"
    printf "  │   %-22s │ %17s us │ %17s us │\n" "Notify latency" "$NS_CO_NT" "$NX_CO_NT"
    printf "  │   %-22s │ %14s GB/s │ %14s GB/s │\n" "RDMA bandwidth" "$NS_CO_RD" "$NX_CO_RD"
    printf "  │   %-22s │ %14s GB/s │ %14s GB/s │\n" "NVLink bandwidth" "$NS_CO_NV" "$NX_CO_NV"
    printf "  └──────────────────────────┴──────────────────────┴──────────────────────┘\n"
    echo ""
    echo "  ═══ LOW-LATENCY ═══"
    echo ""
    printf "  ┌──────────────────────────┬──────────────────────┬──────────────────────┐\n"
    printf "  │ %-24s │ %-20s │ %-20s │\n" "Metric" "NVSHMEM" "NIXL"
    printf "  ├──────────────────────────┼──────────────────────┼──────────────────────┤\n"
    printf "  │ %-24s │ %-20s │ %-20s │\n" "Tests passed (HT+LL)" "$NS_LL_PASS" "$NX_LL_PASS"
    printf "  ├──────────────────────────┼──────────────────────┼──────────────────────┤\n"
    printf "  │   %-22s │ %17s us │ %17s us │\n" "Dispatch latency" "$NS_LL_DISP_US" "$NX_LL_DISP_US"
    printf "  │   %-22s │ %14s GB/s │ %14s GB/s │\n" "Dispatch bandwidth" "$NS_LL_DISP_BW" "$NX_LL_DISP_BW"
    printf "  │   %-22s │ %17s us │ %17s us │\n" "Combine latency" "$NS_LL_COMB_US" "$NX_LL_COMB_US"
    printf "  │   %-22s │ %14s GB/s │ %14s GB/s │\n" "Combine bandwidth" "$NS_LL_COMB_BW" "$NX_LL_COMB_BW"
    printf "  └──────────────────────────┴──────────────────────┴──────────────────────┘\n"
    echo ""
    echo "  Logs:"
    echo "    NVSHMEM HT: $NS_HT_LOG"
    echo "    NVSHMEM LL: $NS_LL_LOG"
    echo "    NIXL HT:    $NX_HT_LOG"
    echo "    NIXL LL:    $NX_LL_LOG"
    echo ""
} | tee "$REPORT"

echo "Report saved to: $REPORT"
