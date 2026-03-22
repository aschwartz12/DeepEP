#pragma once

#include "configs.cuh"

#ifdef USE_NIXL
#include "nixl_types.h"
#endif

#ifdef __CUDACC__
  #ifdef USE_NIXL
    #include "exception.cuh"
    #include "nixl_device.cuh"
  #else
    #include "ibgda_device.cuh"
  #endif
#else
  #ifndef USE_NIXL
    #include <cstdint>
    #ifndef nvshmem_team_t
    typedef int32_t nvshmem_team_t;
    #endif
  #endif
#endif

namespace deep_ep {

#ifdef USE_NIXL

// ─── NIXL transport ──────────────────────────────────────────────────────────

struct gpu_nixl_ctx {
    nixlMemViewH local_mvh;
    nixlMemViewH barrier_mvh;
    nixlMemViewH remote_mvh;
    nixlMemViewH internode_barrier_mvh;
    int *sync_buffer_ptr;
    int *sync_count_ptr;
    void *rdma_buffer_ptr;
    int max_num_ranks;
    int num_rdma_ranks;
    int rank;
    uint64_t *last_barrier_counter;
    uint64_t *local_barrier_counter_ptr;
};

using TransportCtx  = gpu_nixl_ctx;
using tp_counter_t  = uint64_t;

#ifdef __CUDACC__

__device__ __forceinline__ uint64_t tp_offset(const TransportCtx& ctx, uint64_t ptr) {
    return ptr - reinterpret_cast<uint64_t>(ctx.rdma_buffer_ptr);
}

__device__ __forceinline__ void* tp_p2p_ptr(const TransportCtx& ctx, uint64_t dst_ptr, int dst_rank) {
    if (dst_rank == ctx.rank) return reinterpret_cast<void*>(dst_ptr);
    void* base = nixlGetPtr(ctx.remote_mvh, dst_rank);
    if (base == nullptr) return nullptr;
    return reinterpret_cast<void*>(reinterpret_cast<uint64_t>(base) + tp_offset(ctx, dst_ptr));
}

__device__ __forceinline__ void tp_put_nbi_warp(
        const TransportCtx& ctx, uint64_t dst_ptr, uint64_t src_ptr, size_t size,
        int dst_rank, int qp_id, int /*lane_id*/, int /*slot_idx*/, bool flush) {
    nixlMemViewElem src_mdesc{ctx.local_mvh, 0, tp_offset(ctx, src_ptr)};
    nixlMemViewElem dst_mdesc{ctx.remote_mvh, static_cast<size_t>(dst_rank), tp_offset(ctx, dst_ptr)};
    EP_DEVICE_ASSERT(nixlPut<nixl_gpu_level_t::WARP>(
        src_mdesc, dst_mdesc, size, qp_id, flush ? 0 : nixl_gpu_flags::defer) == NIXL_IN_PROG);
}

__device__ __forceinline__ void tp_amo_add(
        const TransportCtx& ctx, uint64_t dst_ptr, int value, int dst_rank, int qp_id,
        bool is_local_copy = false) {
    if (is_local_copy) {
        atomicAdd(reinterpret_cast<unsigned long long*>(dst_ptr), static_cast<unsigned long long>(value));
    } else {
        nixlMemViewElem dst_mdesc{ctx.remote_mvh, static_cast<size_t>(dst_rank), tp_offset(ctx, dst_ptr)};
        EP_DEVICE_ASSERT(nixlAtomicAdd(static_cast<tp_counter_t>(value), dst_mdesc, qp_id) == NIXL_IN_PROG);
    }
}

// Quiet: noop for NIXL.
__device__ __forceinline__ void tp_quiet(const TransportCtx&, int, int) {}

template <bool kLowLatencyMode>
__device__ __forceinline__ void tp_quiet_all(const TransportCtx&, int, int, int, int, int) {}

__device__ __forceinline__ void tp_assert_qp_config(int, int) {}

__device__ __forceinline__ void tp_p2p_counter_store(void* ptr, int value) {
    st_release_sys_global(reinterpret_cast<tp_counter_t*>(ptr), static_cast<tp_counter_t>(value));
}

// NIXL reuses the same QPs for both data and head updates.
__device__ __forceinline__ int tp_head_qp(int channel_id, int /*num_channels*/) { return channel_id; }

// Internode barrier: warp-level send then thread-0 wait.
__device__ __forceinline__ void tp_internode_barrier_send(const TransportCtx& ctx, int num_channels) {
    int rdma_rank = ctx.rank / NUM_MAX_NVL_PEERS;
    int nvl_rank  = ctx.rank % NUM_MAX_NVL_PEERS;
    int lane_id   = get_lane_id();
    for (int j = lane_id; j < num_channels; j += 32) {
        for (int i = 0; i < ctx.num_rdma_ranks; i++) {
            if (i == rdma_rank) continue;
            int dst = i * NUM_MAX_NVL_PEERS + nvl_rank;
            nixlMemViewElem mdesc{ctx.internode_barrier_mvh, static_cast<size_t>(dst), 0};
            EP_DEVICE_ASSERT(nixlAtomicAdd<nixl_gpu_level_t::THREAD>(
                static_cast<uint64_t>(1), mdesc, j, 0) == NIXL_IN_PROG);
        }
    }
}

__device__ __forceinline__ void tp_internode_barrier_wait(const TransportCtx& ctx, int num_channels) {
    uint64_t epoch    = ld_acquire_sys_global(ctx.last_barrier_counter);
    uint64_t expected = (epoch + static_cast<uint64_t>(num_channels)) *
                        static_cast<uint64_t>(ctx.num_rdma_ranks - 1);
    while (ld_acquire_sys_global(ctx.local_barrier_counter_ptr) < expected);
    st_release_sys_global(ctx.last_barrier_counter, epoch + static_cast<uint64_t>(num_channels));
}

// All 32 threads of the designated warp call this.
template <bool kLowLatencyMode = true>
__device__ __forceinline__ void tp_internode_sync_warp(
        const TransportCtx& ctx, int lane_id, int num_channels) {
    tp_internode_barrier_send(ctx, num_channels);
    if (lane_id == 0)
        tp_internode_barrier_wait(ctx, num_channels);
}

// translate_dst: NIXL always maps to global (NVL-peer-mapped) rank.
template <bool kLowLatencyMode = true>
__forceinline__ __device__ int tp_translate_dst(int dst_rdma_rank, int nvl_rank) {
    return dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank;
}

// Clean-buffer barrier helpers.
template <int kNumThreads>
__device__ __forceinline__ void tp_barrier_quiet(const TransportCtx&, int) {}

__device__ __forceinline__ int tp_barrier_send(const TransportCtx& ctx, int dst_rank) {
    int expected_cnt = atomicAdd(ctx.sync_count_ptr + dst_rank, -1) - 1;
    nixlMemViewElem src_mdesc{ctx.local_mvh, 1, dst_rank * sizeof(int)};
    nixlMemViewElem dst_mdesc{ctx.barrier_mvh, static_cast<size_t>(dst_rank), ctx.rank * sizeof(int)};
    EP_DEVICE_ASSERT(nixlPut<nixl_gpu_level_t::THREAD>(src_mdesc, dst_mdesc, sizeof(int), 0) == NIXL_IN_PROG);
    return expected_cnt;
}

__device__ __forceinline__ int tp_barrier_num_ranks(const TransportCtx& ctx) { return ctx.max_num_ranks; }

__device__ __forceinline__ void tp_block_barrier_all() {}

#endif  // __CUDACC__

#else  // !USE_NIXL ──────────────────────────────────────────────────────────────────

// ─── NVSHMEM transport ───────────────────────────────────────────────────────

struct TransportCtx {
    nvshmem_team_t rdma_team;
    int rank;
    int num_ranks;
    int nvl_rank;
    int* sync_buffer_ptr;  // used only by clean_buffer barrier
};

using tp_counter_t = int;

#ifdef __CUDACC__

__device__ __forceinline__ void* tp_p2p_ptr(const TransportCtx& ctx, uint64_t dst_ptr, int dst_rank) {
    return reinterpret_cast<void*>(nvshmemi_get_p2p_ptr(dst_ptr, ctx.rank, dst_rank));
}

__device__ __forceinline__ void tp_put_nbi_warp(
        const TransportCtx&, uint64_t dst_ptr, uint64_t src_ptr, size_t size,
        int dst_rank, int qp_id, int lane_id, int slot_idx, bool /*flush*/) {
    nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr, src_ptr, size, dst_rank, qp_id, lane_id, slot_idx);
}

__device__ __forceinline__ void tp_amo_add(
        const TransportCtx&, uint64_t dst_ptr, int value, int dst_rank, int qp_id,
        bool is_local_copy = false) {
    nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), value, dst_rank, qp_id,
                                    is_local_copy);
}

__device__ __forceinline__ void tp_quiet(const TransportCtx&, int dst_rank, int qp_id) {
    nvshmemi_ibgda_quiet(dst_rank, qp_id);
}

__device__ __forceinline__ void tp_assert_qp_config(int expected, int min_val) {
    EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe == expected ||
                     ibgda_get_state()->num_rc_per_pe >= min_val);
}

__device__ __forceinline__ void tp_p2p_counter_store(void* ptr, int value) {
    st_release_sys_global(reinterpret_cast<tp_counter_t*>(ptr), static_cast<tp_counter_t>(value));
}

// NVSHMEM uses a separate QP range for head updates.
__device__ __forceinline__ int tp_head_qp(int channel_id, int num_channels) { return channel_id + num_channels; }

template <bool kLowLatencyMode>
__forceinline__ __device__ void nvshmem_sync_with_same_gpu_idx(const nvshmem_team_t& rdma_team) {
    kLowLatencyMode ? void(nvshmem_sync(rdma_team)) : nvshmem_sync_all();
}

// Internode barrier: only lane 0 calls nvshmem_sync; others are noops.
template <bool kLowLatencyMode>
__device__ __forceinline__ void tp_internode_sync_warp(
        const TransportCtx& ctx, int lane_id, int /*num_channels*/) {
    if (lane_id == 0)
        nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(ctx.rdma_team);
}

// translate_dst: LL mode uses global rank; normal mode uses rdma rank only.
template <bool kLowLatencyMode>
__forceinline__ __device__ int tp_translate_dst(int dst_rdma_rank, int nvl_rank) {
    return kLowLatencyMode ? (dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank) : dst_rdma_rank;
}

template <bool kLowLatencyMode>
__device__ __forceinline__ void tp_quiet_all(
        const TransportCtx&, int thread_id, int num_threads,
        int rdma_rank, int nvl_rank, int num_rdma_ranks) {
    auto qps_per_rdma_rank = ibgda_get_state()->num_rc_per_pe * ibgda_get_state()->num_devices_initialized;
    for (int i = thread_id; i < qps_per_rdma_rank * (num_rdma_ranks - 1); i += num_threads) {
        auto dst_rdma_rank = (i / qps_per_rdma_rank + rdma_rank + 1) % num_rdma_ranks;
        auto qp_id = i % qps_per_rdma_rank;
        nvshmemi_ibgda_quiet(tp_translate_dst<kLowLatencyMode>(dst_rdma_rank, nvl_rank), qp_id);
    }
}

// Clean-buffer barrier helpers.
template <int kNumThreads>
__device__ __forceinline__ void tp_barrier_quiet(const TransportCtx& ctx, int thread_id) {
    auto qps_per_rank = ibgda_get_state()->num_rc_per_pe * ibgda_get_state()->num_devices_initialized;
    for (int i = thread_id; i < qps_per_rank * (ctx.num_ranks - 1); i += kNumThreads) {
        auto dst_rank = (ctx.rank + 1 + i / qps_per_rank) % ctx.num_ranks;
        auto qp_id = i % qps_per_rank;
        nvshmemi_ibgda_quiet(dst_rank, qp_id);
    }
    if (thread_id == 0)
        atomicAdd(ctx.sync_buffer_ptr + ctx.rank, -1);
    __syncthreads();
}

__device__ __forceinline__ int tp_barrier_send(const TransportCtx& ctx, int dst_rank) {
    int cnt = ctx.sync_buffer_ptr[ctx.rank];
    auto dst_ptr = reinterpret_cast<uint64_t>(ctx.sync_buffer_ptr + ctx.rank);
    auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, ctx.rank, dst_rank);
    if (dst_p2p_ptr == 0) {
        nvshmemi_ibgda_rma_p(reinterpret_cast<int*>(dst_ptr), cnt, dst_rank, 0);
    } else {
        st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), cnt);
    }
    return cnt;
}

__device__ __forceinline__ int tp_barrier_num_ranks(const TransportCtx& ctx) { return ctx.num_ranks; }

__device__ __forceinline__ void tp_block_barrier_all() { nvshmemx_barrier_all_block(); }

#endif  // __CUDACC__

#endif  // USE_NIXL

}  // namespace deep_ep
