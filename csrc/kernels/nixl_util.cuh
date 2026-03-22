#pragma once

#include "configs.cuh"
#include "exception.cuh"
#include "utils.cuh"

#ifdef USE_NIXL
#include "nixl_device.cuh"
#endif

#ifdef USE_NIXL
#define NIXL_CTX_TRAILING_PARAM , gpu_nixl_ctx nixl_ctx
#define NIXL_CTX_TRAILING_ARG , nixl_ctx
#else
#define NIXL_CTX_TRAILING_PARAM
#define NIXL_CTX_TRAILING_ARG
#endif

namespace deep_ep {

#ifdef USE_NIXL

namespace nixl_util {

__device__ __forceinline__ void* get_p2p_ptr(
        const gpu_nixl_ctx& ctx, uint64_t dst_ptr, int dst_rank, int rank) {
    if (dst_rank == rank) return reinterpret_cast<void*>(dst_ptr);
    void* base = nixlGetPtr(ctx.remote_mvh, dst_rank);
    if (base == nullptr) return nullptr;
    return reinterpret_cast<void*>(reinterpret_cast<uint64_t>(base) +
        (dst_ptr - reinterpret_cast<uint64_t>(ctx.rdma_buffer_ptr)));
}

template <int kNumThreads>
__forceinline__ __device__ void barrier(gpu_nixl_ctx nixl_ctx, int* mask_buffer_ptr, int thread_id, int rank, int num_ranks) {
    EP_DEVICE_ASSERT(kNumThreads >= num_ranks);
    if (thread_id < num_ranks && thread_id != rank) {
        const auto dst_rank = thread_id;
        if (not is_rank_masked(mask_buffer_ptr, dst_rank)) {
            int expected_cnt = atomicAdd(nixl_ctx.sync_count_ptr + dst_rank, -1) - 1;
            nixlMemViewElem src_mdesc{nixl_ctx.local_mvh, 1, static_cast<size_t>(dst_rank * sizeof(int))};
            nixlMemViewElem dst_mdesc{nixl_ctx.barrier_mvh, static_cast<size_t>(dst_rank),
                                      static_cast<size_t>(rank * sizeof(int))};
            EP_DEVICE_ASSERT(nixlPut<nixl_gpu_level_t::THREAD>(src_mdesc, dst_mdesc, sizeof(int), 0) == NIXL_IN_PROG);
            auto start_time = clock64();
            uint64_t wait_recv_cost = 0;
            while (ld_acquire_sys_global(nixl_ctx.sync_buffer_ptr + dst_rank) > expected_cnt
                   && (wait_recv_cost = clock64() - start_time) <= NUM_TIMEOUT_CYCLES);
            if (wait_recv_cost > NUM_TIMEOUT_CYCLES) {
                printf("Warning: DeepEP timeout for barrier, rank %d, thread %d, dst_rank %d\n",
                       rank, thread_id, dst_rank);
                if (mask_buffer_ptr == nullptr) trap();
                atomicExch(mask_buffer_ptr + dst_rank, 1);
            }
        }
    }
    __syncthreads();
}

template <bool kLowLatencyMode>
__device__ __forceinline__ void internode_sync_warp(
        const gpu_nixl_ctx& ctx, int lane_id, int num_channels, int rank, int num_rdma_ranks) {
    int lane_rdma_rank = rank / NUM_MAX_NVL_PEERS;
    int lane_nvl_rank  = rank % NUM_MAX_NVL_PEERS;
    for (int j = lane_id; j < num_channels; j += 32) {
        for (int i = 0; i < num_rdma_ranks; i++) {
            if (i == lane_rdma_rank) continue;
            int dst = i * NUM_MAX_NVL_PEERS + lane_nvl_rank;
            nixlMemViewElem mdesc{ctx.internode_barrier_mvh, static_cast<size_t>(dst), 0};
            EP_DEVICE_ASSERT(nixlAtomicAdd<nixl_gpu_level_t::THREAD>(
                static_cast<uint64_t>(1), mdesc, j, 0) == NIXL_IN_PROG);
        }
    }
    if (lane_id == 0) {
        uint64_t epoch    = ld_acquire_sys_global(ctx.last_barrier_counter);
        uint64_t expected = (epoch + static_cast<uint64_t>(num_channels)) *
                            static_cast<uint64_t>(num_rdma_ranks - 1);
        while (ld_acquire_sys_global(ctx.local_barrier_counter_ptr) < expected);
        st_release_sys_global(ctx.last_barrier_counter, epoch + static_cast<uint64_t>(num_channels));
    }
}

}  // namespace nixl_util

#endif  // USE_NIXL

}  // namespace deep_ep
