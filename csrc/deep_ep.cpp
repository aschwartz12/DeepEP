#include "deep_ep.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <cuda_runtime.h>
#include <pybind11/functional.h>
#include <torch/python.h>

#include <chrono>
#include <memory>

#ifdef USE_NIXL
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <cstdio>
#include <fstream>
#include <unistd.h>
#include <stdio.h>
#include "nixl.h"
#include <ifaddrs.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <sstream>
#include <unordered_set>
#endif

#include "kernels/api.cuh"
#include "kernels/configs.cuh"

#ifndef USE_NIXL
namespace shared_memory {
void cu_mem_set_access_all(void* ptr, size_t size) {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    CUmemAccessDesc access_desc[device_count];
    for (int idx = 0; idx < device_count; ++idx) {
        access_desc[idx].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc[idx].location.id = idx;
        access_desc[idx].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }

    CU_CHECK(cuMemSetAccess((CUdeviceptr)ptr, size, access_desc, device_count));
}

void cu_mem_free(void* ptr) {
    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));

    size_t size = 0;
    CU_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

    CU_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemRelease(handle));
}

size_t get_size_align_to_granularity(size_t size_raw, size_t granularity) {
    size_t size = (size_raw + granularity - 1) & ~(granularity - 1);
    if (size == 0)
        size = granularity;
    return size;
}

SharedMemoryAllocator::SharedMemoryAllocator(bool use_fabric) : use_fabric(use_fabric) {}

void SharedMemoryAllocator::malloc(void** ptr, size_t size_raw) {
    if (use_fabric) {
        CUdevice device;
        CU_CHECK(cuCtxGetDevice(&device));

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        prop.location.id = device;

        size_t granularity = 0;
        CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

        size_t size = get_size_align_to_granularity(size_raw, granularity);

        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemCreate(&handle, size, &prop, 0));

        CU_CHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));
        CU_CHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
        cu_mem_set_access_all(*ptr, size);
    } else {
        CUDA_CHECK(cudaMalloc(ptr, size_raw));
    }
}

void SharedMemoryAllocator::free(void* ptr) {
    if (use_fabric) {
        cu_mem_free(ptr);
    } else {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void SharedMemoryAllocator::get_mem_handle(MemHandle* mem_handle, void* ptr) {
    size_t size = 0;
    CU_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

    mem_handle->size = size;

    if (use_fabric) {
        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));

        CU_CHECK(cuMemExportToShareableHandle(&mem_handle->inner.cu_mem_fabric_handle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    } else {
        CUDA_CHECK(cudaIpcGetMemHandle(&mem_handle->inner.cuda_ipc_mem_handle, ptr));
    }
}

void SharedMemoryAllocator::open_mem_handle(void** ptr, MemHandle* mem_handle) {
    if (use_fabric) {
        size_t size = mem_handle->size;

        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemImportFromShareableHandle(&handle, &mem_handle->inner.cu_mem_fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC));

        CU_CHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, 0, 0, 0));
        CU_CHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
        cu_mem_set_access_all(*ptr, size);
    } else {
        CUDA_CHECK(cudaIpcOpenMemHandle(ptr, mem_handle->inner.cuda_ipc_mem_handle, cudaIpcMemLazyEnablePeerAccess));
    }
}

void SharedMemoryAllocator::close_mem_handle(void* ptr) {
    if (use_fabric) {
        cu_mem_free(ptr);
    } else {
        CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
    }
}
}  // namespace shared_memory
#endif // !USE_NIXL

namespace deep_ep {

#ifdef USE_NIXL
#define NIXL_ETCD_WATCH_TIMEOUT std::chrono::microseconds(1000000000) // 1000 seconds

static void sleep_ms(int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}
#endif // USE_NIXL

#ifdef USE_NIXL
Buffer::Buffer(int rank, bool low_latency_mode, bool explicitly_destroy)
    : low_latency_mode(low_latency_mode),
      rank(rank), num_ranks(1),
      explicitly_destroy(explicitly_destroy),
      comm_stream(at::cuda::getStreamFromPool(true)) {}

void Buffer::init(int num_ranks, int num_experts_per_rank, int64_t num_nvl_bytes, int64_t num_rdma_bytes) {
    this->max_num_ranks = num_ranks;
    this->max_experts_per_rank = num_experts_per_rank;
    this->num_nvl_bytes = num_nvl_bytes;
    this->num_rdma_bytes = num_rdma_bytes;

    int64_t barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int);
    int64_t buffer_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(void*);
    int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int*);

    EP_STATIC_ASSERT(NUM_BUFFER_ALIGNMENT_BYTES % sizeof(int4) == 0, "Invalid alignment");
    EP_HOST_ASSERT(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                   (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));
    EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                   (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));
    EP_HOST_ASSERT(0 <= rank and rank < num_ranks and
                   (num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS or low_latency_mode));
    EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);
    if (num_rdma_bytes > 0)
        EP_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode);

    CUDA_CHECK(cudaGetDevice(&device_id));
    rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS), num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);

    cudaDeviceProp device_prop = {};
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
    num_device_sms = device_prop.multiProcessorCount;
    int denom_sms = std::max(1, num_device_sms / 2);
    auto per_channel_bytes = ceil_div<int64_t>(num_rdma_bytes, denom_sms);
    EP_HOST_ASSERT(per_channel_bytes < std::numeric_limits<int>::max());

    if (num_nvl_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&buffer_ptrs[nvl_rank],
                              num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes));
        CUDA_CHECK(cudaIpcGetMemHandle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]));
        buffer_ptrs_gpu = reinterpret_cast<void**>(
            static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes);

        barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(
            static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
        barrier_signal_ptrs_gpu = reinterpret_cast<int**>(
            static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);

        CUDA_CHECK(cudaMemsetAsync(barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes, comm_stream));
    }

    CUDA_CHECK(cudaMalloc(&workspace, NUM_WORKSPACE_BYTES));
    CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));

    CUDA_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));
    *moe_recv_counter = -1;

    CUDA_CHECK(cudaMallocHost(&moe_recv_expert_counter, sizeof(int) * NUM_MAX_LOCAL_EXPERTS, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped, const_cast<int*>(moe_recv_expert_counter), 0));
    for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i)
        moe_recv_expert_counter[i] = -1;

    CUDA_CHECK(cudaMallocHost(&moe_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_rdma_counter_mapped, const_cast<int*>(moe_recv_rdma_counter), 0));
    *moe_recv_rdma_counter = -1;

    EP_HOST_ASSERT(max_experts_per_rank > 0);
    CUDA_CHECK(cudaMalloc(&rdma_buffer_ptr, num_rdma_bytes));
    CUDA_CHECK(cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes));

    mask_buffer_ptr = nullptr;
    sync_buffer_ptr = nullptr;
    int num_mask_buffer_bytes = max_num_ranks * sizeof(int);
    CUDA_CHECK(cudaMalloc(&mask_buffer_ptr, num_mask_buffer_bytes));
    CUDA_CHECK(cudaMemset(mask_buffer_ptr, 0xff, num_mask_buffer_bytes));
    CUDA_CHECK(cudaMemset(mask_buffer_ptr + rank, 0, sizeof(int)));

    int num_sync_buffer_bytes = max_num_ranks * sizeof(int);
    CUDA_CHECK(cudaMalloc(&sync_buffer_ptr, num_sync_buffer_bytes));
    CUDA_CHECK(cudaMemset(sync_buffer_ptr, 0, num_sync_buffer_bytes));
    CUDA_CHECK(cudaMalloc(&sync_count_ptr, num_sync_buffer_bytes));
    CUDA_CHECK(cudaMemset(sync_count_ptr, 0, num_sync_buffer_bytes));
    CUDA_CHECK(cudaMalloc(&local_barrier_cnt_ptr, num_sync_buffer_bytes));
    CUDA_CHECK(cudaMemset(local_barrier_cnt_ptr, 0, num_sync_buffer_bytes));

    CUDA_CHECK(cudaMalloc(&local_barrier_counter, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(local_barrier_counter, 0, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&last_barrier_counter, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(last_barrier_counter, 0, sizeof(uint64_t)));
    CUDA_CHECK(cudaDeviceSynchronize());

    my_peer_info.rdma_buffer_ptr = rdma_buffer_ptr;
    my_peer_info.device_id = get_local_device_id();
    my_peer_info.sync_buffer_ptr = sync_buffer_ptr;
    my_peer_info.barrier_ptr = local_barrier_counter;
    my_peer_info.rank = rank;

    nixl_peer_info.resize(max_num_ranks);
    nixl_peer_info[rank] = my_peer_info;

    _nixl_agent_init();
    _nixl_ep_init();
}

void Buffer::update_memory_buffers(int num_ranks, int num_experts_per_rank, int64_t num_nvl_bytes, int64_t num_rdma_bytes) {
    if (!available) {
        init(num_ranks, num_experts_per_rank, num_nvl_bytes, num_rdma_bytes);
        available = true;
    } else {
        throw std::runtime_error("Multiple calls to update_memory_buffers are not supported");
    }
}

void Buffer::connect_ranks(const std::vector<int>& remote_ranks_list,
                           const std::optional<std::vector<nixl_blob_t>>& remote_mds,
                           const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles) {
    EP_HOST_ASSERT(!remote_ranks_list.empty());
    EP_HOST_ASSERT(!remote_mds.has_value() || remote_mds->size() == remote_ranks_list.size());

    std::vector<int> new_ranks;
    std::vector<nixl_blob_t> new_ranks_mds;
    int max_added_rank = std::max(rank, *std::max_element(remote_ranks_list.begin(), remote_ranks_list.end()));
    num_ranks = std::max(num_ranks, max_added_rank + 1);

    if (all_gathered_handles.size() > 0)
        _ipc_handles_sync(all_gathered_handles);

    for (size_t i = 0; i < remote_ranks_list.size(); i++) {
        int remote_rank = remote_ranks_list[i];
        if (remote_rank == rank or
            std::find(remote_ranks.begin(), remote_ranks.end(), remote_rank) != remote_ranks.end())
            continue;

        new_ranks.push_back(remote_rank);
        CUDA_CHECK(cudaMemset(mask_buffer_ptr + remote_rank, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(sync_count_ptr + remote_rank, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(local_barrier_cnt_ptr + remote_rank, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(sync_buffer_ptr + remote_rank, 0, sizeof(int)));

        if (remote_mds.has_value())
            new_ranks_mds.push_back((*remote_mds)[i]);
    }

    if (new_ranks.empty())
        return;

    _nixl_agents_connect(new_ranks, new_ranks_mds);
    _nixl_agents_peer_info_gather(new_ranks);
    _nixl_ep_memory_views_destroy();
    _nixl_ep_memory_views_create();

    CUDA_CHECK(cudaDeviceSynchronize());
    available = true;
}

void Buffer::disconnect_ranks(const std::vector<int>& remote_ranks_list) {
    EP_HOST_ASSERT(!remote_ranks_list.empty());
    EP_HOST_ASSERT(remote_ranks_list.size() <= remote_ranks.size());

    CUDA_CHECK(cudaDeviceSynchronize());

    for (int removed_rank : remote_ranks_list) {
        update_mask_buffer(removed_rank, true);
    }

    _nixl_ep_memory_views_destroy();
    _nixl_ep_memory_views_create();
    _nixl_agents_peer_info_cleanup(remote_ranks_list);
    _nixl_agents_disconnect(remote_ranks_list);

    for (int removed_rank : remote_ranks_list) {
        remote_ranks.erase(
            std::remove(remote_ranks.begin(), remote_ranks.end(), removed_rank),
            remote_ranks.end());
    }

    int max_rank = rank;
    if (!remote_ranks.empty()) {
        max_rank = std::max(max_rank, *std::max_element(remote_ranks.begin(), remote_ranks.end()));
    }
    num_ranks = max_rank + 1;
}

#else // !USE_NIXL

Buffer::Buffer(int rank,
               int num_ranks,
               int64_t num_nvl_bytes,
               int64_t num_rdma_bytes,
               bool low_latency_mode,
               bool explicitly_destroy,
               bool enable_shrink,
               bool use_fabric)
    : rank(rank),
      num_ranks(num_ranks),
      num_nvl_bytes(num_nvl_bytes),
      num_rdma_bytes(num_rdma_bytes),
      enable_shrink(enable_shrink),
      low_latency_mode(low_latency_mode),
      explicitly_destroy(explicitly_destroy),
      comm_stream(at::cuda::getStreamFromPool(true)),
      shared_memory_allocator(use_fabric) {
    int64_t barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int);
    int64_t buffer_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(void*);
    int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int*);

    EP_STATIC_ASSERT(NUM_BUFFER_ALIGNMENT_BYTES % sizeof(int4) == 0, "Invalid alignment");
    EP_HOST_ASSERT(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                   (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));
    EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                   (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));
    EP_HOST_ASSERT(num_nvl_bytes / sizeof(int4) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_rdma_bytes / sizeof(int4) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(0 <= rank and rank < num_ranks and (num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS or low_latency_mode));
    EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);
    if (num_rdma_bytes > 0)
        EP_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode);

    CUDA_CHECK(cudaGetDevice(&device_id));
    rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS), num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);
#ifdef DISABLE_NVSHMEM
    EP_HOST_ASSERT(num_rdma_ranks == 1 and not low_latency_mode and "NVSHMEM is disabled during compilation");
#endif

    cudaDeviceProp device_prop = {};
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
    num_device_sms = device_prop.multiProcessorCount;

    EP_HOST_ASSERT(ceil_div<int64_t>(num_nvl_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(ceil_div<int64_t>(num_rdma_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());

    if (num_nvl_bytes > 0) {
        shared_memory_allocator.malloc(&buffer_ptrs[nvl_rank],
                                       num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes);
        shared_memory_allocator.get_mem_handle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]);
        buffer_ptrs_gpu = reinterpret_cast<void**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes);

        barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
        barrier_signal_ptrs_gpu =
            reinterpret_cast<int**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);

        CUDA_CHECK(cudaMemsetAsync(barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes, comm_stream));
    }

    CUDA_CHECK(cudaMalloc(&workspace, NUM_WORKSPACE_BYTES));
    CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));

    CUDA_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));
    *moe_recv_counter = -1;

    CUDA_CHECK(cudaMallocHost(&moe_recv_expert_counter, sizeof(int) * NUM_MAX_LOCAL_EXPERTS, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped, const_cast<int*>(moe_recv_expert_counter), 0));
    for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i)
        moe_recv_expert_counter[i] = -1;

    if (num_rdma_ranks > 0) {
        CUDA_CHECK(cudaMallocHost(&moe_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_rdma_counter_mapped, const_cast<int*>(moe_recv_rdma_counter), 0));
        *moe_recv_rdma_counter = -1;
    }
}
#endif // USE_NIXL

Buffer::~Buffer() noexcept(false) {
    if (not explicitly_destroy) {
        destroy();
    } else if (not destroyed) {
        printf("WARNING: destroy() was not called before DeepEP buffer destruction, which can leak resources.\n");
        fflush(stdout);
    }
}

bool Buffer::is_available() const {
    return available;
}

bool Buffer::is_internode_available() const {
    return is_available() and num_ranks > NUM_MAX_NVL_PEERS;
}

int Buffer::get_num_rdma_ranks() const {
    return num_rdma_ranks;
}

int Buffer::get_rdma_rank() const {
    return rdma_rank;
}

int Buffer::get_root_rdma_rank(bool global) const {
    return global ? nvl_rank : 0;
}

int Buffer::get_local_device_id() const {
    return device_id;
}

pybind11::bytearray Buffer::get_local_ipc_handle() const {
#ifdef USE_NIXL
    return {ipc_handles[nvl_rank].reserved, CUDA_IPC_HANDLE_SIZE};
#else
    const shared_memory::MemHandle& handle = ipc_handles[nvl_rank];
    return {reinterpret_cast<const char*>(&handle), sizeof(handle)};
#endif
}

#ifndef USE_NIXL
pybind11::bytearray Buffer::get_local_nvshmem_unique_id() const {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(rdma_rank == 0 and "Only RDMA rank 0 can get NVSHMEM unique ID");
    auto unique_id = internode::get_unique_id();
    return {reinterpret_cast<const char*>(unique_id.data()), unique_id.size()};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
#endif
}
#endif // !USE_NIXL

torch::Tensor Buffer::get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const {
    torch::ScalarType casted_dtype = torch::python::detail::py_object_to_dtype(dtype);
    auto element_bytes = static_cast<int64_t>(elementSize(casted_dtype));
    auto base_ptr = static_cast<uint8_t*>(use_rdma_buffer ? rdma_buffer_ptr : buffer_ptrs[nvl_rank]) + offset;
    auto num_bytes = use_rdma_buffer ? num_rdma_bytes : num_nvl_bytes;
    return torch::from_blob(base_ptr, num_bytes / element_bytes, torch::TensorOptions().dtype(casted_dtype).device(at::kCUDA));
}

torch::Stream Buffer::get_comm_stream() const {
    return comm_stream;
}

void Buffer::destroy() {
    EP_HOST_ASSERT(not destroyed);

    CUDA_CHECK(cudaDeviceSynchronize());

#ifdef USE_NIXL
    _nixl_ep_destroy();

    if (num_nvl_bytes > 0) {
        intranode::barrier(barrier_signal_ptrs_gpu, nvl_rank, num_nvl_ranks, comm_stream);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (is_available()) {
            for (int i = 0; i < num_nvl_ranks; ++i)
                if (i != nvl_rank)
                    CUDA_CHECK(cudaIpcCloseMemHandle(buffer_ptrs[i]));
        }

        CUDA_CHECK(cudaFree(buffer_ptrs[nvl_rank]));
    }

    cudaFree(rdma_buffer_ptr);

    if (nixl_agent_info and nixl_agent_info->agent != nullptr and getenv("NIXL_ETCD_ENDPOINTS")) {
        nixl_agent_info->agent->invalidateLocalMD();
    }

    rdma_buffer_ptr = nullptr;

    cudaFree(mask_buffer_ptr);
    cudaFree(sync_buffer_ptr);
    cudaFree(sync_count_ptr);
    cudaFree(local_barrier_cnt_ptr);
    cudaFree(local_barrier_counter);
    cudaFree(last_barrier_counter);
#else
    if (num_nvl_bytes > 0) {
        intranode::barrier(barrier_signal_ptrs_gpu, nvl_rank, num_nvl_ranks, comm_stream);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (is_available()) {
            for (int i = 0; i < num_nvl_ranks; ++i)
                if (i != nvl_rank)
                    shared_memory_allocator.close_mem_handle(buffer_ptrs[i]);
        }

        shared_memory_allocator.free(buffer_ptrs[nvl_rank]);
    }

#ifndef DISABLE_NVSHMEM
    if (is_available() and num_rdma_bytes > 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
        internode::barrier();
        internode::free(rdma_buffer_ptr);
        if (enable_shrink) {
            internode::free(mask_buffer_ptr);
            internode::free(sync_buffer_ptr);
        }
        internode::finalize();
    }
#endif
#endif // USE_NIXL

    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_counter)));

    CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_expert_counter)));

    destroyed = true;
    available = false;
}

#ifndef USE_NIXL
void Buffer::sync(const std::vector<int>& device_ids,
                  const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
                  const std::optional<pybind11::bytearray>& root_unique_id_opt) {
    EP_HOST_ASSERT(not is_available());

    if (num_nvl_bytes > 0) {
        EP_HOST_ASSERT(num_ranks == device_ids.size());
        EP_HOST_ASSERT(device_ids.size() == all_gathered_handles.size());
        for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++i) {
            EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
            auto handle_str = std::string(all_gathered_handles[offset + i].value());
            EP_HOST_ASSERT(handle_str.size() == shared_memory::HANDLE_SIZE);
            if (offset + i != rank) {
                std::memcpy(&ipc_handles[i], handle_str.c_str(), shared_memory::HANDLE_SIZE);
                shared_memory_allocator.open_mem_handle(&buffer_ptrs[i], &ipc_handles[i]);
                barrier_signal_ptrs[i] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
            } else {
                EP_HOST_ASSERT(std::memcmp(&ipc_handles[i], handle_str.c_str(), shared_memory::HANDLE_SIZE) == 0);
            }
        }

        CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs, sizeof(int*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

#ifndef DISABLE_NVSHMEM
    if (num_rdma_bytes > 0) {
        EP_HOST_ASSERT(root_unique_id_opt.has_value());
        std::vector<uint8_t> root_unique_id(root_unique_id_opt->size());
        auto root_unique_id_str = root_unique_id_opt->cast<std::string>();
        std::memcpy(root_unique_id.data(), root_unique_id_str.c_str(), root_unique_id_opt->size());
        auto nvshmem_rank = low_latency_mode ? rank : rdma_rank;
        auto num_nvshmem_ranks = low_latency_mode ? num_ranks : num_rdma_ranks;
        EP_HOST_ASSERT(nvshmem_rank == internode::init(root_unique_id, nvshmem_rank, num_nvshmem_ranks, low_latency_mode));
        internode::barrier();

        rdma_buffer_ptr = internode::alloc(num_rdma_bytes, NUM_BUFFER_ALIGNMENT_BYTES);

        CUDA_CHECK(cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes));

        if (enable_shrink) {
            int num_mask_buffer_bytes = num_ranks * sizeof(int);
            int num_sync_buffer_bytes = num_ranks * sizeof(int);
            mask_buffer_ptr = reinterpret_cast<int*>(internode::alloc(num_mask_buffer_bytes, NUM_BUFFER_ALIGNMENT_BYTES));
            sync_buffer_ptr = reinterpret_cast<int*>(internode::alloc(num_sync_buffer_bytes, NUM_BUFFER_ALIGNMENT_BYTES));
            CUDA_CHECK(cudaMemset(mask_buffer_ptr, 0, num_mask_buffer_bytes));
            CUDA_CHECK(cudaMemset(sync_buffer_ptr, 0, num_sync_buffer_bytes));
        }

        internode::barrier();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
#endif

    available = true;
}
#endif // !USE_NIXL

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
Buffer::get_dispatch_layout(
    const torch::Tensor& topk_idx, int num_experts, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(topk_idx.dim() == 2);
    EP_HOST_ASSERT(topk_idx.is_contiguous());
    EP_HOST_ASSERT(num_experts > 0);

    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::cuda::setCurrentCUDAStream(comm_stream);
    }

    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    auto num_tokens = static_cast<int>(topk_idx.size(0)), num_topk = static_cast<int>(topk_idx.size(1));
    auto num_tokens_per_rank = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
    auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
    auto num_tokens_per_expert = torch::empty({num_experts}, dtype(torch::kInt32).device(torch::kCUDA));
    auto is_token_in_rank = torch::empty({num_tokens, num_ranks}, dtype(torch::kBool).device(torch::kCUDA));
    if (is_internode_available())
        num_tokens_per_rdma_rank = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

    layout::get_dispatch_layout(topk_idx.data_ptr<topk_idx_t>(),
                                num_tokens_per_rank.data_ptr<int>(),
                                num_tokens_per_rdma_rank.has_value() ? num_tokens_per_rdma_rank.value().data_ptr<int>() : nullptr,
                                num_tokens_per_expert.data_ptr<int>(),
                                is_token_in_rank.data_ptr<bool>(),
                                num_tokens,
                                num_topk,
                                num_ranks,
                                num_experts,
                                comm_stream);

    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {topk_idx, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {num_tokens_per_rdma_rank}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(compute_stream);

    return {num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event};
}

#ifndef USE_NIXL
std::tuple<torch::Tensor,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::vector<int>,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           std::optional<EventHandle>>
Buffer::intranode_dispatch(const torch::Tensor& x,
                           const std::optional<torch::Tensor>& x_scales,
                           const std::optional<torch::Tensor>& topk_idx,
                           const std::optional<torch::Tensor>& topk_weights,
                           const std::optional<torch::Tensor>& num_tokens_per_rank,
                           const torch::Tensor& is_token_in_rank,
                           const std::optional<torch::Tensor>& num_tokens_per_expert,
                           int cached_num_recv_tokens,
                           const std::optional<torch::Tensor>& cached_rank_prefix_matrix,
                           const std::optional<torch::Tensor>& cached_channel_prefix_matrix,
                           int expert_alignment,
                           int num_worst_tokens,
                           const Config& config,
                           std::optional<EventHandle>& previous_event,
                           bool async,
                           bool allocate_on_comm_stream) {
    bool cached_mode = cached_rank_prefix_matrix.has_value();

    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_channel_prefix_matrix.has_value());
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    EP_HOST_ASSERT(is_token_in_rank.scalar_type() == torch::kBool);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->scalar_type() == torch::kInt32);
    } else {
        EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
    }

    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
    EP_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) and is_token_in_rank.size(1) == num_ranks);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->dim() == 2 and cached_rank_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rank_prefix_matrix->size(0) == num_ranks and cached_rank_prefix_matrix->size(1) == num_ranks);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->dim() == 2 and cached_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_channel_prefix_matrix->size(0) == num_ranks and cached_channel_prefix_matrix->size(1) == num_channels);
    } else {
        EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
        EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;

    int num_topk = 0;
    topk_idx_t* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr = topk_idx->data_ptr<topk_idx_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    float* x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or x_scales->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::cuda::setCurrentCUDAStream(comm_stream);
    }

    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    int num_recv_tokens = -1;
    auto rank_prefix_matrix = torch::Tensor();
    auto channel_prefix_matrix = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    int num_memset_int = num_channels * num_ranks * 4;
    if (cached_mode) {
        num_recv_tokens = cached_num_recv_tokens;
        rank_prefix_matrix = cached_rank_prefix_matrix.value();
        channel_prefix_matrix = cached_channel_prefix_matrix.value();

        intranode::cached_notify_dispatch(
            rank_prefix_matrix.data_ptr<int>(), num_memset_int, buffer_ptrs_gpu, barrier_signal_ptrs_gpu, rank, num_ranks, comm_stream);
    } else {
        rank_prefix_matrix = torch::empty({num_ranks, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
        channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));

        *moe_recv_counter = -1;
        for (int i = 0; i < num_local_experts; ++i)
            moe_recv_expert_counter[i] = -1;
        EP_HOST_ASSERT(num_ranks * (num_ranks + num_local_experts) * sizeof(int) <= num_nvl_bytes);
        intranode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(),
                                   moe_recv_counter_mapped,
                                   num_ranks,
                                   num_tokens_per_expert->data_ptr<int>(),
                                   moe_recv_expert_counter_mapped,
                                   num_experts,
                                   num_tokens,
                                   is_token_in_rank.data_ptr<bool>(),
                                   channel_prefix_matrix.data_ptr<int>(),
                                   rank_prefix_matrix.data_ptr<int>(),
                                   num_memset_int,
                                   expert_alignment,
                                   buffer_ptrs_gpu,
                                   barrier_signal_ptrs_gpu,
                                   rank,
                                   comm_stream,
                                   num_channels);

        if (num_worst_tokens > 0) {
            num_recv_tokens = num_worst_tokens;
            EP_HOST_ASSERT(topk_idx.has_value());
            EP_HOST_ASSERT(topk_weights.has_value());
        } else {
            auto start_time = std::chrono::high_resolution_clock::now();
            while (true) {
                num_recv_tokens = static_cast<int>(*moe_recv_counter);
                bool ready = (num_recv_tokens >= 0);
                for (int i = 0; i < num_local_experts and ready; ++i)
                    ready &= moe_recv_expert_counter[i] >= 0;
                if (ready)
                    break;
                if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() >
                    NUM_CPU_TIMEOUT_SECS)
                    throw std::runtime_error("DeepEP error: CPU recv timeout");
            }
            num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
        }
    }

    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_src_idx = torch::empty({num_recv_tokens}, dtype(torch::kInt32).device(torch::kCUDA));
    auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(),
         recv_x_scales = std::optional<torch::Tensor>();
    auto recv_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
    auto send_head = torch::empty({num_tokens, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

    topk_idx_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
        recv_x_scales = x_scales->dim() == 1 ? torch::empty({num_recv_tokens}, x_scales->options())
                                             : torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
    }

    EP_HOST_ASSERT(
        num_ranks * num_ranks * sizeof(int) +
            num_channels * num_ranks * sizeof(int) +
            num_channels * num_ranks * sizeof(int) +
            num_channels * num_ranks * sizeof(int) * 2 +
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * recv_x.element_size() +
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(topk_idx_t) +
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float) +
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(float) * num_scales
        <= num_nvl_bytes);
    intranode::dispatch(recv_x.data_ptr(),
                        recv_x_scales_ptr,
                        recv_src_idx.data_ptr<int>(),
                        recv_topk_idx_ptr,
                        recv_topk_weights_ptr,
                        recv_channel_prefix_matrix.data_ptr<int>(),
                        send_head.data_ptr<int>(),
                        x.data_ptr(),
                        x_scales_ptr,
                        topk_idx_ptr,
                        topk_weights_ptr,
                        is_token_in_rank.data_ptr<bool>(),
                        channel_prefix_matrix.data_ptr<int>(),
                        num_tokens,
                        num_worst_tokens,
                        static_cast<int>(hidden * recv_x.element_size() / sizeof(int4)),
                        num_topk,
                        num_experts,
                        num_scales,
                        scale_token_stride,
                        scale_hidden_stride,
                        buffer_ptrs_gpu,
                        rank,
                        num_ranks,
                        comm_stream,
                        config.num_sms,
                        config.num_max_nvl_chunked_send_tokens,
                        config.num_max_nvl_chunked_recv_tokens);

    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {x,
                        is_token_in_rank,
                        rank_prefix_matrix,
                        channel_prefix_matrix,
                        recv_x,
                        recv_src_idx,
                        recv_channel_prefix_matrix,
                        send_head}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {x_scales,
                         topk_idx,
                         topk_weights,
                         num_tokens_per_rank,
                         num_tokens_per_expert,
                         cached_channel_prefix_matrix,
                         cached_rank_prefix_matrix,
                         recv_topk_idx,
                         recv_topk_weights,
                         recv_x_scales}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(compute_stream);

    return {recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            send_head,
            event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> Buffer::intranode_combine(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& topk_weights,
    const std::optional<torch::Tensor>& bias_0,
    const std::optional<torch::Tensor>& bias_1,
    const torch::Tensor& src_idx,
    const torch::Tensor& rank_prefix_matrix,
    const torch::Tensor& channel_prefix_matrix,
    const torch::Tensor& send_head,
    const Config& config,
    std::optional<EventHandle>& previous_event,
    bool async,
    bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_idx.dim() == 1 and src_idx.is_contiguous() and src_idx.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(send_head.dim() == 2 and send_head.is_contiguous() and send_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rank_prefix_matrix.dim() == 2 and rank_prefix_matrix.is_contiguous() and
                   rank_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(channel_prefix_matrix.dim() == 2 and channel_prefix_matrix.is_contiguous() and
                   channel_prefix_matrix.scalar_type() == torch::kInt32);

    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_recv_tokens = static_cast<int>(send_head.size(0));
    EP_HOST_ASSERT(src_idx.size(0) == num_tokens);
    EP_HOST_ASSERT(send_head.size(1) == num_ranks);
    EP_HOST_ASSERT(rank_prefix_matrix.size(0) == num_ranks and rank_prefix_matrix.size(1) == num_ranks);
    EP_HOST_ASSERT(channel_prefix_matrix.size(0) == num_ranks and channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);

    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::cuda::setCurrentCUDAStream(comm_stream);
    }

    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    int num_topk = 0;
    auto recv_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        num_topk = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->data_ptr<float>();
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }

    EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 <= num_nvl_bytes);
    intranode::cached_notify_combine(buffer_ptrs_gpu,
                                     send_head.data_ptr<int>(),
                                     num_channels,
                                     num_recv_tokens,
                                     num_channels * num_ranks * 2,
                                     barrier_signal_ptrs_gpu,
                                     rank,
                                     num_ranks,
                                     comm_stream);

    auto bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i)
        if (bias_opts[i].has_value()) {
            auto bias = bias_opts[i].value();
            EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
            EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
            EP_HOST_ASSERT(bias.size(0) == num_recv_tokens and bias.size(1) == hidden);
            bias_ptrs[i] = bias.data_ptr();
        }

    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 +
                       num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * x.element_size() +
                       num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +
                       num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float)
                   <= num_nvl_bytes);
    intranode::combine(at::cuda::ScalarTypeToCudaDataType(x.scalar_type()),
                       recv_x.data_ptr(),
                       recv_topk_weights_ptr,
                       x.data_ptr(),
                       topk_weights_ptr,
                       bias_ptrs[0],
                       bias_ptrs[1],
                       src_idx.data_ptr<int>(),
                       rank_prefix_matrix.data_ptr<int>(),
                       channel_prefix_matrix.data_ptr<int>(),
                       send_head.data_ptr<int>(),
                       num_tokens,
                       num_recv_tokens,
                       hidden,
                       num_topk,
                       buffer_ptrs_gpu,
                       rank,
                       num_ranks,
                       comm_stream,
                       config.num_sms,
                       config.num_max_nvl_chunked_send_tokens,
                       config.num_max_nvl_chunked_recv_tokens);

    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {x, src_idx, send_head, rank_prefix_matrix, channel_prefix_matrix, recv_x}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {topk_weights, recv_topk_weights, bias_0, bias_1}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(compute_stream);

    return {recv_x, recv_topk_weights, event};
}
#endif // !USE_NIXL

std::tuple<torch::Tensor,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::vector<int>,
           torch::Tensor,
           torch::Tensor,
           std::optional<torch::Tensor>,
           torch::Tensor,
           std::optional<torch::Tensor>,
           torch::Tensor,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::optional<EventHandle>>
Buffer::internode_dispatch(const torch::Tensor& x,
                           const std::optional<torch::Tensor>& x_scales,
                           const std::optional<torch::Tensor>& topk_idx,
                           const std::optional<torch::Tensor>& topk_weights,
                           const std::optional<torch::Tensor>& num_tokens_per_rank,
                           const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
                           const torch::Tensor& is_token_in_rank,
                           const std::optional<torch::Tensor>& num_tokens_per_expert,
                           int cached_num_recv_tokens,
                           int cached_num_rdma_recv_tokens,
                           const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix,
                           const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,
                           const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix,
                           const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum,
                           int expert_alignment,
#ifndef USE_NIXL
                           int num_worst_tokens,
#endif
                           const Config& config,
                           std::optional<EventHandle>& previous_event,
                           bool async,
                           bool allocate_on_comm_stream) {
#if defined(USE_NIXL) || !defined(DISABLE_NVSHMEM)
    pybind11::gil_scoped_release release;

    const int num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    EP_HOST_ASSERT(0 < get_num_rdma_ranks() and get_num_rdma_ranks() <= NUM_MAX_RDMA_PEERS);

    bool cached_mode = cached_rdma_channel_prefix_matrix.has_value();
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum.has_value());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum.has_value());
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->scalar_type() == torch::kInt32);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
    }

    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->dim() == 2 and cached_rdma_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->size(0) == num_rdma_ranks and
                       cached_rdma_channel_prefix_matrix->size(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->dim() == 1 and cached_recv_rdma_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->dim() == 2 and cached_gbl_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->size(0) == num_ranks and
                       cached_gbl_channel_prefix_matrix->size(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->dim() == 1 and cached_recv_gbl_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->size(0) == num_ranks);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->dim() == 1 and num_tokens_per_rdma_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)),
         hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)),
         num_local_experts = num_experts / num_ranks;

    int num_topk = 0;
    topk_idx_t* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr = topk_idx->data_ptr<topk_idx_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    float* x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or x_scales->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
#ifdef USE_NIXL
        num_scales = static_cast<int>(x_scales->size(1));
#else
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
#endif
        x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::cuda::setCurrentCUDAStream(comm_stream);
    }

    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    int num_recv_tokens = -1, num_rdma_recv_tokens = -1;
    auto rdma_channel_prefix_matrix = torch::Tensor();
    auto recv_rdma_rank_prefix_sum = torch::Tensor();
    auto gbl_channel_prefix_matrix = torch::Tensor();
    auto recv_gbl_rank_prefix_sum = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    if (cached_mode) {
        num_recv_tokens = cached_num_recv_tokens;
        num_rdma_recv_tokens = cached_num_rdma_recv_tokens;
        rdma_channel_prefix_matrix = cached_rdma_channel_prefix_matrix.value();
        recv_rdma_rank_prefix_sum = cached_recv_rdma_rank_prefix_sum.value();
        gbl_channel_prefix_matrix = cached_gbl_channel_prefix_matrix.value();
        recv_gbl_rank_prefix_sum = cached_recv_gbl_rank_prefix_sum.value();

        internode::cached_notify(hidden_int4, num_scales, num_topk, num_topk,
                                 num_ranks, num_channels, 0, nullptr,
                                 nullptr, nullptr, nullptr,
                                 rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
                                 buffer_ptrs_gpu, config.num_max_nvl_chunked_recv_tokens,
                                 barrier_signal_ptrs_gpu, rank, comm_stream,
                                 config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                                 num_nvl_bytes, true, low_latency_mode
#ifdef USE_NIXL
                                 , gpu_ctx
#endif
                                 );
    } else {
        rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        recv_rdma_rank_prefix_sum = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
        gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        recv_gbl_rank_prefix_sum = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

        *moe_recv_counter = -1, *moe_recv_rdma_counter = -1;
        for (int i = 0; i < num_local_experts; ++i)
            moe_recv_expert_counter[i] = -1;
        internode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(),
                                   moe_recv_counter_mapped,
                                   num_ranks,
                                   num_tokens_per_rdma_rank->data_ptr<int>(),
                                   moe_recv_rdma_counter_mapped,
                                   num_tokens_per_expert->data_ptr<int>(),
                                   moe_recv_expert_counter_mapped,
                                   num_experts,
                                   is_token_in_rank.data_ptr<bool>(),
                                   num_tokens,
#ifndef USE_NIXL
                                   num_worst_tokens,
#endif
                                   num_channels,
                                   hidden_int4,
                                   num_scales,
                                   num_topk,
                                   expert_alignment,
                                   rdma_channel_prefix_matrix.data_ptr<int>(),
                                   recv_rdma_rank_prefix_sum.data_ptr<int>(),
                                   gbl_channel_prefix_matrix.data_ptr<int>(),
                                   recv_gbl_rank_prefix_sum.data_ptr<int>(),
                                   rdma_buffer_ptr,
                                   config.num_max_rdma_chunked_recv_tokens,
                                   buffer_ptrs_gpu,
                                   config.num_max_nvl_chunked_recv_tokens,
                                   barrier_signal_ptrs_gpu,
                                   rank,
                                   comm_stream,
                                   config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                                   num_nvl_bytes,
                                   low_latency_mode
#ifdef USE_NIXL
                                   , gpu_ctx
#endif
                                   );

#ifndef USE_NIXL
        if (num_worst_tokens > 0) {
            num_recv_tokens = num_worst_tokens;
            num_rdma_recv_tokens = num_worst_tokens;
        } else {
#endif
        auto start_time = std::chrono::high_resolution_clock::now();
        while (true) {
            num_recv_tokens = static_cast<int>(*moe_recv_counter);
            num_rdma_recv_tokens = static_cast<int>(*moe_recv_rdma_counter);

            bool ready = (num_recv_tokens >= 0) and (num_rdma_recv_tokens >= 0);
            for (int i = 0; i < num_local_experts and ready; ++i)
                ready &= moe_recv_expert_counter[i] >= 0;

            if (ready)
                break;

            if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() >
                NUM_CPU_TIMEOUT_SECS) {
#ifndef USE_NIXL
                    printf("Global rank: %d, num_recv_tokens: %d, num_rdma_recv_tokens: %d\n", rank, num_recv_tokens, num_rdma_recv_tokens);
#endif
                for (int i = 0; i < num_local_experts; ++i)
                    printf("moe_recv_expert_counter[%d]: %d\n", i, moe_recv_expert_counter[i]);
                throw std::runtime_error("DeepEP error: timeout (dispatch CPU)");
            }
        }
        num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
#ifndef USE_NIXL
        }
#endif
    }

    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(),
         recv_x_scales = std::optional<torch::Tensor>();
    auto recv_src_meta = std::optional<torch::Tensor>();
    auto recv_rdma_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto recv_gbl_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto send_rdma_head = std::optional<torch::Tensor>();
    auto send_nvl_head = std::optional<torch::Tensor>();
    if (not cached_mode) {
        recv_src_meta = torch::empty({num_recv_tokens, internode::get_source_meta_bytes()}, dtype(torch::kByte).device(torch::kCUDA));
        recv_rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        recv_gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        send_rdma_head = torch::empty({num_tokens, num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
        send_nvl_head = torch::empty({num_rdma_recv_tokens, NUM_MAX_NVL_PEERS}, dtype(torch::kInt32).device(torch::kCUDA));
    }

    topk_idx_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
#ifdef USE_NIXL
        recv_x_scales = torch::empty({num_recv_tokens, num_scales}, x_scales->options());
#else
        recv_x_scales = x_scales->dim() == 1 ? torch::empty({num_recv_tokens}, x_scales->options())
                                             : torch::empty({num_recv_tokens, num_scales}, x_scales->options());
#endif
        recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
    }

    internode::dispatch(recv_x.data_ptr(),
                        recv_x_scales_ptr,
                        recv_topk_idx_ptr,
                        recv_topk_weights_ptr,
                        cached_mode ? nullptr : recv_src_meta->data_ptr(),
                        x.data_ptr(),
                        x_scales_ptr,
                        topk_idx_ptr,
                        topk_weights_ptr,
                        cached_mode ? nullptr : send_rdma_head->data_ptr<int>(),
                        cached_mode ? nullptr : send_nvl_head->data_ptr<int>(),
                        cached_mode ? nullptr : recv_rdma_channel_prefix_matrix->data_ptr<int>(),
                        cached_mode ? nullptr : recv_gbl_channel_prefix_matrix->data_ptr<int>(),
                        rdma_channel_prefix_matrix.data_ptr<int>(),
                        recv_rdma_rank_prefix_sum.data_ptr<int>(),
                        gbl_channel_prefix_matrix.data_ptr<int>(),
                        recv_gbl_rank_prefix_sum.data_ptr<int>(),
                        is_token_in_rank.data_ptr<bool>(),
                        num_tokens,
#ifndef USE_NIXL
                        num_worst_tokens,
#endif
                        hidden_int4,
                        num_scales,
                        num_topk,
                        num_experts,
                        scale_token_stride,
                        scale_hidden_stride,
                        rdma_buffer_ptr,
                        config.num_max_rdma_chunked_send_tokens,
                        config.num_max_rdma_chunked_recv_tokens,
                        buffer_ptrs_gpu,
                        config.num_max_nvl_chunked_send_tokens,
                        config.num_max_nvl_chunked_recv_tokens,
                        rank,
                        num_ranks,
                        cached_mode,
                        comm_stream,
                        num_channels,
                        low_latency_mode
#ifdef USE_NIXL
                        , gpu_ctx
#endif
                        );

    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {x,
                        is_token_in_rank,
                        recv_x,
                        rdma_channel_prefix_matrix,
                        recv_rdma_rank_prefix_sum,
                        gbl_channel_prefix_matrix,
                        recv_gbl_rank_prefix_sum}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {x_scales,
                         topk_idx,
                         topk_weights,
                         num_tokens_per_rank,
                         num_tokens_per_rdma_rank,
                         num_tokens_per_expert,
                         cached_rdma_channel_prefix_matrix,
                         cached_recv_rdma_rank_prefix_sum,
                         cached_gbl_channel_prefix_matrix,
                         cached_recv_gbl_rank_prefix_sum,
                         recv_topk_idx,
                         recv_topk_weights,
                         recv_x_scales,
                         recv_rdma_channel_prefix_matrix,
                         recv_gbl_channel_prefix_matrix,
                         send_rdma_head,
                         send_nvl_head,
                         recv_src_meta}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(compute_stream);

    return {recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rdma_channel_prefix_matrix,
            gbl_channel_prefix_matrix,
            recv_rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            recv_gbl_channel_prefix_matrix,
            recv_gbl_rank_prefix_sum,
            recv_src_meta,
            send_rdma_head,
            send_nvl_head,
            event};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> Buffer::internode_combine(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& topk_weights,
    const std::optional<torch::Tensor>& bias_0,
    const std::optional<torch::Tensor>& bias_1,
    const torch::Tensor& src_meta,
    const torch::Tensor& is_combined_token_in_rank,
    const torch::Tensor& rdma_channel_prefix_matrix,
    const torch::Tensor& rdma_rank_prefix_sum,
    const torch::Tensor& gbl_channel_prefix_matrix,
    const torch::Tensor& combined_rdma_head,
    const torch::Tensor& combined_nvl_head,
    const Config& config,
    std::optional<EventHandle>& previous_event,
    bool async,
    bool allocate_on_comm_stream) {
#if defined(USE_NIXL) || !defined(DISABLE_NVSHMEM)
    const int num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);

    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_meta.dim() == 2 and src_meta.is_contiguous() and src_meta.scalar_type() == torch::kByte);
    EP_HOST_ASSERT(is_combined_token_in_rank.dim() == 2 and is_combined_token_in_rank.is_contiguous() and
                   is_combined_token_in_rank.scalar_type() == torch::kBool);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.dim() == 2 and rdma_channel_prefix_matrix.is_contiguous() and
                   rdma_channel_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.dim() == 1 and rdma_rank_prefix_sum.is_contiguous() and
                   rdma_rank_prefix_sum.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.dim() == 2 and gbl_channel_prefix_matrix.is_contiguous() and
                   gbl_channel_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.is_contiguous() and
                   combined_rdma_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.is_contiguous() and combined_nvl_head.scalar_type() == torch::kInt32);

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)),
         hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    auto num_combined_tokens = static_cast<int>(is_combined_token_in_rank.size(0));
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(src_meta.size(1) == internode::get_source_meta_bytes());
    EP_HOST_ASSERT(is_combined_token_in_rank.size(1) == num_ranks);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.size(0) == num_rdma_ranks and rdma_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.size(0) == num_rdma_ranks);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.size(0) == num_ranks and gbl_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.size(0) == num_combined_tokens and
                   combined_rdma_head.size(1) == num_rdma_ranks);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.size(1) == NUM_MAX_NVL_PEERS);

    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::cuda::setCurrentCUDAStream(comm_stream);
    }

    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    int num_topk = 0;
    auto combined_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* combined_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        num_topk = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->data_ptr<float>();
        combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
        combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
    }

    EP_HOST_ASSERT(config.num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
    EP_HOST_ASSERT(config.num_max_nvl_chunked_send_tokens <= config.num_max_nvl_chunked_recv_tokens / num_rdma_ranks);

    internode::cached_notify(hidden_int4, 0, 0, num_topk,
                             num_ranks, num_channels,
                             num_combined_tokens, combined_rdma_head.data_ptr<int>(),
                             rdma_channel_prefix_matrix.data_ptr<int>(),
                             rdma_rank_prefix_sum.data_ptr<int>(),
                             combined_nvl_head.data_ptr<int>(),
                             rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
                             buffer_ptrs_gpu, config.num_max_nvl_chunked_recv_tokens,
                             barrier_signal_ptrs_gpu, rank, comm_stream,
                             config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                             num_nvl_bytes, false, low_latency_mode
#ifdef USE_NIXL
                             , gpu_ctx
#endif
                             );

    auto bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i)
        if (bias_opts[i].has_value()) {
            auto bias = bias_opts[i].value();
            EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
            EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
            EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);
            bias_ptrs[i] = bias.data_ptr();
        }

    auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    internode::combine(at::cuda::ScalarTypeToCudaDataType(x.scalar_type()),
                       combined_x.data_ptr(),
                       combined_topk_weights_ptr,
                       is_combined_token_in_rank.data_ptr<bool>(),
                       x.data_ptr(),
                       topk_weights_ptr,
                       bias_ptrs[0],
                       bias_ptrs[1],
                       combined_rdma_head.data_ptr<int>(),
                       combined_nvl_head.data_ptr<int>(),
                       src_meta.data_ptr(),
                       rdma_channel_prefix_matrix.data_ptr<int>(),
                       rdma_rank_prefix_sum.data_ptr<int>(),
                       gbl_channel_prefix_matrix.data_ptr<int>(),
                       num_tokens,
                       num_combined_tokens,
                       hidden,
                       num_topk,
                       rdma_buffer_ptr,
                       config.num_max_rdma_chunked_send_tokens,
                       config.num_max_rdma_chunked_recv_tokens,
                       buffer_ptrs_gpu,
                       config.num_max_nvl_chunked_send_tokens,
                       config.num_max_nvl_chunked_recv_tokens,
                       rank,
                       num_ranks,
                       comm_stream,
                       num_channels,
                       low_latency_mode
#ifdef USE_NIXL
                       , gpu_ctx
#endif
                       );

    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {x,
                        src_meta,
                        is_combined_token_in_rank,
                        rdma_channel_prefix_matrix,
                        rdma_rank_prefix_sum,
                        gbl_channel_prefix_matrix,
                        combined_x,
                        combined_rdma_head,
                        combined_nvl_head}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {topk_weights, combined_topk_weights, bias_0, bias_1}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(compute_stream);

    return {combined_x, combined_topk_weights, event};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

#ifdef USE_NIXL

void Buffer::clean_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) {
    EPLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    auto clean_meta_0 = layout.buffers[0].clean_meta();
    auto clean_meta_1 = layout.buffers[1].clean_meta();

    auto check_boundary = [=](void* ptr, size_t num_bytes) {
        auto offset = reinterpret_cast<int64_t>(ptr) - reinterpret_cast<int64_t>(rdma_buffer_ptr);
        EP_HOST_ASSERT(0 <= offset and offset + num_bytes <= num_rdma_bytes);
    };
    check_boundary(clean_meta_0.first, clean_meta_0.second * sizeof(int));
    check_boundary(clean_meta_1.first, clean_meta_1.second * sizeof(int));

    ep_kernels::clean_buffer(clean_meta_0.first,
                             clean_meta_0.second,
                             clean_meta_1.first,
                             clean_meta_1.second,
                             mask_buffer_ptr,
                             gpu_ctx,
                             at::cuda::getCurrentCUDAStream());
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
Buffer::dispatch(const torch::Tensor& x,
                 const torch::Tensor& topk_idx,
                 const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
                 const std::optional<torch::Tensor>& dispatch_wait_recv_cost_stats,
                 int num_max_dispatch_tokens_per_rank,
                 int num_experts,
                 bool use_fp8,
                 bool round_scale,
                 bool use_ue8m0,
                 bool async,
                 bool return_recv_hook) {
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(1) % sizeof(int4) == 0 and x.size(1) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(x.size(0) == topk_idx.size(0) and x.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(num_experts % num_ranks == 0);

    if (cumulative_local_expert_recv_stats.has_value()) {
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->dim() == 1 and cumulative_local_expert_recv_stats->is_contiguous());
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->size(0) == num_experts / num_ranks);
    }
    if (dispatch_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->dim() == 1 and dispatch_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_topk = static_cast<int>(topk_idx.size(1));
    int num_local_experts = num_experts / num_ranks;

    EPLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    auto buffer = layout.buffers[buffer_idx];
    auto next_buffer = layout.buffers[buffer_idx ^= 1];

    auto compute_stream = at::cuda::getCurrentCUDAStream();
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not(async and return_recv_hook));
    if (not return_recv_hook)
        stream_wait(launch_stream, compute_stream);

    auto packed_recv_x = torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                                      x.options().dtype(use_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16));
    auto packed_recv_src_info =
        torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto packed_recv_layout_range = torch::empty({num_local_experts, num_ranks}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto packed_recv_count = torch::empty({num_local_experts}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    auto packed_recv_x_scales = std::optional<torch::Tensor>();
    void* packed_recv_x_scales_ptr = nullptr;
    EP_HOST_ASSERT((num_ranks * num_max_dispatch_tokens_per_rank) % 4 == 0 and "TMA requires the number of tokens to be multiple of 4");

    if (use_fp8) {
        EP_HOST_ASSERT(hidden % 512 == 0);
        if (not use_ue8m0) {
            packed_recv_x_scales = torch::empty({num_local_experts, hidden / 128, num_ranks * num_max_dispatch_tokens_per_rank},
                                                torch::dtype(torch::kFloat32).device(torch::kCUDA));
        } else {
            EP_HOST_ASSERT(round_scale);
            packed_recv_x_scales = torch::empty({num_local_experts, hidden / 512, num_ranks * num_max_dispatch_tokens_per_rank},
                                                torch::dtype(torch::kInt).device(torch::kCUDA));
        }
        packed_recv_x_scales = torch::transpose(packed_recv_x_scales.value(), 1, 2);
        packed_recv_x_scales_ptr = packed_recv_x_scales->data_ptr();
    }

    auto next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=](int phases) {
        ep_kernels::dispatch(packed_recv_x.data_ptr(),
                             packed_recv_x_scales_ptr,
                             packed_recv_src_info.data_ptr<int>(),
                             packed_recv_layout_range.data_ptr<int64_t>(),
                             packed_recv_count.data_ptr<int>(),
                             mask_buffer_ptr,
                             cumulative_local_expert_recv_stats.has_value() ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr,
                             dispatch_wait_recv_cost_stats.has_value() ? dispatch_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                             buffer.dispatch_rdma_recv_data_buffer,
                             buffer.dispatch_rdma_recv_count_buffer,
                             buffer.dispatch_rdma_send_buffer,
                             x.data_ptr(),
                             topk_idx.data_ptr<topk_idx_t>(),
                             next_clean_meta.first,
                             next_clean_meta.second,
                             num_tokens,
                             hidden,
                             num_max_dispatch_tokens_per_rank,
                             num_topk,
                             num_experts,
                             rank,
                             num_ranks,
                             use_fp8,
                             round_scale,
                             use_ue8m0,
                             workspace,
                             num_device_sms,
                             launch_stream,
                             phases,
                             gpu_ctx);
    };
    launcher(return_recv_hook ? EP_SEND_PHASE : (EP_SEND_PHASE | EP_RECV_PHASE));

    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
        stream_wait(compute_stream, launch_stream);
    }

    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() { launcher(EP_RECV_PHASE); };

    return {packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, event, recv_hook};
}

std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
Buffer::combine(const torch::Tensor& x,
                const torch::Tensor& topk_idx,
                const torch::Tensor& topk_weights,
                const torch::Tensor& src_info,
                const torch::Tensor& layout_range,
                const std::optional<torch::Tensor>& combine_wait_recv_cost_stats,
                int num_max_dispatch_tokens_per_rank,
                int num_experts,
                bool use_logfmt,
                bool zero_copy,
                bool async,
                bool return_recv_hook,
                const std::optional<torch::Tensor>& out) {
    EP_HOST_ASSERT(x.dim() == 3 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(0) == num_experts / num_ranks);
    EP_HOST_ASSERT(x.size(1) == num_ranks * num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(x.size(2) % sizeof(int4) == 0 and x.size(2) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(topk_idx.size(0) == topk_weights.size(0) and topk_idx.size(1) == topk_weights.size(1));
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
    EP_HOST_ASSERT(topk_weights.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
    EP_HOST_ASSERT(src_info.dim() == 2 and src_info.is_contiguous());
    EP_HOST_ASSERT(src_info.scalar_type() == torch::kInt32 and x.size(0) == src_info.size(0));
    EP_HOST_ASSERT(layout_range.dim() == 2 and layout_range.is_contiguous());
    EP_HOST_ASSERT(layout_range.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(layout_range.size(0) == num_experts / num_ranks and layout_range.size(1) == num_ranks);

    if (combine_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->dim() == 1 and combine_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->size(0) == num_ranks);
    }

    auto hidden = static_cast<int>(x.size(2));
    auto num_topk = static_cast<int>(topk_weights.size(1));
    auto num_combined_tokens = static_cast<int>(topk_weights.size(0));

    EPLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    auto buffer = layout.buffers[buffer_idx];
    auto next_buffer = layout.buffers[buffer_idx ^= 1];

    auto compute_stream = at::cuda::getCurrentCUDAStream();
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not(async and return_recv_hook));
    if (not return_recv_hook)
        stream_wait(launch_stream, compute_stream);

    torch::Tensor combined_x;
    if (out.has_value()) {
        EP_HOST_ASSERT(out->dim() == 2 and out->is_contiguous());
        EP_HOST_ASSERT(out->size(0) == num_combined_tokens and out->size(1) == hidden);
        EP_HOST_ASSERT(out->scalar_type() == x.scalar_type());
        combined_x = out.value();
    } else {
        combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    }

    auto next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=](int phases) {
        ep_kernels::combine(combined_x.data_ptr(),
                            buffer.combine_rdma_recv_data_buffer,
                            buffer.combine_rdma_recv_flag_buffer,
                            buffer.combine_rdma_send_buffer,
                            x.data_ptr(),
                            topk_idx.data_ptr<topk_idx_t>(),
                            topk_weights.data_ptr<float>(),
                            src_info.data_ptr<int>(),
                            layout_range.data_ptr<int64_t>(),
                            mask_buffer_ptr,
                            combine_wait_recv_cost_stats.has_value() ? combine_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                            next_clean_meta.first,
                            next_clean_meta.second,
                            num_combined_tokens,
                            hidden,
                            num_max_dispatch_tokens_per_rank,
                            num_topk,
                            num_experts,
                            rank,
                            num_ranks,
                            use_logfmt,
                            workspace,
                            num_device_sms,
                            launch_stream,
                            phases,
                            zero_copy,
                            gpu_ctx);
    };
    launcher(return_recv_hook ? EP_SEND_PHASE : (EP_SEND_PHASE | EP_RECV_PHASE));

    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
        stream_wait(compute_stream, launch_stream);
    }

    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() { launcher(EP_RECV_PHASE); };

    return {combined_x, event, recv_hook};
}

void Buffer::barrier() {
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    ep_kernels::barrier(gpu_ctx, mask_buffer_ptr, compute_stream);
}

torch::Tensor Buffer::get_next_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const {
    EPLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);

    auto buffer = layout.buffers[buffer_idx];
    auto dtype = torch::kBFloat16;
    auto num_msg_elems = static_cast<int>(buffer.num_bytes_per_combine_msg / elementSize(torch::kBFloat16));

    EP_HOST_ASSERT(buffer.num_bytes_per_combine_msg % elementSize(torch::kBFloat16) == 0);
    return torch::from_blob(buffer.combine_rdma_send_buffer_data_start,
                            {num_experts / num_ranks, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                            {num_ranks * num_max_dispatch_tokens_per_rank * num_msg_elems, num_msg_elems, 1},
                            torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
}

void Buffer::update_mask_buffer(int rank_to_mask, bool mask) {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(rank_to_mask >= 0 and rank_to_mask < max_num_ranks);
    ep_kernels::update_mask_buffer(mask_buffer_ptr, rank_to_mask, mask, at::cuda::getCurrentCUDAStream());
}

void Buffer::query_mask_buffer(const torch::Tensor& mask_status) {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(mask_status.numel() == max_num_ranks && mask_status.scalar_type() == torch::kInt32);

    ep_kernels::query_mask_buffer(mask_buffer_ptr, max_num_ranks,
                                  reinterpret_cast<int*>(mask_status.data_ptr()),
                                  at::cuda::getCurrentCUDAStream());
}

void Buffer::clean_mask_buffer() {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    ep_kernels::clean_mask_buffer(mask_buffer_ptr, max_num_ranks, at::cuda::getCurrentCUDAStream());
}

std::string Buffer::get_local_metadata() const {
    EP_HOST_ASSERT(nixl_agent_info != nullptr && nixl_agent_info->agent != nullptr);
    nixl_blob_t metadata_blob;
    nixl_status_t status = nixl_agent_info->agent->getLocalMD(metadata_blob);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to get local metadata, status: " + std::to_string(status));
    }
    return metadata_blob;
}

void Buffer::_nixl_agent_init() {
    std::string agent_name = std::to_string(rank);
    nixlAgentConfig cfg(true, false, 0,
                        nixl_thread_sync_t::NIXL_THREAD_SYNC_RW, 1, 0, 100000, false, NIXL_ETCD_WATCH_TIMEOUT);
    auto agent = std::make_shared<nixlAgent>(agent_name, cfg);

    nixl_mem_list_t mems;
    nixl_b_params_t init_params;

    nixl_status_t status = agent->getPluginParams("UCX", mems, init_params);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to get UCX plugin parameters for agent " + agent_name +
                                 ", status: " + std::to_string(status));
    }

    const char* num_channels_env = std::getenv("NIXL_EP_NUM_CHANNELS");
    init_params["ucx_num_device_channels"] = num_channels_env ? num_channels_env : "4";
    init_params["ucx_error_handling_mode"] = "none";
    init_params["num_workers"] = std::to_string(1);

    nixlBackendH* ucx_backend = nullptr;
    status = agent->createBackend("UCX", init_params, ucx_backend);
    if (status != NIXL_SUCCESS || !ucx_backend) {
        throw std::runtime_error("Failed to create UCX backend for agent " + agent_name +
                                 ", status: " + std::to_string(status));
    }

    nixl_agent_info = std::make_unique<NixlAgentInfo>(agent, ucx_backend, max_num_ranks);
    nixl_agent_info->extra_params.backends.push_back(ucx_backend);
    nixl_agent_info->agent_name = agent_name;

    nixl_reg_dlist_t rdma_ptr_dlist(VRAM_SEG);
    rdma_ptr_dlist.addDesc(nixlBlobDesc((uintptr_t)(rdma_buffer_ptr), num_rdma_bytes, get_local_device_id(), ""));
    EP_HOST_ASSERT(agent->registerMem(rdma_ptr_dlist) == NIXL_SUCCESS);

    nixl_reg_dlist_t sync_dlist(VRAM_SEG);
    sync_dlist.addDesc(nixlBlobDesc((uintptr_t)(sync_buffer_ptr), max_num_ranks * sizeof(int), get_local_device_id(), ""));
    EP_HOST_ASSERT(agent->registerMem(sync_dlist) == NIXL_SUCCESS);

    nixl_reg_dlist_t barrier_cnt_dlist(VRAM_SEG);
    barrier_cnt_dlist.addDesc(nixlBlobDesc((uintptr_t)(sync_count_ptr), max_num_ranks * sizeof(int), get_local_device_id(), ""));
    EP_HOST_ASSERT(agent->registerMem(barrier_cnt_dlist) == NIXL_SUCCESS);

    if (!low_latency_mode && local_barrier_counter) {
        nixl_reg_dlist_t internode_barrier_dlist(VRAM_SEG);
        internode_barrier_dlist.addDesc(nixlBlobDesc((uintptr_t)(local_barrier_counter), sizeof(uint64_t), get_local_device_id(), ""));
        EP_HOST_ASSERT(agent->registerMem(internode_barrier_dlist) == NIXL_SUCCESS);
    }

    if (getenv("NIXL_ETCD_ENDPOINTS")) {
        status = nixl_agent_info->agent->sendLocalMD();
        if (status != NIXL_SUCCESS) {
            throw std::runtime_error("Failed to send local metadata for agent " +
                                     nixl_agent_info->agent_name + ", status: " + std::to_string(status));
        }
    }
}

void Buffer::_nixl_agents_connect(const std::vector<int>& ranks, const std::vector<nixl_blob_t>& remote_mds) {
    EP_HOST_ASSERT(!ranks.empty());
    EP_HOST_ASSERT(remote_mds.empty() || remote_mds.size() == ranks.size());

    remote_ranks.insert(remote_ranks.end(), ranks.begin(), ranks.end());
    for (int remote_rank : ranks) {
        nixl_agent_info->remote_agent_names[remote_rank] = std::to_string(remote_rank);
    }

    for (size_t i = 0; i < ranks.size(); i++) {
        int remote_rank = ranks[i];

        nixl_status_t status = remote_mds.empty()
            ? nixl_agent_info->agent->fetchRemoteMD(nixl_agent_info->remote_agent_names[remote_rank])
            : nixl_agent_info->agent->loadRemoteMD(remote_mds[i], nixl_agent_info->remote_agent_names[remote_rank]);

        if (status != NIXL_SUCCESS) {
            throw std::runtime_error("Failed to get metadata for remote agent " +
                                     std::to_string(remote_rank) + ", status: " + std::to_string(status));
        }
    }

    std::vector<bool> peer_ready(max_num_ranks, false);
    int peers_remaining = static_cast<int>(ranks.size());

    while (peers_remaining > 0) {
        for (int remote_rank : ranks) {
            if (peer_ready[remote_rank]) continue;

            nixl_local_dlist_t empty_descs(VRAM_SEG);
            if (nixl_agent_info->agent->checkRemoteMD(std::to_string(remote_rank), empty_descs) == NIXL_SUCCESS) {
                peer_ready[remote_rank] = true;
                peers_remaining--;
            }
        }
        if (peers_remaining > 0) {
            sleep_ms(10);
        }
    }
}

void Buffer::_nixl_agents_disconnect(const std::vector<int>& ranks) {
    for (int remote_rank : ranks) {
        EP_HOST_ASSERT(remote_rank != rank);
        EP_HOST_ASSERT(remote_rank < num_ranks);
        nixl_xfer_dlist_t empty_descs(VRAM_SEG);
        if (nixl_agent_info->agent->checkRemoteMD(nixl_agent_info->remote_agent_names[remote_rank], empty_descs) == NIXL_SUCCESS) {
            nixl_status_t status = nixl_agent_info->agent->invalidateRemoteMD(nixl_agent_info->remote_agent_names[remote_rank]);
            if (status != NIXL_SUCCESS && status != NIXL_ERR_NOT_FOUND) {
                printf("WARNING: rank %d Failed to invalidate remote rank %d metadata for agent %s, status: %d\n",
                       rank, remote_rank, std::to_string(remote_rank).c_str(), status);
                fflush(stdout);
            }
        }
    }
}

void Buffer::_nixl_agents_peer_info_gather(std::vector<int>& ranks) {
    for (int remote_rank : ranks) {
        std::string my_peer_info_str(reinterpret_cast<const char*>(&my_peer_info), sizeof(NixlPeerInfo));
        nixl_agent_info->agent->genNotif(std::to_string(remote_rank), my_peer_info_str);
    }

    for (int remote_rank : ranks) {
        do {
            nixl_notifs_t notif_map;
            nixl_agent_info->agent->getNotifs(notif_map);
            for (auto& notif : notif_map) {
                std::string my_peer_info_str = notif.second[0];
                NixlPeerInfo remote_peer_info;
                memcpy(&remote_peer_info, my_peer_info_str.c_str(), sizeof(NixlPeerInfo));
                nixl_peer_info[remote_peer_info.rank] = remote_peer_info;
                nixl_agent_info->wire_up_done[remote_peer_info.rank] = true;
            }
        } while (!nixl_agent_info->wire_up_done[remote_rank]);
    }
}

void Buffer::_nixl_agents_peer_info_cleanup(const std::vector<int>& ranks) {
    for (int remote_rank : ranks) {
        nixl_agent_info->wire_up_done[remote_rank] = false;
        nixl_peer_info[remote_rank] = NixlPeerInfo{};
    }
}

void Buffer::_nixl_ep_init(void) {
    gpu_ctx = {
        .sync_buffer_ptr = sync_buffer_ptr,
        .sync_count_ptr = sync_count_ptr,
        .rdma_buffer_ptr = rdma_buffer_ptr,
        .max_num_ranks = max_num_ranks,
        .num_rdma_ranks = num_rdma_ranks,
        .rank = rank,
        .last_barrier_counter = last_barrier_counter,
        .local_barrier_counter_ptr = local_barrier_counter,
    };
}

void Buffer::_nixl_ep_memory_views_create(void) {
    nixl_remote_dlist_t remote_descs(VRAM_SEG);
    nixl_remote_dlist_t barrier_descs(VRAM_SEG);
    nixl_local_dlist_t local_descs(VRAM_SEG);

    local_descs.addDesc(nixlBlobDesc((uintptr_t)(rdma_buffer_ptr), num_rdma_bytes, get_local_device_id(), ""));
    local_descs.addDesc(nixlBlobDesc((uintptr_t)(sync_count_ptr), max_num_ranks * sizeof(int), get_local_device_id(), ""));

    std::unordered_set<int> remote_set(remote_ranks.begin(), remote_ranks.end());
    for (int r = 0; r < max_num_ranks; r++) {
        std::string remote_agent_name = remote_set.count(r) ? nixl_agent_info->remote_agent_names[r] : nixl_null_agent;
        remote_descs.addDesc(nixlRemoteDesc((uintptr_t)nixl_peer_info[r].rdma_buffer_ptr, num_rdma_bytes,
                                            nixl_peer_info[r].device_id, remote_agent_name));
        barrier_descs.addDesc(nixlRemoteDesc((uintptr_t)nixl_peer_info[r].sync_buffer_ptr, max_num_ranks * sizeof(int),
                                             nixl_peer_info[r].device_id, remote_agent_name));
    }

    EP_HOST_ASSERT(nixl_agent_info->agent->prepMemView(local_descs, gpu_ctx.local_mvh, &nixl_agent_info->extra_params) == NIXL_SUCCESS);
    EP_HOST_ASSERT(nixl_agent_info->agent->prepMemView(remote_descs, gpu_ctx.remote_mvh, &nixl_agent_info->extra_params) == NIXL_SUCCESS);
    EP_HOST_ASSERT(nixl_agent_info->agent->prepMemView(barrier_descs, gpu_ctx.barrier_mvh, &nixl_agent_info->extra_params) == NIXL_SUCCESS);

    if (!low_latency_mode && num_ranks > NUM_MAX_NVL_PEERS) {
        nixl_remote_dlist_t internode_barrier_descs(VRAM_SEG);
        for (int r = 0; r < max_num_ranks; r++) {
            std::string remote_agent_name = remote_set.count(r) ? nixl_agent_info->remote_agent_names[r] : nixl_null_agent;
            internode_barrier_descs.addDesc(nixlRemoteDesc((uintptr_t)nixl_peer_info[r].barrier_ptr, sizeof(uint64_t),
                                                           nixl_peer_info[r].device_id, remote_agent_name));
        }
        EP_HOST_ASSERT(nixl_agent_info->agent->prepMemView(internode_barrier_descs, gpu_ctx.internode_barrier_mvh,
                                                            &nixl_agent_info->extra_params) == NIXL_SUCCESS);
    }
}

void Buffer::_nixl_ep_memory_views_destroy(void) {
    if (gpu_ctx.local_mvh) nixl_agent_info->agent->releaseMemView(gpu_ctx.local_mvh);
    if (gpu_ctx.remote_mvh) nixl_agent_info->agent->releaseMemView(gpu_ctx.remote_mvh);
    if (gpu_ctx.barrier_mvh) nixl_agent_info->agent->releaseMemView(gpu_ctx.barrier_mvh);
    if (gpu_ctx.internode_barrier_mvh) nixl_agent_info->agent->releaseMemView(gpu_ctx.internode_barrier_mvh);
    gpu_ctx.local_mvh = nullptr;
    gpu_ctx.remote_mvh = nullptr;
    gpu_ctx.barrier_mvh = nullptr;
    gpu_ctx.internode_barrier_mvh = nullptr;
}

void Buffer::_nixl_ep_destroy(void) {
    _nixl_ep_memory_views_destroy();
}

void Buffer::_ipc_handles_sync(const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles) {
    if (num_nvl_bytes > 0) {
        EP_HOST_ASSERT(all_gathered_handles.size() == max_num_ranks);
        for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++i) {
            EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
            auto handle_str = std::string(all_gathered_handles[offset + i].value());
            EP_HOST_ASSERT(handle_str.size() == CUDA_IPC_HANDLE_SIZE);
            if (offset + i != rank) {
                std::memcpy(ipc_handles[i].reserved, handle_str.c_str(), CUDA_IPC_HANDLE_SIZE);
                CUDA_CHECK(cudaIpcOpenMemHandle(&buffer_ptrs[i], ipc_handles[i], cudaIpcMemLazyEnablePeerAccess));
                barrier_signal_ptrs[i] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
            } else {
                EP_HOST_ASSERT(std::memcmp(ipc_handles[i].reserved, handle_str.c_str(), CUDA_IPC_HANDLE_SIZE) == 0);
            }
        }

        CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs, sizeof(int*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

static std::optional<std::vector<nixl_blob_t>> convert_mds(const std::optional<std::vector<pybind11::bytes>>& remote_mds) {
    if (!remote_mds.has_value()) {
        return std::nullopt;
    }
    std::vector<nixl_blob_t> md_blobs;
    md_blobs.reserve(remote_mds->size());
    for (const auto& md_bytes : *remote_mds) {
        md_blobs.push_back(nixl_blob_t(md_bytes));
    }
    return md_blobs;
}

#else // !USE_NIXL

void Buffer::clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(low_latency_mode);

    auto layout = LowLatencyLayout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    auto clean_meta_0 = layout.buffers[0].clean_meta();
    auto clean_meta_1 = layout.buffers[1].clean_meta();

    auto check_boundary = [=](void* ptr, size_t num_bytes) {
        auto offset = reinterpret_cast<int64_t>(ptr) - reinterpret_cast<int64_t>(rdma_buffer_ptr);
        EP_HOST_ASSERT(0 <= offset and offset + num_bytes <= num_rdma_bytes);
    };
    check_boundary(clean_meta_0.first, clean_meta_0.second * sizeof(int));
    check_boundary(clean_meta_1.first, clean_meta_1.second * sizeof(int));

    internode_ll::clean_low_latency_buffer(clean_meta_0.first,
                                           clean_meta_0.second,
                                           clean_meta_1.first,
                                           clean_meta_1.second,
                                           rank,
                                           num_ranks,
                                           mask_buffer_ptr,
                                           sync_buffer_ptr,
                                           at::cuda::getCurrentCUDAStream());
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
#endif
}

std::tuple<torch::Tensor,
           std::optional<torch::Tensor>,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           std::optional<EventHandle>,
           std::optional<std::function<void()>>>
Buffer::low_latency_dispatch(const torch::Tensor& x,
                             const torch::Tensor& topk_idx,
                             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
                             const std::optional<torch::Tensor>& dispatch_wait_recv_cost_stats,
                             int num_max_dispatch_tokens_per_rank,
                             int num_experts,
                             bool use_fp8,
                             bool round_scale,
                             bool use_ue8m0,
                             bool async,
                             bool return_recv_hook) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(low_latency_mode);

    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(1) % sizeof(int4) == 0 and x.size(1) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(x.size(0) == topk_idx.size(0) and x.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(num_experts % num_ranks == 0);

    if (cumulative_local_expert_recv_stats.has_value()) {
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->dim() == 1 and cumulative_local_expert_recv_stats->is_contiguous());
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->size(0) == num_experts / num_ranks);
    }
    if (dispatch_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->dim() == 1 and dispatch_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_topk = static_cast<int>(topk_idx.size(1));
    auto num_local_experts = num_experts / num_ranks;

    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    auto compute_stream = at::cuda::getCurrentCUDAStream();
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not(async and return_recv_hook));
    if (not return_recv_hook)
        stream_wait(launch_stream, compute_stream);

    auto packed_recv_x = torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                                      x.options().dtype(use_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16));
    auto packed_recv_src_info =
        torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto packed_recv_layout_range = torch::empty({num_local_experts, num_ranks}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto packed_recv_count = torch::empty({num_local_experts}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    auto packed_recv_x_scales = std::optional<torch::Tensor>();
    void* packed_recv_x_scales_ptr = nullptr;
    EP_HOST_ASSERT((num_ranks * num_max_dispatch_tokens_per_rank) % 4 == 0 and "TMA requires the number of tokens to be multiple of 4");

    if (use_fp8) {
        EP_HOST_ASSERT(hidden % 512 == 0);
        if (not use_ue8m0) {
            packed_recv_x_scales = torch::empty({num_local_experts, hidden / 128, num_ranks * num_max_dispatch_tokens_per_rank},
                                                torch::dtype(torch::kFloat32).device(torch::kCUDA));
        } else {
            EP_HOST_ASSERT(round_scale);
            packed_recv_x_scales = torch::empty({num_local_experts, hidden / 512, num_ranks * num_max_dispatch_tokens_per_rank},
                                                torch::dtype(torch::kInt).device(torch::kCUDA));
        }
        packed_recv_x_scales = torch::transpose(packed_recv_x_scales.value(), 1, 2);
        packed_recv_x_scales_ptr = packed_recv_x_scales->data_ptr();
    }

    auto next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=](int phases) {
        internode_ll::dispatch(
            packed_recv_x.data_ptr(),
            packed_recv_x_scales_ptr,
            packed_recv_src_info.data_ptr<int>(),
            packed_recv_layout_range.data_ptr<int64_t>(),
            packed_recv_count.data_ptr<int>(),
            mask_buffer_ptr,
            cumulative_local_expert_recv_stats.has_value() ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr,
            dispatch_wait_recv_cost_stats.has_value() ? dispatch_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
            buffer.dispatch_rdma_recv_data_buffer,
            buffer.dispatch_rdma_recv_count_buffer,
            buffer.dispatch_rdma_send_buffer,
            x.data_ptr(),
            topk_idx.data_ptr<topk_idx_t>(),
            next_clean_meta.first,
            next_clean_meta.second,
            num_tokens,
            hidden,
            num_max_dispatch_tokens_per_rank,
            num_topk,
            num_experts,
            rank,
            num_ranks,
            use_fp8,
            round_scale,
            use_ue8m0,
            workspace,
            num_device_sms,
            launch_stream,
            phases);
    };
    launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
        stream_wait(compute_stream, launch_stream);
    }

    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

    return {packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, event, recv_hook};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> Buffer::low_latency_combine(
    const torch::Tensor& x,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights,
    const torch::Tensor& src_info,
    const torch::Tensor& layout_range,
    const std::optional<torch::Tensor>& combine_wait_recv_cost_stats,
    int num_max_dispatch_tokens_per_rank,
    int num_experts,
    bool use_logfmt,
    bool zero_copy,
    bool async,
    bool return_recv_hook,
    const std::optional<torch::Tensor>& out) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(low_latency_mode);

    EP_HOST_ASSERT(x.dim() == 3 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(0) == num_experts / num_ranks);
    EP_HOST_ASSERT(x.size(1) == num_ranks * num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(x.size(2) % sizeof(int4) == 0 and x.size(2) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(topk_idx.size(0) == topk_weights.size(0) and topk_idx.size(1) == topk_weights.size(1));
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
    EP_HOST_ASSERT(topk_weights.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
    EP_HOST_ASSERT(src_info.dim() == 2 and src_info.is_contiguous());
    EP_HOST_ASSERT(src_info.scalar_type() == torch::kInt32 and x.size(0) == src_info.size(0));
    EP_HOST_ASSERT(layout_range.dim() == 2 and layout_range.is_contiguous());
    EP_HOST_ASSERT(layout_range.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(layout_range.size(0) == num_experts / num_ranks and layout_range.size(1) == num_ranks);

    if (combine_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->dim() == 1 and combine_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->size(0) == num_ranks);
    }

    auto hidden = static_cast<int>(x.size(2));
    auto num_topk = static_cast<int>(topk_weights.size(1));
    auto num_combined_tokens = static_cast<int>(topk_weights.size(0));

    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    auto compute_stream = at::cuda::getCurrentCUDAStream();
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not(async and return_recv_hook));
    if (not return_recv_hook)
        stream_wait(launch_stream, compute_stream);

    torch::Tensor combined_x;
    if (out.has_value()) {
        EP_HOST_ASSERT(out->dim() == 2 and out->is_contiguous());
        EP_HOST_ASSERT(out->size(0) == num_combined_tokens and out->size(1) == hidden);
        EP_HOST_ASSERT(out->scalar_type() == x.scalar_type());
        combined_x = out.value();
    } else {
        combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    }

    auto next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=](int phases) {
        internode_ll::combine(combined_x.data_ptr(),
                              buffer.combine_rdma_recv_data_buffer,
                              buffer.combine_rdma_recv_flag_buffer,
                              buffer.combine_rdma_send_buffer,
                              x.data_ptr(),
                              topk_idx.data_ptr<topk_idx_t>(),
                              topk_weights.data_ptr<float>(),
                              src_info.data_ptr<int>(),
                              layout_range.data_ptr<int64_t>(),
                              mask_buffer_ptr,
                              combine_wait_recv_cost_stats.has_value() ? combine_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                              next_clean_meta.first,
                              next_clean_meta.second,
                              num_combined_tokens,
                              hidden,
                              num_max_dispatch_tokens_per_rank,
                              num_topk,
                              num_experts,
                              rank,
                              num_ranks,
                              use_logfmt,
                              workspace,
                              num_device_sms,
                              launch_stream,
                              phases,
                              zero_copy);
    };
    launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
        stream_wait(compute_stream, launch_stream);
    }

    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

    return {combined_x, event, recv_hook};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

torch::Tensor Buffer::get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const {
#ifndef DISABLE_NVSHMEM
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);

    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto dtype = torch::kBFloat16;
    auto num_msg_elems = static_cast<int>(buffer.num_bytes_per_combine_msg / elementSize(torch::kBFloat16));

    EP_HOST_ASSERT(buffer.num_bytes_per_combine_msg % elementSize(torch::kBFloat16) == 0);
    return torch::from_blob(buffer.combine_rdma_send_buffer_data_start,
                            {num_experts / num_ranks, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                            {num_ranks * num_max_dispatch_tokens_per_rank * num_msg_elems, num_msg_elems, 1},
                            torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

void Buffer::low_latency_update_mask_buffer(int rank_to_mask, bool mask) {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(rank_to_mask >= 0 and rank_to_mask < num_ranks);
    internode_ll::update_mask_buffer(mask_buffer_ptr, rank_to_mask, mask, at::cuda::getCurrentCUDAStream());
}

void Buffer::low_latency_query_mask_buffer(const torch::Tensor& mask_status) {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(mask_status.numel() == num_ranks && mask_status.scalar_type() == torch::kInt32);

    internode_ll::query_mask_buffer(
        mask_buffer_ptr, num_ranks, reinterpret_cast<int*>(mask_status.data_ptr()), at::cuda::getCurrentCUDAStream());
}

void Buffer::low_latency_clean_mask_buffer() {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    internode_ll::clean_mask_buffer(mask_buffer_ptr, num_ranks, at::cuda::getCurrentCUDAStream());
}

#endif // USE_NIXL

bool is_sm90_compiled() {
#ifndef DISABLE_SM90_FEATURES
    return true;
#else
    return false;
#endif
}

}  // namespace deep_ep

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepEP: an efficient expert-parallel communication library";

#ifdef USE_NIXL
    m.def("get_rdma_size_hint", &deep_ep::get_rdma_size_hint);
#endif

    pybind11::class_<deep_ep::Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>(),
             py::arg("num_sms") = 20,
             py::arg("num_max_nvl_chunked_send_tokens") = 6,
             py::arg("num_max_nvl_chunked_recv_tokens") = 256,
             py::arg("num_max_rdma_chunked_send_tokens") = 6,
             py::arg("num_max_rdma_chunked_recv_tokens") = 256)
        .def("get_nvl_buffer_size_hint", &deep_ep::Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint", &deep_ep::Config::get_rdma_buffer_size_hint);

#ifndef USE_NIXL
    m.def("get_low_latency_rdma_size_hint", &deep_ep::get_low_latency_rdma_size_hint);
#endif

    pybind11::class_<deep_ep::EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

#ifdef USE_NIXL
    pybind11::class_<deep_ep::Buffer>(m, "Buffer")
        .def(pybind11::init<int, bool, bool>())
        .def("update_memory_buffers", &deep_ep::Buffer::update_memory_buffers)
        .def("barrier", &deep_ep::Buffer::barrier)
        .def("connect_ranks", [](deep_ep::Buffer& buffer,
                                  const std::vector<int>& remote_ranks,
                                  const std::optional<std::vector<pybind11::bytes>>& remote_mds,
                                  const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles) {
            buffer.connect_ranks(remote_ranks, deep_ep::convert_mds(remote_mds), all_gathered_handles);
        }, py::arg("remote_ranks"), py::arg("remote_mds") = std::nullopt,
           py::arg("ipc_handles") = std::vector<std::optional<pybind11::bytearray>>{})
        .def("disconnect_ranks", &deep_ep::Buffer::disconnect_ranks)
        .def("is_available", &deep_ep::Buffer::is_available)
        .def("get_num_rdma_ranks", &deep_ep::Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &deep_ep::Buffer::get_rdma_rank)
        .def("get_root_rdma_rank", &deep_ep::Buffer::get_root_rdma_rank)
        .def("get_local_device_id", &deep_ep::Buffer::get_local_device_id)
        .def("get_local_ipc_handle", &deep_ep::Buffer::get_local_ipc_handle)
        .def("get_local_buffer_tensor", &deep_ep::Buffer::get_local_buffer_tensor)
        .def("get_comm_stream", &deep_ep::Buffer::get_comm_stream)
        .def("destroy", &deep_ep::Buffer::destroy)
        .def("get_dispatch_layout", &deep_ep::Buffer::get_dispatch_layout)
        .def("dispatch", &deep_ep::Buffer::dispatch)
        .def("combine", &deep_ep::Buffer::combine)
        .def("internode_dispatch", &deep_ep::Buffer::internode_dispatch)
        .def("internode_combine", &deep_ep::Buffer::internode_combine)
        .def("update_mask_buffer", &deep_ep::Buffer::update_mask_buffer)
        .def("query_mask_buffer", &deep_ep::Buffer::query_mask_buffer)
        .def("clean_mask_buffer", &deep_ep::Buffer::clean_mask_buffer)
        .def("clean_buffer", &deep_ep::Buffer::clean_buffer)
        .def("get_next_combine_buffer", &deep_ep::Buffer::get_next_combine_buffer)
        .def("get_local_metadata", [](const deep_ep::Buffer& buffer) -> pybind11::bytes {
            return pybind11::bytes(buffer.get_local_metadata());
        });
#else
    pybind11::class_<deep_ep::Buffer>(m, "Buffer")
        .def(pybind11::init<int, int, int64_t, int64_t, bool, bool, bool, bool>())
        .def("is_available", &deep_ep::Buffer::is_available)
        .def("get_num_rdma_ranks", &deep_ep::Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &deep_ep::Buffer::get_rdma_rank)
        .def("get_root_rdma_rank", &deep_ep::Buffer::get_root_rdma_rank)
        .def("get_local_device_id", &deep_ep::Buffer::get_local_device_id)
        .def("get_local_ipc_handle", &deep_ep::Buffer::get_local_ipc_handle)
        .def("get_local_nvshmem_unique_id", &deep_ep::Buffer::get_local_nvshmem_unique_id)
        .def("get_local_buffer_tensor", &deep_ep::Buffer::get_local_buffer_tensor)
        .def("get_comm_stream", &deep_ep::Buffer::get_comm_stream)
        .def("sync", &deep_ep::Buffer::sync)
        .def("destroy", &deep_ep::Buffer::destroy)
        .def("get_dispatch_layout", &deep_ep::Buffer::get_dispatch_layout)
        .def("intranode_dispatch", &deep_ep::Buffer::intranode_dispatch)
        .def("intranode_combine", &deep_ep::Buffer::intranode_combine)
        .def("internode_dispatch", &deep_ep::Buffer::internode_dispatch)
        .def("internode_combine", &deep_ep::Buffer::internode_combine)
        .def("clean_low_latency_buffer", &deep_ep::Buffer::clean_low_latency_buffer)
        .def("low_latency_dispatch", &deep_ep::Buffer::low_latency_dispatch)
        .def("low_latency_combine", &deep_ep::Buffer::low_latency_combine)
        .def("low_latency_update_mask_buffer", &deep_ep::Buffer::low_latency_update_mask_buffer)
        .def("low_latency_query_mask_buffer", &deep_ep::Buffer::low_latency_query_mask_buffer)
        .def("low_latency_clean_mask_buffer", &deep_ep::Buffer::low_latency_clean_mask_buffer)
        .def("get_next_low_latency_combine_buffer", &deep_ep::Buffer::get_next_low_latency_combine_buffer);
#endif

    m.def("is_sm90_compiled", deep_ep::is_sm90_compiled);
    m.attr("topk_idx_t") =
        py::reinterpret_borrow<py::object>((PyObject*)torch::getTHPDtype(c10::CppTypeToScalarType<deep_ep::topk_idx_t>::value));
}
