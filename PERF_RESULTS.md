# DeepEP NIXL Backend Integration

## Performance evaluation

### Test environment

- **Cluster**: NVIDIA EOS, 2 nodes x 8 H100 GPUs (16 ranks), ConnectX-7 InfiniBand
- **Configuration**: 4096 tokens (HT) / 16 tokens (LL), hidden=7168, 256 experts, 24 SMs
- **NIXL settings**: `NIXL_EP_NUM_CHANNELS=10`, `UCX_RC_GDA_NUM_CHANNELS=10`

### How to build and test

**Build NVSHMEM backend:**

```bash
cd /workspace/deepep
export NVSHMEM_DIR=/workspace/nvshmem_src/build
pip install -e .
```

**Build NIXL backend:**

```bash
cd /workspace/deepep
export NIXL_LIB_PATH=/workspace/nixl/build/src
export NIXL_INCLUDE_PATHS=/workspace/nixl/src/api/gpu/ucx:/workspace/nixl/src/api/cpp:/workspace/ucx/rfs/include:/workspace/doca/build/install/include
pip install -e .
```

**Run NVSHMEM internode test (2 nodes):**

```bash
# Node 0
cd /workspace/deepep
WORLD_SIZE=2 RANK=0 MASTER_ADDR=$(hostname -i) MASTER_PORT=8491 \
  python3 tests/test_internode.py --num-processes 8

# Node 1
WORLD_SIZE=2 RANK=1 MASTER_ADDR=<node0_ip> MASTER_PORT=8491 \
  python3 tests/test_internode.py --num-processes 8
```

**Run NIXL internode test (2 nodes):**

```bash
# Node 0
cd /workspace/deepep/tests/nixl
WORLD_SIZE=2 RANK=0 MASTER_ADDR=$(hostname -i) MASTER_PORT=8691 \
  NIXL_EP_NUM_CHANNELS=10 UCX_RC_GDA_NUM_CHANNELS=10 \
  PYTHONPATH=/workspace/deepep/tests/nixl:/workspace/deepep/tests \
  python3 test_internode.py --num-processes 8

# Node 1
cd /workspace/deepep/tests/nixl
WORLD_SIZE=2 RANK=1 MASTER_ADDR=<node0_ip> MASTER_PORT=8691 \
  NIXL_EP_NUM_CHANNELS=10 UCX_RC_GDA_NUM_CHANNELS=10 \
  PYTHONPATH=/workspace/deepep/tests/nixl:/workspace/deepep/tests \
  python3 test_internode.py --num-processes 8 --tcp-server <node0_ip>
```

Add `--test-ll-compatibility` to both commands to also run the low-latency tests.

### High-Throughput results


| Mode          | Impl    | Transmit (us) | Notify (us) | BW RDMA (GB/s) | BW NVL (GB/s) |
| ------------- | ------- | ------------- | ----------- | -------------- | ------------- |
| Dispatch FP8  | NVSHMEM | 756           | 53          | 79.89          | 260.75        |
|               | NIXL    | 765           | 67          | 78.89          | 257.50        |
| Dispatch BF16 | NVSHMEM | 1377          | 83          | 85.02          | 277.50        |
|               | NIXL    | 1362          | 94          | 85.95          | 280.55        |
| Combine BF16  | NVSHMEM | 1825          | 76          | 64.15          | 209.38        |
|               | NIXL    | 1827          | 74          | 64.08          | 209.15        |


NIXL and NVSHMEM are within 1-3% for all high-throughput operations. NIXL slightly wins on BF16 dispatch (85.95 vs 85.02 GB/s RDMA). Combine is identical.

### Low-Latency results


| Metric                    | NVSHMEM | NIXL  | Delta |
| ------------------------- | ------- | ----- | ----- |
| Dispatch latency (us)     | 20.77   | 36.31 | 1.75x |
| Dispatch bandwidth (GB/s) | 34.17   | 19.55 |       |
| Combine latency (us)      | 47.09   | 48.04 | 1.02x |
| Combine bandwidth (GB/s)  | 29.14   | 28.57 |       |




