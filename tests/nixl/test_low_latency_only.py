"""
NIXL low-latency only test — extracted from test_internode.py --test-ll-compatibility.
Skips the HT internode test and goes directly to LL buffer creation + test_main().

Usage:
  Node 0: WORLD_SIZE=2 RANK=0 MASTER_ADDR=<ip> MASTER_PORT=18491 \
           python3 test_low_latency_only.py --num-processes 8

  Node 1: WORLD_SIZE=2 RANK=1 MASTER_ADDR=<ip> MASTER_PORT=18491 \
           python3 test_low_latency_only.py --num-processes 8 --tcp-server <ip>
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'elastic'))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import store_group

import torch
import torch.distributed as dist

import deep_ep
from utils import init_dist
import test_low_latency

TCP_STORE_PORT = 9999


def _init_dist_single_gpu(local_rank: int, num_local_ranks: int):
    """Like init_dist but works with CUDA_VISIBLE_DEVICES=<single GPU>."""
    import inspect
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8361"))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))

    sig = inspect.signature(dist.init_process_group)
    params = {
        "backend": "nccl",
        "init_method": f"tcp://{ip}:{port}",
        "world_size": num_nodes * num_local_ranks,
        "rank": node_rank * num_local_ranks + local_rank,
    }
    if "device_id" in sig.parameters:
        params["device_id"] = torch.device("cuda:0")
    dist.init_process_group(**params)
    return dist.get_rank(), dist.get_world_size(), dist.group.WORLD


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank % 8)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(0)

    rank, num_ranks, group = _init_dist_single_gpu(local_rank, num_local_ranks)

    ll_num_tokens = args.num_tokens
    ll_hidden = args.hidden
    ll_num_experts = args.num_experts
    ll_num_topk = args.num_topk

    tcp_server = args.tcp_server if args.tcp_server else "127.0.0.1"
    tcp_store = store_group.create_client_store(master_addr=tcp_server, port=TCP_STORE_PORT)

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(ll_num_tokens, ll_hidden, num_ranks, ll_num_experts)
    if local_rank == 0:
        print(f'[config] rank={rank}, num_ranks={num_ranks}, local_rank={local_rank}, '
              f'num_tokens={ll_num_tokens}, hidden={ll_hidden}, num_experts={ll_num_experts}, '
              f'num_topk={ll_num_topk}, experts_per_rank={ll_num_experts // num_ranks}', flush=True)
        print(f'[buffer] rdma={num_rdma_bytes / 1e6} MB, nvl=0.0 MB', flush=True)

    buffer = deep_ep.Buffer.nixl_buffer(
        rank=rank,
        low_latency_mode=True,
        explicitly_destroy=True,
        tcp_store_group=tcp_store,
        disable_ll_nvlink=args.disable_nvlink,
    )
    buffer.update_memory_buffers(
        num_ranks=num_ranks,
        num_experts_per_rank=ll_num_experts // num_ranks,
        num_nvl_bytes=0,
        num_rdma_bytes=num_rdma_bytes,
    )
    buffer.connect_ranks([i for i in range(num_ranks) if i != rank])

    dist.barrier()
    test_low_latency.test_main(
        ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk,
        rank, num_ranks, group, buffer, seed=1,
    )

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


def run_server():
    _store = store_group.create_master_store(port=TCP_STORE_PORT)
    while True:
        time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NIXL low-latency only test')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-tokens', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=5120)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--num-topk', type=int, default=9)
    parser.add_argument('--tcp-server', type=str, default=None)
    parser.add_argument('--disable-nvlink', action='store_true')
    args = parser.parse_args()

    if not args.tcp_server:
        print("Starting TCPStore server locally", flush=True)
        server_proc = torch.multiprocessing.Process(target=run_server, daemon=True)
        server_proc.start()
        time.sleep(0.5)

    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
