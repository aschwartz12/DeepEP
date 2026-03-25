import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nixl', 'elastic'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.distributed as dist

import deep_ep
from utils import init_dist
from test_low_latency import test_main

TCP_STORE_PORT = 9999
RANK_SERVER_PORT = 10000


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    import store_group
    import rank_server as rs

    server_addr = args.tcp_server if args.tcp_server else "127.0.0.1"
    rank_client = rs.RankClient(server_addr, RANK_SERVER_PORT)
    _, global_rank, _ = rank_client.get_rank()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank % 8)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(0)

    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    num_ranks = num_nodes * num_local_ranks

    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{ip}:{port}',
        world_size=num_ranks,
        rank=global_rank,
    )
    group = dist.new_group(list(range(num_ranks)))

    tcp_store = store_group.create_client_store(master_addr=server_addr, port=TCP_STORE_PORT)

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
    if local_rank == 0:
        print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)

    buffer = deep_ep.Buffer.nixl_buffer(
        rank=global_rank,
        low_latency_mode=True,
        explicitly_destroy=True,
        tcp_store_group=tcp_store,
        disable_ll_nvlink=args.disable_nvlink,
    )
    buffer.update_memory_buffers(
        num_ranks=num_ranks,
        num_experts_per_rank=num_experts // num_ranks,
        num_nvl_bytes=0,
        num_rdma_bytes=num_rdma_bytes,
    )

    peer_ranks = [i for i in range(num_ranks) if i != global_rank]
    print(f'global_rank={global_rank} -> connecting to {peer_ranks}', flush=True)
    buffer.connect_ranks(peer_ranks)

    dist.barrier()
    test_main(num_tokens, hidden, num_experts, num_topk,
              global_rank, num_ranks, group, buffer, seed=1)

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()
    rank_client.release_rank()


def run_server():
    import store_group
    import rank_server as rs
    _store = store_group.create_master_store(port=TCP_STORE_PORT)  # noqa: F841
    rs.start_server(port=RANK_SERVER_PORT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NIXL low-latency EP benchmark')
    parser.add_argument('--num-processes', type=int, default=1)
    parser.add_argument('--num-tokens', type=int, default=128)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=32)
    parser.add_argument('--disable-nvlink', action='store_true')
    parser.add_argument('--tcp-server', type=str, default=None)
    args = parser.parse_args()

    if not args.tcp_server:
        print("Starting TCPStore and rank server locally", flush=True)
        server_proc = torch.multiprocessing.Process(target=run_server, daemon=True)
        server_proc.start()
        time.sleep(0.5)

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
