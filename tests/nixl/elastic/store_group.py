from datetime import timedelta

import torch.distributed as dist


def create_master_store(
    port: int = 9999,
    timeout_sec: float = 300.0,
) -> dist.TCPStore:
    return dist.TCPStore(
        host_name="0.0.0.0",
        port=port,
        is_master=True,
        wait_for_workers=False,
        timeout=timedelta(seconds=timeout_sec),
    )


def create_client_store(
    master_addr: str = "127.0.0.1",
    port: int = 9999,
    timeout_sec: float = 300.0,
) -> dist.TCPStore:
    return dist.TCPStore(
        host_name=master_addr,
        port=port,
        is_master=False,
        wait_for_workers=False,
        timeout=timedelta(seconds=timeout_sec),
    )
