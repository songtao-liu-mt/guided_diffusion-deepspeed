"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3

def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    #os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

    #comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    port = _find_free_port()
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")

def set_env():
    #os.environ["NCCL_DEBUG"] = "INFO"

#    os.environ["NCCL_SOCKET_IFNAME"] = "ib,eth,en,em,bond"
#    os.environ["GLOO_SOCKET_IFNAME"] = "eth"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_SHM_DISABLE"] = "0"

    #os.environ["NCCL_IB_CUDA_SUPPORT"] = "1"
    #os.environ["NCCL_NET_GDR_READ"] = "1"

    os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
    os.environ["NCCL_IB_HCA"] = "mlx5_1:1"
    #os.environ["NCCL_IB_GID_INDEX"] = "3"
    #os.environ["NCCL_IB_TC"] = "106"
    os.environ["NCCL_NET"] = "IB"



def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    if dist.get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = bytes()

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
