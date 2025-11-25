import torch
import argparse
import os
import time
import logging
import torch.distributed as dist
import gc
import socket

LOGGER = logging.getLogger(__name__)


def measure(init, m, args):
    times = []
    for _ in range(args.num_iters + args.num_warmup):
        xs = init()
        torch.cuda.synchronize()
        start = time.time_ns()
        y = m(*xs)
        torch.cuda.synchronize()
        end = time.time_ns()
        times.append(end - start)
        if isinstance(y, torch.Tensor):
            y.untyped_storage().resize_(0)
        del y
        for x in xs:
            if isinstance(x, list):
                for q in x:
                    q.untyped_storage().resize_(0)
            else:
                x.untyped_storage().resize_(0)
        del xs
    times = times[args.num_warmup :]
    total_ns = sum(times)
    total_s = total_ns * 1e-9
    return total_s


def cluster_mean_std(item, group=None):
    cluster_avg = torch.tensor(item, dtype=torch.float32)
    dist.all_reduce(cluster_avg, op=dist.ReduceOp.SUM, group=group)
    cluster_avg /= dist.get_world_size(group=group)

    stddev = (item - cluster_avg).square()
    dist.all_reduce(stddev, op=dist.ReduceOp.SUM, group=group)
    stddev /= dist.get_world_size(group=group) - 1
    stddev = stddev.sqrt()
    return cluster_avg.item(), stddev.item()


def _ranks_str(ranks):
    ranks = sorted(ranks)
    if len(ranks) > 2 and all(r + 1 == ranks[i + 1] for i, r in enumerate(ranks[:-1])):
        start = ranks[0]
        end = ranks[-1]
        ranks_display = f"{start}-{end}"
    else:
        ranks_display = ",".join(map(str, ranks))
    return ranks_display


MODELS = {
    "Llama-8B": {
        "mlp": (4096, 14336),
        "block_numel": 218112000,
    },
    "Llama-70B": {
        "mlp": (8192, 28672),
        "block_numel": 855654400,
    },
    "Llama-405B": {
        "mlp": (16384, 53248),
        "block_numel": 3187703808,
    },
}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", default=16384),
    parser.add_argument(
        "-m", "--model", default="Llama-70B", choices=list(MODELS.keys())
    )
    parser.add_argument("-i", "--num-iters", default=20, type=int)
    parser.add_argument("-w", "--num-warmup", default=5, type=int)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    logging.basicConfig(
        format=f"[%(asctime)s] [rank={rank:03}] %(message)s",
        level=logging.INFO,
    )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)

    device_name = torch.cuda.get_device_name(device).split(" ")[1]
    host = socket.gethostname()
    if rank == 0:
        LOGGER.info("=== Node Assignments ===")
    dist.barrier()

    LOGGER.info(f"Joined from {host}")

    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)

    dist.barrier()

    model = MODELS[args.model]

    b = args.batch
    m, n = model["mlp"]
    if rank == 0:
        LOGGER.info(f"=== {args.model} MLP Performance ({b}x{m} @ {m}x{n}) ===")
    dist.barrier()

    gc.disable()

    """
    TFLOPS
    """
    layer = torch.nn.Linear(m, n, bias=False)
    flops_per_token = 2 * m * n
    seconds = measure(lambda: (torch.randn(b, m),), layer.forward, args)
    tokens_per_s = args.num_iters * b / seconds
    flops_per_s = flops_per_token * tokens_per_s
    tflops = flops_per_s * 1e-12
    LOGGER.info(f"{tokens_per_s:.0f} tokens/s/gpu ({tflops:>8.1f} tflops)")
    cluster_tflops, tflops_stddev = cluster_mean_std(tflops)
    dist.barrier()
    if rank == 0:
        LOGGER.info(f"Mean TFLOPS is {cluster_tflops:.1f} ± {tflops_stddev:.3f}")

    layer.to("meta")
    del layer
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    """
    All Reduce
    """
    n = model["block_numel"]
    gb = n * torch.bfloat16.itemsize * 1e-9
    intra = 2
    while intra <= world_size:
        inter = world_size // intra
        assert inter * intra == world_size

        if rank == 0:
            LOGGER.info(f"=== ~{gb:.1f}gb All Reduce across {intra}x{device_name} ===")
        dist.barrier()

        mesh = dist.device_mesh.init_device_mesh(
            "cuda", (inter, intra), mesh_dim_names=("inter", "intra")
        )
        group = mesh.get_group(mesh_dim="intra")

        my_rank = torch.tensor([rank], dtype=torch.int64)
        group_ranks = [torch.empty(1, dtype=torch.int64) for _ in range(intra)]
        group.allgather(group_ranks, my_rank)

        seconds = measure(
            lambda: (torch.randn(n),),
            group.allreduce,
            args,
        )
        gb_per_s = args.num_iters * gb / seconds
        group_gbps, gbps_stddev = cluster_mean_std(gb_per_s, group=group)
        if group.rank() == 0:
            ranks = _ranks_str(sorted([g.item() for g in group_ranks]))
            LOGGER.info(f"Ranks {ranks:>10}: {group_gbps:>4.0f} gb/s ± {gbps_stddev:.2f}")

        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        intra *= 2

    """
    All Gather
    """
    n = model["block_numel"]
    gb = n * torch.bfloat16.itemsize * 1e-9
    intra = 2
    while intra <= world_size:
        inter = world_size // intra
        assert inter * intra == world_size

        if rank == 0:
            LOGGER.info(f"=== ~{gb:.1f}gb All-Gather across {intra}x{device_name}  ===")
        dist.barrier()

        mesh = dist.device_mesh.init_device_mesh(
            "cuda", (inter, intra), mesh_dim_names=("inter", "intra")
        )
        group = mesh.get_group(mesh_dim="intra")

        my_rank = torch.tensor([rank], dtype=torch.int64)
        group_ranks = [torch.empty(1, dtype=torch.int64) for _ in range(intra)]
        group.allgather(group_ranks, my_rank)

        seconds = measure(
            lambda: (
                [torch.empty(n) for _ in range(intra)],
                torch.randn(n),
            ),
            group.allgather,
            args,
        )
        gb_per_s = args.num_iters * gb / seconds
        group_gbps, gbps_stddev = cluster_mean_std(gb_per_s, group=group)
        if group.rank() == 0:
            ranks = _ranks_str(sorted([g.item() for g in group_ranks]))
            LOGGER.info(f"Ranks {ranks:>10}: {group_gbps:.2f} gb/s ± {gbps_stddev:.4f}")

        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        intra *= 2

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
