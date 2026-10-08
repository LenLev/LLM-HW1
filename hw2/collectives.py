import torch
import torch.distributed as dist


def ring_reduce_scatter(tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    chunks = [chunk.clone() for chunk in tensor.chunk(world_size)]

    send_idx = (rank - 1) % world_size

    for _ in range(world_size - 1):
        recv_idx = (send_idx - 1) % world_size

        send_req = dist.isend(
            chunks[send_idx],
            dst=(rank + 1) % world_size,
        )

        recv = torch.empty_like(chunks[recv_idx])
        dist.recv(
            recv,
            src=(rank - 1) % world_size,
        )

        send_req.wait()

        chunks[recv_idx] += recv
        send_idx = recv_idx

    return chunks[rank]


def ring_all_gather(shard):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    chunks = [torch.empty_like(shard) for _ in range(world_size)]
    chunks[rank].copy_(shard)

    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size

        send_req = dist.isend(
            chunks[send_idx],
            dst=(rank + 1) % world_size,
        )

        dist.recv(
            chunks[recv_idx],
            src=(rank - 1) % world_size,
        )

        send_req.wait()

    return torch.cat(chunks)

def ring_all_reduce(tensor):
    shard = ring_reduce_scatter(tensor)
    return ring_all_gather(shard)
