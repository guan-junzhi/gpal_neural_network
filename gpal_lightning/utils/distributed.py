import logging

import numpy as np
from torch import distributed


def all_gather_by_chunks(data_list: list, n_chunks: int, desc: str):
    """Gather a list of data across all ranks. This function uses NCCL all_gather
    as the backend, therefore, it will transfer data_list to GPU memory first
    before sending across GPUs.

    Example:
        Assuming world_size is 2, rank 0 has data_list = [1, 2, 3], rank 1 has
        data_list = [8, 9]. This function will return [1, 2, 3, 8, 9].

    Args:
        data_list: the data list in the current rank
        n_chunk: sometime the data_list is too large to fit in GPU memory, this
                 argument allows user to break it into n smaller chunks to gather
        desc: some text to be printed, mainly to avoid triggering hung detections

    Return:
        A list equivalent to sum([data_list_rank_0, data_list_rank_1, ...], []).
        The order within each rank's data_list is preserverd. The order over rank
        is also preserved.
    """
    assert n_chunks >= 1
    data_chunks = [list(p) for p in np.array_split(data_list, n_chunks)]
    world_size = distributed.get_world_size()
    pool = [[] for _ in range(world_size)]
    for idx, chunk in enumerate(data_chunks):
        if distributed.get_rank() == 0:
            logging.info("gathering %s, step %d of %d", desc, idx, n_chunks - 1)
        chunks = ["" for _ in range(world_size)]
        distributed.all_gather_object(chunks, chunk)
        for per_rank, chk in zip(pool, chunks):
            per_rank += chk
    return sum(pool, [])
