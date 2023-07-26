# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging
from functools import wraps

import torch.distributed as dist

from detectron2.utils import comm as d2_comm

LOG = logging.getLogger(__name__)

_NESTED_BROADCAST_FROM_MASTER = False


def is_distributed():
    return d2_comm.get_world_size() > 1


def broadcast_from_master(fn):
    """If distributed, only the master executes the function and broadcast the results to other workers.

    Usage:
    @broadcast_from_master
    def foo(a, b): ...
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):  # pylint: disable=unused-argument
        global _NESTED_BROADCAST_FROM_MASTER

        if not is_distributed():
            return fn(*args, **kwargs)

        if _NESTED_BROADCAST_FROM_MASTER:
            assert d2_comm.is_main_process()
            LOG.warning(f"_NESTED_BROADCAST_FROM_MASTER = True, {fn.__name__}")
            return fn(*args, **kwargs)

        if d2_comm.is_main_process():
            _NESTED_BROADCAST_FROM_MASTER = True
            ret = [fn(*args, **kwargs), ]
            _NESTED_BROADCAST_FROM_MASTER = False
        else:
            ret = [None, ]
        if dist.is_initialized():
            dist.broadcast_object_list(ret)
        ret = ret[0]

        assert ret is not None
        return ret

    return wrapper


def master_only(fn):
    """If distributed, only the master executes the function.

    Usage:
    @master_only
    def foo(a, b): ...
    """
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if d2_comm.is_main_process():
            ret = fn(*args, **kwargs)
        d2_comm.synchronize()
        if d2_comm.is_main_process():
            return ret

    return wrapped_fn


def gather_dict(dikt):
    """Gather python dictionaries from all workers to the rank=0 worker.

    Assumption: the keys of `dikt` are disjoint across all workers.

    If rank = 0, then returned aggregated dict.
    If rank > 0, then return `None`.
    """
    dict_lst = d2_comm.gather(dikt, dst=0)
    if d2_comm.is_main_process():
        gathered_dict = {}
        for dic in dict_lst:
            for k in dic.keys():
                assert k not in gathered_dict, f"Dictionary key overlaps: {k}"
            gathered_dict.update(dic)
        return gathered_dict
    else:
        return None


def reduce_sum(tensor):
    """
    Adapted from AdelaiDet:
        https://github.com/aim-uofa/AdelaiDet/blob/master/adet/utils/comm.py
    """
    if not is_distributed():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor
