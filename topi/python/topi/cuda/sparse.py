import tvm

from .. import generic 
from .injective import schedule_injective

@generic.schedule_sparse_dense.register(["gpu", "cuda"])
def _schedule_sparse_dense(outs):
    print("_schedule_sparse_dense")
    s = schedule_injective(outs)
    return s
