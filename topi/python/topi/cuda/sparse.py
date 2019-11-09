import tvm

from .. import generic 
from .injective import schedule_injective

@generic.schedule_sparse_dense.register(["gpu", "cuda"])
def _schedule_sparse_dense(outs):
    target = tvm.target.current_target()
    if target.target_name == "cuda" and "cusparse" in target.libs:
        return generic.schedule_extern(outs)

    s = schedule_injective(outs)

    return s
