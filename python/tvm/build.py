"""The build pipeline in python.

Eventually some of these pipelines will be moved to C++.
But the first pipeline will be kept in python for ease of change and evolving.
"""
# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments

from . import api
from . import tensor
from . import schedule
from . import expr
from . import ir_pass
from . import codegen

def build(sch,
          args,
          target,
          target_host="stackvm",
          name="default_function",
          binds=None,
          max_auto_unroll_step=8):
    """Build a function with arguments as signiture.

    Parameters
    ----------
    sch : tvm.Schedule
        The schedule to be builded

    args : list of Buffer or Tensor or Var
        The argument lists to the function.

    target : str
        The target of the compilation.

    target_host :
        Host compilation target, if target is device.

    name : str
        The name of result function.

    binds : dict, optional
        Dictionary that maps the binding of symbolic buffer to Tensor.
        By default, a new buffer is created for each tensor in the argument.

    max_auto_unroll_step: int
        Maximum step to perform automatic unrolling

    Returns
    -------
    f : Function, or pair of functions
       The result function.
       If the function requires host space allocation,
       a pair of functions will be returned.
    """
    binds = {} if binds is None else binds.copy()
    arg_list = []
    for x in args:
        if isinstance(x, tensor.Tensor):
            buf = api.Buffer(x.shape, dtype=x.dtype, name=x.op.name)
            assert x not in binds
            binds[x] = buf
            arg_list.append(buf)
        elif isinstance(x, schedule.Buffer):
            arg_list.append(x)
        elif isinstance(x, expr.Var):
            arg_list.append(x)
        else:
            raise ValueError("args must be Tensor, Buffer or Var")
    # normalize schedule first
    sch.normalize()
    bounds = schedule.InferBound(sch)
    stmt = schedule.ScheduleOps(sch, bounds)
    stmt = ir_pass.StorageFlatten(stmt, binds)
    stmt = ir_pass.CanonicalSimplify(stmt)
    stmt = ir_pass.VectorizeLoop(stmt)
    stmt = ir_pass.InjectVirtualThread(stmt)
    stmt = ir_pass.LiftAllocate(stmt)
    stmt = ir_pass.UnrollLoop(stmt, max_auto_unroll_step)
    stmt = ir_pass.Simplify(stmt)
    fapi = ir_pass.MakeAPI(stmt, name, arg_list, 0)
    fsplits = ir_pass.SplitHostDevice(fapi)
    fsplits = [x for x in fsplits]
    for i in range(1, len(fsplits)):
        fsplits[i] = ir_pass.StorageSync(fsplits[i], "shared")

    if len(fsplits) > 1:
        mhost = codegen.build(fsplits[0], target_host)
        if target:
            mdev = codegen.build(fsplits[1:], target)
            mhost.import_module(mdev)
        return mhost
    else:
        return codegen.build(fsplits[0], target)
