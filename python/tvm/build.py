"""The build pipeline in python.

Eventually some of these pipelines will be moved to C++.
But the first pipeline will be kept in python for ease of change and evolving.
"""

from . import api
from . import tensor
from . import schedule
from . import expr
from . import ir_pass
from . import collections
from . import codegen

def lower(sch,
          args,
          name="default_function",
          binds=None,
          max_auto_unroll_step=8):
    """Lowering step before build into target.

    Parameters
    ----------
    sch : tvm.Schedule
        The schedule to be builded

    args : list of Buffer or Tensor or Var
        The argument lists to the function.

    name : str, optional
        The name of result function.

    binds : dict, optional
        Dictionary that maps the binding of symbolic buffer to Tensor.
        By default, a new buffer is created for each tensor in the argument.

    max_auto_unroll_step: int, optional
        Maximum step to perform automatic unrolling

    Returns
    -------
    f : LoweredFunc
       The result function.
    """
    binds = {} if binds is None else binds.copy()
    arg_list = []
    for x in args:
        if isinstance(x, tensor.Tensor):
            buf = api.decl_buffer(x.shape, dtype=x.dtype, name=x.op.name)
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
    return fapi


def build(sch,
          args=None,
          target="llvm",
          target_host="stackvm",
          name="default_function",
          binds=None,
          max_auto_unroll_step=8,
          detect_global_barrier=True):
    """Build a function with arguments as signiture.

    Parameters
    ----------
    sch : tvm.Schedule, or LoweredFunc
        The schedule to be builded

    args : list of Buffer or Tensor or Var, optional
        The argument lists to the function.

    target : str, optional
        The target of the compilation.

    target_host : str, optional
        Host compilation target, if target is device.

    name : str, optional
        The name of result function.

    binds : dict, optional
        Dictionary that maps the binding of symbolic buffer to Tensor.
        By default, a new buffer is created for each tensor in the argument.

    max_auto_unroll_step: int, optional
        Maximum step to perform automatic unrolling

    detect_global_barrier: boolean, optional
        Whether detect and inser global barrier

    Returns
    -------
    f : Function, or pair of functions
       The result function.
    """
    if isinstance(sch, schedule.Schedule):
        if args is None:
            raise ValueError("args must be given for build from schedule")
        fapi = lower(sch, args,
                     name=name,
                     binds=binds,
                     max_auto_unroll_step=max_auto_unroll_step)
    elif isinstance(sch, collections.LoweredFunc):
        if args:
            raise ValueError("args must be done when build from LoweredFunc")
        fapi = sch
    else:
        raise ValueError("sch have to be Schedule or LoweredFunc")
    # device related lowering
    if detect_global_barrier:
        fapi = ir_pass.StorageSync(fapi, "global")
    fapi = ir_pass.StorageSync(fapi, "shared")
    warp_size = 32 if target == "cuda" else 1
    fapi = ir_pass.LowerThreadAllreduce(fapi, warp_size)
    fsplits = [s for s in ir_pass.SplitHostDevice(fapi)]
    if len(fsplits) > 1:
        mhost = codegen.build_module(fsplits[0], target_host)
        if target:
            mdev = codegen.build_module(fsplits[1:], target)
            mhost.import_module(mdev)
        return mhost
    else:
        return codegen.build_module(fsplits[0], target)
