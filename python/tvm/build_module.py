"""The build utils in python.

This module provides the functions to transform schedule to
LoweredFunc and compiled Module.
"""
from __future__ import absolute_import as _abs
import warnings

from . import api
from . import tensor
from . import schedule
from . import expr
from . import ir_pass
from . import container
from . import module
from . import codegen
from . import ndarray
from . import target as _target

class BuildConfig(object):
    """Configuration scope to set a build config option.

    Parameters
    ----------
    kwargs
        Keyword arguments of configurations to set.
    """
    current = None
    defaults = {
        "auto_unroll_max_step": 0,
        "auto_unroll_max_depth": 8,
        "auto_unroll_max_extent": 0,
        "unroll_explicit": True,
        "detect_global_barrier": False,
        "offset_factor": 0,
        "data_alignment": -1,
        "restricted_func": True,
        "double_buffer_split_loop": 1,
        "add_lower_pass": None
    }
    def __init__(self, **kwargs):
        self._old_scope = None
        for k, _ in kwargs.items():
            if k not in BuildConfig.defaults:
                raise ValueError(
                    "invalid argument %s, candidates are %s" % (k, BuildConfig.defaults.keys()))
        self._attr = kwargs

    def __getattr__(self, name):
        if name not in self._attr:
            return BuildConfig.defaults[name]
        return self._attr[name]

    def __enter__(self):
        # pylint: disable=protected-access
        self._old_scope = BuildConfig.current
        attr = BuildConfig.current._attr.copy()
        attr.update(self._attr)
        self._attr = attr
        BuildConfig.current = self
        return self

    def __exit__(self, ptype, value, trace):
        assert self._old_scope
        BuildConfig.current = self._old_scope


BuildConfig.current = BuildConfig()

def build_config(**kwargs):
    """Configure the build behavior by setting config variables.

    Parameters
    ----------
    auto_unroll_max_step: int, default=0
        Threshold of number of steps in the loop to be automatically unrolled.
        This takes inner loop count into consideration.

    auto_unroll_max_depth: int, default=4
        The maximum nested level of loops that can be automatically unrolled.

    unroll_explicit: bool, default=True
        Whether explicitly unroll the loop, if set false, the unroll hint will
        be passed to the CodeGen phase, which may generate pragma unroll hint.
        Set this to be true if CodeGen support unroll pragma and
        when we want to be more readable.

    detect_global_barrier: bool, default=True
        Whether detect global barrier.

    data_alignment: int, optional
        The alignment of data pointer in bytes.
        If -1 is passed, the alignment will be set to TVM's internal default.

    offset_factor: int, default=0
        The factor used in default buffer declaration.
        If specified as 0, offset field is not used.

    restricted_func: bool, default=True
        Whether build restricted function.
        That is each buffer argument to the function are guaranteed
        not to overlap. This enables more optimization.
        Corresponds to restricted keyword in C99

    double_buffer_split_loop: int, default=2
        Whether split the loop with factor. If it is zero, no splitting will happen.
        It it is bigger than one, the logic will do a split with factor equals the integer
        and unroll the inner loop. This allows the buffer fetching won't contain condition.

    add_lower_pass: list of tuiple (phase, function(Stmt->Stmt)), default=None
        phase contains an integer on which optimization pass we apply the pass.
        Additional lowering passes to be applied before make_api.

    Returns
    -------
    config: BuildConfig
        The build configuration
    """
    return BuildConfig(**kwargs)


def get_binds(args, binds=None):
    """Internal function to get binds and arg_list given arguments.

    Parameters
    ----------
    args : list of Buffer or Tensor or Var
        The argument lists to the function.

    binds : dict of :any:`Tensor` to :any:`Buffer`, optional
        Dictionary that maps the Tensor to Buffer which specified the data layout
        requirement of the function. By default, a new compact buffer is created
        for each tensor in the argument.

    Returns
    -------
    binds: dict
        The bind specification

    arg_list: list
        The list of symbolic buffers of arguments.
    """
    binds = {} if binds is None else binds.copy()
    cfg = BuildConfig.current
    arg_list = []
    for x in args:
        if isinstance(x, tensor.Tensor):
            if x not in binds:
                buf = api.decl_buffer(x.shape,
                                      dtype=x.dtype,
                                      name=x.name,
                                      data_alignment=cfg.data_alignment,
                                      offset_factor=cfg.offset_factor)
                binds[x] = buf
                arg_list.append(buf)
            else:
                arg_list.append(binds[x])
        elif isinstance(x, schedule.Buffer):
            arg_list.append(x)
        elif isinstance(x, expr.Var):
            arg_list.append(x)
        else:
            raise ValueError("args must be Tensor, Buffer or Var")
    return binds, arg_list


def lower(sch,
          args,
          name="default_function",
          binds=None,
          simple_mode=False):
    """Lowering step before build into target.

    Parameters
    ----------
    sch : tvm.Schedule
        The schedule to be builded

    args : list of Buffer or Tensor or Var
        The argument lists to the function.

    name : str, optional
        The name of result function.

    binds : dict of :any:`Tensor` to :any:`Buffer`, optional
        Dictionary that maps the Tensor to Buffer which specified the data layout
        requirement of the function. By default, a new compact buffer is created
        for each tensor in the argument.

    simple_mode : bool, optional
        Whether only output simple and compact statement, this will skip
        LoopPartition, api wrapper generation and Unrolling.

    Returns
    -------
    f : LoweredFunc or Stmt
       The result function, if with_api_wrapper=False
       Then the Stmt before make api is returned.
    """
    binds, arg_list = get_binds(args, binds)
    cfg = BuildConfig.current
    add_lower_pass = cfg.add_lower_pass if cfg.add_lower_pass else []
    lower_phase0 = [x[1] for x in add_lower_pass if x[0] == 0]
    lower_phase1 = [x[1] for x in add_lower_pass if x[0] == 1]
    lower_phase2 = [x[1] for x in add_lower_pass if x[0] == 2]
    lower_phase3 = [x[1] for x in add_lower_pass if x[0] > 2]
    # normalize schedule first
    sch = sch.normalize()
    # Phase 0
    bounds = schedule.InferBound(sch)
    stmt = schedule.ScheduleOps(sch, bounds)
    stmt = ir_pass.InjectPrefetch(stmt)
    for f in lower_phase0:
        stmt = f(stmt)
    # Phase 1
    stmt = ir_pass.StorageFlatten(stmt, binds, 64)
    stmt = ir_pass.CanonicalSimplify(stmt)
    for f in lower_phase1:
        stmt = f(stmt)
    # Phase 2
    if not simple_mode:
        stmt = ir_pass.LoopPartition(stmt)
    stmt = ir_pass.VectorizeLoop(stmt)
    stmt = ir_pass.InjectVirtualThread(stmt)
    stmt = ir_pass.InjectDoubleBuffer(stmt, cfg.double_buffer_split_loop)
    stmt = ir_pass.StorageRewrite(stmt)
    stmt = ir_pass.UnrollLoop(
        stmt,
        cfg.auto_unroll_max_step,
        cfg.auto_unroll_max_depth,
        cfg.auto_unroll_max_extent,
        cfg.unroll_explicit)
    for f in lower_phase2:
        stmt = f(stmt)
    # Phase 2
    stmt = ir_pass.Simplify(stmt)
    stmt = ir_pass.LowerStorageAccessInfo(stmt)
    stmt = ir_pass.RemoveNoOp(stmt)
    stmt = ir_pass.RewriteUnsafeSelect(stmt)
    for f in lower_phase3:
        stmt = f(stmt)
    if simple_mode:
        return stmt
    return ir_pass.MakeAPI(stmt, name, arg_list, 0, cfg.restricted_func)


def build(sch,
          args=None,
          target=None,
          target_host=None,
          name="default_function",
          binds=None):
    """Build a function with arguments as signiture.

    Parameters
    ----------
    sch : tvm.Schedule, or LoweredFunc
        The schedule to be builded

    args : list of Buffer or Tensor or Var, optional
        The argument lists to the function.

    target : str or :any:`tvm.target.Target`, optional
        The target and option of the compilation.

    target_host : str or :any:`tvm.target.Target` optional
        Host compilation target, if target is device.
        When TVM compiles device specific program such as CUDA,
        we also need host(CPU) side code to interact with the driver
        setup the dimensions and parameters correctly.
        target_host is used to specify the host side codegen target.
        By default, llvm is used if it is enabled,
        otherwise a stackvm intepreter is used.

    name : str, optional
        The name of result function.

    binds : dict, optional
        Dictionary that maps the binding of symbolic buffer to Tensor.
        By default, a new buffer is created for each tensor in the argument.

    Returns
    -------
    f : Function, or pair of functions
       The result function.

    Note
    ----
    See the note on :any:`tvm.target` on target string format.
    """
    if isinstance(sch, schedule.Schedule):
        if args is None:
            raise ValueError("args must be given for build from schedule")
        flist = lower(sch, args,
                      name=name,
                      binds=binds)
        if isinstance(flist, container.LoweredFunc):
            flist = [flist]
    elif isinstance(sch, container.LoweredFunc):
        if args:
            raise ValueError("args must be done when build from LoweredFunc")
        flist = [sch]
    elif isinstance(sch, (list, tuple, container.Array)):
        flist = sch
    else:
        raise ValueError("sch have to be Schedule, LoweredFunc or list of LoweredFunc")
    fname_set = set()
    for x in flist:
        if not isinstance(x, container.LoweredFunc):
            raise ValueError("sch have to be Schedule, LoweredFunc or list of LoweredFunc")
        if x.name in fname_set:
            raise ValueError("Duplicate function name %s" % x.name)

    target = _target.current_target() if target is None else target
    target = _target.create(target) if target else _target.create("llvm")

    fhost = []
    fdevice = []
    for func in flist:
        if func.func_type == container.LoweredFunc.MixedFunc:
            if BuildConfig.current.detect_global_barrier:
                func = ir_pass.ThreadSync(func, "global")
            func = ir_pass.ThreadSync(func, "shared")
            warp_size = target.thread_warp_size
            func = ir_pass.LowerThreadAllreduce(func, warp_size)
            fsplits = [s for s in ir_pass.SplitHostDevice(func)]
            fhost.append(fsplits[0])
            for x in fsplits[1:]:
                fdevice.append(x)
        elif func.func_type == container.LoweredFunc.HostFunc:
            fhost.append(func)
        elif func.func_type == container.LoweredFunc.DeviceFunc:
            fdevice.append(func)
        else:
            raise ValueError("unknown function type %d" % func.func_type)

    if "gpu" in target.keys and not fdevice:
        warnings.warn(
            "Specified target %s, but cannot find device code, did you do bind?" % target)

    device_type = ndarray.context(target.target_name, 0).device_type
    fhost = [ir_pass.BindDeviceType(x, device_type) for x in fhost]
    fhost = [ir_pass.LowerTVMBuiltin(x) for x in fhost]

    if not target_host:
        if device_type == ndarray.cpu(0).device_type:
            target_host = target
            assert not fdevice
        else:
            target_host = "llvm" if module.enabled("llvm") else "stackvm"
    target_host = _target.create(target_host)
    target_device = target
    fdevice = [ir_pass.LowerIntrin(x, target_device.target_name) for x in fdevice]
    fhost = [ir_pass.LowerIntrin(x, target_host.target_name) for x in fhost]
    fhost = [ir_pass.CombineContextCall(x) for x in fhost]
    mhost = codegen.build_module(fhost, str(target_host))

    if fdevice:
        mdev = codegen.build_module(fdevice, str(target_device))
        mhost.import_module(mdev)
    return mhost
