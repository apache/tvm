"""The build utils in python.

This module provides the functions to transform schedule to
LoweredFunc and compiled Module.
"""
from __future__ import absolute_import as _abs
import warnings
import types

from ._ffi.node import NodeBase, register_node
from . import api
from . import _api_internal
from . import tensor
from . import schedule
from . import expr
from . import ir_pass
from . import stmt as _stmt
from . import container
from . import module
from . import codegen
from . import ndarray
from . import target as _target
from . import make

class DumpIR(object):
    """
    Dump IR for each pass.
    With it, you can dump ir just like gcc/llvm.

    How to use:
    -----------
    .. code-block:: python

        with tvm.build_config(dump_pass_ir=True)
            run()
    """
    scope_level = 0
    def __init__(self):
        self._pass_id = 0
        self._recover_list = []

    def decorate(self, func):
        """ decorate the pass function"""
        def dump(*args, **kwargs):
            """dump function"""
            retv = func(*args, **kwargs)
            if not isinstance(retv, (_stmt.Stmt, container.LoweredFunc, container.Array)):
                return retv
            fname = func.func_name if hasattr(func, 'func_name') else func.__name__
            pname = str(self._pass_id) + "_" + fname + "_ir.cc"
            with open(pname, "a") as f:
                out = retv.body if isinstance(retv, container.LoweredFunc) else retv
                f.write(str(out))
                if isinstance(retv, container.Array):
                    for x in retv:
                        out = x.body if isinstance(x, container.LoweredFunc) else x
                        f.write("---------%s\n%s\n-----------\n"%(x.name, str(out)))
                self._pass_id += 1
            return retv
        return dump

    def decorate_irpass(self):
        """decorate ir_pass and ScheduleOps"""
        self._old_sgpass = schedule.ScheduleOps
        schedule.ScheduleOps = self.decorate(schedule.ScheduleOps)
        vset = vars(ir_pass)
        k = v = 0
        def recover():
            vset[k] = v
        for k, v in vset.items():
            self._recover_list.append(recover)
            vset[k] = self.decorate(v) if isinstance(v, types.FunctionType) else v

    def decorate_custompass(self, custom_pass):
        """decorate given list of custom passes, and return decorated passes"""
        custom_pass = custom_pass if custom_pass else []
        pass_list = []
        for idx, x in enumerate(custom_pass):
            x[1].__name__ = "custom{}_phase{}".format(idx, x[0])
            pass_list += [(x[0], self.decorate(x[1]))]
        return pass_list

    def enter(self):
        """only decorate outermost nest"""
        if DumpIR.scope_level > 0:
            return
        self.decorate_irpass()
        self._pass_id = 0
        DumpIR.scope_level += 1

    def exit(self):
        """recover outermost nest"""
        if DumpIR.scope_level > 1:
            return
        # recover decorated functions
        for f in self._recover_list:
            f()
        schedule.ScheduleOps = self._old_sgpass
        DumpIR.scope_level -= 1


@register_node
class BuildConfig(NodeBase):
    """Configuration scope to set a build config option.

    Note
    ----
    This object is backed by node system in C++, with arguments that can be
    exchanged between python and C++.

    Do not construct directly, use build_config instead.

    The fields that are backed by the C++ node are immutable once an instance
    is constructed. See _node_defaults for the fields.
    """

    _node_defaults = {
        "auto_unroll_max_step": 0,
        "auto_unroll_max_depth": 8,
        "auto_unroll_max_extent": 0,
        "unroll_explicit": True,
        "detect_global_barrier": False,
        "partition_const_loop": False,
        "offset_factor": 0,
        "data_alignment": -1,
        "restricted_func": True,
        "double_buffer_split_loop": 1,
        "dump_pass_ir": False
    }
    _dump_ir = DumpIR()

    # pylint: disable=no-member
    def __init__(self, handle):
        """Initialize the function with handle

        Parameters
        ----------
        handle : SymbolHandle
            the handle to the underlying C++ Symbol
        """
        super(BuildConfig, self).__init__(handle)
        self.handle = handle

    @property
    def add_lower_pass(self):
        size = _api_internal._BuildConfigGetAddLowerPassInfo(self)
        result = []
        for i in range(size):
            phase = _api_internal._BuildConfigGetAddLowerPassInfo(self, i, True)
            func = _api_internal._BuildConfigGetAddLowerPassInfo(self, i, False)
            result += [(phase, func)]
        return result

    @add_lower_pass.setter
    def add_lower_pass(self, value):
        add_lower_pass_args = []
        for x in value:
            add_lower_pass_args += [x[0], x[1]]
        _api_internal._BuildConfigSetAddLowerPass(self, *add_lower_pass_args)

    def __enter__(self):
        # pylint: disable=protected-access
        _api_internal._EnterBuildConfigScope(self)
        if self.dump_pass_ir:
            BuildConfig._dump_ir.enter()
        return self

    def __exit__(self, ptype, value, trace):
        if self.dump_pass_ir:
            BuildConfig._dump_ir.exit()
        _api_internal._ExitBuildConfigScope()

    def __setattr__(self, name, value):
        if name in BuildConfig._node_defaults:
            raise AttributeError(
                "'%s' object cannot set attribute '%s'" % (str(type(self)), name))
        return super(BuildConfig, self).__setattr__(name, value)


def current_build_config():
    """Get the current build configuration."""
    return _api_internal._GetCurrentBuildConfig()


def build_config(**kwargs):
    """Configure the build behavior by setting config variables.

    Parameters
    ----------
    auto_unroll_max_step: int, default=0
        Threshold of number of steps in the loop to be automatically unrolled.
        This takes inner loop count into consideration.

    auto_unroll_max_depth: int, default=8
        The maximum nested level of loops that can be automatically unrolled.

    unroll_explicit: bool, default=True
        Whether explicitly unroll the loop, if set false, the unroll hint will
        be passed to the CodeGen phase, which may generate pragma unroll hint.
        Set this to be true if CodeGen support unroll pragma and
        when we want to be more readable.

    detect_global_barrier: bool, default=True
        Whether detect global barrier.

    partition_const_loop: bool, default=False
        Whether partition const loop

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

    add_lower_pass: list of tuple (phase, function(Stmt->Stmt)), default=None
        phase contains an integer on which optimization pass we apply the pass.
        Additional lowering passes to be applied before make_api.

    dump_pass_ir: dump ir of each pass into file idx_passname_ir.cc, default=False

    Returns
    -------
    config: BuildConfig
        The build configuration
    """
    node_args = {k: v if k not in kwargs else kwargs[k]
                 for k, v in BuildConfig._node_defaults.items()}
    config = make.node("BuildConfig", **node_args)

    if "add_lower_pass" in kwargs:
        config.add_lower_pass = kwargs["add_lower_pass"]

    return config

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
    cfg = current_build_config()
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
    cfg = current_build_config()
    add_lower_pass = cfg.add_lower_pass if cfg.add_lower_pass else []
    if cfg.dump_pass_ir:
        add_lower_pass = BuildConfig._dump_ir.decorate_custompass(add_lower_pass)
    lower_phase0 = [x[1] for x in add_lower_pass if x[0] == 0]
    lower_phase1 = [x[1] for x in add_lower_pass if x[0] == 1]
    lower_phase2 = [x[1] for x in add_lower_pass if x[0] == 2]
    lower_phase3 = [x[1] for x in add_lower_pass if x[0] > 2]

    # Phase 0
    if isinstance(sch, schedule.Schedule):
        # normalize schedule first
        sch = sch.normalize()
        bounds = schedule.InferBound(sch)
        stmt = schedule.ScheduleOps(sch, bounds)
        stmt = ir_pass.InjectPrefetch(stmt)
    else:
        #So far there is no op for hybrid script, so a plain ir body is given
        if not isinstance(sch, _stmt.Stmt):
            raise ValueError("sch should be either a Schedule or a Stmt")
        stmt = sch

    for f in lower_phase0:
        stmt = f(stmt)
    # Phase 1
    stmt = ir_pass.StorageFlatten(stmt, binds, 64)
    stmt = ir_pass.CanonicalSimplify(stmt)
    for f in lower_phase1:
        stmt = f(stmt)
    # Phase 2
    if not simple_mode:
        stmt = ir_pass.LoopPartition(stmt, cfg.partition_const_loop)
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
    # Phase 3
    stmt = ir_pass.Simplify(stmt)
    stmt = ir_pass.LowerStorageAccessInfo(stmt)
    stmt = ir_pass.RemoveNoOp(stmt)
    stmt = ir_pass.RewriteUnsafeSelect(stmt)
    for f in lower_phase3:
        stmt = f(stmt)
    if simple_mode:
        return stmt
    return ir_pass.MakeAPI(stmt, name, arg_list, 0, cfg.restricted_func)


def _build_for_device(flist, target, target_host):
    """Build the lowered functions for a device with the given compilation
    target.

    Parameters
    ----------
    flist : list of LoweredFunc
        The schedule to be built.

    target : str or :any:`tvm.target.Target`
        The target and option of the compilation.

    target_host : str or :any:`tvm.target.Target`
        The host compilation target.

    Returns
    -------
    fhost : list of LoweredFunc
        A list of lowered functions for the host.

    mdev : tvm.module
        A module that contains device code.
    """
    target = _target.create(target)
    device_type = ndarray.context(target.target_name, 0).device_type
    fhost = []
    fdevice = []
    for func in flist:
        if not ir_pass.VerifyMemory(func, device_type):
            raise ValueError(
                "Direct host side access to device memory is detected in %s. "
                "Did you forget to bind?" % func.name)
        if func.func_type == container.LoweredFunc.MixedFunc:
            if current_build_config().detect_global_barrier:
                func = ir_pass.ThreadSync(func, "global")
            func = ir_pass.ThreadSync(func, "shared")
            func = ir_pass.ThreadSync(func, "warp")
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

    for i, func in enumerate(fdevice):
        warp_size = target.thread_warp_size
        fdevice[i] = ir_pass.LowerWarpMemory(func, warp_size)

    if "gpu" in target.keys and not fdevice:
        warnings.warn(
            "Specified target %s, but cannot find device code, did you do "
            "bind?" % target)

    fhost = [ir_pass.BindDeviceType(x, device_type) for x in fhost]
    fhost = [ir_pass.LowerTVMBuiltin(x) for x in fhost]

    if device_type == ndarray.cpu(0).device_type and target_host == target:
        assert not fdevice

    target_host = _target.create(target_host)
    fdevice = [ir_pass.LowerIntrin(x, target.target_name) for x in fdevice]
    fhost = [ir_pass.LowerIntrin(x, target_host.target_name) for x in fhost]
    fhost = [ir_pass.CombineContextCall(x) for x in fhost]
    mdev = codegen.build_module(fdevice, str(target)) if fdevice else None

    return fhost, mdev


def build(inputs,
          args=None,
          target=None,
          target_host=None,
          name="default_function",
          binds=None):
    """Build a function with arguments as signature. Code will be generated
    for devices coupled with target information.

    Parameters
    ----------
    inputs : tvm.Schedule, LoweredFunc, or dict of target to LoweredFunc list
        The schedule to be built

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
    ret : tvm.module
        A module that combines both host and device code.

    Examples
    ________
    There are two typical example uses of this function depending on the type
    of the argument `inputs`:
    1. it is a list of lowered functions:

    .. code-block:: python

        n = 2
        A = tvm.placeholder((n,), name='A')
        B = tvm.placeholder((n,), name='B')
        C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
        s = tvm.create_schedule(C.op)
        f = tvm.lower(s, [A, B, C], name="test_add")
        m = tvm.build(f, target="llvm")

    2. it is a dict of compilation target to list of lowered functions:

    .. code-block:: python

        n = 2
        A = tvm.placeholder((n,), name='A')
        B = tvm.placeholder((n,), name='B')
        C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
        s1 = tvm.create_schedule(C.op)
        s2 = topi.cpp.cuda.schedule_injective("cuda", [C])
        f1 = tvm.lower(s1, [A, B, C], name="test_add1")
        f2 = tvm.lower(s2, [A, B, C], name="test_add2")
        m = tvm.build({"llvm": [f1], "cuda": [f2]}, target_host="llvm")

    Note
    ----
    See the note on :any:`tvm.target` on target string format.
    """
    if isinstance(inputs, schedule.Schedule):
        if args is None:
            raise ValueError("args must be given for build from schedule")
        flist = lower(inputs, args,
                      name=name,
                      binds=binds)
        if isinstance(flist, container.LoweredFunc):
            flist = [flist]
    elif isinstance(inputs, container.LoweredFunc):
        if args:
            raise ValueError("args must be done when build from LoweredFunc.")
        flist = [inputs]
    elif isinstance(inputs, (list, tuple, container.Array)):
        flist = inputs
    elif not isinstance(inputs, (dict, container.Map)):
        raise ValueError("inputs must be Schedule, LoweredFunc, list of "
                         "LoweredFunc, or dict of target to list of "
                         "LoweredFunc.")

    if not isinstance(inputs, (dict, container.Map)):
        target = _target.current_target() if target is None else target
        target = target if target else "llvm"
        target_flist = {target: flist}
    else:
        target_flist = inputs

    for tar, flist in target_flist.items():
        if not isinstance(tar, (str, _target.Target)):
            raise ValueError("The key of inputs must be str or "
                             "_target.Target when inputs is dict.")
        fname_set = set()
        for x in flist:
            if not isinstance(x, container.LoweredFunc):
                raise ValueError("inputs must be Schedule, LoweredFunc, list "
                                 "of LoweredFunc, or dict of str to list of "
                                 "LoweredFunc.")
            if x.name in fname_set:
                raise ValueError("Duplicate function name %s" % x.name)
            fname_set.add(x.name)

    if not target_host:
        for tar, _ in target_flist.items():
            tar = _target.create(tar)
            device_type = ndarray.context(tar.target_name, 0).device_type
            if device_type == ndarray.cpu(0).device_type:
                target_host = tar
                break
    if not target_host:
        target_host = "llvm" if module.enabled("llvm") else "stackvm"

    fhost_all = []
    device_modules = []
    for tar, flist in target_flist.items():
        fhost, mdev = _build_for_device(flist, tar, target_host)
        # Save the current lowered functions of the host and the device module.
        fhost_all += fhost
        device_modules.append(mdev)

    # Generate a unified host module.
    mhost = codegen.build_module(fhost_all, str(target_host))

    # Import all modules.
    for mdev in device_modules:
        if mdev:
            mhost.import_module(mdev)
    return mhost
