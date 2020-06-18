# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=invalid-name
"""The build utils in python.
"""
import warnings

import tvm.tir

from tvm.runtime import ndarray
from tvm.ir import container
from tvm.ir import CallingConv
from tvm.target import codegen, BuildConfig
from tvm.tir import ir_pass
from tvm.te import tensor
from tvm.te import schedule
from tvm import target as _target


def get_binds(args, compact=False, binds=None):
    """Internal function to get binds and arg_list given arguments.

    Parameters
    ----------
    args : list of Buffer or Tensor or Var
        The argument lists to the function.

    compact : bool
        If the statement has already bound to a compact buffer.

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
    cfg = BuildConfig.current()
    arg_list = []
    for x in args:
        if isinstance(x, tensor.Tensor):
            any_dim = any(isinstance(i, tvm.tir.Var) for i in x.shape)
            buffer_type = "auto_broadcast" if any_dim and not compact else ""
            if x not in binds:
                buf = tvm.tir.decl_buffer(
                    x.shape,
                    dtype=x.dtype,
                    name=x.name,
                    data_alignment=cfg.data_alignment,
                    offset_factor=cfg.offset_factor,
                    buffer_type=buffer_type)
                binds[x] = buf
                arg_list.append(buf)
            else:
                arg_list.append(binds[x])
        elif isinstance(x, schedule.Buffer):
            arg_list.append(x)
        elif isinstance(x, tvm.tir.Var):
            arg_list.append(x)
        else:
            raise ValueError("args must be Tensor, Buffer or Var")
    return binds, arg_list


def form_body(sch):
    """According to the given schedule, form the raw body
    Parameters
    ----------
    sch : tvm.te.schedule.Schedule
    The given scheduler to form the raw body

    Returns
    -------
    The body formed according to the given schedule
    """
    # normalize schedule first
    sch = sch.normalize()
    bounds = schedule.InferBound(sch)
    stmt = schedule.ScheduleOps(sch, bounds)
    stmt = ir_pass.InjectPrefetch(stmt)
    return stmt


def lower(sch,
          args,
          name="default_function",
          binds=None,
          simple_mode=False):
    """Lowering step before build into target.

    Parameters
    ----------
    sch : tvm.te.schedule.Schedule
        The schedule to be built

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
    m : IRModule or Stmt
       The result IRModule, if simple_mode=False
       Then the Stmt before make api is returned.
    """
    cfg = BuildConfig.current()
    add_lower_pass = cfg.add_lower_pass if cfg.add_lower_pass else []
    if cfg.dump_pass_ir:
        add_lower_pass = BuildConfig._dump_ir.decorate_custompass(add_lower_pass)
    lower_phase0 = [x[1] for x in add_lower_pass if x[0] == 0]
    lower_phase1 = [x[1] for x in add_lower_pass if x[0] == 1]
    lower_phase2 = [x[1] for x in add_lower_pass if x[0] == 2]
    lower_phase3 = [x[1] for x in add_lower_pass if x[0] > 2]

    # Phase 0
    if isinstance(sch, schedule.Schedule):
        stmt = form_body(sch)

    for f in lower_phase0:
        stmt = f(stmt)

    compact = ir_pass.VerifyCompactBuffer(stmt)
    binds, arg_list = get_binds(args, compact, binds)

    # Phase 1
    stmt = ir_pass.RewriteForTensorCore(stmt, sch, binds)
    stmt = ir_pass.StorageFlatten(stmt, binds, 64, cfg.instrument_bound_checkers)
    stmt = ir_pass.NarrowDataType(stmt, 32)
    stmt = ir_pass.CanonicalSimplify(stmt)
    for f in lower_phase1:
        stmt = f(stmt)

    # Phase 2
    if not simple_mode:
        stmt = ir_pass.LoopPartition(stmt, cfg.partition_const_loop)
    if cfg.disable_vectorize:
        stmt = ir_pass.SkipVectorize(stmt)
    else:
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
    stmt = ir_pass.RemoveNoOp(stmt)
    if not cfg.disable_select_rewriting:
        stmt = ir_pass.RewriteUnsafeSelect(stmt)
    for f in lower_phase3:
        stmt = f(stmt)
    # Instrument BoundCheckers
    if cfg.instrument_bound_checkers:
        stmt = ir_pass.InstrumentBoundCheckers(stmt)
    if simple_mode:
        return stmt

    f = tvm.tir.PrimFunc(arg_list, stmt).with_attr(
        "global_symbol", tvm.runtime.String(name))
    if cfg.restricted_func:
        f = f.with_attr("tir.noalias", True)
    mod = tvm.IRModule({name: f})
    return mod


def _build_for_device(input_mod, target, target_host):
    """Build the lowered functions for a device with the given compilation
    target.

    Parameters
    ----------
    input_mod : IRModule
        The schedule to be built.

    target : str or :any:`tvm.target.Target`
        The target and option of the compilation.

    target_host : str or :any:`tvm.target.Target`
        The host compilation target.

    Returns
    -------
    fhost : IRModule
        The host IRModule.

    mdev : tvm.module
        A module that contains device code.
    """
    target = _target.create(target)
    target_host = _target.create(target_host)
    device_type = ndarray.context(target.target_name, 0).device_type

    mod_mixed = input_mod
    mod_mixed = tvm.tir.transform.Apply(lambda f: f.with_attr("target", target))(mod_mixed)
    tvm.tir.analysis.verify_memory(mod_mixed)

    opt_mixed = []
    if len(mod_mixed.functions) == 1:
        opt_mixed += [tvm.tir.transform.Apply(lambda f: f.with_attr("tir.is_entry_func", True))]
    if BuildConfig.current().detect_global_barrier:
        opt_mixed += [tvm.tir.transform.ThreadSync("global")]
    opt_mixed += [tvm.tir.transform.ThreadSync("shared"),
                  tvm.tir.transform.ThreadSync("warp"),
                  tvm.tir.transform.InferFragment(),
                  tvm.tir.transform.LowerThreadAllreduce(),
                  tvm.tir.transform.MakePackedAPI(),
                  tvm.tir.transform.SplitHostDevice()]
    mod_mixed = tvm.transform.Sequential(opt_mixed)(mod_mixed)


    # device optimizations
    opt_device = tvm.transform.Sequential(
        [tvm.tir.transform.Filter(
            lambda f: "calling_conv" in f.attrs and
            f.attrs["calling_conv"].value == CallingConv.DEVICE_KERNEL_LAUNCH),
         tvm.tir.transform.LowerWarpMemory(),
         tvm.tir.transform.LowerDeviceStorageAccessInfo(),
         tvm.tir.transform.LowerIntrin()])
    mod_dev = opt_device(mod_mixed)

    # host optimizations
    opt_host = tvm.transform.Sequential(
        [tvm.tir.transform.Filter(
            lambda f: "calling_conv" not in f.attrs or
            f.attrs["calling_conv"].value != CallingConv.DEVICE_KERNEL_LAUNCH),
         tvm.tir.transform.Apply(lambda f: f.with_attr("target", target)),
         tvm.tir.transform.LowerTVMBuiltin(),
         tvm.tir.transform.LowerDeviceStorageAccessInfo(),
         tvm.tir.transform.LowerIntrin(),
         tvm.tir.transform.CombineContextCall()])
    mod_host = opt_host(mod_mixed)

    if device_type == ndarray.cpu(0).device_type and target_host == target:
        assert len(mod_dev.functions) == 0
    if "gpu" in target.keys and len(mod_dev.functions) == 0:
        warnings.warn(
            "Specified target %s, but cannot find device code, did you do "
            "bind?" % target)

    rt_mod_dev = codegen.build_module(mod_dev, target) if len(mod_dev.functions) != 0 else None
    return mod_host, rt_mod_dev


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
    inputs : tvm.te.Schedule, IRModule, or dict of target to IRModule
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
    1. it is an IRModule.

    .. code-block:: python

        n = 2
        A = te.placeholder((n,), name='A')
        B = te.placeholder((n,), name='B')
        C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
        s = tvm.te.create_schedule(C.op)
        m = tvm.lower(s, [A, B, C], name="test_add")
        rt_mod = tvm.build(m, target="llvm")

    2. it is a dict of compilation target to IRModule.

    .. code-block:: python

        n = 2
        A = te.placeholder((n,), name='A')
        B = te.placeholder((n,), name='B')
        C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
        s1 = tvm.te.create_schedule(C.op)
        with tvm.target.cuda() as cuda_tgt:
          s2 = topi.cuda.schedule_injective(cuda_tgt, [C])
          m1 = tvm.lower(s1, [A, B, C], name="test_add1")
          m2 = tvm.lower(s2, [A, B, C], name="test_add2")
          rt_mod = tvm.build({"llvm": m1, "cuda": m2}, target_host="llvm")

    Note
    ----
    See the note on :any:`tvm.target` on target string format.
    """
    if isinstance(inputs, schedule.Schedule):
        if args is None:
            raise ValueError("args must be given for build from schedule")
        input_mod = lower(inputs, args,
                          name=name,
                          binds=binds)
    elif isinstance(inputs, (list, tuple, container.Array)):
        merged_mod = tvm.IRModule({})
        for x in inputs:
            merged_mod.update(x)
        input_mod = merged_mod
    elif isinstance(inputs, tvm.IRModule):
        input_mod = inputs
    elif not isinstance(inputs, (dict, container.Map)):
        raise ValueError("inputs must be Schedule, IRModule or dict of target to IRModule")

    if not isinstance(inputs, (dict, container.Map)):
        target = _target.Target.current() if target is None else target
        target = target if target else "llvm"
        target_input_mod = {target: input_mod}
    else:
        target_input_mod = inputs

    for tar, mod in target_input_mod.items():
        if not isinstance(tar, (str, _target.Target)):
            raise ValueError("The key of inputs must be str or "
                             "_target.Target when inputs is dict.")
        if not isinstance(mod, tvm.IRModule):
            raise ValueError("inputs must be Schedule, IRModule,"
                             "or dict of str to IRModule.")

    if not target_host:
        for tar, _ in target_input_mod.items():
            tar = _target.create(tar)
            device_type = ndarray.context(tar.target_name, 0).device_type
            if device_type == ndarray.cpu(0).device_type:
                target_host = tar
                break
    if not target_host:
        target_host = "llvm" if tvm.runtime.enabled("llvm") else "stackvm"

    mod_host_all = tvm.IRModule({})

    device_modules = []
    for tar, input_mod in target_input_mod.items():
        mod_host, mdev = _build_for_device(input_mod, tar, target_host)
        mod_host_all.update(mod_host)
        device_modules.append(mdev)

    # Generate a unified host module.
    rt_mod_host = codegen.build_module(mod_host_all, target_host)

    # Import all modules.
    for mdev in device_modules:
        if mdev:
            rt_mod_host.import_module(mdev)
    return rt_mod_host
