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
"""The build utils in python."""
from typing import Union, Optional, Dict, Tuple
import enum
import tvm
from tvm import tir, ir
from tvm.runtime import ndarray
from tvm.tir import PrimFunc
from tvm.ir.module import IRModule
from tvm.target import Target
from tvm._ffi.runtime_ctypes import Device


def create_pass_list(disable_loop_partition: bool):
    """Create a list of passes based on pass context configurations.

    Parameters
    ----------
    disable_loop_partition : bool
        Whether to disable loop partition pass.

    Returns
    -------
    List[tvm.tir.transform.Pass]
        List of passes to run.
    """
    pass_ctx = tvm.transform.PassContext.current()
    config = pass_ctx.config
    # Retrieve configuration flags.
    disable_vectorize = bool(config.get("tir.disable_vectorize", False))
    disable_storage_rewrite = bool(config.get("tir.disable_storage_rewrite", False))
    instrument_bound_checkers = bool(config.get("tir.instrument_bound_checkers", False))
    disable_cse_tir = bool(config.get("tir.disable_cse_tir", False))
    enable_equiv_terms_in_cse_tir = bool(config.get("tir.enable_equiv_terms_in_cse_tir", False))
    ptx_ldg32 = bool(config.get("tir.ptx_ldg32", False))
    instrument_lwp = bool(config.get("tir.instrument_lwp", False))
    add_lower_pass = config.get("tir.add_lower_pass", [])

    # Group user passes by phase (phases 0, 1, 2, and 3 where phase>=3 goes to 3)
    user_passes = {0: [], 1: [], 2: [], 3: []}
    for phase, p in add_lower_pass:
        if not isinstance(phase, int) or phase < 0:
            raise ValueError(
                f"Phase number must be a non-negative integer, got {phase} of type {type(phase)}"
            )
        user_passes[phase if phase < 3 else 3].append(p)

    # Construct phase-specific passes.
    phase0 = user_passes[0]

    phase1 = [
        tir.transform.InjectPrefetch(),
        tir.transform.TextureFlatten(),
        tir.transform.StorageFlatten(64, instrument_bound_checkers),
        tir.transform.LowerCrossThreadReduction(),
        tir.transform.LowerInitBlock(),
        tir.transform.PlanAndUpdateBufferAllocationLocation(),
        tir.transform.ConvertBlocksToOpaque(),
        tir.transform.LiftThreadBinding(),
        tir.transform.ManifestSharedMemoryLocalStage(),
        tir.transform.CompactBufferAllocation(),
        tir.transform.LowerAutoCopy(),
        tir.transform.UnifyThreadBinding(),
        tir.transform.LowerMatchBuffer(),
        tir.transform.Simplify(),
        tir.transform.InjectPermutedLayout(),
        tir.transform.Simplify(),
        tir.transform.InjectSoftwarePipeline(),
        tir.transform.TransformMmaBufferLayout(),
        tir.transform.LowerOpaqueBlock(),
        tir.transform.FlattenBuffer(),
        tir.transform.BF16ComputeLegalize(),
        tir.transform.NarrowDataType(32),
        tir.transform.Simplify(),
    ] + user_passes[1]

    phase2 = []
    if not disable_loop_partition:
        phase2.append(tir.transform.LoopPartition())
    phase2.extend(
        [
            tir.transform.VectorizeLoop(not disable_vectorize),
            tir.transform.InjectVirtualThread(),
            tir.transform.InjectDoubleBuffer(),
        ]
    )
    if not disable_storage_rewrite:
        phase2.append(tir.transform.StorageRewrite())
    if config.get("tir.use_async_copy", False):
        phase2.append(tir.transform.LowerAsyncDMA())
    phase2.extend(
        [
            tir.transform.HoistIfThenElse(),
            tir.transform.UnrollLoop(),
        ]
    )
    phase2 += user_passes[2]

    phase3 = [
        tir.transform.RenormalizeSplitPattern(),
        tir.transform.Simplify(),
        tir.transform.RemoveNoOp(),
        tir.transform.RewriteUnsafeSelect(),
    ] + user_passes[3]

    # Additional passes based on configuration.
    extras = []
    if instrument_bound_checkers:
        extras.append(tir.transform.InstrumentBoundCheckers())
    if ptx_ldg32:
        extras.append(tir.transform.InjectPTXLDG32(True))
    extras.append(
        tir.transform.CommonSubexprElimTIR(not disable_cse_tir, enable_equiv_terms_in_cse_tir)
    )
    if instrument_lwp:
        extras.append(tir.transform.InstrumentProfileIntrinsics())

    return phase0 + phase1 + phase2 + phase3 + extras


def lower_module(inp: IRModule, simple_mode: bool = False) -> IRModule:
    """Lowering step before building the target.

    Parameters
    ----------
    inp : IRModule
        The IRModule to be lowered.
    simple_mode : bool
        Whether to output only a simple, compact statement.

    Returns
    -------
    IRModule
        The lowered IRModule.
    """
    return tvm.ir.transform.Sequential(create_pass_list(simple_mode))(inp)


def lower_primfunc(inp: PrimFunc, name: str = "main", simple_mode: bool = False) -> IRModule:
    """Lowering step before building the target for a PrimFunc.

    Parameters
    ----------
    inp : PrimFunc
        The PrimFunc to be lowered.
    name : str
        The name of the resulting function.
    simple_mode : bool
        Whether to output only a simple, compact statement.

    Returns
    -------
    IRModule
        The lowered IRModule.
    """
    pass_ctx = tvm.ir.transform.PassContext.current()
    f = inp.with_attr("global_symbol", name)
    if pass_ctx.config.get("tir.noalias", True):
        f = f.with_attr("tir.noalias", True)
    mod = tvm.ir.IRModule({tvm.ir.GlobalVar(name): f})
    return tvm.ir.transform.Sequential(create_pass_list(simple_mode))(mod)


def lower(
    inp: Union[PrimFunc, IRModule], name: str = "main", simple_mode: bool = False
) -> IRModule:
    """Lowering step before building the target.

    Parameters
    ----------
    inp : Union[PrimFunc, IRModule]
        The PrimFunc or IRModule to be lowered.
    name : str
        The name of the resulting function (if applicable).
    simple_mode : bool
        Whether to output only a simple, compact statement.

    Returns
    -------
    IRModule
        The lowered IRModule.
    """
    if isinstance(inp, IRModule):
        return lower_module(inp, simple_mode)
    if isinstance(inp, PrimFunc):
        return lower_primfunc(inp, name, simple_mode)
    raise ValueError(f"Expected input to be IRModule or PrimFunc, but got {type(inp)}")


def check_and_update_host_consistency(targets: dict, host):
    """
    Check and update the host field of the given legacy heterogeneous targets
    for legacy target API compatibility.

    Parameters
    ----------
    targets : dict
        Dictionary mapping Target objects to IRModule objects.
    host : Target
        The target host to be updated.
    """
    for tgt in list(targets):
        if getattr(tgt, "host", None) is None:
            tgt.host = host


def mixed_module_pass_manager(target: Target) -> tvm.ir.transform.Sequential:
    """
    Constructs a Sequential transformation pass pipeline for a mixed module.

    Parameters
    ----------
    target : Target
        The target device for which the module is intended.

    Returns
    -------
    tvm.ir.transform.Sequential
        A sequential pass pipeline for the mixed module.
    """
    pass_ctx = tvm.ir.transform.PassContext.current()
    mixed_pass_list = [
        # Bind the target first so that target-specific attributes are available.
        tir.transform.BindTarget(target),
        tir.transform.FP8ComputeLegalize(),
        # VerifyVTCMLimit must occur before LowerVtcmAlloc.
        tir.transform.VerifyVTCMLimit(target),
        tir.transform.LowerVtcmAlloc(),
        tir.transform.VerifyMemory(),
        tir.transform.AnnotateEntryFunc(),
    ]
    if pass_ctx.config.get("tir.detect_global_barrier", False):
        mixed_pass_list.append(tir.transform.ThreadSync("global"))
    mixed_pass_list.extend(
        [
            tir.transform.ThreadSync("shared"),
            tir.transform.ThreadSync("shared.dyn"),
            tir.transform.ThreadSync("warp"),
            tir.transform.InferFragment(),
            tir.transform.LowerThreadAllreduce(),
        ]
    )
    if pass_ctx.config.get("tir.use_async_copy", False):
        mixed_pass_list.append(tir.transform.InjectPTXAsyncCopy())
    if pass_ctx.config.get("tir.ptx_ldg32", False):
        mixed_pass_list.append(tir.transform.InjectPTXLDG32())
    mixed_pass_list.extend(
        [
            tir.transform.AnnotateDeviceRegions(),
            tir.transform.SplitHostDevice(),
            # MergeSharedMemoryAllocations must follow SplitHostDevice.
            tir.transform.MergeSharedMemoryAllocations(),
            tir.transform.MakePackedAPI(),
            tir.transform.FP8StorageLegalize(),
            tir.transform.BF16StorageLegalize(),
            tir.transform.LowerDeviceKernelLaunch(),
        ]
    )
    return tvm.ir.transform.Sequential(mixed_pass_list)


class CallConv(enum.IntEnum):
    """
    Enum representing different calling conventions.
    Corresponds to the C++ tvm::ir::CallingConv enum.
    """

    kDefault = 0
    kCPackedFunc = 1
    kDeviceKernelLaunch = 2


def host_module_pass_manager(target_host: Target) -> tvm.ir.transform.Sequential:
    """
    Build a sequential pass pipeline for lowering the host part of a mixed module.

    Parameters
    ----------
    target_host : Target
        The host target for which to lower the module.

    Returns
    -------
    tvm.ir.transform.Sequential
        A sequential pass pipeline for host-specific transformations.
    """
    host_pass_list = [
        # Filter out device kernel launches.
        tir.transform.Filter(
            lambda f: int(f.attrs.get("calling_conv", CallConv.kDefault))
            != int(CallConv.kDeviceKernelLaunch)
        ),
        tir.transform.BindTarget(target_host),
        tir.transform.LowerTVMBuiltin(),
        tir.transform.LowerCustomDatatypes(),
        tir.transform.LowerIntrin(),
        tir.transform.LowerDeviceStorageAccessInfo(),
        tir.transform.CombineContextCall(),
    ]
    return tvm.ir.transform.Sequential(host_pass_list)


def device_module_pass_manager(target: Target) -> tvm.ir.transform.Sequential:
    """
    Build a sequential pass pipeline for lowering the device part of a mixed module.

    Parameters
    ----------
    target : Target
        The target for device-specific transformations.

    Returns
    -------
    tvm.ir.transform.Sequential
        A sequential pass pipeline for device-specific transformations.
    """
    device_pass_list = [
        # Select only device kernel launches.
        tir.transform.Filter(
            lambda f: int(f.attrs.get("calling_conv", CallConv.kDefault))
            == int(CallConv.kDeviceKernelLaunch)
        ),
        tir.transform.BindTarget(target),
        tir.transform.LowerWarpMemory(),
        tir.transform.Simplify(),
        tir.transform.LowerCustomDatatypes(),
        tir.transform.LowerDeviceStorageAccessInfo(),
        tir.transform.LowerIntrin(),
    ]
    return tvm.ir.transform.Sequential(device_pass_list)


def split_mixed_module(
    mod_mixed: IRModule, target_arg: Target, target_host_arg: Target
) -> Tuple[IRModule, IRModule]:
    """
    Split a mixed module containing both device and host parts into separate modules,
    applying appropriate transformations on each.

    Parameters
    ----------
    mod_mixed : IRModule
        The input module containing both device and host code.
    target_arg : Target
        The target for device-specific transformations.
    target_host_arg : Target
        The host target for lowering.

    Returns
    -------
    Tuple[IRModule, IRModule]
        (host module, device module)
    """
    target, target_host = target_arg, target_host_arg
    if getattr(target, "host", None) is None:
        target.host = target_host
    if mod_mixed is None:
        raise ValueError("Module must be defined")

    mod_mixed = mixed_module_pass_manager(target)(mod_mixed)
    host_mod = host_module_pass_manager(target_host)(mod_mixed)
    device_mod = device_module_pass_manager(target)(mod_mixed)

    # Warn if target is GPU but no device code was generated.
    if "gpu" in target.keys and len(device_mod.functions) == 0:
        print(
            f"Warning: Specified target {target} but cannot find device code. "
            "Did you forget to bind?"
        )

    return host_mod, device_mod


def default_target_host(target: Target) -> Target:
    """
    Determine the default target host for a given target.
    """
    if target is not None and target.device_type == Device.kDLCPU:
        return target
    # In practice, llvm_enabled should be determined dynamically.
    llvm_enabled = True
    return Target("llvm") if llvm_enabled else Target("stackvm")


def codegen_build(mod: IRModule, target: Target) -> tvm.runtime.Module:
    """
    Build a runtime module from an IRModule and a Target.

    If the "tir.disable_assert" flag is set in the pass context,
    the SkipAssert transformation is applied.

    Parameters
    ----------
    mod : IRModule
        The input IRModule.
    target : Target
        The target for which to build the module.

    Returns
    -------
    tvm.runtime.Module
        The built runtime module.
    """
    if tvm.ir.transform.PassContext.current().config.get("tir.disable_assert", False):
        mod = tvm.tir.transform.SkipAssert()(mod)
    build_f_name = "target.build." + target.kind.name
    bf = tvm.get_global_func(build_f_name)
    if bf is None:
        raise ValueError(f"{build_f_name} is not enabled")
    return bf(mod, target)


def tir_to_runtime(inputs: Dict[Target, Tuple[IRModule, IRModule]], target_host: Target):
    """
    Convert a collection of TIR IRModules (keyed by Target) into a single runtime Module.

    Parameters
    ----------
    inputs : dict
        Mapping from Target to Tuple[IRModule, IRModule].
    target_host : Target
        The initial host target.

    Returns
    -------
    tvm.runtime.Module
        The final runtime module.
    """

    # Get the first module to get the attributes
    # necessary for tests/python/codegen/test_target_codegen_blob.py::test_cuda_multi_lib
    first_module = next(iter(inputs.values()))[0]
    mhost_all = ir.IRModule({}, attrs=first_module.attrs)

    device_modules = []
    for tgt, (host_mod, device_mod) in inputs.items():
        overrides_host_target = tgt.get_target_device_type() == target_host.get_target_device_type()
        non_host_target_kind = tgt.kind != target_host.kind
        if overrides_host_target and non_host_target_kind:
            device_modules.append(codegen_build(host_mod, tgt))
        else:
            mhost_all.update(host_mod)
        if len(device_mod.functions) != 0:
            device_modules.append(codegen_build(device_mod, tgt))

    mhost = codegen_build(mhost_all, target_host)
    for dev_mod in device_modules:
        if dev_mod is not None:
            mhost.import_module(dev_mod)
    return mhost


def build(
    inputs: Union[PrimFunc, IRModule],
    target: Optional[Union[str, Target]] = None,
    name: str = "main",
):
    """
    Build a function with a signature, generating code for devices
    coupled with target information.

    Parameters
    ----------
    inputs : Union[PrimFunc, IRModule]
        The input to be built.
    target : Optional[Union[str, Target]]
        The target for compilation.
    name : str
        The name of the result function.

    Returns
    -------
    tvm.runtime.Module
        A module combining both host and device code.
    """
    # Convert PrimFunc to IRModule
    pass_ctx = tvm.ir.transform.PassContext.current()
    if isinstance(inputs, PrimFunc):
        f = inputs.with_attr("global_symbol", name)
        if pass_ctx.config.get("tir.noalias", True):
            f = f.with_attr("tir.noalias", True)
        input_mod = tvm.ir.IRModule({tvm.ir.GlobalVar(name): f})
    elif isinstance(inputs, tvm.IRModule):
        input_mod = inputs
    else:
        raise ValueError("Inputs must be IRModule or PrimFunc")

    # Get target and target_host
    target = Target.current() if target is None else target
    if target is None and isinstance(input_mod, tvm.IRModule):
        target_mod = {}
        for gvar, func in input_mod.functions.items():
            tgt = func.attrs.get("target", "llvm")
            target_mod.setdefault(tgt, {})[gvar] = func
        target_input_mod = {
            tgt: tvm.IRModule(funcs).with_attrs(input_mod.attrs)
            for tgt, funcs in target_mod.items()
        }
    else:
        target_input_mod = {target: input_mod}

    annotated_mods = {}
    for tgt, mod in target_input_mod.items():
        if not isinstance(tgt, (str, Target)):
            raise ValueError("The key of inputs must be str or Target.")
        if not isinstance(mod, tvm.IRModule):
            raise ValueError("inputs must be IRModule, or dict of str to IRModule.")
        annotated_mods[tgt] = mod

    annotated_mods, target_host = Target.canon_target_map_and_host(annotated_mods)
    if not target_host:
        for tar, mod in annotated_mods.items():
            if ndarray.device(tar.kind.name, 0).device_type == ndarray.cpu(0).device_type:
                target_host = tar
                break
    if not target_host:
        target_host = "llvm" if tvm.runtime.enabled("llvm") else "stackvm"
    annotated_mods, target_host = Target.canon_target_map_and_host(annotated_mods, target_host)

    assert annotated_mods is not None and target_host is not None
    check_and_update_host_consistency(annotated_mods, target_host)

    # Lower the module
    for tgt, mod in annotated_mods.items():
        mod = lower_module(mod, simple_mode=False)
        host_mod, device_mod = split_mixed_module(mod, tgt, target_host)
        annotated_mods[tgt] = (host_mod, device_mod)

    # Convert TIR IRModules to runtime Module by calling target.build
    return tir_to_runtime(annotated_mods, target_host)


tvm.register_func("tir.build", build)
