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
from typing import Union, Optional, Dict
import enum

import tvm
from tvm import ir
from tvm.runtime import ndarray
from tvm.tir import PrimFunc
from tvm.ir.module import IRModule
from tvm.target import Target


def split_host_device_mods(mod):
    """Split an IRModule into host and device modules.

    Parameters
    ----------
    mod : tvm.IRModule
        The input module to split

    Returns
    -------
    host_mod : tvm.IRModule
        The module containing host functions
    device_mod_dict : Dict[Target, tvm.IRModule]
        A dict mapping targets to device modules
    """

    class CallConv(enum.IntEnum):
        """Enum representing different calling conventions.
        Corresponds to the C++ tvm::ir::CallingConv enum.
        """

        kDefault = 0
        kCPackedFunc = 1
        kDeviceKernelLaunch = 2

    host_mod = tvm.tir.transform.Filter(
        lambda f: int(f.attrs.get("calling_conv", CallConv.kDefault))
        != int(CallConv.kDeviceKernelLaunch)
    )(mod)
    device_mod = tvm.tir.transform.Filter(
        lambda f: int(f.attrs.get("calling_conv", CallConv.kDefault))
        == int(CallConv.kDeviceKernelLaunch)
    )(mod)
    device_mod_dict = {}
    for gv, func in device_mod.functions.items():
        device_mod_dict.setdefault(func.attrs.get("target", None), dict()).update({gv: func})
    for target, funcs in device_mod_dict.items():
        device_mod_dict[target] = tvm.IRModule(funcs, attrs=device_mod.attrs)
    return host_mod, device_mod_dict


def codegen_build(mod: IRModule, target: Target) -> tvm.runtime.Module:
    """Build a runtime module from an IRModule and a Target."""
    if tvm.ir.transform.PassContext.current().config.get("tir.disable_assert", False):
        mod = tvm.tir.transform.SkipAssert()(mod)
    build_f_name = "target.build." + target.kind.name
    bf = tvm.get_global_func(build_f_name)
    if bf is None:
        raise ValueError(f"{build_f_name} is not enabled")
    return bf(mod, target)


def tir_to_runtime(
    host_mod: IRModule, device_mod_dict: Dict[Target, IRModule], target_host: Target
):
    """Convert a collection of TIR IRModules (keyed by Target) into a single runtime Module."""

    # Get the first module to get the attributes
    # necessary for tests/python/codegen/test_target_codegen_blob.py::test_cuda_multi_lib
    mhost_all = ir.IRModule({}, attrs=host_mod.attrs)

    mhost_all.update(host_mod)
    device_modules = []
    for target, device_mod in device_mod_dict.items():
        if len(device_mod.functions) != 0:
            device_modules.append(codegen_build(device_mod, target))

    mhost = codegen_build(mhost_all, target_host)
    for dev_mod in device_modules:
        if dev_mod is not None:
            mhost.import_module(dev_mod)
    return mhost


def build(
    mod: Union[PrimFunc, IRModule],
    target: Optional[Union[str, Target]] = None,
    pipeline: Union[None, str, tvm.transform.Pass] = "default",
):
    """Build a function with a signature, generating code for devices
    coupled with target information.

    Parameters
    ----------
    mod : Union[PrimFunc, IRModule]
        The input to be built.
    target : Optional[Union[str, Target]]
        The target for compilation.
    pipeline : Union[None, str, tvm.transform.Pass]
        The pipeline to use for compilation.

    Returns
    -------
    tvm.runtime.Module
        A module combining both host and device code.
    """
    # Convert PrimFunc to IRModule
    if isinstance(mod, PrimFunc):
        mod = tvm.IRModule.from_expr(mod)
    else:
        assert isinstance(mod, tvm.IRModule)

    # Step 0: Determine the target in environment
    # It's used to bind the PrimFunc without target attr to serve as a default target
    target_to_bind = Target.current() if target is None else target
    if target_to_bind is None:
        target_to_bind = "llvm"
    assert target_to_bind is not None
    target_to_bind = Target.canon_target(target_to_bind)

    # Step 1: Determine the target to search for tir pipeline
    target = Target.current() if target is None else target
    if target is None:
        for func in mod.functions.values():
            f_target = func.attrs.get("target", None)
            if f_target is not None:
                target = f_target
                break
    if target is not None:
        target = Target.canon_target(target)

    # Step 2: Determine the host target
    target_host = "llvm" if tvm.runtime.enabled("llvm") else "stackvm"
    if target is not None:
        if target.host is not None:
            target_host = target.host
        elif ndarray.device(target.kind.name, 0).device_type == ndarray.cpu(0).device_type:
            target_host = target
    target_host = Target.canon_target(target_host)
    target_to_bind = target_to_bind.with_host(target_host)

    # Step 3: Bind the target to the input module
    mod = tvm.tir.transform.BindTarget(target_to_bind)(mod)

    # Step 4: Apply the tir  pipeline
    if pipeline is not None:
        # custom pipeline
        if isinstance(pipeline, str):
            pipeline = tvm.tir.get_tir_pipeline(pipeline)
    else:
        # default pipeline depends on the target
        pipeline = tvm.tir.get_default_tir_pipeline(target)
    mod = pipeline(mod)

    # Step 5: Get host and device modules
    host_mod, device_mod_dict = split_host_device_mods(mod)

    # Step 6: Apply finalization passes
    host_mod = tvm.tir.pipeline.finalize_host_passes()(host_mod)
    device_mod_dict = {
        target: tvm.tir.pipeline.finalize_device_passes()(device_mod)
        for target, device_mod in device_mod_dict.items()
    }

    # Convert TIR IRModules to runtime Module by calling target.build
    return tir_to_runtime(host_mod, device_mod_dict, target_host)


tvm.register_func("tir.build", build)
