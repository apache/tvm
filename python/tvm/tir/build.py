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


def tir_to_runtime(host_mod: IRModule, device_mod: IRModule, target, target_host: Target):
    """
    Convert a collection of TIR IRModules (keyed by Target) into a single runtime Module.

    Parameters
    ----------
    host_mod : IRModule
        The host module.
    device_mod : IRModule
        The device module.
    target : Target
        The target.
    target_host : Target
        The initial host target.

    Returns
    -------
    tvm.runtime.Module
        The final runtime module.
    """

    # Get the first module to get the attributes
    # necessary for tests/python/codegen/test_target_codegen_blob.py::test_cuda_multi_lib
    mhost_all = ir.IRModule({}, attrs=host_mod.attrs)

    device_modules = []
    overrides_host_target = target.get_target_device_type() == target_host.get_target_device_type()
    non_host_target_kind = target.kind != target_host.kind
    if overrides_host_target and non_host_target_kind:
        device_modules.append(codegen_build(host_mod, target))
    else:
        mhost_all.update(host_mod)
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
    pipeline: Union[None, str, tvm.transform.Pass] = "default_tir",
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
    target = Target.current() if target is None else target
    if target is None:
        target = "llvm"
    assert target is not None
    target = Target.canon_target(target)

    # Step 1: Determine the host
    target_host = "llvm" if tvm.runtime.enabled("llvm") else "stackvm"
    if target is not None:
        if target.host is not None:
            target_host = target.host
        elif ndarray.device(target.kind.name, 0).device_type == ndarray.cpu(0).device_type:
            target_host = target
    else:
        for func in mod.functions.values():
            f_target = func.attrs.get("target", None)
            if f_target is not None and f_target.host is not None:
                target_host = f_target.host
    assert target_host is not None
    target_host = Target.canon_target(target_host)
    target = target.with_host(target_host)

    # Step 2: Bind the target to the input module
    mod = tvm.tir.transform.BindTarget(target)(mod)
    # Step 3: Apply the pipeline
    if pipeline is not None:
        if isinstance(pipeline, str):
            pipeline = tvm.tir.get_pipeline(pipeline)
        mod = pipeline(mod)

    # Step 4: Finalize the host and device modules
    host_mod = tvm.tir.pipeline.finalize_host_passes()(mod)
    device_mod = tvm.tir.pipeline.finalize_device_passes()(mod)

    # Convert TIR IRModules to runtime Module by calling target.build
    return tir_to_runtime(host_mod, device_mod, target, target_host)


tvm.register_func("tir.build", build)
