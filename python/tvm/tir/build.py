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
from typing import Dict, Optional, Tuple, Union

import tvm
from tvm import ir
from tvm.ir.module import IRModule
from tvm.target import Target
from tvm.tir import PrimFunc


def split_host_device_mods(mod: IRModule) -> Tuple[IRModule, Dict[Target, IRModule]]:
    """Split an IRModule into host and device modules.

    This function takes an IRModule containing functions with different target attributes
    and separates them into host (CPU) and device (GPU/accelerator) modules. Functions
    are categorized based on their target attribute in func_attr.

    Parameters
    ----------
    mod : tvm.IRModule
        The input module to split.
        The module should contain functions with target attributes in their func_attr.
        Functions with "cpu" in their target string are considered host functions,
        while others are considered device functions.

    Returns
    -------
    host_mod : tvm.IRModule
        The module containing host functions (CPU-targeted functions)
    device_mod_dict : Dict[Target, tvm.IRModule]
        A dict mapping targets to device modules. Each device module contains
        functions targeting the same device (e.g., CUDA GPU, OpenCL, etc.)

        Examples
    --------
    Given an IRModule with the following functions:

    .. code-block:: python

        @I.ir_module
        class Module:
            @T.prim_func(private=True)
            def add(a: T.int32, b: T.int32) -> T.int32:
                T.func_attr({"target": T.target({"arch": "sm_90", "keys": ["cuda", "gpu"],
                                                "kind": "cuda", "max_num_threads": 1024}))
                return a + b

            @T.prim_func(private=True)
            def add_host(a: T.int32, b: T.int32) -> T.int32:
                T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "c"}))
                return a + b

            @T.prim_func
            def main_kernel(A: T.handle, B: T.handle, C: T.handle, length: T.int32):
                T.func_attr({"target": T.target({"arch": "sm_90", "keys": ["cuda", "gpu"],
                                                "kind": "cuda"}),
                            "calling_conv": 2,  # kDeviceKernelLaunch for device kernels
                            "tir.is_global_func": True})
                # ... kernel implementation

            @T.prim_func
            def main(self_handle: T.handle, args: T.handle, num_args: T.int32, result: T.handle):
                T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "c"}),
                            "calling_conv": 1,  # kCPackedFunc for entry functions
                            "tir.is_entry_func": True})
                # ... main function implementation

    The function will return:
    - host_mod: Contains `add_host` and `main` functions (CPU targets)
    - device_mod_dict: Contains a CUDA module with `add` and `main_kernel` functions

    Notes
    -----
    - Functions are categorized based on string matching of their target attribute
    - Functions with "cpu" in the target string are considered host functions
    - Device functions are grouped by their target to create separate modules
    - The function uses string-based target matching due to target hash limitations
    - All functions must have a `calling_conv` attribute in their func_attr:
        - Private helper functions (private=True): use `calling_conv: 0` (kDefault, by default)
        - Public entry functions: use `calling_conv: 1` (kCPackedFunc)
        - Device kernel functions: use `calling_conv: 2` (kDeviceKernelLaunch)
    """

    def is_host_func(f):
        target = f.attrs.get("target", tvm.target.Target("llvm"))
        return str(target.kind) in ["llvm", "c"]

    host_mod = tvm.tir.transform.Filter(is_host_func)(mod)
    device_mod = tvm.tir.transform.Filter(lambda f: not is_host_func(f))(mod)
    # TODO(syfeng): Here we use str as key since target hash is not correct
    target_str2target = {}
    device_func_dict = {}
    device_mod_dict: Dict[Target, IRModule] = {}
    for gv, func in device_mod.functions.items():
        target = func.attrs.get("target", None)
        target_str = str(target) if target is not None else ""
        target_str2target[target_str] = target  # This might be overridden by the last one
        device_func_dict.setdefault(target_str, dict()).update({gv: func})
    for target_str in target_str2target.keys():
        target = target_str2target[target_str]
        device_mod_dict[target] = tvm.IRModule(device_func_dict[target_str], attrs=device_mod.attrs)
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
    target_host = "llvm" if tvm.runtime.enabled("llvm") else "c"
    if target is not None:
        if target.host is not None:
            target_host = target.host
        elif (
            tvm.device(target.kind.name, 0).dlpack_device_type() == tvm.cpu(0).dlpack_device_type()
        ):
            target_host = target
    target_host = Target.canon_target(target_host)
    target_to_bind = target_to_bind.with_host(target_host)

    # Step 3: Bind the target to the input module
    mod = tvm.tir.transform.BindTarget(target_to_bind)(mod)

    # Step 4: Apply the tir pipeline
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


tvm.register_global_func("tir.build", build)
