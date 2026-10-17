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
"""Native TIRx compilation and target code generation."""

import tvm_ffi
from tvm_ffi import DLDeviceType

import tvm_tirx
from tvm_tirx import ir
from tvm_tirx.codegen import Executable
from tvm_tirx.ir.module import IRModule
from tvm_tirx.target import Target
from tvm_tirx.tirx import Function


def _split_host_device_modules(
    mod: IRModule,
) -> tuple[IRModule, dict[tuple[Target, str], IRModule]]:
    """Split an IRModule into host and device modules.

    This function takes an IRModule containing functions with different target attributes
    and separates them into host (CPU) and device (GPU/accelerator) modules. Functions
    are categorized based on their target attribute in func_attr.

    Parameters
    ----------
    mod : tvm_tirx.IRModule
        The input module to split.
        The module should contain functions with target attributes in their func_attr.
        Functions with "cpu" in their target string are considered host functions,
        while others are considered device functions.

    Returns
    -------
    host_mod : tvm_tirx.IRModule
        The module containing host functions (CPU-targeted functions)
    device_mod_dict : Dict[Tuple[Target, str], tvm_tirx.IRModule]
        A dict mapping (target, serialized compile configuration) to device modules.
        Functions in each module share both target and compiler settings.

        Examples
    --------
    Given an IRModule with the following functions:

    .. code-block:: python

        @I.ir_module
        class Module:
            @T.function(private=True)
            def add(a: T.int32, b: T.int32) -> T.int32:
                T.func_attr({"target": T.target({"arch": "sm_90", "keys": ["cuda", "gpu"],
                                                "kind": "cuda", "max_num_threads": 1024}))
                return a + b

            @T.function(private=True)
            def add_host(a: T.int32, b: T.int32) -> T.int32:
                T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "c"}))
                return a + b

            @T.function
            def main_kernel(A: T.handle, B: T.handle, C: T.handle, length: T.int32):
                T.func_attr({"target": T.target({"arch": "sm_90", "keys": ["cuda", "gpu"],
                                                "kind": "cuda"}),
                            "calling_conv": 2,  # kDeviceKernelLaunch for device kernels
                            "tirx.is_global_func": True})
                # ... kernel implementation

            @T.function
            def main(self_handle: T.handle, args: T.handle, num_args: T.int32, result: T.handle):
                T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "c"}),
                            "calling_conv": 1,  # kCPackedFunc for entry functions
                            "tirx.is_entry_func": True})
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

    def is_host_function(f):
        target = f.attrs.get("target", tvm_tirx.target.Target("llvm"))
        return target.kind.name in ["llvm", "c", "cuda_host"]

    host_mod = tvm_tirx.tirx.transform.Filter(is_host_function)(mod)
    device_mod = tvm_tirx.tirx.transform.Filter(lambda f: not is_host_function(f))(mod)
    # Options participate in grouping even when two kernels share a Target.
    groups = {}
    for gv, func in device_mod.functions.items():
        target = func.attrs["target"]
        config = func.attrs.get("backend_config", "")
        key = (str(target), config)
        if key not in groups:
            groups[key] = (target, {})
        groups[key][1][gv] = func
    device_mod_dict = {}
    for (_, config), (target, functions) in groups.items():
        group = tvm_tirx.IRModule(
            functions, attrs=device_mod.attrs, global_infos=device_mod.global_infos
        )
        if config:
            group = group.with_attr("backend_config", config)
        device_mod_dict[(target, config)] = group
    return host_mod, device_mod_dict


def _build_for_target(mod: IRModule, target: Target) -> tvm_ffi.Module:
    """Build a runtime module from an IRModule and a Target."""
    if tvm_tirx.ir.transform.PassContext.current().config.get("tirx.disable_assert", False):
        mod = tvm_tirx.tirx.transform.SkipAssert()(mod)
    build_f_name = "target.build." + target.kind.name
    bf = tvm_ffi.get_global_func(build_f_name)
    if bf is None:
        raise ValueError(f"{build_f_name} is not enabled")
    return bf(mod, target)


def _codegen_host_and_device(
    host_mod: IRModule, device_mod_dict: dict[tuple[Target, str], IRModule], target_host: Target
) -> tvm_ffi.Module:
    """Compile each device group and attach it to the host runtime module."""

    # Get the first module to get the attributes
    # necessary for tirx/tests/python/backend/cuda/test_module_export.py::test_cuda_multi_lib
    mhost_all = ir.IRModule({}, attrs=host_mod.attrs)

    mhost_all.update(host_mod)
    device_modules = []
    for (target, _config), device_mod in device_mod_dict.items():
        if len(device_mod.functions) != 0:
            device_modules.append(_build_for_target(device_mod, target))

    mhost = _build_for_target(mhost_all, target_host)
    for dev_mod in device_modules:
        if dev_mod is not None:
            mhost.import_module(dev_mod)
    for external_mod in host_mod.get_attr("external_mods") or []:
        mhost.import_module(external_mod)
    return mhost


def compile(  # pylint: disable=redefined-builtin
    mod: Function | IRModule,
    target: str | Target | None = None,
    pipeline: str | tvm_tirx.ir.transform.Pass | None = "default",
) -> Executable:
    """Compile a function or module to an executable for the target.

    Parameters
    ----------
    mod : Union[Function, IRModule]
        The input to be compiled.
    target : Optional[Union[str, Target]]
        The target for compilation, including backend configuration defaults.
    pipeline : Union[None, str, tvm_tirx.ir.transform.Pass]
        The pipeline to use for compilation.

    Returns
    -------
    tvm_tirx.codegen.Executable
        An executable combining both host and device code.
    """
    # Convert Function to IRModule
    if isinstance(mod, Function):
        mod = tvm_tirx.IRModule.from_expr(mod)
    else:
        assert isinstance(mod, tvm_tirx.IRModule)

    from tvm_tirx.backend.config import prepare_target

    target = prepare_target(target, mod)

    # Step 0: Determine the target in environment
    # It's used to bind the Function without target attr to serve as a default target
    target_to_bind = Target.current() if target is None else target
    if target_to_bind is None:
        target_to_bind = "llvm"
    assert target_to_bind is not None
    target_to_bind = Target(target_to_bind)

    # Step 1: Determine the target to search for tirx pipeline
    target = Target.current() if target is None else target
    if target is None:
        for func in mod.functions.values():
            f_target = func.attrs.get("target", None)
            if f_target is not None:
                target = f_target
                break
    if target is not None:
        target = Target(target)

    # Step 2: Determine the host target
    target_host = "llvm" if tvm_tirx.runtime.enabled("llvm") else "c"
    if target is not None:
        if target.host is not None:
            target_host = target.host
        elif target.get_target_device_type() == DLDeviceType.kDLCPU:
            target_host = target
    target_host = Target(target_host)
    target_to_bind = target_to_bind.with_host(target_host)

    # Step 3: Select a pipeline that binds targets after lowering thread launches.
    # Classification and helper specialization require the complete mixed module.
    if pipeline not in (None, "default"):
        # custom pipeline
        assert isinstance(pipeline, str)
        pipeline, finalize_host_passes, finalize_device_passes = tvm_tirx.tirx.get_tir_pipeline(
            pipeline, target=target_to_bind
        )
    else:
        # default pipeline depends on the target
        pipeline, finalize_host_passes, finalize_device_passes = (
            tvm_tirx.tirx.get_default_tir_pipeline(target, mod, target_to_bind=target_to_bind)
        )
    mod = pipeline(mod)

    # Step 4: Get host and device modules
    host_mod, device_mod_dict = _split_host_device_modules(mod)

    # Step 5: Apply finalization passes
    host_mod = finalize_host_passes()(host_mod)
    device_mod_dict = {
        target: finalize_device_passes()(device_mod)
        for target, device_mod in device_mod_dict.items()
    }

    # Convert TIR IRModules to runtime Module by calling target.build
    return Executable(_codegen_host_and_device(host_mod, device_mod_dict, target_host))


# C++ compiler consumers link the generated runtime module directly.
tvm_ffi.register_global_func("tirx.compile", lambda *args, **kwargs: compile(*args, **kwargs).mod)
