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
"""The compiler for TL programs."""

import os
import os.path as osp
from typing import Literal
import tvm
from tvm import tir, tl, relay
from tvm.contrib import nvcc, hipcc, rocm
from typing import Literal
try:
    from tvm.tl.code_replace import replace_code
except ImportError:
    def replace_code(code):
        return code

def is_device_call(func: tir.PrimFunc):
    return bool(func.attrs and "calling_conv" in func.attrs and func.attrs["calling_conv"] == 2)


def is_host_call(func: tir.PrimFunc):
    return not is_device_call(func)


@tvm.register_func("tvm_callback_cuda_compile", override=True)
def tvm_callback_cuda_compile(code, target):
    tvm_root = osp.join(osp.dirname(__file__), "../../..")
    tl_template_path = osp.abspath(osp.join(tvm_root, "src/tl"))
    # TODO(lei): actually this indeed should be renamed into
    # TL_CUTLASS_INCLUDE_PATH
    if "TL_CUTLASS_PATH" in os.environ:
        cutlass_path = os.environ["TL_CUTLASS_PATH"]
    else:
        cutlass_path = osp.abspath(osp.join(tvm_root, "3rdparty/cutlass/include"))
    compute_version = "".join(nvcc.get_target_compute_version(target).split("."))

    # special handle for Hopper
    if compute_version == "90":
        arch = [f"-arch=sm_90a"]
        format = "cubin"
    else:
        arch = [f"-arch=sm_{compute_version}"]
        format = "cubin"

    # printing out number of registers
    debug_option = "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage"
    ptx = nvcc.compile_cuda(
        code,
        format,
        arch,
        options=[
            "-std=c++17",
            debug_option,
            "--use_fast_math",
            "-I" + tl_template_path,
            "-I" + cutlass_path,
        ],
        verbose=False,
    )

    return ptx

@tvm.register_func("tvm_callback_hip_compile", override=True)
def tvm_callback_hip_compile(code, target):
    tvm_root = osp.join(osp.dirname(__file__), "../../..")
    tl_template_path = osp.abspath(osp.join(tvm_root, "src/tl"))

    # TODO(lei): actually this indeed should be renamed into
    # TL_COMPOSABLE_KERNEL_INCLUDE_PATH
    if "TL_COMPOSABLE_KERNEL_PATH" in os.environ:
        ck_path = os.environ["TL_COMPOSABLE_KERNEL_PATH"]
    else:
        ck_path = osp.abspath(osp.join(tvm_root, "3rdparty/composable_kernel/include"))

    hsaco = hipcc.compile_hip(
        code,
        target_format="hsaco",
        options=[
            "-std=c++17",
            "-I" + tl_template_path,
            "-I" + ck_path,
            # "-I" + osp.join(ck_path, "../build/include/"),
        ],
        verbose=False,
    )

    return hsaco

def extrac_params(func: tir.PrimFunc):
    buffers = [func.buffer_map[var] for var in func.params]
    tensor_types = [relay.TensorType(buffer.shape, buffer.dtype) for buffer in buffers]
    return tensor_types

def detect_target(target: str = "auto") -> str:
    """Detect the computing target (CUDA or ROCm) based on the environment.

    Args:
        target (str): The target to detect. Use "auto" for automatic detection.
                      Can also specify "cuda" or "hip" explicitly.

    Returns:
        str: The detected target, either "cuda" or "hip".

    Raises:
        ValueError: If auto-detection is enabled and no valid target is found.
    """

    def is_cuda_available() -> bool:
        """Check if CUDA is available."""
        try:
            nvcc.find_cuda_path()
            return True
        except RuntimeError:
            return False

    def is_rocm_available() -> bool:
        """Check if ROCm is available."""
        try:
            rocm.find_rocm_path()
            return True
        except RuntimeError:
            return False

    if target == "auto":
        if is_cuda_available():
            return "cuda"
        if is_rocm_available():
            return "hip"
        raise ValueError("Cannot detect the target: no CUDA or ROCm installation found.")

    if target not in {"cuda", "hip"}:
        raise ValueError(f"Invalid target: {target}. Must be 'cuda', 'hip', or 'auto'.")
    
    return target

# TODO(lei): Should enhance to support IRModule with multiple functions
def lower(func, target: Literal["auto", "cuda", "hip"]="auto", target_host="llvm", runtime_only=False):
    # TODO(lei): Append C Source code host generation to the runtime
    params = extrac_params(func) if not runtime_only else None
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    
    target = detect_target(target)
    target_host = tvm.target.Target.canon_target(target_host)
    if target == "auto":
        is_cuda_available = False
        is_hip_available = False
        try:
            nvcc.find_cuda_path()
            is_cuda_available = True
        except:
            is_cuda_available = False
        
        try:
            rocm.find_rocm_path()
            is_hip_available = True
        except:
            is_hip_available = False

        if is_cuda_available:
            target = "cuda"
        elif is_hip_available:
            target = "hip"
        else:
            raise ValueError("No CUDA or HIP available")
    else:
        assert target in ["cuda", "hip"], f"Target {target} is not supported"

    target = tvm.target.Target(target, target_host)

    mod = tir.transform.BindTarget(target)(mod)

    mod = tl.transform.FrontendLegalize()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tl.transform.LayoutInference()(mod)
    mod = tl.transform.LowerTileOp()(mod)
    mod = tir.transform.Simplify()(mod)

    if target.arch == "sm_90":
        mod = tl.transform.MultiVersionBuffer()(mod)
        mod = tl.transform.WarpSpecialized()(mod)
        mod = tl.transform.InjectSoftwarePipeline()(mod)
        mod = tir.transform.LowerOpaqueBlock()(mod)
        # mod = tl.transform.WarpSpecializedPipeline()(mod)
        mod = tl.transform.InjectFenceProxy()(mod)
    else:
        mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        mod = tl.transform.PipelinePlanning()(mod)
        mod = tl.transform.InjectSoftwarePipeline()(mod)

    mod = tir.transform.LowerOpaqueBlock()(mod)
    mod = tir.transform.FlattenBuffer()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tir.transform.Simplify()(mod)

    mod = tir.transform.VectorizeLoop()(mod)
    mod = tir.transform.StorageRewrite()(mod)
    mod = tir.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tir.transform.RewriteUnsafeSelect()(mod)
    mod = tir.transform.HoistIfThenElse()(mod)

    mod = tir.transform.VerifyMemory()(mod)
    mod = tir.transform.AnnotateEntryFunc()(mod)
    # TODO(lei): This is a hack to make sure the
    # thread level allreduce pass can be applied
    # in TL. As Tl ony use one thread dimension
    # the var binding information will be lost
    # in the lowering process with Legalization
    # and Simplify pass.
    # We can find a way better to create var instead
    # of putting the LowerThreadAllreduce before
    # the Legalization.
    mod = tir.transform.ThreadPartialSync("shared.dyn")(mod)
    mod = tir.transform.LowerThreadAllreduce()(mod)
    mod = tl.transform.LowerHopperIntrin()(mod)
    mod = tir.transform.InjectPTXAsyncCopy()(mod)

    mod = tir.transform.AnnotateDeviceRegions()(mod)
    mod = tir.transform.SplitHostDevice()(mod)
    mod = tir.transform.MergeSharedMemoryAllocations()(mod)
    mod = tir.transform.ThreadSync("shared")(mod)
    mod = tir.transform.ThreadSync("shared.dyn")(mod)

    mod = tir.transform.MakePackedAPI()(mod)
    mod = tir.transform.LowerDeviceKernelLaunch()(mod)
    host_mod = tir.transform.Filter(is_host_call)(mod)
    host_mod = tir.transform.BindTarget(target_host)(host_mod)
    host_mod = tir.transform.FP8StorageLegalize()(host_mod)
    host_mod = tir.transform.BF16StorageLegalize()(host_mod)
    host_mod = tir.transform.LowerTVMBuiltin()(host_mod)
    host_mod = tir.transform.LowerCustomDatatypes()(host_mod)
    host_mod = tir.transform.LowerIntrin()(host_mod)
    host_mod = tir.transform.LowerDeviceStorageAccessInfo()(host_mod)
    host_mod = tir.transform.CombineContextCall()(host_mod)

    if target_host.kind.name == "llvm":
        host_mod = tvm._ffi.get_global_func("target.build.llvm")(host_mod, target_host)
    else:
        raise ValueError("Target host is not supported")

    device_mod = tir.transform.Filter(is_device_call)(mod)
    device_mod = tir.transform.LowerDeviceStorageAccessInfo()(device_mod)
    device_mod = tir.transform.LowerIntrin()(device_mod)
    device_mod = tir.transform.Simplify()(device_mod)
    
    if target.kind.name == "cuda":
        # Debug to get the code
        # code = tvm._ffi.get_global_func("target.build.tl_debug_codegen")(device_mod, target)
        device_mod = tvm._ffi.get_global_func("target.build.tilelang_cuda")(device_mod, target)
    elif target.kind.name == "hip":
        device_mod = tvm._ffi.get_global_func("target.build.tilelang_hip")(device_mod, target)
    else:
        raise ValueError("Target is not supported")

    host_mod.import_module(device_mod)

    if runtime_only is True:
        return host_mod
    else:
        return host_mod, params
