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
"""Driver for partitioning and building a Relay module for CUTLASS offload."""
import logging
import os
import multiprocessing
import tvm
from tvm import runtime, relay
from tvm.contrib.nvcc import find_cuda_path, get_cuda_version
from .gen_gemm import CutlassGemmProfiler
from .gen_conv2d import CutlassConv2DProfiler

logger = logging.getLogger("cutlass")


def _get_cutlass_path():
    tvm_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../../")
    cutlass_path = os.path.join(tvm_root, "3rdparty/cutlass")
    assert os.path.exists(
        cutlass_path
    ), """The CUTLASS root directory not found in {}.
        Currently, using CUTLASS requires building TVM from source.""".format(
        cutlass_path
    )
    return cutlass_path


def _get_cutlass_compile_options(sm, threads, use_fast_math=False):
    cutlass_root = _get_cutlass_path()
    cutlass_include = os.path.join(cutlass_root, "include")
    cutlass_util_include = os.path.join(cutlass_root, "tools/util/include")

    kwargs = {}
    kwargs["cc"] = "nvcc"
    kwargs["options"] = [
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-gencode=arch=compute_%d,code=[sm_%d,compute_%d]" % (sm, sm, sm),
        "-Xcompiler=-fPIC",
        "-Xcompiler=-Wconversion",
        "-Xcompiler=-fno-strict-aliasing",
        "-O3",
        "-std=c++14",
        "-I" + cutlass_include,
        "-I" + cutlass_util_include,
    ]
    if use_fast_math:
        kwargs["options"].append("-DCUTLASS_USE_TANH_FOR_SIGMOID")
    cuda_path = find_cuda_path()
    cuda_ver = get_cuda_version(cuda_path)
    if cuda_ver >= 11.2:
        ncpu = multiprocessing.cpu_count() if threads < 0 else threads
        kwargs["options"].append("-t %d" % ncpu)
    return kwargs


class OpAnnotator(tvm.relay.ExprVisitor):
    """Annotates partitioned functions with shape and dtype information."""

    def __init__(self):
        super().__init__()
        self.signature = {}

    def visit_call(self, call):
        op = call.op
        if isinstance(op, relay.Function) and "PartitionedFromPattern" in op.attrs:
            self.signature["op_type"] = op.attrs["Composite"]
            for i, arg in enumerate(op.params):
                self.signature["arg%d_shape" % i] = arg.checked_type.shape
                self.signature["arg%d_dtype" % i] = arg.checked_type.dtype
            self.signature["ret_shape"] = op.ret_type.shape
            self.signature["ret_dtype"] = op.ret_type.dtype
            self.visit(op.body)

        if str(op) == "nn.conv2d":
            self.op_attrs = call.attrs

        for arg in call.args:
            self.visit(arg)


def select_gemm_kernel(
    cutlass_profiler,
    op_type,
    MM,
    KK,
    NN,
    out_dtype,
    arg0_dtype,
    arg1_dtype,
    use_3xtf32,
    batched,
    profile_all,
    use_multiprocessing,
):
    """Run CUTLASS profiler to select the best kernel, or return the default one for dynamic
    workloads."""
    if any(isinstance(s, tvm.tir.Any) for s in [MM, KK, NN]):
        out = cutlass_profiler.get_default(
            op_type, out_dtype, arg0_dtype, arg1_dtype, use_3xtf32, batched=batched
        )
        name, cutlass_op_def = out["name"], out["opdef"]
        logger.info("Picked the default kernel %s", name)
    else:
        name, cutlass_op_def, _ = cutlass_profiler.profile(
            op_type,
            MM,
            NN,
            KK,
            out_dtype,
            arg0_dtype,
            arg1_dtype,
            use_3xtf32,
            batched=batched,
            profile_all=profile_all,
            use_multiprocessing=use_multiprocessing,
        )
        if profile_all:
            logger.info("The best kernel is %s", name)
        else:
            logger.info("Picked the first kernel found %s", name)

    return name, cutlass_op_def


def handle_batch_matmul(
    cutlass_profiler,
    op_type,
    arg0_shape,
    arg1_shape,
    out_dtype,
    arg0_dtype,
    arg1_dtype,
    use_3xtf32,
    profile_all,
    use_multiprocessing,
):
    """Profile and select a kernel for batch_matmul op workload."""
    MM = arg0_shape[1]
    KK = arg0_shape[2]
    NN = arg1_shape[1]

    name, cutlass_op_def = select_gemm_kernel(
        cutlass_profiler,
        op_type,
        MM,
        KK,
        NN,
        out_dtype,
        arg0_dtype,
        arg1_dtype,
        use_3xtf32,
        True,
        profile_all,
        use_multiprocessing,
    )

    return {
        "batch": arg0_shape[0],
        "batch_stride_A": arg0_shape[1] * arg0_shape[2],
        "batch_stride_B": arg1_shape[1] * arg1_shape[2],
        "batch_stride_C": arg0_shape[1] * arg1_shape[1],
        "cutlass_op_def": cutlass_op_def,
        "cutlass_op_name": name,
        "lda": "K",
        "ldb": "K",
        "ldc": "N",
    }


def handle_dense(
    cutlass_profiler,
    op_type,
    arg0_shape,
    arg1_shape,
    out_dtype,
    arg0_dtype,
    arg1_dtype,
    use_3xtf32,
    profile_all,
    use_multiprocessing,
):
    """Profile and select a kernel for dense op workload."""
    MM = arg0_shape[0]
    KK = arg0_shape[1]
    NN = arg1_shape[0]

    name, cutlass_op_def = select_gemm_kernel(
        cutlass_profiler,
        op_type,
        MM,
        KK,
        NN,
        out_dtype,
        arg0_dtype,
        arg1_dtype,
        use_3xtf32,
        False,
        profile_all,
        use_multiprocessing,
    )

    assert "tn_align" in name, "Only supports (row_major, col_major) input layout for now."

    return {
        "cutlass_op_def": cutlass_op_def,
        "cutlass_op_name": name,
        "lda": "K",
        "ldb": "K",
        "ldc": "N",
    }


def handle_conv2d(
    cutlass_profiler,
    op_type,
    d_shape,
    w_shape,
    padding,
    strides,
    dilation,
    out_dtype,
    data_dtype,
    weight_dtype,
    use_3xtf32,
    profile_all,
    use_multiprocessing,
):
    """Profile and select a kernel for conv2d op workload."""
    if any(isinstance(s, tvm.tir.Any) for s in d_shape):
        out = cutlass_profiler.get_default(op_type, out_dtype, data_dtype, weight_dtype, use_3xtf32)
        name, cutlass_op_def = out["name"], out["opdef"]
        logger.info("Picked the default kernel %s", name)
    else:
        name, cutlass_op_def, _ = cutlass_profiler.profile(
            op_type,
            d_shape,
            w_shape,
            padding,
            strides,
            dilation,
            out_dtype,
            data_dtype,
            weight_dtype,
            use_3xtf32,
            profile_all=profile_all,
            use_multiprocessing=use_multiprocessing,
        )
        if profile_all:
            logger.info("The best kernel is %s", name)
        else:
            logger.info("Picked the first kernel found %s", name)

    return {
        "cutlass_op_def": cutlass_op_def,
        "cutlass_op_name": name,
    }


def tune_cutlass_kernels(
    mod, sm, use_3xtf32=True, profile_all=True, use_multiprocessing=False, tmp_dir="./tmp"
):
    """Given a module partitioned for CUTLASS offloading, profile each workload to select which
    kernels to emit.

    Parameters
    ----------
    mod : IRModule
        The Relay module with cutlass partitions.

    sm : int
        An integer specifying the compute capability. For example, 75 for Turing and
        80 or 86 for Ampere.

    profile_all : bool
        Whether or not profile all candidate kernels, or stop profiling after
        the first applicable kernel is found.

    use_multiprocessing : bool
        Whether or not compile profiler executables for different kernels in parallel.

    tmp_dir : string, optional
        A temporary directory where intermediate compiled artifacts will be stored.

    Returns
    -------
    mod : IRModule
        The updated module annotated with cutlass profiling information.

    num_cutlass_partition : int
        The number of partitioned functions created for CUTLASS.
    """
    gemm_profiler = CutlassGemmProfiler(sm, _get_cutlass_path(), tmp_dir)
    conv2d_profiler = CutlassConv2DProfiler(sm, _get_cutlass_path(), tmp_dir)
    num_cutlass_partition = 0
    for var in mod.get_global_vars():
        fun_name = var.name_hint
        func = mod[fun_name]
        annotator = OpAnnotator()
        if "cutlass" in fun_name:
            num_cutlass_partition += 1
            annotator.visit(func)
            out_dtype = annotator.signature["ret_dtype"]
            op_type = annotator.signature["op_type"]

            new_attrs = {"op_type": op_type}
            new_attrs.update(annotator.signature)
            new_attrs.update(func.attrs)
            arg0_shape = new_attrs["arg0_shape"]
            arg1_shape = new_attrs["arg1_shape"]
            arg0_dtype = new_attrs["arg0_dtype"]
            arg1_dtype = new_attrs["arg1_dtype"]

            if "conv2d" in op_type:
                new_attrs["padding"] = annotator.op_attrs.padding
                new_attrs["strides"] = annotator.op_attrs.strides
                new_attrs["dilation"] = annotator.op_attrs.dilation
                new_attrs.update(
                    handle_conv2d(
                        conv2d_profiler,
                        op_type,
                        arg0_shape,
                        arg1_shape,
                        annotator.op_attrs.padding,
                        annotator.op_attrs.strides,
                        annotator.op_attrs.dilation,
                        out_dtype,
                        arg0_dtype,
                        arg1_dtype,
                        use_3xtf32,
                        profile_all,
                        use_multiprocessing,
                    )
                )
            elif "batch_matmul" in op_type:
                new_attrs.update(
                    handle_batch_matmul(
                        gemm_profiler,
                        op_type,
                        arg0_shape,
                        arg1_shape,
                        out_dtype,
                        arg0_dtype,
                        arg1_dtype,
                        use_3xtf32,
                        profile_all,
                        use_multiprocessing,
                    )
                )
            elif "dense" in op_type:
                new_attrs.update(
                    handle_dense(
                        gemm_profiler,
                        op_type,
                        arg0_shape,
                        arg1_shape,
                        out_dtype,
                        arg0_dtype,
                        arg1_dtype,
                        use_3xtf32,
                        profile_all,
                        use_multiprocessing,
                    )
                )
            else:
                raise ValueError("%s unsupported composite" % op_type)

            new_attrs = tvm.ir.make_node("DictAttrs", **new_attrs)
            new_func = relay.Function(
                func.params,
                func.body,
                ret_type=func.ret_type,
                type_params=func.type_params,
                attrs=new_attrs,
            )
            mod.update_func(var, new_func)

    return mod, num_cutlass_partition


def build_cutlass_kernels(
    lib, sm, tmp_dir="./tmp", lib_path="compile.so", threads=-1, use_fast_math=False
):
    """Compile CUTLASS kernels in lib and return the runtime module ready to run.

    Parameters
    ----------
    lib : GraphExecutorFactoryModule
        The output from relay.build containing compiled host code and non-cutlass kernels.

    sm : int
        An integer specifying the compute capability. For example, 75 for Turing and
        80 or 86 for Ampere.

    tmp_dir : string, optional
        A temporary directory where intermediate compiled artifacts will be stored.

    lib_path : string, optional
        The path to a shared library which will be generated as the result of the build process.

    threads : int, optional
        The number of threads to use for compiling generated kernels. Only available for
        CUDA 11.2 or later. Use all physical cores by default.

    use_fast_math : bool, optional
        Whether or not to use faster but less accurate math intrinsics.

    Returns
    -------
    updated_lib : runtime.Module
        The updated module with compiled cutlass kernels.
    """
    kwargs = _get_cutlass_compile_options(sm, threads, use_fast_math)
    lib.export_library(lib_path, workspace_dir=tmp_dir, **kwargs)
    return runtime.load_module(lib_path)


def build_cutlass_kernels_vm(
    vm_exec,
    sm,
    tmp_dir="./tmp",
    lib_path="compile.so",
    vmcode_path="vmcode.ro",
    threads=-1,
    use_fast_math=False,
):
    """Compile CUTLASS kernels in vm_exec and return a VM executable ready to run.

    Parameters
    ----------
    vm_exec : vm.Executable
        The output from relay.vm.compile containing compiled host code and non-cutlass kernels.

    sm : int
        An integer specifying the compute capability. For example, 75 for Turing and
        80 or 86 for Ampere.

    tmp_dir : string, optional
        A temporary directory where intermediate compiled artifacts will be stored.

    lib_path : string, optional
        The path to a shared library which will be generated as the result of the build process.

    vmcode_path : string, optional
        The path where the VM bytecode will be serialized to.

    threads : int, optional
        The number of threads to use for compiling generated kernels. Only available for
        CUDA 11.2 or later. Use all physical cores by default.

    use_fast_math : bool, optional
        Whether or not to use faster but less accurate math intrinsics.

    Returns
    -------
    updated_vm_exec: vm.Executable
        The updated exectuable with compiled cutlass kernels.
    """
    code, lib = vm_exec.save()
    kwargs = _get_cutlass_compile_options(sm, threads, use_fast_math)
    lib_path = os.path.join(tmp_dir, lib_path)
    vmcode_path = os.path.join(tmp_dir, vmcode_path)
    lib.export_library(lib_path, workspace_dir=tmp_dir, **kwargs)
    with open(vmcode_path, "wb") as fo:
        fo.write(code)
    lib = tvm.runtime.load_module(lib_path)
    code = bytearray(open(vmcode_path, "rb").read())
    return tvm.runtime.vm.Executable.load_exec(code, lib)
