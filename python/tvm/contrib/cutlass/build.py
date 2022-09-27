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
# pylint: disable=invalid-name, dangerous-default-value
"""Driver for partitioning and building a Relay module for CUTLASS offload."""
import logging
import os
import multiprocessing
import tvm
from tvm import runtime, relay
from tvm.contrib.nvcc import get_cuda_version
from tvm._ffi.registry import register_func
from .gen_gemm import CutlassGemmProfiler
from .gen_conv2d import CutlassConv2DProfiler
from .library import ConvKind

logger = logging.getLogger("cutlass")


def has_cutlass():
    """Returns true if the CUTLASS custom codegen is available"""
    return tvm.get_global_func("relay.ext.cutlass.create_c_source_module", True) is not None


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
        "-c",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-gencode=arch=compute_%d,code=[sm_%d,compute_%d]" % (sm, sm, sm),
        "-Xcompiler=-fPIC",
        "-Xcompiler=-Wconversion",
        "-Xcompiler=-fno-strict-aliasing",
        "-O3",
        "-std=c++17",
        "-I" + cutlass_include,
        "-I" + cutlass_util_include,
    ]
    if use_fast_math:
        kwargs["options"].append("-DCUTLASS_USE_TANH_FOR_SIGMOID")
    cuda_ver = get_cuda_version()
    if cuda_ver >= (11, 2):
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
        if isinstance(op, relay.Function) and "Composite" in op.attrs:
            self.signature["op_type"] = op.attrs["Composite"]
            for i, arg in enumerate(op.params):
                self.signature["arg%d_shape" % i] = arg.checked_type.shape
                self.signature["arg%d_dtype" % i] = arg.checked_type.dtype
            self.signature["ret_shape"] = op.ret_type.shape
            self.signature["ret_dtype"] = op.ret_type.dtype
            self.visit(op.body)

        if str(op) in ["nn.conv2d", "nn.conv2d_transpose", "nn.conv2d_backward_weight"]:
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
    find_first_valid,
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
            find_first_valid=find_first_valid,
            use_multiprocessing=use_multiprocessing,
        )
        if not find_first_valid:
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
    find_first_valid,
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
        find_first_valid,
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
    find_first_valid,
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
        find_first_valid,
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
    split_k_slices,
    profile_all_alignments,
    find_first_valid,
    use_multiprocessing,
):
    """Profile and select a kernel for conv2d op workload."""
    if "conv2d_transpose" in op_type:
        conv_kind = ConvKind.Dgrad
    elif "backward_weight" in op_type:
        conv_kind = ConvKind.Wgrad
    else:
        conv_kind = ConvKind.Fprop

    if any(isinstance(s, tvm.tir.Any) for s in d_shape):
        out = cutlass_profiler.get_default(
            op_type, out_dtype, data_dtype, weight_dtype, use_3xtf32, conv_kind, strides
        )
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
            conv_kind,
            split_k_slices,
            profile_all_alignments,
            find_first_valid=find_first_valid,
            use_multiprocessing=use_multiprocessing,
        )
        if not find_first_valid:
            logger.info("The best kernel is %s", name)
        else:
            logger.info("Picked the first kernel found %s", name)

    return {
        "cutlass_op_def": cutlass_op_def,
        "cutlass_op_name": name,
    }


def num_cutlass_partitions(mod):
    return sum([(1 if "cutlass" in var.name_hint else 0) for var in mod.get_global_vars()])


def tune_cutlass_kernels(
    mod,
    sm,
    use_3xtf32=True,
    split_k_slices=[1],
    profile_all_alignments=False,
    find_first_valid=False,
    use_multiprocessing=False,
    tmp_dir="./tmp",
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

    use_3xtf32 : bool
        Wheter or not use slower but very accurate (compared to tf32) 3xtf32 mode for
        fp32 inputs on tensorcore.

    split_k_slices : list of int
        Split factor candidates for split-K GEMM. If split-K > 1, the GEMM K-loop is computed in
        parallel across split-K blocks, and a separate global reduction kernel is launched to
        accumulate partial reductions. The profiler will pick the best split-k factor from the
        given candidate list. Note that the larger split-K factor requires a larger workspace.
        Currently, parallel split-k has been tested only for wgrad. For GEMM and other conv2d
        kinds, split_k_slices is ignored.

    profile_all_alignments : bool
        When True, profile all kernal variants with smaller alignments than the largest possible.

    find_first_valid : bool
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
        if "cutlass" in fun_name:
            num_cutlass_partition += 1
            new_func = tune_cutlass_function(
                func,
                use_3xtf32,
                split_k_slices,
                profile_all_alignments,
                find_first_valid,
                use_multiprocessing,
                gemm_profiler,
                conv2d_profiler,
            )
            mod.update_func(var, new_func)

    return mod, num_cutlass_partition


def tune_cutlass_function(
    func,
    use_3xtf32,
    split_k_slices,
    profile_all_alignments,
    find_first_valid,
    use_multiprocessing,
    gemm_profiler,
    conv2d_profiler,
):
    """Given a function intended to be offloaded to CUTLASS,  profile each workload to select which
    kernels to emit.

    Parameters
    ----------
    func : IRModule
        The Relay Function to tune for.

    use_3xtf32 : bool
        Wheter or not use slower but very accurate (compared to tf32) 3xtf32 mode for
        fp32 inputs on tensorcore.

    split_k_slices : list of int
        Split factor candidates for split-K GEMM. If split-K > 1, the GEMM K-loop is computed in
        parallel accross split-K blocks, and a seperate global reduction kernel is launched to
        accumulate partial reductions. The profiler will pick the best split-k factor from the
        given candidate list. Note that the larger split-K factor requires a larger workspace.
        Currently, parallel split-k has been tested only for wgrad. For GEMM and other conv2d
        kinds, split_k_slices is ignored.

    profile_all_alignments : bool
        When True, profile all kernal variants with smaller alignments than the largest possible.

    find_first_valid : bool
        Whether or not profile all candidate kernels, or stop profiling after
        the first applicable kernel is found.

    use_multiprocessing : bool
        Whether or not compile profiler executables for different kernels in parallel.

    gemm_profiler : CutlassGemmProfiler
        Profiler for dense operators. May cache results between tuned functions.

    conv2d_profiler : CutlassConv2DProfiler
        Profiler for conv2d operators. May cach results between tuned functions.

    Returns
    -------
    annot_func : Function
        The input function with attributes capturing the best CUTLASS kernel found by tuning.
    """
    annotator = OpAnnotator()
    annotator.visit(func)
    out_shape = annotator.signature["ret_shape"]
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

        if "conv2d_transpose" in op_type:
            d_shape = out_shape
            w_shape = arg1_shape
        elif "conv2d_backward_weight" in op_type:
            d_shape = arg1_shape
            w_shape = out_shape
        else:
            d_shape = arg0_shape
            w_shape = arg1_shape

        new_attrs.update(
            handle_conv2d(
                conv2d_profiler,
                op_type,
                d_shape,
                w_shape,
                annotator.op_attrs.padding,
                annotator.op_attrs.strides,
                annotator.op_attrs.dilation,
                out_dtype,
                arg0_dtype,
                arg1_dtype,
                use_3xtf32,
                split_k_slices,
                profile_all_alignments,
                find_first_valid,
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
                find_first_valid,
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
                find_first_valid,
                use_multiprocessing,
            )
        )
    else:
        raise ValueError("%s unsupported composite" % op_type)

    new_attrs = tvm.ir.make_node("DictAttrs", **new_attrs)
    return relay.Function(
        func.params,
        func.body,
        ret_type=func.ret_type,
        type_params=func.type_params,
        attrs=new_attrs,
    )


@register_func("relay.ext.cutlass.compile_for_cutlass")
def compile_for_cutlass(mod, cutlass_target):
    """Given an IRModule with at least one Compiler='cutlass' Relay function, return a
    LibraryModule with all such functions compiled into their PackedFunc-compatible form.
     - First runs CUTLASS tuning to decide on the best kernels, which itself requires the
       repeated compilation and execution of CUDA code using nvcc. The results of this
       is captured as annotation on each relevant function. Kernel performance is cached
       overall all functions.
     - Then generates a single CSourceModule containing C code implementing all the
       Compiler='cutlass' Relay functions, accounting for the tuning done above.
     - Then compiles that CSourceModule with the appropriate nvcc arguments to yield
       a static .o library. An export_library step will be required on the final runtime
       module to link that library into the overall .so library.
     See CompileForCutlass in src/relay/backend/contrib/cutlass/codegen.cc for where this
     helper function is used to implement the RelayToTIR pass hook for CUTLASS."""

    # Recover options from the current 'cutlass' Target
    assert cutlass_target.kind.name == "cutlass"
    tuning_config = {
        key: cutlass_target.attrs.get(key)
        for key in [
            "sm",
            "use_3xtf32",
            "split_k_slices",
            "profile_all_alignments",
            "find_first_valid",
            "use_multiprocessing",
        ]
    }
    compile_config = {
        key: cutlass_target.attrs.get(key) for key in ["sm", "threads", "use_fast_math"]
    }
    tmp_dir = cutlass_target.attrs.get("tmp_dir")

    # Tune
    logger.info("Tuning for CUTLASS")
    mod, _ = tune_cutlass_kernels(mod, tmp_dir=tmp_dir, **tuning_config)

    # Compile
    logger.info("Creating CSource module for CUTLASS")
    create_c_source_module = tvm._ffi.get_global_func("relay.ext.cutlass.create_c_source_module")
    c_module = create_c_source_module(mod)
    function_names = c_module.get_function("get_func_names")()
    compile_options = _get_cutlass_compile_options(**compile_config)
    lib_path = os.path.join(tmp_dir, "cutlass.o")
    logger.info("Compiling generated CUTLASS code")
    c_module.export_library(lib_path, workspace_dir=tmp_dir, **compile_options)

    # Recover static library
    logger.info("Loading compiled CUTLASS code")
    final_mod = tvm.runtime.load_static_library(lib_path, function_names)

    logger.info("Done with CUTLASS compilation")
    return final_mod


def finalize_modules(lib, lib_path="compile.so", tmp_dir="./tmp"):
    """Returns lib with any C source, LLVM and static library modules complied and linked in ready
    for use by the graph or AOT executors. This method is not specific to CUTLASS, however it does
    assume nvcc will be used for final compilation and linking. It is provided here for
    convenience.

    Parameters
    ----------
    lib : runtime.Module
        The output from relay.build.

    lib_path : string
        The path to a shared library which will be generated as the result of the build process.

    tmp_dir : string
        A temporary directory where intermediate compiled artifacts will be stored.

    Returns
    -------
    updated_lib : runtime.Module
        The updated library with all compilation and linking completed.

    """
    lib_path = os.path.join(tmp_dir, lib_path)
    lib.export_library(lib_path, workspace_dir=tmp_dir, cc="nvcc")
    return runtime.load_module(lib_path)


def finalize_modules_vm(vm_exec, lib_path="compile.so", vmcode_path="vmcode.ro", tmp_dir="./tmp"):
    """Returns vm_exec with any C source, LLVM and static library modules compiled and linked in
    ready for use by the VM executor. This method is not specific to CUTLASS, however it does
    assume nvcc will be used for final compilation and linking. It is provided here for
    convenience.

    Parameters
    ----------
    vm_exec : vm.Executable
        The output from relay.vm.compile containing compiled host code and kernels.

    lib_path : string
        The path to a shared library which will be generated as the result of the build process.

    vmcode_path : string
        The path where the VM bytecode will be serialized to as a side-effect.

    tmp_dir : string
        A temporary directory where intermediate compiled artifacts will be stored.

    Returns
    -------
    updated_vm_exec : vm.Executable
        The updated VM executable with all compilation and linking completed.
    """
    code, lib = vm_exec.save()
    lib_path = os.path.join(tmp_dir, lib_path)
    vmcode_path = os.path.join(tmp_dir, vmcode_path)
    lib.export_library(lib_path, workspace_dir=tmp_dir, cc="nvcc")
    with open(vmcode_path, "wb") as fo:
        fo.write(code)
    lib = tvm.runtime.load_module(lib_path)
    return tvm.runtime.vm.Executable.load_exec(code, lib)
